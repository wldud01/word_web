"""
mk_unet_3d.py
3D version of MKRF_UNet — uses 3D convolutions to model the window/depth dimension.

Input/Output convention (same interface as MKRF_UNet):
  x    : (B, W, H, Hs)  — W is the window / slice depth
  cond : (B, W, H, Hs) or None
  → output : (B, W, H, Hs)

Internally, the depth dimension W is treated as the 3D depth axis:
  x  →  unsqueeze(1)  →  (B, 1, W, H, Hs)
  processed through 3D encoder-decoder
  → squeeze(1) → (B, W, H, Hs)

Spatial pooling only (kernel/stride=(1,2,2)) so depth W is preserved throughout.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

__all__ = ['MKRF_UNet_3D']


# ============================================================
# Utilities
# ============================================================

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights_3d(module, name, scheme=''):
    if isinstance(module, (nn.Conv3d,)):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        else:
            fan_out = 1
            for k in module.kernel_size:
                fan_out *= k
            fan_out = fan_out * module.out_channels // module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    elif act == 'relu6':
        return nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'hswish':
        return nn.Hardswish(inplace)
    raise NotImplementedError(f'activation layer [{act}] is not found')


def channel_shuffle_3d(x, groups):
    B, C, D, H, W = x.shape
    cpg = C // groups
    x = x.view(B, groups, cpg, D, H, W).transpose(1, 2).contiguous()
    return x.view(B, C, D, H, W)


# ============================================================
# 3D Attention Modules
# ============================================================

class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        reduced = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(in_planes, reduced, 1, bias=False)
        self.fc2 = nn.Conv3d(reduced, in_planes, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        named_apply(partial(_init_weights_3d, scheme='normal'), self)

    def forward(self, x):
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7, 11)
        pad = kernel_size // 2
        # (1, k, k) kernel — spatial attention per depth slice
        self.conv = nn.Conv3d(2, 1, (1, kernel_size, kernel_size),
                              padding=(0, pad, pad), bias=False)
        self.sigmoid = nn.Sigmoid()
        named_apply(partial(_init_weights_3d, scheme='normal'), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class GroupedAttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=1):
        super().__init__()
        if kernel_size > 1:
            k = (1, kernel_size, kernel_size)
            p = (0, kernel_size // 2, kernel_size // 2)
        else:
            k, p = 1, 0

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, k, padding=p, bias=True),
            nn.BatchNorm3d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, k, padding=p, bias=True),
            nn.BatchNorm3d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, 1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
        named_apply(partial(_init_weights_3d, scheme='normal'), self)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


# ============================================================
# 3D Multi-Kernel Depthwise Building Blocks
# ============================================================

class MultiKernelDepthwiseConv3D(nn.Module):
    """
    Multi-scale depthwise conv on spatial (H, W) dimensions.
    Kernel applied as (1, k, k) to preserve depth dimension.
    """
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super().__init__()
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels,
                          (1, k, k), stride=(1, stride, stride),
                          padding=(0, k // 2, k // 2),
                          groups=in_channels, bias=False),
                nn.BatchNorm3d(in_channels),
                act_layer(activation, inplace=True),
            )
            for k in kernel_sizes
        ])
        named_apply(partial(_init_weights_3d, scheme='normal'), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MultiKernelInvertedResidualBlock3D(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion_factor=2,
                 dw_parallel=True, add=True, kernel_sizes=(1, 3, 5), activation='relu6'):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip = (stride == 1)

        ex_c = int(in_c * expansion_factor)
        self.ex_c = ex_c

        self.pconv1 = nn.Sequential(
            nn.Conv3d(in_c, ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm3d(ex_c),
            act_layer(activation, inplace=True),
        )
        self.multi_scale_dwconv = MultiKernelDepthwiseConv3D(
            ex_c, kernel_sizes, stride, activation, dw_parallel
        )
        combined = ex_c if add else ex_c * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv3d(combined, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_c),
        )
        if self.use_skip and in_c != out_c:
            self.conv1x1 = nn.Conv3d(in_c, out_c, 1, 1, 0, bias=False)

        named_apply(partial(_init_weights_3d, scheme='normal'), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        if self.add:
            dout = sum(dwconv_outs)
        else:
            dout = torch.cat(dwconv_outs, dim=1)
        dout = channel_shuffle_3d(dout, gcd(self.ex_c if self.add else self.ex_c * self.n_scales, self.out_c))
        out = self.pconv2(dout)
        if self.use_skip:
            skip = self.conv1x1(x) if self.in_c != self.out_c else x
            return skip + out
        return out


def mk_irb_bottleneck_3d(in_c, out_c, n, s, expansion_factor=2, dw_parallel=True,
                          add=True, kernel_sizes=(1, 3, 5), activation='relu6'):
    blocks = [MultiKernelInvertedResidualBlock3D(
        in_c, out_c, s, expansion_factor, dw_parallel, add, kernel_sizes, activation
    )]
    for _ in range(n - 1):
        blocks.append(MultiKernelInvertedResidualBlock3D(
            out_c, out_c, 1, expansion_factor, dw_parallel, add, kernel_sizes, activation
        ))
    return nn.Sequential(*blocks)


# ============================================================
# Time Embedding
# ============================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(
            torch.arange(half, device=device) * -(torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        emb = t * emb
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def apply_scale_shift_3d(x, t_emb):
    """x: (B, C, D, H, W)  t_emb: (B, 2C)"""
    # unsqueeze for broadcast over D, H, W
    t = t_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (B, 2C, 1, 1, 1)
    scale, shift = t.chunk(2, dim=1)
    return x * (scale + 1) + shift


# ============================================================
# MKRF_UNet_3D
# ============================================================

class MKRF_UNet_3D(nn.Module):
    """
    3D UNet backbone for window-based (multi-slice) inputs.

    Interface matches MKRF_UNet:
      forward(x, times, cond=None, seg=None)
      x    : (B, W, H, Hs)  — W = window size (used as depth dim internally)
      cond : (B, W, H, Hs) or None
      → (B, W, H, Hs)
    """

    def __init__(
        self,
        in_channels: int = 5,         # window_size (kept for interface compat)
        out_channels: int = 5,        # window_size (kept for interface compat)
        channels=(64, 128, 256, 384, 512),
        depths=(1, 1, 1, 1, 1),
        kernel_sizes=(1, 3, 5),
        expansion_factor: int = 2,
        gag_kernel: int = 3,
        time_embed_dim: int = 256,
        deep_supervision: bool = True,
        **kwargs,                      # absorbs regis=, model= etc.
    ):
        super().__init__()

        C0, C1, C2, C3, C4 = channels

        self.in_channels = in_channels     # = window_size
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision

        # ── Time embedding ───────────────────────────────────
        self.time_pos_emb = SinusoidalPosEmb(time_embed_dim)
        self.time_mlp = nn.ModuleDict({
            "enc5":  nn.Sequential(nn.Linear(time_embed_dim, C4 * 2), nn.GELU(), nn.Linear(C4 * 2, C4 * 2)),
            "dec4":  nn.Sequential(nn.Linear(time_embed_dim, C3 * 2), nn.GELU(), nn.Linear(C3 * 2, C3 * 2)),
            "dec3":  nn.Sequential(nn.Linear(time_embed_dim, C2 * 2), nn.GELU(), nn.Linear(C2 * 2, C2 * 2)),
            "dec2":  nn.Sequential(nn.Linear(time_embed_dim, C1 * 2), nn.GELU(), nn.Linear(C1 * 2, C1 * 2)),
            "dec1":  nn.Sequential(nn.Linear(time_embed_dim, C0 * 2), nn.GELU(), nn.Linear(C0 * 2, C0 * 2)),
            "final": nn.Sequential(nn.Linear(time_embed_dim, C0 * 2), nn.GELU(), nn.Linear(C0 * 2, C0 * 2)),
        })

        # ── Encoder ──────────────────────────────────────────
        # encoder1_flow : no cond  → input (B, 1, W, H, Hs)
        # encoder1_img  : with cond → input (B, 2, W, H, Hs)   [x + cond]
        self.encoder1_flow = mk_irb_bottleneck_3d(1, C0, depths[0], 1, expansion_factor, True, True, kernel_sizes)
        self.encoder1_img  = mk_irb_bottleneck_3d(2, C0, depths[0], 1, expansion_factor, True, True, kernel_sizes)

        self.encoder2 = mk_irb_bottleneck_3d(C0, C1, depths[1], 1, expansion_factor, True, True, kernel_sizes)
        self.encoder3 = mk_irb_bottleneck_3d(C1, C2, depths[2], 1, expansion_factor, True, True, kernel_sizes)
        self.encoder4 = mk_irb_bottleneck_3d(C2, C3, depths[3], 1, expansion_factor, True, True, kernel_sizes)
        self.encoder5 = mk_irb_bottleneck_3d(C3, C4, depths[4], 1, expansion_factor, True, True, kernel_sizes)

        # ── Attention Gates ───────────────────────────────────
        self.AG1 = GroupedAttentionGate3D(C3, C3, C3 // 2, gag_kernel)
        self.AG2 = GroupedAttentionGate3D(C2, C2, C2 // 2, gag_kernel)
        self.AG3 = GroupedAttentionGate3D(C1, C1, C1 // 2, gag_kernel)
        self.AG4 = GroupedAttentionGate3D(C0, C0, C0 // 2, gag_kernel)

        # ── Decoder ──────────────────────────────────────────
        self.decoder1 = mk_irb_bottleneck_3d(C4, C3, 1, 1, expansion_factor, True, True, kernel_sizes)
        self.decoder2 = mk_irb_bottleneck_3d(C3, C2, 1, 1, expansion_factor, True, True, kernel_sizes)
        self.decoder3 = mk_irb_bottleneck_3d(C2, C1, 1, 1, expansion_factor, True, True, kernel_sizes)
        self.decoder4 = mk_irb_bottleneck_3d(C1, C0, 1, 1, expansion_factor, True, True, kernel_sizes)
        self.decoder5 = mk_irb_bottleneck_3d(C0, C0, 1, 1, expansion_factor, True, True, kernel_sizes)

        # ── Channel + Spatial Attention (3D) ─────────────────
        self.CA1 = ChannelAttention3D(C4)
        self.SA1 = SpatialAttention3D()

        self.CA2 = ChannelAttention3D(C3)
        self.SA2 = SpatialAttention3D()

        self.CA3 = ChannelAttention3D(C2)
        self.SA3 = SpatialAttention3D()

        self.CA4 = ChannelAttention3D(C1)
        self.SA4 = SpatialAttention3D()

        self.CA5 = ChannelAttention3D(C0)
        self.SA5 = SpatialAttention3D()

        # ── Deep supervision heads ────────────────────────────
        # Each head outputs (B, 1, W, H, Hs) → reshaped to (B, W, H, Hs) outside
        self.ds1 = nn.Conv3d(C2, 1, 1)
        self.ds2 = nn.Conv3d(C1, 1, 1)
        self.ds3 = nn.Conv3d(C0, 1, 1)

        # ── Final output ─────────────────────────────────────
        self.final = nn.Conv3d(C0, 1, 1)

    # ----------------------------------------------------------------
    def forward(self, x, times=None, cond=None, seg=None):
        """
        x     : (B, W, H, Hs)
        times : (B,) or (B, 1)
        cond  : (B, W, H, Hs) or None
        """
        if times.dim() == 1:
            times = times.unsqueeze(1)
        t = self.time_pos_emb(times)          # (B, time_embed_dim)

        # ── Build 3D input ────────────────────────────────────
        x3d = x.unsqueeze(1)                  # (B, 1, W, H, Hs)

        if cond is not None:
            c3d = cond.unsqueeze(1)            # (B, 1, W, H, Hs)
            x3d = torch.cat([x3d, c3d], dim=1) # (B, 2, W, H, Hs)
            is_flow = False
        else:
            is_flow = True

        if seg is not None:
            s3d = seg.unsqueeze(1)
            x3d = torch.cat([x3d, s3d], dim=1)

        # ── Encoder ──────────────────────────────────────────
        if is_flow:
            out = self.encoder1_flow(x3d)     # (B, C0, W, H, Hs)
        else:
            out = self.encoder1_img(x3d)      # (B, C0, W, H, Hs)

        # Spatial-only pooling: (1, 2, 2) keeps depth W intact
        out = F.max_pool3d(out, (1, 2, 2)); t1 = out   # (B, C0, W, H/2, Hs/2)
        out = F.max_pool3d(self.encoder2(out), (1, 2, 2)); t2 = out
        out = F.max_pool3d(self.encoder3(out), (1, 2, 2)); t3 = out
        out = F.max_pool3d(self.encoder4(out), (1, 2, 2)); t4 = out
        out = F.max_pool3d(self.encoder5(out), (1, 2, 2))

        out = apply_scale_shift_3d(out, self.time_mlp["enc5"](t))

        # ── Decoder ──────────────────────────────────────────
        preds = []

        # Stage 4
        out = self.CA1(out) * out
        out = self.SA1(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), size=t4.shape[2:],
                                   mode='trilinear', align_corners=False))
        out = apply_scale_shift_3d(out, self.time_mlp["dec4"](t))
        out = out + self.AG1(out, t4)

        # Stage 3 (DS-1)
        out = self.CA2(out) * out
        out = self.SA2(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), size=t3.shape[2:],
                                   mode='trilinear', align_corners=False))
        out = apply_scale_shift_3d(out, self.time_mlp["dec3"](t))
        if self.training and self.deep_supervision:
            p1 = self.ds1(out)
            p1 = F.interpolate(p1, size=x3d.shape[2:], mode='trilinear', align_corners=False)
            preds.append(p1.squeeze(1))        # (B, W, H, Hs)
        out = out + self.AG2(out, t3)

        # Stage 2 (DS-2)
        out = self.CA3(out) * out
        out = self.SA3(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), size=t2.shape[2:],
                                   mode='trilinear', align_corners=False))
        out = apply_scale_shift_3d(out, self.time_mlp["dec2"](t))
        if self.training and self.deep_supervision:
            p2 = self.ds2(out)
            p2 = F.interpolate(p2, size=x3d.shape[2:], mode='trilinear', align_corners=False)
            preds.append(p2.squeeze(1))
        out = out + self.AG3(out, t2)

        # Stage 1 (DS-3)
        out = self.CA4(out) * out
        out = self.SA4(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), size=t1.shape[2:],
                                   mode='trilinear', align_corners=False))
        out = apply_scale_shift_3d(out, self.time_mlp["dec1"](t))
        if self.training and self.deep_supervision:
            p3 = self.ds3(out)
            p3 = F.interpolate(p3, size=x3d.shape[2:], mode='trilinear', align_corners=False)
            preds.append(p3.squeeze(1))
        out = out + self.AG4(out, t1)

        # Final
        out = self.CA5(out) * out
        out = self.SA5(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), size=x3d.shape[2:],
                                   mode='trilinear', align_corners=False))
        out = apply_scale_shift_3d(out, self.time_mlp["final"](t))

        flow_out = self.final(out).squeeze(1)  # (B, W, H, Hs)

        if self.training and self.deep_supervision:
            preds.append(flow_out)
            return preds[::-1]   # [final, ds3, ds2, ds1]

        return flow_out
