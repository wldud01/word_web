"""
mk_unet_ldm_attn.py
MKRF_UNet with LDM-style SpatialTransformer (Self-Attn + Cross-Attn + FFN).
CA/SA modules removed; SpatialTransformer inserted at:
  Encoder : after enc3, enc4, enc5 (bottleneck)
  Decoder : decoder1↑→scale_shift→ST, decoder2↑→scale_shift→ST, decoder3↑→scale_shift→ST

Context for cross-attention = seg (forward parameter).
NECT condition              = cond (forward parameter, concatenated with x in encoder1).
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial

from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


# ── shared helpers ────────────────────────────────────────────────────────────

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None: nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None: nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':      return nn.ReLU(inplace)
    if act == 'relu6':     return nn.ReLU6(inplace)
    if act == 'leakyrelu': return nn.LeakyReLU(neg_slope, inplace)
    if act == 'prelu':     return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    if act == 'gelu':      return nn.GELU()
    if act == 'hswish':    return nn.Hardswish(inplace)
    raise NotImplementedError(f'activation layer [{act}] not found')


def channel_shuffle(x, groups):
    b, c, h, w = x.shape
    cg = c // groups
    x = x.view(b, groups, cg, h, w).transpose(1, 2).contiguous()
    return x.view(b, -1, h, w)


# ── MK-IRB building blocks (identical to mk_unet_img_attn_sup) ───────────────

class GroupedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1, activation='relu'):
        super().__init__()
        if kernel_size == 1: groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, 1, kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, 1, kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.act = act_layer(activation, inplace=True)
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, g, x):
        return x * self.psi(self.act(self.W_g(g) + self.W_x(x)))


class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super().__init__()
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, ks, stride, ks//2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True)
            ) for ks in kernel_sizes
        ])
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, x):
        outputs = []
        for dw in self.dwconvs:
            out = dw(x)
            outputs.append(out)
            if not self.dw_parallel:
                x = x + out
        return outputs


class MultiKernelInvertedResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True,
                 add=True, kernel_sizes=[1,3,5], activation='relu6'):
        super().__init__()
        assert stride in [1, 2]
        self.use_skip = stride == 1
        self.in_c, self.out_c, self.add = in_c, out_c, add
        self.n_scales = len(kernel_sizes)
        ex_c = int(in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, ex_c, 1, bias=False), nn.BatchNorm2d(ex_c), act_layer(activation, inplace=True))
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(ex_c, kernel_sizes, stride, activation, dw_parallel)
        combined = ex_c if add else ex_c * self.n_scales
        self.pconv2 = nn.Sequential(nn.Conv2d(combined, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
        if self.use_skip and in_c != out_c:
            self.conv1x1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, x):
        p = self.pconv1(x)
        outs = self.multi_scale_dwconv(p)
        d = sum(outs) if self.add else torch.cat(outs, dim=1)
        combined = p.shape[1] if self.add else p.shape[1] * self.n_scales
        d = channel_shuffle(d, gcd(combined, self.out_c))
        out = self.pconv2(d)
        if self.use_skip:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        return out


def mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2, dw_parallel=True,
                      add=True, kernel_sizes=[1,3,5], activation='relu6'):
    blocks = [MultiKernelInvertedResidualBlock(
        in_c, out_c, s, expansion_factor, dw_parallel, add, kernel_sizes, activation)]
    for _ in range(1, n):
        blocks.append(MultiKernelInvertedResidualBlock(
            out_c, out_c, 1, expansion_factor, dw_parallel, add, kernel_sizes, activation))
    return nn.Sequential(*blocks)


# ── Time embedding ────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=device) * -(math.log(10000.0) / (half - 1)))
        emb = t.float() * emb
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def apply_scale_shift(x, t_emb):
    t = t_emb.unsqueeze(-1).unsqueeze(-1)
    scale, shift = t.chunk(2, dim=1)
    return x * (scale + 1) + shift


# ── LDM-style Spatial Transformer (self-contained, no ldm dependency) ────────

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        inner = int(dim * 4)
        self.net = nn.Sequential(GEGLU(dim, inner), nn.Dropout(dropout), nn.Linear(inner, dim))

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner = heads * dim_head
        context_dim = context_dim or query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q  = nn.Linear(query_dim,   inner, bias=False)
        self.to_k  = nn.Linear(context_dim, inner, bias=False)
        self.to_v  = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        ctx = context if context is not None else x
        k, v = self.to_k(ctx), self.to_v(ctx)
        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(rearrange(out, '(b h) n d -> b n (h d)', h=h))


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None):
        super().__init__()
        self.attn1 = CrossAttention(dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.attn2 = CrossAttention(dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff    = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x           # self-attention
        x = self.attn2(self.norm2(x), context=context) + x  # cross-attention with seg
        x = self.ff(self.norm3(x)) + x               # feed-forward
        return x


class SpatialTransformerBlock(nn.Module):
    """Flatten feature map → transformer → unflatten. Residual connection included."""

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None):
        super().__init__()
        inner = n_heads * d_head
        self.norm     = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.proj_in  = nn.Conv2d(in_channels, inner, 1)
        self.blocks   = nn.ModuleList([
            BasicTransformerBlock(inner, n_heads, d_head, dropout=dropout, context_dim=context_dim)
            for _ in range(depth)
        ])
        self.proj_out = nn.Conv2d(inner, in_channels, 1)
        # zero-init for stable early training (LDM practice)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        residual = x
        x = self.proj_in(self.norm(x))
        x = rearrange(x, 'b c h w -> b (h w) c')
        for blk in self.blocks:
            x = blk(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return self.proj_out(x) + residual


# ── Main backbone ─────────────────────────────────────────────────────────────

class MKRF_UNet_LDM(nn.Module):
    """
    MK-IRB UNet with LDM-style SpatialTransformer attention.

    Attention positions
    -------------------
    Encoder:
      enc3 → SpatialTransformer → pool  (t3)
      enc4 → SpatialTransformer → pool  (t4)
      enc5 → scale_shift → SpatialTransformer        (bottleneck)

    Decoder:
      dec1 upsample → scale_shift → SpatialTransformer → AG1 + t4
      dec2 upsample → scale_shift → SpatialTransformer → AG2 + t3
      dec3 upsample → scale_shift → SpatialTransformer → AG3 + t2

    Parameters
    ----------
    seg_channels : int   channels of the seg input used as cross-attention context
    context_dim  : int   projected dim for seg context tokens
    attn_depth   : int   transformer depth per attention position
    """

    def __init__(
        self,
        in_channels=5,
        out_channels=5,
        channels=[64, 128, 256, 384, 512],
        depths=[1, 1, 1, 1, 1],
        kernel_sizes=[1, 3, 5],
        expansion_factor=2,
        gag_kernel=3,
        time_embed_dim=256,
        deep_supervision=True,
        regis=False,
        seg_channels=1,
        context_dim=256,
        attn_depth=1,
        dropout=0.,
    ):
        super().__init__()
        C0, C1, C2, C3, C4 = channels
        self.in_channels      = in_channels
        self.deep_supervision = deep_supervision

        # ── time embedding ────────────────────────────────────────────────────
        self.time_pos_emb = SinusoidalPosEmb(time_embed_dim)
        self.time_mlp = nn.ModuleDict({
            "enc5":  nn.Sequential(nn.Linear(time_embed_dim, C4*2), nn.GELU(), nn.Linear(C4*2, C4*2)),
            "dec4":  nn.Sequential(nn.Linear(time_embed_dim, C3*2), nn.GELU(), nn.Linear(C3*2, C3*2)),
            "dec3":  nn.Sequential(nn.Linear(time_embed_dim, C2*2), nn.GELU(), nn.Linear(C2*2, C2*2)),
            "dec2":  nn.Sequential(nn.Linear(time_embed_dim, C1*2), nn.GELU(), nn.Linear(C1*2, C1*2)),
            "dec1":  nn.Sequential(nn.Linear(time_embed_dim, C0*2), nn.GELU(), nn.Linear(C0*2, C0*2)),
            "final": nn.Sequential(nn.Linear(time_embed_dim, C0*2), nn.GELU(), nn.Linear(C0*2, C0*2)),
        })

        # ── encoder ───────────────────────────────────────────────────────────
        # encoder1_flow : no condition  (cond is None)
        self.encoder1_flow = mk_irb_bottleneck(
            in_channels, C0, depths[0], 1, expansion_factor, True, True, kernel_sizes)
        # encoder1_img  : x cat cond (NECT same shape) → 2*in_channels
        self.encoder1_img  = mk_irb_bottleneck(
            in_channels * 2, C0, depths[0], 1, expansion_factor, True, True, kernel_sizes)

        self.encoder2 = mk_irb_bottleneck(C0, C1, depths[1], 1, expansion_factor, True, True, kernel_sizes)
        self.encoder3 = mk_irb_bottleneck(C1, C2, depths[2], 1, expansion_factor, True, True, kernel_sizes)
        self.encoder4 = mk_irb_bottleneck(C2, C3, depths[3], 1, expansion_factor, True, True, kernel_sizes)
        self.encoder5 = mk_irb_bottleneck(C3, C4, depths[4], 1, expansion_factor, True, True, kernel_sizes)

        # ── spatial transformers ──────────────────────────────────────────────
        def _n_heads(ch): return max(1, ch // 64)
        def _make_st(ch):
            return SpatialTransformerBlock(
                ch, _n_heads(ch), 64, depth=attn_depth, dropout=dropout, context_dim=context_dim)

        self.st_enc3 = _make_st(C2)   # encoder stage 3
        self.st_enc4 = _make_st(C3)   # encoder stage 4
        self.st_enc5 = _make_st(C4)   # bottleneck
        self.st_dec1 = _make_st(C3)   # decoder stage 4
        self.st_dec2 = _make_st(C2)   # decoder stage 3
        self.st_dec3 = _make_st(C1)   # decoder stage 2

        # ── attention gates ───────────────────────────────────────────────────
        self.AG1 = GroupedAttentionGate(C3, C3, C3//2, gag_kernel, C3//2)
        self.AG2 = GroupedAttentionGate(C2, C2, C2//2, gag_kernel, C2//2)
        self.AG3 = GroupedAttentionGate(C1, C1, C1//2, gag_kernel, C1//2)
        self.AG4 = GroupedAttentionGate(C0, C0, C0//2, gag_kernel, C0//2)

        # ── decoder ───────────────────────────────────────────────────────────
        self.decoder1 = mk_irb_bottleneck(C4, C3, 1, 1, expansion_factor, True, True, kernel_sizes)
        self.decoder2 = mk_irb_bottleneck(C3, C2, 1, 1, expansion_factor, True, True, kernel_sizes)
        self.decoder3 = mk_irb_bottleneck(C2, C1, 1, 1, expansion_factor, True, True, kernel_sizes)
        self.decoder4 = mk_irb_bottleneck(C1, C0, 1, 1, expansion_factor, True, True, kernel_sizes)
        self.decoder5 = mk_irb_bottleneck(C0, C0, 1, 1, expansion_factor, True, True, kernel_sizes)

        # ── deep supervision heads ────────────────────────────────────────────
        self.ds1 = nn.Conv2d(C2, out_channels, 1)
        self.ds2 = nn.Conv2d(C1, out_channels, 1)
        self.ds3 = nn.Conv2d(C0, out_channels, 1)

        # ── seg context projection ────────────────────────────────────────────
        self.seg_proj = nn.Sequential(
            nn.Conv2d(seg_channels, context_dim, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(context_dim, context_dim, 1, bias=False),
        )

        # ── output ────────────────────────────────────────────────────────────
        self.final = nn.Conv2d(C0, 2 if regis else out_channels, 1)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_ctx(self, seg_feat, h, w):
        """Resize seg feature map to (h, w) and return flat token sequence (B, h*w, C)."""
        if seg_feat is None:
            return None
        ctx = F.adaptive_avg_pool2d(seg_feat, (h, w))   # (B, context_dim, h, w)
        return rearrange(ctx, 'b c h w -> b (h w) c')   # (B, h*w, context_dim)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x, times=None, cond=None, seg=None):
        """
        x     : noised input  (B, W, H, Hs)
        cond  : NECT window   (B, W, H, Hs)  — concatenated with x
        seg   : segmentation  (B, seg_ch, H, Hs) — cross-attention context
        """
        if times.dim() == 1:
            times = times.unsqueeze(1)
        t = self.time_pos_emb(times)   # (B, time_embed_dim)

        # project seg to context feature map
        seg_feat = self.seg_proj(seg) if seg is not None else None   # (B, context_dim, H, Hs)

        # ── encoder ──────────────────────────────────────────────────────────
        # seg (=NECT)이 있으면 enc1 early fusion + cross-attn 모두 활용
        # cond만 있으면 enc1 early fusion만
        # 둘 다 없으면 flow path
        if seg is not None and seg.shape[1] == x.shape[1]:
            out = self.encoder1_img(torch.cat([x, seg], dim=1))
        elif cond is not None:
            out = self.encoder1_img(torch.cat([x, cond], dim=1))
        else:
            out = self.encoder1_flow(x)

        out = F.max_pool2d(out, 2);  t1 = out                              # C0
        out = F.max_pool2d(self.encoder2(out), 2);  t2 = out               # C1

        out = self.encoder3(out)
        out = self.st_enc3(out, context=self._get_ctx(seg_feat, *out.shape[-2:]))
        out = F.max_pool2d(out, 2);  t3 = out                              # C2

        out = self.encoder4(out)
        out = self.st_enc4(out, context=self._get_ctx(seg_feat, *out.shape[-2:]))
        out = F.max_pool2d(out, 2);  t4 = out                              # C3

        out = self.encoder5(out)                                            # C4
        out = apply_scale_shift(out, self.time_mlp["enc5"](t))
        out = self.st_enc5(out, context=self._get_ctx(seg_feat, *out.shape[-2:]))

        # ── decoder ──────────────────────────────────────────────────────────
        preds = []

        # stage 4
        out = F.relu(F.interpolate(self.decoder1(out), size=t4.shape[-2:], mode='bilinear', align_corners=False))
        out = apply_scale_shift(out, self.time_mlp["dec4"](t))
        out = self.st_dec1(out, context=self._get_ctx(seg_feat, *out.shape[-2:]))
        out = out + self.AG1(out, t4)

        # stage 3  (DS-1)
        out = F.relu(F.interpolate(self.decoder2(out), size=t3.shape[-2:], mode='bilinear', align_corners=False))
        out = apply_scale_shift(out, self.time_mlp["dec3"](t))
        out = self.st_dec2(out, context=self._get_ctx(seg_feat, *out.shape[-2:]))
        if self.training and self.deep_supervision:
            preds.append(F.interpolate(self.ds1(out), size=x.shape[-2:], mode='bilinear', align_corners=False))
        out = out + self.AG2(out, t3)

        # stage 2  (DS-2)
        out = F.relu(F.interpolate(self.decoder3(out), size=t2.shape[-2:], mode='bilinear', align_corners=False))
        out = apply_scale_shift(out, self.time_mlp["dec2"](t))
        out = self.st_dec3(out, context=self._get_ctx(seg_feat, *out.shape[-2:]))
        if self.training and self.deep_supervision:
            preds.append(F.interpolate(self.ds2(out), size=x.shape[-2:], mode='bilinear', align_corners=False))
        out = out + self.AG3(out, t2)

        # stage 1  (DS-3)
        out = F.relu(F.interpolate(self.decoder4(out), size=t1.shape[-2:], mode='bilinear', align_corners=False))
        out = apply_scale_shift(out, self.time_mlp["dec1"](t))
        if self.training and self.deep_supervision:
            preds.append(F.interpolate(self.ds3(out), size=x.shape[-2:], mode='bilinear', align_corners=False))
        out = out + self.AG4(out, t1)

        # final
        out = F.relu(F.interpolate(self.decoder5(out), size=x.shape[-2:], mode='bilinear', align_corners=False))
        out = apply_scale_shift(out, self.time_mlp["final"](t))
        flow_out = self.final(out)

        if self.training and self.deep_supervision:
            preds.append(flow_out)
            return preds[::-1]   # [final, ds3, ds2, ds1]
        return flow_out
