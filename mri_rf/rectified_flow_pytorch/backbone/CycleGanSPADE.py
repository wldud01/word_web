"""
CycleGanSPADE.py
-----------------
RegGAN ResNet Generator + Time Embedding + SPADE conditioning

아키텍처:
  Encoder : cond(NECT)를 입력으로 받아 특징 추출
  Bottleneck : 9개 ResidualBlock에 time embedding 주입(AdaIN)
  Decoder : ConvTranspose 업샘플링 후 SPADE로 seg(nect_warped) 조건화

RectifiedFlow 인터페이스와 호환:
  forward(x, times=None, cond=None, seg=None)
  - x     : noised 입력 (사용하지 않고 cond를 encoder 입력으로 씀)
  - times : 시간 스텝 (B,) or (B,1)
  - cond  : NECT 원본 → encoder 입력
  - seg   : warped NECT → SPADE 조건
  출력    : velocity field (flow), unbounded
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Time-conditioned Residual Block ─────────────────────────────────────────

class ResBlockTime(nn.Module):
    """RegGAN ResidualBlock + AdaIN time conditioning."""

    def __init__(self, channels: int, time_embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )
        # AdaIN: time → (scale, shift) for each channel
        self.time_proj = nn.Linear(time_embed_dim, channels * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None):
        h = self.conv(x)
        if t_emb is not None:
            params = self.time_proj(t_emb)              # (B, 2C)
            scale, shift = params.chunk(2, dim=-1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            h = h * (1.0 + scale) + shift
        return x + h


# ─── Generator with SPADE ────────────────────────────────────────────────────

class GeneratorSPADE(nn.Module):
    """
    RegGAN ResNet Generator with time embedding + SPADE decoder conditioning.

    Parameters
    ----------
    input_nc : int
        Input channels (NECT, typically 1).
    output_nc : int
        Output channels (flow velocity, same as input_nc).
    n_residual_blocks : int
        Number of residual blocks in bottleneck.
    time_embed_dim : int
        Sinusoidal time embedding dimension.
    spade_cond_channels : int
        Channels of the SPADE conditioning image (seg = warped NECT, typically 1).
    spade_hidden_channels : int
        Hidden channels in SPADE MLP.
    """

    def __init__(
        self,
        input_nc: int = 1,
        output_nc: int = 1,
        n_residual_blocks: int = 9,
        time_embed_dim: int = 256,
        spade_cond_channels: int = 1,
        spade_hidden_channels: int = 64,
    ):
        super().__init__()

        # ── Time embedding ──────────────────────────────────────────────────
        self.time_pos_emb = SinusoidalPosEmb(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.GELU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # ── Encoder (입력: cond = NECT) ──────────────────────────────────────
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ── Bottleneck: time-conditioned residual blocks ─────────────────────
        self.res_blocks = nn.ModuleList([
            ResBlockTime(256, time_embed_dim)
            for _ in range(n_residual_blocks)
        ])

        # ── Decoder with SPADE ───────────────────────────────────────────────
        self.up1_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.spade1 = SPADE2d(
            128,
            cond_channels=spade_cond_channels,
            hidden_channels=spade_hidden_channels,
        )

        self.up2_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.spade2 = SPADE2d(
            64,
            cond_channels=spade_cond_channels,
            hidden_channels=spade_hidden_channels,
        )

        # ── Output: image translation → Tanh to [-1, 1] ────────────────────
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh(),
        )

    # ─── forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x    : noised (사용하지 않음; cond를 encoder 입력으로 사용)
        times: (B,) or (B,1) time step
        cond : NECT 원본 → encoder 입력
        seg  : warped NECT → SPADE 조건
        """
        # ── Time embedding ──────────────────────────────────────────────────
        if times is not None:
            if times.dim() == 0:
                times = times.unsqueeze(0)
            if times.dim() == 1:
                times = times.unsqueeze(1)          # (B, 1)
            t = self.time_pos_emb(times)            # (B, 1, D)
            t = t.squeeze(1)                        # (B, D)
            t_emb = self.time_mlp(t)               # (B, D)
        else:
            t_emb = None

        # ── Encoder: cond(NECT)를 입력으로 사용 ─────────────────────────────
        enc_in = cond if cond is not None else x
        h = self.head(enc_in)
        h = self.down1(h)
        h = self.down2(h)

        # ── Bottleneck ──────────────────────────────────────────────────────
        for res_block in self.res_blocks:
            h = res_block(h, t_emb)

        # ── Decoder + SPADE ─────────────────────────────────────────────────
        h = self.up1_conv(h)
        h = self.spade1(h, seg)

        h = self.up2_conv(h)
        h = self.spade2(h, seg)

        return self.output_layer(h)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(
            torch.arange(half, device=device)
            * -(torch.log(torch.tensor(10000.0, device=device)) / (half - 1))
        )
        emb = t * emb
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def apply_scale_shift(x, t_emb):
    t = t_emb.unsqueeze(-1).unsqueeze(-1)
    scale, shift = t.chunk(2, dim=1)
    return x * (scale + 1) + shift


class SPADE2d(nn.Module):
    def __init__(self, num_features, cond_channels=1, hidden_channels=128, norm_type="instance"):
        super().__init__()
        if norm_type == "instance":
            self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=1e-5)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm2d(num_features, affine=False, eps=1e-5)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden_channels, num_features, kernel_size=3, padding=1)
        self.mlp_beta  = nn.Conv2d(hidden_channels, num_features, kernel_size=3, padding=1)
        # gate starts at 0 so modulation begins as identity
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.mlp_gamma.weight)
        nn.init.zeros_(self.mlp_gamma.bias)
        nn.init.zeros_(self.mlp_beta.weight)
        nn.init.zeros_(self.mlp_beta.bias)

    def forward(self, x, cond):
        if cond is None:
            return x
        if cond.ndim == 5 and cond.shape[2] == 1:
            cond = cond.squeeze(2)
        cond_resized = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
        h = self.mlp_shared(cond_resized)
        gamma = self.mlp_gamma(h)
        beta  = self.mlp_beta(h)
        spade_out = self.norm(x) * (1.0 + gamma) + beta
        return x + torch.tanh(self.gate) * spade_out
