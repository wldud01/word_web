from __future__ import annotations
"""
rectified_flow_mri.py
=====================
T1 → T2 MRI 변환용 학습 파이프라인.

training_mode = 1, registration_mode 지원:
  - 2 : MIND descriptor 기반 정합 (구조적 지문)
  - 6 : Raw image 기반 Cycle 정합

CT 버전에서 MRI 불필요 요소 제거:
  seg, F_xx/F_xx_, F_yx_(pretrain), CT validator
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from ema_pytorch import EMA
import lpips

# ── 내부 모듈 ─────────────────────────────────────────────────────
from rectified_flow_pytorch.rectified_flow_5_pred_img_attn_sup_seg_regis import (
    RectifiedFlow,
    MSELoss,
)
from rectified_flow_pytorch.reg import Reg
from rectified_flow_pytorch.transformer import Transformer_2D
from rectified_flow_pytorch.utils import (
    smooothing_loss,
    default,
    save_sample,
    divisible_by,
    set_random_seed,
    MIND,
    MINDLoss,
)
from rectified_flow_pytorch.dataset.dataset_mri import MRIWindowDataset

set_random_seed(42)


# ──────────────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────────────

def collate_mri_window(batch_list):
    out = {
        "t1_window":    torch.stack([b["t1_window"]    for b in batch_list], dim=0),
        "t2_window":    torch.stack([b["t2_window"]    for b in batch_list], dim=0),
        "center_index": torch.tensor([b["center_index"] for b in batch_list], dtype=torch.long),
        "patient_id":   [b["patient_id"] for b in batch_list],
    }
    return out


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


# ──────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────

class MRITrainer(Module):
    """
    T1 → T2 MRI 변환 학습기.

    Parameters
    ----------
    forward_model     : RectifiedFlow  — F_xy (T1 → T2)
    backward_model    : RectifiedFlow  — F_yx (T2 → T1)
    registration_mode : int
        2 = MIND descriptor 기반 정합
        6 = Raw image + Cycle 정합
    """

    def __init__(
        self,
        forward_model: RectifiedFlow,
        backward_model: RectifiedFlow,
        *,
        dataset: dict | MRIWindowDataset,
        registration_mode: int = 2,
        num_train_steps: int = 1_000_000,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        checkpoints_folder: str = "./checkpoints_mri",
        results_folder: str = "./results_mri",
        checkpoint_every: int = 1,
        max_grad_norm: float = 0.5,
        warmup_epochs: int = 30,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        use_ema: bool = True,
        window_size: int = 1,
        save_root: str = "checkpoints_mri",
    ):
        super().__init__()
        assert registration_mode in (2, 6), \
            f"registration_mode must be 2 or 6, got {registration_mode}"

        self.accelerator = Accelerator(**accelerate_kwargs)
        self.registration_mode = registration_mode

        # ── Dataset ─────────────────────────────────────────────
        if isinstance(dataset, dict):
            dataset = MRIWindowDataset(**dataset)

        self.dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_mri_window,
            num_workers=4,
            pin_memory=True,
        )
        self.window_size = window_size

        # ── Models ──────────────────────────────────────────────
        self.F_xy = forward_model    # T1 → T2
        self.F_yx = backward_model   # T2 → T1

        # R_A 입력 채널: mode 2 = MIND 9ch, mode 6 = raw 1ch
        in_ch = 9 if registration_mode == 2 else 1
        self.R_A = Reg(256, 256, in_ch, in_ch, dim=2).cuda()
        self.spatial_transform = Transformer_2D().cuda()

        # ── EMA ─────────────────────────────────────────────────
        use_ema &= not getattr(self.F_xy, "use_consistency", False)
        self.use_ema = use_ema
        self.ema_model = None
        if self.is_main and use_ema:
            self.ema_model = EMA(
                self.F_xy,
                forward_method_names=("sample",),
                **ema_kwargs,
            )
            self.ema_model.to(self.accelerator.device)

        # ── Optimizer ───────────────────────────────────────────
        self.optimizer = Adam([
            {"params": self.F_xy.parameters(), "lr": learning_rate},
            {"params": self.F_yx.parameters(), "lr": learning_rate},
            {"params": self.R_A.parameters(),  "lr": 1e-4},
        ])

        # ── Accelerate ──────────────────────────────────────────
        (
            self.F_xy,
            self.F_yx,
            self.R_A,
            self.optimizer,
            self.dl,
        ) = self.accelerator.prepare(
            self.F_xy, self.F_yx, self.R_A, self.optimizer, self.dl
        )

        # ── Training settings ───────────────────────────────────
        self.num_train_steps    = num_train_steps
        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder     = Path(results_folder)
        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.checkpoint_every = checkpoint_every
        self.max_grad_norm    = float(max_grad_norm)
        self.warmup_epochs    = warmup_epochs
        self.epoch_start      = 1
        self.save_root        = save_root

        # ── Loss weights ────────────────────────────────────────
        self.w_rf         = 1.0
        self.w_rec        = 0.8
        self.w_ssim       = 0.1
        self.w_sm         = 1e-4
        self.w_align      = 1.0
        self.w_regularize = 0.2   # mode 6: LPIPS texture

        self.ssim_loss_fn = StructuralSimilarityIndexMeasure(
            data_range=2.0
        ).to(self.accelerator.device)

        # ── Mode-specific modules ────────────────────────────────
        if registration_mode == 2:
            self.mind_extractor = MIND(
                non_local_region_size=3,
                patch_size=7,
                neighbor_size=3,
                gaussian_patch_sigma=3.0,
            ).cuda()
            self.mind_loss_fn = MINDLoss(non_local_region_size=3).cuda()
            self.lpips_fn = None

        elif registration_mode == 6:
            self.mind_extractor = None
            self.mind_loss_fn   = None
            self.lpips_fn = lpips.LPIPS(net="vgg").to(self.accelerator.device)
            self.lpips_fn.eval()

    # ────────────────────────────────────────────────────────────
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # ────────────────────────────────────────────────────────────
    def save(self, path):
        if not self.is_main:
            return
        pkg = {
            "F_xy":              self.accelerator.unwrap_model(self.F_xy).state_dict(),
            "F_yx":              self.accelerator.unwrap_model(self.F_yx).state_dict(),
            "RA":                self.accelerator.unwrap_model(self.R_A).state_dict(),
            "optimizer":         self.optimizer.state_dict(),
            "registration_mode": self.registration_mode,
        }
        if self.ema_model is not None:
            pkg["ema_model"] = self.ema_model.state_dict()
        torch.save(pkg, str(self.checkpoints_folder / path))

    def load(self, path):
        self.epoch_start = int(path.split(".")[-2]) + 1
        pkg = torch.load(path, map_location=self.accelerator.device)
        self.accelerator.unwrap_model(self.F_xy).load_state_dict(pkg["F_xy"])
        self.accelerator.unwrap_model(self.F_yx).load_state_dict(pkg["F_yx"])
        self.accelerator.unwrap_model(self.R_A).load_state_dict(pkg["RA"])
        self.optimizer.load_state_dict(pkg["optimizer"])
        if "ema_model" in pkg and self.ema_model is not None:
            self.ema_model.load_state_dict(pkg["ema_model"])
        print(f"[✓] Checkpoint loaded: {path}")

    # ────────────────────────────────────────────────────────────
    def _set_requires_grad(self, modules, flag: bool):
        if not isinstance(modules, (list, tuple)):
            modules = [modules]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = flag

    # ────────────────────────────────────────────────────────────
    def forward(self):
        num_epochs = math.ceil(self.num_train_steps / len(self.dl))
        self.accelerator.print(
            f"[INFO] reg_mode={self.registration_mode} | "
            f"{num_epochs} epochs | warmup={self.warmup_epochs}"
        )

        self.accelerator.wait_for_everyone()
        if self.is_main and wandb.run is None:
            wandb.init(
                project="MRI_T1_to_T2_RectifiedFlow",
                name=f"reg{self.registration_mode}_run_{np.random.randint(10000)}",
                config={
                    "steps": self.num_train_steps,
                    "registration_mode": self.registration_mode,
                },
                reinit=True,
            )
        self.accelerator.wait_for_everyone()

        global_step = getattr(self, "global_step", 0)

        for epoch in range(self.epoch_start, num_epochs + 1):

            in_warmup = epoch <= self.warmup_epochs

            self.F_xy.train(); self.F_yx.train(); self.R_A.train()

            if in_warmup:
                self._set_requires_grad([self.F_xy, self.F_yx], True)
                self._set_requires_grad(self.R_A, False)
            else:
                self._set_requires_grad([self.F_xy, self.F_yx, self.R_A], True)

            epoch_loss = 0.0
            pbar = tqdm(self.dl, desc=f"[Epoch {epoch}]", ncols=120)

            for batch in pbar:
                global_step += 1

                t1 = batch["t1_window"].to(self.accelerator.device, dtype=torch.float32)
                t2 = batch["t2_window"].to(self.accelerator.device, dtype=torch.float32)

                if t1.ndim == 5: t1 = t1.squeeze(2)
                if t2.ndim == 5: t2 = t2.squeeze(2)

                # ================================================
                # WARMUP: F_yx(T2→T1) + F_xy(T1→T2)
                # ================================================
                if in_warmup:
                    L_yx, pred_t1, _ = self.F_yx(t1, noise=t2)   # x0=T2 → T1
                    L_t1 = F.l1_loss(pred_t1, t1)

                    L_xy, pred_t2, _ = self.F_xy(t2, noise=t1)   # x0=T1 → T2
                    L_cycle = F.l1_loss(pred_t2, t2)

                    loss = L_yx + L_xy + 0.8 * (L_t1 + L_cycle)

                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(
                        list(self.F_xy.parameters()) + list(self.F_yx.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    epoch_loss += loss.item()
                    pbar.set_postfix({"phase": "warmup", "loss": f"{loss.item():.4f}"})
                    wandb.log(
                        {"train/warmup/L_yx": L_yx.item(), "train/warmup/L_t1": L_t1.item(),
                         "epoch": epoch},
                        step=global_step,
                    )
                    if self.is_main and global_step % 1000 == 0:
                        with torch.no_grad():
                            sd = str(self.results_folder / f"warmup/epoch_{epoch}/")
                            save_sample(sd, t1, global_step, "t1")
                            save_sample(sd, t2, global_step, "t2")
                            save_sample(sd, pred_t1, global_step, "pred_t1")
                    continue

                # ================================================
                # FULL PHASE
                # ================================================

                # ── reg_mode 2: MIND descriptor ────────────────
                if self.registration_mode == 2:

                    reg_src = self.mind_extractor(t1)  # (B, 9, H, W)
                    reg_tgt = self.mind_extractor(t2)

                    Trans     = self.R_A(reg_src, reg_tgt)
                    t1_warped = self.spatial_transform(t1, Trans)

                    L_align = self.mind_loss_fn(t1_warped, t2)
                    sm_loss = smooothing_loss(Trans)

                    L_rf, pred_t2, _ = self.F_xy(t2, noise=t1_warped)  # x0=T1_warped → T2
                    L_rec  = F.l1_loss(pred_t2, t2)
                    L_ssim = 1.0 - self.ssim_loss_fn(pred_t2, t2)

                    total_loss = (
                        self.w_rf    * L_rf
                        + self.w_rec   * L_rec
                        + self.w_ssim  * L_ssim
                        + self.w_align * L_align
                        + self.w_sm    * sm_loss
                    )

                    wandb.log(
                        {"train/loss": total_loss.item(), "train/L_rf": L_rf.item(),
                         "train/L_rec": L_rec.item(), "train/L_ssim": L_ssim.item(),
                         "train/L_align": L_align.item(), "train/sm_loss": sm_loss.item(),
                         "epoch": epoch},
                        step=global_step,
                    )

                # ── reg_mode 6: Raw image + Cycle ──────────────
                elif self.registration_mode == 6:

                    # Step 1: F_yx로 pseudo T1 생성 (정합 기준점)
                    with torch.no_grad():
                        _, pred_t1, _ = self.F_yx(t1, noise=t2)  # x0=T2 → T1

                    # Forward registration: T1 → pred_T1 방향 정합
                    Trans     = self.R_A(t1, pred_t1)
                    t1_warped = self.spatial_transform(t1, Trans)

                    L_align     = F.l1_loss(t1_warped, pred_t1)
                    sm_loss     = smooothing_loss(Trans)
                    L_tex_shape = self.lpips_fn(t1_warped, t1).mean()

                    # Forward generation: T1_warped → T2
                    L_rf, pred_t2, _ = self.F_xy(t2, noise=t1_warped)  # x0=T1_warped → T2
                    L_rec  = F.l1_loss(pred_t2, t2)
                    L_ssim = 1.0 - self.ssim_loss_fn(pred_t2, t2)

                    # Cycle backward: pred_T2 → T1 재생성
                    L_rf_back, pred_t1_hat, _ = self.F_yx(t1, noise=pred_t2)  # x0=pred_T2 → T1

                    # Backward registration: pred_T1_hat → T1 방향 정합
                    Trans_back      = self.R_A(pred_t1_hat, t1)
                    t1_hat_warped   = self.spatial_transform(pred_t1_hat, Trans_back)
                    L_align_hat     = F.l1_loss(t1_hat_warped, t1)
                    sm_loss_back    = smooothing_loss(Trans_back)

                    total_loss = (
                        self.w_rf         * (L_rf + L_rf_back)
                        + self.w_rec      * L_rec
                        + self.w_ssim     * L_ssim
                        + self.w_align    * (L_align + L_align_hat)
                        + self.w_sm       * (sm_loss + sm_loss_back)
                        + self.w_regularize * L_tex_shape
                    )

                    wandb.log(
                        {"train/loss": total_loss.item(), "train/L_rf": L_rf.item(),
                         "train/L_rf_back": L_rf_back.item(),
                         "train/L_rec": L_rec.item(), "train/L_ssim": L_ssim.item(),
                         "train/L_align": L_align.item(), "train/L_align_hat": L_align_hat.item(),
                         "train/sm_loss": sm_loss.item(), "train/L_tex_shape": L_tex_shape.item(),
                         "epoch": epoch},
                        step=global_step,
                    )

                # ── backward & update ──────────────────────────
                self.accelerator.backward(total_loss)
                self.accelerator.clip_grad_norm_(
                    list(self.F_xy.parameters())
                    + list(self.F_yx.parameters())
                    + list(self.R_A.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += total_loss.item()
                pbar.set_postfix({"phase": "full", "loss": f"{total_loss.item():.4f}"})

                if self.is_main and global_step % 100 == 0:
                    with torch.no_grad():
                        sd = str(self.results_folder / f"epoch_{epoch}/")
                        save_sample(sd, t1,        global_step, "t1")
                        save_sample(sd, t2,        global_step, "t2")
                        save_sample(sd, t1_warped, global_step, "t1_warped")
                        save_sample(sd, pred_t2,   global_step, "pred_t2")

            # ── epoch end ─────────────────────────────────────
            avg_loss = epoch_loss / len(self.dl)
            if self.is_main:
                wandb.log({"train/loss_avg": avg_loss, "epoch": epoch})
            self.accelerator.print(f"[Epoch {epoch}] Avg Loss {avg_loss:.4f}")

            if self.is_main and divisible_by(epoch, self.checkpoint_every):
                self.save(f"checkpoint.{epoch}.pt")

        self.accelerator.print("Training finished.")
