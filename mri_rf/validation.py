from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader

# ── 내부 모듈 (경로 확인 필요) ─────────────────────────────────────────────────
from rectified_flow_pytorch.dataset.dataset_mri import MRIWindowDataset
from rectified_flow_pytorch.backbone.mk_unet_img_attn_sup import MKRF_UNet
from rectified_flow_pytorch.rectified_flow_5_pred_img_attn_sup_seg_regis import RectifiedFlow
from rectified_flow_pytorch.utils import MIND
from rectified_flow_pytorch.rectified_flow_mri import collate_mri_window

# ============================================================
# CONFIG
# ============================================================
VAL_ROOT    = Path("/home/user/yeong/MRI/val")
IMAGE_SIZE  = 256
WINDOW_SIZE = 1
STEPS       = 4         # ODE steps (학습 시 설정과 동일하게)
DEVICE      = "cuda"
BASE_SAVE   = Path("./tests_mri_warmup")

EXPERIMENTS = [
    {
        "name":        "mri_warmup_eval_final",
        "ckpt_dir":    "/home/user/yeong/rectified-flow-pytorch/checkpoints/mri/base/",
        "start_ckpt":  80, # 확인하고자 하는 체크포인트 번호
        "end_ckpt":    80,
        "save_images": True,
        "save_every":  5,
    },
]

# ============================================================
# UTILS
# ============================================================
unnorm_01  = lambda t: (t.clamp(-1, 1) + 1) / 2.0
unnorm_255 = lambda t: unnorm_01(t) * 255.0

_psnr_fn = PeakSignalNoiseRatio(data_range=255.0).to(DEVICE)
_ssim_fn = StructuralSimilarityIndexMeasure(data_range=255.0).to(DEVICE)
_mae_fn  = MeanAbsoluteError().to(DEVICE)
_mse_fn  = MeanSquaredError().to(DEVICE)

def compute_metrics(pred, target):
    if pred is None or target is None:
        return (np.nan,) * 5
    p = unnorm_255(pred)
    t = unnorm_255(target)
    psnr = _psnr_fn(p, t).item()
    ssim = _ssim_fn(p, t).item()
    mae  = _mae_fn(p, t).item()
    mse  = _mse_fn(p, t).item()
    nmse = mse / (torch.mean(t ** 2).item() + 1e-12)
    return psnr, ssim, mae, mse, nmse

def nan_mean(lst):
    return np.nanmean(lst, axis=0) if lst else np.full(5, np.nan)

def ensure_csv(path, header):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header)

def _to_jet(arr_01: np.ndarray) -> torch.Tensor:
    rgb = plt.get_cmap("jet")(arr_01)[..., :3]
    return torch.from_numpy(rgb).permute(2, 0, 1).float()

WARMUP_SLICE_HEADER = "patient_id,slice_idx,PSNR_T2,SSIM_T2,MAE_T2,MSE_T2,NMSE_T2,PSNR_T1,SSIM_T1,MAE_T1,MSE_T1,NMSE_T1\n"
WARMUP_EPOCH_HEADER = "model,epoch,PSNR_T2,SSIM_T2,MAE_T2,MSE_T2,NMSE_T2,PSNR_T1,SSIM_T1,MAE_T1,MSE_T1,NMSE_T1\n"

# ============================================================
# MODEL LOADERS
# ============================================================

def build_generator(window_size=WINDOW_SIZE):
    backbone = MKRF_UNet(in_channels=window_size, out_channels=window_size, model=3)
    return RectifiedFlow(backbone, data_normalize_fn=None, data_unnormalize_fn=None)

def build_warmup_models(pkg, window_size=WINDOW_SIZE):
    F_xy = build_generator(window_size)
    F_xy.load_state_dict(pkg["F_xy"], strict=False)
    F_xy = F_xy.to(DEVICE).eval()

    F_yx = build_generator(window_size)
    F_yx.load_state_dict(pkg["F_yx"], strict=False)
    F_yx = F_yx.to(DEVICE).eval()
    return F_xy, F_yx

# ============================================================
# INFERENCE & VISUALIZATION
# ============================================================

@torch.no_grad()
def infer_warmup(t1, t2, F_xy, F_yx, window_size=WINDOW_SIZE):
    W = window_size
    shape = (W, IMAGE_SIZE, IMAGE_SIZE)
    if t1.ndim == 5: t1 = t1.squeeze(2)
    if t2.ndim == 5: t2 = t2.squeeze(2)

    # [핵심] 학습 코드에서 seg 없이 학습했으므로 sample 호출 시에도 seg를 넣지 않음
    # T1 -> T2 (Forward)
    pred_t2_full = F_xy.sample(batch_size=1, data_shape=shape, noise=t1, steps=STEPS + 1)
    pred_t2 = pred_t2_full[:, W // 2:W // 2 + 1] if W > 1 else pred_t2_full

    # T2 -> T1 (Backward)
    pred_t1_full = F_yx.sample(batch_size=1, data_shape=shape, noise=t2, steps=STEPS + 1)
    pred_t1 = pred_t1_full[:, W // 2:W // 2 + 1] if W > 1 else pred_t1_full

    return pred_t2, pred_t1

def save_vis_warmup(t1, t2, pred_t2, pred_t1, path):
    """
    6열 그리드 구성:
    1행: [T1_In] [T2_Gen] [T2_GT] | [T2_In] [T1_Gen] [T1_GT]
    2행: [Blank] [Err_T2]  [Blank] | [Blank] [Err_T1]  [Blank]
    """
    W = WINDOW_SIZE
    t1_c = t1[:, W // 2:W // 2 + 1] if t1.shape[1] > 1 else t1
    t2_c = t2[:, W // 2:W // 2 + 1] if t2.shape[1] > 1 else t2
    
    t1_in = unnorm_01(t1_c).cpu().expand(-1, 3, -1, -1)
    t2_gt = unnorm_01(t2_c).cpu().expand(-1, 3, -1, -1)
    p_t2  = unnorm_01(pred_t2).cpu().expand(-1, 3, -1, -1)
    p_t1  = unnorm_01(pred_t1).cpu().expand(-1, 3, -1, -1)

    def get_err_map(pred, gt):
        diff = torch.abs(unnorm_01(pred) - unnorm_01(gt))[0, 0].cpu().numpy()
        diff = diff / (diff.max() + 1e-6)
        return _to_jet(diff).unsqueeze(0)

    err_t2 = get_err_map(pred_t2, t2_c)
    err_t1 = get_err_map(pred_t1, t1_c)
    blank = torch.zeros_like(t1_in)

    # 1행 (6열)
    row1 = [t1_in, p_t2, t2_gt, t2_gt, p_t1, t1_in]
    # 2행 (6열)
    row2 = [blank, err_t2, blank, blank, err_t1, blank]
    
    all_imgs = torch.cat(row1 + row2, dim=0)
    grid = make_grid(all_imgs, nrow=6, padding=4, pad_value=1) # 흰색 배경 패딩
    
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)

# ============================================================
# RUNNER
# ============================================================

def run_eval(exp: dict, epoch: int):
    name = exp["name"]
    ckpt_path = Path(exp["ckpt_dir"]) / f"checkpoint.{epoch}.pt"
    if not ckpt_path.exists(): return print(f"[SKIP] {ckpt_path} not found")

    save_dir = BASE_SAVE / name / f"epoch_{epoch:04d}"
    ensure_csv(save_dir / "csv" / "slice_metrics.csv", WARMUP_SLICE_HEADER)
    ensure_csv(BASE_SAVE / name / "epoch_metrics.csv", WARMUP_EPOCH_HEADER)

    pkg = torch.load(ckpt_path, map_location="cpu")
    F_xy, F_yx = build_warmup_models(pkg, WINDOW_SIZE)

    val_ds = MRIWindowDataset(root=VAL_ROOT, image_size=IMAGE_SIZE, window_size=WINDOW_SIZE, use_cache=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_mri_window, num_workers=4)

    t2_all, t1_all = [], []
    for idx, batch in enumerate(tqdm(val_dl, desc=f"[{name}] ep{epoch}")):
        t1 = batch["t1_window"].to(DEVICE)
        t2 = batch["t2_window"].to(DEVICE)
        gt_t2 = t2[:, WINDOW_SIZE//2:WINDOW_SIZE//2+1] if WINDOW_SIZE > 1 else t2
        gt_t1 = t1[:, WINDOW_SIZE//2:WINDOW_SIZE//2+1] if WINDOW_SIZE > 1 else t1

        pred_t2, pred_t1 = infer_warmup(t1, t2, F_xy, F_yx, WINDOW_SIZE)
        
        m_t2 = compute_metrics(pred_t2, gt_t2)
        m_t1 = compute_metrics(pred_t1, gt_t1)
        t2_all.append(m_t2)
        t1_all.append(m_t1)

        with open(save_dir / "csv" / "slice_metrics.csv", "a") as f:
            f.write(f"{batch['patient_id'][0]},{batch['center_index'][0].item()}," + 
                    ",".join(f"{v:.6f}" for v in m_t2) + "," + 
                    ",".join(f"{v:.6f}" for v in m_t1) + "\n")
        
        if exp.get("save_images") and idx % exp.get("save_every", 5) == 0:
            save_vis_warmup(t1, t2, pred_t2, pred_t1, 
                            save_dir / "vis" / f"{batch['patient_id'][0]}_sl{batch['center_index'][0].item():04d}.png")

    avg_t2, avg_t1 = nan_mean(t2_all), nan_mean(t1_all)
    with open(BASE_SAVE / name / "epoch_metrics.csv", "a") as f:
        f.write(f"{name},{epoch}," + 
                ",".join(f"{v:.6f}" for v in avg_t2) + "," + 
                ",".join(f"{v:.6f}" for v in avg_t1) + "\n")
    
    print(f"\n--- [Epoch {epoch}] Result ---")
    print(f"T2 PSNR: {avg_t2[0]:.2f}, SSIM: {avg_t2[1]:.4f}")
    print(f"T1 PSNR: {avg_t1[0]:.2f}, SSIM: {avg_t1[1]:.4f}")

def main():
    for exp in EXPERIMENTS:
        for epoch in range(exp["start_ckpt"], exp["end_ckpt"] + 1):
            run_eval(exp, epoch)

if __name__ == "__main__":
    main()