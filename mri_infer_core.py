"""
mri_infer_core.py
------------------
T1→T2 추론 공용 코어. server.py(로컬 상시 서버)와 api/[...path].py(Vercel
서버리스 함수)가 공유해서 쓴다.

주의: 지금 들어있는 checkpoint.88.pt는 원래 T2→T1용으로 학습된 가중치다.
실제 T1→T2 체크포인트로 교체하기 전까지는 자리표시자(placeholder)로만 쓴다.
patient_mri/p1의 *_t2_*.png 파일들도 마찬가지로, 진짜 T1 원본 슬라이스가
준비되기 전까지 "원본(T1)"으로 표시하기 위한 자리표시자다.
"""

from __future__ import annotations

import io
import sys
import threading
import traceback
from pathlib import Path

from PIL import Image

BASE       = Path(__file__).parent
MRI_DIR    = BASE / "patient_mri"
MRI_RF_DIR = BASE / "mri_rf"
CKPT_PATH  = MRI_RF_DIR / "checkpoint.88.pt"

IMAGE_SIZE  = 256
WINDOW_SIZE = 1
STEPS       = 5
DEVICE      = "cpu"

MRI_PATHS = sorted(MRI_DIR.rglob("*.png")) if MRI_DIR.exists() else []

MODEL           = None
MODEL_LOCK       = threading.Lock()
INFER_CACHE: dict[int, bytes] = {}
_model_error: str | None = None
_load_started    = False
_load_started_lock = threading.Lock()


def ensure_model_load_started() -> None:
    """모델 로딩을 백그라운드 스레드에서 (최초 1회) 시작시킨다."""
    global _load_started
    with _load_started_lock:
        if _load_started:
            return
        _load_started = True
    threading.Thread(target=_load_model, daemon=True).start()


def _load_model() -> None:
    global MODEL, _model_error
    if not CKPT_PATH.exists():
        _model_error = f"체크포인트 없음: {CKPT_PATH}"
        print(f"[WARN] {_model_error}")
        return
    try:
        import torch
        sys.path.insert(0, str(MRI_RF_DIR))
        from rectified_flow_pytorch.backbone.CycleGanSPADE import GeneratorSPADE
        from rectified_flow_pytorch.rectified_flow_5_pred_img_attn_sup_seg_regis import RectifiedFlow

        print(f"[INFO] T1→T2 모델 로딩 중: {CKPT_PATH}")
        pkg = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
        backbone = GeneratorSPADE(input_nc=WINDOW_SIZE, output_nc=WINDOW_SIZE)
        # RectifiedFlow가 backbone을 `self.model`로 감싸면서 state_dict 키에
        # "model." 접두사가 붙기 때문에, 체크포인트(F_yx, 접두사 없음)는
        # 감싸기 전에 backbone에 직접 로드해야 한다. strict=True로 정확히
        # 맞는 체크포인트인지 확인한다.
        backbone.load_state_dict(pkg["F_yx"], strict=True)
        model = RectifiedFlow(backbone, data_normalize_fn=None, data_unnormalize_fn=None)
        model.eval()
        MODEL = model
        print(f"[INFO] 모델 로드 완료 (params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M)")
    except Exception as e:
        _model_error = str(e)
        traceback.print_exc()
        print(f"[WARN] 모델 로딩 실패: {e}")


def model_status() -> dict:
    return {
        "mri_slices":    len(MRI_PATHS),
        "model_ready":   MODEL is not None,
        "model_error":   _model_error,
        "cached_slices": len(INFER_CACHE),
        "device":        DEVICE,
        "steps":         STEPS,
    }


def _wait_for_model(timeout: int = 300) -> None:
    import time
    waited = 0
    while MODEL is None and _model_error is None and waited < timeout:
        time.sleep(1)
        waited += 1
    if MODEL is None:
        if _model_error:
            raise RuntimeError(f"모델 로딩 실패: {_model_error}")
        raise RuntimeError(f"모델 로딩 시간 초과 ({timeout}초)")


def _run_sample(t2: "torch.Tensor") -> bytes:
    import torch
    with MODEL_LOCK:
        with torch.no_grad():
            pred = MODEL.sample(
                batch_size=1,
                data_shape=(WINDOW_SIZE, IMAGE_SIZE, IMAGE_SIZE),
                noise=t2,
                steps=STEPS,
            )
    pred_np = ((pred.clamp(-1, 1) + 1) / 2 * 255).byte().squeeze().cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(pred_np, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _png_to_tensor(png_bytes: bytes) -> "torch.Tensor":
    import numpy as np
    import torch
    img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    t  = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return (t - 0.5) * 2.0


def infer_from_bytes(png_bytes: bytes) -> bytes:
    ensure_model_load_started()
    _wait_for_model()
    return _run_sample(_png_to_tensor(png_bytes))


def infer_slice(idx: int) -> bytes:
    ensure_model_load_started()
    _wait_for_model()
    if idx in INFER_CACHE:
        return INFER_CACHE[idx]

    mri_path = MRI_PATHS[idx]
    if mri_path.stat().st_size == 0:
        raise RuntimeError(f"MRI 파일이 비어 있습니다: {mri_path.name}")

    png_bytes = _run_sample(_png_to_tensor(mri_path.read_bytes()))
    INFER_CACHE[idx] = png_bytes
    return png_bytes


def find_mri_file(filename: str) -> Path | None:
    candidates = list(MRI_DIR.rglob(filename))
    if candidates:
        return candidates[0]
    direct = MRI_DIR / filename
    return direct if direct.is_file() else None
