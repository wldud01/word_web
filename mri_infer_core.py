"""
mri_infer_core.py
------------------
T1→T2 추론 공용 코어. server.py(로컬 상시 서버)가 쓴다.

GeneratorSPADE는 RectifiedFlow처럼 여러 스텝 ODE로 샘플링하는 모델이 아니라,
학습/검증 스크립트(validation_mri.py)의 reg_mode=3 경로가 보여주듯 단일
forward pass로 쓰는 모델이다: `G(t1, cond=t1, seg=t1)` — 실제 배포 환경에는
등록(registration)용 T2 ground truth가 없으므로 cond/seg 모두 T1 자기 자신을
넣는 self-conditioning으로 추론한다 (검증 스크립트의 `pred_raw`에 해당,
`pred_e2e`/`pred_mind`는 T2 ground truth로 등록한 warp를 조건으로 쓰는
오라클 지표라 실제 추론에는 못 쓴다).

체크포인트의 F_xy/F_yx는 방향이 반대인 별개의 학습된 가중치다
(F_xy = NECT(T1)→CECT(T2), F_yx = 그 반대) — T1→T2를 원하므로 F_xy를 쓴다.
"""

from __future__ import annotations

import io
import os
import sys
import threading
import traceback
import urllib.request
from pathlib import Path

from PIL import Image

BASE       = Path(__file__).parent
MRI_DIR    = BASE / "patient_mri"
MRI_RF_DIR = BASE / "mri_rf"
CKPT_PATH  = MRI_RF_DIR / "checkpoint.88.pt"

# Vercel 서버리스 함수처럼 저장소에 체크포인트를 못 담는 환경에서는
# CHECKPOINT_URL(예: Hugging Face Hub의 파일 URL)에서 받아와 /tmp에 캐싱한다.
# 로컬(server.py)에서는 CKPT_PATH가 이미 존재하므로 다운로드가 필요 없다.
CKPT_URL        = os.environ.get("CHECKPOINT_URL")
CKPT_CACHE_PATH = Path(os.environ.get("CHECKPOINT_CACHE_DIR", "/tmp")) / "checkpoint.88.pt"

IMAGE_SIZE  = 256
WINDOW_SIZE = 1
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


def _resolve_ckpt_path() -> Path | None:
    if CKPT_PATH.exists():
        return CKPT_PATH
    if CKPT_CACHE_PATH.exists() and CKPT_CACHE_PATH.stat().st_size > 0:
        return CKPT_CACHE_PATH
    if CKPT_URL:
        print(f"[INFO] 체크포인트 다운로드 중: {CKPT_URL}")
        CKPT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = CKPT_CACHE_PATH.with_suffix(".tmp")
        urllib.request.urlretrieve(CKPT_URL, tmp_path)
        tmp_path.rename(CKPT_CACHE_PATH)
        print(f"[INFO] 체크포인트 다운로드 완료 ({CKPT_CACHE_PATH.stat().st_size/1e6:.1f}MB)")
        return CKPT_CACHE_PATH
    return None


def _load_model() -> None:
    global MODEL, _model_error
    try:
        ckpt_path = _resolve_ckpt_path()
        if ckpt_path is None:
            _model_error = f"체크포인트 없음: {CKPT_PATH} (CHECKPOINT_URL도 설정되지 않음)"
            print(f"[WARN] {_model_error}")
            return

        import torch
        sys.path.insert(0, str(MRI_RF_DIR))
        from rectified_flow_pytorch.backbone.CycleGanSPADE import GeneratorSPADE

        print(f"[INFO] T1→T2 모델 로딩 중: {ckpt_path}")
        pkg = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model = GeneratorSPADE(input_nc=WINDOW_SIZE, output_nc=WINDOW_SIZE)
        # F_xy = 정방향(T1→T2). strict=True로 정확히 맞는 체크포인트인지 확인한다.
        model.load_state_dict(pkg["F_xy"], strict=True)
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
    }


def wait_for_model_ready(timeout: int = 50) -> None:
    """서버리스 환경용: 에러를 던지지 않고 최대 timeout초까지 로딩 완료(또는 실패)를
    기다리기만 한다. 백그라운드 스레드가 요청 사이에 이어서 실행된다는 보장이 없는
    환경(Vercel Functions)에서, 한 요청 안에서 다운로드+로딩이 끝날 기회를 준다."""
    import time
    waited = 0
    while MODEL is None and _model_error is None and waited < timeout:
        time.sleep(1)
        waited += 1


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


def _run_inference(t1: "torch.Tensor") -> bytes:
    import torch
    with MODEL_LOCK:
        with torch.no_grad():
            # self-conditioning: 실제 추론엔 T2 ground truth로 등록(warp)할 대상이
            # 없으므로 cond와 seg 모두 T1 자기 자신을 넣는다.
            pred = MODEL(t1, cond=t1, seg=t1)
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
    return _run_inference(_png_to_tensor(png_bytes))


def infer_slice(idx: int) -> bytes:
    ensure_model_load_started()
    _wait_for_model()
    if idx in INFER_CACHE:
        return INFER_CACHE[idx]

    mri_path = MRI_PATHS[idx]
    if mri_path.stat().st_size == 0:
        raise RuntimeError(f"MRI 파일이 비어 있습니다: {mri_path.name}")

    png_bytes = _run_inference(_png_to_tensor(mri_path.read_bytes()))
    INFER_CACHE[idx] = png_bytes
    return png_bytes


def find_mri_file(filename: str) -> Path | None:
    candidates = list(MRI_DIR.rglob(filename))
    if candidates:
        return candidates[0]
    direct = MRI_DIR / filename
    return direct if direct.is_file() else None
