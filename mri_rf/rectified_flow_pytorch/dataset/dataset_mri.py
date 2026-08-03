from __future__ import annotations
"""
dataset_mri.py
==============
BraTS MRI용 Dataset.  MultiPatientWindowDataset의 MRI(T1/T2) 버전.

입력 폴더 구조:
  {root}/t1/{pid}_t1_{z:04d}.png
  {root}/t2/{pid}_t2_{z:04d}.png

반환 키:
  t1_window  : (W, H, W) 텐서  — window_size 슬라이스
  t2_window  : (W, H, W) 텐서
  center_index: int
  patient_id  : str

Patient ID 추출:
  파일명 예시: Brats18_2013_10_1_t1_0008.png
  stem.rsplit('_', 2)[0]  →  Brats18_2013_10_1
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
from tqdm import tqdm


class MRIWindowDataset(Dataset):
    """
    T1 / T2 MRI 슬라이스 pair를 window 단위로 반환.

    Parameters
    ----------
    root : str | Path
        t1/, t2/ 폴더가 있는 루트 경로.
        예) /home/user/yeong/MRI/train
    image_size : int
        Resize 크기 (정방형).
    window_size : int, 홀수
        컨텍스트 윈도우 크기. 1이면 단일 슬라이스.
    use_cache : bool
        볼륨 .pt 파일 캐시 사용 여부.
    """

    def __init__(
        self,
        root,
        image_size: int,
        window_size: int = 1,
        use_cache: bool = True,
    ):
        super().__init__()
        assert window_size % 2 == 1, "window_size는 홀수여야 합니다."

        self.window_size = window_size
        self.half_window = window_size // 2
        self.image_size  = image_size
        self.use_cache   = use_cache

        root = Path(root)
        self.t1_dir = root / "t1"
        self.t2_dir = root / "t2"
        assert self.t1_dir.exists(), f"t1 폴더 없음: {self.t1_dir}"
        assert self.t2_dir.exists(), f"t2 폴더 없음: {self.t2_dir}"

        # 볼륨 캐시 폴더
        self.cache_root = root / "volume"
        self.cache_t1   = self.cache_root / "t1"
        self.cache_t2   = self.cache_root / "t2"
        self.cache_t1.mkdir(parents=True, exist_ok=True)
        self.cache_t2.mkdir(parents=True, exist_ok=True)

        # 이미지 전처리: [0,1] → [-1,1]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Lambda(lambda t: (t - 0.5) * 2.0),
            T.Lambda(lambda t: t.to(torch.float32)),
        ])

        # ── 환자별 파일 그룹화 ──────────────────────────────────────
        self.patient_groups = self._group_by_patient()

        # ── 볼륨 로드 (캐시 우선) ───────────────────────────────────
        self.patient_data = []

        for pid, paths in tqdm(self.patient_groups.items(), desc="Loading MRI volumes"):
            t1_pt = self.cache_t1 / f"{pid}.pt"
            t2_pt = self.cache_t2 / f"{pid}.pt"

            if use_cache and t1_pt.exists() and t2_pt.exists():
                t1_vol = torch.load(t1_pt)
                t2_vol = torch.load(t2_pt)
            else:
                t1_vol = self._load_volume(paths["t1"])
                t2_vol = self._load_volume(paths["t2"])
                if use_cache:
                    torch.save(t1_vol, t1_pt)
                    torch.save(t2_vol, t2_pt)

            D = min(t1_vol.shape[0], t2_vol.shape[0])
            if D <= window_size:
                continue  # 너무 짧은 볼륨은 스킵

            # t1/t2 슬라이스 수가 다를 경우 짧은 쪽에 맞춤
            if t1_vol.shape[0] != t2_vol.shape[0]:
                t1_vol = t1_vol[:D]
                t2_vol = t2_vol[:D]

            valid_indices = list(range(self.half_window, D - self.half_window))

            self.patient_data.append({
                "patient_id":   pid,
                "t1_volume":    t1_vol,       # (D, H, W)
                "t2_volume":    t2_vol,
                "valid_indices": valid_indices,
            })

        # ── Index map (pid_idx, local_idx) ────────────────────────
        self.index_map = [
            (pid_i, li)
            for pid_i, pdata in enumerate(self.patient_data)
            for li in range(len(pdata["valid_indices"]))
        ]

    # ──────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        pid_i, local_i = self.index_map[idx]
        pdata  = self.patient_data[pid_i]

        center = pdata["valid_indices"][local_i]
        lo = center - self.half_window
        hi = center + self.half_window + 1

        return {
            "t1_window":    pdata["t1_volume"][lo:hi],   # (W, H, W)
            "t2_window":    pdata["t2_volume"][lo:hi],
            "center_index": center,
            "patient_id":   pdata["patient_id"],
        }

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    def _extract_pid(self, stem: str) -> str:
        """
        Brats18_2013_10_1_t1_0008  →  Brats18_2013_10_1
        stem.rsplit('_', 2) 로 마지막 2개(modality, index) 제거.
        """
        return stem.rsplit("_", 2)[0]

    def _group_by_patient(self) -> dict:
        """
        t1/, t2/ 폴더의 PNG를 환자별로 그룹화.
        반환: { pid: {"t1": [Path, ...], "t2": [Path, ...]} }
        """
        groups: dict = {}

        for mod, d in [("t1", self.t1_dir), ("t2", self.t2_dir)]:
            for p in sorted(d.glob("*.png")):
                pid = self._extract_pid(p.stem)
                groups.setdefault(pid, {"t1": [], "t2": []})
                groups[pid][mod].append(p)

        # t1, t2 모두 있는 환자만 유지
        groups = {
            pid: v for pid, v in groups.items()
            if v["t1"] and v["t2"]
        }
        return groups

    def _load_volume(self, paths: list) -> torch.Tensor:
        """PNG 리스트 → (D, H, W) float32 텐서."""
        slices = []
        for p in sorted(paths):
            img = Image.open(p).convert("L")
            img = self.transform(img)   # (1, H, W)
            slices.append(img.squeeze(0))   # (H, W)
        return torch.stack(slices, dim=0)   # (D, H, W)
