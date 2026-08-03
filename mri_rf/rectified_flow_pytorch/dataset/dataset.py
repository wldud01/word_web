from __future__ import annotations
##########################################################
# dataset classes
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T


class PairedImageWindowDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int, window_size: int = 5):
        """
        root: 데이터 루트 (하위에 NECT/, CECT/ 존재)
        image_size: 이미지 resize 크기
        window_size: 홀수만 허용 (예: 1, 3, 5, ...)
        """
        super().__init__()
        assert window_size % 2 == 1, "window_size는 홀수여야 합니다."
        self.window_size = window_size
        self.half_window = window_size // 2

        root = Path(root)
        self.nect_dir = root / "NECT"
        self.cect_dir = root / "CECT"
        assert self.nect_dir.exists() and self.cect_dir.exists(), "NECT, CECT 폴더가 모두 있어야 합니다."

        # 동일 파일명 기준으로 매칭
        self.nect_paths = sorted(self.nect_dir.glob("*.png"))
        self.cect_paths = sorted(self.cect_dir.glob("*.png"))
        assert len(self.nect_paths) == len(self.cect_paths), "NECT와 CECT 이미지 수가 일치해야 합니다."

        # 유효 인덱스: 가장자리 제외
        self.valid_indices = list(range(self.half_window, len(self.nect_paths) - self.half_window))

        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            # [0,1] → [-1,1] 변환
            T.Lambda(lambda t: (t - 0.5) * 2.0),
            T.Lambda(lambda t: t.to(torch.float32))
        ])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        윈도우 기반 입력 구성:
          window_size=3 → [index-1, index, index+1]
          window_size=5 → [index-2, index-1, index, index+1, index+2]
        환자 별로 가장자리는 제외 (데이터 구성 X)
        
        환자 id 출력: 이후, 환자 별 3d info embedding을 만들기 위해 필요함 / NECT, CECT 둘다
        """
        index = self.valid_indices[idx]

        # 윈도우 인덱스 범위
        indices = list(range(index - self.half_window, index + self.half_window + 1))

        # 🔹 NECT 윈도우 구성
        nect_slices = []
        for i in indices:
            nect_img = Image.open(self.nect_paths[i]).convert("L")
            nect_img = self.transform(nect_img)
            nect_slices.append(nect_img)
        nect_stack = torch.stack(nect_slices, dim=0)  # (C=window_size, H, W)

        # 🔹 CECT 윈도우 구성 (NECT와 동일한 인덱스)
        cect_slices = []
        for i in indices:
            cect_img = Image.open(self.cect_paths[i]).convert("L")
            cect_img = self.transform(cect_img)
            cect_slices.append(cect_img)
        cect_stack = torch.stack(cect_slices, dim=0)  # (C=window_size, H, W)

        return nect_stack, cect_stack

##########################################################
# Multi-patient dataset (patient grouped by filename)
##########################################################

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
from collections import defaultdict
from tqdm import tqdm
import time

def collate_patient_window(batch_list):
    """
    batch_list: list of dicts from MultiPatientWindowDataset.__getitem__
    - window 텐서는 stack
    - volume 텐서는 환자마다 D가 달라서 list로 유지
    """
    out = {}
    out["nect_window"] = torch.stack([b["nect_window"] for b in batch_list], dim=0)  # (B,W,H,W)
    out["cect_window"] = torch.stack([b["cect_window"] for b in batch_list], dim=0)

    # variable-length volumes -> list
    out["nect_volume"] = [b["nect_volume"] for b in batch_list]  # list[(D,H,W)]
    out["cect_volume"] = [b["cect_volume"] for b in batch_list]

    out["center_index"] = torch.tensor([b["center_index"] for b in batch_list], dtype=torch.long)
    out["patient_id"] = [b["patient_id"] for b in batch_list]
    return out

from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MultiPatientWindowDataset(Dataset):
    def __init__(
        self,
        root,
        image_size,
        window_size: int | None = 5,
        use_cache=True,
        patient_level=False,
        seg_root=None,              # 🔹 segmentation root 추가
    ):
        super().__init__()

        self.window_size = window_size
        self.patient_level = patient_level
        self.use_cache = use_cache
        self.image_size = image_size

        if window_size is not None:
            assert window_size % 2 == 1
            self.half_window = window_size // 2

        root = Path(root)
        self.nect_dir = root / "NECT"
        self.cect_dir = root / "CECT"
        assert self.nect_dir.exists() and self.cect_dir.exists()

        # ============================
        # Segmentation directories
        # ============================
        self.use_seg = seg_root is not None
        if self.use_seg:
            seg_root = Path(seg_root)
            self.seg_nect_dir = seg_root / "NECT"
            self.seg_cect_dir = seg_root / "CECT"
            assert self.seg_nect_dir.exists()
            assert self.seg_cect_dir.exists()

        # ============================
        # Cache dirs (volume)
        # ============================
        self.volume_root = root / "volume"
        self.volume_nect_dir = self.volume_root / "NECT"
        self.volume_cect_dir = self.volume_root / "CECT"
        self.volume_nect_dir.mkdir(parents=True, exist_ok=True)
        self.volume_cect_dir.mkdir(parents=True, exist_ok=True)

        if self.use_seg:
            self.volume_seg_nect_dir = self.volume_root / "SEG_NECT"
            self.volume_seg_cect_dir = self.volume_root / "SEG_CECT"
            self.volume_seg_nect_dir.mkdir(parents=True, exist_ok=True)
            self.volume_seg_cect_dir.mkdir(parents=True, exist_ok=True)

        # ============================
        # Transforms
        # ============================
        self.img_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Lambda(lambda t: (t - 0.5) * 2.0),
            T.Lambda(lambda t: t.to(torch.float32)) # 추가, Paired dataset
        ])

        self.seg_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),   # (1,H,W), 값은 0~1
        ])

        # ============================
        # Group by patient
        # ============================
        self.patient_groups = self._group_by_patient()

        # ============================
        # Preload volumes
        # ============================
        self.patient_data = []

        for pid, paths in tqdm(self.patient_groups.items(), desc="Loading volumes"):
            nect_pt = self.volume_nect_dir / f"{pid}.pt"
            cect_pt = self.volume_cect_dir / f"{pid}.pt"

            if use_cache and nect_pt.exists():
                nect_vol = torch.load(nect_pt)
                cect_vol = torch.load(cect_pt)
            else:
                nect_vol = self._load_volume(paths["NECT"], self.img_transform)
                cect_vol = self._load_volume(paths["CECT"], self.img_transform)
                if use_cache:
                    torch.save(nect_vol, nect_pt)
                    torch.save(cect_vol, cect_pt)

            D = nect_vol.shape[0]

            # ----------------------------
            # Segmentation volume
            # ----------------------------
            if self.use_seg:
                seg_nect_pt = self.volume_seg_nect_dir / f"{pid}.pt"
                seg_cect_pt = self.volume_seg_cect_dir / f"{pid}.pt"

                if use_cache and seg_nect_pt.exists():
                    seg_nect_vol = torch.load(seg_nect_pt)
                    seg_cect_vol = torch.load(seg_cect_pt)
                else:
                    seg_nect_vol = self._load_seg_volume(pid, "NECT", D)
                    seg_cect_vol = self._load_seg_volume(pid, "CECT", D)
                    if use_cache:
                        torch.save(seg_nect_vol, seg_nect_pt)
                        torch.save(seg_cect_vol, seg_cect_pt)
            else:
                seg_nect_vol = None
                seg_cect_vol = None

            # ----------------------------
            # Valid indices
            # ----------------------------
            if window_size is None:
                valid_indices = None
            else:
                if D <= window_size:
                    continue
                valid_indices = list(range(self.half_window, D - self.half_window))

            self.patient_data.append({
                "patient_id": pid,
                "nect_volume": nect_vol,            # (D,H,W)
                "cect_volume": cect_vol,
                "seg_nect_volume": seg_nect_vol,    # (D,1,H,W) or None
                "seg_cect_volume": seg_cect_vol,
                "valid_indices": valid_indices,
            })

        # ============================
        # Index map
        # ============================
        self.index_map = []

        if patient_level or window_size is None:
            for pid in range(len(self.patient_data)):
                self.index_map.append((pid, None))
        else:
            for pid, pdata in enumerate(self.patient_data):
                for li in range(len(pdata["valid_indices"])):
                    self.index_map.append((pid, li))

    # -------------------------------------------------
    def __len__(self):
        return len(self.index_map)

    # -------------------------------------------------
    def __getitem__(self, idx):
        pid, local_i = self.index_map[idx]
        pdata = self.patient_data[pid]

        nect_vol = pdata["nect_volume"]
        cect_vol = pdata["cect_volume"]
        seg_nect_vol = pdata["seg_nect_volume"]
        seg_cect_vol = pdata["seg_cect_volume"]

        # ============================
        # Patient-level return
        # ============================
        if self.window_size is None or self.patient_level:
            return {
                "nect_volume": nect_vol,
                "cect_volume": cect_vol,
                "seg_nect_volume": seg_nect_vol,
                "seg_cect_volume": seg_cect_vol,
                "patient_id": pdata["patient_id"],
                "num_slices": nect_vol.shape[0],
            }

        # ============================
        # Window-level return
        # ============================
        center = pdata["valid_indices"][local_i]
        lo = center - self.half_window
        hi = center + self.half_window + 1

        sample = {
            "nect_window": nect_vol[lo:hi],          # (W,H,W)
            "cect_window": cect_vol[lo:hi],
            "center_index": center,
            "patient_id": pdata["patient_id"],
        }

        if self.use_seg:
            sample.update({
                "seg_nect_window": seg_nect_vol[lo:hi],   # (W,1,H,W)
                "seg_cect_window": seg_cect_vol[lo:hi],
            })

        return sample

    # =================================================
    # Helper functions
    # =================================================
    def _group_by_patient(self):
        groups = {}
        for p in sorted(self.nect_dir.glob("*.png")):
            pid = p.name.split("_")[0]
            groups.setdefault(pid, {"NECT": [], "CECT": []})
            groups[pid]["NECT"].append(p)

        for p in sorted(self.cect_dir.glob("*.png")):
            pid = p.name.split("_")[0]
            groups.setdefault(pid, {"NECT": [], "CECT": []})
            groups[pid]["CECT"].append(p)

        return groups

    def _load_volume(self, paths, transform):
        slices = []
        for p in sorted(paths):
            img = Image.open(p).convert("L")
            img = transform(img)
            slices.append(img.squeeze(0))   # (H,W)
        return torch.stack(slices, dim=0)  # (D,H,W)

    def _load_seg_volume(self, pid, modality, D):
        seg_dir = self.seg_nect_dir if modality == "NECT" else self.seg_cect_dir
        seg_slices = []

        for i in range(1, D + 1):
            fname = f"{pid}_{modality}_{i:04d}.png"
            fpath = seg_dir / fname

            if not fpath.exists():
                seg_slices.append(torch.zeros(1, self.image_size, self.image_size))
                continue

            img = Image.open(fpath).convert("L")
            img = self.seg_transform(img)   # (1,H,W)
            seg_slices.append(img)

        return torch.stack(seg_slices, dim=0)  # (D,1,H,W)


