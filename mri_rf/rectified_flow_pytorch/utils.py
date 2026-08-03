#import wandb
# helpers
from __future__ import annotations

import numpy as np
import random
from pathlib import Path
from torchvision.utils import save_image

import torch
from torch import nn, pi, cat, stack, from_numpy

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

# normalizing helpers

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# noise schedules

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))

def cosine_time_warp(t: torch.Tensor):
    # t in [0,1] → t' in [0,1], sin^2 커브 (cos^2와 보완적)
    return torch.sin((pi / 2) * t) ** 2

def set_random_seed(SEED=1234):
    """
    재현성을 위한 시드 고정
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def divisible_by(num, den):
    return (num % den) == 0


def save_sample(save_dir, slice, step, name):                                            
    with torch.no_grad():
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        center = slice.shape[1] // 2
        unnorm_01  = lambda t: (t.clamp(-1, 1) + 1) / 2.0
        unnorm_255 = lambda t: unnorm_01(t) * 255.0
        
        save_image(
            unnorm_01(slice[:, center:center+1]),
            save_dir / f"step_{step}_{name}.png"
        )

def collate_patient_window(batch_list):
    """
    Supports:
      1) window-level batch  (nect_window)
      2) patient-level batch (nect_volume)
    """

    out = {}

    # ===============================
    # CASE 1: window-based
    # ===============================
    if "nect_window" in batch_list[0]:
        out["nect_window"] = torch.stack(
            [b["nect_window"] for b in batch_list], dim=0
        )  # (B,W,H,W)

        out["cect_window"] = torch.stack(
            [b["cect_window"] for b in batch_list], dim=0
        )

        out["center_index"] = torch.tensor(
            [b["center_index"] for b in batch_list], dtype=torch.long
        )

    # ===============================
    # CASE 2: patient-level (FULL volume)
    # ===============================
    elif "nect_volume" in batch_list[0]:
        #  variable depth → batch size must be 1
        assert len(batch_list) == 1, \
            "Patient-level volume batching requires batch_size=1"

        b = batch_list[0]
        out["nect_volume"] = b["nect_volume"].unsqueeze(0)  # (1,D,H,W)
        out["cect_volume"] = b["cect_volume"].unsqueeze(0)
        out["num_slices"] = b["num_slices"]

    else:
        raise KeyError("Unknown batch format keys: " + str(batch_list[0].keys()))

    # common
    out["patient_id"] = [b["patient_id"] for b in batch_list]

    return out


def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx*dx
    dy = dy*dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d 
    return d

def get_registration_losses(flow):
    # flow shape: [B, 3, D, H, W]
    b, c, d, h, w = flow.shape

    # 1. Smoothing Loss (공간 내 매끄러움, xy 평면)
    dy = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
    dx = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
    dz = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    l_smooth = torch.mean(dy) + torch.mean(dx) + torch.mean(dz)

    # 2. Trajectory Continuity (2차 차분)
    # dx, dy에 대해서만 계산
    xy_flow = flow[:, :2, :, :, :]
    
    # 가속도(Acceleration)를 억제
    accel = xy_flow[:, :, 2:, :, :] - 2 * xy_flow[:, :, 1:-1, :, :] + xy_flow[:, :, :-2, :, :]
    l_trajectory = torch.mean(accel**2)

    return l_smooth, l_trajectory



import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*vgg[:4])   # relu1_2
        self.slice2 = nn.Sequential(*vgg[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*vgg[9:16]) # relu3_3
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x shape: [B, D, H, W] (4차원) 또는 [B, C, D, H, W] (5차원)
        # 에러 로그상 [4, 5, 256, 256]이 들어오므로 4차원 케이스임
        
        if x.dim() == 4:
            B, D, H, W = x.shape
            # D(윈도우)를 배치 차원으로 보내서 [B*D, 1, H, W]로 만듦
            x = x.view(B * D, 1, H, W)
            is_windowed = True
        elif x.dim() == 5:
            B, C, D, H, W = x.shape
            # [B, C, D, H, W] -> [B*D, C, H, W]
            x = x.transpose(1, 2).reshape(B * D, C, H, W)
            is_windowed = True
        else:
            is_windowed = False

        # VGG는 3채널 입력을 받으므로 1채널(의료영상)일 경우 복제
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) != 3:
            # 혹시라도 채널이 3이나 1이 아니면 에러 방지를 위해 강제 조정 (예: 5채널 등)
            # 여기서는 윈도우 차원이 채널로 잘못 인식된 경우를 대비
            pass 

        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)

        features = [h1, h2, h3]

        # 다시 원래의 [B, D, C, H, W] 구조로 복원하여 리턴
        if is_windowed:
            restored_features = []
            for f in features:
                f_B_D, f_C, f_H, f_W = f.shape
                restored_features.append(f.view(B, D, f_C, f_H, f_W))
            return restored_features
        
        return features
    
class HardNegativePatchNCELoss(nn.Module):
    """윈도우 차원 내의 모든 패치 또는 특정 샘플링된 패치에 대해 NCE Loss 계산"""
    def __init__(self, tau=0.07, num_patches=256):
        super().__init__()
        self.tau = tau
        self.num_patches = num_patches
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, feat_anchor, feat_pos, feat_neg_hard):
        """
        각 입력은 [B, D, C, H, W] 형상의 리스트 (VGG 레이어별 특징)
        """
        loss = 0.0
        for fa, fp, fn in zip(feat_anchor, feat_pos, feat_neg_hard):
            # 5D 대응: [B, D, C, H, W] -> [B, D*H*W, C]
            B, D, C, H, W = fa.shape
            
            fa = fa.reshape(B, D * H * W, C)
            fp = fp.reshape(B, D * H * W, C)
            fn = fn.reshape(B, D * H * W, C)

            # 윈도우 전체 공간(D*H*W)에서 랜덤 패치 샘플링
            sample_ids = torch.randperm(D * H * W, device=fa.device)[:self.num_patches]
            
            fa_sampled = F.normalize(fa[:, sample_ids, :], p=2, dim=-1)
            fp_sampled = F.normalize(fp[:, sample_ids, :], p=2, dim=-1)
            fn_sampled = F.normalize(fn[:, sample_ids, :], p=2, dim=-1)

            # Positive/Negative Logits 계산
            # [B, num_patches]
            l_pos = torch.einsum('b n c, b n c -> b n', fa_sampled, fp_sampled).unsqueeze(-1)
            l_neg = torch.einsum('b n c, b n c -> b n', fa_sampled, fn_sampled).unsqueeze(-1)

            logits = torch.cat([l_pos, l_neg], dim=-1) / self.tau
            logits = logits.view(-1, 2)
            
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
            loss += self.cross_entropy_loss(logits, labels)

        return loss / len(feat_anchor)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import math


class MIND(torch.nn.Module):

    def __init__(self, non_local_region_size =9, patch_size =7, neighbor_size =5, gaussian_patch_sigma =3.0):
        super(MIND, self).__init__()
        self.nl_size =non_local_region_size
        self.p_size =patch_size
        self.n_size =neighbor_size
        self.sigma2 =gaussian_patch_sigma *gaussian_patch_sigma


        # calc shifted images in non local region
        self.image_shifter =torch.nn.Conv2d(in_channels =1, out_channels =self.nl_size *self.nl_size,
                                            kernel_size =(self.nl_size, self.nl_size),
                                            stride=1, padding=((self.nl_size-1)//2, (self.nl_size-1)//2),
                                            dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size):
            t =torch.zeros((1, self.nl_size, self.nl_size))
            t[0, i%self.nl_size, i//self.nl_size] =1
            self.image_shifter.weight.data[i] =t


        # patch summation
        self.summation_patcher =torch.nn.Conv2d(in_channels =self.nl_size*self.nl_size, out_channels =self.nl_size*self.nl_size,
                                              kernel_size =(self.p_size, self.p_size),
                                              stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2),
                                              dilation=1, groups=self.nl_size*self.nl_size, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size):
            # gaussian kernel
            t =torch.zeros((1, self.p_size, self.p_size))
            cx =(self.p_size-1)//2
            cy =(self.p_size-1)//2
            for j in range(self.p_size *self.p_size):
                x=j%self.p_size
                y=j//self.p_size
                d2 =torch.norm( torch.tensor([x-cx, y-cy]).float(), 2)
                t[0, x, y] =math.exp(-d2 / self.sigma2)

            self.summation_patcher.weight.data[i] =t


        # neighbor images
        self.neighbors =torch.nn.Conv2d(in_channels =1, out_channels =self.n_size*self.n_size,
                                        kernel_size =(self.n_size, self.n_size),
                                        stride=1, padding=((self.n_size-1)//2, (self.n_size-1)//2),
                                        dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t =torch.zeros((1, self.n_size, self.n_size))
            t[0, i%self.n_size, i//self.n_size] =1
            self.neighbors.weight.data[i] =t


        # neighbor patcher
        self.neighbor_summation_patcher =torch.nn.Conv2d(in_channels =self.n_size*self.n_size, out_channels =self.n_size*self.n_size,
                                               kernel_size =(self.p_size, self.p_size),
                                               stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2),
                                               dilation=1, groups=self.n_size*self.n_size, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t =torch.ones((1, self.p_size, self.p_size))
            self.neighbor_summation_patcher.weight.data[i] =t



    def forward(self, orig):
        assert(len(orig.shape) ==4)
        assert(orig.shape[1] ==1)
        H, W = orig.shape[2], orig.shape[3]

        # get original image channel stack
        orig_stack =torch.stack([orig.squeeze(dim=1) for i in range(self.nl_size*self.nl_size)], dim=1)

        # get shifted images (even kernel sizes produce H-1 × W-1 → crop back)
        shifted =self.image_shifter(orig)
        shifted =shifted[..., :H, :W]

        # get image diff
        diff_images =shifted -orig_stack

        # diff's L2 norm (even patch_size → crop back)
        Dx_alpha =self.summation_patcher(torch.pow(diff_images, 2.0))
        Dx_alpha =Dx_alpha[..., :H, :W]

        # calc neighbor's variance (even neighbor_size → crop back)
        neighbor_images =self.neighbor_summation_patcher( self.neighbors(orig) )
        neighbor_images =neighbor_images[..., :H, :W]
        Vx =neighbor_images.var(dim =1).unsqueeze(dim =1)

        # output mind
        nume =torch.exp(-Dx_alpha /(Vx +1e-8))
        denomi =nume.sum(dim =1).unsqueeze(dim =1)
        mind =nume /denomi
        return mind


class MINDLoss(torch.nn.Module):

    def __init__(self, non_local_region_size =9, patch_size =7, neighbor_size =3, gaussian_patch_sigma =3.0):
        super(MINDLoss, self).__init__()
        self.nl_size =non_local_region_size
        self.MIND =MIND(non_local_region_size =non_local_region_size,
                        patch_size =patch_size,
                        neighbor_size =neighbor_size,
                        gaussian_patch_sigma =gaussian_patch_sigma)

    def forward(self, input, target):
        in_mind =self.MIND(input)
        tar_mind =self.MIND(target)
        mind_diff =in_mind -tar_mind
        l1 =torch.norm( mind_diff, 1)
        return l1/(input.shape[2] *input.shape[3] *self.nl_size *self.nl_size)


class MIND3D(torch.nn.Module):
    """3D MIND: shifts/patches operate in (H, W) planes, processes all depth slices in parallel.
    Input:  (B, 1, D, H, W)
    Output: (B, nl_size^2, D, H, W)
    """

    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=5, gaussian_patch_sigma=3.0):
        super().__init__()
        nl     = non_local_region_size
        p      = patch_size
        n      = neighbor_size
        sigma2 = gaussian_patch_sigma ** 2

        self.nl_size = nl
        self.p_size  = p
        self.n_size  = n
        self.sigma2  = sigma2

        # shift in (H, W) only — kernel (1, nl, nl)
        self.image_shifter = torch.nn.Conv3d(
            1, nl*nl, kernel_size=(1, nl, nl),
            stride=1, padding=(0, (nl-1)//2, (nl-1)//2),
            bias=False, padding_mode='zeros')
        for i in range(nl*nl):
            t = torch.zeros(1, 1, nl, nl)
            t[0, 0, i % nl, i // nl] = 1
            self.image_shifter.weight.data[i] = t

        # gaussian patch summation in (H, W)
        self.summation_patcher = torch.nn.Conv3d(
            nl*nl, nl*nl, kernel_size=(1, p, p),
            stride=1, padding=(0, (p-1)//2, (p-1)//2),
            groups=nl*nl, bias=False, padding_mode='zeros')
        cx, cy = (p-1)//2, (p-1)//2
        for i in range(nl*nl):
            t = torch.zeros(1, 1, p, p)
            for j in range(p*p):
                x, y = j % p, j // p
                d2 = math.sqrt((x-cx)**2 + (y-cy)**2)
                t[0, 0, x, y] = math.exp(-d2 / sigma2)
            self.summation_patcher.weight.data[i] = t

        # neighbor extraction in (H, W)
        self.neighbors = torch.nn.Conv3d(
            1, n*n, kernel_size=(1, n, n),
            stride=1, padding=(0, (n-1)//2, (n-1)//2),
            bias=False, padding_mode='zeros')
        for i in range(n*n):
            t = torch.zeros(1, 1, n, n)
            t[0, 0, i % n, i // n] = 1
            self.neighbors.weight.data[i] = t

        # neighbor variance patcher in (H, W)
        self.neighbor_summation_patcher = torch.nn.Conv3d(
            n*n, n*n, kernel_size=(1, p, p),
            stride=1, padding=(0, (p-1)//2, (p-1)//2),
            groups=n*n, bias=False, padding_mode='zeros')
        for i in range(n*n):
            t = torch.ones(1, 1, p, p)
            self.neighbor_summation_patcher.weight.data[i] = t

    def forward(self, orig):
        # orig: (B, 1, D, H, W)
        assert orig.ndim == 5 and orig.shape[1] == 1
        B, _, D, H, W = orig.shape

        orig_stack = orig.expand(B, self.nl_size*self.nl_size, D, H, W)

        shifted = self.image_shifter(orig)
        shifted = shifted[..., :H, :W]

        diff_images = shifted - orig_stack
        Dx_alpha = self.summation_patcher(torch.pow(diff_images, 2.0))
        Dx_alpha = Dx_alpha[..., :H, :W]

        neighbor_images = self.neighbor_summation_patcher(self.neighbors(orig))
        neighbor_images = neighbor_images[..., :H, :W]
        Vx = neighbor_images.var(dim=1, keepdim=True)  # (B, 1, D, H, W)

        nume  = torch.exp(-Dx_alpha / (Vx + 1e-8))
        denomi = nume.sum(dim=1, keepdim=True)
        return nume / denomi  # (B, nl^2, D, H, W)


class MINDLoss3D(torch.nn.Module):

    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0):
        super().__init__()
        self.nl_size = non_local_region_size
        self.MIND = MIND3D(
            non_local_region_size=non_local_region_size,
            patch_size=patch_size,
            neighbor_size=neighbor_size,
            gaussian_patch_sigma=gaussian_patch_sigma,
        )

    def forward(self, input, target):
        in_mind  = self.MIND(input)
        tar_mind = self.MIND(target)
        l1 = torch.norm(in_mind - tar_mind, 1)
        D = input.shape[2]
        return l1 / (D * input.shape[3] * input.shape[4] * self.nl_size * self.nl_size)


def scaling_and_squaring(spatial_transform, velocity=None, k=7):
        """
        Diffeomorphic integration of a stationary velocity field.
        velocity: (B, 2, H, W) 형태의 속도 벡터장
        k: Squaring 횟수 (기본값 7이면 2^7 = 128 스텝 적분과 동일한 효과)
        """
        # 1. Scaling: 속도를 아주 작은 단위로 나눔
        phi = velocity / (2 ** k)
        
        # 2. Squaring: 작은 변형장을 반복적으로 자기 합성(Composition)
        for _ in range(k):
            # phi = phi + phi(phi) -> spatial_transform이 변형장 자체를 warp 해줌
            phi = phi + spatial_transform(phi, phi)
            
        return phi