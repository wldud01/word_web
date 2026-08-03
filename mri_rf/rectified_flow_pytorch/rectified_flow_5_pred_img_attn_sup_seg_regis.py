from __future__ import annotations

import math
from copy import deepcopy
from collections import namedtuple
from typing import Literal, Callable
from networkx import sigma
import numpy as np
import random
from torchvision.utils import make_grid
import os
from pathlib import Path

import torch
from torch import Tensor
from torch import nn, pi, cat, stack, from_numpy
from torch.nn import Module, ModuleList
from torch.distributions import Normal
import torch.nn.functional as F

# 미분 관련 모듈
from torchdiffeq import odeint

import torchvision
from torchvision.utils import save_image
from torchvision.models import VGG16_Weights

import einx
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange

from hyper_connections.hyper_connections_channel_first import get_init_and_expand_reduce_stream_functions, Residual

from scipy.optimize import linear_sum_assignment
from rectified_flow_pytorch.utils import smooothing_loss, get_registration_losses, exists, identity, save_sample, divisible_by, default, append_dims, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, cosine_time_warp, cosmap, set_random_seed
from rectified_flow_pytorch.utils import VGGFeatureExtractor, HardNegativePatchNCELoss, scaling_and_squaring, MIND, MINDLoss, MIND3D, MINDLoss3D
from rectified_flow_pytorch.backbone.unet_baseline import Unet
from rectified_flow_pytorch.dataset.dataset import MultiPatientWindowDataset
import lpips


from validation_module import EpochValidatorRG, EarlyStopping
    
set_random_seed(42)
def collate_patient_window(batch_list):
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

        # 🔹 segmentation (optional)
        if "seg_nect_window" in batch_list[0]:
            out["seg_nect_window"] = torch.stack(
                [b["seg_nect_window"] for b in batch_list], dim=0
            )  # (B,W,1,H,W)

        if "seg_cect_window" in batch_list[0]:
            out["seg_cect_window"] = torch.stack(
                [b["seg_cect_window"] for b in batch_list], dim=0
            )

        out["center_index"] = torch.tensor(
            [b["center_index"] for b in batch_list], dtype=torch.long
        )

    # ===============================
    # CASE 2: patient-level
    # ===============================
    elif "nect_volume" in batch_list[0]:
        assert len(batch_list) == 1

        b = batch_list[0]
        out["nect_volume"] = b["nect_volume"].unsqueeze(0)
        out["cect_volume"] = b["cect_volume"].unsqueeze(0)
        out["num_slices"] = b["num_slices"]

        if "seg_nect_volume" in b:
            out["seg_nect_volume"] = b["seg_nect_volume"].unsqueeze(0)
        if "seg_cect_volume" in b:
            out["seg_cect_volume"] = b["seg_cect_volume"].unsqueeze(0)

    else:
        raise KeyError(f"Unknown batch keys: {batch_list[0].keys()}")

    out["patient_id"] = [b["patient_id"] for b in batch_list]
    return out


#############################################################
# losses
class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        
        if isinstance(pred, torch.Tensor) and pred.shape[1] == 2:
            return pred.new_tensor(0.0)

        # deep supervision
        if isinstance(pred, list):
            # Each of decoder layers' output is supervised, with different weights
            weights = [1.0, 0.5, 0.25, 0.125]
            loss = 0.0

            for w, p in zip(weights, pred):
                if p.shape[1] == 2:
                    return p.new_tensor(0.0)

                loss = loss + w * F.mse_loss(p, target)
            return loss

        return F.mse_loss(pred, target)



# =========================================================
# RectifiedFlow (one-direction only)
# =========================================================


######################################################
# main class
class RectifiedFlow(Module):
    def __init__(
        self,
        model: dict | Module,
        mean_variance_net: bool | None = None,
        time_cond_kwarg: str | None = 'times',  # time scheduling 관련 인자
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        predict: Literal['flow', 'noise'] = 'flow',
        loss_fn: Literal[
            'mse',
        ] | Module = 'mse',
        noise_schedule: Literal[
            'cosine_time_warp'
        ] | Callable = identity,
        loss_fn_kwargs: dict = dict(),
        ema_update_after_step: int = 100,
        ema_kwargs: dict = dict(),
        data_shape: tuple[int, ...] | None = None,
        
        use_consistency = False,
        consistency_decay = 0.9999,
        consistency_velocity_match_alpha = 1e-5,
        consistency_delta_time = 1e-3,
        consistency_loss_weight = 1.,
        
        data_normalize_fn = normalize_to_neg_one_to_one,
        data_unnormalize_fn = unnormalize_to_zero_to_one,
        
        clip_during_sampling = False,
        clip_values: tuple[float, float] = (-1., 1.),
        clip_flow_during_sampling = None, # this seems to help a lot when training with predict epsilon, at least for me
        clip_flow_values: tuple[float, float] = (-3., 3)
    ):
        super().__init__()

        if isinstance(model, dict):
            model = Unet(**model)

        self.model = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

        # allow for mean variance output prediction
        if not exists(mean_variance_net):
            mean_variance_net = default(model.mean_variance_net if isinstance(model, Unet) else mean_variance_net, False)

        self.mean_variance_net = mean_variance_net

        # objective - either flow or noise (proposed by Esser / Rombach et al in SD3)
        self.predict = predict

        # automatically default to a working setting for predict epsilon
        clip_flow_during_sampling = default(clip_flow_during_sampling, predict == 'noise')

        # loss fn
        if loss_fn == 'mse':
            loss_fn = MSELoss()

        elif not isinstance(loss_fn, Module):
            raise ValueError(f'unknown loss function {loss_fn}')

        self.loss_fn = loss_fn

        # noise schedules
        if noise_schedule == 'cosmap':
            noise_schedule = cosmap
        elif noise_schedule == 'cosine_time_warp': #NOTE - cosine_time_warp 추가
            noise_schedule = cosine_time_warp
        elif not callable(noise_schedule):
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        self.noise_schedule = noise_schedule

        # sampling
        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape

        # clipping for epsilon prediction
        self.clip_during_sampling = clip_during_sampling
        self.clip_flow_during_sampling = clip_flow_during_sampling

        self.clip_values = clip_values
        self.clip_flow_values = clip_flow_values
        self.noise_mode = 'gaussian'  # default; Trainer.__call__ may override via set_noise_mode()

        # consistency flow matching
        self.use_consistency = use_consistency
        self.consistency_decay = consistency_decay
        self.consistency_velocity_match_alpha = consistency_velocity_match_alpha
        self.consistency_delta_time = consistency_delta_time
        self.consistency_loss_weight = consistency_loss_weight

        if use_consistency:
            self.ema_model = EMA(
                model,
                beta = consistency_decay,
                update_after_step = ema_update_after_step,
                include_online_model = False,
                **ema_kwargs
            )

        # normalizing fn
        self.data_normalize_fn = default(data_normalize_fn, identity)
        self.data_unnormalize_fn = default(data_unnormalize_fn, identity)

    @property
    def device(self):
        return next(self.model.parameters()).device

    # flow를 예측하는 함수
    def predict_flow(self, model: Module, noised, *, times, adj = None, seg=None, regis=False, eps = 1e-10, **model_kwargs):
        """
        returns the model output as well as the derived flow, depending on the `predict` objective
        """
        if noised.ndim == 5:
            noised = noised.squeeze(2)
        batch = noised.shape[0]
        #print(noised.shape)     # channel 5
        
        # prepare maybe time conditioning for model
        time_kwarg = self.time_cond_kwarg   # 기본: 'times'

        if exists(time_kwarg):
            times = rearrange(times, '... -> (...)')    # 1D 벡터로 평탄화
            if times.numel() == 1:
                times = repeat(times, '1 -> b', b = batch)  # batch 크기로 times 복제
            
            #print(' \npredict_flow times',times)
            model_kwargs.update(**{time_kwarg: times})  # time_kwarg = 'times' → model_kwargs['times'] = times
            
        output = self.model(noised, cond = adj, seg=seg, **model_kwargs)   # Unet, output은 중심 슬라이스에 대한 velocity가 출력
        #print(output.max(), output.min())

        # depending on objective, derive flow

        if self.predict == 'flow':
            flow = output

        elif self.predict == 'noise':
            noise = output
            padded_times = append_dims(times, noised.ndim - 1)

            flow = (noised - noise) / padded_times.clamp(min = eps)
        else:
            raise ValueError(f'unknown objective {self.predict}')

        return output, flow

    # RectifiedFlow 클래스 안 - sampling 함수
    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        steps = 16,
        noise = None,
        adj=None,
        seg=None,
        cond=None,           # MRI 등 neighbor/condition (adj 별칭)
        regis=False,
        data_shape: tuple[int, ...] | None = None,
        temperature: float = 1.,
        use_ema: bool = False,
        **model_kwargs
    ):
        use_ema = default(use_ema, self.use_consistency)
        assert not (use_ema and not self.use_consistency), 'in order to sample from an ema model, you must have `use_consistency` turned on'
        model = self.ema_model if use_ema else self.model

        was_training = self.training
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'


        maybe_clip      = (lambda t: t.clamp_(*self.clip_values))      if self.clip_during_sampling      else identity
        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity

        # seg = NECT (cross-attention condition); start ODE from Gaussian noise
        if seg is not None and seg.ndim == 5:
            seg = seg.squeeze(2)

        noise = default(noise, torch.randn((batch_size, *data_shape), device=self.device) * temperature)

        _adj = cond if cond is not None else adj
        def ode_fn(t, x):
            x = maybe_clip(x)
            _, output = self.predict_flow(model, x, adj=_adj, seg=seg, times=t, **model_kwargs)
            return maybe_clip_flow(output)

        times = torch.linspace(0., 1., steps, device=self.device)
        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)  # (T, B, C, H, W)

        # 마지막으로 나온 결과만 반환
        sampled_data = trajectory[-1]
        #print('sampled_data',trajectory.shape )

        self.train(was_training)

        return self.data_unnormalize_fn(sampled_data)



    def forward(
        self,
        data,                           # CECT
        noise: Tensor | None = None,    # Gaussian noise (optional override)
        times=None,
        seg=None,
        cond=None,                      # adjacent / neighbor condition (MRI 등)
        use_neighbor=False,
        return_loss_breakdown=False,
        **model_kwargs,
    ):
        batch, *data_shape = data.shape

        data = self.data_normalize_fn(data)
        data = data.to(dtype=next(self.model.parameters()).dtype)

        self.data_shape = default(self.data_shape, data_shape)

        # seg = NECT (or warped NECT) as cross-attention condition
        if seg is not None and seg.ndim == 5:
            seg = seg.squeeze(2)
        if data.ndim == 5:
            data = data.squeeze(2)

        # noise_mode 결정: 명시적으로 noise가 넘어오면 그것 사용
        # 아니면 self.noise_mode에 따라 결정
        # 'gaussian': x0 = Gaussian noise  (새 모델, LDM)
        # 'nect'    : x0 = seg (NECT)       (구 모델, window=5 3D conv)
        if noise is None:
            if getattr(self, 'noise_mode', 'gaussian') == 'nect' and seg is not None:
                noise = seg.clone()
            else:
                noise = torch.randn_like(data)

        center_idx = data.shape[1] // 2 if data.shape[1] > 1 else 0
        noise_center = noise[:, center_idx:center_idx+1, ...]
        data_center  = data[:, center_idx:center_idx+1, ...]

        if times is not None:
            times = times
        else:
            times = torch.rand(batch, device=self.device)
        padded_times = append_dims(times, data.ndim - 1)

        def get_noised_and_flows(model, t):
            t = self.noise_schedule(t)

            # (1-t)*Gaussian + t*CECT
            noised = noise.lerp(data, t)

            if use_neighbor:
                noised_center   = noised[:, center_idx:center_idx+1, ...]
                noised_neighbor = torch.cat(
                    [noised[:, :center_idx, ...], noised[:, center_idx+1:, ...]], dim=1)
                flow = data_center - noise_center   # CECT_c - Gaussian_c
                model_output, _ = self.predict_flow(
                    model, noised_center, adj=noised_neighbor, seg=seg, times=t, **model_kwargs)
                pred_flow = model_output
                main_out  = pred_flow[0] if isinstance(pred_flow, list) else pred_flow
                pred_data = noised_center + main_out * (1. - t)
            else:
                flow = data - noise                 # CECT - Gaussian
                model_output, _ = self.predict_flow(
                    model, noised, adj=cond, seg=seg, times=t, **model_kwargs)
                pred_flow = model_output
                main_out  = pred_flow[0] if isinstance(pred_flow, list) else pred_flow
                pred_data = noised + main_out * (1. - t)

            return model_output, flow, pred_flow, pred_data

        output, flow, pred_flow, pred_data = get_noised_and_flows(self.model, padded_times)

        if self.predict == 'flow':
            target = flow
        elif self.predict == 'noise':
            target = noise
        else:
            raise ValueError(f'unknown objective {self.predict}')

        main_loss  = self.loss_fn(output, target, pred_data=pred_data, times=times, data=data)
        total_loss = main_loss

        return total_loss, pred_data, pred_flow


# =========================================================
# Backward Model = same RF class (no change needed)
# =========================================================

class BackwardRectifiedFlow(RectifiedFlow):
    pass



##########################################################
# trainer
from torch.optim import Adam, AdamW
from accelerate import Accelerator
from torch.utils.data import DataLoader
from ema_pytorch import EMA
import wandb

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError
import torch
import math
from torchvision.utils import save_image
from einops import rearrange
from tqdm import tqdm
from rectified_flow_pytorch.transformer  import Transformer_2D, Transformer_3D, Transformer_2D_win
from rectified_flow_pytorch.reg import Reg
import torchvision.transforms.functional as TF
from scipy.ndimage import distance_transform_edt


def cycle(dl):
    while True:
        for batch in dl:
            yield batch
            
            
class Trainer(Module):
    def __init__(
        self,
        forward_model: RectifiedFlow,     # F_xy
        backward_model: RectifiedFlow,    # F_yx
        *,
        dataset: dict | MultiPatientWindowDataset,
        num_train_steps = 100_000_000,
        learning_rate = 1e-4,
        batch_size = 4,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './results',
        save_results_every: int = 1000,
        checkpoint_every: int = 1,
        sample_temperature: float = 1.,
        num_samples: int = 16,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        use_ema = True,
        max_grad_norm = 0.5,

        warmup_epochs = 30,
        seg_ = False,
        training_mode = -1,       # -1: baseline, 0: G→R, 1: R→G
        registration_mode = 0,    # mode 1 only: 0=raw(window), 1=MIND, 2=cycle
        window_size = 1,
        save_root = "checkpoints",
        pre_train_ckpt = None,
        mind_params: dict | None = None,
        loss_weights: dict | None = None,  # loss weight 튜닝용
        conv_mode: str = '2d',             # '2d' or '3d'
        backbone_mode: str = 'mk_unet',    # 'base' | 'mk_unet' | 'mk_unet_ldm'
        noise_mode: str = 'gaussian',      # 'gaussian': x0=noise / 'nect': x0=nect (구 방식)
        model_type: int | None = None,     # None: 기본 / 3: MIND feature condition
    ):
        super().__init__()
        assert training_mode in (-1, 0, 1, 3), f"training_mode must be -1, 0, 1, or 3. Got {training_mode}"
        if training_mode == 1:
            assert registration_mode in (0, 1, 2), \
                f"registration_mode for mode 1 must be 0 (raw), 1 (MIND), or 2 (cycle). Got {registration_mode}"
        assert conv_mode in ('2d', '3d'), f"conv_mode must be '2d' or '3d'. Got {conv_mode}"

        self.accelerator = Accelerator(**accelerate_kwargs)

        if isinstance(dataset, dict):
            dataset = MultiPatientWindowDataset(**dataset)

        self.dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_patient_window)
        self.window_size = window_size
        self.training_mode = training_mode
        self.registration_mode = registration_mode
        self.conv_mode = conv_mode
        self.backbone_mode = backbone_mode
        self.noise_mode = noise_mode
        self.model_type = model_type
        # forward/sample에서 사용하도록 모델에 noise_mode 주입
        self.F_xy.noise_mode = noise_mode
        self.F_yx.noise_mode = noise_mode

        # Models
        self.F_xy = forward_model   # NECT → CECT
        self.F_yx = backward_model  # CECT → NECT (pretrain / warmup)

        # Registration — only modes 0 and 1 use R_A
        # conv_mode '3d': R_A uses 3D convs + Transformer_3D (except MIND reg which is always 2D center-slice)
        _mp = mind_params or {}
        _dim_reg = 3 if conv_mode == '3d' else 2
        self.R_A = None
        self.spatial_transform = Transformer_3D().cuda() if conv_mode == '3d' else Transformer_2D().cuda()
        self.spatial_transform_2d = Transformer_2D().cuda()  # MIND reg is always 2D

        if training_mode == 0:
            self.R_A = Reg(256, 256, 1, 1, dim=_dim_reg).cuda()

        elif training_mode == 1:
            if registration_mode == 0:       # raw image
                self.R_A = Reg(256, 256, 1, 1, dim=_dim_reg).cuda()
            elif registration_mode == 1:     # MIND — 2D center-slice or 3D full-volume
                _nlr = _mp.get("non_local_region_size", 9)
                in_ch = _nlr * _nlr
                _dim_mind = 3 if conv_mode == '3d' else 2
                self.R_A = Reg(256, 256, in_ch, in_ch, dim=_dim_mind).cuda()
            elif registration_mode == 2:     # cycle
                self.R_A = Reg(256, 256, 1, 1, dim=_dim_reg).cuda()

        # MIND — mode 1 reg 1 only
        if training_mode == 1 and registration_mode == 1:
            _MINDCls     = MIND3D     if conv_mode == '3d' else MIND
            _MINDLossCls = MINDLoss3D if conv_mode == '3d' else MINDLoss
            self.mind_extractor = _MINDCls(
                non_local_region_size=_mp.get("non_local_region_size", 9),
                patch_size=_mp.get("patch_size", 7),
                neighbor_size=_mp.get("neighbor_size", 3),
                gaussian_patch_sigma=_mp.get("gaussian_patch_sigma", 3.0),
            ).cuda()
            self.mind_loss_fn = _MINDLossCls(
                non_local_region_size=_mp.get("non_local_region_size", 9),
                patch_size=_mp.get("patch_size", 7),
                neighbor_size=_mp.get("neighbor_size", 3),
                gaussian_patch_sigma=_mp.get("gaussian_patch_sigma", 3.0),
            ).cuda()
        else:
            self.mind_extractor = None
            self.mind_loss_fn = None

        # LPIPS — mode 1 reg 2 (cycle) only
        if training_mode == 1 and registration_mode == 2:
            self.lpips_fn = lpips.LPIPS(net='vgg').to(self.accelerator.device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None

        # EMA
        use_ema &= not getattr(self.F_xy, 'use_consistency', False)
        self.use_ema = use_ema
        self.ema_model = None
        if self.is_main and use_ema:
            self.ema_model = EMA(self.F_xy, forward_method_names=('sample',), **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

        # Optimizer — only include models actually in use
        params = [
            {"params": self.F_xy.parameters(), "lr": learning_rate},
            {"params": self.F_yx.parameters(), "lr": learning_rate},
        ]
        if self.R_A is not None:
            params.append({"params": self.R_A.parameters(), "lr": 1e-4})
        self.optimizer = Adam(params)

        # Accelerate prepare
        to_prepare = [self.F_xy, self.F_yx]
        if self.R_A is not None:
            to_prepare.append(self.R_A)
        to_prepare.extend([self.optimizer, self.dl])
        prepared = self.accelerator.prepare(*to_prepare)
        _i = 0
        self.F_xy = prepared[_i]; _i += 1
        self.F_yx = prepared[_i]; _i += 1
        if self.R_A is not None:
            self.R_A = prepared[_i]; _i += 1
        self.optimizer = prepared[_i]; _i += 1
        self.dl = prepared[_i]; _i += 1

        # Training setup
        self.num_train_steps = num_train_steps
        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every
        self.sample_temperature = sample_temperature
        self.num_samples = num_samples
        self.num_sample_rows = int(math.sqrt(num_samples))

        self.max_grad_norm = float(max_grad_norm)

        self.get_registration_losses = get_registration_losses

        # Loss weights (configurable — used for WandB sweep)
        lw = loss_weights or {}
        self.w_align      = lw.get("w_align",      1.0)
        self.w_rf         = lw.get("w_rf",          1.0)
        self.w_cect_rec   = lw.get("w_cect_rec",    0.8)
        self.w_ssim       = lw.get("w_ssim",        0.1)
        self.w_sm         = lw.get("w_sm",          1e-2)
        self.w_adj_sm     = lw.get("w_adj_sm",      1e-5)
        self.w_regularize = lw.get("w_regularize",  0.2)

        self.epoch_start = 1
        self.warmup_epochs = warmup_epochs
        self.ssim_loss_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.accelerator.device)

        self.seg_ = seg_
        self.save_root = save_root
        self.pre_train_ckpt = pre_train_ckpt

        self.validator = EpochValidatorRG(
            val_root="/home/user/yeong/PMC/input_data_new/final_data/val",
            seg_root="/home/user/yeong/PMC/input_data_new/segmentation/model/val",
            save_root=self.save_root,
            pre_ckpt_path=self.pre_train_ckpt,
            window_size=window_size,
            steps=4 + 1,
            training_mode=self.training_mode,
            registration_mode=self.registration_mode,
            device="cuda",
        )

        self.stop_training = False
        self.early_stopper = EarlyStopping(patience=30, mode="max")
        
    # --------------------------------------------------------
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # --------------------------------------------------------
    # 3D registration helpers
    # --------------------------------------------------------
    def _reg_wrap(self, x):
        """(B, W, H, Hs) → (B, 1, W, H, Hs) for 3D R_A; no-op in 2D mode."""
        return x.unsqueeze(1) if self.conv_mode == '3d' else x

    def _reg_unwrap(self, x):
        """(B, 1, W, H, Hs) → (B, W, H, Hs) after spatial_transform; no-op in 2D mode."""
        return x.squeeze(1) if self.conv_mode == '3d' else x

    # --------------------------------------------------------
    def save(self, path):
        if not self.is_main:
            return
        unwrap = self.accelerator.unwrap_model
        package = {
            "F_xy":      unwrap(self.F_xy).state_dict(),
            "F_yx":      unwrap(self.F_yx).state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.R_A is not None:
            package["RA"] = unwrap(self.R_A).state_dict()
        if self.ema_model is not None:
            package["ema_model"] = self.ema_model.state_dict()
        torch.save(package, str(self.checkpoints_folder / path))

    def load(self, path):
        self.epoch_start = int(path.split('.')[-2]) + 1
        print(self.epoch_start)
        pkg = torch.load(path, map_location=self.accelerator.device)
        unwrap = self.accelerator.unwrap_model
        if "F_xy" in pkg:  unwrap(self.F_xy).load_state_dict(pkg["F_xy"])
        if "F_yx" in pkg:  unwrap(self.F_yx).load_state_dict(pkg["F_yx"])
        if "RA" in pkg and self.R_A is not None:
            unwrap(self.R_A).load_state_dict(pkg["RA"])
        if "optimizer" in pkg: self.optimizer.load_state_dict(pkg["optimizer"])
        if "ema_model" in pkg and self.ema_model is not None:
            self.ema_model.load_state_dict(pkg["ema_model"])
        print(f"[✓] Checkpoint loaded from {path}")

    # --------------------------------------------------------
    def sample(self, fname: str):

        os.makedirs(os.path.dirname(fname), exist_ok=True)
        eval_model = default(self.ema_model, self.F_xy)
        dl = cycle(self.dl)
        nect, cect = next(dl)
        
        data_shape = cect.shape[1:]

        with torch.no_grad():
            sampled = eval_model.sample(
                batch_size=self.num_samples,
                data_shape=data_shape,
                seg=nect.to(self.accelerator.device, dtype=torch.float32),
                temperature=self.sample_temperature
            )

        B, C, H, W = sampled.shape
        sampled_reshaped = sampled.reshape(B * C, 1, H, W)

        grid = make_grid(sampled_reshaped, nrow=5)
        save_image(grid, fname)
        self.accelerator.print(f"[Saved] {fname}")
        return fname



    def get_edge(self, x, low=100, high=200):
        import cv2

        # x: (B,1,H,W), range [-1,1] 또는 [0,1] 둘 다 대응
        x_np = x.detach().cpu().numpy()

        # [-1,1] → [0,255]
        if x_np.min() < 0:
            x_np = ((x_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
        else:
            x_np = (x_np * 255.0).astype(np.uint8)

        edges = []
        for i in range(x_np.shape[0]):
            e = cv2.Canny(x_np[i, 0], low, high)

            # 🔹 background 완전 제거 (edge만 1, 나머지 0)
            e = (e > 0).astype(np.float32)

            edges.append(e)

        edges = torch.from_numpy(np.stack(edges)).unsqueeze(1).to(x.device)  # (B,1,H,W), {0,1}
        return edges
    

    def get_edge_and_seg(self, x, seg, use_image=False, mode=None, use_dt=True):
        import cv2
        if seg is not None and seg.ndim == 5:
            seg = seg.squeeze(2)
        
        # 1. 기본 에지 추출
        edge = self.get_edge(x) 
        
        # 2. Distance Transform 적용 (추가된 부분)
        if use_dt:
            feat = self.get_distance_transform(edge)
        else:
            feat = seg
            
        # 3. 추가 채널 결합 (기존 로직)
        if use_image:
            feat = torch.cat([x, feat], dim=1)
            
        if mode == 4 and seg is not None:
            feat = feat + seg # Edge(DT)와 SegMap 결합
            
        return feat

    # --------------------------------------------------------
    def forward(self):

        num_epochs = math.ceil(self.num_train_steps / len(self.dl))
        print(f"[INFO] Training for {num_epochs} epochs ({self.num_train_steps} steps)")
        print(f"training mode: {self.training_mode}, registration mode: {self.registration_mode}")

        def set_requires_grad(modules, flag: bool):
            if not isinstance(modules, (list, tuple)):
                modules = [modules]
            for m in modules:
                for p in m.parameters():
                    p.requires_grad = flag

        # Load frozen pretrain model (F_yx_) — used in main phase for modes 0, 1
        # reg_mode==1 (MIND)은 real nect/cect를 직접 사용하므로 F_yx_ 불필요
        _need_pretrain = not (self.training_mode == 1 and self.registration_mode == 1)
        if self.pre_train_ckpt is not None and _need_pretrain:
            if self.backbone_mode == 'mk_unet_ldm':
                from rectified_flow_pytorch.backbone.mk_unet_ldm_attn import MKRF_UNet_LDM as _MK
            elif self.conv_mode == '3d':
                from rectified_flow_pytorch.backbone.mk_unet_3d import MKRF_UNet_3D as _MK
            else:
                from rectified_flow_pytorch.backbone.mk_unet_img_attn_sup import MKRF_UNet as _MK
            _pre_backbone = _MK(in_channels=self.window_size, out_channels=self.window_size)
            self.F_yx_ = RectifiedFlow(_pre_backbone, data_normalize_fn=None, data_unnormalize_fn=None)
            self.F_yx_ = self.F_yx_.to(self.accelerator.device)
            _state = torch.load(self.pre_train_ckpt, map_location=self.accelerator.device)
            self.F_yx_.load_state_dict(_state["F_yx"])
            self.F_yx_.eval()
            for p in self.F_yx_.parameters():
                p.requires_grad = False

        # WandB init
        self.accelerator.wait_for_everyone()
        if self.is_main and wandb.run is None:
            wandb.init(
                project="NECT_to_CECT_RectifiedFlow",
                name=f"run_{np.random.randint(10000)}",
                config={"steps": self.num_train_steps},
                reinit=True,
            )
        self.accelerator.wait_for_everyone()

        global_step = getattr(self, "global_step", 0)

        for epoch in range(self.epoch_start, num_epochs + 1):

            if self.stop_training:
                print(f"[EarlyStop] triggered at epoch {epoch}")
                break

            in_warmup = (epoch <= self.warmup_epochs)

            self.F_xy.train()
            self.F_yx.train()
            if self.R_A is not None:
                self.R_A.train()

            if in_warmup:
                set_requires_grad(self.F_yx, True)
                set_requires_grad(self.F_xy, True)
            else:
                set_requires_grad(self.F_xy, True)
                set_requires_grad(self.F_yx, True)
                if self.R_A is not None:
                    set_requires_grad(self.R_A, True)

            epoch_loss = 0.0
            pbar = tqdm(self.dl, desc=f"[Epoch {epoch}] Training", ncols=120)

            for batch in pbar:
                global_step += 1

                nect = batch["nect_window"].to(self.accelerator.device, dtype=torch.float32)
                cect = batch["cect_window"].to(self.accelerator.device, dtype=torch.float32)
                if nect.ndim == 5: nect = nect.squeeze(2)
                if cect.ndim == 5: cect = cect.squeeze(2)

                center_idx = nect.shape[1] // 2
                nect_center = nect[:, center_idx:center_idx+1]
                cect_center = cect[:, center_idx:center_idx+1]
                
                
                
                # ── WARMUP PHASE ────────────────────────────���───────
                if in_warmup:
                    L_rf_yx, pred_nect, _ = self.F_yx(nect, seg=cect)
                    L_nect = F.l1_loss(pred_nect, nect)
                    L_rf_xy, pred_cect, _ = self.F_xy(cect, seg=pred_nect)
                    L_cycle_cect = F.l1_loss(pred_cect, cect)
                    loss = L_rf_yx + L_rf_xy + (L_nect + L_cycle_cect) * self.w_cect_rec

                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(
                        list(self.F_yx.parameters()) + list(self.F_xy.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_loss += loss.item()
                    pbar.set_postfix({"phase": "warmup", "loss": f"{loss.item():.4f}"})
                    wandb.log({'train/warmup/L_rf_yx': L_rf_yx.item(), 'train/warmup/L_nect': L_nect.item(),
                               'epoch': epoch}, step=global_step)
                    if self.is_main and global_step % 1000 == 0:
                        with torch.no_grad():
                            save_dir = f'/home/user/yeong/rectified-flow-pytorch/results_window/epoch_{epoch}/'
                            save_sample(save_dir, nect, global_step, 'nect')
                            save_sample(save_dir, cect, global_step, 'cect')
                            save_sample(save_dir, pred_nect, global_step, 'nect_hat')
                    continue
                
                
                # ── FULL TRAINING PHASE ──────────────────────────────
                # Seg masks
                if self.seg_:
                    seg_cect = batch["seg_cect_window"].to(self.accelerator.device, dtype=torch.float32)
                    seg_nect = batch["seg_nect_window"].to(self.accelerator.device, dtype=torch.float32)
                    if seg_cect.ndim == 5: seg_cect = seg_cect.squeeze(2)
                    if seg_nect.ndim == 5: seg_nect = seg_nect.squeeze(2)
                    seg_cect_center = seg_cect[:, center_idx:center_idx+1]
                    seg_nect_center = seg_nect[:, center_idx:center_idx+1]
                        


                # Pretrain model: CECT → pseudo-NECT (frozen, used by reg 0 and 2)
                # reg 1 (MIND) uses real nect/cect directly — skip inference to save compute
                if self.training_mode in (0,) or (self.training_mode == 1 and self.registration_mode != 1):
                    with torch.no_grad():
                        _, pred_nect, _ = self.F_yx_(nect, seg=cect)
                    pred_nect_center = pred_nect[:, center_idx:center_idx+1]
                else:
                    pred_nect = None
                    pred_nect_center = None

                # training_mode -1: Baseline
                # training_mode  0: G→R (generate then register)
                # training_mode  1: R→G (register then generate)
                #   reg 0: raw image (window-aware)
                #   reg 1: MIND-based (nect vs cect directly)
                #   reg 2: cycle

                # ── training_mode == -1: Baseline ────────────────────
                if self.training_mode == -1:
                    L_contrast, pred_cect, _ = self.F_xy(cect, seg=nect)
                        
                        # Prediction loss
                    L_cect_rec = F.l1_loss(pred_cect, cect)

                        # SSIM loss - registration
                    ssim_val = self.ssim_loss_fn(pred_cect, cect)
                    L_ssim = 1.0 - ssim_val
                    
                    # total_loss
                    if any(p.requires_grad for p in self.F_yx.parameters()):
                        total_loss = (
                            self.w_rf * (L_contrast)
                            + self.w_cect_rec * L_cect_rec
                            + self.w_ssim * (L_ssim )
                        )
                        wandb.log({'train/loss': total_loss.item(), 'train/L_contrast': L_contrast.item(),
                                   'train/L_cect_rec': L_cect_rec.item(), 'train/L_ssim': L_ssim.item(), 'epoch': epoch}, step=global_step)

                # ── training_mode == 0: G→R ──────────────────────────
                elif self.training_mode == 0:
                    L_contrast, pred_cect, _ = self.F_xy(cect, seg=nect)
                    pred_r = self._reg_wrap(pred_cect)
                    Trans = self.R_A(pred_r, self._reg_wrap(cect))
                    cect_warped = self._reg_unwrap(self.spatial_transform(pred_r, Trans))
                    L_cect_rec = F.l1_loss(cect_warped, cect)
                    L_ssim = 1.0 - self.ssim_loss_fn(cect_warped, cect)
                    sm_loss = smooothing_loss(Trans)
                    total_loss = (
                        self.w_rf * L_contrast
                        + self.w_cect_rec * L_cect_rec
                        + self.w_ssim * L_ssim
                        + self.w_sm * sm_loss
                    )
                    wandb.log({'train/loss': total_loss.item(), 'train/L_contrast': L_contrast.item(),
                               'train/L_cect_rec': L_cect_rec.item(), 'train/L_ssim': L_ssim.item(),
                               'train/sm_loss': sm_loss.item(), 'epoch': epoch}, step=global_step)
                    
                # ── training_mode == 1: R→G ──────────────────────────
                elif self.training_mode == 1:

                    if self.registration_mode == 0:       # reg 0: raw
                        nect_r = self._reg_wrap(nect)
                        pred_nect_r = self._reg_wrap(pred_nect)
                        Trans = self.R_A(nect_r, pred_nect_r)
                        nect_warped = self._reg_unwrap(self.spatial_transform(nect_r, Trans))
                        pred_nect_u = self._reg_unwrap(pred_nect_r)

                        L_align = F.l1_loss(nect_warped, pred_nect_u)
                        sm_loss = smooothing_loss(Trans)

                        # Generation
                        L_contrast, pred_cect, _ = self.F_xy(cect, seg=nect_warped)
                        L_cect_rec = F.l1_loss(pred_cect, cect)
                        L_ssim = 1.0 - self.ssim_loss_fn(pred_cect, cect)

                        if any(p.requires_grad for p in self.F_yx.parameters()):
                            total_loss = (
                                self.w_rf * (L_contrast)
                                + self.w_cect_rec * L_cect_rec
                                + self.w_ssim * (L_ssim)
                                + self.w_sm * (sm_loss)
                                + self.w_rf * (L_align)
                            )
                            wandb.log({'train/loss': total_loss.item(), 'train/L_contrast': L_contrast.item(),
                                        'train/L_cect_rec': L_cect_rec.item(), 'train/L_ssim': L_ssim.item(),
                                        'train/L_align': L_align.item(), 'train/l_smooth': sm_loss.item(), 'epoch': epoch,}, step=global_step)

                    elif self.registration_mode == 1:     # reg 1: MIND
                        if self.conv_mode == '3d':
                            # 3D MIND: full-volume warp
                            nect_r = self._reg_wrap(nect)   # (B, 1, D, H, W)
                            cect_r = self._reg_wrap(cect)
                            reg_src = self.mind_extractor(nect_r)
                            reg_tgt = self.mind_extractor(cect_r)
                            Trans = self.R_A(reg_src, reg_tgt)
                            nect_r_warped = self.spatial_transform(nect_r, Trans)
                            nect_warped = self._reg_unwrap(nect_r_warped)
                            L_align = self.mind_loss_fn(nect_r_warped, cect_r)
                        else:
                            # 2D MIND: center slice only, warp applied to all slices
                            reg_src = self.mind_extractor(nect_center)
                            reg_tgt = self.mind_extractor(cect_center)
                            Trans = self.R_A(reg_src, reg_tgt)
                            nect_warped = self.spatial_transform_2d(nect, Trans)
                            L_align = self.mind_loss_fn(nect_warped[:, center_idx:center_idx+1], cect_center)
                        sm_loss = smooothing_loss(Trans)

                        # Generation
                        L_contrast, pred_cect, _ = self.F_xy(cect, seg=nect_warped)
                        L_cect_rec = F.l1_loss(pred_cect, cect)
                        L_ssim = 1.0 - self.ssim_loss_fn(pred_cect, cect)

                        if any(p.requires_grad for p in self.F_yx.parameters()):
                            total_loss = (
                                self.w_rf * (L_contrast)
                                + self.w_cect_rec * L_cect_rec
                                + self.w_ssim * (L_ssim)
                                + self.w_rf * (L_align)
                                + self.w_sm * (sm_loss)
                            )
                            wandb.log({'train/loss': total_loss.item(), 'train/L_contrast': L_contrast.item(),
                                        'train/L_cect_rec': L_cect_rec.item(), 'train/L_ssim': L_ssim.item(),
                                        'train/sm_loss': sm_loss.item(), 'epoch': epoch,}, step=global_step)

                    elif self.registration_mode == 2:     # reg 2: cycle
                        nect_r = self._reg_wrap(nect)
                        pred_nect_r = self._reg_wrap(pred_nect)
                        Trans = self.R_A(nect_r, pred_nect_r)
                        nect_warped = self._reg_unwrap(self.spatial_transform(nect_r, Trans))
                        L_align = F.l1_loss(nect_warped, self._reg_unwrap(pred_nect_r))
                        sm_loss = smooothing_loss(Trans)
                        L_tex_shape = self.lpips_fn(nect_warped[:, center_idx:center_idx+1],
                                                    nect[:, center_idx:center_idx+1]).mean()

                        # Generation (forward)
                        L_contrast, pred_cect, _ = self.F_xy(cect, seg=nect_warped)
                        L_cect_rec = F.l1_loss(pred_cect, cect)
                        L_ssim = 1.0 - self.ssim_loss_fn(pred_cect, cect)

                        # Backward cycle
                        L_contrast_back, pred_nect_hat, _ = self.F_yx(nect, seg=pred_cect)
                        pnh_r = self._reg_wrap(pred_nect_hat)
                        Trans_back = self.R_A(pnh_r, self._reg_wrap(nect))
                        nect_hat_warped = self._reg_unwrap(self.spatial_transform(pnh_r, Trans_back))
                        L_align_hat = F.l1_loss(nect_hat_warped, nect)
                        sm_loss_back = smooothing_loss(Trans_back)

                        if any(p.requires_grad for p in self.F_yx.parameters()):
                            total_loss = (
                                self.w_rf * (L_contrast + L_contrast_back)
                                + self.w_cect_rec * L_cect_rec
                                + self.w_ssim * (L_ssim)
                                + self.w_align * (L_align + L_align_hat)
                                + self.w_sm * (sm_loss + sm_loss_back)
                                + self.w_regularize * (L_tex_shape)
                            )
                            wandb.log({'train/loss': total_loss.item(), 'train/L_contrast': L_contrast.item(),
                                        'train/L_cect_rec': L_cect_rec.item(), 'train/L_ssim': L_ssim.item(),
                                        'train/sm_loss': sm_loss.item(),
                                        'train/L_tex_shape': L_tex_shape.item(), 'epoch': epoch,}, step=global_step)

                    else:
                        print("Invalid registration mode. Please choose from 0, 1, 2.")

                # ── training_mode == 3: source as x0 ─────────────────
                elif self.training_mode == 3:
                    # x0=NECT (source), no seg, no cond — simple NECT→CECT rectified flow
                    L_contrast, pred_cect, _ = self.F_xy(cect, noise=nect)
                    L_cect_rec = F.l1_loss(pred_cect, cect)
                    L_ssim = 1.0 - self.ssim_loss_fn(pred_cect, cect)
                    total_loss = (
                        self.w_rf * L_contrast
                        + self.w_cect_rec * L_cect_rec
                        + self.w_ssim * L_ssim
                    )
                    wandb.log({'train/loss': total_loss.item(), 'train/L_contrast': L_contrast.item(),
                               'train/L_cect_rec': L_cect_rec.item(), 'train/L_ssim': L_ssim.item(),
                               'epoch': epoch}, step=global_step)

                else:
                    print("Invalid training mode. Please choose from -1, 0, 1, 3.")
                
                

                # ============================================================
                #                      TOTAL LOSS
                # ============================================================
                loss = total_loss
                self.accelerator.backward(loss)

                # Gradient clipping
                params_to_clip = list(self.F_xy.parameters()) + list(self.F_yx.parameters())
                if self.R_A is not None:
                    params_to_clip += list(self.R_A.parameters())
                self.accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                pbar.set_postfix({"phase": "full", "loss": f"{loss.item():.4f}"})
                
                # save image
                if self.is_main and global_step % 100 == 0:
                    with torch.no_grad():
                        save_dir = '/home/user/yeong/rectified-flow-pytorch/results_window/' + f"{self.training_mode}/" + f"epoch_{epoch}/"
                        save_sample(save_dir, nect, global_step, 'nect')
                        save_sample(save_dir, cect, global_step, 'cect')
                        #save_sample(save_dir, warped_edge, global_step, 'nect_edge_warp')
                        if self.training_mode in [1]:
                            save_sample(save_dir, nect_warped, global_step, 'nect_hat') 
                        save_sample(save_dir, pred_cect, global_step, 'cect_hat')
    

            avg_loss = epoch_loss / len(self.dl)

            if self.is_main:
                wandb.log({"train/loss_avg": avg_loss, "epoch": epoch})
            self.accelerator.print(f"[Epoch {epoch}] Avg Loss {avg_loss:.4f}")

            if self.is_main and divisible_by(epoch, self.checkpoint_every):
                self.save(f'checkpoint.{epoch}.pt')
                
                # main train부터 validation 및 early stopping 적용
                if self.warmup_epochs < epoch and epoch >= 300:
                    # (1) validate
                    # 수정: 저장된 체크포인트의 정확한 경로를 지정 (슬래시 추가 혹은 체크포인트 폴더 참조)
                    ckpt_path = str(self.checkpoints_folder / f'checkpoint.{epoch}.pt')
                    metrics = self.validator.evaluate(epoch=epoch, ckpt_path=ckpt_path)
                    score = metrics["primary"]  # warp PSNR

                    # (2) early stopping update
                    should_stop = self.early_stopper.step(score)

                    print(f"[VAL] epoch={epoch} warpPSNR={score:.4f} rawPSNR={metrics['raw']['PSNR']:.4f}")

                    if should_stop:
                        # 수정: 하드코딩된 patience 숫자 대신 설정된 변수값 출력
                        print(f"[EarlyStop] patience={self.early_stopper.patience} triggered at epoch {epoch}")
                        # 여기서 break / stop flag 세팅
                        self.stop_training = True


        print("Training finished.")


# ============================================================
# RegistrationPretrainer  (training_mode = -2)
# MIND 기반 registration 전용 사전 학습 — R_A 만 저장
# ============================================================

class RegistrationPretrainer:
    """
    Trains only R_A using MIND loss between real NECT and CECT.
    No generation model involved. Saves only R_A in the checkpoint.
    """

    def __init__(
        self,
        *,
        dataset,
        num_train_steps: int,
        batch_size:      int  = 4,
        lr:              float = 1e-4,
        max_grad_norm:   float = 1.0,
        checkpoints_folder: str,
        checkpoint_every:   int = 10,
        window_size:     int = 1,
        mind_params:     dict | None = None,
    ):
        from accelerate import Accelerator
        from torch.utils.data import DataLoader

        self.checkpoints_folder = Path(checkpoints_folder)
        self.checkpoints_folder.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every
        self.max_grad_norm    = max_grad_norm
        self.num_train_steps  = num_train_steps
        self.window_size      = window_size

        # ── MIND ────────────────────────────────────────────
        _mp = mind_params or {}
        _nlr = _mp.get("non_local_region_size", 9)
        in_ch = _nlr * _nlr
        self.mind_extractor = MIND(
            non_local_region_size=_nlr,
            patch_size=_mp.get("patch_size", 7),
            neighbor_size=_mp.get("neighbor_size", 3),
            gaussian_patch_sigma=_mp.get("gaussian_patch_sigma", 3.0),
        ).cuda()
        self.mind_loss_fn = MINDLoss(
            non_local_region_size=_nlr,
            patch_size=_mp.get("patch_size", 7),
            neighbor_size=_mp.get("neighbor_size", 3),
            gaussian_patch_sigma=_mp.get("gaussian_patch_sigma", 3.0),
        ).cuda()

        # ── Registration model ───────────────────────────────
        self.R_A = Reg(256, 256, in_ch, in_ch, dim=2).cuda()
        self.spatial_transform = Transformer_2D().cuda()

        # ── Optimizer ───────────────────────────────────────
        self.optimizer = torch.optim.Adam(self.R_A.parameters(), lr=lr)

        # ── DataLoader ───────────────────────────────────────
        self.dl = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_patient_window, num_workers=4, drop_last=True,
        )

        # ── Accelerate ───────────────────────────────────────
        self.accelerator = Accelerator()
        self.R_A, self.spatial_transform, self.mind_extractor, self.mind_loss_fn, \
            self.optimizer, self.dl = self.accelerator.prepare(
                self.R_A, self.spatial_transform, self.mind_extractor, self.mind_loss_fn,
                self.optimizer, self.dl,
            )
        self.is_main = self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return
        package = {"RA": self.accelerator.unwrap_model(self.R_A).state_dict()}
        torch.save(package, str(self.checkpoints_folder / path))
        print(f"[RegistrationPretrainer] Saved → {self.checkpoints_folder / path}")

    def load(self, path):
        pkg = torch.load(path, map_location=self.accelerator.device)
        if "RA" in pkg:
            self.accelerator.unwrap_model(self.R_A).load_state_dict(pkg["RA"])
        print(f"[RegistrationPretrainer] Loaded from {path}")

    def __call__(self):
        import wandb
        from tqdm import tqdm

        steps_per_epoch = len(self.dl)
        total_epochs    = math.ceil(self.num_train_steps / max(steps_per_epoch, 1))
        global_step     = 0

        for epoch in range(1, total_epochs + 1):
            epoch_loss = 0.0
            pbar = tqdm(self.dl, desc=f"[RegPretrain Epoch {epoch}] Training", ncols=120)

            for batch in pbar:
                global_step += 1
                nect = batch["nect_window"].to(self.accelerator.device, dtype=torch.float32)
                cect = batch["cect_window"].to(self.accelerator.device, dtype=torch.float32)
                if nect.ndim == 5: nect = nect.squeeze(2)
                if cect.ndim == 5: cect = cect.squeeze(2)

                # Use center slice only
                c = nect.shape[1] // 2
                nect_c = nect[:, c:c+1]
                cect_c = cect[:, c:c+1]

                reg_src = self.mind_extractor(nect_c)
                reg_tgt = self.mind_extractor(cect_c)
                Trans   = self.R_A(reg_src, reg_tgt)
                nect_warped = self.spatial_transform(nect_c, Trans)

                L_align  = self.mind_loss_fn(nect_warped, cect_c)
                sm_loss  = smooothing_loss(Trans)
                loss     = L_align + 1e-2 * sm_loss

                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(
                    list(self.R_A.parameters()), self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                if self.is_main:
                    wandb.log({'regpre/L_align': L_align.item(), 'regpre/sm_loss': sm_loss.item(),
                               'regpre/loss': loss.item(), 'epoch': epoch}, step=global_step)

                if global_step >= self.num_train_steps:
                    break

            if self.is_main and divisible_by(epoch, self.checkpoint_every):
                self.save(f'checkpoint.{epoch}.pt')

            if global_step >= self.num_train_steps:
                break

        print("RegistrationPretrainer finished.")