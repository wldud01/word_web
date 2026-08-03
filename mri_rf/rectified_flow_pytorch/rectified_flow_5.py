from __future__ import annotations

import math
from copy import deepcopy
from collections import namedtuple
from typing import Literal, Callable
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
from rectified_flow_pytorch.utils import exists, identity, save_sample, divisible_by,collate_patient_window,   default, append_dims, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, cosine_time_warp, cosmap, set_random_seed
from rectified_flow_pytorch.backbone.unet_baseline import Unet
from rectified_flow_pytorch.dataset.dataset import MultiPatientWindowDataset
    
set_random_seed(42)

#############################################################

# 기장 기본이 되는 loss
class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)

# 
class MeanVarianceNetLoss(Module):
    def forward(self, pred, target, **kwargs):
        dist = Normal(*pred)
        return -dist.log_prob(target).mean()

# loss breakdown

LossBreakdown = namedtuple('LossBreakdown', ['total', 'main', 'data_match', 'velocity_match'])
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
        time_cond_kwarg: str | None = 'times',
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
            'cosmap'
        ] | Callable = identity,
        loss_fn_kwargs: dict = dict(),
        ema_update_after_step: int = 100,
        ema_kwargs: dict = dict(),
        data_shape: tuple[int, ...] | None = None,
        immiscible = False,
        
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

        if mean_variance_net:
            loss_fn = MeanVarianceNetLoss()

        # objective - either flow or noise (proposed by Esser / Rombach et al in SD3)
        self.predict = predict

        # automatically default to a working setting for predict epsilon
        clip_flow_during_sampling = default(clip_flow_during_sampling, predict == 'noise')

        # loss fn: option 3
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
    def predict_flow(self, model: Module, noised, *, times, alpha = None, eps = 1e-10, **model_kwargs):
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
            

        output = self.model(noised, cond=alpha, **model_kwargs)   # Unet, output은 중심 슬라이스에 대한 velocity가 출력
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
        alpha=None,
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


        # 학습 때 clipping x
        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity
        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity

        # ode step function
        def ode_fn(t, x):
            
            x = maybe_clip(x)
 
            # output = flow
            _, output = self.predict_flow(model, x, times = t, alpha=alpha, **model_kwargs)
            #print('ode_fn predict',output.max(), output.min())
            flow = output

            # 분포 모델링을 하는 경우
            if self.mean_variance_net:
                mean, variance = output
                std = variance.clamp(min = 1e-5).sqrt()
                flow = torch.normal(mean, std * temperature)

            flow = maybe_clip_flow(flow)

            return flow

        # start with random gaussian noise - y0

        #ANCHOR - noise = default(noise, torch.randn((batch_size, *data_shape), device = self.device))
        noise = model_kwargs.pop("cond")  # window 모델링의 경우, NECT 윈도우 (B, 5, H, W)
        if noise.ndim == 5: # NECT 윈도우 (B, 5, 1, H, W), 차원 맞추기
            noise = noise.squeeze(2)

        
        # steps 만큼 linear하게 시간 생성, 적분할 때 사용
        times = torch.linspace(0., 1., steps, device = self.device)
        #print(times) step 0 --> 0

        # ode / odeint(적분 함수) 를 이용한 샘플링
        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)  # (T, B, C, H, W)

        # 마지막으로 나온 결과만 반환
        sampled_data = trajectory[-1]
        #print('sampled_data',trajectory.shape )

        self.train(was_training)

        return self.data_unnormalize_fn(sampled_data)



    def forward(
        self,
        data,   # CECT
        noise: Tensor | None = None,    # NECT
        alpha=None,
        return_loss_breakdown = False,
        **model_kwargs
    ):
        batch, *data_shape = data.shape

        data = self.data_normalize_fn(data) # CECT
        data = data.to(dtype=next(self.model.parameters()).dtype)

        self.data_shape = default(self.data_shape, data_shape)

        # x0 - gaussian noise, x1 - data
        #noise = default(noise, torch.randn_like(data))
        assert "cond" in model_kwargs, "Please provide NECT as 'cond' input"
        
        noise = model_kwargs.pop("cond")  # NECT 윈도우 (B, 5, H, W)
        if noise.ndim == 5: # NECT 윈도우 (B, 5, 1, H, W)
            noise = noise.squeeze(2)
            data = data.squeeze(2)

        # times, and times with dimension padding on right
        times = torch.rand(batch, device = self.device)
        padded_times = append_dims(times, data.ndim - 1)


        def get_noised_and_flows(model, t):

            # maybe noise schedule
            t = self.noise_schedule(t)

            # Algorithm 2 in paper
            # linear interpolation of noise with data using random times
            # x1 * t + x0 * (1 - t) - so from noise (time = 0) to data (time = 1.)
            # (1 - t) * NECT + t * CECT 
            center_idx = noise.shape[1] // 2 if noise.shape[1] > 1 else 0
            # source slice
            noise_center = noise[:, center_idx:center_idx+1, ...]
            data_center = data[:, center_idx:center_idx+1, ...]
            
            #!NOTE t 시점의 NECT와 CECT의 보간, data도 window shape이여야 함, 각 nect slice마다 flow를 가지고 있기 때문에
            #print(noise.shape, data.shape)
            noised = noise.lerp(data, t) # noise -> data from 0. to 1.

            #!NOTE Ground truth target slice에 대한 flow
            #flow = data_center - noise_center
            flow = data - noise

            # the model predicts the flow from the noised data
            #our task = CECT - NECT
            model_output, model_output = self.predict_flow(model, noised, alpha = alpha, times = t,**model_kwargs)    # output , flow

            # if mean variance network, sample from normal
            pred_flow = model_output

            if self.mean_variance_net:
                mean, variance = model_output
                pred_flow = torch.normal(mean, variance)

            # predicted data will be the noised xt + flow * (1. - t)
            pred_data = noised + pred_flow * (1. - t)

            return model_output, flow, pred_flow, pred_data

        # getting flow and pred flow for main model
        output, flow, pred_flow, pred_data = get_noised_and_flows(self.model, padded_times)


        # determine target, depending on objective
        if self.predict == 'flow':
            target = flow
        elif self.predict == 'noise':
            target = noise
        else:
            raise ValueError(f'unknown objective {self.predict}')

        # losses
        main_loss = self.loss_fn(output, target, pred_data = pred_data, times = times, data = data)

        consistency_loss = data_match_loss = velocity_match_loss = 0.

        # total loss

        total_loss = main_loss + consistency_loss * self.consistency_loss_weight

        if not return_loss_breakdown:
            return total_loss, pred_data

        # loss breakdown

        return total_loss, flow, times, LossBreakdown(total_loss, main_loss, data_match_loss, velocity_match_loss)
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

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError
import torch
import math
from torchvision.utils import save_image
from einops import rearrange
from tqdm import tqdm

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

class Trainer(Module):
    def __init__(
        self,
        rectified_flow: dict | RectifiedFlow ,
        *,
        dataset: dict | MultiPatientWindowDataset,
        num_train_steps = 70_000,
        learning_rate = 1e-4,   #NOTE - learning rate 수정 default = 3e-4
        batch_size = 8,
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
        max_grad_norm = 0.5
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        if isinstance(dataset, dict):
            dataset = MultiPatientWindowDataset(**dataset)

        if isinstance(rectified_flow, dict):
            rectified_flow = RectifiedFlow(**rectified_flow)

        self.model = rectified_flow
        self.model = self.model.to(torch.float32)
        # determine whether to keep track of EMA (if not using consistency FM)
        # which will determine which model to use for sampling

        use_ema &= not getattr(self.model, 'use_consistency', False)

        self.use_ema = use_ema
        self.ema_model = None

        if self.is_main and use_ema:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)

        # optimizer, dataloader, and all that
        #NOTE -  optimaizer 수정
        #self.optimizer = AdamW(rectified_flow.parameters(), lr = learning_rate, **adam_kwargs)
        self.optimizer = Adam(rectified_flow.parameters(), lr = learning_rate, **adam_kwargs)
        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True, collate_fn=collate_patient_window )

        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)

        self.num_train_steps = num_train_steps

        self.return_loss_breakdown = isinstance(rectified_flow, RectifiedFlow)

        # folders

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every
        self.sample_temperature = sample_temperature

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

        self.max_grad_norm = max_grad_norm
        self.epoch_start = 1

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = {
            "model": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        if self.ema_model is not None:
            save_package["ema_model"] = self.ema_model.state_dict()

        torch.save(save_package, str(self.checkpoints_folder / path))

    def load(self, path):
        if not self.is_main:
            return

        load_package = torch.load(path)
        
        self.epoch_start = int(path.split('.')[-2]) + 1
        print(self.epoch_start)
        
        self.model.load_state_dict(load_package["model"])
        self.optimizer.load_state_dict(load_package["optimizer"])

        if "ema_model" in load_package and self.ema_model is not None:
            self.ema_model.load_state_dict(load_package["ema_model"])

    def log(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def log_images(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    # Trainer.sample 교체
    def sample(self, fname: str):
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        eval_model = default(self.ema_model, self.model)
        dl = cycle(self.dl)
        mock_data = next(dl)

        # 🔹 Paired dataset (NECT, CECT) 대응
        if isinstance(mock_data, (list, tuple)):
            nect, cect = mock_data
            cond = nect.to(self.accelerator.device, dtype=torch.float32)
            data_shape = cect.shape[1:]
        else:
            print("[WARN] Using unpaired dataset for sampling. No condition will be used.")
            cond = None
            cect = mock_data.to(self.accelerator.device, dtype=torch.float32)
            data_shape = cect.shape[1:]

        add_kwargs = {}
        if isinstance(eval_model.model, RectifiedFlow):
            add_kwargs.update(temperature=self.sample_temperature)

        with torch.no_grad():
            # 🔹 cond (NECT) 입력 지원
            sampled = eval_model.sample(
                batch_size=self.num_samples,
                data_shape=data_shape,
                cond=cond,
                **add_kwargs
            )

            

        if sampled.ndim != 4:
            raise ValueError(f"Unexpected sampled shape: {sampled.shape}")
        B, C, H, W = sampled.shape
        sampled_reshaped = sampled.reshape(B * C, 1, H, W)   # (B*5, 1, H, W)

        grid = make_grid(sampled_reshaped, nrow=5)
        save_image(grid, fname)
        self.accelerator.print(f"[Saved sample] {fname}")
        return fname



    # -------------------------
    # Trainer.forward() 수정
    # -------------------------
    def forward(self):
        num_epochs = math.ceil(self.num_train_steps / len(self.dl))  # 1epoch 단위로 변경
        print(f"[INFO] Training for {num_epochs} epochs ({self.num_train_steps} steps total)")
        
        self.accelerator.wait_for_everyone()
        if self.is_main:
            if wandb.run is None:
                wandb.init(
                    project="NECT_to_CECT_RectifiedFlow",
                    name=f"run_{np.random.randint(10000)}",
                    config={
                        "num_train_steps": self.num_train_steps,
                        "batch_size": self.dl.batch_size,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                    reinit=True
                )
                self.wandb_run = wandb.run
        self.accelerator.wait_for_everyone()
        
        global_step = 0
        for epoch in range(self.epoch_start, num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(self.dl, desc=f"[Epoch {epoch}] Training", ncols=100)

            for batch in pbar:
                global_step += 1


                nect = batch["nect_window"].to(self.accelerator.device, dtype=torch.float32)
                cect = batch["cect_window"].to(self.accelerator.device, dtype=torch.float32)
                
                slice_name= batch['patient_id']

                center = nect.shape[1] // 2
                nect_c = nect[:, center:center+1, :, :]  # (1,1,H,W)
                cect_c = cect[:, center:center+1, :, :]  # (1,1,H,W)
                
                # (B,1,1,H,W) 같이 들어오는 경우 squeeze
                if nect.ndim == 5:
                    nect = nect.squeeze(2)
                if cect.ndim == 5:
                    cect = cect.squeeze(2)
                    
                B = nect.shape[0]
                
                
                # segmentation label
                if "seg_cect_window" in batch:
                    seg = batch["seg_cect_window"].to(
                        self.accelerator.device, dtype=torch.float32
                    )

                    center = seg.shape[1] // 2
                    seg = seg[:, center:center+1, :, :]   # (B,1,H,W)

                    if seg.ndim == 5:
                        seg = seg.squeeze(2)              # (B,1,H,W)


                # forward: condition = NECT, target = CECT
                L_contrast, pred_cect = self.model(cect, cond=nect)  # rectified flow  alpha = seg
                L_cect_rec = F.l1_loss(pred_cect, cect)
                
                
                loss = L_contrast + L_cect_rec
                
                self.accelerator.backward(loss)

                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                wandb.log({"train/loss": loss.item(), "epoch": epoch})

                # save image
                if self.is_main and global_step % 1000 == 0:
                    with torch.no_grad():
                        save_dir = f"{self.results_folder}" + "/full_train_idea_ac/" + f"epoch_{epoch}/"
                        save_sample(save_dir, nect, global_step, 'nect')
                        save_sample(save_dir, cect, global_step, 'cect')
                        save_sample(save_dir, pred_cect, global_step, 'cect_hat')


            avg_loss = epoch_loss / len(self.dl)
            
            # 로깅
            if self.is_main:
                wandb.log({"train/loss_avg": avg_loss, "epoch": epoch})
            self.accelerator.print(f"Epoch [{epoch}] Avg Loss: {avg_loss:.4f}")
        

            # Validation 수행
            if self.is_main and divisible_by(epoch, 100):  # 매 epoch마다 수행
                val_psnr, val_ssim, val_mse, val_nmse = self.validate()
                wandb.log({
                    "val/PSNR": val_psnr,
                    "val/SSIM": val_ssim,
                    "val/MSE": val_mse,
                    "val/NMSE": val_nmse,
                    "epoch": epoch
                })

            # 모델 및 샘플 저장
            if self.is_main and divisible_by(epoch, self.checkpoint_every):
                self.save(f'checkpoint.{epoch}.pt')
                

        print("Training complete ")


    @torch.no_grad()
    def validate(self):
        """
        Validation:
        NECT → (condition)
        CECT → (target)
        """
        self.model.eval()

        psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.accelerator.device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.accelerator.device)
        mse = MeanSquaredError().to(self.accelerator.device)

        total_psnr, total_ssim, total_mse = 0.0, 0.0, 0.0
        n = 0

        # 🔹 validation 전용 DataLoader 사용 (없으면 train dataloader 사용)
        val_loader = getattr(self, "val_dl", self.dl)

        pbar = tqdm(val_loader, desc="[Validation]", ncols=100)
        for nect, cect in pbar:
            nect = nect.to(self.accelerator.device, dtype=torch.float32)
            cect = cect.to(self.accelerator.device, dtype=torch.float32)

            # 🔹 RectifiedFlow 샘플링 (NECT → CECT)
            pred = self.model.sample(
                batch_size=nect.size(0),
                data_shape=nect.shape[1:],
                cond=nect
            )

            # 후처리
            pred = pred.clamp(0, 1)

            # 🔹 metric 계산
            total_psnr += psnr(pred, cect).item()
            total_ssim += ssim(pred, cect).item()
            total_mse += mse(pred, cect).item()
            n += 1

            pbar.set_postfix({
                "PSNR": f"{total_psnr/n:.2f}",
                "SSIM": f"{total_ssim/n:.3f}"
            })

        # 평균 계산
        val_psnr = total_psnr / n
        val_ssim = total_ssim / n
        val_mse  = total_mse / n
        val_nmse = val_mse / torch.mean(cect ** 2).item()

        # 🔹 결과 시각화 (마지막 배치)
        grid = make_grid(pred, nrow=min(pred.size(0), 4))
        save_image(grid, str(self.results_folder / f"val_sample.png"))

        self.accelerator.print(
            f"[Validation] PSNR={val_psnr:.3f} | SSIM={val_ssim:.4f} | MSE={val_mse:.5f} | NMSE={val_nmse:.5f}"
        )

        return val_psnr, val_ssim, val_mse, val_nmse



   