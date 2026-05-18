"""
Watermark-integrated Trainer for RevMark-SinSR.

Inherits from TrainerDistillDifIR and overrides training_step to:
  - Replace random noise with watermarked Gaussian noise (all phases)
  - Add L_bin loss for quantile-bin constraint (Phase B/C)
  - Add differentiable attack augmentation (Phase C)
"""

import os
import sys
import math
import functools
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from trainer import TrainerDistillDifIR
from watermark_codec import WatermarkCodec
from models.gaussian_diffusion import _extract_into_tensor
from models.basic_ops import mean_flat


class TrainerWatermarkDifIR(TrainerDistillDifIR):
    def __init__(self, configs):
        super().__init__(configs)

        # Watermark codec（watermark.enabled=false 时关闭：训练噪声与 vanilla SinSR 一致）
        wm_cfg = configs.get('watermark', {})
        self.wm_enabled = bool(wm_cfg.get('enabled', True))
        if self.wm_enabled:
            self.wm_codec = WatermarkCodec(
                num_channels=configs.autoencoder.params.ddconfig.z_channels,  # 3
                latent_size=configs.model.params.image_size,  # 64
                ch_factor=wm_cfg.get('ch_factor', 1),
                hw_factor=wm_cfg.get('hw_factor', 8),
                use_chacha=wm_cfg.get('use_chacha', True),
                fpr=wm_cfg.get('fpr', 1e-6),
                user_number=wm_cfg.get('user_number', 1),
            )
        else:
            self.wm_codec = None

        # Phase-specific loss weights
        self.phase = configs.train.get('phase', 'A')
        self.lambda_xT_wm = configs.train.get('lambda_xT', 0.0)
        self.lambda_bin = configs.train.get('lambda_bin', 0.0)
        self.lambda_lr = configs.train.get('lambda_lr', 0.0)
        self.bin_margin = configs.train.get('bin_margin', 0.05)  # delta for L_bin

        # Phase C: attack augmentation
        self.use_attack_aug = configs.train.get('use_attack_aug', False)
        self.attack_cfg = configs.train.get('attack_aug', {})

        if self.rank == 0:
            from loguru import logger
            logger.info(f"[WM] Phase: {self.phase}")
            if self.wm_enabled:
                logger.info(f"[WM] Watermark enabled; capacity: {self.wm_codec.mark_length} bits")
                logger.info(f"[WM] lambda_xT={self.lambda_xT_wm}, lambda_bin={self.lambda_bin}, lambda_lr={self.lambda_lr}")
                logger.info(f"[WM] Detection thresholds: tau_onebit={self.wm_codec.tau_onebit}, tau_bits={self.wm_codec.tau_bits}")
            else:
                logger.info("[WM] Watermark disabled (watermark.enabled=false); noise = torch.randn like TrainerDistillDifIR")
                if self.lambda_xT_wm > 0 or self.lambda_bin > 0 or self.lambda_lr > 0:
                    logger.warning("[WM] Watermark losses are ignored while enabled=false")

    @property
    def lr_recovery_module(self):
        """Get the unwrapped lr_recovery module (strips DDP wrapper if present)."""
        if self.lr_recovery is None:
            return None
        return self.lr_recovery.module if isinstance(self.lr_recovery, DDP) else self.lr_recovery

    def build_model(self):
        super().build_model()  # UNet, teacher, autoencoder, EMA, diffusion

        lr_cfg = self.configs.get('lr_recovery', {})
        if lr_cfg.get('enabled', False):
            from models.lr_recovery import LRRecoveryNet
            self.lr_recovery = LRRecoveryNet(
                in_channels=3,
                hidden_channels=lr_cfg.get('hidden_channels', 64),
                num_res_blocks=lr_cfg.get('num_res_blocks', 4),
                dropout=lr_cfg.get('dropout', 0.0),
            ).cuda()

            # Load pretrained g_ψ if available
            lr_ckpt = lr_cfg.get('ckpt_path', None)
            if lr_ckpt is not None:
                if self.rank == 0:
                    self.logger.info(f"[WM] Loading LR recovery from {lr_ckpt}")
                state = torch.load(lr_ckpt, map_location=f"cuda:{self.rank}")
                self.lr_recovery.load_state_dict(state)

            # EMA state initialized BEFORE DDP wrapping (no module. prefix)
            if self.rank == 0:
                self.ema_state_lr = OrderedDict(
                    (k, deepcopy(v.data)) for k, v in self.lr_recovery.state_dict().items()
                )

            # DDP wrap (must be after loading weights and EMA init)
            if self.num_gpus > 1:
                self.lr_recovery = DDP(
                    self.lr_recovery, device_ids=[self.rank],
                    broadcast_buffers=False
                )

            if self.rank == 0:
                n_params = sum(p.numel() for p in self.lr_recovery.parameters())
                self.logger.info(f"[WM] LR recovery net: {n_params/1e3:.1f}K params, enabled=True")
        else:
            self.lr_recovery = None
            if self.rank == 0:
                self.logger.info("[WM] LR recovery net: disabled")

    def setup_optimizaton(self):
        params = list(self.model.parameters())
        if self.lr_recovery is not None:
            params += list(self.lr_recovery.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=self.configs.train.lr, weight_decay=self.configs.train.weight_decay
        )

    def update_ema_model(self):
        super().update_ema_model()  # UNet EMA
        if self.rank == 0 and hasattr(self, 'ema_state_lr'):
            source = self.lr_recovery_module.state_dict()
            rate = self.ema_rate
            for k, v in self.ema_state_lr.items():
                v.mul_(rate).add_(source[k].detach().data, alpha=1 - rate)

    def save_ckpt(self):
        super().save_ckpt()  # UNet + UNet EMA
        if self.rank == 0 and self.lr_recovery is not None:
            torch.save(
                self.lr_recovery_module.state_dict(),
                self.ckpt_dir / f'lr_recovery_{self.current_iters}.pth',
            )
            torch.save(
                self.ema_state_lr,
                self.ema_ckpt_dir / f'ema_lr_recovery_{self.current_iters}.pth',
            )

    def resume_from_ckpt(self):
        super().resume_from_ckpt()  # UNet + UNet EMA + iters
        if self.lr_recovery is not None and hasattr(self, 'iters_start') and self.iters_start > 0:
            # Derive g_ψ checkpoint path from model checkpoint
            lr_ckpt_path = self.ckpt_dir / f'lr_recovery_{self.iters_start}.pth'
            if lr_ckpt_path.exists():
                if self.rank == 0:
                    self.logger.info(f"[WM] Resuming LR recovery from {lr_ckpt_path}")
                state = torch.load(lr_ckpt_path, map_location=f"cuda:{self.rank}")
                self.lr_recovery_module.load_state_dict(state)

            if self.rank == 0 and hasattr(self, 'ema_state_lr'):
                ema_lr_path = self.ema_ckpt_dir / f'ema_lr_recovery_{self.iters_start}.pth'
                if ema_lr_path.exists():
                    self.logger.info(f"[WM] Resuming LR recovery EMA from {ema_lr_path}")
                    ema_ckpt = torch.load(ema_lr_path, map_location=f"cuda:{self.rank}")
                    for k in self.ema_state_lr:
                        if k in ema_ckpt:
                            self.ema_state_lr[k] = deepcopy(ema_ckpt[k].data)

    def training_step(self, data):
        """
        Override training_step to inject watermarked noise.
        The core change: replace `noise = torch.randn(...)` with watermark-encoded noise.
        """
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        if self.configs.train.use_fp16:
            scaler = amp.GradScaler()

        self.optimizer.zero_grad()
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key: value[jj:jj+micro_batchsize] for key, value in data.items()}
            last_batch = (jj + micro_batchsize >= current_batchsize)
            micro_bs = micro_data['gt'].shape[0]

            tt = torch.randint(
                0, self.base_diffusion.num_timesteps,
                size=(micro_bs,),
                device=f"cuda:{self.rank}",
            )
            if not self.use_reflow:
                tt = torch.ones_like(tt) * (self.base_diffusion.num_timesteps - 1)

            latent_downsamping_sf = 2 ** (len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf

            # ===== Noise: 水印关闭时与 trainer.TrainerDistillDifIR 相同（标准高斯）=====
            has_wm_losses = self.wm_enabled and (
                self.lambda_bin > 0 or self.lambda_xT_wm > 0 or self.lambda_lr > 0
            )
            if not self.wm_enabled:
                noise = torch.randn(
                    size=micro_data['gt'].shape[:2] + (latent_resolution,) * 2,
                    device=f"cuda:{self.rank}",
                    dtype=micro_data['gt'].dtype,
                )
                watermarks = None
                target_bins = None
            elif has_wm_losses:
                noise, watermarks, target_bins = self.wm_codec.encode_batch_with_targets(
                    micro_bs, device=f"cuda:{self.rank}"
                )
            else:
                noise, watermarks = self.wm_codec.encode_batch(
                    micro_bs, device=f"cuda:{self.rank}"
                )
                target_bins = None

            noise = noise.to(dtype=micro_data['gt'].dtype)

            model_kwargs = {'lq': micro_data['lq']} if self.configs.model.params.cond_lq else None

            # Standard distillation losses (unchanged)
            compute_losses = functools.partial(
                self.base_diffusion.training_losses_distill,
                self.model,
                self.teacher_model,
                micro_data['gt'],
                micro_data['lq'],
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
                distill_ddpm=self.distill_ddpm,
                uncertainty_hyper=self.uncertainty_hyper,
                uncertainty_num_aux=self.uncertainty_num_aux,
                learn_xT=self.learn_xT,
                finetune_use_gt=self.finetune_use_gt,
                reformulated_reflow=self.reformulated_reflow,
                xT_cov_loss=self.xT_cov_loss,
                loss_in_image_space=self.loss_in_image_space,
            )

            if self.configs.train.use_fp16:
                with amp.autocast():
                    if last_batch or self.num_gpus <= 1:
                        losses, z_t, z0_pred = compute_losses()
                    else:
                        with self.model.no_sync():
                            losses, z_t, z0_pred = compute_losses()

                    # Add watermark losses (Phase B/C)
                    if has_wm_losses:
                        wm_losses = self._compute_wm_losses(
                            z0_pred, micro_data, tt, noise, z_t, target_bins, model_kwargs
                        )
                        for k, v in wm_losses.items():
                            losses[k] = v
                        losses["loss"] = losses["loss"] + wm_losses.get("loss_wm", 0)

                    loss = losses["loss"].mean() / num_grad_accumulate
                scaler.scale(loss).backward()
            else:
                if last_batch or self.num_gpus <= 1:
                    losses, z_t, z0_pred = compute_losses()
                else:
                    with self.model.no_sync():
                        losses, z_t, z0_pred = compute_losses()

                # Add watermark losses (Phase B/C)
                if has_wm_losses:
                    wm_losses = self._compute_wm_losses(
                        z0_pred, micro_data, tt, noise, z_t, target_bins, model_kwargs
                    )
                    for k, v in wm_losses.items():
                        losses[k] = v
                    losses["loss"] = losses["loss"] + wm_losses.get("loss_wm", 0)

                loss = losses["loss"].mean() / num_grad_accumulate
                loss.backward()

            # Log
            self.log_step_train(losses, tt * 0 if not self.use_reflow else tt,
                                micro_data, z_t, z0_pred, last_batch)

        if self.configs.train.use_fp16:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        self.update_ema_model()

    def _compute_wm_losses(self, z0_pred, micro_data, tt, noise, z_t, target_bins, model_kwargs):
        """
        Compute watermark-specific losses for Phase B/C.

        L_lr:     L1 between g_ψ(z0_pred) and z_y_gt (LR recovery in latent space)
        L_xT_wm:  MSE between original z_t (watermarked) and the model's inversion prediction
        L_bin:    quantile-bin constraint ensuring recovered noise falls in correct bins
        """
        wm_losses = {}
        loss_wm = 0

        # Get z_y_gt (encoded upsampled LR) — ground truth for LR recovery
        z_y_gt = self.base_diffusion.encode_first_stage(
            micro_data['lq'], self.autoencoder, up_sample=True
        )

        # === L_lr: latent-space LR recovery loss ===
        if self.lr_recovery is not None and self.lambda_lr > 0:
            z_y_est = self.lr_recovery(z0_pred.detach())  # detach: don't backprop to UNet
            wm_losses["loss_lr"] = F.l1_loss(z_y_est, z_y_gt)
            loss_wm = loss_wm + self.lambda_lr * wm_losses["loss_lr"]
            z_y_for_extraction = z_y_est  # use g_ψ output for noise recovery (matches test-time)
        else:
            z_y_for_extraction = z_y_gt  # fallback: use GT z_y

        # Compute predicted_xT once (shared by L_xT_wm and L_bin)
        predicted_xT = None
        if self.lambda_xT_wm > 0 or self.lambda_bin > 0:
            if z0_pred is not None:
                z0_for_inversion = z0_pred.detach()
            else:
                z0_for_inversion = self.base_diffusion.encode_first_stage(
                    micro_data['gt'], self.autoencoder, up_sample=False
                )
            # The model with t_input=0 acts as inverter: x_0 -> x_T
            predicted_xT = self.model(
                self.base_diffusion._scale_input(z0_for_inversion, tt),
                tt * 0,  # t=0 triggers inversion branch
                **model_kwargs
            )

        # === L_xT_wm: inversion accuracy loss ===
        if self.lambda_xT_wm > 0:
            wm_losses["mse_xT_wm"] = mean_flat((z_t - predicted_xT) ** 2)
            loss_wm = loss_wm + self.lambda_xT_wm * wm_losses["mse_xT_wm"]

        # === L_bin: quantile bin constraint ===
        if self.lambda_bin > 0:

            # eps_hat = (x_T_pred - z_y) / (kappa * sqrt_etas[T-1])
            kappa = self.base_diffusion.kappa
            sqrt_etas_T = _extract_into_tensor(
                self.base_diffusion.sqrt_etas, tt, z_t.shape
            )
            # Use z_y_for_extraction (g_ψ output) so L_bin gradient trains g_ψ
            eps_hat = (predicted_xT - z_y_for_extraction) / (kappa * sqrt_etas_T + 1e-8)

            # CDF of standard normal: Phi(eps_hat)
            cdf_hat = 0.5 * (1.0 + torch.erf(eps_hat.flatten(1) / math.sqrt(2.0)))

            # Target bins from encrypted message: bin in {0, 1}
            # bit=0 -> target interval [0, 0.5], bit=1 -> [0.5, 1.0]
            lower = target_bins / 2.0  # (B, latent_length)
            upper = (target_bins + 1.0) / 2.0

            delta = self.bin_margin
            # Hinge loss: penalize if cdf_hat falls outside [lower+delta, upper-delta]
            loss_below = F.relu(lower + delta - cdf_hat)
            loss_above = F.relu(cdf_hat - upper + delta)
            wm_losses["loss_bin"] = mean_flat(loss_below + loss_above)
            loss_wm = loss_wm + self.lambda_bin * wm_losses["loss_bin"]

        wm_losses["loss_wm"] = loss_wm
        return wm_losses
