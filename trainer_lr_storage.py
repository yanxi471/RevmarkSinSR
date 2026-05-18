import math
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from inference_wm import WatermarkSampler
from lr_storage_codec import LRStorageCodec, integer_haar_ll_levels_torch
from models.lr_storage_predictor import LRStoragePredictorNet
from trainer import TrainerBase, _resolve_stage_snapshot_iters
from watermark_codec import WatermarkCodec


def _laplace_cdf(x, mu, scale):
    delta = x - mu
    return torch.where(
        delta < 0,
        0.5 * torch.exp(delta / scale),
        1.0 - 0.5 * torch.exp(-delta / scale),
    )


def discretized_laplace_bits(target, mu, log_scale):
    bin_width = 1.0 / 255.0
    scale = torch.exp(log_scale).clamp(min=1e-4, max=1.0)
    upper = torch.clamp(target + 0.5 * bin_width, 0.0, 1.0)
    lower = torch.clamp(target - 0.5 * bin_width, 0.0, 1.0)
    probs = (_laplace_cdf(upper, mu, scale) - _laplace_cdf(lower, mu, scale)).clamp(min=1e-9)
    return -torch.log2(probs)


class TrainerLRStoragePredictor(TrainerBase):
    def __init__(self, configs):
        self.ema_rate = configs.train.get('ema_rate', 0.999)
        super().__init__(configs)
        self.lambda_l1 = float(configs.train.get('lambda_l1', 0.02))
        self.analysis_levels = int(configs.lr_storage.get('analysis_levels', 2))
        self.use_ema_val = bool(configs.train.get('use_ema_val', True))

    def _predictor_id(self):
        predictor_cfg = self.configs.lr_storage.get('predictor', {})
        predictor_id = predictor_cfg.get('predictor_id', None)
        if predictor_id:
            return str(predictor_id)
        ckpt_path = predictor_cfg.get('ckpt_path', None)
        if ckpt_path:
            return str(ckpt_path).split('/')[-1].rsplit('.', 1)[0]
        return "phasee_predictor_v1"

    def _build_eval_codec(self, predictor_model):
        carrier_cfg = self.configs.lr_storage.get('carrier', {})
        return LRStorageCodec(
            mode=self.configs.lr_storage.get('mode', 'ihaar_residual_lsb_v1'),
            packet_mode=self.configs.lr_storage.get('packet_mode', None),
            carrier_type=carrier_cfg.get('type', None),
            carrier_config=carrier_cfg,
            bitplanes=self.configs.lr_storage.get('bitplanes', 1),
            carrier_fraction=self.configs.lr_storage.get('carrier_fraction', 'auto'),
            use_chacha=self.configs.lr_storage.get('use_chacha', True),
            predictor=predictor_model,
            predictor_id=self._predictor_id(),
            analysis_levels=self.analysis_levels,
            compressor_type=self.configs.lr_storage.get('compressor_type', 'zlib'),
            compressor_level=self.configs.lr_storage.get('compressor_level', self.configs.lr_storage.get('compressor', {}).get('level', 9)),
            residual_pack=self.configs.lr_storage.get('residual_pack', self.configs.lr_storage.get('compressor', {}).get('residual_pack', 'zigzag_u16')),
        )

    def build_model(self):
        predictor_cfg = self.configs.lr_storage.get('predictor', {})
        self.predictor = LRStoragePredictorNet(
            in_channels=3,
            hidden_channels=predictor_cfg.get('hidden_channels', 64),
            num_res_blocks=predictor_cfg.get('num_res_blocks', 4),
            dropout=predictor_cfg.get('dropout', 0.0),
            init_scale=predictor_cfg.get('init_scale', 0.05),
        ).cuda()

        if self.rank == 0:
            self.ema_predictor = deepcopy(self.predictor).cuda().eval()
            self.ema_state = OrderedDict(
                (k, deepcopy(v.data)) for k, v in self.predictor.state_dict().items()
            )

        sampler_cfg = OmegaConf.create(OmegaConf.to_container(self.configs, resolve=True))
        if 'lr_storage' in sampler_cfg and sampler_cfg.lr_storage is not None:
            sampler_cfg.lr_storage.enabled = False

        wm_cfg = self.configs.get('watermark', {})
        wm_codec = WatermarkCodec(
            num_channels=self.configs.autoencoder.params.ddconfig.z_channels,
            latent_size=self.configs.model.params.image_size,
            ch_factor=wm_cfg.get('ch_factor', 1),
            hw_factor=wm_cfg.get('hw_factor', 8),
            use_chacha=wm_cfg.get('use_chacha', True),
            fpr=wm_cfg.get('fpr', 1e-6),
            user_number=wm_cfg.get('user_number', 1),
        )
        self.cover_sampler = WatermarkSampler(
            sampler_cfg,
            wm_codec,
            lr_recovery_net=None,
            sf=self.configs.diffusion.params.sf,
            seed=self.configs.train.get('seed', 123456),
        )
        for module in [self.cover_sampler.model, self.cover_sampler.autoencoder]:
            if module is None:
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=self.configs.train.lr,
            weight_decay=self.configs.train.weight_decay,
        )
        milestones = self.configs.train.get('milestones', [])
        self.lr_sheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.5
        )

    def update_ema_model(self):
        if self.rank != 0:
            return
        source = self.predictor.state_dict()
        for key, value in self.ema_state.items():
            value.mul_(self.ema_rate).add_(source[key].detach().data, alpha=1 - self.ema_rate)

    def reload_ema_model(self):
        if self.rank != 0:
            return
        self.ema_predictor.load_state_dict(self.ema_state)

    def save_ckpt(self):
        if self.rank != 0:
            return
        ckpt_path = self.ckpt_dir / f'lr_storage_predictor_{self.current_iters}.pth'
        torch.save(
            {
                'iters_start': self.current_iters,
                'log_step': dict(self.log_step),
                'state_dict': self.predictor.state_dict(),
            },
            ckpt_path,
        )
        ema_ckpt_path = self.ema_ckpt_dir / f'ema_lr_storage_predictor_{self.current_iters}.pth'
        torch.save(self.ema_state, ema_ckpt_path)

    def resume_from_ckpt(self):
        if not self.configs.resume:
            self.iters_start = 0
            return

        ckpt = torch.load(self.configs.resume, map_location=f'cuda:{self.rank}')
        state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        self.predictor.load_state_dict(state)
        self.iters_start = int(ckpt.get('iters_start', 0))
        if self.rank == 0:
            self.logger.info(f"=> Loaded predictor checkpoint from {self.configs.resume}")
            if 'log_step' in ckpt:
                self.log_step = ckpt['log_step']

            ema_path = self.ema_ckpt_dir / ("ema_" + self.configs.resume.split('/')[-1])
            if ema_path.exists():
                ema_ckpt = torch.load(ema_path, map_location=f'cuda:{self.rank}')
                for key in self.ema_state:
                    if key in ema_ckpt:
                        self.ema_state[key] = deepcopy(ema_ckpt[key].data)
        for _ in range(self.iters_start):
            self.adjust_lr()

    def _prepare_predictor_batch(self, lq_tensor):
        lq_01 = (lq_tensor * 0.5 + 0.5).clamp(0.0, 1.0)
        with torch.no_grad():
            sr_cover, _, _ = self.cover_sampler.encode_sr(
                lq_01,
                payload_bits=None,
                enable_lr_storage=False,
            )
        sr_cover_u8 = torch.round(sr_cover.clamp(0.0, 1.0) * 255.0).to(torch.int32)
        lr_u8 = torch.round(lq_01 * 255.0).to(torch.int32)
        ll2_u8 = integer_haar_ll_levels_torch(sr_cover_u8, levels=self.analysis_levels)
        return sr_cover, sr_cover_u8, ll2_u8.float() / 255.0, lr_u8.float() / 255.0

    def training_step(self, data):
        current_batchsize = data['lq'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        self.optimizer.zero_grad()
        loss_sum = rate_sum = l1_sum = proxy_bpp_sum = 0.0
        num_micro = 0

        for jj in range(0, current_batchsize, micro_batchsize):
            micro = {key: value[jj:jj + micro_batchsize] for key, value in data.items()}
            _, _, ll2, target = self._prepare_predictor_batch(micro['lq'])

            mu, log_scale = self.predictor(ll2)
            rate_map = discretized_laplace_bits(target, mu, log_scale)
            loss_rate = rate_map.mean()
            loss_l1 = F.l1_loss(mu, target)
            loss = loss_rate + self.lambda_l1 * loss_l1
            (loss / num_grad_accumulate).backward()

            loss_sum += float(loss.item())
            rate_sum += float(loss_rate.item())
            l1_sum += float(loss_l1.item())
            proxy_bpp_sum += float(loss_rate.item() * target.shape[1])
            num_micro += 1

        self.optimizer.step()
        self.update_ema_model()

        denom = max(num_micro, 1)
        return {
            'loss': loss_sum / denom,
            'loss_rate': rate_sum / denom,
            'loss_l1': l1_sum / denom,
            'proxy_bpp': proxy_bpp_sum / denom,
        }

    @torch.no_grad()
    def validation(self):
        if self.rank != 0:
            return

        predictor_model = self.predictor
        if self.use_ema_val:
            self.reload_ema_model()
            predictor_model = self.ema_predictor
        predictor_model.eval()

        codec = self._build_eval_codec(predictor_model)
        max_batches = self.configs.train.get('max_val_batches', None)
        num_batches = 0
        proxy_bpp = loss_l1 = 0.0
        raw_bits = compressed_bits = payload_bits = 0.0
        count = 0

        for data in self.dataloaders['val']:
            data = self.prepare_data(data, phase='val')
            sr_cover, sr_cover_u8, ll2, target = self._prepare_predictor_batch(data['lq'])
            mu, log_scale = predictor_model(ll2)
            rate_map = discretized_laplace_bits(target, mu, log_scale)

            proxy_bpp += float(rate_map.mean().item() * target.shape[1])
            loss_l1 += float(F.l1_loss(mu, target).item())

            lr_u8 = torch.round(target * 255.0).to(torch.uint8)
            sr_cover_u8 = sr_cover_u8.to(torch.uint8)
            batch_size = int(sr_cover.shape[0])
            for idx in range(batch_size):
                sr_np = sr_cover_u8[idx].permute(1, 2, 0).cpu().numpy()
                lr_np = lr_u8[idx].permute(1, 2, 0).cpu().numpy()
                stats = codec.analyze_payload(sr_np, lr_np)
                raw_bits += float(stats['raw_bits'])
                compressed_bits += float(stats['compressed_bits'])
                payload_bits += float(stats['payload_bits'])
                count += 1

            num_batches += 1
            if max_batches is not None and num_batches >= int(max_batches):
                break

        if count == 0:
            self.logger.info("Validation skipped: no samples processed")
        else:
            self.logger.info(
                "Validation: "
                f"proxy_bpp={proxy_bpp / count:.4f} "
                f"l1={loss_l1 / max(num_batches, 1):.5f} "
                f"raw_bits={raw_bits / count:.1f} "
                f"compressed_bits={compressed_bits / count:.1f} "
                f"payload_bits={payload_bits / count:.1f} "
                f"savings={(raw_bits - compressed_bits) / count:.1f}"
            )

        self.predictor.train()

    def train(self):
        self.init_logger()
        self.build_model()
        self.setup_optimizaton()
        self.resume_from_ckpt()
        self.build_dataloader()

        self.predictor.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1
            data = self.prepare_data(next(self.dataloaders['train']))

            tic = time.time()
            losses = self.training_step(data)

            if self.current_iters == 1 or self.current_iters % self.configs.train.log_freq[0] == 0:
                elapsed = time.time() - tic
                self.logger.info(
                    f"Train: {self.current_iters:06d}/{self.configs.train.iterations:06d} "
                    f"loss={losses['loss']:.4f} "
                    f"rate={losses['loss_rate']:.4f} "
                    f"l1={losses['loss_l1']:.5f} "
                    f"proxy_bpp={losses['proxy_bpp']:.4f} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e} "
                    f"time={elapsed:.2f}s"
                )

            self.adjust_lr()

            save_freq = int(self.configs.train.save_freq)
            ms = _resolve_stage_snapshot_iters(self.configs)
            need_save = self.current_iters % save_freq == 0
            if ms and self.current_iters in ms and self.current_iters % save_freq != 0:
                need_save = True
            if need_save:
                self.save_ckpt()
                self._maybe_mirror_stage_snapshot()

            if 'val' in self.dataloaders and self.current_iters % self.configs.train.get('val_freq', 1000) == 0:
                self.validation()

            if self.current_iters % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(self.current_iters)

        self.close_logger()
