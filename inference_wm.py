"""
Inference pipeline for RevMark-SinSR.

Two modes:
  1. Encode SR: Input LQ + payload -> watermarked SR image
  2. Extract:   Input SR image -> recovered payload

Usage:
  # Encode: embed watermark during SR
  python inference_wm.py --mode encode -i input/ -o output/ \
      --ckpt weights/SinSR_wm.pth --scale 4

  # Extract: recover watermark from SR image
  python inference_wm.py --mode extract -i sr_image.png \
      --ckpt weights/SinSR_wm.pth --scale 4 --wm_keys keys.pt
"""

import os
import sys
import math
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

from lr_storage_codec import LRStorageCodec, RAW_PACKET_MODE, resolve_lr_storage_layout
from sampler import BaseSampler
from watermark_codec import WatermarkCodec
from models.lr_storage_predictor import LRStoragePredictorNet
from utils import util_image, util_common
from utils.roi_wm_mask import build_roi_wm_masks_for_lq_batch
from models.gaussian_diffusion import _extract_into_tensor


class WatermarkSampler(BaseSampler):
    """Extends BaseSampler with watermark encode/extract capabilities."""

    def __init__(self, configs, wm_codec, lr_recovery_net=None, lr_storage_predictor=None, **kwargs):
        wm_cfg = configs.get('watermark', {})
        self.wm_enabled = bool(wm_cfg.get('enabled', True))
        if self.wm_enabled and wm_codec is None:
            raise ValueError("watermark.enabled=true 时需要提供 wm_codec（WatermarkCodec）")
        self.wm_codec = wm_codec if self.wm_enabled else None
        self.lr_recovery_net = lr_recovery_net
        self.lr_storage_cfg = configs.get('lr_storage', {})
        self.lr_storage_predictor = lr_storage_predictor
        self.lr_storage_codec = None
        if self.lr_storage_cfg.get('enabled', False):
            if self.lr_storage_predictor is None:
                self.lr_storage_predictor = self._build_lr_storage_predictor(self.lr_storage_cfg)
            self.lr_storage_codec = self._build_lr_storage_codec(self.lr_storage_cfg)
        super().__init__(configs, **kwargs)

    def _resolve_lr_storage_predictor_id(self, cfg=None, key_info=None):
        if key_info is not None and key_info.get('predictor_id', None):
            return key_info['predictor_id']
        cfg = cfg or {}
        predictor_cfg = cfg.get('predictor', {})
        predictor_id = predictor_cfg.get('predictor_id', None)
        if predictor_id:
            return str(predictor_id)
        ckpt_path = predictor_cfg.get('ckpt_path', None)
        if ckpt_path:
            return Path(ckpt_path).stem
        return "none"

    def _build_lr_storage_predictor(self, cfg):
        packet_mode, _, _ = resolve_lr_storage_layout(
            mode=cfg.get('mode', 'lsb_raw_v1'),
            packet_mode=cfg.get('packet_mode', None),
            carrier_type=cfg.get('carrier', {}).get('type', None),
        )
        if packet_mode == RAW_PACKET_MODE:
            return None
        predictor_cfg = cfg.get('predictor', {})
        ckpt_path = predictor_cfg.get('ckpt_path', None)
        if not ckpt_path:
            return None
        predictor = LRStoragePredictorNet(
            in_channels=3,
            hidden_channels=predictor_cfg.get('hidden_channels', 64),
            num_res_blocks=predictor_cfg.get('num_res_blocks', 4),
            dropout=predictor_cfg.get('dropout', 0.0),
            init_scale=predictor_cfg.get('init_scale', 0.05),
        ).cuda().eval()
        state = torch.load(ckpt_path, map_location='cuda')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        predictor.load_state_dict(state)
        return predictor

    def _build_lr_storage_codec_kwargs(self, cfg, key_info=None):
        predictor_cfg = cfg.get('predictor', {})
        carrier_cfg = cfg.get('carrier', {})
        return dict(
            mode=(key_info or {}).get('mode', cfg.get('mode', 'lsb_raw_v1')),
            packet_mode=(key_info or {}).get('packet_mode', cfg.get('packet_mode', None)),
            carrier_type=(key_info or {}).get('carrier_type', carrier_cfg.get('type', None)),
            carrier_config=carrier_cfg,
            bitplanes=cfg.get('bitplanes', 1),
            carrier_fraction=cfg.get('carrier_fraction', 'auto'),
            use_chacha=cfg.get('use_chacha', True),
            predictor=self.lr_storage_predictor,
            predictor_id=self._resolve_lr_storage_predictor_id(cfg=cfg),
            analysis_levels=cfg.get('analysis_levels', predictor_cfg.get('analysis_levels', 2)),
            compressor_type=cfg.get('compressor_type', 'zlib'),
            compressor_level=cfg.get('compressor_level', cfg.get('compressor', {}).get('level', 9)),
            residual_pack=cfg.get('residual_pack', cfg.get('compressor', {}).get('residual_pack', 'zigzag_u16')),
        )

    def _build_lr_storage_codec(self, cfg):
        return LRStorageCodec(**self._build_lr_storage_codec_kwargs(cfg))

    def _use_lr_storage(self, enable_lr_storage=None):
        if enable_lr_storage is None:
            return self.lr_storage_codec is not None
        return bool(enable_lr_storage)

    def _ensure_lr_storage_codec(self):
        if self.lr_storage_codec is None:
            self.lr_storage_codec = self._build_lr_storage_codec(self.lr_storage_cfg)
        return self.lr_storage_codec

    def _build_lr_storage_codec_from_keys(self, key_info):
        kwargs = self._build_lr_storage_codec_kwargs(self.lr_storage_cfg, key_info=key_info)
        kwargs.update(
            {
                'bitplanes': key_info.get('bitplanes', kwargs['bitplanes']),
                'carrier_fraction': key_info.get('carrier_fraction', kwargs['carrier_fraction']),
                'use_chacha': key_info.get('use_chacha', kwargs['use_chacha']),
                'predictor_id': self._resolve_lr_storage_predictor_id(cfg=self.lr_storage_cfg, key_info=key_info),
                'analysis_levels': key_info.get('analysis_levels', kwargs['analysis_levels']),
            }
        )
        return LRStorageCodec(**kwargs)

    def _tensor_to_uint8_image(self, image_tensor):
        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError(f"Expected image tensor with shape (1, C, H, W), got {tuple(image_tensor.shape)}")
        return util_image.tensor2img(
            image_tensor[0], rgb2bgr=False, min_max=(0.0, 1.0)
        )

    def _uint8_image_to_tensor(self, image_u8, device):
        image_f32 = image_u8.astype(np.float32) / 255.0
        return util_image.img2tensor(image_f32).to(device)

    def _canonicalize_lq_uint8(self, y0):
        return self._tensor_to_uint8_image(y0.clamp(0.0, 1.0))

    def apply_lr_storage(self, sr_image, y0, enable_lr_storage=None, lr_storage_keys=None):
        if not self._use_lr_storage(enable_lr_storage):
            return sr_image, None

        codec = self._ensure_lr_storage_codec()
        sr_u8 = self._tensor_to_uint8_image(sr_image.clamp(0.0, 1.0))
        lr_u8 = self._canonicalize_lq_uint8(y0)
        sr_stego_u8, lr_storage_info = codec.embed(sr_u8, lr_u8, key_info=lr_storage_keys)
        sr_stego = self._uint8_image_to_tensor(sr_stego_u8, device=sr_image.device)
        return sr_stego, lr_storage_info

    def extract_exact_lr(self, sr_image, keys_info):
        lr_storage_keys = None
        if isinstance(keys_info, dict):
            lr_storage_keys = keys_info.get('lr_storage', None)
        if lr_storage_keys is None:
            return None, {
                'lr_u8': None,
                'crc_ok': False,
                'header': None,
                'reason': 'no_lr_storage',
            }

        if self.lr_storage_codec is not None:
            codec = self.lr_storage_codec
        else:
            codec = self._build_lr_storage_codec_from_keys(lr_storage_keys)

        sr_u8 = self._tensor_to_uint8_image(sr_image.clamp(0.0, 1.0))
        lr_result = codec.extract(sr_u8, lr_storage_keys)
        if lr_result.get('crc_ok', False) and lr_result.get('lr_u8') is not None:
            lr_tensor = self._uint8_image_to_tensor(lr_result['lr_u8'], device=sr_image.device)
        else:
            lr_tensor = None
        return lr_tensor, lr_result

    def z_y_from_lq_normalized(self, y0):
        """
        与 sample_func_with_noise 使用相同的 reflect padding，再经 VAE 编码得到 z_y。
        用于构造与 prior_sample 形状一致的 noise（随机噪声或水印噪声）。
        y0: (N,3,H,W), [-1,1]
        """
        desired_min_size = self.desired_min_size
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        return self.base_diffusion.encode_first_stage(
            y0, self.autoencoder, up_sample=True
        )

    def sample_func_with_noise(self, y0, noise, one_step=True, apply_decoder=True):
        """
        Run SR with a specific noise tensor (watermarked noise).

        Args:
            y0: (N, 3, H, W) LQ image in [-1, 1]
            noise: latent 空间噪声，形状须与 z_y_from_lq_normalized(y0) 一致
            one_step: use single-step denoising
            apply_decoder: decode from latent to image space
        """
        desired_min_size = self.desired_min_size
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False

        model_kwargs = {'lq': y0} if self.configs.model.params.cond_lq else None

        results = self.base_diffusion.p_sample_loop(
            y=y0,
            model=self.model,
            first_stage_model=self.autoencoder,
            noise=noise,
            noise_repeat=False,
            clip_denoised=(self.autoencoder is None),
            denoised_fn=None,
            model_kwargs=model_kwargs,
            progress=False,
            one_step=one_step,
            apply_decoder=apply_decoder,
        )

        if flag_pad and apply_decoder:
            results = results[:, :, :ori_h * self.sf, :ori_w * self.sf]

        if not apply_decoder:
            return results["pred_xstart"]
        return results.clamp_(-1.0, 1.0)

    def encode_sr(
        self,
        y0,
        payload_bits=None,
        enable_lr_storage=None,
        roi_segmenter=None,
        roi_threshold=0.5,
        roi_wm_mask=None,
        roi_wm_in_roi=True,
    ):
        """
        Embed watermark during SR.

        Args:
            y0: (N, 3, H, W) LQ image tensor in [0, 1]
            payload_bits: optional 1D or 2D array of payload bits
            enable_lr_storage: override config-controlled exact LR storage
            roi_segmenter: 可选，含 segment_rgb01(HWC [0,1])；与 latent 网格对齐后在宏块网格上
                将 ROI 外比特置 0（区外不嵌入随机载荷）。
            roi_threshold: 分割概率 >= 该阈值视为 ROI
            roi_wm_mask: 可选，显式 (B,1,wm_h,wm_w) 或 (wm_h,wm_w) 0/1；与 roi_segmenter 二选一。
                若提供 ROI：载荷比特仍在宏块上 ROI 外置 0；且在**潜空间**做
                ``eps = M*eps_wm + (1-M)*N(0,I)``（M 与 spread 对齐），非 ROI 用纯高斯，
                以减轻背景诊断区受水印结构化噪声影响。keys_info 会附带 ``ciphertext_bits``
                与 ``latent_roi_blend``，供提取端拼接 sign(eps_hat) 与密文以完成 ChaCha 解密。
            roi_wm_in_roi: True 表示 ROI 区域使用含水印噪声；False 表示 ROI 使用随机高斯、
                非 ROI 使用含水印噪声。

        Returns:
            sr_image: (1, 3, H*sf, W*sf) SR image tensor in [0, 1]
            watermark: stored watermark bits（关闭水印时为 None）
            keys: encryption keys for extraction（关闭时 keys_info 含 watermark_disabled）
        """
        y0_norm = (y0 - 0.5) / 0.5  # [0,1] -> [-1,1]

        if not self.wm_enabled:
            with torch.no_grad():
                z_ref = self.z_y_from_lq_normalized(y0_norm)
            noise = torch.randn_like(z_ref)
            sr_norm = self.sample_func_with_noise(y0_norm, noise, one_step=True)
            sr_image = sr_norm * 0.5 + 0.5
            keys_info = {'watermark_disabled': True, 'keys': None, 'latent_size': None}
            batch_size = int(y0.shape[0])
            if self._use_lr_storage(enable_lr_storage):
                if batch_size != 1:
                    raise ValueError("LR storage embedding currently expects batch size 1")
                sr_image, lr_storage_info = self.apply_lr_storage(
                    sr_image, y0, enable_lr_storage=True
                )
                keys_info['lr_storage'] = lr_storage_info
            return sr_image, None, keys_info

        # 与 z_y / sample 路径一致：先按 desired_min_size 对 LR 做 reflect pad，再 VAE 编码。
        # latent 空间边长须被 hw_factor 整除；不能用原始 LQ H/W（如 118）直接建 Codec。
        with torch.no_grad():
            z_ref = self.z_y_from_lq_normalized(y0_norm)
        latent_h, latent_w = int(z_ref.shape[2]), int(z_ref.shape[3])
        if latent_h != latent_w:
            raise ValueError(
                f"WatermarkCodec 当前仅支持方形 latent，得到 {latent_h}x{latent_w}。"
                "请对输入做中心正方形裁切后再评估，或扩展编解码器以支持矩形 latent。"
            )
        latent_sz = latent_h
        hw_f = int(self.wm_codec.hw_factor)
        if latent_sz % hw_f != 0:
            raise ValueError(
                f"latent 边长 {latent_sz} 不能被 watermark.hw_factor={hw_f} 整除；"
                "请检查输入尺寸或配置。"
            )

        # Create a temporary codec matching the actual latent size if different
        if latent_sz != self.wm_codec.latent_size:
            wm_codec_local = WatermarkCodec(
                num_channels=self.wm_codec.num_channels,
                latent_size=latent_sz,
                ch_factor=self.wm_codec.ch_factor,
                hw_factor=self.wm_codec.hw_factor,
                use_chacha=self.wm_codec.use_chacha,
            )
        else:
            wm_codec_local = self.wm_codec

        wm_ch = int(wm_codec_local.num_channels // wm_codec_local.ch_factor)
        wm_h = int(latent_sz // hw_f)
        wm_w = wm_h

        roi_m = None
        if roi_segmenter is not None and roi_wm_mask is not None:
            raise ValueError("roi_segmenter 与 roi_wm_mask 请勿同时传入")
        if roi_segmenter is not None:
            roi_m = build_roi_wm_masks_for_lq_batch(
                y0, roi_segmenter, roi_threshold, wm_h, wm_w
            )
        elif roi_wm_mask is not None:
            roi_m = torch.as_tensor(roi_wm_mask, dtype=torch.float32, device=y0.device)
            if roi_m.ndim == 2:
                roi_m = roi_m.view(1, 1, wm_h, wm_w).expand(y0.shape[0], 1, wm_h, wm_w)
            elif roi_m.ndim == 3:
                roi_m = roi_m.unsqueeze(1)
            if roi_m.shape != (y0.shape[0], 1, wm_h, wm_w):
                raise ValueError(
                    f"roi_wm_mask 展开后期望 (B,1,{wm_h},{wm_w})，得到 {tuple(roi_m.shape)}"
                )

        # Generate watermarked noise
        batch_size = int(y0.shape[0])

        wm_region_mask = roi_m
        if roi_m is not None and not roi_wm_in_roi:
            wm_region_mask = 1.0 - roi_m

        if payload_bits is not None:
            payload_t = torch.as_tensor(payload_bits, dtype=torch.int32)
            if payload_t.ndim == 1:
                payload_t = payload_t.reshape(1, -1)
            if payload_t.shape[0] == 1 and batch_size > 1:
                payload_t = payload_t.repeat(batch_size, 1)
            if payload_t.shape[0] != batch_size:
                raise ValueError(
                    f"payload_bits batch dimension {payload_t.shape[0]} does not match input batch {batch_size}"
                )
            wmr = payload_t.reshape(batch_size, wm_ch, wm_h, wm_w)
            if wm_region_mask is not None:
                wmr = wmr * (wm_region_mask.expand(batch_size, wm_ch, wm_h, wm_w) > 0.5).long()
            eps_wm, watermarks = wm_codec_local.encode_batch(
                batch_size, device=y0.device, payloads=wmr.reshape(batch_size, -1)
            )
        else:
            watermarks_rand = torch.randint(
                0, 2, (batch_size, wm_ch, wm_h, wm_w), device=y0.device
            )
            if wm_region_mask is not None:
                watermarks_rand = watermarks_rand * (
                    wm_region_mask.expand(batch_size, wm_ch, wm_h, wm_w) > 0.5
                ).long()
            eps_wm, watermarks = wm_codec_local.encode_batch(
                batch_size,
                device=y0.device,
                payloads=watermarks_rand.reshape(batch_size, -1),
            )

        eps_wm = eps_wm.to(dtype=y0.dtype)

        # 潜空间 ROI：ROI 内用水印噪声 eps_wm，非 ROI 用标准高斯，减轻非诊断区扰动。
        # 提取时用保存的密文比特填补非 ROI 的 sign 槽位，保持 ChaCha 全局解密一致。
        eps_for_sr = eps_wm
        if wm_region_mask is not None:
            roi_exp = wm_region_mask.expand(batch_size, wm_ch, wm_h, wm_w).float()
            m_lat = roi_exp.repeat(1, wm_codec_local.ch_factor, hw_f, hw_f)
            eps_rand = torch.empty_like(eps_wm)
            blend_seeds = []
            for b in range(batch_size):
                seed = int(torch.randint(0, 2**31 - 1, (1,), device=y0.device).item())
                blend_seeds.append(seed)
                gen = torch.Generator(device=y0.device)
                gen.manual_seed(seed)
                eps_rand[b : b + 1] = torch.randn(
                    eps_wm[b : b + 1].shape,
                    device=y0.device,
                    dtype=eps_wm.dtype,
                    generator=gen,
                )
            eps_for_sr = m_lat * eps_wm + (1.0 - m_lat) * eps_rand

        # SR with (possibly blended) noise
        sr_norm = self.sample_func_with_noise(y0_norm, eps_for_sr, one_step=True)
        sr_image = sr_norm * 0.5 + 0.5  # [-1,1] -> [0,1]

        keys = wm_codec_local.get_keys()
        # Store the latent size used for this encoding
        keys_info = {
            'keys': keys,
            'latent_size': latent_sz,
        }
        if roi_m is not None:
            keys_info['roi_wm_mask'] = (
                roi_m[0, 0].detach().float().cpu().numpy()
            )  # (wm_h, wm_w)，评测用
            keys_info['roi_wm_in_roi'] = bool(roi_wm_in_roi)
            keys_info['roi_threshold'] = float(roi_threshold)
            keys_info['latent_roi_blend'] = True
            if batch_size == 1:
                keys_info['ciphertext_bits'] = np.copy(
                    wm_codec_local._messages[0]
                ).astype(np.uint8)
            else:
                keys_info['ciphertext_bits'] = [
                    np.copy(wm_codec_local._messages[b]).astype(np.uint8)
                    for b in range(batch_size)
                ]
            keys_info['roi_blend_eps_seeds'] = blend_seeds

        if self._use_lr_storage(enable_lr_storage):
            if batch_size != 1:
                raise ValueError("LR storage embedding currently expects batch size 1")
            sr_image, lr_storage_info = self.apply_lr_storage(
                sr_image, y0, enable_lr_storage=True
            )
            keys_info['lr_storage'] = lr_storage_info

        return sr_image, watermarks, keys_info

    def extract_dual_channel(self, sr_image, keys_info, sf=4):
        """
        Extract watermark and, when available, recover the exact LR bytes.

        If an LR storage capsule is present and CRC passes, use the recovered LR
        for both model conditioning and z_y reconstruction. Otherwise fall back to
        the existing g_psi/bicubic path.

        Args:
            sr_image: (1, 3, H, W) SR image tensor in [0, 1]
            keys_info: dict with 'keys' and 'latent_size' from encode step
            sf: super-resolution scale factor

        Returns:
            dict with decoded watermark, recovered LR tensor, and LR status flags
        """
        device = sr_image.device

        if not self.wm_enabled or keys_info.get('watermark_disabled'):
            raise RuntimeError(
                "水印已关闭（watermark.enabled=false）或该 SR 为无水印 vanilla 模式，无法 extract；"
                "对比实验请仅用 encode 输出或与 inference.py 的 Sampler 对照。"
            )

        # Create codec matching the latent size used during encoding
        latent_size = keys_info.get('latent_size', self.wm_codec.latent_size)
        if latent_size != self.wm_codec.latent_size:
            wm_codec_local = WatermarkCodec(
                num_channels=self.wm_codec.num_channels,
                latent_size=latent_size,
                ch_factor=self.wm_codec.ch_factor,
                hw_factor=self.wm_codec.hw_factor,
                use_chacha=self.wm_codec.use_chacha,
            )
        else:
            wm_codec_local = self.wm_codec

        keys = keys_info.get('keys', keys_info)  # backward compat

        # Normalize SR to [-1, 1]
        sr_norm = (sr_image - 0.5) / 0.5
        z_sr = self.base_diffusion.encode_first_stage(
            sr_norm, self.autoencoder, up_sample=False
        )

        lr_recovered, lr_result = self.extract_exact_lr(sr_image, keys_info)
        used_exact_lr = lr_recovered is not None and lr_result.get('crc_ok', False)

        if used_exact_lr:
            lr_norm = (lr_recovered - 0.5) / 0.5
            z_y = self.base_diffusion.encode_first_stage(
                lr_norm, self.autoencoder, up_sample=True
            )
        elif self.lr_recovery_net is not None:
            # === g_ψ path: operate in latent space ===
            # g_ψ: z_sr → z_y_est
            z_y = self.lr_recovery_net(z_sr)
            # Decode z_y to image space then downsample for model conditioning
            # Model expects lq at LR resolution (H/sf x W/sf), same as training
            if self.autoencoder is not None:
                ae_dtype = next(self.autoencoder.parameters()).dtype
                lr_decoded = self.autoencoder.decode(z_y.to(ae_dtype))
                lr_decoded = lr_decoded.float().clamp(-1.0, 1.0)
                H, W = sr_image.shape[2:]
                lr_norm = F.interpolate(
                    lr_decoded, size=(H // sf, W // sf),
                    mode='bicubic', align_corners=False
                ).clamp(-1.0, 1.0)
            else:
                lr_norm = z_y
        else:
            # === Bicubic fallback (Phase A compatible) ===
            H, W = sr_image.shape[2:]
            lr_est = F.interpolate(
                sr_image, size=(H // sf, W // sf), mode='bicubic', align_corners=False
            )
            lr_est = lr_est.clamp(0, 1)
            lr_norm = (lr_est - 0.5) / 0.5

            # VAE encode
            z_y = self.base_diffusion.encode_first_stage(
                lr_norm, self.autoencoder, up_sample=True
            )

        # Model inversion (t_input=0 triggers inversion branch)
        model_kwargs = {'lq': lr_norm} if self.configs.model.params.cond_lq else None
        t = torch.tensor([self.base_diffusion.num_timesteps - 1], device=device).long()

        with torch.no_grad():
            predicted_xT = self.model(
                self.base_diffusion._scale_input(z_sr, t),
                t * 0,  # t=0 for inversion
                **model_kwargs
            )

        # Recover noise
        kappa = self.base_diffusion.kappa
        sqrt_etas_T = _extract_into_tensor(
            self.base_diffusion.sqrt_etas, t, z_sr.shape
        )
        eps_hat = (predicted_xT - z_y) / (kappa * sqrt_etas_T)

        # Decode watermark
        wm_codec_local.set_keys(keys)
        if keys_info.get("latent_roi_blend") and keys_info.get("ciphertext_bits") is not None:
            B = int(eps_hat.shape[0])
            hw_ff = int(wm_codec_local.hw_factor)
            wm_ch = int(wm_codec_local.num_channels // wm_codec_local.ch_factor)
            wm_hn = int(latent_size // hw_ff)
            roi_np = keys_info["roi_wm_mask"]
            cipher_store = keys_info["ciphertext_bits"]
            decoded_chunks = []
            for b in range(B):
                roi_t = torch.as_tensor(roi_np, dtype=torch.float32, device=device)
                roi_m1 = roi_t.view(1, 1, wm_hn, wm_hn).expand(1, wm_ch, wm_hn, wm_hn)
                m_lat = roi_m1.repeat(1, wm_codec_local.ch_factor, hw_ff, hw_ff)
                rev = (
                    (eps_hat[b : b + 1] > 0).int().reshape(-1).cpu().numpy().astype(np.uint8)
                )
                m_stored = cipher_store if B == 1 else cipher_store[b]
                m_stored = np.asarray(m_stored, dtype=np.uint8).reshape(-1)
                m_flat = (m_lat[0] > 0.5).reshape(-1).detach().cpu().numpy()
                roi_wm_in_roi = bool(keys_info.get("roi_wm_in_roi", True))
                if roi_wm_in_roi:
                    hybrid = np.where(m_flat > 0.5, rev, m_stored)
                else:
                    hybrid = np.where(m_flat > 0.5, m_stored, rev)
                dec_b = wm_codec_local.decode_from_flat_ciphertext(
                    hybrid, sample_id=b, device=device
                )
                decoded_chunks.append(dec_b)
            decoded_watermark = torch.cat(decoded_chunks, dim=0)
        else:
            decoded_watermark = wm_codec_local.decode_batch(eps_hat)

        return {
            'decoded_watermark': decoded_watermark,
            'lr_recovered': lr_recovered,
            'lr_crc_ok': bool(lr_result.get('crc_ok', False)),
            'used_exact_lr': bool(used_exact_lr),
            'lr_result': lr_result,
        }

    def extract_watermark(self, sr_image, keys_info, sf=4):
        return self.extract_dual_channel(sr_image, keys_info, sf=sf)['decoded_watermark']


def main():
    parser = argparse.ArgumentParser(description="RevMark-SinSR Inference")
    parser.add_argument("--mode", type=str, required=True, choices=["encode", "extract"],
                        help="encode: SR + watermark embed; extract: recover watermark")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image/folder path")
    parser.add_argument("-o", "--output", type=str, default="./output_wm", help="Output folder")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--cfg_path", type=str, default="configs/SinSR_wm_phaseA.yaml",
                        help="Config yaml path")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--wm_keys", type=str, default=None,
                        help="Path to watermark keys (for extract mode)")
    parser.add_argument("--payload", type=str, default=None,
                        help="Hex string payload to embed (for encode mode)")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    # Load config
    configs = OmegaConf.load(args.cfg_path)
    configs.model.ckpt_path = args.ckpt

    # Setup watermark codec（enabled=false 时不构建，推理与 SinSR 一致）
    wm_cfg = configs.get('watermark', {})
    wm_enabled = bool(wm_cfg.get('enabled', True))
    if wm_enabled:
        wm_codec = WatermarkCodec(
            num_channels=configs.autoencoder.params.ddconfig.z_channels,
            latent_size=configs.model.params.image_size,
            ch_factor=wm_cfg.get('ch_factor', 1),
            hw_factor=wm_cfg.get('hw_factor', 8),
            use_chacha=wm_cfg.get('use_chacha', True),
        )
    else:
        wm_codec = None
        print("[INFO] watermark.enabled=false：encode 使用标准高斯噪声，与原始 SinSR 行为一致")

    # Build sampler (with optional LR recovery net)
    lr_recovery_net = None
    lr_cfg = configs.get('lr_recovery', {})
    if lr_cfg.get('enabled', False):
        from models.lr_recovery import LRRecoveryNet
        lr_recovery_net = LRRecoveryNet(
            in_channels=3,
            hidden_channels=lr_cfg.get('hidden_channels', 64),
            num_res_blocks=lr_cfg.get('num_res_blocks', 4),
            dropout=lr_cfg.get('dropout', 0.0),
        ).cuda().eval()
        lr_ckpt = lr_cfg.get('ckpt_path', None)
        if lr_ckpt is not None:
            state = torch.load(lr_ckpt, map_location='cuda')
            lr_recovery_net.load_state_dict(state)
            print(f"Loaded LR recovery net from {lr_ckpt}")

    if args.mode == "extract" and not wm_enabled:
        raise SystemExit("extract 模式需要水印开启；请使用 watermark.enabled=true 的配置或改用 vanilla 推理脚本。")

    sampler = WatermarkSampler(configs, wm_codec, lr_recovery_net=lr_recovery_net,
                               sf=args.scale, seed=args.seed)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.mode == "encode":
        # Parse payload if provided
        payload_bits = None
        if args.payload:
            if not wm_enabled:
                print("[WARN] --payload 在水印关闭时被忽略")
            else:
                hex_bytes = bytes.fromhex(args.payload)
                bits = np.unpackbits(np.frombuffer(hex_bytes, dtype=np.uint8))
                payload_bits = bits[:wm_codec.mark_length]
                if len(payload_bits) < wm_codec.mark_length:
                    payload_bits = np.pad(payload_bits, (0, wm_codec.mark_length - len(payload_bits)))

        # Process images
        if in_path.is_dir():
            im_paths = sorted(in_path.glob("*.[jpJP][pnPN]*[gG]"))
        else:
            im_paths = [in_path]

        all_keys = {}
        for im_path in im_paths:
            print(f"Processing: {im_path.name}")
            im_lq = util_image.imread(str(im_path), chn='rgb', dtype='float32')
            im_lq_tensor = util_image.img2tensor(im_lq).cuda()  # 1 x c x h x w, [0,1]

            sr_image, watermark, keys_info = sampler.encode_sr(im_lq_tensor, payload_bits)
            all_keys[im_path.stem] = {
                'keys_info': keys_info,
                'watermark': watermark.cpu(),
            }

            # Save SR image
            im_sr = util_image.tensor2img(sr_image[0], rgb2bgr=True, min_max=(0.0, 1.0))
            util_image.imwrite(im_sr, out_path / f"{im_path.stem}_sr.png", chn='bgr', dtype_in='uint8')

        # Save keys
        keys_path = out_path / "wm_keys.pkl"
        with open(keys_path, 'wb') as f:
            pickle.dump(all_keys, f)
        print(f"Keys saved to {keys_path}")
        if wm_enabled:
            print(f"Watermark capacity: {wm_codec.mark_length} bits")
        if configs.get('lr_storage', {}).get('enabled', False):
            print("LR storage channel: enabled")

    elif args.mode == "extract":
        assert args.wm_keys is not None, "Must provide --wm_keys for extraction"
        with open(args.wm_keys, 'rb') as f:
            all_keys = pickle.load(f)

        if in_path.is_dir():
            im_paths = sorted(in_path.glob("*.[jpJP][pnPN]*[gG]"))
        else:
            im_paths = [in_path]

        for im_path in im_paths:
            stem = im_path.stem.replace("_sr", "")
            if stem not in all_keys:
                print(f"Warning: no keys found for {im_path.name}, skipping")
                continue

            print(f"Extracting from: {im_path.name}")
            im_sr = util_image.imread(str(im_path), chn='rgb', dtype='float32')
            im_sr_tensor = util_image.img2tensor(im_sr).cuda()

            info = all_keys[stem]
            keys_info = info.get('keys_info', info.get('keys', {}))
            dual_result = sampler.extract_dual_channel(im_sr_tensor, keys_info, sf=args.scale)
            decoded_wm = dual_result['decoded_watermark']

            original_wm = info['watermark'].to(decoded_wm.device)
            # Use matching codec for accuracy
            latent_sz = keys_info.get('latent_size', wm_codec.latent_size) if isinstance(keys_info, dict) else wm_codec.latent_size
            if latent_sz != wm_codec.latent_size:
                codec_local = WatermarkCodec(
                    num_channels=wm_codec.num_channels,
                    latent_size=latent_sz,
                    ch_factor=wm_codec.ch_factor,
                    hw_factor=wm_codec.hw_factor,
                    use_chacha=wm_codec.use_chacha,
                )
            else:
                codec_local = wm_codec
            acc, mean_acc = codec_local.compute_bit_accuracy(original_wm, decoded_wm)
            print(f"  Bit accuracy: {mean_acc:.4f} ({int(mean_acc * codec_local.mark_length)}/{codec_local.mark_length})")
            print(f"  Exact LR used: {dual_result['used_exact_lr']} | CRC: {dual_result['lr_crc_ok']}")

            lr_recovered = dual_result.get('lr_recovered', None)
            if lr_recovered is not None:
                lr_img = util_image.tensor2img(lr_recovered[0], rgb2bgr=True, min_max=(0.0, 1.0))
                lr_path = out_path / f"{stem}_lr_recovered.png"
                util_image.imwrite(lr_img, lr_path, chn='bgr', dtype_in='uint8')
                print(f"  Recovered LR saved to {lr_path}")

            # Detection
            if wm_codec.tau_onebit and mean_acc >= wm_codec.tau_onebit:
                print(f"  Detection: POSITIVE (acc={mean_acc:.4f} >= tau={wm_codec.tau_onebit:.4f})")
            else:
                print(f"  Detection: NEGATIVE (acc={mean_acc:.4f} < tau={wm_codec.tau_onebit})")


if __name__ == "__main__":
    main()
