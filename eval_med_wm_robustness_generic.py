#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from eval_busi_wm_e2e import (
    bicubic_degrade,
    center_crop_square,
    crop_to_multiple_of,
    read_image_list,
    setup_cuda,
)
from inference_wm import WatermarkSampler
from utils import util_image
from watermark_codec import WatermarkCodec


def ensure_min_size_square(im: np.ndarray, min_size: int) -> tuple[np.ndarray, bool]:
    h, w = im.shape[:2]
    if h >= min_size and w >= min_size:
        return im, False
    scale = float(min_size) / float(min(h, w))
    out = np.clip(util_image.imresize_np(im, scale=scale), 0.0, 1.0)
    return out, True


def _jpeg_attack(img_u8: np.ndarray, q: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        return img_u8
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def _gaussian_noise(img_u8: np.ndarray, sigma: float) -> np.ndarray:
    n = np.random.normal(0.0, sigma, img_u8.shape).astype(np.float32)
    return np.clip(img_u8.astype(np.float32) + n, 0, 255).astype(np.uint8)


def _gaussian_blur(img_u8: np.ndarray, k: int, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img_u8, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)


def _rescale_back(img_u8: np.ndarray, ratio: float) -> np.ndarray:
    h, w = img_u8.shape[:2]
    h2 = max(8, int(round(h * ratio)))
    w2 = max(8, int(round(w * ratio)))
    small = cv2.resize(img_u8, (w2, h2), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _center_crop_back(img_u8: np.ndarray, ratio: float) -> np.ndarray:
    h, w = img_u8.shape[:2]
    h2 = max(8, int(round(h * ratio)))
    w2 = max(8, int(round(w * ratio)))
    top = (h - h2) // 2
    left = (w - w2) // 2
    crop = img_u8[top : top + h2, left : left + w2]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)


def _brightness_contrast(img_u8: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    out = img_u8.astype(np.float32) * float(alpha) + float(beta)
    return np.clip(out, 0, 255).astype(np.uint8)


def get_attacks():
    return [
        ("clean", lambda x: x),
        ("jpeg_q95", lambda x: _jpeg_attack(x, 95)),
        ("jpeg_q85", lambda x: _jpeg_attack(x, 85)),
        ("jpeg_q75", lambda x: _jpeg_attack(x, 75)),
        ("jpeg_q60", lambda x: _jpeg_attack(x, 60)),
        ("noise_s2", lambda x: _gaussian_noise(x, 2)),
        ("noise_s5", lambda x: _gaussian_noise(x, 5)),
        ("noise_s10", lambda x: _gaussian_noise(x, 10)),
        ("blur_k3_s08", lambda x: _gaussian_blur(x, 3, 0.8)),
        ("blur_k5_s12", lambda x: _gaussian_blur(x, 5, 1.2)),
        ("rescale_09", lambda x: _rescale_back(x, 0.9)),
        ("rescale_08", lambda x: _rescale_back(x, 0.8)),
        ("crop_95", lambda x: _center_crop_back(x, 0.95)),
        ("crop_90", lambda x: _center_crop_back(x, 0.9)),
        ("bc_b+10", lambda x: _brightness_contrast(x, 1.0, +10.0)),
        ("bc_b-10", lambda x: _brightness_contrast(x, 1.0, -10.0)),
        ("bc_c09", lambda x: _brightness_contrast(x, 0.9, 0.0)),
        ("bc_c11", lambda x: _brightness_contrast(x, 1.1, 0.0)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="weights/SinSR_v1.pth")
    ap.add_argument("--cfg_path", type=str, default="configs/SinSR_wm_busi_smoke.yaml")
    ap.add_argument("--image_list", type=str, required=True)
    ap.add_argument("--mask_root", type=str, default="")
    ap.add_argument("--sf", type=int, default=4)
    ap.add_argument("--roi_threshold", type=float, default=0.5)
    ap.add_argument("--pixel_roi_fusion", action="store_true")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    setup_cuda()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.load(str(project_root / args.cfg_path))
    cfg.model.ckpt_path = str(project_root / args.ckpt)
    wm_cfg = cfg.get("watermark", {})
    if not wm_cfg.get("enabled", True):
        raise ValueError("需要 watermark.enabled=true")

    codec = WatermarkCodec(
        num_channels=cfg.autoencoder.params.ddconfig.z_channels,
        latent_size=cfg.model.params.image_size,
        ch_factor=wm_cfg.get("ch_factor", 1),
        hw_factor=wm_cfg.get("hw_factor", 8),
        use_chacha=wm_cfg.get("use_chacha", True),
    )
    tau = float(codec.tau_onebit)
    sampler = WatermarkSampler(cfg, codec, lr_recovery_net=None, sf=args.sf, seed=42)
    gt_align = int(args.sf) * int(cfg.model.params.image_size)

    image_list = read_image_list(project_root / args.image_list, project_root)
    mask_root = (project_root / args.mask_root).resolve() if args.mask_root.strip() else None
    attacks = get_attacks()

    rows = []
    stat = {k: [] for k, _ in attacks}
    fallback_count = 0
    for idx, p in enumerate(image_list, 1):
        if not p.exists():
            continue
        name = p.name
        im_gt = util_image.imread(str(p), chn="rgb", dtype="float32")
        im_gt = center_crop_square(im_gt)
        mask_hr = None
        if mask_root is not None:
            mask_path = mask_root / name
            if mask_path.exists():
                mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_raw is not None:
                    mask_f = (mask_raw.astype(np.float32) / 255.0)
                    mask_f = center_crop_square(mask_f[..., None])[..., 0]
                else:
                    mask_f = None
            else:
                mask_f = None
        else:
            mask_f = None

        im_gt, _ = ensure_min_size_square(im_gt, gt_align)
        im_gt = crop_to_multiple_of(im_gt, gt_align)
        if mask_f is not None:
            mask_f, _ = ensure_min_size_square(mask_f[..., None], gt_align)
            mask_f = mask_f[..., 0]
            mask_f = crop_to_multiple_of(mask_f[..., None], gt_align)[..., 0]
            mask_hr = mask_f >= float(args.roi_threshold)

        im_lq = bicubic_degrade(im_gt, sf=args.sf)
        lq_t = util_image.img2tensor(im_lq).to(device)

        encode_kwargs = {}
        roi_eval = None
        with torch.no_grad():
            z_ref = sampler.z_y_from_lq_normalized((lq_t - 0.5) / 0.5)
        latent_sz = int(z_ref.shape[2])
        wm_h = latent_sz // int(codec.hw_factor)
        if mask_f is not None:
            roi_grid = cv2.resize(mask_f, (wm_h, wm_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            roi_grid = (roi_grid >= float(args.roi_threshold)).astype(np.float32)
            if float(roi_grid.max()) >= 0.5:
                encode_kwargs = {"roi_wm_mask": roi_grid, "roi_threshold": float(args.roi_threshold)}
                roi_eval = roi_grid
            else:
                fallback_count += 1
                roi_eval = np.ones((wm_h, wm_h), dtype=np.float32)

        with torch.no_grad():
            sr_wm, watermark, keys_info = sampler.encode_sr(lq_t, **encode_kwargs)
            sr_eval = sr_wm
            if args.pixel_roi_fusion and mask_hr is not None and bool(mask_hr.any()):
                y0_norm = (lq_t - 0.5) / 0.5
                z_ref_nowm = sampler.z_y_from_lq_normalized(y0_norm)
                noise_nowm = torch.randn_like(z_ref_nowm)
                sr_nowm_norm = sampler.sample_func_with_noise(y0_norm, noise_nowm, one_step=True)
                sr_nowm = sr_nowm_norm * 0.5 + 0.5

                sr_wm_u8 = util_image.tensor2img(sr_wm[0], rgb2bgr=False, min_max=(0.0, 1.0)).astype(np.uint8)
                sr_nowm_u8 = util_image.tensor2img(sr_nowm[0], rgb2bgr=False, min_max=(0.0, 1.0)).astype(np.uint8)
                sr_mix_u8 = sr_wm_u8.copy()
                sr_mix_u8[mask_hr] = sr_nowm_u8[mask_hr]
                sr_eval = util_image.img2tensor(sr_mix_u8.astype(np.float32) / 255.0).to(device)

        latent_used = keys_info.get("latent_size", codec.latent_size)
        codec_eval = codec if latent_used == codec.latent_size else WatermarkCodec(
            num_channels=codec.num_channels,
            latent_size=int(latent_used),
            ch_factor=codec.ch_factor,
            hw_factor=codec.hw_factor,
            use_chacha=codec.use_chacha,
        )

        sr_u8 = util_image.tensor2img(sr_eval[0], rgb2bgr=False, min_max=(0.0, 1.0)).astype(np.uint8)
        for atk_name, fn in attacks:
            atk_u8 = fn(sr_u8)
            atk_t = util_image.img2tensor(atk_u8.astype(np.float32) / 255.0).to(device)
            with torch.no_grad():
                decoded = sampler.extract_watermark(atk_t, keys_info, sf=args.sf)
            if roi_eval is None:
                _, acc = codec_eval.compute_bit_accuracy(watermark, decoded)
            else:
                _, acc = codec_eval.compute_bit_accuracy_masked(watermark, decoded, roi_eval)
            stat[atk_name].append(float(acc))
            rows.append(
                {
                    "image": name,
                    "attack": atk_name,
                    "bit_accuracy": float(acc),
                    "ge_tau": bool(float(acc) >= tau),
                }
            )
        if idx % 5 == 0:
            print(f"[{idx}/{len(image_list)}] done")

    out_dir = project_root / args.out_dir
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "robustness_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary_rows = []
    for atk_name, vals in stat.items():
        arr = np.array(vals, dtype=np.float64)
        summary_rows.append(
            {
                "attack": atk_name,
                "num_images": int(arr.size),
                "bit_accuracy_mean": float(arr.mean()),
                "bit_accuracy_std": float(arr.std()),
                "bit_accuracy_ge_tau_rate": float(np.mean(arr >= tau)),
            }
        )
    with open(metrics_dir / "robustness_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    out_json = {
        "ckpt": Path(args.ckpt).name,
        "num_images": len(image_list),
        "tau_onebit": tau,
        "pixel_roi_fusion": bool(args.pixel_roi_fusion),
        "use_roi_mask": bool(mask_root is not None),
        "num_fallback_fullwm": int(fallback_count),
        "attacks": summary_rows,
    }
    with open(metrics_dir / "robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(json.dumps(out_json, indent=2, ensure_ascii=False))
    print(f"\n已写入: {out_dir}")


if __name__ == "__main__":
    main()

