#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

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


def masked_psnr_u8(gt_u8: np.ndarray, sr_u8: np.ndarray, mask_hw: np.ndarray) -> float | None:
    m = mask_hw.astype(bool)
    if m.ndim != 2 or gt_u8.shape[:2] != m.shape:
        raise ValueError("mask_hw 尺寸需与图像 H×W 一致")
    n = int(m.sum())
    if n == 0:
        return None
    gt = gt_u8.astype(np.float64)
    sr = sr_u8.astype(np.float64)
    mse = float(((gt - sr) ** 2)[m].mean())
    if mse <= 0:
        return 99.0
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def class_from_name(name: str) -> str:
    if "__" in name:
        return name.split("__", 1)[0]
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="weights/SinSR_v1.pth")
    ap.add_argument("--cfg_path", type=str, default="configs/SinSR_wm_busi_smoke.yaml")
    ap.add_argument("--image_list", type=str, required=True)
    ap.add_argument("--mask_root", type=str, required=True, help="mask目录，文件名需与image_list中的name一致")
    ap.add_argument("--sf", type=int, default=4)
    ap.add_argument("--roi_threshold", type=float, default=0.5)
    ap.add_argument(
        "--pixel_roi_fusion",
        action="store_true",
        help="像素域 ROI 合成：ROI 用无水印 SR，非 ROI 保留有水印 SR。",
    )
    ap.add_argument("--num_images", type=int, default=0)
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
    sampler = WatermarkSampler(cfg, codec, lr_recovery_net=None, sf=args.sf, seed=42)
    tau = float(codec.tau_onebit)
    gt_align = int(args.sf) * int(cfg.model.params.image_size)

    image_list = read_image_list(project_root / args.image_list, project_root)
    if args.num_images > 0:
        image_list = image_list[: args.num_images]
    mask_root = (project_root / args.mask_root).resolve()
    if not mask_root.is_dir():
        raise FileNotFoundError(f"mask_root 不存在: {mask_root}")

    out_dir = project_root / args.out_dir
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows_sr, rows_wm = [], []
    acc_full, acc_roi, psnrs, ssims = [], [], [], []
    skipped = 0
    fallback_fullwm = 0

    for p in image_list:
        if not p.exists():
            skipped += 1
            continue
        name = p.name
        mask_path = mask_root / name
        if not mask_path.exists():
            skipped += 1
            continue

        t0 = time.time()
        im_gt = util_image.imread(str(p), chn="rgb", dtype="float32")
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            skipped += 1
            continue

        im_gt = center_crop_square(im_gt)
        mask_f = (mask_raw.astype(np.float32) / 255.0)
        mask_f = center_crop_square(mask_f[..., None])[..., 0]

        im_gt, _ = ensure_min_size_square(im_gt, gt_align)
        mask_f, _ = ensure_min_size_square(mask_f[..., None], gt_align)
        mask_f = mask_f[..., 0]

        im_gt = crop_to_multiple_of(im_gt, gt_align)
        mask_f = crop_to_multiple_of(mask_f[..., None], gt_align)[..., 0]
        gt_u8 = (im_gt * 255.0).astype(np.uint8)
        mask_hr = mask_f >= float(args.roi_threshold)

        im_lq = bicubic_degrade(im_gt, sf=args.sf)
        lq_t = util_image.img2tensor(im_lq).to(device)

        with torch.no_grad():
            z_ref = sampler.z_y_from_lq_normalized((lq_t - 0.5) / 0.5)
        latent_sz = int(z_ref.shape[2])
        hw_f = int(codec.hw_factor)
        wm_h = latent_sz // hw_f
        wm_w = wm_h

        roi_grid = cv2.resize(mask_f, (wm_w, wm_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        roi_grid = (roi_grid >= float(args.roi_threshold)).astype(np.float32)

        with torch.no_grad():
            if float(roi_grid.max()) < 0.5:
                sr, watermark, keys = sampler.encode_sr(lq_t)
                fallback_fullwm += 1
                roi_mode = "fallback_fullwm"
                roi_eval = np.ones((wm_h, wm_w), dtype=np.float32)
            else:
                sr, watermark, keys = sampler.encode_sr(
                    lq_t, roi_wm_mask=roi_grid, roi_threshold=float(args.roi_threshold)
                )
                roi_mode = "roi_on"
                roi_eval = roi_grid

            sr_eval = sr
            if args.pixel_roi_fusion and bool(mask_hr.any()):
                y0_norm = (lq_t - 0.5) / 0.5
                z_ref_nowm = sampler.z_y_from_lq_normalized(y0_norm)
                noise_nowm = torch.randn_like(z_ref_nowm)
                sr_nowm_norm = sampler.sample_func_with_noise(y0_norm, noise_nowm, one_step=True)
                sr_nowm = sr_nowm_norm * 0.5 + 0.5

                sr_wm_u8 = util_image.tensor2img(sr[0], rgb2bgr=False, min_max=(0.0, 1.0)).astype(np.uint8)
                sr_nowm_u8 = util_image.tensor2img(sr_nowm[0], rgb2bgr=False, min_max=(0.0, 1.0)).astype(np.uint8)
                sr_mix_u8 = sr_wm_u8.copy()
                sr_mix_u8[mask_hr] = sr_nowm_u8[mask_hr]
                sr_eval = util_image.img2tensor(sr_mix_u8.astype(np.float32) / 255.0).to(device)
                roi_mode = "pixel_roi_fusion"

            decoded = sampler.extract_watermark(sr_eval, keys, sf=args.sf)

        latent_used = int(keys.get("latent_size", codec.latent_size))
        codec_eval = codec if latent_used == codec.latent_size else WatermarkCodec(
            num_channels=codec.num_channels,
            latent_size=latent_used,
            ch_factor=codec.ch_factor,
            hw_factor=codec.hw_factor,
            use_chacha=codec.use_chacha,
        )
        _, ba_full = codec_eval.compute_bit_accuracy(watermark, decoded)
        _, ba_roi = codec_eval.compute_bit_accuracy_masked(watermark, decoded, roi_eval)

        sr_u8 = util_image.tensor2img(sr_eval[0], rgb2bgr=False, min_max=(0.0, 1.0)).astype(np.uint8)
        psnr = float(psnr_fn(gt_u8, sr_u8, data_range=255))
        ssim = float(ssim_fn(gt_u8, sr_u8, data_range=255, channel_axis=2))
        psnr_roi = masked_psnr_u8(gt_u8, sr_u8, mask_hr)
        psnr_bg = masked_psnr_u8(gt_u8, sr_u8, ~mask_hr)

        cls = class_from_name(name)
        elapsed = time.time() - t0
        rows_sr.append(
            {
                "image": name,
                "class": cls,
                "psnr_wm": round(psnr, 4),
                "ssim_wm": round(ssim, 6),
                "psnr_roi": "" if psnr_roi is None else round(float(psnr_roi), 4),
                "psnr_bg": "" if psnr_bg is None else round(float(psnr_bg), 4),
                "roi_pix_fraction": round(float(mask_hr.mean()), 6),
                "time_encode_extract_s": round(elapsed, 4),
                "roi_mode": roi_mode,
            }
        )
        rows_wm.append(
            {
                "image": name,
                "class": cls,
                "bit_accuracy_full": round(float(ba_full), 6),
                "bit_accuracy_roi": round(float(ba_roi), 6),
                "ge_tau_roi": bool(float(ba_roi) >= tau),
                "tau_onebit": tau,
                "roi_mode": roi_mode,
            }
        )
        acc_full.append(float(ba_full))
        acc_roi.append(float(ba_roi))
        psnrs.append(psnr)
        ssims.append(ssim)
        print(f"[{len(rows_sr)}/{len(image_list)}] {name} PSNR={psnr:.2f} BitAccROI={ba_roi:.4f}")

    if not rows_sr:
        raise RuntimeError("无有效样本")

    summary = {
        "num_images": len(rows_sr),
        "num_skipped": int(skipped),
        "num_fallback_fullwm": int(fallback_fullwm),
        "pixel_roi_fusion": bool(args.pixel_roi_fusion),
        "psnr_wm_mean": float(np.mean(psnrs)),
        "psnr_wm_std": float(np.std(psnrs)),
        "ssim_wm_mean": float(np.mean(ssims)),
        "ssim_wm_std": float(np.std(ssims)),
        "bit_accuracy_full_mean": float(np.mean(acc_full)),
        "bit_accuracy_roi_mean": float(np.mean(acc_roi)),
        "bit_accuracy_roi_ge_tau_rate": float(np.mean(np.array(acc_roi) >= tau)),
        "time_mean_s": float(np.mean([float(r["time_encode_extract_s"]) for r in rows_sr])),
        "tau_onebit": tau,
    }

    with open(metrics_dir / "sr_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sr[0].keys()))
        w.writeheader()
        w.writerows(rows_sr)
    with open(metrics_dir / "wm_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_wm[0].keys()))
        w.writeheader()
        w.writerows(rows_wm)
    with open(metrics_dir / "e2e_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"written: {out_dir}")


if __name__ == "__main__":
    main()

