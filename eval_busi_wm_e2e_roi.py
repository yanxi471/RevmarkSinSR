#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BUSI：超分 + 水印 E2E，**ROI** 控制：宏块载荷 ROI 外置 0；潜空间 **ROI 内 eps_wm、非 ROI 标准高斯混合**，
并保存密文辅助提取（与 Gaussian-Shading / ChaCha 全局解密一致）。

输出与 eval_busi_wm_e2e.py 类似，另含 bit_accuracy_roi（仅统计 ROI 内比特）、roi_bit_fraction。
新增 ROI 像素级合成控制：先分别生成有/无水印 SR，再按 ROI 掩码合成
（ROI 用无水印 SR，非 ROI 用有水印 SR），并在合成图上执行水印提取与评估。

用法：
  python eval_busi_wm_e2e_roi.py --use_gt_roi --out_dir result/busi_wm_e2e_roi_gt
  python eval_busi_wm_e2e_roi.py --seg_ckpt weights/busi_unet.pth --out_dir result/busi_wm_e2e_roi
  # 无权重（随机 U-Net，仅通路/格式联调，分割无医学意义）：
  python eval_busi_wm_e2e_roi.py --out_dir result/busi_wm_e2e_roi_smoke --num_images 5
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from eval_busi_wm_e2e import (
    bicubic_degrade,
    busi_class_from_name,
    center_crop_square,
    crop_to_multiple_of,
    load_baseline_csv,
    read_image_list,
    setup_cuda,
    try_lpips,
)
from inference_wm import WatermarkSampler
from utils import util_image
from utils.busi_gt_mask import (
    align_mask_to_cropped_gt,
    load_merged_gt_mask_paths,
    parse_busi_processed_name,
    prepare_gt_roi_wm_mask,
)
from utils.busiqa import score_sr_rgb_nr, try_pyiqa
from utils.roi_segmenter import create_smp_unet_segmenter
from watermark_codec import WatermarkCodec


def _write_analysis_roi(summary: dict, baseline_csv: Path, tau: float) -> str:
    lines = [
        "BUSI 测试集 — 超分 + 水印 E2E（ROI 分割约束载荷网格）",
        "=" * 52,
        "",
        "一、实验设置",
        f"  权重: {summary.get('ckpt', '')}",
        f"  配置: {summary.get('cfg', '')}",
        f"  ROI 来源: {summary.get('roi_source', '')}",
        f"  分割权重: {summary.get('seg_ckpt', '(无)')}",
        f"  ROI 阈值: {summary.get('roi_threshold', '')}",
        f"  载荷容量: {summary.get('mark_length_bits', '')} bit；tau_onebit = {tau:.4f}",
        f"  有效样本数: {summary['num_images']}（文件缺失等跳过 {summary.get('num_skipped', 0)} 张；"
        f"ROI 为空回退全图水印 {summary.get('num_fallback_fullwm', 0)} 张）",
        f"  基线 CSV: {baseline_csv if baseline_csv.exists() else '(未找到)'}",
        "",
        "  说明：BUSI「normal」类 GT 掩码常为全黑（无病灶），下采样到宏块网格后 ROI 为空。",
        "  本脚本不再跳过此类样本，而是自动回退为全图水印模式以保证全链路覆盖。",
        "",
        "  嵌入策略：先生成含水印 SR 与无水印 SR，再按 ROI 掩码像素级合成：",
        "  ROI 区域回填无水印 SR，非 ROI 保留含水印 SR；并在合成图上完成提取评估。",
        "",
        "二、水印（ROI 内比特准确率）",
        f"  bit_accuracy_roi 均值: {summary.get('bit_accuracy_roi_mean', 0):.4f} ± {summary.get('bit_accuracy_roi_std', 0):.4f}",
        f"  全网格 bit_accuracy 均值（参考）: {summary.get('bit_accuracy_full_mean', 0):.4f}",
        f"  ROI 比特占比（相对总比特）均值: {summary.get('roi_bit_fraction_mean', 0):.4f}",
        f"  ge_tau（按 ROI 准确率）: {summary.get('bit_accuracy_roi_ge_tau_rate', 0)*100:.1f}%",
        "",
        "三、超分（含水印 SR）",
        f"  PSNR 均值: {summary['psnr_wm_mean']:.2f} ± {summary['psnr_wm_std']:.2f} dB",
        f"  SSIM 均值: {summary['ssim_wm_mean']:.4f} ± {summary['ssim_wm_std']:.4f}",
        "",
        "四、按类别（ROI 比特准确率）",
    ]
    for cls, st in summary.get("per_class", {}).items():
        lines.append(
            f"  {cls}: n={st['n']}, PSNR_wm={st['psnr_wm_mean']:.2f}, BitAcc_roi={st['bit_accuracy_roi_mean']:.4f}"
        )
    lines.append("")
    return "\n".join(lines)


def masked_psnr_u8(gt_u8: np.ndarray, sr_u8: np.ndarray, mask_hw: np.ndarray) -> float | None:
    """
    在 mask==True 的像素上计算 PSNR（按 RGB 三通道联合 MSE）。
    gt_u8/sr_u8: H×W×3 uint8
    mask_hw: H×W bool/0-1
    """
    m = mask_hw.astype(bool)
    if m.ndim != 2 or gt_u8.shape[:2] != m.shape:
        raise ValueError("mask_hw 尺寸需与图像 H×W 一致")
    n = int(m.sum())
    if n == 0:
        return None
    gt = gt_u8.astype(np.float64)
    sr = sr_u8.astype(np.float64)
    diff = (gt - sr)  # HWC
    diff2 = diff * diff
    mse = float(diff2[m].mean())  # 选中像素，三通道一起平均
    if mse <= 0:
        return 99.0
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def masked_ssim_u8(gt_u8: np.ndarray, sr_u8: np.ndarray, mask_hw: np.ndarray) -> float | None:
    """
    在 ROI 掩码区域近似计算 SSIM：
    先取 ROI 最小外接框，再将框内非 ROI 像素在两图中同时置零，最后计算 SSIM。
    """
    m = mask_hw.astype(bool)
    if m.ndim != 2 or gt_u8.shape[:2] != m.shape:
        raise ValueError("mask_hw 尺寸需与图像 H×W 一致")
    if int(m.sum()) == 0:
        return None
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    gt_c = gt_u8[y0:y1, x0:x1].copy()
    sr_c = sr_u8[y0:y1, x0:x1].copy()
    m_c = m[y0:y1, x0:x1]
    gt_c[~m_c] = 0
    sr_c[~m_c] = 0
    return float(ssim_fn(gt_c, sr_c, data_range=255, channel_axis=2))


def ensure_min_size_square(im: np.ndarray, min_size: int) -> tuple[np.ndarray, bool]:
    """
    若图像边长小于 min_size，则先双三次放大到 min_size（正方形输入下等比放大）。
    返回 (image, was_resized)。
    """
    h, w = im.shape[:2]
    if h >= min_size and w >= min_size:
        return im, False
    scale = float(min_size) / float(min(h, w))
    out = np.clip(util_image.imresize_np(im, scale=scale), 0.0, 1.0)
    return out, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="weights/SinSR_v1.pth")
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="configs/SinSR_wm_busi_smoke.yaml",
    )
    parser.add_argument("--image_list", type=str, default="traindata/busi_test.txt")
    parser.add_argument("--sf", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="result/busi_wm_e2e_roi")
    parser.add_argument("--baseline_csv", type=str, default="result/busi_sr_baseline_SinSR_v1/metrics/sr_metrics.csv")
    parser.add_argument("--num_images", type=int, default=0)
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--no_iqa", action="store_true")
    parser.add_argument("--musiq_ckpt", type=str, default="")
    parser.add_argument(
        "--seg_ckpt",
        type=str,
        default="",
        help="SMP U-Net .pth；留空则随机初始化编码器（仅联调）",
    )
    parser.add_argument("--roi_threshold", type=float, default=0.5)
    parser.add_argument(
        "--imagenet_encoder",
        action="store_true",
        help="分割骨干加载 ImageNet 权重（需联网）",
    )
    parser.add_argument(
        "--use_gt_roi",
        action="store_true",
        help="使用 data/Dataset_BUSI_with_GT 真值掩码（与 --seg_ckpt 互斥）",
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        default="data/Dataset_BUSI_with_GT",
        help="BUSI GT 掩码根目录（相对项目根）",
    )
    parser.add_argument(
        "--gt_grid_thr",
        type=float,
        default=0.5,
        help="水印宏块网格上 ROI 二值阈值（对 area 下采样后的值）",
    )
    parser.add_argument(
        "--wm_in_nonroi",
        action="store_true",
        help="启用后：ROI 用高斯噪声，非 ROI 用含水印噪声（默认相反）。",
    )
    parser.add_argument(
        "--pixel_roi_fusion",
        action="store_true",
        help="启用像素级 ROI 合成：ROI 回填无水印 SR，非 ROI 保留有水印 SR。",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    setup_cuda()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path = project_root / args.cfg_path
    ckpt_path = project_root / args.ckpt
    configs = OmegaConf.load(str(cfg_path))
    configs.model.ckpt_path = str(ckpt_path)
    wm_cfg = configs.get("watermark", {})
    if not wm_cfg.get("enabled", True):
        raise ValueError("需要 watermark.enabled=true")

    seg_ckpt = args.seg_ckpt.strip()
    if args.use_gt_roi and seg_ckpt:
        raise ValueError("不可同时使用 --use_gt_roi 与 --seg_ckpt")

    seg = None
    gt_root: Path | None = None
    if args.use_gt_roi:
        gt_root = (project_root / args.gt_root.strip()).resolve()
        if not gt_root.is_dir():
            raise FileNotFoundError(f"GT 根目录不存在: {gt_root}")
    else:
        # 无 seg_ckpt 时 U-Net 随机初始化；固定种子使换超分权重时 ROI 掩码一致、便于对比
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        seg = create_smp_unet_segmenter(
            ckpt_path=seg_ckpt if seg_ckpt else None,
            encoder_weights="imagenet" if args.imagenet_encoder else None,
        )

    codec = WatermarkCodec(
        num_channels=configs.autoencoder.params.ddconfig.z_channels,
        latent_size=configs.model.params.image_size,
        ch_factor=wm_cfg.get("ch_factor", 1),
        hw_factor=wm_cfg.get("hw_factor", 8),
        use_chacha=wm_cfg.get("use_chacha", True),
    )
    sampler = WatermarkSampler(configs, codec, lr_recovery_net=None, sf=args.sf, seed=42)
    lpips_model = None if args.no_lpips else (try_lpips() if device == "cuda" else None)
    musiq_path = args.musiq_ckpt.strip() or None
    clip_m, mus_m = (
        (None, None)
        if args.no_iqa
        else try_pyiqa(device, musiq_ckpt_path=musiq_path)
    )
    tau = float(codec.tau_onebit)
    gt_align = int(args.sf) * int(configs.model.params.image_size)

    list_path = project_root / args.image_list
    im_paths = read_image_list(list_path, project_root)
    if args.num_images > 0:
        im_paths = im_paths[: args.num_images]
    baseline_path = project_root / args.baseline_csv
    baseline_by_name = load_baseline_csv(baseline_path)

    out_dir = project_root / args.out_dir
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows_sr = []
    rows_wm = []
    bit_accs_full = []
    bit_accs_roi = []
    roi_fracs = []
    psnr_wm_list = []
    psnr_delta_list = []
    skipped = 0
    num_small_upscaled = 0
    fallback_full_wm = 0

    for idx, im_path in enumerate(im_paths):
        if not im_path.exists():
            print(f"[跳过] {im_path}")
            skipped += 1
            continue
        name = im_path.name
        t0 = time.time()
        im_gt = util_image.imread(str(im_path), chn="rgb", dtype="float32")
        hw0 = (int(im_gt.shape[0]), int(im_gt.shape[1]))
        im_gt = center_crop_square(im_gt)
        im_gt, was_upscaled = ensure_min_size_square(im_gt, gt_align)
        if was_upscaled:
            num_small_upscaled += 1
        im_gt = crop_to_multiple_of(im_gt, gt_align)
        im_lq = bicubic_degrade(im_gt, sf=args.sf)
        im_lq_tensor = util_image.img2tensor(im_lq).to(device)
        mask_hr_u8 = None

        with torch.no_grad():
            y0_n = (im_lq_tensor - 0.5) / 0.5
            z_ref_shape = sampler.z_y_from_lq_normalized(y0_n)
            lh = int(z_ref_shape.shape[2])
            lw = int(z_ref_shape.shape[3])
        if lh != lw:
            raise ValueError(f"LQ 经 VAE 后 latent 非方形 {lh}x{lw}，无法对齐水印网格")
        hw_f = int(codec.hw_factor)
        if lh % hw_f != 0:
            raise ValueError(f"latent 边长 {lh} 不能被 hw_factor={hw_f} 整除")
        wm_h_act = lh // hw_f
        wm_w_act = wm_h_act

        roi_mode = "seg_or_full"
        roi_mask_eval = None
        with torch.no_grad():
            if args.use_gt_roi:
                assert gt_root is not None
                roi_grid = prepare_gt_roi_wm_mask(
                    name,
                    gt_root,
                    im_hw_before_crop=hw0,
                    im_lq_shape_hw=(int(im_lq.shape[0]), int(im_lq.shape[1])),
                    gt_align=gt_align,
                    wm_h=wm_h_act,
                    wm_w=wm_w_act,
                    grid_thr=float(args.gt_grid_thr),
                )
                # 同时准备像素域（裁切后 HR）掩码，用于 ROI 内/外画质指标
                cls_dir, stem = parse_busi_processed_name(name)
                raw = load_merged_gt_mask_paths(gt_root, cls_dir, stem)
                if raw is not None:
                    mask_hr_u8 = align_mask_to_cropped_gt(raw, hw0, gt_align)
                if roi_grid is None or float(roi_grid.max()) < 0.5:
                    # 不再跳过：GT 在宏块网格为空时回退为全图水印，确保全链路覆盖所有图像
                    roi_mode = "fallback_fullwm"
                    fallback_full_wm += 1
                    sr_wm, watermark, keys_info = sampler.encode_sr(
                        im_lq_tensor,
                        roi_wm_in_roi=not args.wm_in_nonroi,
                    )
                    roi_mask_eval = np.ones((wm_h_act, wm_w_act), dtype=np.float32)
                else:
                    roi_mode = "gt_roi"
                    sr_wm, watermark, keys_info = sampler.encode_sr(
                        im_lq_tensor,
                        roi_wm_mask=roi_grid,
                        roi_threshold=float(args.gt_grid_thr),
                        roi_wm_in_roi=not args.wm_in_nonroi,
                    )
                    roi_mask_eval = roi_grid
            else:
                sr_wm, watermark, keys_info = sampler.encode_sr(
                    im_lq_tensor,
                    roi_segmenter=seg,
                    roi_threshold=float(args.roi_threshold),
                    roi_wm_in_roi=not args.wm_in_nonroi,
                )
                roi_mode = "seg_roi"

            # ROI 像素级保护：ROI 区域替换为无水印 SR，非 ROI 保留有水印 SR
            # 若没有像素级 ROI 掩码，则回退为直接评估有水印 SR。
            sr_eval = sr_wm
            if args.pixel_roi_fusion and mask_hr_u8 is not None:
                roi_bool = mask_hr_u8 > 127
                if bool(roi_bool.any()):
                    y0_norm = (im_lq_tensor - 0.5) / 0.5
                    z_ref_nowm = sampler.z_y_from_lq_normalized(y0_norm)
                    noise_nowm = torch.randn_like(z_ref_nowm)
                    sr_nowm_norm = sampler.sample_func_with_noise(y0_norm, noise_nowm, one_step=True)
                    sr_nowm = sr_nowm_norm * 0.5 + 0.5

                    sr_wm_u8 = util_image.tensor2img(sr_wm[0], rgb2bgr=False, min_max=(0.0, 1.0)).astype(np.uint8)
                    sr_nowm_u8 = util_image.tensor2img(sr_nowm[0], rgb2bgr=False, min_max=(0.0, 1.0)).astype(np.uint8)
                    sr_mix_u8 = sr_wm_u8.copy()
                    sr_mix_u8[roi_bool] = sr_nowm_u8[roi_bool]
                    sr_eval = util_image.img2tensor(sr_mix_u8.astype(np.float32) / 255.0).to(device)
                    roi_mode = "pixel_roi_fusion"

            decoded = sampler.extract_watermark(sr_eval, keys_info, sf=args.sf)

        latent_sz = keys_info.get("latent_size", codec.latent_size)
        if latent_sz != codec.latent_size:
            codec_eval = WatermarkCodec(
                num_channels=codec.num_channels,
                latent_size=int(latent_sz),
                ch_factor=codec.ch_factor,
                hw_factor=codec.hw_factor,
                use_chacha=codec.use_chacha,
            )
        else:
            codec_eval = codec

        _, bit_acc_full = codec_eval.compute_bit_accuracy(watermark, decoded)
        roi_mask = keys_info.get("roi_wm_mask")
        if roi_mask is None:
            if roi_mask_eval is None:
                roi_mask_eval = np.ones((wm_h_act, wm_w_act), dtype=np.float32)
            roi_mask = roi_mask_eval

        wm_ch = int(codec_eval.num_channels // codec_eval.ch_factor)
        wm_h = int(latent_sz // codec_eval.hw_factor)
        wm_w = wm_h
        roi_t = torch.as_tensor(roi_mask, dtype=torch.float32, device=watermark.device)
        m_exp = roi_t.view(1, 1, wm_h, wm_w).expand(1, wm_ch, wm_h, wm_w)
        roi_bit_fraction = float((m_exp > 0.5).float().mean().item())

        _, bit_acc_roi = codec_eval.compute_bit_accuracy_masked(
            watermark, decoded, roi_mask
        )
        elapsed = time.time() - t0

        sr_rgb = util_image.tensor2img(sr_eval[0], rgb2bgr=False, min_max=(0.0, 1.0))
        gt_u8 = (im_gt * 255.0).astype(np.uint8)
        sr_u8 = sr_rgb.astype(np.uint8)
        psnr_wm = psnr_fn(gt_u8, sr_rgb, data_range=255)
        ssim_wm = float(ssim_fn(gt_u8, sr_rgb, data_range=255, channel_axis=2))
        psnr_roi = psnr_bg = None
        ssim_roi = ssim_bg = None
        roi_pix_fraction = ""
        if args.use_gt_roi and mask_hr_u8 is not None:
            roi_bool = mask_hr_u8 > 127
            roi_pix_fraction = float(roi_bool.mean())
            psnr_roi = masked_psnr_u8(gt_u8, sr_u8, roi_bool)
            psnr_bg = masked_psnr_u8(gt_u8, sr_u8, ~roi_bool)
            ssim_roi = masked_ssim_u8(gt_u8, sr_u8, roi_bool)
            ssim_bg = masked_ssim_u8(gt_u8, sr_u8, ~roi_bool)
        lp = None
        if lpips_model is not None:
            sr_t = util_image.img2tensor(sr_rgb.astype(np.float32) / 255.0).to(device)
            gt_t = util_image.img2tensor(im_gt).to(device)
            with torch.no_grad():
                lp = lpips_model(gt_t * 2 - 1, sr_t * 2 - 1).item()

        clipiqa_v = musiq_v = None
        if clip_m is not None or mus_m is not None:
            clipiqa_v, musiq_v = score_sr_rgb_nr(clip_m, mus_m, sr_rgb)

        cls = busi_class_from_name(name)
        bit_accs_full.append(float(bit_acc_full))
        bit_accs_roi.append(float(bit_acc_roi))
        roi_fracs.append(roi_bit_fraction)
        psnr_wm_list.append(float(psnr_wm))

        base = baseline_by_name.get(name, {})
        psnr_b = base.get("psnr", "")
        ssim_b = base.get("ssim", "")
        lp_b = base.get("lpips", "")
        mus_b = base.get("musiq", "")
        cli_b = base.get("clipiqa", "")
        d_psnr = d_ssim = d_lp = d_mus = d_cli = ""
        if psnr_b != "":
            d_psnr = round(float(psnr_wm) - float(psnr_b), 4)
            psnr_delta_list.append(float(d_psnr))
        if ssim_b != "":
            d_ssim = round(ssim_wm - float(ssim_b), 6)
        if lp_b != "" and lp is not None:
            d_lp = round(float(lp) - float(lp_b), 6)
        if mus_b != "" and musiq_v is not None:
            d_mus = round(float(musiq_v) - float(mus_b), 4)
        if cli_b != "" and clipiqa_v is not None:
            d_cli = round(float(clipiqa_v) - float(cli_b), 6)

        rows_sr.append(
            {
                "image": name,
                "class": cls,
                "psnr_wm": round(psnr_wm, 4),
                "psnr_roi": "" if psnr_roi is None else round(float(psnr_roi), 4),
                "psnr_bg": "" if psnr_bg is None else round(float(psnr_bg), 4),
                "ssim_roi": "" if ssim_roi is None else round(float(ssim_roi), 6),
                "ssim_bg": "" if ssim_bg is None else round(float(ssim_bg), 6),
                "roi_pix_fraction": "" if roi_pix_fraction == "" else round(float(roi_pix_fraction), 6),
                "ssim_wm": round(ssim_wm, 6),
                "lpips_wm": "" if lp is None else round(lp, 6),
                "musiq_wm": "" if musiq_v is None else round(musiq_v, 4),
                "clipiqa_wm": "" if clipiqa_v is None else round(clipiqa_v, 6),
                "psnr_baseline": psnr_b,
                "ssim_baseline": ssim_b,
                "lpips_baseline": lp_b,
                "musiq_baseline": mus_b,
                "clipiqa_baseline": cli_b,
                "delta_psnr_wm_minus_baseline": d_psnr,
                "delta_ssim_wm_minus_baseline": d_ssim,
                "delta_lpips_wm_minus_baseline": d_lp,
                "delta_musiq_wm_minus_baseline": d_mus,
                "delta_clipiqa_wm_minus_baseline": d_cli,
                "time_encode_extract_s": round(elapsed, 4),
                "roi_bit_fraction": round(roi_bit_fraction, 6),
                "roi_mode": roi_mode,
            }
        )
        rows_wm.append(
            {
                "image": name,
                "class": cls,
                "bit_accuracy_full": round(float(bit_acc_full), 6),
                "bit_accuracy_roi": round(float(bit_acc_roi), 6),
                "roi_bit_fraction": round(roi_bit_fraction, 6),
                "ge_tau_roi": bool(bit_acc_roi >= tau),
                "tau_onebit": tau,
                "roi_mode": roi_mode,
            }
        )
        iq_extra = ""
        if musiq_v is not None:
            iq_extra += f"  MUSIQ_wm={musiq_v:.2f}"
        if clipiqa_v is not None:
            iq_extra += f"  CLIPIQA_wm={clipiqa_v:.4f}"
        print(
            f"[{len(rows_sr)}/{len(im_paths)}] {name}  PSNR_wm={psnr_wm:.2f}  "
            f"BitAcc_roi={bit_acc_roi:.4f}  BitAcc_full={bit_acc_full:.4f}  "
            f"roi_frac={roi_bit_fraction:.3f}{iq_extra}"
        )

    if not rows_sr:
        raise RuntimeError("无有效样本")

    af = np.array(bit_accs_full, dtype=np.float64)
    ar = np.array(bit_accs_roi, dtype=np.float64)
    rf = np.array(roi_fracs, dtype=np.float64)
    psnr_arr = np.array(psnr_wm_list, dtype=np.float64)
    psnr_roi_list = [float(r["psnr_roi"]) for r in rows_sr if r.get("psnr_roi", "") != ""]
    psnr_bg_list = [float(r["psnr_bg"]) for r in rows_sr if r.get("psnr_bg", "") != ""]
    ssim_roi_list = [float(r["ssim_roi"]) for r in rows_sr if r.get("ssim_roi", "") != ""]
    ssim_bg_list = [float(r["ssim_bg"]) for r in rows_sr if r.get("ssim_bg", "") != ""]

    roi_source = (
        f"GT_masks:{args.gt_root}"
        if args.use_gt_roi
        else ("smp_unet:" + (seg_ckpt if seg_ckpt else "random_init"))
    )
    summary = {
        "ckpt": ckpt_path.name,
        "cfg": cfg_path.name,
        "roi_source": roi_source,
        "seg_ckpt": None if args.use_gt_roi else (seg_ckpt if seg_ckpt else None),
        "roi_threshold": float(args.gt_grid_thr if args.use_gt_roi else args.roi_threshold),
        "mark_length_bits": int(codec.mark_length),
        "tau_onebit": tau,
        "num_images": len(rows_sr),
        "num_skipped": int(skipped),
        "num_fallback_fullwm": int(fallback_full_wm),
        "num_small_upscaled": int(num_small_upscaled),
        "psnr_wm_mean": float(psnr_arr.mean()),
        "psnr_wm_std": float(psnr_arr.std()),
        "psnr_roi_mean": "" if not psnr_roi_list else float(np.mean(psnr_roi_list)),
        "psnr_bg_mean": "" if not psnr_bg_list else float(np.mean(psnr_bg_list)),
        "ssim_roi_mean": "" if not ssim_roi_list else float(np.mean(ssim_roi_list)),
        "ssim_bg_mean": "" if not ssim_bg_list else float(np.mean(ssim_bg_list)),
        "bit_accuracy_full_mean": float(af.mean()),
        "bit_accuracy_full_std": float(af.std()),
        "bit_accuracy_roi_mean": float(ar.mean()),
        "bit_accuracy_roi_std": float(ar.std()),
        "bit_accuracy_roi_min": float(ar.min()),
        "bit_accuracy_roi_ge_tau_rate": float(np.mean(ar >= tau)),
        "roi_bit_fraction_mean": float(rf.mean()),
        "roi_bit_fraction_std": float(rf.std()),
        "time_mean_s": float(np.mean([float(r["time_encode_extract_s"]) for r in rows_sr])),
    }
    ssims = np.array([float(r["ssim_wm"]) for r in rows_sr], dtype=np.float64)
    summary["ssim_wm_mean"] = float(ssims.mean())
    summary["ssim_wm_std"] = float(ssims.std())
    if psnr_delta_list:
        da = np.array(psnr_delta_list, dtype=np.float64)
        summary["delta_psnr_mean_wm_minus_baseline"] = float(da.mean())
        summary["delta_psnr_std"] = float(da.std())
    lp_wm = [float(r["lpips_wm"]) for r in rows_sr if r["lpips_wm"] != ""]
    if lp_wm:
        summary["lpips_wm_mean"] = float(np.mean(lp_wm))
    mus_wm = [float(r["musiq_wm"]) for r in rows_sr if r.get("musiq_wm", "") != ""]
    cli_wm = [float(r["clipiqa_wm"]) for r in rows_sr if r.get("clipiqa_wm", "") != ""]
    if mus_wm:
        summary["musiq_wm_mean"] = float(np.mean(mus_wm))
        summary["musiq_wm_std"] = float(np.std(mus_wm))
    if cli_wm:
        summary["clipiqa_wm_mean"] = float(np.mean(cli_wm))
        summary["clipiqa_wm_std"] = float(np.std(cli_wm))

    by_class = defaultdict(lambda: {"psnr": [], "bit_roi": []})
    for r, w in zip(rows_sr, rows_wm):
        c = r["class"]
        by_class[c]["psnr"].append(float(r["psnr_wm"]))
        by_class[c]["bit_roi"].append(float(w["bit_accuracy_roi"]))
    summary["per_class"] = {
        k: {
            "n": len(v["psnr"]),
            "psnr_wm_mean": float(np.mean(v["psnr"])),
            "bit_accuracy_roi_mean": float(np.mean(v["bit_roi"])),
        }
        for k, v in by_class.items()
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

    analysis = _write_analysis_roi(summary, baseline_path, tau)
    with open(out_dir / "ANALYSIS.txt", "w", encoding="utf-8") as f:
        f.write(analysis)

    with open(out_dir / "CHANGELOG.txt", "w", encoding="utf-8") as f:
        roi_line = (
            f"- ROI: Dataset_BUSI_with_GT ({args.gt_root})\n"
            if args.use_gt_roi
            else f"- 分割: {seg_ckpt or '（无 ckpt，随机 U-Net）'}\n"
        )
        f.write(
            "BUSI E2E + ROI 分割（载荷网格掩码）\n"
            f"- 配置: {args.cfg_path}\n"
            f"{roi_line}"
            f"- 列表: {args.image_list}\n"
            f"- 跳过样本数: {skipped}\n"
        )

    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n已写入: {out_dir}")


if __name__ == "__main__":
    main()
