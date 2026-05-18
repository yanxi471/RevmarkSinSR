#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BUSI 测试集：超分 + 水印嵌入 + 提取（端到端）。
与 eval_busi_sr_baseline.py 相同预处理（中心正方形裁切 + GT 边长为 sf×image_size 倍数，
默认 4×64=256，与训练 latent 网格一致）、相同 bicubic 退化，便于与基线 CSV 对比超分指标变化。

输出：
  metrics/sr_metrics.csv      — 含水印 SR 的 PSNR/SSIM/LPIPS 及相对基线的差值
  metrics/wm_metrics.csv      — 比特准确率等
  metrics/e2e_summary.json
  ANALYSIS.txt                — 重点分析嵌入是否「破坏」超分与水印可恢复性

用法：
  python eval_busi_wm_e2e.py \\
    --baseline_csv result/busi_sr_baseline_SinSR_v1/metrics/sr_metrics.csv \\
    --out_dir result/busi_wm_e2e_SinSR_v1
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

from inference_wm import WatermarkSampler
from utils import util_image
from utils.busiqa import score_sr_rgb_nr, try_pyiqa
from watermark_codec import WatermarkCodec


def center_crop_square(im: np.ndarray) -> np.ndarray:
    """取短边中心裁成正方形，使 LR latent 为方形以匹配 WatermarkCodec。"""
    h, w = im.shape[:2]
    m = min(h, w)
    if m < 1:
        raise ValueError("图像过小")
    top = (h - m) // 2
    left = (w - m) // 2
    return im[top : top + m, left : left + m].copy()


def crop_to_multiple_of(im: np.ndarray, sf: int) -> np.ndarray:
    h, w = im.shape[:2]
    h2 = (h // sf) * sf
    w2 = (w // sf) * sf
    if h2 < sf or w2 < sf:
        raise ValueError(f"图像过小: {h}x{w}")
    return im[:h2, :w2].copy()


def bicubic_degrade(im_gt: np.ndarray, sf: int) -> np.ndarray:
    return np.clip(util_image.imresize_np(im_gt, scale=1.0 / sf), 0.0, 1.0)


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


def busi_class_from_name(name: str) -> str:
    if "__" in name:
        return name.split("__", 1)[0]
    return "unknown"


def setup_cuda():
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def read_image_list(list_path: Path, project_root: Path) -> list[Path]:
    paths = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if not p.is_absolute():
                p = project_root / p
            paths.append(p)
    return paths


def load_baseline_csv(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["image"]] = row
    return out


def try_lpips():
    try:
        import lpips  # noqa: F401

        return lpips.LPIPS(net="alex").cuda()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="weights/SinSR_v1.pth")
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="configs/SinSR_wm_busi_smoke.yaml",
        help="须 watermark.enabled=true（默认 smoke 配置）",
    )
    parser.add_argument("--image_list", type=str, default="traindata/busi_test.txt")
    parser.add_argument("--sf", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="result/busi_wm_e2e_SinSR_v1")
    parser.add_argument(
        "--baseline_csv",
        type=str,
        default="result/busi_sr_baseline_SinSR_v1/metrics/sr_metrics.csv",
        help="无水印基线 per-image CSV；不存在则仅输出绝对指标",
    )
    parser.add_argument("--num_images", type=int, default=0)
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument(
        "--no_iqa",
        action="store_true",
        help="不计算 MUSIQ / CLIPIQA（需 pip install pyiqa）",
    )
    parser.add_argument(
        "--musiq_ckpt",
        type=str,
        default="",
        help="本地 MUSIQ 权重 .pth；未设置时尝试 weights/musiq_koniq_ckpt-e95806b9.pth 或在线下载",
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
        raise ValueError("本脚本需要水印开启；请使用 watermark.enabled=true 的 yaml（如 SinSR_wm_busi_smoke.yaml）")

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
    if not args.no_iqa:
        if clip_m is None:
            print("[IQA] CLIPIQA 未加载。")
        if mus_m is None:
            print("[IQA] MUSIQ 未加载：可放置权重到 weights/ 或使用 --musiq_ckpt。")
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
    bit_accs = []
    psnr_wm_list = []
    psnr_delta_list = []

    for idx, im_path in enumerate(im_paths):
        if not im_path.exists():
            print(f"[跳过] {im_path}")
            continue
        name = im_path.name
        t0 = time.time()
        im_gt = util_image.imread(str(im_path), chn="rgb", dtype="float32")
        im_gt = center_crop_square(im_gt)
        im_gt, _ = ensure_min_size_square(im_gt, gt_align)
        im_gt = crop_to_multiple_of(im_gt, gt_align)
        im_lq = bicubic_degrade(im_gt, sf=args.sf)
        im_lq_tensor = util_image.img2tensor(im_lq).to(device)

        with torch.no_grad():
            sr_wm, watermark, keys_info = sampler.encode_sr(im_lq_tensor)
            decoded = sampler.extract_watermark(sr_wm, keys_info, sf=args.sf)

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
        _, bit_acc = codec_eval.compute_bit_accuracy(watermark, decoded)
        elapsed = time.time() - t0

        sr_rgb = util_image.tensor2img(sr_wm[0], rgb2bgr=False, min_max=(0.0, 1.0))
        gt_u8 = (im_gt * 255.0).astype(np.uint8)
        psnr_wm = psnr_fn(gt_u8, sr_rgb, data_range=255)
        ssim_wm = float(ssim_fn(gt_u8, sr_rgb, data_range=255, channel_axis=2))
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
        bit_accs.append(float(bit_acc))
        psnr_wm_list.append(float(psnr_wm))

        base = baseline_by_name.get(name, {})
        psnr_b = base.get("psnr", "")
        ssim_b = base.get("ssim", "")
        lp_b = base.get("lpips", "")
        mus_b = base.get("musiq", "")
        cli_b = base.get("clipiqa", "")
        d_psnr = ""
        d_ssim = ""
        d_lp = ""
        d_mus = ""
        d_cli = ""
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
            }
        )
        rows_wm.append(
            {
                "image": name,
                "class": cls,
                "bit_accuracy": round(float(bit_acc), 6),
                "ge_tau": bool(bit_acc >= tau),
                "tau_onebit": tau,
            }
        )
        iq_extra = ""
        if musiq_v is not None:
            iq_extra += f"  MUSIQ_wm={musiq_v:.2f}"
        if clipiqa_v is not None:
            iq_extra += f"  CLIPIQA_wm={clipiqa_v:.4f}"
        print(
            f"[{idx+1}/{len(im_paths)}] {name}  PSNR_wm={psnr_wm:.2f}  Δvs基线={d_psnr if d_psnr != '' else 'N/A'}  BitAcc={bit_acc:.4f}{iq_extra}"
        )

    if not rows_sr:
        raise RuntimeError("无有效样本")

    bit_accs_arr = np.array(bit_accs, dtype=np.float64)
    psnr_arr = np.array(psnr_wm_list, dtype=np.float64)
    summary = {
        "ckpt": ckpt_path.name,
        "cfg": cfg_path.name,
        "mark_length_bits": int(codec.mark_length),
        "tau_onebit": tau,
        "num_images": len(rows_sr),
        "psnr_wm_mean": float(psnr_arr.mean()),
        "psnr_wm_std": float(psnr_arr.std()),
        "bit_accuracy_mean": float(bit_accs_arr.mean()),
        "bit_accuracy_std": float(bit_accs_arr.std()),
        "bit_accuracy_min": float(bit_accs_arr.min()),
        "bit_accuracy_ge_tau_rate": float(np.mean(bit_accs_arr >= tau)),
        "time_mean_s": float(np.mean([float(r["time_encode_extract_s"]) for r in rows_sr])),
    }
    ssims = np.array([float(r["ssim_wm"]) for r in rows_sr], dtype=np.float64)
    summary["ssim_wm_mean"] = float(ssims.mean())
    summary["ssim_wm_std"] = float(ssims.std())
    if psnr_delta_list:
        da = np.array(psnr_delta_list, dtype=np.float64)
        summary["delta_psnr_mean_wm_minus_baseline"] = float(da.mean())
        summary["delta_psnr_std"] = float(da.std())
        summary["delta_psnr_min"] = float(da.min())
        summary["delta_psnr_max"] = float(da.max())
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
    dmus = [float(r["delta_musiq_wm_minus_baseline"]) for r in rows_sr if r.get("delta_musiq_wm_minus_baseline", "") != ""]
    dcli = [float(r["delta_clipiqa_wm_minus_baseline"]) for r in rows_sr if r.get("delta_clipiqa_wm_minus_baseline", "") != ""]
    if dmus:
        summary["delta_musiq_mean_wm_minus_baseline"] = float(np.mean(dmus))
        summary["delta_musiq_std"] = float(np.std(dmus))
    if dcli:
        summary["delta_clipiqa_mean_wm_minus_baseline"] = float(np.mean(dcli))
        summary["delta_clipiqa_std"] = float(np.std(dcli))

    by_class = defaultdict(lambda: {"psnr": [], "bit": []})
    for r, w in zip(rows_sr, rows_wm):
        c = r["class"]
        by_class[c]["psnr"].append(float(r["psnr_wm"]))
        by_class[c]["bit"].append(float(w["bit_accuracy"]))
    summary["per_class"] = {
        k: {
            "n": len(v["psnr"]),
            "psnr_wm_mean": float(np.mean(v["psnr"])),
            "bit_accuracy_mean": float(np.mean(v["bit"])),
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

    analysis = _write_analysis(summary, baseline_path, tau)
    with open(out_dir / "ANALYSIS.txt", "w", encoding="utf-8") as f:
        f.write(analysis)

    with open(out_dir / "CHANGELOG.txt", "w", encoding="utf-8") as f:
        f.write(
            "BUSI 测试集 超分+水印嵌入+提取 E2E\n"
            f"- 配置: {args.cfg_path}\n"
            f"- 基线对照: {args.baseline_csv}（须与基线同为「中心正方形裁切 + sf×image_size 裁边」）\n"
            f"- 列表: {args.image_list}\n"
        )

    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n已写入: {out_dir}")


def _write_analysis(summary: dict, baseline_csv: Path, tau: float) -> str:
    lines = [
        "BUSI 测试集 — 超分 + 水印嵌入 + 提取（端到端）",
        "=" * 52,
        "",
        "一、实验设置",
        f"  权重: {summary.get('ckpt', '')}",
        f"  配置: {summary.get('cfg', '')}（水印开启，Gaussian-Shading 式噪声嵌入）",
        f"  载荷容量: {summary.get('mark_length_bits', '')} bit；检测阈值 tau_onebit = {tau:.4f}",
        f"  样本数: {summary['num_images']}",
        f"  基线对照文件: {baseline_csv if baseline_csv.exists() else '(未找到，未计算 Δ)'}",
        "",
        "二、水印性能（嵌入是否仍可被模型路径恢复）",
        f"  比特准确率 均值: {summary['bit_accuracy_mean']:.4f} ± {summary['bit_accuracy_std']:.4f}",
        f"  最小值: {summary['bit_accuracy_min']:.4f}",
        f"  达到检测阈值比例 (acc >= tau): {summary['bit_accuracy_ge_tau_rate']*100:.1f}%",
        "",
        "  解读：",
        "  - 若均值接近 1.0 且 ge_tau 比例高，说明在当前「预训练 SinSR + 单步」条件下，",
        "    嵌入未破坏可逆链路，提取功能未被破坏。",
        "  - 若均值接近 0.5（随机猜），说明模型尚未针对水印训练（Phase B/C），",
        "    这不等于「嵌入被破坏」，而是提取头/逆向映射未专门优化。",
        "",
        "三、超分指标相对无水印基线（嵌入对画质的影响）",
        f"  含水印 SR — PSNR 均值: {summary['psnr_wm_mean']:.2f} ± {summary['psnr_wm_std']:.2f} dB",
        f"  SSIM 均值: {summary['ssim_wm_mean']:.4f} ± {summary['ssim_wm_std']:.4f}",
    ]
    if "musiq_wm_mean" in summary:
        lines.append(
            f"  MUSIQ（无参考，含水印 SR）: {summary['musiq_wm_mean']:.2f} ± {summary.get('musiq_wm_std', 0):.2f}"
        )
    if "clipiqa_wm_mean" in summary:
        lines.append(
            f"  CLIPIQA（无参考）: {summary['clipiqa_wm_mean']:.4f} ± {summary.get('clipiqa_wm_std', 0):.4f}"
        )
    if "delta_musiq_mean_wm_minus_baseline" in summary:
        lines.append(
            f"  ΔMUSIQ（水印−无水印基线）均值: {summary['delta_musiq_mean_wm_minus_baseline']:+.4f} ± {summary.get('delta_musiq_std', 0):.4f}"
        )
    if "delta_clipiqa_mean_wm_minus_baseline" in summary:
        lines.append(
            f"  ΔCLIPIQA（水印−无水印基线）均值: {summary['delta_clipiqa_mean_wm_minus_baseline']:+.6f} ± {summary.get('delta_clipiqa_std', 0):.4f}"
        )
    lines.append("")
    if "delta_psnr_mean_wm_minus_baseline" in summary:
        dm = summary["delta_psnr_mean_wm_minus_baseline"]
        dmin = summary["delta_psnr_min"]
        dmax = summary["delta_psnr_max"]
        lines += [
            f"  相对基线 PSNR 差（水印 − 无水印）均值: {dm:+.3f} dB（范围 [{dmin:+.3f}, {dmax:+.3f}]）",
            "",
            "  解读：",
            "  - 若 Δ 均值接近 0 且 |Δ| 很小（例如多数在 ±0.3 dB 内），可认为嵌入对像素级",
            "    超分质量影响在工程上可接受；论文中可报告均值±范围。",
            "  - 若 Δ 明显为负且幅度大，需检查是否与随机种子、与基线是否严格同裁边/同退化有关，",
            "    或考虑 Phase A 仅换噪声分布带来的固有差异。",
        ]
    else:
        lines += ["  （未加载基线 CSV，未计算 PSNR 差值。）"]

    lines += [
        "",
        "四、按类别摘要",
    ]
    for cls, st in summary.get("per_class", {}).items():
        lines.append(
            f"  {cls}: n={st['n']}, PSNR_wm={st['psnr_wm_mean']:.2f}, BitAcc={st['bit_accuracy_mean']:.4f}"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
