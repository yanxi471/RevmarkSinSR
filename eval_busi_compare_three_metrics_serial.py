#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BUSI×20（mixed_med4x20/busi20_list.txt）三类指标对比实验——串行入口。

口径摘要：
- 水印有效比特：与其它方案对齐，固定载荷 40 bit（默认 5 字符 ASCII：`WMK26`），RevMark 潜空间向量其余位置填 0；鲁棒性/准确率只对前 K 比特统计。
- RevMark：`no_roi` | `roi_a`（ROI 潜空间高斯 + 非 ROI 水印：`roi_wm_in_roi=False`）| `roi_b`（与 roi_a 同潜空间嵌入 + 像素域 ROI 回填无水印 SR；无水印噪声支路使用 Generator 种子 `seed_nowm_branch + idx`）。
- 图像质量（对比方案：原图 256×256 vs 水印图）：RevMark 将「无水超分」「有水超分」先缩放到 256×256 再互相比 PSNR/SSIM。
- 鲁棒性：默认与 `KEEP_ATTACKS` 一致；`--attack_set ext_no_crop` 时为无裁剪且每类 4 档（25 项含 clean），见 `get_keep_attacks`。
- 恢复质量：RevMark 将含水印 HR 双三次下采样到 LR 与真 LR 比；HS/DE/PEE 用可逆逆变换；TrustMark 用 **TrustMark‑RM（ReMark）** `remove_watermark` 去水印恢复后与 cover 比；CRMark 用官方 **`recover()`** 可逆重建后与 cover 比；InvisibleWM 仍无统一可逆路径，记 nan。

依赖：本项目虚拟环境（torch）；CRMark 子脚本须在独立包含 `crmark` 的 venv 中 `--crmark_python` 指定。

基线可通过 `--baseline_methods trustmark,hs,de,pee` 等形式裁剪（不写 `invisiblewm` 即不跑 InvisibleWM）。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from eval_busi_reference_wm_methods import (
    DEAdapter,
    HSAdapter,
    PEEAdapter,
    TrustMarkAdapter,
    finite_psnr_u8,
    get_keep_attacks,
    invert_de_cover,
    invert_hs_cover,
    invert_pee_cover,
    load_rgb_u8,
    resize_u8,
    text_to_bits,
)
from eval_busi_wm_e2e import bicubic_degrade, center_crop_square, crop_to_multiple_of, read_image_list, setup_cuda
from eval_med_wm_e2e_roi_generic import ensure_min_size_square
from inference_wm import WatermarkSampler
from utils import util_image
from watermark_codec import WatermarkCodec

PAYLOAD_BITS = 40


def resolve_mask_path(mask_root: Path, image_name: str) -> Path | None:
    p1 = mask_root / image_name
    if p1.is_file():
        return p1
    stem = Path(image_name).stem
    for cand in (mask_root / f"{stem}_mask.png", mask_root / f"{stem}.png"):
        if cand.is_file():
            return cand
    return None


def upscale_to_compare(im_u8: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """双三次 resize 至与对比方案一致的显示/评测分辨率（默认 256）。"""
    h, w = hw
    return cv2.resize(im_u8, (int(w), int(h)), interpolation=cv2.INTER_CUBIC).astype(np.uint8)


def bicubic_lr_from_hr(sr_u8: np.ndarray, lq_hw: tuple[int, int]) -> np.ndarray:
    h, w = lq_hw
    return cv2.resize(sr_u8, (int(w), int(h)), interpolation=cv2.INTER_CUBIC).astype(np.uint8)


def bit_accuracy_first_k(orig: torch.Tensor, dec: torch.Tensor, k: int = PAYLOAD_BITS) -> float:
    o = orig.reshape(-1).float()[:k]
    d = dec.reshape(-1).float()[:k]
    if o.numel() == 0:
        return 0.0
    return float((o == d).float().mean().item())


def build_payload_grid(codec: WatermarkCodec, message: str) -> np.ndarray:
    """(wm_ch, wm_h, wm_w) int {0,1}，前 PAYLOAD_BITS 来自 message，其余 0。"""
    wm_ch = int(codec.num_channels // codec.ch_factor)
    wm_h = int(codec.latent_size // codec.hw_factor)
    total = wm_ch * wm_h * wm_h
    flat = np.zeros(total, dtype=np.int64)
    bits = text_to_bits(message).flatten()
    n = min(PAYLOAD_BITS, int(bits.size), total)
    flat[:n] = bits[:n]
    return flat.reshape(1, wm_ch, wm_h, wm_h).astype(np.int32)


def codec_for_latent(codec: WatermarkCodec, latent_sz: int) -> WatermarkCodec:
    if latent_sz == codec.latent_size:
        return codec
    return WatermarkCodec(
        num_channels=codec.num_channels,
        latent_size=int(latent_sz),
        ch_factor=codec.ch_factor,
        hw_factor=codec.hw_factor,
        use_chacha=codec.use_chacha,
    )


def maybe_pixel_roi_blend(
    sampler_wm: WatermarkSampler,
    lq_t: torch.Tensor,
    sr_wm_tensor: torch.Tensor,
    mask_hr: np.ndarray,
    device: str,
    gen_seed: int,
) -> torch.Tensor:
    """与 eval_med_wm_e2e_roi_generic.pixel_roi_fusion 一致：ROI 填入无水印单步 SR，种子独立。"""
    if not mask_hr.any():
        return sr_wm_tensor
    with torch.no_grad():
        y0_norm = (lq_t - 0.5) / 0.5
        z_ref_nowm = sampler_wm.z_y_from_lq_normalized(y0_norm)
        g = torch.Generator(device=device)
        g.manual_seed(int(gen_seed))
        noise_nowm = torch.randn(z_ref_nowm.shape, device=device, dtype=z_ref_nowm.dtype, generator=g)
        sr_nowm_norm = sampler_wm.sample_func_with_noise(y0_norm, noise_nowm, one_step=True)
        sr_nowm = sr_nowm_norm * 0.5 + 0.5
    sr_wm_u8 = util_image.tensor2img(sr_wm_tensor[0], rgb2bgr=False, min_max=(0.0, 1.0)).astype(np.uint8)
    sr_nowm_u8 = util_image.tensor2img(sr_nowm[0], rgb2bgr=False, min_max=(0.0, 1.0)).astype(np.uint8)
    mix = sr_wm_u8.copy()
    mix[mask_hr] = sr_nowm_u8[mask_hr]
    return util_image.img2tensor(mix.astype(np.float32) / 255.0).to(device)


def run_revmark_branches(
    *,
    project_root: Path,
    image_list: Path,
    mask_root: Path,
    out_metrics: Path,
    ckpt: Path,
    cfg_wm: Path,
    cfg_nowm: Path,
    sf: int,
    message: str,
    seed_base: int,
    seed_nowm_branch: int,
    cmp_hw: tuple[int, int],
    device: str,
    attack_set: str = "default",
    num_images: int = 0,
) -> None:
    setup_cuda()
    cfg_b = OmegaConf.load(str(cfg_wm))
    cfg_n = OmegaConf.load(str(cfg_nowm))
    cfg_b.model.ckpt_path = str(ckpt)
    cfg_n.model.ckpt_path = str(ckpt)
    wm_cfg = cfg_b.get("watermark", {})
    codec0 = WatermarkCodec(
        num_channels=cfg_b.autoencoder.params.ddconfig.z_channels,
        latent_size=cfg_b.model.params.image_size,
        ch_factor=wm_cfg.get("ch_factor", 1),
        hw_factor=wm_cfg.get("hw_factor", 8),
        use_chacha=wm_cfg.get("use_chacha", True),
    )
    sampler_wm = WatermarkSampler(cfg_b, codec0, lr_recovery_net=None, sf=sf, seed=int(seed_base))
    sampler_nowm = WatermarkSampler(cfg_n, wm_codec=None, lr_recovery_net=None, sf=sf, seed=int(seed_base))

    gt_align = int(sf) * int(cfg_b.model.params.image_size)
    im_paths = read_image_list(image_list, project_root)
    if int(num_images) > 0:
        im_paths = im_paths[: int(num_images)]
    attacks = list(get_keep_attacks(attack_set))

    rows_iq: list[dict] = []
    rows_rob: list[dict] = []
    rows_rec: list[dict] = []

    for idx, p in enumerate(im_paths):
        if not p.exists():
            print(f"[skip missing] {p}", flush=True)
            continue
        name = p.name
        mask_path = resolve_mask_path(mask_root, name)
        if mask_path is None or not mask_path.is_file():
            print(f"[skip no mask] {name}", flush=True)
            continue

        sampler_wm.setup_seed(int(seed_base + idx))
        sampler_nowm.setup_seed(int(seed_base + idx))
        np.random.seed(int(seed_base + idx * 1315423911) % (2**31 - 1))

        im_gt = util_image.imread(str(p), chn="rgb", dtype="float32")
        im_gt = center_crop_square(im_gt)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            print(f"[skip bad mask] {name}", flush=True)
            continue
        mask_f = (mask_raw.astype(np.float32) / 255.0)
        mask_f = center_crop_square(mask_f[..., None])[..., 0]

        im_gt, _ = ensure_min_size_square(im_gt, gt_align)
        mask_f, _ = ensure_min_size_square(mask_f[..., None], gt_align)
        mask_f = mask_f[..., 0]

        try:
            im_gt = crop_to_multiple_of(im_gt, gt_align)
            mask_f = crop_to_multiple_of(mask_f[..., None], gt_align)[..., 0]
        except ValueError:
            print(f"[skip crop] {name}", flush=True)
            continue

        mask_hr = mask_f >= 0.5
        im_lq = bicubic_degrade(im_gt, sf=sf)
        lq_hw = (int(im_lq.shape[0]), int(im_lq.shape[1]))
        lq_t = util_image.img2tensor(im_lq).to(device)

        with torch.no_grad():
            z_ref = sampler_wm.z_y_from_lq_normalized((lq_t - 0.5) / 0.5)
        latent_sz = int(z_ref.shape[2])
        codec = codec_for_latent(codec0, latent_sz)
        payload = build_payload_grid(codec, message)

        roi_grid = cv2.resize(mask_f.astype(np.float32), (codec.latent_size // codec.hw_factor, codec.latent_size // codec.hw_factor), interpolation=cv2.INTER_NEAREST)
        roi_grid = (roi_grid >= 0.5).astype(np.float32)
        roi_ok = float(roi_grid.max()) >= 0.5

        branches = [("no_roi", {})]
        if roi_ok:
            roi_kw = dict(roi_wm_mask=roi_grid, roi_threshold=0.5, roi_wm_in_roi=False)
            branches.append(("roi_a", dict(roi_kw)))
            branches.append(("roi_b", {**roi_kw, "pixel_fuse": True}))

        for bname, enc_extra_raw in branches:
            enc_extra = dict(enc_extra_raw)
            pixel_fuse = bool(enc_extra.pop("pixel_fuse", False))

            with torch.no_grad():
                sr_nowm, _, _ = sampler_nowm.encode_sr(lq_t)
                sr_wm, watermark, keys = sampler_wm.encode_sr(lq_t, payload_bits=torch.as_tensor(payload, device=device), **enc_extra)
                if pixel_fuse:
                    sr_eval = maybe_pixel_roi_blend(
                        sampler_wm,
                        lq_t,
                        sr_wm,
                        mask_hr,
                        device,
                        gen_seed=int(seed_nowm_branch + idx),
                    )
                else:
                    sr_eval = sr_wm

            nowm_u8 = util_image.tensor2img(sr_nowm[0], rgb2bgr=False, min_max=(0.0, 1.0)).astype(np.uint8)
            wm_u8 = util_image.tensor2img(sr_eval[0], rgb2bgr=False, min_max=(0.0, 1.0)).astype(np.uint8)
            a256 = upscale_to_compare(nowm_u8, cmp_hw)
            b256 = upscale_to_compare(wm_u8, cmp_hw)
            rows_iq.append(
                {
                    "method": "revmark",
                    "branch": bname,
                    "image": name,
                    "psnr_nowm_vs_wm_at256": float(psnr_fn(a256, b256, data_range=255)),
                    "ssim_nowm_vs_wm_at256": float(ssim_fn(a256, b256, data_range=255, channel_axis=2)),
                }
            )

            with torch.no_grad():
                decoded0 = sampler_wm.extract_watermark(sr_eval, keys, sf=sf)
            rows_rob.append(
                {
                    "method": "revmark",
                    "branch": bname,
                    "image": name,
                    "attack": "clean",
                    "bitacc_first40": bit_accuracy_first_k(watermark, decoded0, PAYLOAD_BITS),
                }
            )
            for atk_name, fn in attacks:
                if atk_name == "clean":
                    continue
                atk_u8 = fn(wm_u8)
                atk_t = util_image.img2tensor(atk_u8.astype(np.float32) / 255.0).to(device)
                with torch.no_grad():
                    decoded = sampler_wm.extract_watermark(atk_t, keys, sf=sf)
                rows_rob.append(
                    {
                        "method": "revmark",
                        "branch": bname,
                        "image": name,
                        "attack": atk_name,
                        "bitacc_first40": bit_accuracy_first_k(watermark, decoded, PAYLOAD_BITS),
                    }
                )

            rec_u8 = bicubic_lr_from_hr(wm_u8, lq_hw)
            lq_u8 = (np.clip(im_lq, 0.0, 1.0) * 255.0).astype(np.uint8)
            rows_rec.append(
                {
                    "method": "revmark",
                    "branch": bname,
                    "image": name,
                    "psnr_rec_lr": float(psnr_fn(lq_u8, rec_u8, data_range=255)),
                    "ssim_rec_lr": float(ssim_fn(lq_u8, rec_u8, data_range=255, channel_axis=2)),
                }
            )
        print(f"[revmark] {idx + 1}/{len(im_paths)} {name}", flush=True)

    out_metrics.mkdir(parents=True, exist_ok=True)
    for fname, rows in (
        ("revmark_image_quality.csv", rows_iq),
        ("revmark_robustness.csv", rows_rob),
        ("revmark_recovery_lr.csv", rows_rec),
    ):
        if not rows:
            continue
        with (out_metrics / fname).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def run_subprocess(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def resolve_crmark_python(project_root: Path, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    for rel in ("external/crmark_venv/Scripts/python.exe", "external/crmark_venv/bin/python"):
        p = project_root / rel
        if p.is_file():
            return str(p.resolve())
    return ""


def recovery_baseline_block(
    *,
    project_root: Path,
    image_list: Path,
    out_csv: Path,
    message: str,
    crmark_python: str,
) -> None:
    im_paths = read_image_list(image_list, project_root)
    adapters = {
        "hs": (HSAdapter(), invert_hs_cover),
        "de": (DEAdapter(), invert_de_cover),
        "pee": (PEEAdapter(), invert_pee_cover),
    }
    rows: list[dict] = []
    ref_bits_flat = text_to_bits(message).flatten()
    for p in im_paths:
        if not p.exists():
            continue
        cover = resize_u8(center_crop_square(load_rgb_u8(p)), (256, 256))
        for mname, (adp, inv) in adapters.items():
            stego, meta = adp.encode(cover, message)
            kb = ref_bits_flat[: int(meta["n_bits"])]
            rec = inv(stego, meta, kb)
            rows.append(
                {
                    "method": mname,
                    "image": p.name,
                    "psnr_rec_vs_orig": finite_psnr_u8(cover, rec),
                    "ssim_rec_vs_orig": float(ssim_fn(cover, rec, data_range=255, channel_axis=2)),
                }
            )

    try:
        tm = TrustMarkAdapter(load_remover=True, device="", verbose=False)
        for p in im_paths:
            if not p.exists():
                continue
            cover = resize_u8(center_crop_square(load_rgb_u8(p)), (256, 256))
            stego, _meta = tm.encode(cover, message)
            rec = tm.remove_watermark_remark(stego)
            rows.append(
                {
                    "method": "trustmark_rm",
                    "image": p.name,
                    "psnr_rec_vs_orig": finite_psnr_u8(cover, rec),
                    "ssim_rec_vs_orig": float(ssim_fn(cover, rec, data_range=255, channel_axis=2)),
                }
            )
    except Exception as exc:
        print(f"[recovery] TrustMark-RM 跳过: {exc}", flush=True)
        for p in im_paths:
            if not p.exists():
                continue
            rows.append({"method": "trustmark_rm", "image": p.name, "psnr_rec_vs_orig": float("nan"), "ssim_rec_vs_orig": float("nan")})

    for p in im_paths:
        if not p.exists():
            continue
        rows.append({"method": "invisiblewm", "image": p.name, "psnr_rec_vs_orig": float("nan"), "ssim_rec_vs_orig": float("nan")})

    cr_part = out_csv.parent / "baseline_recovery_crmark_part.csv"
    if crmark_python:
        try:
            run_subprocess(
                [
                    crmark_python,
                    str(project_root / "scripts/run_crmark_recovery_metrics.py"),
                    "--image_list",
                    str(image_list),
                    "--out_csv",
                    str(cr_part),
                    "--message",
                    message,
                ],
                cwd=project_root,
            )
            with cr_part.open(encoding="utf-8") as f:
                part = list(csv.DictReader(f))
            for r in part:

                def _fcell(x: str) -> float:
                    if x is None or x == "":
                        return float("nan")
                    try:
                        return float(x)
                    except ValueError:
                        return float("nan")

                rows.append(
                    {
                        "method": r.get("method", "crmark"),
                        "image": r["image"],
                        "psnr_rec_vs_orig": _fcell(r.get("psnr_rec_vs_orig")),
                        "ssim_rec_vs_orig": _fcell(r.get("ssim_rec_vs_orig")),
                    }
                )
        except Exception as exc:
            print(f"[recovery] CRMark 跳过: {exc}", flush=True)
            for p in im_paths:
                if not p.exists():
                    continue
                rows.append({"method": "crmark", "image": p.name, "psnr_rec_vs_orig": float("nan"), "ssim_rec_vs_orig": float("nan")})
    else:
        for p in im_paths:
            if not p.exists():
                continue
            rows.append({"method": "crmark", "image": p.name, "psnr_rec_vs_orig": float("nan"), "ssim_rec_vs_orig": float("nan")})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "image", "psnr_rec_vs_orig", "ssim_rec_vs_orig"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_list", type=str, default="data/mixed_med4x20/busi20_list.txt")
    ap.add_argument("--mask_root", type=str, default="data/mixed_med4x20/masks")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="result/busi_compare_three_metrics_serial",
        help="输出根目录。切换 attack_set 时请改用新目录，避免与旧版 8+1 攻击 CSV 混用。",
    )
    ap.add_argument("--ckpt", type=str, default="weights/SinSR_v1.pth")
    ap.add_argument("--cfg_wm", type=str, default="configs/SinSR_wm_busi_smoke.yaml")
    ap.add_argument("--cfg_nowm", type=str, default="configs/SinSR_wm_busi_smoke_nowm.yaml")
    ap.add_argument("--sf", type=int, default=4)
    ap.add_argument("--message", type=str, default="WMK26")
    ap.add_argument("--seed_base", type=int, default=2026)
    ap.add_argument("--seed_nowm_branch", type=int, default=913337)
    ap.add_argument("--cmp_hw", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--skip_revmark", action="store_true")
    ap.add_argument("--skip_baselines", action="store_true")
    ap.add_argument("--skip_recovery", action="store_true", help="跳过 baseline 恢复质量（HS/DE/PEE + TrustMark‑RM + CRMark）")
    ap.add_argument("--skip_crmark", action="store_true")
    ap.add_argument(
        "--crmark_python",
        type=str,
        default="",
        help="含 crmark 的 python.exe；为空时尝试 external/crmark_venv。用于 CRMark 鲁棒子脚本与 recover 恢复指标。",
    )
    ap.add_argument(
        "--attack_set",
        type=str,
        default="default",
        choices=("default", "ext_no_crop"),
        help="鲁棒性攻击集：ext_no_crop=无裁剪，每类攻击 4 档参数（与 eval_busi_reference_wm_methods 一致）。",
    )
    ap.add_argument("--num_images", type=int, default=0, help="仅处理前 N 张图（0 表示全部）；用于快速试跑。")
    ap.add_argument(
        "--baseline_methods",
        type=str,
        default="trustmark,invisiblewm,hs,de,pee",
        help="逗号分隔，子进程 eval_busi_reference_wm_methods 的 --method。例：trustmark,hs,de,pee（不含 invisiblewm）。",
    )
    args = ap.parse_args()

    _baseline_allowed = {"trustmark", "invisiblewm", "hs", "de", "pee"}
    baseline_methods = [x.strip().lower() for x in str(args.baseline_methods).split(",") if x.strip()]
    for m in baseline_methods:
        if m not in _baseline_allowed:
            raise SystemExit(f"--baseline_methods 含非法项 {m!r}，允许: {sorted(_baseline_allowed)}")

    project_root = Path(__file__).resolve().parent
    image_list = (project_root / args.image_list).resolve()
    mask_root = (project_root / args.mask_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    metrics = out_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)

    cmp_hw_t = (int(args.cmp_hw[0]), int(args.cmp_hw[1]))
    py = sys.executable

    wall0 = time.time()
    manifest = {
        "image_list": str(image_list.relative_to(project_root)),
        "payload_bits": PAYLOAD_BITS,
        "message": args.message,
        "attack_set": args.attack_set,
        "num_images": int(args.num_images),
        "baseline_methods": baseline_methods,
    }

    if not args.skip_revmark:
        run_revmark_branches(
            project_root=project_root,
            image_list=image_list,
            mask_root=mask_root,
            out_metrics=metrics,
            ckpt=(project_root / args.ckpt).resolve(),
            cfg_wm=(project_root / args.cfg_wm).resolve(),
            cfg_nowm=(project_root / args.cfg_nowm).resolve(),
            sf=int(args.sf),
            message=args.message,
            seed_base=int(args.seed_base),
            seed_nowm_branch=int(args.seed_nowm_branch),
            cmp_hw=cmp_hw_t,
            device="cuda" if torch.cuda.is_available() else "cpu",
            attack_set=args.attack_set,
            num_images=int(args.num_images),
        )

    if not args.skip_baselines:
        for meth in baseline_methods:
            for mode in ["clean", "robustness"]:
                sub_out = out_dir / f"baseline_{meth}_{mode}"
                cmd = [
                    py,
                    str(project_root / "eval_busi_reference_wm_methods.py"),
                    "--method",
                    meth,
                    "--mode",
                    mode,
                    "--image_list",
                    str(image_list),
                    "--out_dir",
                    str(sub_out),
                    "--message",
                    args.message,
                    "--attack_set",
                    args.attack_set,
                ]
                if int(args.num_images) > 0:
                    cmd += ["--num_images", str(int(args.num_images))]
                run_subprocess(cmd, cwd=project_root)

    crmark_py = resolve_crmark_python(project_root, args.crmark_python)
    crmark_py_recovery = "" if args.skip_crmark else crmark_py
    if not args.skip_recovery:
        recovery_baseline_block(
            project_root=project_root,
            image_list=image_list,
            out_csv=metrics / "baseline_recovery_hs_de_pee.csv",
            message=args.message,
            crmark_python=crmark_py_recovery,
        )

    if not args.skip_crmark and crmark_py:
        cr_cmd = [
            crmark_py,
            str(project_root / "scripts/run_crmark_busi_three_metrics.py"),
            "--image_list",
            str(image_list),
            "--out_dir",
            str(out_dir / "baseline_crmark"),
            "--message",
            args.message,
            "--attack_set",
            args.attack_set,
        ]
        if int(args.num_images) > 0:
            cr_cmd += ["--num_images", str(int(args.num_images))]
        run_subprocess(cr_cmd, cwd=project_root)

    (metrics / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["elapsed_s"] = round(time.time() - wall0, 3)
    (metrics / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"完成，输出目录: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
