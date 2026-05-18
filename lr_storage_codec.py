"""
Lossless LR storage codec for Phase E.

The exact-LR channel is split into two layers:
  - packet mode: how exact LR bytes are packed/encrypted
  - carrier type: where those packet bytes are embedded in the SR image

Legacy aliases remain supported:
  - `lsb_raw_v1`
  - `ihaar_residual_lsb_v1`
  - `ihaar_residual_h1stc_v1`
"""

import os
import struct
import zlib

import numpy as np
import torch
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes


RAW_MODE = "lsb_raw_v1"
PREDICTIVE_MODE = "ihaar_residual_lsb_v1"
H1_STC_MODE = "ihaar_residual_h1stc_v1"
SUPPORTED_MODES = {RAW_MODE, PREDICTIVE_MODE, H1_STC_MODE}

RAW_PACKET_MODE = "raw_v1"
PREDICTIVE_PACKET_MODE = "predictive_residual_v1"
SUPPORTED_PACKET_MODES = {RAW_PACKET_MODE, PREDICTIVE_PACKET_MODE}

PIXEL_LSB_CARRIER = "pixel_lsb_v1"
IHAAR_H1STC_CARRIER = "ihaar_h1stc_v1"
SUPPORTED_CARRIER_TYPES = {PIXEL_LSB_CARRIER, IHAAR_H1STC_CARRIER}

MODE_ALIAS_TO_LAYOUT = {
    RAW_MODE: (RAW_PACKET_MODE, PIXEL_LSB_CARRIER),
    PREDICTIVE_MODE: (PREDICTIVE_PACKET_MODE, PIXEL_LSB_CARRIER),
    H1_STC_MODE: (PREDICTIVE_PACKET_MODE, IHAAR_H1STC_CARRIER),
}
LAYOUT_TO_MODE_ALIAS = {value: key for key, value in MODE_ALIAS_TO_LAYOUT.items()}


def predictor_hash_from_id(predictor_id):
    predictor_id = str(predictor_id or "none")
    return zlib.crc32(predictor_id.encode("utf-8")) & 0xFFFFFFFF


def canonical_mode_from_layout(packet_mode, carrier_type):
    layout = (str(packet_mode), str(carrier_type))
    if layout in LAYOUT_TO_MODE_ALIAS:
        return LAYOUT_TO_MODE_ALIAS[layout]
    return f"{layout[0]}__{layout[1]}"


def resolve_lr_storage_layout(mode=None, packet_mode=None, carrier_type=None):
    if packet_mode is None and carrier_type is None:
        mode = str(mode or RAW_MODE)
        if mode not in MODE_ALIAS_TO_LAYOUT:
            raise ValueError(f"Unsupported LR storage mode: {mode}")
        packet_mode, carrier_type = MODE_ALIAS_TO_LAYOUT[mode]
        resolved_mode = mode
    else:
        packet_mode = str(packet_mode or RAW_PACKET_MODE)
        carrier_type = str(carrier_type or PIXEL_LSB_CARRIER)
        if packet_mode not in SUPPORTED_PACKET_MODES:
            raise ValueError(f"Unsupported LR packet mode: {packet_mode}")
        if carrier_type not in SUPPORTED_CARRIER_TYPES:
            raise ValueError(f"Unsupported LR carrier type: {carrier_type}")
        resolved_mode = canonical_mode_from_layout(packet_mode, carrier_type)
    return str(packet_mode), str(carrier_type), str(resolved_mode)


def _validate_hwc_uint8(image, name):
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise TypeError(f"{name} must be an HWC uint8 numpy array")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} must use uint8 dtype")
    return np.ascontiguousarray(image)


def _zigzag_encode_np(values):
    values = np.ascontiguousarray(values).astype(np.int32)
    encoded = (values << 1) ^ (values >> 31)
    return encoded.astype(np.uint16)


def _zigzag_decode_np(values_u16):
    values_u16 = np.ascontiguousarray(values_u16).astype(np.int32)
    return (values_u16 >> 1) ^ (-(values_u16 & 1))


def _haar_split_np(array, axis):
    slicer_even = [slice(None)] * array.ndim
    slicer_odd = [slice(None)] * array.ndim
    slicer_even[axis] = slice(0, None, 2)
    slicer_odd[axis] = slice(1, None, 2)
    even = array[tuple(slicer_even)]
    odd = array[tuple(slicer_odd)]
    detail = odd - even
    approx = even + np.floor_divide(detail, 2)
    return approx.astype(np.int32), detail.astype(np.int32)


def _haar_merge_np(approx, detail, axis):
    even = approx - np.floor_divide(detail, 2)
    odd = detail + even
    out_shape = list(approx.shape)
    out_shape[axis] = approx.shape[axis] * 2
    output = np.empty(out_shape, dtype=np.int32)
    slicer_even = [slice(None)] * output.ndim
    slicer_odd = [slice(None)] * output.ndim
    slicer_even[axis] = slice(0, None, 2)
    slicer_odd[axis] = slice(1, None, 2)
    output[tuple(slicer_even)] = even
    output[tuple(slicer_odd)] = odd
    return output


def integer_haar_forward_np(image_hwc):
    image_hwc = np.ascontiguousarray(image_hwc).astype(np.int32)
    h, w = image_hwc.shape[:2]
    if (h % 2) != 0 or (w % 2) != 0:
        raise ValueError("integer Haar transform requires even image dimensions")

    low_rows, high_rows = _haar_split_np(image_hwc, axis=0)
    ll, lh = _haar_split_np(low_rows, axis=1)
    hl, hh = _haar_split_np(high_rows, axis=1)
    return ll, (lh, hl, hh)


def integer_haar_inverse_np(ll, bands):
    lh, hl, hh = bands
    low_rows = _haar_merge_np(ll.astype(np.int32), lh.astype(np.int32), axis=1)
    high_rows = _haar_merge_np(hl.astype(np.int32), hh.astype(np.int32), axis=1)
    return _haar_merge_np(low_rows, high_rows, axis=0)


def integer_haar_forward_levels_np(image_hwc, levels=2):
    current = np.ascontiguousarray(image_hwc).astype(np.int32)
    bands = []
    for _ in range(int(levels)):
        current, band = integer_haar_forward_np(current)
        bands.append(band)
    return current, bands


def integer_haar_inverse_levels_np(lowband, bands):
    current = np.ascontiguousarray(lowband).astype(np.int32)
    for band in reversed(list(bands)):
        current = integer_haar_inverse_np(current, band)
    return current


def integer_haar_ll_levels_np(image_hwc, levels=2):
    lowband, _ = integer_haar_forward_levels_np(image_hwc, levels=levels)
    return np.clip(lowband, 0, 255).astype(np.uint8)


def _haar_split_torch(tensor, axis):
    index_even = [slice(None)] * tensor.ndim
    index_odd = [slice(None)] * tensor.ndim
    index_even[axis] = slice(0, None, 2)
    index_odd[axis] = slice(1, None, 2)
    even = tensor[tuple(index_even)]
    odd = tensor[tuple(index_odd)]
    detail = odd - even
    approx = even + torch.div(detail, 2, rounding_mode="floor")
    return approx.to(torch.int32), detail.to(torch.int32)


def integer_haar_ll_levels_torch(image_bchw_u8, levels=2):
    if image_bchw_u8.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(image_bchw_u8.shape)}")
    current = image_bchw_u8.to(torch.int32)
    if (current.shape[-2] % (2 ** levels)) != 0 or (current.shape[-1] % (2 ** levels)) != 0:
        raise ValueError("integer Haar transform requires height/width divisible by 2**levels")
    for _ in range(int(levels)):
        low_rows, _ = _haar_split_torch(current, axis=-2)
        current, _ = _haar_split_torch(low_rows, axis=-1)
    return current


class LRStorageCodec:
    MAGIC = b"RMLS"
    RAW_VERSION = 1
    PREDICTIVE_VERSION = 2
    RAW_HEADER_STRUCT = struct.Struct(">4sBBHHBII")
    PREDICTIVE_HEADER_STRUCT = struct.Struct(">4sBBBBHHBIII")

    FLAG_CHACHA20 = 1 << 0
    PACK_ZIGZAG_U16 = 1
    PACK_NAME_TO_CODE = {"zigzag_u16": PACK_ZIGZAG_U16}
    PACK_CODE_TO_NAME = {PACK_ZIGZAG_U16: "zigzag_u16"}

    def __init__(
        self,
        mode=RAW_MODE,
        *,
        packet_mode=None,
        carrier_type=None,
        carrier_config=None,
        bitplanes=1,
        carrier_fraction="auto",
        use_chacha=True,
        predictor=None,
        predictor_id="none",
        analysis_levels=2,
        compressor_type="zlib",
        compressor_level=9,
        residual_pack="zigzag_u16",
    ):
        packet_mode, carrier_type, resolved_mode = resolve_lr_storage_layout(
            mode=mode,
            packet_mode=packet_mode,
            carrier_type=carrier_type,
        )
        if int(bitplanes) < 1:
            raise ValueError("bitplanes must be >= 1")
        if carrier_fraction != "auto":
            carrier_fraction = float(carrier_fraction)
            if carrier_fraction <= 0.0 or carrier_fraction > 1.0:
                raise ValueError("carrier_fraction must be in (0, 1] or 'auto'")
        if int(analysis_levels) < 1:
            raise ValueError("analysis_levels must be >= 1")
        if compressor_type != "zlib":
            raise ValueError(f"Unsupported compressor_type: {compressor_type}")
        if residual_pack not in self.PACK_NAME_TO_CODE:
            raise ValueError(f"Unsupported residual_pack: {residual_pack}")

        self.mode = resolved_mode
        self.packet_mode = packet_mode
        self.carrier_type = carrier_type
        self.bitplanes = int(bitplanes)
        self.carrier_fraction = carrier_fraction
        self.use_chacha = bool(use_chacha)
        self.predictor = predictor
        self.predictor_id = str(predictor_id or "none")
        self.predictor_hash = predictor_hash_from_id(self.predictor_id)
        self.analysis_levels = int(analysis_levels)
        self.compressor_type = compressor_type
        self.compressor_level = int(compressor_level)
        self.residual_pack = residual_pack
        self.carrier_config = dict(carrier_config or {})
        self.embed_retry_trials = max(1, int(self.carrier_config.get("embed_retry_trials", 1)))
        self.carrier = self._build_carrier(self.carrier_config)

    def _build_carrier(self, carrier_config):
        if self.carrier_type == PIXEL_LSB_CARRIER:
            return None
        if self.carrier_type != IHAAR_H1STC_CARRIER:
            raise ValueError(f"Unsupported carrier_type: {self.carrier_type}")
        from carriers.hf_stc_carrier import IHaarHFSTCCarrier

        return IHaarHFSTCCarrier(
            analysis_levels=self.analysis_levels,
            forward_levels_fn=integer_haar_forward_levels_np,
            inverse_levels_fn=integer_haar_inverse_levels_np,
            inverse_single_level_fn=integer_haar_inverse_np,
            subbands=carrier_config.get("subbands", ["hh1", "hl1", "lh1"]),
            spill_to_hh2=carrier_config.get("spill_to_hh2", False),
            max_fill_ratio=carrier_config.get("max_fill_ratio", 1.0),
            seed_trials=carrier_config.get("seed_trials", 1),
            alpha_vis=carrier_config.get("alpha_vis", 1.0),
            beta_secret=carrier_config.get("beta_secret", 0.0),
            gamma_lowpass=carrier_config.get("gamma_lowpass", 0.0),
            delta_level=carrier_config.get("delta_level", 1.0),
            eta_clip=carrier_config.get("eta_clip", 1000.0),
            forbid_zero_abs_le=carrier_config.get("forbid_zero_abs_le", 0),
            secret_map_path=carrier_config.get("secret_map_path", None),
            lowpass_map_path=carrier_config.get("lowpass_map_path", None),
            seed=carrier_config.get("seed", None),
            seed_key_env=carrier_config.get("seed_key_env", None),
        )

    def _resolve_carrier_seed(self, key_info=None):
        if key_info is not None and key_info.get("carrier_seed", None) is not None:
            return int(key_info["carrier_seed"])
        if key_info is not None and key_info.get("perm_seed", None) is not None:
            return int(key_info["perm_seed"])
        if self.carrier is not None:
            default_seed = self.carrier.default_seed()
            if default_seed is not None:
                return int(default_seed)
        env_name = self.carrier_config.get("seed_key_env", None)
        if env_name:
            raw_value = os.environ.get(str(env_name), None)
            if raw_value:
                return int(raw_value, 0)
        if self.carrier_config.get("seed", None) is not None:
            return int(self.carrier_config["seed"])
        return int(np.random.randint(0, 2**31 - 1))

    def _encrypt_bytes(self, raw_bytes, key_info=None):
        if self.use_chacha:
            if key_info is None:
                key = get_random_bytes(32)
                nonce = get_random_bytes(12)
            else:
                key = key_info["enc_key"]
                nonce = key_info["nonce"]
            cipher = ChaCha20.new(key=key, nonce=nonce)
            return cipher.encrypt(raw_bytes), {"enc_key": key, "nonce": nonce}

        if key_info is None:
            xor_seed = int(np.random.randint(0, 2**31 - 1))
        else:
            xor_seed = int(key_info["xor_seed"])
        rng = np.random.default_rng(xor_seed)
        pad = rng.integers(0, 256, size=len(raw_bytes), dtype=np.uint8).tobytes()
        encrypted = bytes(a ^ b for a, b in zip(raw_bytes, pad))
        return encrypted, {"xor_seed": xor_seed}

    def _decrypt_bytes(self, encrypted_bytes, key_info):
        if self.use_chacha:
            cipher = ChaCha20.new(key=key_info["enc_key"], nonce=key_info["nonce"])
            return cipher.decrypt(encrypted_bytes)

        xor_seed = int(key_info["xor_seed"])
        rng = np.random.default_rng(xor_seed)
        pad = rng.integers(0, 256, size=len(encrypted_bytes), dtype=np.uint8).tobytes()
        return bytes(a ^ b for a, b in zip(encrypted_bytes, pad))

    def _predictor_device(self):
        if self.predictor is None:
            return torch.device("cpu")
        if isinstance(self.predictor, torch.nn.Module):
            try:
                return next(self.predictor.parameters()).device
            except StopIteration:
                return torch.device("cpu")
        return torch.device("cpu")

    def _predictor_anchor_u8(self, sr_u8):
        sr_u8 = _validate_hwc_uint8(sr_u8, "sr_u8")
        if self.carrier_type != PIXEL_LSB_CARRIER:
            return sr_u8
        low_mask = (1 << self.bitplanes) - 1
        keep_mask = np.uint8(0xFF ^ low_mask)
        return np.bitwise_and(sr_u8, keep_mask)

    def _predict_lr_u8(self, sr_u8, ll2_override=None):
        if self.predictor is None:
            raise ValueError("predictor_required")

        sr_u8 = _validate_hwc_uint8(sr_u8, "sr_u8")
        h, w = sr_u8.shape[:2]
        if (h % (2 ** self.analysis_levels)) != 0 or (w % (2 ** self.analysis_levels)) != 0:
            raise ValueError("sr_u8 dimensions must be divisible by 2**analysis_levels")

        if hasattr(self.predictor, "predict_from_cover"):
            anchor_u8 = self._predictor_anchor_u8(sr_u8)
            pred_u8 = self.predictor.predict_from_cover(anchor_u8)
            pred_u8 = _validate_hwc_uint8(pred_u8, "pred_u8")
            return pred_u8, None

        if ll2_override is None:
            anchor_u8 = self._predictor_anchor_u8(sr_u8)
            ll2_u8 = integer_haar_ll_levels_np(anchor_u8, levels=self.analysis_levels)
        else:
            ll2_u8 = _validate_hwc_uint8(ll2_override, "ll2_override")

        if hasattr(self.predictor, "predict_mean"):
            predictor_fn = self.predictor.predict_mean
        else:
            predictor_fn = self.predictor

        device = self._predictor_device()
        ll2_tensor = torch.from_numpy(ll2_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = predictor_fn(ll2_tensor)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]

        if torch.is_tensor(pred):
            pred_np = pred.detach().float().clamp(0.0, 1.0).cpu().permute(0, 2, 3, 1).numpy()[0]
        else:
            pred_np = np.asarray(pred)
            if pred_np.ndim == 4:
                pred_np = pred_np[0]
            if pred_np.ndim == 3 and pred_np.shape[0] == ll2_u8.shape[2]:
                pred_np = np.transpose(pred_np, (1, 2, 0))
        if pred_np.shape != ll2_u8.shape:
            raise ValueError(
                f"Predictor output shape {pred_np.shape} does not match LL shape {ll2_u8.shape}"
            )

        pred_u8 = np.clip(np.rint(pred_np * 255.0), 0, 255).astype(np.uint8)
        return pred_u8, ll2_u8

    def _build_raw_payload_bytes(self, lr_u8, key_info=None):
        lr_u8 = _validate_hwc_uint8(lr_u8, "lr_u8")
        h, w, c = [int(v) for v in lr_u8.shape]
        raw_bytes = lr_u8.tobytes()
        crc32 = zlib.crc32(raw_bytes) & 0xFFFFFFFF

        encrypted_bytes, enc_meta = self._encrypt_bytes(raw_bytes, key_info=key_info)
        flags = self.FLAG_CHACHA20 if self.use_chacha else 0
        header = self.RAW_HEADER_STRUCT.pack(
            self.MAGIC,
            self.RAW_VERSION,
            flags,
            h,
            w,
            c,
            len(raw_bytes),
            crc32,
        )
        payload_bytes = header + encrypted_bytes

        return payload_bytes, {
            "lr_shape": (h, w, c),
            "crc32": crc32,
            "header_version": self.RAW_VERSION,
            "raw_bits": int(len(raw_bytes) * 8),
            "compressed_bits": int(len(raw_bytes) * 8),
            "payload_savings": 0,
            "analysis_levels": 0,
            "predictor_id": None,
            "predictor_hash": None,
            "compressed_len": int(len(raw_bytes)),
            "pack_code": None,
            "packet_mode": RAW_PACKET_MODE,
            "mode": self.mode,
            **enc_meta,
        }

    def _pack_predictive_residual(self, residual):
        residual = np.ascontiguousarray(residual).astype(np.int16)
        if np.min(residual) < -255 or np.max(residual) > 255:
            raise ValueError("Predictive residual is outside the supported [-255, 255] range")
        packed = _zigzag_encode_np(residual.astype(np.int32))
        return packed.astype("<u2", copy=False).tobytes()

    def _unpack_predictive_residual(self, residual_bytes, shape):
        expected_bytes = int(np.prod(shape)) * 2
        if len(residual_bytes) != expected_bytes:
            raise ValueError(
                f"Expected {expected_bytes} residual bytes, got {len(residual_bytes)}"
            )
        packed = np.frombuffer(residual_bytes, dtype="<u2").reshape(shape)
        return _zigzag_decode_np(packed).astype(np.int16)

    def _build_predictive_payload_bytes(self, sr_u8, lr_u8, key_info=None):
        sr_u8 = _validate_hwc_uint8(sr_u8, "sr_u8")
        lr_u8 = _validate_hwc_uint8(lr_u8, "lr_u8")
        h, w, c = [int(v) for v in lr_u8.shape]

        pred_u8, ll2_u8 = self._predict_lr_u8(sr_u8)
        if pred_u8.shape != lr_u8.shape:
            raise ValueError(
                f"Predictor output shape {pred_u8.shape} does not match LR shape {lr_u8.shape}"
            )

        residual = lr_u8.astype(np.int16) - pred_u8.astype(np.int16)
        residual_bytes = self._pack_predictive_residual(residual)
        compressed_bytes = zlib.compress(residual_bytes, level=self.compressor_level)
        raw_lr_bytes = lr_u8.tobytes()
        crc32 = zlib.crc32(raw_lr_bytes) & 0xFFFFFFFF

        encrypted_bytes, enc_meta = self._encrypt_bytes(compressed_bytes, key_info=key_info)
        flags = self.FLAG_CHACHA20 if self.use_chacha else 0
        pack_code = self.PACK_NAME_TO_CODE[self.residual_pack]
        header = self.PREDICTIVE_HEADER_STRUCT.pack(
            self.MAGIC,
            self.PREDICTIVE_VERSION,
            flags,
            self.analysis_levels,
            pack_code,
            h,
            w,
            c,
            len(compressed_bytes),
            crc32,
            self.predictor_hash,
        )
        payload_bytes = header + encrypted_bytes

        return payload_bytes, {
            "lr_shape": (h, w, c),
            "crc32": crc32,
            "header_version": self.PREDICTIVE_VERSION,
            "raw_bits": int(len(raw_lr_bytes) * 8),
            "compressed_bits": int(len(compressed_bytes) * 8),
            "payload_savings": int(len(raw_lr_bytes) * 8 - len(compressed_bytes) * 8),
            "analysis_levels": self.analysis_levels,
            "predictor_id": self.predictor_id,
            "predictor_hash": int(self.predictor_hash),
            "compressed_len": int(len(compressed_bytes)),
            "pack_code": int(pack_code),
            "ll2_shape": tuple(int(v) for v in ll2_u8.shape) if ll2_u8 is not None else None,
            "packet_mode": PREDICTIVE_PACKET_MODE,
            "mode": self.mode,
            **enc_meta,
        }

    def _build_payload_bytes(self, sr_u8, lr_u8, key_info=None):
        if self.packet_mode == RAW_PACKET_MODE:
            return self._build_raw_payload_bytes(lr_u8, key_info=key_info)
        if self.packet_mode == PREDICTIVE_PACKET_MODE:
            return self._build_predictive_payload_bytes(sr_u8, lr_u8, key_info=key_info)
        raise ValueError(f"Unsupported LR packet mode: {self.packet_mode}")

    def _build_payload(self, sr_u8, lr_u8, key_info=None):
        payload_bytes, payload_meta = self._build_payload_bytes(sr_u8, lr_u8, key_info=key_info)
        payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
        return payload_bits.astype(np.uint8), payload_meta

    def _analyze_carrier(self, sr_u8, payload_bits):
        sr_u8 = _validate_hwc_uint8(sr_u8, "sr_u8")
        if self.carrier_type == PIXEL_LSB_CARRIER:
            total_carriers = int(sr_u8.size) * self.bitplanes
            if self.carrier_fraction == "auto":
                capacity_bits = total_carriers
            else:
                capacity_bits = int(total_carriers * self.carrier_fraction)
            usable_ratio = 0.0 if total_carriers <= 0 else float(capacity_bits / max(total_carriers, 1))
            return {
                "carrier_backend": "pixel_lsb",
                "carrier_subbands": None,
                "spill_used": False,
                "carrier_capacity_bits": int(capacity_bits),
                "usable_coeff_ratio": float(usable_ratio),
                "total_coeffs": int(total_carriers),
            }

        analysis = self.carrier.analyze(sr_u8, int(payload_bits))
        return {
            "carrier_backend": self.carrier.backend_name,
            "carrier_subbands": list(analysis.active_bands),
            "spill_used": bool(analysis.spill_used),
            "carrier_capacity_bits": int(analysis.carrier_capacity_bits),
            "usable_coeff_ratio": float(analysis.usable_coeff_ratio),
            "total_coeffs": int(analysis.total_coeffs),
        }

    def analyze_payload(self, sr_u8, lr_u8):
        payload_bytes, payload_meta = self._build_payload_bytes(sr_u8, lr_u8, key_info=None)
        payload_bits = int(len(payload_bytes) * 8)
        carrier_stats = self._analyze_carrier(sr_u8, payload_bits)
        stats = {
            "mode": self.mode,
            "packet_mode": self.packet_mode,
            "carrier_type": self.carrier_type,
            "payload_bits": payload_bits,
            "raw_bits": int(payload_meta["raw_bits"]),
            "compressed_bits": int(payload_meta["compressed_bits"]),
            "payload_savings": int(payload_meta["payload_savings"]),
            "lr_shape": tuple(payload_meta["lr_shape"]),
            "crc32": int(payload_meta["crc32"]),
            "header_version": int(payload_meta["header_version"]),
            "analysis_levels": int(payload_meta.get("analysis_levels", 0)),
            "compressed_len": int(payload_meta.get("compressed_len", 0)),
            "predictor_id": payload_meta.get("predictor_id", None),
            "predictor_hash": payload_meta.get("predictor_hash", None),
            "pack_code": payload_meta.get("pack_code", None),
            "ll2_shape": payload_meta.get("ll2_shape", None),
            "carrier_capacity_bits": int(carrier_stats["carrier_capacity_bits"]),
            "carrier_fill_ratio": 0.0
            if int(carrier_stats["carrier_capacity_bits"]) <= 0
            else float(payload_bits / max(int(carrier_stats["carrier_capacity_bits"]), 1)),
            "usable_coeff_ratio": float(carrier_stats["usable_coeff_ratio"]),
            "spill_used": bool(carrier_stats["spill_used"]),
            "carrier_backend": carrier_stats["carrier_backend"],
            "carrier_subbands": carrier_stats["carrier_subbands"],
        }
        return stats

    def _make_carrier_indices(self, sr_u8, payload_bits, perm_seed):
        total_carriers = int(sr_u8.size) * self.bitplanes
        if payload_bits > total_carriers:
            raise ValueError(
                f"Payload needs {payload_bits} carriers but image only has {total_carriers}"
            )

        if self.carrier_fraction != "auto":
            max_allowed = int(total_carriers * self.carrier_fraction)
            if payload_bits > max_allowed:
                raise ValueError(
                    f"Payload needs {payload_bits} carriers but carrier_fraction only allows {max_allowed}"
                )

        rng = np.random.default_rng(int(perm_seed))
        carriers = rng.permutation(total_carriers)[:payload_bits]
        flat_indices = carriers // self.bitplanes
        plane_indices = carriers % self.bitplanes
        return flat_indices.astype(np.int64), plane_indices.astype(np.int64)

    def _embed_payload_bytes(self, sr_u8, payload_bytes, key_info=None):
        sr_u8 = _validate_hwc_uint8(sr_u8, "sr_u8")
        payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8)).astype(np.uint8)
        if self.carrier_type == PIXEL_LSB_CARRIER:
            if key_info is None or "perm_seed" not in key_info:
                perm_seed = int(np.random.randint(0, 2**31 - 1))
            else:
                perm_seed = int(key_info["perm_seed"])

            flat_indices, plane_indices = self._make_carrier_indices(sr_u8, payload_bits.size, perm_seed)
            stego_flat = sr_u8.reshape(-1).copy()
            for plane in range(self.bitplanes):
                plane_mask = plane_indices == plane
                if not np.any(plane_mask):
                    continue
                idx = flat_indices[plane_mask]
                bits = payload_bits[plane_mask].astype(np.uint8)
                clear_mask = np.uint8(0xFF ^ (1 << plane))
                stego_flat[idx] = (stego_flat[idx] & clear_mask) | (bits << plane)
            return stego_flat.reshape(sr_u8.shape), {
                "perm_seed": int(perm_seed),
                "carrier_backend": "pixel_lsb",
                "carrier_subbands": None,
                "spill_used": False,
                "carrier_capacity_bits": int(sr_u8.size) * self.bitplanes
                if self.carrier_fraction == "auto"
                else int(int(sr_u8.size) * self.bitplanes * float(self.carrier_fraction)),
                "usable_coeff_ratio": 1.0,
            }

        carrier_seed = self._resolve_carrier_seed(key_info=key_info)
        return self.carrier.embed(sr_u8, payload_bytes, seed=carrier_seed)

    def embed(self, sr_u8, lr_u8, key_info=None):
        sr_u8 = _validate_hwc_uint8(sr_u8, "sr_u8")
        payload_bytes, payload_meta = self._build_payload_bytes(sr_u8, lr_u8, key_info=key_info)
        payload_bits = int(len(payload_bytes) * 8)
        base_seed = self._resolve_carrier_seed(key_info=key_info)
        last_error = None
        for retry_idx in range(self.embed_retry_trials):
            retry_key_info = {} if key_info is None else dict(key_info)
            if self.carrier_type == IHAAR_H1STC_CARRIER:
                retry_key_info["carrier_seed"] = int(base_seed + retry_idx * 7919)
            stego_u8, carrier_info = self._embed_payload_bytes(sr_u8, payload_bytes, key_info=retry_key_info)

            keys_info = {
                "mode": self.mode,
                "packet_mode": self.packet_mode,
                "carrier_type": self.carrier_type,
                "bitplanes": self.bitplanes,
                "carrier_fraction": self.carrier_fraction,
                "use_chacha": self.use_chacha,
                "payload_bits": payload_bits,
                "lr_shape": payload_meta["lr_shape"],
                "crc32": int(payload_meta["crc32"]),
                "header_version": int(payload_meta["header_version"]),
                "raw_bits": int(payload_meta["raw_bits"]),
                "compressed_bits": int(payload_meta["compressed_bits"]),
                "payload_savings": int(payload_meta["payload_savings"]),
                "analysis_levels": int(payload_meta.get("analysis_levels", 0)),
                "compressed_len": int(payload_meta.get("compressed_len", 0)),
                "predictor_id": payload_meta.get("predictor_id", None),
                "predictor_hash": payload_meta.get("predictor_hash", None),
                "pack_code": payload_meta.get("pack_code", None),
                "carrier_backend": carrier_info.get("carrier_backend", None),
                "carrier_subbands": carrier_info.get("carrier_subbands", None),
                "spill_used": bool(carrier_info.get("spill_used", False)),
                "carrier_capacity_bits": int(carrier_info.get("carrier_capacity_bits", payload_bits)),
                "usable_coeff_ratio": float(carrier_info.get("usable_coeff_ratio", 1.0)),
            }
            if payload_meta.get("ll2_shape", None) is not None:
                keys_info["ll2_shape"] = payload_meta["ll2_shape"]
            if "perm_seed" in carrier_info:
                keys_info["perm_seed"] = int(carrier_info["perm_seed"])
            if "carrier_seed" in carrier_info:
                keys_info["carrier_seed"] = int(carrier_info["carrier_seed"])
            if self.use_chacha:
                keys_info["enc_key"] = payload_meta["enc_key"]
                keys_info["nonce"] = payload_meta["nonce"]
            else:
                keys_info["xor_seed"] = int(payload_meta["xor_seed"])

            if self.carrier_type == IHAAR_H1STC_CARRIER:
                self_check = self.extract(stego_u8, keys_info)
                if self_check.get("crc_ok", False):
                    return stego_u8, keys_info
                last_error = ValueError(f"h1stc_self_check_failed:{self_check.get('reason', 'unknown')}")
                continue
            return stego_u8, keys_info

        if last_error is not None:
            raise last_error
        raise RuntimeError("LR storage embedding failed without a recorded error")

    def _extract_payload_bytes(self, sr_u8, key_info):
        sr_u8 = _validate_hwc_uint8(sr_u8, "sr_u8")
        if self.carrier_type == PIXEL_LSB_CARRIER:
            payload_bits = int(key_info["payload_bits"])
            flat_indices, plane_indices = self._make_carrier_indices(
                sr_u8, payload_bits, int(key_info["perm_seed"])
            )

            flat = sr_u8.reshape(-1)
            extracted_bits = np.zeros(payload_bits, dtype=np.uint8)
            for plane in range(int(key_info.get("bitplanes", self.bitplanes))):
                plane_mask = plane_indices == plane
                if not np.any(plane_mask):
                    continue
                idx = flat_indices[plane_mask]
                extracted_bits[plane_mask] = (flat[idx] >> plane) & 1

            payload_bytes = np.packbits(extracted_bits).tobytes()
            return payload_bytes, {
                "carrier_backend": "pixel_lsb",
                "carrier_subbands": None,
                "spill_used": False,
            }

        payload_num_bytes = (int(key_info["payload_bits"]) + 7) // 8
        carrier_seed = self._resolve_carrier_seed(key_info=key_info)
        carrier_subbands = key_info.get("carrier_subbands", None)
        return self.carrier.extract(
            sr_u8,
            payload_num_bytes=payload_num_bytes,
            seed=carrier_seed,
            carrier_subbands=carrier_subbands,
        )

    def _extract_raw(self, payload_bytes, key_info):
        header_size = self.RAW_HEADER_STRUCT.size
        if len(payload_bytes) < header_size:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": None,
                "reason": "payload_too_short",
            }

        try:
            magic, version, flags, h, w, c, payload_len, crc32 = self.RAW_HEADER_STRUCT.unpack(
                payload_bytes[:header_size]
            )
        except struct.error:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": None,
                "reason": "header_unpack_failed",
            }

        header = {
            "magic": magic,
            "version": int(version),
            "flags": int(flags),
            "lr_shape": (int(h), int(w), int(c)),
            "payload_len": int(payload_len),
            "crc32": int(crc32),
            "packet_mode": RAW_PACKET_MODE,
            "mode": self.mode,
        }
        if magic != self.MAGIC or version != self.RAW_VERSION:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": header,
                "reason": "bad_magic_or_version",
            }

        encrypted_bytes = payload_bytes[header_size:header_size + payload_len]
        if len(encrypted_bytes) != payload_len:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": header,
                "reason": "payload_length_mismatch",
            }

        raw_bytes = self._decrypt_bytes(encrypted_bytes, key_info)
        crc_ok = (zlib.crc32(raw_bytes) & 0xFFFFFFFF) == crc32
        expected_values = int(h) * int(w) * int(c)
        if len(raw_bytes) != expected_values:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": header,
                "reason": "decoded_length_mismatch",
            }

        lr_u8 = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(int(h), int(w), int(c)).copy()
        return {
            "lr_u8": lr_u8,
            "crc_ok": bool(crc_ok),
            "header": header,
            "reason": None if crc_ok else "crc_mismatch",
        }

    def _extract_predictive(self, sr_u8, payload_bytes, key_info, ll2_override=None):
        header_size = self.PREDICTIVE_HEADER_STRUCT.size
        if len(payload_bytes) < header_size:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": None,
                "reason": "payload_too_short",
            }

        try:
            magic, version, flags, analysis_levels, pack_code, h, w, c, payload_len, crc32, predictor_hash = (
                self.PREDICTIVE_HEADER_STRUCT.unpack(payload_bytes[:header_size])
            )
        except struct.error:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": None,
                "reason": "header_unpack_failed",
            }

        header = {
            "magic": magic,
            "version": int(version),
            "flags": int(flags),
            "analysis_levels": int(analysis_levels),
            "pack_code": int(pack_code),
            "lr_shape": (int(h), int(w), int(c)),
            "payload_len": int(payload_len),
            "crc32": int(crc32),
            "predictor_hash": int(predictor_hash),
            "packet_mode": PREDICTIVE_PACKET_MODE,
            "mode": self.mode,
        }
        if magic != self.MAGIC or version != self.PREDICTIVE_VERSION:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": header,
                "reason": "bad_magic_or_version",
            }
        if pack_code not in self.PACK_CODE_TO_NAME:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": header,
                "reason": "unsupported_pack_code",
            }
        if int(analysis_levels) != int(self.analysis_levels):
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": header,
                "reason": "analysis_levels_mismatch",
            }
        if self.predictor is None:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": header,
                "reason": "missing_predictor",
            }
        if int(predictor_hash) != int(self.predictor_hash):
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": header,
                "reason": "predictor_mismatch",
            }

        encrypted_bytes = payload_bytes[header_size:header_size + payload_len]
        if len(encrypted_bytes) != payload_len:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": header,
                "reason": "payload_length_mismatch",
            }

        try:
            compressed_bytes = self._decrypt_bytes(encrypted_bytes, key_info)
            residual_bytes = zlib.decompress(compressed_bytes)
            residual = self._unpack_predictive_residual(residual_bytes, (int(h), int(w), int(c)))
            pred_u8, _ = self._predict_lr_u8(sr_u8, ll2_override=ll2_override)
        except ValueError as exc:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": header,
                "reason": str(exc),
            }
        except zlib.error:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": header,
                "reason": "decompression_failed",
            }

        lr_i32 = pred_u8.astype(np.int32) + residual.astype(np.int32)
        if np.min(lr_i32) < 0 or np.max(lr_i32) > 255:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": header,
                "reason": "residual_out_of_range",
            }

        lr_u8 = lr_i32.astype(np.uint8)
        raw_bytes = lr_u8.tobytes()
        crc_ok = (zlib.crc32(raw_bytes) & 0xFFFFFFFF) == crc32
        return {
            "lr_u8": lr_u8,
            "crc_ok": bool(crc_ok),
            "header": header,
            "reason": None if crc_ok else "crc_mismatch",
        }

    def _decode_payload_bytes(self, sr_u8, payload_bytes, key_info, carrier_info=None):
        if len(payload_bytes) < 5:
            return {
                "lr_u8": None,
                "crc_ok": False,
                "header": None,
                "reason": "payload_too_short",
            }

        version = int(payload_bytes[4])
        if version == self.RAW_VERSION:
            return self._extract_raw(payload_bytes, key_info)
        if version == self.PREDICTIVE_VERSION:
            ll2_override = None if carrier_info is None else carrier_info.get("ll2_u8", None)
            return self._extract_predictive(sr_u8, payload_bytes, key_info, ll2_override=ll2_override)
        return {
            "lr_u8": None,
            "crc_ok": False,
            "header": {
                "magic": payload_bytes[:4],
                "version": version,
            },
            "reason": "unknown_payload_version",
        }

    def extract(self, sr_u8, key_info):
        sr_u8 = _validate_hwc_uint8(sr_u8, "sr_u8")
        if key_info is None:
            raise ValueError("key_info is required for extraction")

        payload_bytes, carrier_info = self._extract_payload_bytes(sr_u8, key_info)
        return self._decode_payload_bytes(sr_u8, payload_bytes, key_info, carrier_info=carrier_info)
