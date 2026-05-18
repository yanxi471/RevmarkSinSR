"""
Watermark Codec for RevMark-SinSR.

Adapted from Gaussian-Shading/watermark.py to work with SinSR's 3-channel 64x64 latent space.
Key differences from original Gaussian Shading:
  - latent shape: (B, 3, 64, 64) instead of (1, 4, 64, 64)
  - Uses PyTorch erfinv for GPU-friendly batch quantile sampling
  - Supports batch encoding/decoding
  - ch_factor must divide 3 (use 1 or 3)
"""

import math
import numpy as np
import torch
from scipy.special import betainc
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes


class WatermarkCodec:
    """
    Gaussian Shading watermark encoder/decoder for SinSR latent space.

    Encodes a binary payload into Gaussian noise eps ~ N(0, I) via quantile mapping,
    preserving the standard normal marginal distribution.

    Args:
        num_channels: number of latent channels (3 for SinSR's VQ-f4 autoencoder)
        latent_size: spatial size of latent (64 for 256x256 GT with f=4)
        ch_factor: channel replication factor (must divide num_channels)
        hw_factor: spatial replication factor
        use_chacha: if True, use ChaCha20 encryption; else use XOR
        fpr: false positive rate for detection threshold
        user_number: number of users for traceability threshold
    """

    def __init__(
        self,
        num_channels=3,
        latent_size=64,
        ch_factor=1,
        hw_factor=8,
        use_chacha=True,
        fpr=1e-6,
        user_number=1,
    ):
        assert num_channels % ch_factor == 0, (
            f"ch_factor={ch_factor} must divide num_channels={num_channels}"
        )
        assert latent_size % hw_factor == 0, (
            f"hw_factor={hw_factor} must divide latent_size={latent_size}"
        )

        self.num_channels = num_channels
        self.latent_size = latent_size
        self.ch_factor = ch_factor
        self.hw_factor = hw_factor
        self.use_chacha = use_chacha

        self.latent_length = num_channels * latent_size * latent_size  # 3*64*64 = 12288
        self.mark_length = self.latent_length // (ch_factor * hw_factor * hw_factor)
        # ch=1, hw=8 -> 12288 / 64 = 192 bits
        # ch=3, hw=8 -> 12288 / 192 = 64 bits

        self.num_copies = ch_factor * hw_factor * hw_factor
        self.threshold = 1 if (hw_factor == 1 and ch_factor == 1) else self.num_copies // 2

        # Compute detection thresholds from FPR using incomplete beta function
        self.tau_onebit = None
        self.tau_bits = None
        for i in range(self.mark_length):
            fpr_onebit = betainc(i + 1, self.mark_length - i, 0.5)
            fpr_bits = fpr_onebit * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.mark_length
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.mark_length

        # Per-sample encryption keys (for batch support)
        self._keys = {}  # sample_id -> (key, nonce) or xor_key
        self._u_seeds = {}  # sample_id -> int seed for quantile sampling RNG
        self._messages = {}  # sample_id -> np.array of encrypted message bits

    def _spread(self, watermark):
        """
        Spread watermark bits by replication.
        watermark: (B, C//ch, H//hw, W//hw) -> (B, C, H, W)
        """
        return watermark.repeat(1, self.ch_factor, self.hw_factor, self.hw_factor)

    def _majority_vote(self, sd):
        """
        Inverse of spreading: split replicated tensor and majority-vote.
        sd: (B, C, H, W) -> (B, C//ch, H//hw, W//hw)
        """
        ch_stride = self.num_channels // self.ch_factor
        hw_stride = self.latent_size // self.hw_factor

        B = sd.shape[0]
        results = []
        for b in range(B):
            x = sd[b:b+1]  # (1, C, H, W)
            # Split along channel dim
            split_ch = torch.cat(torch.split(x, ch_stride, dim=1), dim=0)  # (ch_factor, ch_stride, H, W)
            # Split along height dim
            split_h = torch.cat(torch.split(split_ch, hw_stride, dim=2), dim=0)  # (ch*hw, ch_stride, hw_stride, W)
            # Split along width dim
            split_w = torch.cat(torch.split(split_h, hw_stride, dim=3), dim=0)  # (ch*hw*hw, ch_stride, hw_stride, hw_stride)

            vote = torch.sum(split_w, dim=0, keepdim=True).clone()  # (1, ch_stride, hw_stride, hw_stride)
            vote = (vote > self.threshold).int()
            results.append(vote)

        return torch.cat(results, dim=0)  # (B, C//ch, H//hw, W//hw)

    def _chacha_encrypt(self, sd_flat_np, sample_id=0):
        """Encrypt flat bit array with ChaCha20. Returns encrypted bits."""
        key = get_random_bytes(32)
        nonce = get_random_bytes(12)
        self._keys[sample_id] = (key, nonce)

        cipher = ChaCha20.new(key=key, nonce=nonce)
        m_byte = cipher.encrypt(np.packbits(sd_flat_np).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit[:self.latent_length]  # trim to exact length

    def _chacha_decrypt(self, m_flat_np, sample_id=0):
        """Decrypt flat bit array with ChaCha20. Returns decrypted bits."""
        key, nonce = self._keys[sample_id]
        cipher = ChaCha20.new(key=key, nonce=nonce)
        sd_byte = cipher.decrypt(np.packbits(m_flat_np).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        return sd_bit[:self.latent_length]

    def _xor_encrypt(self, sd_flat_np, sample_id=0):
        """XOR encryption with random key."""
        key = np.random.randint(0, 2, size=self.latent_length, dtype=np.uint8)
        self._keys[sample_id] = key
        return ((sd_flat_np.astype(np.uint8) + key) % 2).astype(np.uint8)

    def _xor_decrypt(self, m_flat_np, sample_id=0):
        """XOR decryption."""
        key = self._keys[sample_id]
        return ((m_flat_np.astype(np.uint8) + key) % 2).astype(np.uint8)

    def _quantile_sample_batch(self, messages, device='cuda'):
        """
        GPU-friendly quantile sampling for batch of encrypted message bits.

        For 1-bit quantile (denominator=2):
          bit=0 -> sample from (-inf, 0], i.e. Phi^{-1}(u/2) where u ~ U(0,1)
          bit=1 -> sample from [0, +inf), i.e. Phi^{-1}((u+1)/2) where u ~ U(0,1)

        Uses per-sample seeded RNG so u values are reproducible via stored seeds.

        messages: numpy array (B, latent_length), values in {0, 1}
        Returns: torch tensor (B, C, H, W) on device
        """
        B = messages.shape[0]
        msg_t = torch.from_numpy(messages.astype(np.float32)).to(device)  # (B, latent_length)

        # Generate u with per-sample reproducible seeds
        u_list = []
        for b in range(B):
            seed = torch.randint(0, 2**62, (1,)).item()
            self._u_seeds[b] = seed
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            u_list.append(torch.rand(1, self.latent_length, device=device, generator=gen))
        u = torch.cat(u_list, dim=0)  # (B, latent_length)

        # p = (u + message) / 2  maps to the correct half of [0,1]
        # bit=0 -> p in [0, 0.5), bit=1 -> p in [0.5, 1)
        p = (u + msg_t) / 2.0

        # Clamp to avoid inf from erfinv
        p = p.clamp(1e-6, 1.0 - 1e-6)

        # Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1)
        z = torch.erfinv(2.0 * p - 1.0) * math.sqrt(2.0)

        return z.reshape(B, self.num_channels, self.latent_size, self.latent_size)

    def encode_batch(self, batch_size, device='cuda', payloads=None):
        """
        Encode watermarked Gaussian noise for a batch.

        Args:
            batch_size: number of samples
            device: torch device
            payloads: optional (B, mark_length) tensor of pre-defined payloads.
                      If None, random payloads are generated.

        Returns:
            eps_wm: (B, C, H, W) watermarked noise ~ N(0, I)
            watermarks: (B, C//ch, H//hw, W//hw) the raw watermark bits
        """
        wm_ch = self.num_channels // self.ch_factor
        wm_h = self.latent_size // self.hw_factor
        wm_w = wm_h

        if payloads is not None:
            # Convert flat payload to spatial watermark shape
            watermarks = payloads.reshape(batch_size, wm_ch, wm_h, wm_w).int()
        else:
            watermarks = torch.randint(0, 2, (batch_size, wm_ch, wm_h, wm_w))

        # Spread, encrypt, quantile-sample per sample
        all_messages = np.zeros((batch_size, self.latent_length), dtype=np.uint8)

        for b in range(batch_size):
            sd = self._spread(watermarks[b:b+1])  # (1, C, H, W)
            sd_flat = sd.flatten().cpu().numpy().astype(np.uint8)

            if self.use_chacha:
                m = self._chacha_encrypt(sd_flat, sample_id=b)
            else:
                m = self._xor_encrypt(sd_flat, sample_id=b)

            all_messages[b] = m

        eps_wm = self._quantile_sample_batch(all_messages, device=device)

        # Store messages for noise reconstruction
        for b in range(batch_size):
            self._messages[b] = all_messages[b]

        return eps_wm, watermarks.to(device)

    def decode_batch(self, eps_hat):
        """
        Decode watermark from recovered noise.

        Args:
            eps_hat: (B, C, H, W) recovered noise tensor

        Returns:
            decoded_watermarks: (B, C//ch, H//hw, W//hw) decoded watermark bits
            bit_accuracies: list of per-sample bit accuracies (vs stored watermarks)
        """
        B = eps_hat.shape[0]
        device = eps_hat.device

        # Step 1: Binarize by sign (quantile decode)
        reversed_m = (eps_hat > 0).int()  # (B, C, H, W)

        decoded_watermarks = []
        bit_accuracies = []

        for b in range(B):
            m_flat = reversed_m[b].flatten().cpu().numpy().astype(np.uint8)

            # Step 2: Decrypt
            if self.use_chacha:
                sd_flat = self._chacha_decrypt(m_flat, sample_id=b)
            else:
                sd_flat = self._xor_decrypt(m_flat, sample_id=b)

            sd = torch.from_numpy(sd_flat.astype(np.float32)).reshape(
                1, self.num_channels, self.latent_size, self.latent_size
            ).to(device)

            # Step 3: Majority vote to recover watermark
            wm = self._majority_vote(sd.int())  # (1, C//ch, H//hw, W//hw)
            decoded_watermarks.append(wm)

        decoded_watermarks = torch.cat(decoded_watermarks, dim=0)
        return decoded_watermarks

    def decode_from_flat_ciphertext(self, m_flat_np, sample_id=0, device="cuda"):
        """
        从已构造的整幅密文比特向量解密（用于潜空间 ROI 混合：非 ROI 处不用 sign(eps_hat) 而用嵌入端保存的密文）。
        m_flat_np: (latent_length,) uint8 in {0,1}
        """
        m_flat_np = np.asarray(m_flat_np, dtype=np.uint8).reshape(-1)
        if m_flat_np.shape[0] != self.latent_length:
            raise ValueError(
                f"m_flat 长度 {m_flat_np.shape[0]} != latent_length {self.latent_length}"
            )
        if self.use_chacha:
            sd_flat = self._chacha_decrypt(m_flat_np, sample_id=sample_id)
        else:
            sd_flat = self._xor_decrypt(m_flat_np, sample_id=sample_id)
        sd = torch.from_numpy(sd_flat.astype(np.float32)).reshape(
            1, self.num_channels, self.latent_size, self.latent_size
        ).to(device)
        return self._majority_vote(sd.int())

    def compute_bit_accuracy(self, original_watermarks, decoded_watermarks):
        """
        Compute per-sample bit accuracy.

        Args:
            original_watermarks: (B, ...) original watermark tensor
            decoded_watermarks: (B, ...) decoded watermark tensor

        Returns:
            accuracies: (B,) tensor of bit accuracies per sample
            mean_accuracy: scalar mean accuracy
        """
        B = original_watermarks.shape[0]
        orig_flat = original_watermarks.reshape(B, -1).float()
        dec_flat = decoded_watermarks.reshape(B, -1).float()
        accuracies = (orig_flat == dec_flat).float().mean(dim=1)
        return accuracies, accuracies.mean().item()

    def compute_bit_accuracy_masked(
        self, original_watermarks, decoded_watermarks, mask
    ):
        """
        仅在 mask==1 的比特位置上统计准确率（用于 ROI 载荷：区外固定为 0）。

        Args:
            original_watermarks: (B, wm_ch, wm_h, wm_w)
            decoded_watermarks: 同形
            mask: (wm_h, wm_w) 或 (wm_ch, wm_h, wm_w) 或 (1,1,wm_h,wm_w)，>0.5 计入

        Returns:
            accuracies: (B,) 每样本 ROI 内准确率；若某样本 ROI 比特数为 0 则记为 1.0
            mean_accuracy: 标量均值
        """
        B = int(original_watermarks.shape[0])
        wm_ch = int(self.num_channels // self.ch_factor)
        wm_h = int(self.latent_size // self.hw_factor)
        wm_w = wm_h

        m = torch.as_tensor(mask, dtype=torch.float32, device=original_watermarks.device)
        if m.ndim == 2:
            m = m.view(1, 1, wm_h, wm_w).expand(B, wm_ch, wm_h, wm_w)
        elif m.ndim == 3:
            m = m.view(1, wm_ch, wm_h, wm_w).expand(B, wm_ch, wm_h, wm_w)
        elif m.ndim == 4:
            if m.shape[0] == 1 and B > 1:
                m = m.expand(B, -1, -1, -1)
        else:
            raise ValueError(f"mask 维度不支持: {tuple(m.shape)}")

        m_bin = m > 0.5
        orig_flat = original_watermarks.reshape(B, -1)
        dec_flat = decoded_watermarks.reshape(B, -1)
        m_flat = m_bin.reshape(B, -1)

        acc_list = []
        for b in range(B):
            sel = m_flat[b]
            n = int(sel.sum().item())
            if n == 0:
                acc_list.append(1.0)
            else:
                acc_list.append(
                    float((orig_flat[b, sel] == dec_flat[b, sel]).float().mean().item())
                )
        accuracies = torch.tensor(acc_list, device=original_watermarks.device)
        return accuracies, float(accuracies.mean().item())

    def get_bin_targets(self, device='cuda'):
        """
        Get the encrypted message bits for each sample in the current batch.
        Used to compute L_bin loss during training.

        Returns:
            dict mapping sample_id -> encrypted message tensor (latent_length,)
        """
        return self._current_messages

    def encode_batch_with_targets(self, batch_size, device='cuda', payloads=None):
        """
        Like encode_batch but also returns the encrypted message bits needed for L_bin.

        Returns:
            eps_wm: (B, C, H, W)
            watermarks: (B, C//ch, H//hw, W//hw)
            target_bins: (B, latent_length) encrypted message bits in {0, 1}
        """
        wm_ch = self.num_channels // self.ch_factor
        wm_h = self.latent_size // self.hw_factor
        wm_w = wm_h

        if payloads is not None:
            watermarks = payloads.reshape(batch_size, wm_ch, wm_h, wm_w).int()
        else:
            watermarks = torch.randint(0, 2, (batch_size, wm_ch, wm_h, wm_w))

        all_messages = np.zeros((batch_size, self.latent_length), dtype=np.uint8)

        for b in range(batch_size):
            sd = self._spread(watermarks[b:b+1])
            sd_flat = sd.flatten().cpu().numpy().astype(np.uint8)

            if self.use_chacha:
                m = self._chacha_encrypt(sd_flat, sample_id=b)
            else:
                m = self._xor_encrypt(sd_flat, sample_id=b)

            all_messages[b] = m

        eps_wm = self._quantile_sample_batch(all_messages, device=device)
        target_bins = torch.from_numpy(all_messages.astype(np.float32)).to(device)  # (B, latent_length)

        return eps_wm, watermarks.to(device), target_bins

    def encode_payload(self, payload_bits, device='cuda'):
        """
        Encode a specific payload into watermarked noise (single sample).

        Args:
            payload_bits: 1D tensor or array of length mark_length

        Returns:
            eps_wm: (1, C, H, W) watermarked noise
        """
        assert len(payload_bits) == self.mark_length
        payload = torch.tensor(payload_bits, dtype=torch.int32).reshape(
            1, self.num_channels // self.ch_factor,
            self.latent_size // self.hw_factor,
            self.latent_size // self.hw_factor
        )
        eps_wm, _ = self.encode_batch(1, device=device, payloads=payload.reshape(1, -1))
        return eps_wm

    def get_keys(self):
        """Return current encryption keys, u-seeds, and messages for serialization."""
        return {
            'enc_keys': dict(self._keys),
            'u_seeds': dict(self._u_seeds),
            'messages': {k: v.copy() for k, v in self._messages.items()},
        }

    def set_keys(self, keys):
        """Load encryption keys (for extraction). Backward compatible."""
        if isinstance(keys, dict) and 'enc_keys' in keys:
            self._keys = dict(keys['enc_keys'])
            self._u_seeds = dict(keys.get('u_seeds', {}))
            self._messages = {k: v.copy() for k, v in keys.get('messages', {}).items()}
        else:
            # Backward compatibility: old format was just enc_keys dict
            self._keys = dict(keys)
            self._u_seeds = {}
            self._messages = {}

    def reconstruct_noise(self, sample_id=0, device='cuda'):
        """
        Reconstruct exact epsilon^wm from stored u-seed and encrypted message.

        This allows precise z_y recovery: z_y = predicted_xT - sigma_T * eps_wm.

        Returns:
            eps_wm: (1, C, H, W) exact watermarked noise tensor
        """
        if sample_id not in self._u_seeds:
            raise ValueError(
                f"No u_seed stored for sample {sample_id}. "
                "Keys were saved without seed info (old format)."
            )

        m = self._messages[sample_id]  # (latent_length,) uint8
        seed = self._u_seeds[sample_id]

        msg_t = torch.from_numpy(m.astype(np.float32)).to(device).unsqueeze(0)  # (1, L)

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        u = torch.rand(1, self.latent_length, device=device, generator=gen)

        p = (u + msg_t) / 2.0
        p = p.clamp(1e-6, 1.0 - 1e-6)
        z = torch.erfinv(2.0 * p - 1.0) * math.sqrt(2.0)

        return z.reshape(1, self.num_channels, self.latent_size, self.latent_size)
