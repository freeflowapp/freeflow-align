"""Vocal separation before alignment: Mel-Band RoFormer (vocals), pinned checkpoint.

The model class is lucidrains' BS-RoFormer 0.3.10 from PyPI (MIT), the release whose layers match the checkpoint
exactly. The checkpoint is Kimberley Jensen's MelBandRoformer.ckpt (huggingface.co/KimberleyJSN/melbandroformer,
commit ac9b0614), baked into the image at build time. Its SHA-256 is checked before loading, and it is loaded as plain
tensors (torch weights_only=True) with a strict key match: nothing in the file is executed, and no code from the
model's own repository is used.

Weights licence: "mit" in the Hugging Face metadata; the model card is empty and the training data is not stated.
"""

from __future__ import annotations

import hashlib
import os
import subprocess

import numpy as np

def _checkpoint_path() -> str:
    """The baked-in checkpoint (/weights, from cog.yaml), else ALIGN_SEPARATOR_CKPT, else weights/ in the repo."""
    for p in (os.environ.get("ALIGN_SEPARATOR_CKPT"), "/weights/MelBandRoformer.ckpt",
              os.path.join(os.path.dirname(__file__), "..", "weights", "MelBandRoformer.ckpt")):
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("MelBandRoformer.ckpt not found (expected /weights/ in the image)")
CHECKPOINT_SHA256 = "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e"
MODEL = "KimberleyJSN/melbandroformer@ac9b0614 on BS-RoFormer 0.3.10"
WEIGHTS_LICENSE = "MIT (HF metadata; training data unverified)"
SAMPLE_RATE = 44100
CHUNK = 352800  # 8 s, the length the model was trained on

# From the model's config_vocals_mel_band_roformer.yaml.
CONFIG = dict(
    dim=384, depth=6, stereo=True, num_stems=1, time_transformer_depth=1, freq_transformer_depth=1, num_bands=60,
    dim_head=64, heads=8, attn_dropout=0, ff_dropout=0, flash_attn=True, dim_freqs_in=1025, sample_rate=SAMPLE_RATE,
    stft_n_fft=2048, stft_hop_length=441, stft_win_length=2048, stft_normalized=False, mask_estimator_depth=2,
    multi_stft_resolution_loss_weight=1.0, multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
    multi_stft_hop_size=147, multi_stft_normalized=False,
)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Separator:
    def __init__(self, device: str = "cpu", checkpoint: str | None = None) -> None:
        import torch
        from bs_roformer import MelBandRoformer

        checkpoint = checkpoint or _checkpoint_path()
        digest = sha256(checkpoint)
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError(f"separator checkpoint hash {digest} does not match the pinned {CHECKPOINT_SHA256}")
        model = MelBandRoformer(**CONFIG)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.torch = torch
        self.device = device
        self.model = model.to(device).eval()

    def vocals(self, path: str, sample_rate: int) -> np.ndarray:
        """The vocal stem of the audio at ``path``, mono float32 at ``sample_rate``."""
        stereo = _decode_stereo(path)
        n = stereo.shape[1]
        hop = CHUNK // 2
        win = np.hanning(CHUNK).astype(np.float32) + 1e-3
        out = np.zeros_like(stereo)
        wsum = np.zeros(n, dtype=np.float32)
        with self.torch.inference_mode():
            for s in range(0, n, hop):
                seg = stereo[:, s : s + CHUNK]
                length = seg.shape[1]
                if length < CHUNK:
                    seg = np.pad(seg, ((0, 0), (0, CHUNK - length)))
                y = self.model(self.torch.from_numpy(np.ascontiguousarray(seg))[None].to(self.device))[0].float().cpu().numpy()
                out[:, s : s + length] += y[:, :length] * win[:length]
                wsum[s : s + length] += win[:length]
        vox = (out / np.maximum(wsum, 1e-6)).mean(axis=0).astype(np.float32)
        return _resample(vox, SAMPLE_RATE, sample_rate)


def _decode_stereo(path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-ac", "2", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    wave = np.frombuffer(raw, dtype=np.float32)
    if wave.size == 0:
        raise ValueError("audio decoded to zero samples")
    return wave.reshape(-1, 2).T.copy()


def _resample(wave: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return wave
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "f32le", "-ac", "1", "-ar", str(src), "-i", "-",
           "-ar", str(dst), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, input=wave.tobytes(), check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()
