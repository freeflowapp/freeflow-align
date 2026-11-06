"""Decode any audio file to mono float32 at a given rate, using ffmpeg."""

from __future__ import annotations

import subprocess

import numpy as np


def load(path: str, sample_rate: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
        "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    wave = np.frombuffer(raw, dtype=np.float32)
    if wave.size == 0:
        raise ValueError("audio decoded to zero samples")
    return wave
