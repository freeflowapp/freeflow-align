"""Forced alignment of known text to audio."""

from __future__ import annotations

import numpy as np

from . import audio, backends
from .run import MODES, align_transcript, parse_transcript

__all__ = ["align_file", "MODES"]


def align_file(
    path: str,
    transcript: str,
    aligner: str = backends.DEFAULT,
    mode: str = "segments",
    padding: float = 1.5,
    device: str | None = None,
    backend: backends.Backend | None = None,
) -> dict:
    """Align ``transcript`` (JSON or plain text) to the audio at ``path``."""
    segments = parse_transcript(transcript)
    if backend is None:
        backend = backends.get(aligner)
        backend.load(device or _default_device())
    wave = audio.load(path, backend.sample_rate)
    results = align_transcript(wave, backend, segments, mode, padding)
    if backend.time_offset:
        for seg in results:
            for w in seg["words"]:
                for t in [w] + ([w["whole"]] if "whole" in w else []):
                    for k in ("start", "end"):
                        if t[k] is not None:
                            t[k] = round(max(0.0, t[k] + backend.time_offset), 3)
    return {
        "aligner": backend.name,
        "model": backend.model_id,
        "weights_license": backend.weights_license,
        "mode": mode,
        "duration": round(len(wave) / backend.sample_rate, 3),
        "frame_seconds": backend.frame_seconds,
        "segments": results,
    }


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
