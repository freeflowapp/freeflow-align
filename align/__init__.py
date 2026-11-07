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
    separate: bool = False,
    separator=None,
) -> dict:
    """Align ``transcript`` (JSON or plain text) to the audio at ``path``.

    With ``separate``, the vocal is also separated from the audio (``align.separate``) and the same passes run on it:
    each word then carries ``stem`` (its timings on the vocal stem, with ``whole`` in mode ``both``), while its own
    timings stay those of the mix, exactly as without ``separate``.
    """
    segments = parse_transcript(transcript)
    if backend is None:
        backend = backends.get(aligner)
        backend.load(device or _default_device())
    wave = audio.load(path, backend.sample_rate)
    results = align_transcript(wave, backend, segments, mode, padding)
    _shift(results, backend.time_offset)
    out = {
        "aligner": backend.name,
        "model": backend.model_id,
        "weights_license": backend.weights_license,
        "mode": mode,
        "duration": round(len(wave) / backend.sample_rate, 3),
        "frame_seconds": backend.frame_seconds,
        "segments": results,
    }
    if separate:
        from . import separate as sep

        if separator is None:
            separator = sep.Separator(device or _default_device())
        stem = separator.vocals(path, backend.sample_rate)
        stem_results = align_transcript(stem, backend, segments, mode, padding)
        _shift(stem_results, backend.time_offset)
        for seg, stem_seg in zip(results, stem_results):
            for w, sw in zip(seg["words"], stem_seg["words"]):
                w["stem"] = {k: sw[k] for k in ("start", "end", "score")}
                if "whole" in sw:
                    w["stem"]["whole"] = sw["whole"]
            if stem_seg["status"] != "ok":
                seg["stem_status"] = stem_seg["status"]
                seg["stem_error"] = stem_seg.get("error")
        out["separator"] = {"model": sep.MODEL, "checkpoint_sha256": sep.CHECKPOINT_SHA256, "weights_license": sep.WEIGHTS_LICENSE}
    return out


def _shift(results: list[dict], offset: float) -> None:
    """Apply the backend's fixed time offset to every timing, in place."""
    if not offset:
        return
    for seg in results:
        for w in seg["words"]:
            for t in [w] + ([w["whole"]] if "whole" in w else []):
                for k in ("start", "end"):
                    if t[k] is not None:
                        t[k] = round(max(0.0, t[k] + offset), 3)


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
