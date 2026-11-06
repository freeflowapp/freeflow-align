"""Turn a transcript plus frame log probs into word timings.

Transcript format (JSON):

    {"segments": [{"id": "a", "text": "some words here", "start": 12.0, "end": 30.5}, ...]}

``start`` and ``end`` are optional hints in seconds. A plain string is treated
as a single segment with no hints.

Modes:
  * ``segments``: each segment is aligned inside its own hinted window (plus
    padding). Segments without hints use the whole-audio pass.
  * ``whole``: hints are ignored; the full transcript is aligned in one pass.
  * ``both``: ``segments`` timings, and each word also carries the ``whole``
    timings so callers can compare the two passes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import numpy as np

from .ctc import AlignmentError, forced_align

MODES = ("segments", "whole", "both")


@dataclass
class Segment:
    id: str
    text: str
    start: float | None = None
    end: float | None = None
    words: list[str] = field(default_factory=list)


def parse_transcript(raw: str) -> list[Segment]:
    raw = raw.strip()
    if not raw:
        raise ValueError("transcript is empty")
    if raw[0] in "{[":
        data = json.loads(raw)
        items = data["segments"] if isinstance(data, dict) else data
        segments = []
        for i, item in enumerate(items):
            start, end = item.get("start"), item.get("end")
            if start is not None and end is not None and end <= start:
                raise ValueError(f"segment {item.get('id', i)}: end must be after start")
            segments.append(
                Segment(
                    id=str(item.get("id", i)),
                    text=str(item.get("text", "")),
                    start=None if start is None else float(start),
                    end=None if end is None else float(end),
                )
            )
    else:
        segments = [Segment(id="0", text=raw)]
    for seg in segments:
        seg.words = seg.text.split()
    return segments


Timing = dict  # {"start": float | None, "end": float | None, "score": float | None}
EMPTY: Timing = {"start": None, "end": None, "score": None}


def _align_words(
    log_probs: np.ndarray,
    offset: float,
    frame_seconds: float,
    word_tokens: list[list[int]],
    blank: int,
) -> list[Timing]:
    """Align a list of words (each a token list, possibly empty) to the frames."""
    flat: list[int] = []
    owner: list[int] = []
    for w, toks in enumerate(word_tokens):
        flat.extend(toks)
        owner.extend([w] * len(toks))
    out: list[Timing] = [dict(EMPTY) for _ in word_tokens]
    if not flat:
        return out
    spans = forced_align(log_probs, flat, blank=blank)
    for w in range(len(word_tokens)):
        mine = [s for s, o in zip(spans, owner) if o == w]
        if not mine:
            continue
        out[w] = {
            "start": round(offset + mine[0].start * frame_seconds, 3),
            "end": round(offset + mine[-1].end * frame_seconds, 3),
            "score": round(float(np.mean([s.score for s in mine])), 4),
        }
    return out


def align_transcript(
    wave: np.ndarray,
    backend,
    segments: list[Segment],
    mode: str = "segments",
    padding: float = 1.5,
) -> list[dict]:
    """Align every segment's words to ``wave`` (mono float32 at backend.sample_rate).

    The whole-file pass runs the model once over the full audio. Each windowed
    segment runs the model again on just its window, so the two passes are
    independent opinions and their disagreement means something.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    sr = backend.sample_rate
    fsec = backend.frame_seconds
    tokens = [[backend.tokenize(w) for w in seg.words] for seg in segments]

    need_whole = mode in ("whole", "both") or any(s.start is None or s.end is None for s in segments)
    whole: list[list[Timing]] | None = None
    whole_error: str | None = None
    if need_whole:
        flat_words = [t for seg_tokens in tokens for t in seg_tokens]
        try:
            timings = _align_words(backend.emissions(wave), 0.0, fsec, flat_words, backend.blank)
        except AlignmentError as e:
            timings = [dict(EMPTY) for _ in flat_words]
            whole_error = str(e)
        whole, i = [], 0
        for seg_tokens in tokens:
            whole.append(timings[i : i + len(seg_tokens)])
            i += len(seg_tokens)

    results = []
    for k, seg in enumerate(segments):
        error = None
        hinted = seg.start is not None and seg.end is not None
        if mode == "whole" or not hinted:
            primary = whole[k]
            error = whole_error
        else:
            lo = max(0, int((seg.start - padding) * sr))
            hi = min(len(wave), int((seg.end + padding) * sr))
            try:
                if hi <= lo:
                    raise AlignmentError("window is outside the audio")
                lp = backend.emissions(wave[lo:hi])
                primary = _align_words(lp, lo / sr, fsec, tokens[k], backend.blank)
            except AlignmentError as e:
                primary = [dict(EMPTY) for _ in tokens[k]]
                error = str(e)
        words = []
        for j, text in enumerate(seg.words):
            word = {"text": text, **primary[j]}
            if not tokens[k][j]:
                word["skipped"] = True  # nothing the model can hear (e.g. digits only)
            if mode == "both":
                word["whole"] = whole[k][j]
            words.append(word)
        entry = {"id": seg.id, "status": "failed" if error else "ok", "words": words}
        if error:
            entry["error"] = error
        results.append(entry)
    return results
