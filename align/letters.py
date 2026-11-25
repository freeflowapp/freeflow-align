"""The letters pass: when each letter of a placed word starts, heard by a fine character model.

The word aligner (parakeet, 80 ms frames) places whole words. A word of several syllables then has one time, and the
caller can only spread its syllables evenly across it. The letters pass listens again inside each placed word, with
a character model on 20 ms frames (``backends.wav2vec2``), and reports when each of the word's letters starts. The
caller cuts the word into syllables its own way and starts each syllable at its first letter's time.

Measured before it was built (Freeflow's fine-listening test, 2026-11-25; 13 songs, syllables a person tapped, songs
split in two by a hash and scored on the half not used for tuning): syllables inside the words parakeet placed, within
50 ms of the tap, 33% for the even split and 60% from the letters' times. It does not help word starts, and it does not
tell how many syllables a word is sung on.

Each word is aligned alone, inside its own placed time plus ``LETTER_PADDING`` on each side, on frames computed once
for the whole file. Letters are the word's a-z and internal apostrophes, lower case (``letters_of``): the caller must
cut them out of the word the same way. A word with nothing placed, nothing to hear, or too little time for its letters
gets ``None``.
"""

from __future__ import annotations

import math

import numpy as np

from .backends.wav2vec2 import letters_of
from .ctc import AlignmentError, forced_align

#: Seconds of audio either side of a word's placed time that its letters may fall in (the value the test used).
LETTER_PADDING = 0.15


def _bounds(word: dict) -> tuple[float, float] | None:
    """The time the letters are looked for in: the vocal stem's own pass where it placed the word, else the mix's."""
    for t in (word.get("stem"), word):
        if t and t.get("start") is not None:
            end = t.get("end")
            return t["start"], end if end is not None and end > t["start"] else t["start"]
    return None


def word_letters(
    log_probs: np.ndarray,
    frame_seconds: float,
    text: str,
    bounds: tuple[float, float],
    tokenize,
    blank: int,
    padding: float = LETTER_PADDING,
) -> list[float] | None:
    """Start times (seconds into the file) of each of ``letters_of(text)``, aligned inside ``bounds`` ± ``padding``."""
    letters = letters_of(text)
    tokens = tokenize(text)
    if not letters or len(tokens) != len(letters):
        return None
    lo = max(0, int(math.floor((bounds[0] - padding) / frame_seconds)))
    hi = min(log_probs.shape[0], int(math.ceil((bounds[1] + padding) / frame_seconds)))
    if hi <= lo:
        return None
    try:
        spans = forced_align(log_probs[lo:hi], tokens, blank=blank)
    except AlignmentError:
        return None
    return [round((lo + s.start) * frame_seconds, 3) for s in spans]


def add_letters(results: list[dict], log_probs: np.ndarray, backend, time_offset: float = 0.0) -> None:
    """Give every word in ``results`` its ``letters`` (start times, or None), in place."""
    for seg in results:
        for w in seg["words"]:
            bounds = None if w.get("skipped") else _bounds(w)
            times = None if bounds is None else word_letters(
                log_probs, backend.frame_seconds, w["text"], bounds, backend.tokenize, backend.blank,
            )
            if times is not None and time_offset:
                times = [round(max(0.0, t + time_offset), 3) for t in times]
            w["letters"] = times
