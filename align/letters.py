"""Letter times inside each word, on the separated vocal: where each sung letter starts.

The word aligner (parakeet, 80 ms frames, word pieces) places words. Inside each word it placed, a character CTC model
with 20 ms frames (wav2vec 2.0 large, LV-60 self-training, fine-tuned on LibriSpeech 960 h) is forced to spell the word,
and each letter's first frame is reported. A client cuts a word into syllables and times each one at its letters.

Measured 2026-11-25 on 13 House of Bread songs (580 hand-timed syllables after a word's first; freeflow-ai-eval
`bench/`, run S2): syllables at their first vowel's time, kept only inside the heard word and in order, on the nearest
sixteenth: 33% -> 58% within 50 ms of the hand time on the songs held out from tuning (162 closer, 46 further), 40% ->
64% on all 13. The window (the word +- PAD) and the shift (TIME_OFFSET) were tuned on the other songs.

Weights: facebook/wav2vec2-large-960h-lv60-self, Apache-2.0.
"""

from __future__ import annotations

import numpy as np

from .backends.parakeet import normalize
from .ctc import AlignmentError, forced_align

MODEL = "facebook/wav2vec2-large-960h-lv60-self"
REVISION = "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"
WEIGHTS_LICENSE = "Apache-2.0 (Meta AI)"
SAMPLE_RATE = 16000
CHUNK = 30 * SAMPLE_RATE  # run as separate 30 s pieces, as measured
MIN_CHUNK = SAMPLE_RATE // 2
#: seconds around the word's own start and end that its letters may use
PAD = 0.1
#: added to every letter time: CTC fires late on a sung vowel (tuned; see the module note)
TIME_OFFSET = -0.12


class LetterModel:
    """wav2vec2 character CTC: log probs at 20 ms, and a word as letter ids."""

    frame_seconds = 0.02
    sample_rate = SAMPLE_RATE

    def load(self, device: str) -> None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self._torch = torch
        self._device = device
        self._proc = Wav2Vec2Processor.from_pretrained(MODEL, revision=REVISION)
        self._model = Wav2Vec2ForCTC.from_pretrained(MODEL, revision=REVISION).to(device).eval()
        self._vocab = self._proc.tokenizer.get_vocab()
        self.blank = self._proc.tokenizer.pad_token_id

    def emissions(self, wave: np.ndarray) -> tuple[np.ndarray, float]:
        """Log probs [frames, vocab] and seconds per frame (measured: chunk seconds / chunk frames)."""
        torch = self._torch
        wave = np.ascontiguousarray(wave, dtype=np.float32)
        outs, fs = [], None
        for a in range(0, max(1, len(wave)), CHUNK):
            x = wave[a : a + CHUNK]
            full = len(x) == CHUNK
            if len(x) < MIN_CHUNK:
                x = np.pad(x, (0, MIN_CHUNK - len(x)))
            inp = self._proc(x, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(self._device)
            with torch.inference_mode():
                lp = torch.log_softmax(self._model(inp).logits[0].float(), -1).cpu().numpy()
            outs.append(lp)
            if full and fs is None:
                fs = 30.0 / lp.shape[0]
        lp = np.concatenate(outs, 0)
        return lp, fs or len(wave) / SAMPLE_RATE / lp.shape[0]

    def spell(self, word: str) -> list[int]:
        """One id per letter of ``normalize(word)``; empty if any letter has no id (the times would not line up)."""
        ids = [self._vocab.get(ch) for ch in normalize(word).upper()]
        return [] if any(i is None for i in ids) else ids


def word_letters(lp: np.ndarray, fs: float, t0: float, ids: list[int], lo: float, hi: float, blank: int) -> list[float] | None:
    """Start time of each letter, aligned inside [lo, hi] (seconds) of emissions that begin at ``t0``; None if it can't fit."""
    if not ids:
        return None
    a = max(0, int(np.floor((lo - t0) / fs)))
    b = min(lp.shape[0], int(np.ceil((hi - t0) / fs)))
    if b - a < 2:
        return None
    try:
        spans = forced_align(lp[a:b].astype(np.float64), ids, blank=blank)
    except AlignmentError:
        return None
    return [round(max(0.0, t0 + (a + s.start) * fs + TIME_OFFSET), 3) for s in spans]


def add_letters(wave: np.ndarray, segments, results: list[dict], model, padding: float) -> None:
    """Give every word ``results`` placed (its own windowed times) a ``letters`` list, in place.

    One model pass per segment window (the segment's hint +- ``padding``, or the whole file when unhinted); each word is
    then spelled alone inside its own start/end +- PAD. A word that can't be spelled gets ``letters: null``.
    """
    sr = model.sample_rate
    whole = None
    for seg, res in zip(segments, results):
        hinted = seg.start is not None and seg.end is not None
        if hinted:
            lo = max(0, int((seg.start - padding) * sr))
            hi = min(len(wave), int((seg.end + padding) * sr))
            if hi <= lo:
                for w in res["words"]:
                    w["letters"] = None
                continue
            lp, fs = model.emissions(wave[lo:hi])
            t0 = lo / sr
        else:
            if whole is None:
                whole = model.emissions(wave)
            (lp, fs), t0 = whole, 0.0
        for text, w in zip(seg.words, res["words"]):
            if w.get("start") is None or w.get("end") is None:
                w["letters"] = None
                continue
            w["letters"] = word_letters(lp, fs, t0, model.spell(text), w["start"] - PAD, w["end"] + PAD, model.blank)
