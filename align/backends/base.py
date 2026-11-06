"""What every acoustic model backend provides."""

from __future__ import annotations

import abc

import numpy as np


class Backend(abc.ABC):
    #: name used in the ``aligner`` input
    name: str
    #: model card / weights identifier, echoed in the output
    model_id: str
    #: licence of the weights, echoed in the output
    weights_license: str
    sample_rate: int = 16000
    #: seconds per output frame
    frame_seconds: float
    #: samples per output frame (frame_seconds * sample_rate)
    frame_samples: int
    #: id of the CTC blank token
    blank: int
    #: seconds of audio per forward pass; None runs the whole file at once
    chunk_seconds: float | None = None
    #: extra audio fed on each side of a chunk, then discarded
    context_seconds: float = 2.0
    #: shift added to every reported time, to correct the model's systematic lag/lead
    time_offset: float = 0.0

    @abc.abstractmethod
    def load(self, device: str) -> None: ...

    @abc.abstractmethod
    def forward(self, wave: np.ndarray) -> np.ndarray:
        """Log probs [frames, vocab] for a mono float32 waveform at ``sample_rate``."""

    @abc.abstractmethod
    def tokenize(self, word: str) -> list[int]:
        """Token ids for one word; empty when the model has no way to hear it."""

    def emissions(self, wave: np.ndarray) -> np.ndarray:
        """Log probs for the whole file, chunked so long audio fits in memory.

        Frame ``i`` always refers to sample ``i * frame_samples``, whether the
        file ran in one pass or many.
        """
        if self.chunk_seconds is None:
            return self.forward(wave)
        fs = self.frame_samples
        chunk = int(self.chunk_seconds * self.sample_rate) // fs * fs
        ctx = int(self.context_seconds * self.sample_rate) // fs * fs
        n = len(wave)
        if n <= chunk + 2 * ctx:
            return self.forward(wave)
        total = n // fs
        parts: list[np.ndarray] = []
        for core_start in range(0, n, chunk):
            core_end = min(n, core_start + chunk)
            pad_start = max(0, core_start - ctx)
            pad_end = min(n, core_end + ctx)
            lp = self.forward(wave[pad_start:pad_end])
            first = core_start // fs - pad_start // fs
            last = first + (core_end - core_start + fs - 1) // fs
            piece = lp[first:min(last, lp.shape[0])]
            parts.append(piece)
        return np.concatenate(parts)[:total]
