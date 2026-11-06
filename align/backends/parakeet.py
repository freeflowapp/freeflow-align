"""NVIDIA parakeet-ctc-0.6b (FastConformer CTC, SentencePiece BPE) through NeMo.

Weights: https://huggingface.co/nvidia/parakeet-ctc-0.6b, licensed CC-BY-4.0.
"""

from __future__ import annotations

import re

import numpy as np

from .base import Backend

MODEL = "nvidia/parakeet-ctc-0.6b"
CHUNK = 480_000  # 30 s at 16 kHz, run as separate, non-overlapping pieces
MIN_CHUNK = 8_000  # pieces shorter than 0.5 s are zero-padded to this


def normalize(word: str) -> str:
    """Lowercase a-z with internal apostrophes; digits and other marks dropped."""
    w = re.sub(r"[^a-z']", "", word.lower().replace("’", "'")).strip("'")
    return w or "a"  # keep every word placeable so the passes stay one to one


class ParakeetCTC(Backend):
    name = "parakeet-ctc-0.6b"
    model_id = MODEL
    weights_license = "CC-BY-4.0 (NVIDIA)"
    frame_seconds = 0.08  # 10 ms features, 8x subsampling
    frame_samples = 1280
    chunk_seconds = None  # forward() does its own chunking

    def load(self, device: str) -> None:
        import torch
        from nemo.collections.asr.models import ASRModel

        self._torch = torch
        self._model = ASRModel.from_pretrained(MODEL, map_location=device).eval()
        self._tok = self._model.tokenizer
        self.blank = self._tok.vocab_size  # 1024; the CTC blank sits after the vocabulary

    def forward(self, wave: np.ndarray) -> np.ndarray:
        torch = self._torch
        wave = np.ascontiguousarray(wave, dtype=np.float32)
        chunks = [wave[i : i + CHUNK] for i in range(0, len(wave), CHUNK)] or [wave]
        if len(chunks[-1]) < MIN_CHUNK:
            chunks[-1] = np.pad(chunks[-1], (0, MIN_CHUNK - len(chunks[-1])))
        with torch.inference_mode():
            hyps = self._model.transcribe(chunks, return_hypotheses=True, batch_size=1, verbose=False)
        if isinstance(hyps, tuple):
            hyps = hyps[0]
        lp = torch.cat(
            [torch.log_softmax(torch.as_tensor(h.y_sequence).float(), -1) for h in hyps], 0
        )
        return lp.cpu().numpy()

    def tokenize(self, word: str) -> list[int]:
        return list(self._tok.text_to_ids(normalize(word)))
