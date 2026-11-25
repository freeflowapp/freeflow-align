"""facebook/wav2vec2-large-960h-lv60-self (wav2vec 2.0 large, character CTC) through Hugging Face transformers.

Used for the letters pass only (``align.letters``), never as the word aligner: it hears letters on 20 ms frames, which
places the syllables inside a word much better than an even split, but it finds words on their own worse than
parakeet (Freeflow's fine-listening test, 2026-11-25).

Weights: https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self, licensed Apache-2.0, pinned to a commit.
"""

from __future__ import annotations

import re

import numpy as np

from .base import Backend

MODEL = "facebook/wav2vec2-large-960h-lv60-self"
REVISION = "9fc4550a36e97a2a446c46c3baa6a45678098861"


def letters_of(word: str) -> str:
    """The letters of ``word`` the model can hear: lowercase a-z with internal apostrophes; "" when there are none."""
    return re.sub(r"[^a-z']", "", word.lower().replace("’", "'")).strip("'")


class Wav2Vec2CTC(Backend):
    name = "wav2vec2-large-960h-lv60-self"
    model_id = f"{MODEL}@{REVISION[:8]}"
    weights_license = "Apache-2.0 (Meta AI)"
    frame_seconds = 0.02  # 320 samples at 16 kHz
    frame_samples = 320
    chunk_seconds = 30.0  # the large model on a whole song at once does not fit comfortably in memory
    context_seconds = 2.0

    def load(self, device: str) -> None:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

        self._torch = torch
        self._device = device
        self._features = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, revision=REVISION)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL, revision=REVISION)
        self._model = Wav2Vec2ForCTC.from_pretrained(MODEL, revision=REVISION).to(device).eval()
        self.set_vocab(tokenizer.get_vocab(), tokenizer.pad_token_id)

    def set_vocab(self, vocab: dict[str, int], blank: int) -> None:
        """The model's letters (upper case in its vocabulary) by id; the CTC blank is its pad token."""
        self._ids = {k.lower(): v for k, v in vocab.items() if len(k) == 1 and (k.isalpha() or k == "'")}
        self.blank = blank

    def forward(self, wave: np.ndarray) -> np.ndarray:
        torch = self._torch
        wave = np.ascontiguousarray(wave, dtype=np.float32)
        if len(wave) < 800:  # under 50 ms: too short for the model's convolutions
            wave = np.pad(wave, (0, 800 - len(wave)))
        inputs = self._features(wave, sampling_rate=self.sample_rate, return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(inputs.input_values.to(self._device)).logits[0]
        return torch.log_softmax(logits.float(), -1).cpu().numpy()

    def tokenize(self, word: str) -> list[int]:
        return [self._ids[c] for c in letters_of(word) if c in self._ids]
