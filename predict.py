"""Cog entry point: audio + known text in, word timings out (JSON string)."""

from __future__ import annotations

import json

from cog import BasePredictor, Input, Path

import align
from align import backends


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = align._default_device()
        self.loaded: dict[str, backends.Backend] = {}
        self._backend(backends.DEFAULT)

    def _backend(self, name: str) -> backends.Backend:
        if name not in self.loaded:
            b = backends.get(name)
            b.load(self.device)
            self.loaded[name] = b
        return self.loaded[name]

    def predict(
        self,
        audio: Path = Input(description="Audio file (any format ffmpeg reads)"),
        transcript: str = Input(
            description='Plain text, or JSON {"segments": [{"id": "a", "text": "...", '
            '"start": 12.0, "end": 30.5}]} where start/end are optional window hints in seconds'
        ),
        aligner: str = Input(default=backends.DEFAULT, choices=sorted(backends.BACKENDS)),
        mode: str = Input(
            default="segments",
            choices=list(align.MODES),
            description="segments: align each segment inside its hinted window; whole: one pass "
            "over the full audio; both: segments timings plus whole-pass timings per word",
        ),
        padding: float = Input(default=1.5, ge=0, le=30, description="Seconds added around each window hint"),
    ) -> str:
        result = align.align_file(
            str(audio), transcript, aligner=aligner, mode=mode, padding=padding,
            backend=self._backend(aligner),
        )
        return json.dumps(result)
