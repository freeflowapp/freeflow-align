"""Cog entry point: audio + known text in, word timings out (JSON string)."""

import json

from cog import BasePredictor, Input, Path

import align
from align import backends


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = align._default_device()
        self.loaded: dict[str, backends.Backend] = {}
        self._backend(backends.DEFAULT)
        self.separator = None  # loaded on the first request that asks for it

    def _separator(self):
        if self.separator is None:
            from align.separate import Separator

            self.separator = Separator(self.device)
        return self.separator

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
        separate: bool = Input(
            default=False,
            description="Also separate the vocal (Mel-Band RoFormer) and run the same passes on it; each word then "
            "carries its stem timings under `stem`, and its own timings stay the mix's",
        ),
    ) -> str:
        result = align.align_file(
            str(audio), transcript, aligner=aligner, mode=mode, padding=padding,
            backend=self._backend(aligner), separate=separate,
            separator=self._separator() if separate else None,
        )
        return json.dumps(result)
