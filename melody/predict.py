"""Cog entry point: a song recording (full mix) in, lead-vocal pitch track out (JSON string, see README.md)."""

from cog import BasePredictor, Input, Path

import pitch


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.separator = pitch.Separator()
        print("device", self.separator.device, flush=True)

    def predict(
        self,
        audio: Path = Input(description="The full mix (any format ffmpeg reads)"),
        lead_upload_url: str = Input(
            default="",
            description="Optional https URL to PUT the separated lead vocal to, as FLAC (a signed storage upload URL)",
        ),
    ) -> str:
        return pitch.run(self.separator, str(audio), lead_upload_url)
