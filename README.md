# align

Forced alignment of known text to audio, packaged with [Cog](https://cog.run)
for [Replicate](https://replicate.com).

You give it an audio file and the words you already know are in it. It returns
when each word starts and ends, with a confidence score, using a CTC acoustic
model and a Viterbi search that must follow your text exactly.

## Input

| Input | Meaning |
|---|---|
| `audio` | Any file ffmpeg can decode. |
| `transcript` | Plain text, or JSON `{"segments": [{"id": "a", "text": "...", "start": 12.0, "end": 30.5}]}`. `start`/`end` are optional window hints in seconds. |
| `aligner` | Which acoustic model to use (see below). |
| `mode` | `segments`: align each segment inside its window (unhinted segments use the whole-file pass). `whole`: one pass over the full file, hints ignored. `both`: `segments` timings, plus the whole-file timings on every word under `whole`, so you can see where the two passes disagree. |
| `padding` | Seconds added on each side of a window hint (default 1.5). Each window runs the model again on that stretch of audio alone, so the two passes in `both` are independent. |
| `separate` | Default `false`. When `true`, the vocal is first separated from the audio (see below) and the same passes run on it too: each word gains `stem` (its timings on the vocal stem, with `whole` in mode `both`), while its own timings stay the mix's, exactly as with `separate` off. The output gains `separator`. |

## Output

A JSON string:

```json
{
  "aligner": "parakeet-ctc-0.6b",
  "model": "nvidia/parakeet-ctc-0.6b",
  "weights_license": "CC-BY-4.0 (NVIDIA)",
  "mode": "both",
  "duration": 241.3,
  "frame_seconds": 0.08,
  "segments": [
    {"id": "a", "status": "ok", "words": [
      {"text": "hello", "start": 12.42, "end": 12.8, "score": 0.91,
       "whole": {"start": 12.44, "end": 12.8, "score": 0.9}}
    ]}
  ]
}
```

`score` is the mean model probability of the word's tokens over the frames
they occupy (0 to 1). A word the model cannot represent (for example digits
only) comes back with `"skipped": true` and null times. A segment whose window
is too short for its text comes back with `"status": "failed"`.

## Vocal separation (`separate: true`)

The vocal is separated with a Mel-Band RoFormer: the checkpoint `MelBandRoformer.ckpt` from
[KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer) at commit `ac9b0614`,
SHA-256 `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e`, run with
[BS-RoFormer](https://pypi.org/project/BS-RoFormer/) 0.3.10 (MIT). The checkpoint is downloaded and checked against
that hash when the image is built, checked again when it loads, and loaded as plain tensors (`weights_only=True`)
with a strict key match. No code from the model's own repository is used. Its Hugging Face metadata says MIT; the
model card is empty and the training data is not stated, which is what `separator.weights_license` says.

```json
"separator": {"model": "KimberleyJSN/melbandroformer@ac9b0614 on BS-RoFormer 0.3.10",
              "checkpoint_sha256": "87201f4d…7559e",
              "weights_license": "MIT (HF metadata; training data unverified)"}
```

## Aligners

| Name | Weights | Licence |
|---|---|---|
| `parakeet-ctc-0.6b` (default) | [nvidia/parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b) through [NeMo](https://github.com/NVIDIA/NeMo) | CC-BY-4.0 |

The default model is NVIDIA's parakeet-ctc-0.6b, used under the
[CC-BY-4.0 licence](https://creativecommons.org/licenses/by/4.0/). Credit:
NVIDIA. The code in this repository is MIT licensed; the model weights keep
their own licence.

Words are normalised before alignment: lowercase, a to z plus internal
apostrophes; digits and other marks are dropped, and a word left empty is
aligned as "a" so every word still gets a time.

Adding one means subclassing `align/backends/base.py:Backend` (load, per-frame
log probs, per-word tokenizer, frame length) and registering it in
`align/backends/__init__.py`. The alignment itself (`align/ctc.py`) is shared.

## Development

```
pip install numpy pytest && python -m pytest tests     # no model needed
pip install -r requirements.txt && python cli.py song.mp3 words.json --mode both
```

## Publishing

`.github/workflows/push.yml` runs the tests, builds the image and pushes a new
version to `r8.im/$REPLICATE_MODEL` on every change to the model code on
`main` (or by hand from the Actions tab). It needs:

- repository variable `REPLICATE_MODEL`, for example `owner/align`
- repository secret `REPLICATE_CLI_AUTH_TOKEN`, the Cog CLI token from
  https://replicate.com/auth/token (an ordinary API token is refused by `cog login`)
