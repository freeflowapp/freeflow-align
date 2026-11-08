# melody-pitch

A **private** Replicate model, `freeflowdev/melody-pitch`: a song recording (the full mix) in, a lead-vocal pitch track
out. It is separate from the align model in the repo root and has its own `cog.yaml`; `.github/workflows/push-melody.yml`
pushes it when anything under `melody/` changes on `main`.

## Licence of the weights — read this first

The separation weights are the Mel-Band RoFormer **karaoke** checkpoint by aufr33 & viperx
(`mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt`). **They carry no stated licence** (open question: https://github.com/Anjok07/ultimatevocalremovergui/issues/2295): the
GitHub release they come from has no text, and its repository has no licence file. Freeflow uses them **provisionally**,
until they are replaced, and the model is kept private for that reason. The code in this folder is MIT like the rest of
the repo; the model class is lucidrains' BS-RoFormer 0.3.10 from PyPI (MIT).

- Download (at image build time, never committed):
  https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt
- SHA-256 `1de20d459332fe8869aeb01327a31df0032262706e1365114e852dc271779813` (913,096,801 bytes), checked by the build
  and again when the model loads. Loaded with `torch.load(weights_only=True)` and a strict key match.

## What it does

The same steps as freeflow-ai-eval's `scripts/melody/separate_karaoke.py` followed by `scripts/melody/f0.py`
(see `pitch.py`):

1. ffmpeg decodes the mix to 44.1 kHz stereo.
2. Karaoke separation: 8 s chunks with a 4 s hop, Hann window (+1e-3), overlap-added; the single output stem is the
   lead vocal (backing vocals count as accompaniment).
3. Stereo mean, resampled to 16 kHz mono (`scipy.signal.resample_poly(160, 441)`).
4. `librosa.pyin(fmin=E2, fmax=C6, sr=16000, frame_length=1024, hop_length=160)`; semitones = `hz_to_midi(f0) - 12`
   (C0 = 0).

## Output

A JSON string:

| key | value |
|---|---|
| `model` | `"mel_band_roformer_karaoke_aufr33_viperx"` |
| `weights_sha256` | the checkpoint's SHA-256 |
| `tracker` | `"pyin"` |
| `sample_rate` | `16000` |
| `frame_seconds` | `0.01` |
| `frames` | N |
| `duration` | seconds of audio |
| `semis` | base64 of N little-endian int16: `round(semitones * 100)`, `-32768` where unvoiced |
| `voicing` | base64 of N uint8: `round(voiced_probability * 255)` |

Frame `i` is at `i * 0.01` s from the start of the file (librosa's centred frames). `pitch.decode()` reverses it.

## Running it locally

No Docker needed:

```bash
MELODY_CKPT=/path/to/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt python melody/run_local.py song.m4a out.json
```
