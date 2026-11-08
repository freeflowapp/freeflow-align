"""Full mix in, lead-vocal pitch track out.

Reproduces freeflow-ai-eval's scripts/melody/separate_karaoke.py + scripts/melody/f0.py step for step:
  ffmpeg -> 44.1 kHz stereo WAV (16-bit, ffmpeg's default for .wav) -> soundfile float32
  -> Mel-Band RoFormer karaoke (aufr33 & viperx), 8 s chunks, 50% hop, Hann window + 1e-3, overlap-add
  -> stereo mean -> scipy resample_poly(160, 441) -> 16 kHz WAV (soundfile, 16-bit) -> librosa.load(sr=16000)
  -> librosa.pyin(E2..C6, frame_length 1024, hop 160) -> semitones with C0 = 0.
The intermediate WAV files are kept on purpose: the reference quantised to 16 bit at both points, and so do we.

Model class: lucidrains' BS-RoFormer 0.3.10 (PyPI). Weights: loaded as plain tensors (weights_only=True), strict key
match, SHA-256 checked first. The weights carry no stated licence (see README.md).
"""

import base64
import hashlib
import json
import os
import subprocess
import tempfile
import time

import numpy as np

MODEL = "mel_band_roformer_karaoke_aufr33_viperx"
CHECKPOINT_NAME = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
CHECKPOINT_SHA256 = "1de20d459332fe8869aeb01327a31df0032262706e1365114e852dc271779813"
CHUNK = 352800  # audio.chunk_size in config_mel_band_roformer_karaoke.yaml (8 s at 44.1 kHz)
SR = 16000
HOP = 160  # 10 ms frames
UNVOICED = -32768
LEAD_SR = 22050  # the listening copy of the lead vocal (`lead_flac`)

# The `model:` section of config_mel_band_roformer_karaoke.yaml (TRvlvr/application_data), verbatim.
CONFIG = dict(
    dim=384, depth=6, stereo=True, num_stems=1, time_transformer_depth=1, freq_transformer_depth=1, num_bands=60,
    dim_head=64, heads=8, attn_dropout=0, ff_dropout=0, flash_attn=True, dim_freqs_in=1025, sample_rate=44100,
    stft_n_fft=2048, stft_hop_length=441, stft_win_length=2048, stft_normalized=False, mask_estimator_depth=2,
    multi_stft_resolution_loss_weight=1.0, multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
    multi_stft_hop_size=147, multi_stft_normalized=False,
)


def checkpoint_path():
    for p in (os.environ.get("MELODY_CKPT"), f"/weights/{CHECKPOINT_NAME}"):
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"{CHECKPOINT_NAME} not found (set MELODY_CKPT, or /weights/ in the image)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def default_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Separator:
    def __init__(self, device=None, checkpoint=None):
        import torch
        from bs_roformer import MelBandRoformer

        checkpoint = checkpoint or checkpoint_path()
        digest = sha256(checkpoint)
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError(f"checkpoint sha256 {digest} does not match the pinned {CHECKPOINT_SHA256}")
        model = MelBandRoformer(**CONFIG)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.torch = torch
        self.device = device or default_device()
        self.model = model.to(self.device).eval()

    def lead(self, x):
        """x: (2, n) float32 at 44.1 kHz -> (2, n) lead vocal."""
        n = x.shape[1]
        hop = CHUNK // 2
        out = np.zeros_like(x)
        wsum = np.zeros(n, dtype=np.float32)
        win = np.hanning(CHUNK).astype(np.float32) + 1e-3
        with self.torch.inference_mode():
            for s in range(0, n, hop):
                seg = x[:, s:s + CHUNK]
                length = seg.shape[1]
                if length < CHUNK:
                    seg = np.pad(seg, ((0, 0), (0, CHUNK - length)))
                y = self.model(self.torch.from_numpy(np.ascontiguousarray(seg))[None].to(self.device))[0]
                y = y.float().cpu().numpy()
                out[:, s:s + length] += y[:, :length] * win[:length]
                wsum[s:s + length] += win[:length]
        return out / np.maximum(wsum, 1e-6)


def lead_vocal_16k(separator, audio_path, workdir):
    """The lead vocal of the mix at audio_path, 16 kHz mono float32, via the same WAV round trips as the reference."""
    import soundfile as sf
    from scipy.signal import resample_poly

    mix = os.path.join(workdir, "mix44.wav")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(audio_path), "-ar", "44100", "-ac", "2", mix],
                   check=True)
    wav, sr = sf.read(mix, dtype="float32", always_2d=True)
    assert sr == 44100 and wav.shape[1] == 2, (sr, wav.shape)
    v = separator.lead(np.ascontiguousarray(wav.T))
    corr = float(np.corrcoef(v.mean(0), wav.mean(1))[0, 1])
    if corr > 0.95:
        raise RuntimeError(f"the lead stem is the mix (corr {corr:.2f}); separation did not work")
    mono = v.mean(0)
    y = resample_poly(mono, 160, 441).astype("float32")
    lead = os.path.join(workdir, "lead16.wav")
    sf.write(lead, y, SR)
    import librosa

    y16, _ = librosa.load(lead, sr=SR, mono=True)
    return y16, corr, mono


def lead_flac(mono44, workdir):
    """The lead vocal (44.1 kHz mono float) as a FLAC file to listen to: 22.05 kHz mono 16-bit, lossless, so it lines
    up with the recording sample for sample (no encoder delay, unlike MP3 or AAC). About 1.4 MB a minute."""
    import soundfile as sf

    wav = os.path.join(workdir, "lead44.wav")
    sf.write(wav, np.clip(mono44, -1.0, 1.0).astype("float32"), 44100, subtype="FLOAT")
    out = os.path.join(workdir, "lead.flac")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", wav, "-ar", str(LEAD_SR), "-ac", "1",
                    "-sample_fmt", "s16", "-c:a", "flac", out], check=True)
    return out


def upload_lead(path, url, timeout=120):
    """PUT the file to a signed upload URL (Supabase Storage's `createSignedUploadUrl`). Returns the bytes sent."""
    import urllib.request

    if not url.startswith("https://"):
        raise ValueError("lead_upload_url must be https")
    data = open(path, "rb").read()
    req = urllib.request.Request(url, data=data, method="PUT",
                                 headers={"Content-Type": "audio/flac", "x-upsert": "true"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        if res.status >= 300:
            raise RuntimeError(f"upload refused ({res.status})")
    return len(data)


def pitch_track(y):
    import warnings

    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, vflag, vprob = librosa.pyin(y, fmin=librosa.note_to_hz("E2"), fmax=librosa.note_to_hz("C6"), sr=SR,
                                        frame_length=1024, hop_length=HOP, fill_na=np.nan)
    semi = np.full_like(f0, np.nan)
    ok = ~np.isnan(f0)
    semi[ok] = librosa.hz_to_midi(f0[ok]) - 12
    return semi, vprob


def encode(semi, vprob, duration):
    cents = np.full(len(semi), UNVOICED, dtype="<i2")
    ok = ~np.isnan(semi)
    cents[ok] = np.clip(np.round(semi[ok] * 100), -32767, 32767).astype("<i2")
    vo = np.clip(np.round(np.nan_to_num(vprob, nan=0.0) * 255), 0, 255).astype(np.uint8)
    return {
        "model": MODEL, "weights_sha256": CHECKPOINT_SHA256, "tracker": "pyin", "sample_rate": SR,
        "frame_seconds": HOP / SR, "frames": int(len(semi)), "duration": round(float(duration), 3),
        "semis": base64.b64encode(cents.tobytes()).decode("ascii"),
        "voicing": base64.b64encode(vo.tobytes()).decode("ascii"),
    }


def decode(result):
    """Inverse of encode: (semi float array with NaN unvoiced, vprob float array in 0..1)."""
    cents = np.frombuffer(base64.b64decode(result["semis"]), dtype="<i2")
    semi = np.where(cents == UNVOICED, np.nan, cents / 100.0)
    vprob = np.frombuffer(base64.b64decode(result["voicing"]), dtype=np.uint8) / 255.0
    assert len(semi) == len(vprob) == result["frames"]
    return semi, vprob


def keep_lead(mono44, url, workdir):
    """The `lead` entry of the result: the lead vocal uploaded to `url`, or why it wasn't. Never fails the run: the
    pitch track is the model's job, the listening copy a convenience."""
    try:
        sent = upload_lead(lead_flac(mono44, workdir), url)
        return {"uploaded": True, "format": "flac", "sample_rate": LEAD_SR, "bytes": sent}
    except Exception as e:  # noqa: BLE001 - reported in the result instead
        return {"uploaded": False, "error": str(e)[:300]}


def run(separator, audio_path, lead_upload_url=""):
    """Full mix -> JSON string (see README.md for the format). Logs timings to stdout. With `lead_upload_url`, the
    separated lead vocal is also uploaded there as FLAC and the result says so under `lead`."""
    lead = None
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        y, corr, mono44 = lead_vocal_16k(separator, audio_path, tmp)
        t1 = time.time()
        semi, vprob = pitch_track(y)
        t2 = time.time()
        if lead_upload_url:
            lead = keep_lead(mono44, lead_upload_url, tmp)
            print(f"lead vocal {lead} in {time.time() - t2:.1f} s", flush=True)
    duration = len(y) / SR
    print(f"audio {duration:.1f} s; separation {t1 - t0:.1f} s (corr with mix {corr:.2f}); pyin {t2 - t1:.1f} s; "
          f"voiced {np.mean(~np.isnan(semi)) * 100:.0f}% of {len(semi)} frames", flush=True)
    out = encode(semi, vprob, duration)
    if lead is not None:
        out["lead"] = lead
    return json.dumps(out, separators=(",", ":"))
