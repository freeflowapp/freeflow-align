import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "melody"))
import pitch  # noqa: E402


def test_encode_decode_round_trip():
    semi = np.array([np.nan, 40.123, -0.004, 12.5, np.nan])
    vprob = np.array([0.0, 0.5, 1.0, 0.2, 0.01])
    out = pitch.encode(semi, vprob, 0.05)
    assert out["frames"] == 5 and out["frame_seconds"] == 0.01 and out["sample_rate"] == 16000
    assert set(out) == {"model", "weights_sha256", "tracker", "sample_rate", "frame_seconds", "frames", "duration",
                        "semis", "voicing"}
    s, v = pitch.decode(out)
    assert np.isnan(s[0]) and np.isnan(s[4])
    np.testing.assert_allclose(s[1:4], [40.12, 0.0, 12.5])
    np.testing.assert_allclose(v, np.round(vprob * 255) / 255)


def test_lead_is_only_put_to_https():
    import pytest

    with pytest.raises(ValueError):
        pitch.upload_lead(__file__, "http://example.com/upload")


def test_upload_lead_puts_the_file(monkeypatch, tmp_path):
    import urllib.request

    sent = {}

    class Res:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        sent.update(method=req.get_method(), url=req.full_url, body=req.data, type=req.get_header("Content-type"))
        return Res()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    f = tmp_path / "lead.flac"
    f.write_bytes(b"fLaC1234")
    assert pitch.upload_lead(str(f), "https://store.example/upload/sign/x.flac?token=t") == 8
    assert sent == {"method": "PUT", "url": "https://store.example/upload/sign/x.flac?token=t", "body": b"fLaC1234",
                    "type": "audio/flac"}


def test_keep_lead_reports_a_failed_upload_instead_of_failing(monkeypatch, tmp_path):
    def refuse(path, url):
        raise RuntimeError("upload refused (403)")

    monkeypatch.setattr(pitch, "upload_lead", refuse)
    monkeypatch.setattr(pitch, "lead_flac", lambda mono, workdir: str(tmp_path / "lead.flac"))
    out = pitch.keep_lead(np.zeros(10, dtype="float32"), "https://store.example/u", str(tmp_path))
    assert out == {"uploaded": False, "error": "upload refused (403)"}


def test_lead_flac_is_lossless_mono_22k_and_the_same_length(tmp_path):
    import soundfile as sf

    t = np.arange(44100 * 2) / 44100
    mono = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    path = pitch.lead_flac(mono, str(tmp_path))
    y, sr = sf.read(path)
    assert sr == 22050 and y.ndim == 1
    assert abs(len(y) - len(mono) / 2) <= 2
