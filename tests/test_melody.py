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
