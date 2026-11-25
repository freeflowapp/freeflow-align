import json
import shutil
import subprocess

import numpy as np
import pytest

from align.backends.base import Backend
from align.ctc import AlignmentError, forced_align, min_frames
from align.run import align_transcript, parse_transcript

BLANK = 0


def emissions(path, vocab=6, sharp=8.0, frames=None):
    """Log probs that strongly favour the token at each frame of ``path``."""
    T = frames or len(path)
    logits = np.zeros((T, vocab))
    for t, tok in enumerate(path):
        logits[t, tok] = sharp
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def test_recovers_token_positions():
    path = [0, 0, 1, 1, 0, 2, 2, 2, 0, 0, 3, 0]
    spans = forced_align(emissions(path), [1, 2, 3])
    assert [(s.start, s.end) for s in spans] == [(2, 4), (5, 8), (10, 11)]
    assert all(s.score > 0.9 for s in spans)


def test_repeated_token_needs_blank_between():
    path = [0, 4, 0, 4, 0]
    spans = forced_align(emissions(path), [4, 4])
    assert [(s.start, s.end) for s in spans] == [(1, 2), (3, 4)]
    assert min_frames([4, 4]) == 3
    with pytest.raises(AlignmentError):
        forced_align(emissions([4, 4]), [4, 4])


def test_follows_transcript_not_the_loudest_token():
    # The audio "says" 1 2 3, but the transcript is 1 3: 3 must still land late.
    path = [1, 1, 0, 2, 2, 0, 3, 3]
    spans = forced_align(emissions(path), [1, 3])
    assert spans[0].start == 0
    assert spans[1].start >= 5


def test_too_short_raises():
    with pytest.raises(AlignmentError):
        forced_align(emissions([1, 2]), [1, 2, 3])


TOKENS = {"a": [1], "bb": [2, 2], "c": [3], "9": []}


class PathBackend(Backend):
    """A pretend model that hears, at each 0.1 s frame, the token written into the audio."""

    name = model_id = weights_license = "path"
    sample_rate = 100
    frame_seconds = 0.1
    frame_samples = 10
    blank = 0

    def load(self, device):
        pass

    def tokenize(self, word):
        return TOKENS[word.lower()]

    def forward(self, wave):
        return emissions([int(v) for v in wave[:: self.frame_samples]])

    @staticmethod
    def audio(path):
        return np.repeat(np.array(path, dtype=np.float32), 10)


def test_segments_windows_and_whole_pass():
    # "a" at 1.0 s, "bb" at 1.2-1.5 s, then "c" at 5.0 s
    path = [0] * 60
    path[10] = 1
    path[12], path[13], path[14] = 2, 0, 2
    path[50] = 3
    segs = parse_transcript(json.dumps({"segments": [
        {"id": "x", "text": "a bb 9", "start": 0.8, "end": 2.0},
        {"id": "y", "text": "c", "start": 4.5, "end": 5.5},
    ]}))
    out = align_transcript(PathBackend.audio(path), PathBackend(), segs, mode="both", padding=0.2)
    x, y = out
    assert x["status"] == "ok" and [w["text"] for w in x["words"]] == ["a", "bb", "9"]
    assert x["words"][0]["start"] == pytest.approx(1.0)
    assert x["words"][1]["start"] == pytest.approx(1.2) and x["words"][1]["end"] == pytest.approx(1.5)
    assert x["words"][2]["skipped"] and x["words"][2]["start"] is None
    assert y["words"][0]["start"] == pytest.approx(5.0)
    assert y["words"][0]["whole"]["start"] == pytest.approx(5.0)
    assert x["words"][1]["whole"]["end"] == pytest.approx(1.5)


def test_unhinted_segment_uses_whole_pass_and_bad_window_fails_alone():
    path = [0] * 30
    path[5], path[20] = 1, 3
    segs = parse_transcript(json.dumps({"segments": [
        {"id": "x", "text": "a"},
        {"id": "y", "text": "c bb bb bb", "start": 2.0, "end": 2.1},
        {"id": "z", "text": "c", "start": 9.0, "end": 9.5},
    ]}))
    out = align_transcript(PathBackend.audio(path), PathBackend(), segs, padding=0)
    assert out[0]["words"][0]["start"] == pytest.approx(0.5)
    assert out[1]["status"] == "failed" and out[1]["words"][0]["start"] is None
    assert out[2]["status"] == "failed"


def test_parakeet_normalization():
    from align.backends.parakeet import normalize

    assert normalize("Don\u2019t,") == "don't"
    assert normalize("'Tis") == "tis"
    assert normalize("H\u00e9llo!") == "hllo"
    assert normalize("2") == "a"


def test_plain_text_and_bad_input():
    assert [s.words for s in parse_transcript("a bb  c")] == [["a", "bb", "c"]]
    with pytest.raises(ValueError):
        parse_transcript("   ")
    with pytest.raises(ValueError):
        parse_transcript('{"segments": [{"text": "a", "start": 3, "end": 2}]}')


class Fake(Backend):
    """Each frame's log probs encode the absolute sample it starts at."""

    name = model_id = weights_license = "fake"
    sample_rate = 100
    frame_seconds = 0.1
    frame_samples = 10
    blank = 0
    chunk_seconds = 3.0
    context_seconds = 1.0

    def load(self, device):
        pass

    def tokenize(self, word):
        return []

    def forward(self, wave):
        n = len(wave) // self.frame_samples
        return wave[: n * self.frame_samples : self.frame_samples, None].astype(np.float64)


@pytest.mark.parametrize("n", [50, 499, 500, 501, 1234, 2000])
def test_chunked_emissions_line_up_with_samples(n):
    wave = np.arange(n, dtype=np.float32)
    lp = Fake().emissions(wave)
    assert lp.shape[0] == n // 10
    assert np.array_equal(lp[:, 0], np.arange(n // 10) * 10)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_audio_load(tmp_path):
    from align import audio

    f = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=1:sample_rate=44100",
                    "-ac", "2", str(f)], check=True)
    wave = audio.load(str(f), 16000)
    assert abs(len(wave) - 16000) < 100 and wave.dtype == np.float32


# ── The letters pass ───────────────────────────────────────────────────────

LETTER_IDS = {"p": 1, "o": 2, "w": 3, "e": 4, "r": 5, "'": 6}


class LetterBackend(Backend):
    """A pretend character model: at each 0.1 s frame it hears the letter id written into the audio."""

    name = model_id = weights_license = "letters"
    sample_rate = 100
    frame_seconds = 0.1
    frame_samples = 10
    blank = 0

    def load(self, device):
        pass

    def tokenize(self, word):
        from align.backends.wav2vec2 import letters_of

        return [LETTER_IDS[c] for c in letters_of(word) if c in LETTER_IDS]

    def forward(self, wave):
        return emissions([int(v) for v in wave[:: self.frame_samples]], vocab=8)


def test_letters_of_matches_the_word_aligner_normalisation():
    from align.backends.wav2vec2 import letters_of

    assert letters_of("Pow-er,") == "power"
    assert letters_of("Don’t") == "don't"
    assert letters_of("'Tis") == "tis"
    assert letters_of("42") == ""


def test_word_letters_inside_the_placed_word():
    from align.letters import word_letters

    path = [0] * 40
    for t, c in zip([10, 11, 14, 15, 16], "power"):
        path[t] = LETTER_IDS[c]
    b = LetterBackend()
    lp = b.forward(PathBackend.audio(path))
    times = word_letters(lp, 0.1, "Power!", (1.0, 1.7), b.tokenize, b.blank, padding=0.15)
    assert times == pytest.approx([1.0, 1.1, 1.4, 1.5, 1.6])
    # The same letters heard again later are outside this word's time and do not pull it.
    path2 = path[:30] + [LETTER_IDS[c] for c in "power"] + [0] * 5
    lp2 = b.forward(PathBackend.audio(path2))
    assert word_letters(lp2, 0.1, "power", (1.0, 1.7), b.tokenize, b.blank, padding=0.15)[0] == pytest.approx(1.0)


def test_word_letters_gives_none_when_it_cannot_place():
    from align.letters import word_letters

    b = LetterBackend()
    lp = b.forward(PathBackend.audio([0] * 20))
    assert word_letters(lp, 0.1, "42", (0.5, 0.9), b.tokenize, b.blank) is None  # nothing to hear
    assert word_letters(lp, 0.1, "power", (0.5, 0.55), b.tokenize, b.blank, padding=0) is None  # too short
    assert word_letters(lp, 0.1, "power", (5.0, 6.0), b.tokenize, b.blank) is None  # outside the audio


def test_add_letters_uses_the_stem_pass_first_and_skips_unplaced_words():
    from align.letters import add_letters

    path = [0] * 40
    for t, c in zip([20, 21, 22, 23, 24], "power"):
        path[t] = LETTER_IDS[c]
    b = LetterBackend()
    lp = b.forward(PathBackend.audio(path))
    results = [{"id": "x", "status": "ok", "words": [
        {"text": "power", "start": 0.5, "end": 0.9, "score": 0.9, "stem": {"start": 2.0, "end": 2.5, "score": 0.9}},
        {"text": "power", "start": None, "end": None, "score": None},
        {"text": "9", "start": 1.0, "end": 1.2, "score": 0.5, "skipped": True},
    ]}]
    add_letters(results, lp, b)
    w = results[0]["words"]
    assert w[0]["letters"] == pytest.approx([2.0, 2.1, 2.2, 2.3, 2.4])
    assert w[1]["letters"] is None and w[2]["letters"] is None


def test_wav2vec2_tokenizes_letters_from_its_upper_case_vocabulary():
    from align.backends.wav2vec2 import Wav2Vec2CTC

    b = Wav2Vec2CTC()
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "W": 18,
             "R": 13, "P": 23, "D": 14, "'": 27}
    b.set_vocab(vocab, 0)
    assert b.blank == 0
    assert b.tokenize("Pow-er!") == [23, 8, 18, 5, 13]
    assert b.tokenize("Don’t") == [14, 8, 9, 27, 6]
    assert b.tokenize("4") == []
