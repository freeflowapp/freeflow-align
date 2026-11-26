import json

import numpy as np
import pytest

from align import letters as let
from align.letters import add_letters, word_letters
from align.run import parse_transcript

from test_align import emissions


class PathLetters:
    """A pretend letter model: at each 0.02 s frame it hears the letter id written into the audio (0 = blank)."""

    sample_rate = 100
    frame_seconds = 0.02
    blank = 0
    IDS = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz'")}

    def emissions(self, wave):
        path = [int(v) for v in wave[::2]]  # 2 samples a frame at 100 Hz = 20 ms
        return emissions(path, vocab=len(self.IDS) + 1), 0.02

    def spell(self, word):
        return [self.IDS[c] for c in word.lower()]

    @staticmethod
    def audio(path):
        return np.repeat(np.array(path, dtype=np.float32), 2)


def frames_for(spec, n):
    """Frame path of n frames with letter ids at given frames."""
    path = [0] * n
    for frame, letter in spec:
        path[frame] = PathLetters.IDS[letter]
    return path


def test_letters_land_where_they_are_sung_shifted_by_the_offset():
    # "go" sung with g at frame 50 (1.00 s) and o at frame 60 (1.20 s)
    path = frames_for([(50, "g"), (60, "o")], 150)
    segs = parse_transcript(json.dumps({"segments": [{"id": "s", "text": "go", "start": 0.5, "end": 2.0}]}))
    results = [{"words": [{"text": "go", "start": 0.98, "end": 1.4, "score": 0.9}]}]
    add_letters(PathLetters.audio(path), segs, results, PathLetters(), padding=0.2)
    got = results[0]["words"][0]["letters"]
    assert got == pytest.approx([1.0 + let.TIME_OFFSET, 1.2 + let.TIME_OFFSET], abs=1e-6)


def test_letters_stay_inside_the_word_and_unplaced_words_get_none():
    # the same letters sung again after the word, outside its own window, must not be used
    path = frames_for([(50, "g"), (55, "o"), (125, "g"), (130, "o")], 150)
    segs = parse_transcript(json.dumps({"segments": [{"id": "s", "text": "go go", "start": 0.2, "end": 2.8}]}))
    results = [{"words": [{"text": "go", "start": 0.98, "end": 1.3}, {"text": "go", "start": None, "end": None}]}]
    add_letters(PathLetters.audio(path), segs, results, PathLetters(), padding=0.0)
    first, second = (w["letters"] for w in results[0]["words"])
    assert first[0] == pytest.approx(1.0 + let.TIME_OFFSET, abs=1e-6)
    assert second is None


def test_a_held_word_may_run_on_to_the_next_words_start():
    # "go": g at 1.0 s, but o held late, at 1.5 s, after where the word aligner heard it end (1.1 s); next word at 1.7 s
    path = frames_for([(50, "g"), (75, "o"), (85, "l"), (88, "a")], 150)
    segs = parse_transcript(json.dumps({"segments": [{"id": "s", "text": "go la", "start": 0.5, "end": 2.5}]}))
    results = [{"words": [{"text": "go", "start": 0.98, "end": 1.1}, {"text": "la", "start": 1.7, "end": 1.8}]}]
    add_letters(PathLetters.audio(path), segs, results, PathLetters(), padding=0.2)
    assert results[0]["words"][0]["letters"][1] == pytest.approx(1.5 + let.TIME_OFFSET, abs=1e-6)


def test_word_too_long_for_its_window_is_none():
    lp = emissions([0] * 10, vocab=5)
    assert word_letters(lp, 0.02, 0.0, [1, 2, 3, 4], 0.0, 0.04, 0) is None
    assert word_letters(lp, 0.02, 0.0, [], 0.0, 0.2, 0) is None


def test_letters_need_separate():
    from align import align_file

    with pytest.raises(ValueError, match="separate"):
        align_file("missing.wav", "go", letters=True, separate=False, backend=object())
