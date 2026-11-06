"""CTC forced alignment (Viterbi) in plain numpy.

Given per-frame log probabilities from any CTC acoustic model and the token
ids of a known transcript, find the single most likely path that emits exactly
those tokens in order, and report where each token (and so each word) sits.

Kept free of torch so it can be tested quickly and reused by every backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEG_INF = -np.inf


class AlignmentError(ValueError):
    """The transcript cannot fit in the given frames."""


@dataclass
class TokenSpan:
    token: int
    start: int  # first frame (inclusive)
    end: int  # last frame + 1
    score: float  # mean probability of the token over its frames


def min_frames(tokens: list[int]) -> int:
    """Frames needed to emit ``tokens``: one each, plus a blank between repeats."""
    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return len(tokens) + repeats


def forced_align(log_probs: np.ndarray, tokens: list[int], blank: int = 0) -> list[TokenSpan]:
    """Align ``tokens`` to ``log_probs`` (shape [frames, vocab]).

    Returns one span per token, in order. Raises AlignmentError when the
    transcript is empty or longer than the audio allows.
    """
    if log_probs.ndim != 2:
        raise ValueError("log_probs must be [frames, vocab]")
    if not tokens:
        raise AlignmentError("nothing to align")
    if any(t == blank for t in tokens):
        raise ValueError("transcript tokens must not contain the blank id")
    T = log_probs.shape[0]
    if T < min_frames(tokens):
        raise AlignmentError(f"{len(tokens)} tokens need {min_frames(tokens)} frames, got {T}")

    lp = np.asarray(log_probs, dtype=np.float64)
    # Extended label sequence: blank, t1, blank, t2, ..., tL, blank
    L = len(tokens)
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = tokens
    # A skip from s-2 to s is allowed onto a token that differs from the previous token.
    can_skip = np.zeros(S, dtype=bool)
    can_skip[3::2] = ext[3::2] != ext[1:-2:2]

    alpha = np.full(S, NEG_INF)
    alpha[0] = lp[0, blank]
    alpha[1] = lp[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2

    stay = np.empty(S)
    step = np.empty(S)
    skip = np.empty(S)
    for t in range(1, T):
        stay[:] = alpha
        step[0] = NEG_INF
        step[1:] = alpha[:-1]
        skip[:2] = NEG_INF
        skip[2:] = alpha[:-2]
        skip[~can_skip] = NEG_INF
        stacked = np.stack((stay, step, skip))
        choice = np.argmax(stacked, axis=0)
        back[t] = choice
        alpha = stacked[choice, np.arange(S)] + lp[t, ext]

    # The path must end on the last token or the blank after it.
    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    if not np.isfinite(alpha[s]):
        raise AlignmentError("no valid path")
    states = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s -= int(back[t, s])

    spans: list[TokenSpan] = []
    probs = np.exp(lp)
    for i in range(L):
        frames = np.nonzero(states == 2 * i + 1)[0]
        # Viterbi guarantees every token gets at least one frame.
        start, end = int(frames[0]), int(frames[-1]) + 1
        score = float(probs[frames, tokens[i]].mean())
        spans.append(TokenSpan(tokens[i], start, end, score))
    return spans
