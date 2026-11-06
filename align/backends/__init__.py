"""Acoustic model backends. Add one by subclassing Backend and listing it here."""

from __future__ import annotations

from .base import Backend
from .parakeet import ParakeetCTC

BACKENDS: dict[str, type[Backend]] = {
    ParakeetCTC.name: ParakeetCTC,
}
DEFAULT = ParakeetCTC.name


def get(name: str) -> Backend:
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ValueError(f"unknown aligner {name!r}; choose from {sorted(BACKENDS)}") from None
