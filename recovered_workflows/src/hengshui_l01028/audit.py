"""Shared audit predicates for L01028 final cleanup."""

from __future__ import annotations

from pathlib import Path


def path_is_authoritative(path: Path, protected_substrings: tuple[str, ...]) -> bool:
    text = str(path)
    return any(item in text for item in protected_substrings)
