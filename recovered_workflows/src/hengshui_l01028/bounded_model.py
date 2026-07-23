"""Bounded model math helpers used in tests and audits."""

from __future__ import annotations

import numpy as np


def stable_sigmoid(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def bounded_ske_from_eta(eta: np.ndarray, ske_min: float, ske_max: float) -> tuple[np.ndarray, np.ndarray]:
    sig = stable_sigmoid(eta)
    span = float(ske_max) - float(ske_min)
    return float(ske_min) + span * sig, span * sig * (1.0 - sig)
