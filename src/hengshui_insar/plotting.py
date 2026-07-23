"""Plotting helpers for release figures."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


def cyclic_phase_limits(period_days: float) -> tuple[float, float, str]:
    return 0.0, float(period_days), "twilight_shifted"


def symmetric_real_imag_norm(values: np.ndarray) -> Normalize:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    vmax = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1e-12)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def amplitude_colormap() -> str:
    return "viridis"
