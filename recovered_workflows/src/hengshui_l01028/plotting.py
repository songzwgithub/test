"""Plotting configuration helpers for publication figures."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


def diverging_norm(values: np.ndarray) -> Normalize:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return Normalize(vmin=-1.0, vmax=1.0)
    vmax = float(np.nanmax(np.abs(finite)))
    vmax = max(vmax, 1e-12)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
