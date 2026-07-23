"""Confined seasonal elastic storage release checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Geod
import rasterio

from .constants import ANNUAL_PERIOD_DAYS, EXPECTED_STORAGE, LAG_C_DAYS, RELEASE_ROOT
from .harmonics import phase_days
from .io import read_json


def geodesic_pixel_area_rows(transform: rasterio.Affine, width: int, height: int) -> np.ndarray:
    geod = Geod(ellps="WGS84")
    areas = np.empty(height, dtype=float)
    x0 = transform.c
    x1 = transform.c + transform.a
    for row in range(height):
        y0 = transform.f + row * transform.e
        y1 = transform.f + (row + 1) * transform.e
        area, _ = geod.polygon_area_perimeter([x0, x1, x1, x0], [y0, y0, y1, y1])
        areas[row] = abs(area)
    return areas


def storage_summary(release_root: Path = RELEASE_ROOT) -> dict[str, Any]:
    path = release_root / "storage" / "confined_harmonic_storage" / "confined_storage_harmonic_regional_summary.json"
    return read_json(path)


def recalculate_storage(release_root: Path = RELEASE_ROOT, tolerance: float = 1e-6) -> dict[str, Any]:
    summary = storage_summary(release_root)
    checks = {
        "coherent_amplitude_m3": summary["regional_coherent_amplitude_m3"],
        "local_amplitude_sum_m3": summary["sum_local_amplitudes_m3"],
        "peak_to_trough_m3": summary["seasonal_max_minus_min_m3"],
        "delayed_peak_shift_days": summary["delayed_peak_shift_days"],
        "delayed_direction": summary["delayed_peak_shift_sign"],
    }
    ok = (
        abs(checks["coherent_amplitude_m3"] - EXPECTED_STORAGE["coherent_amplitude_m3"]) <= tolerance
        and abs(checks["local_amplitude_sum_m3"] - EXPECTED_STORAGE["local_amplitude_sum_m3"]) <= tolerance
        and abs(checks["peak_to_trough_m3"] - EXPECTED_STORAGE["peak_to_trough_m3"]) <= tolerance
        and abs(checks["delayed_peak_shift_days"] - LAG_C_DAYS) < 0.05
        and checks["delayed_direction"] == "positive_delay"
    )
    return {
        "storage_recalculation_status": "passed" if ok else "failed",
        "delayed_positive_shift_status": "passed" if checks["delayed_direction"] == "positive_delay" and abs(checks["delayed_peak_shift_days"] - LAG_C_DAYS) < 0.05 else "failed",
        "metrics": checks,
        "phase_day_check": float(phase_days(summary["regional_real_m3"], summary["regional_imag_m3"], ANNUAL_PERIOD_DAYS)),
    }
