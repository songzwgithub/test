"""Confined seasonal elastic storage source-level release checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .constants import ANNUAL_PERIOD_DAYS, EXPECTED_STORAGE, LAG_C_DAYS, RELEASE_ROOT
from .harmonics import phase_days
from .io import read_json
from .source_recompute import StreamInputs, geodesic_pixel_area_rows, recompute_storage_metrics


def storage_summary(release_root: Path = RELEASE_ROOT) -> dict[str, Any]:
    path = release_root / "storage" / "confined_harmonic_storage" / "confined_storage_harmonic_regional_summary.json"
    return read_json(path)


def recalculate_storage(release_root: Path = RELEASE_ROOT, tolerance: float = 1e-3) -> dict[str, Any]:
    summary = recompute_storage_metrics(StreamInputs(release_root=release_root))
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
        "source_level_recalculation": True,
    }
