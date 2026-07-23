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
    envelope_path = release_root / "storage" / "uncertainty" / "uncertainty_acceptance.json"
    envelope = read_json(envelope_path) if envelope_path.exists() else {}
    expected_envelope = {
        "p2_5_m3": 81529443.78,
        "p97_5_m3": 99245376.13,
    }
    envelope_checks = {
        "uncertainty_file_present": envelope_path.exists(),
        "uncertainty_name": envelope.get("confined_storage_uncertainty_status") == "passed_structural_amplitude_envelope",
        "p2_5_m3": abs(float(envelope.get("p2_5_m3", float("nan"))) - expected_envelope["p2_5_m3"]) <= 1.0,
        "p97_5_m3": abs(float(envelope.get("p97_5_m3", float("nan"))) - expected_envelope["p97_5_m3"]) <= 1.0,
    }
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
        and summary["Ske_tif_max_abs_diff"] <= 1e-8
        and abs(summary["lag_c_days_from_final_parameters"] - LAG_C_DAYS) < 0.05
        and all(envelope_checks.values())
    )
    return {
        "storage_recalculation_status": "passed" if ok else "failed",
        "delayed_positive_shift_status": "passed" if checks["delayed_direction"] == "positive_delay" and abs(checks["delayed_peak_shift_days"] - LAG_C_DAYS) < 0.05 else "failed",
        "metrics": checks,
        "Ske_parameter_to_tif_check": {
            "max_abs_diff": summary["Ske_tif_max_abs_diff"],
            "rms_diff": summary["Ske_tif_rms_diff"],
            "comparison_count": summary["Ske_tif_comparison_count"],
            "status": "passed" if summary["Ske_tif_max_abs_diff"] <= 1e-8 else "failed",
        },
        "structural_envelope_check": envelope_checks,
        "phase_day_check": float(phase_days(summary["regional_real_m3"], summary["regional_imag_m3"], ANNUAL_PERIOD_DAYS)),
        "source_level_recalculation": True,
    }
