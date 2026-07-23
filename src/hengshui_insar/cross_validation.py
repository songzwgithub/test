"""Formal cross-validation source-level release checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .constants import EXPECTED_FOLD_RMSE, RELEASE_ROOT
from .io import read_json
from .constants import LAG_C_DAYS, LAG_U_DAYS, SKE_MAX, SKE_MIN
from .source_recompute import StreamInputs, fold_partition_audit, recompute_cv_metrics


def formal_cv_metrics(release_root: Path = RELEASE_ROOT) -> dict[str, Any]:
    cv_path = release_root / "formal_cv" / "formal_cv_acceptance.json"
    if not cv_path.exists():
        cv_path = release_root / "formal_cv" / "formal_cv_summary.json"
    payload = read_json(cv_path)
    rows = payload.get("rows", [])
    result = {}
    for row in rows:
        fold = int(row["fold_id"])
        result[fold] = float(row["validation_rmse"])
    return {"fold_rmse": result, "source": str(cv_path)}


def _expected_fold_metrics(release_root: Path, fold: int) -> dict[str, Any]:
    path = release_root / "formal_cv" / f"fold_{fold:02d}" / "fit_summary.json"
    payload = read_json(path)
    val = payload["validation"]
    return {
        "rmse": val["rmse"],
        "mae": val["mae"],
        "pixel_count": val["pixel_count"],
        "Ske_min": val["Ske_min"],
        "Ske_max": val["Ske_max"],
        "Cu_global": val["Cu_global"],
        "lag_c_days": val["lag_c_days"],
        "lag_u_days": val["lag_u_days"],
        "gamma_norm": val["gamma_norm"],
        "prediction_abs_p99": val["prediction_abs_p99"],
        "nonfinite_prediction_count": val["nonfinite_prediction_count"],
    }


def recalculate_formal_cv(release_root: Path = RELEASE_ROOT, tolerance: float = 1e-8, inputs: StreamInputs | None = None) -> dict[str, Any]:
    inputs = inputs or StreamInputs(release_root=release_root)
    metrics = recompute_cv_metrics(inputs)
    partition = fold_partition_audit(inputs)
    rows = []
    ok = partition["fold_partition_status"] == "passed"
    for fold, expected in EXPECTED_FOLD_RMSE.items():
        actual = metrics[fold]
        stored = _expected_fold_metrics(release_root, fold)
        gates = {
            "validation_rmse": abs(float(actual["rmse"]) - expected) <= tolerance,
            "validation_mae": abs(float(actual["mae"]) - float(stored["mae"])) <= tolerance,
            "validation_pixel_count": int(actual["pixel_count"]) == int(stored["pixel_count"]) == int(partition["fold_counts"][fold]),
            "nonfinite_prediction_count": int(actual["nonfinite_prediction_count"]) == 0,
            "Ske_min_bound": float(actual["Ske_min"]) >= SKE_MIN - 1e-12,
            "Ske_max_bound": float(actual["Ske_max"]) <= SKE_MAX + 1e-12,
            "Ske_min_value": abs(float(actual["Ske_min"]) - float(stored["Ske_min"])) <= 1e-6,
            "Ske_max_value": abs(float(actual["Ske_max"]) - float(stored["Ske_max"])) <= 1e-6,
            "prediction_abs_p99": np.isfinite(float(actual["prediction_abs_p99"])) and float(actual["prediction_abs_p99"]) <= 100.0,
            "Cu_global": abs(float(actual["Cu_global"]) - float(stored["Cu_global"])) <= 1e-12,
            "lag_c_days": abs(float(actual["lag_c_days"]) - float(stored["lag_c_days"])) <= 1e-12,
            "lag_u_days": abs(float(actual["lag_u_days"]) - LAG_U_DAYS) <= 1e-12,
        }
        status = "passed" if all(gates.values()) else "failed"
        ok = ok and status == "passed"
        rows.append(
            {
                "fold_id": fold,
                "actual_rmse_mm": actual["rmse"],
                "expected_rmse_mm": expected,
                "actual_mae_mm": actual["mae"],
                "expected_mae_mm": stored["mae"],
                "validation_pixel_count": actual["pixel_count"],
                "Ske_min": actual["Ske_min"],
                "Ske_max": actual["Ske_max"],
                "Cu_global": actual["Cu_global"],
                "lag_c_days": actual["lag_c_days"],
                "lag_u_days": actual["lag_u_days"],
                "prediction_abs_p99": actual["prediction_abs_p99"],
                "nonfinite_prediction_count": actual["nonfinite_prediction_count"],
                "gates": gates,
                "status": status,
            }
        )
    return {
        "formal_cv_recalculation_status": "passed" if ok else "failed",
        "fold_partition_audit": partition,
        "rows": rows,
        "source_level_recalculation": True,
    }


def recalculate_final_refit(release_root: Path = RELEASE_ROOT, expected: dict[str, float] | None = None, tolerance: float = 1e-8, inputs: StreamInputs | None = None) -> dict[str, Any]:
    from .constants import EXPECTED_FINAL
    from .source_recompute import recompute_final_metrics

    expected = expected or EXPECTED_FINAL
    inputs = inputs or StreamInputs(release_root=release_root)
    actual = recompute_final_metrics(inputs)
    rows = {}
    ok = True
    mapping = {"rmse": "rmse", "mae": "mae", "Ske_min": "Ske_min", "Ske_p50": "Ske_p50", "Ske_max": "Ske_max", "Cu_global": "Cu_global"}
    for key, actual_key in mapping.items():
        metric_tol = {"rmse": 1e-8, "mae": 1e-8, "Ske_min": 1e-10, "Ske_p50": 1e-9, "Ske_max": 1e-10, "Cu_global": 1e-13}[key]
        diff = abs(float(actual[actual_key]) - float(expected[key]))
        rows[key] = {"actual": actual[actual_key], "expected": expected[key], "abs_diff": diff}
        ok = ok and diff <= metric_tol
    extra_gates = {
        "lag_c_days": abs(float(actual["lag_c_days"]) - LAG_C_DAYS) <= 1e-12,
        "lag_u_days": abs(float(actual["lag_u_days"]) - LAG_U_DAYS) <= 1e-12,
        "gamma_norm_finite": np.isfinite(float(actual["gamma_norm"])),
        "nonfinite_prediction_count": int(actual["nonfinite_prediction_count"]) == 0,
        "Ske_min_bound": float(actual["Ske_min"]) >= SKE_MIN - 1e-12,
        "Ske_max_bound": float(actual["Ske_max"]) <= SKE_MAX + 1e-12,
        "prediction_abs_p99_finite": np.isfinite(float(actual["prediction_abs_p99"])),
        "parameter_count": len(np.load(release_root / "final_refit" / "parameters.npy")) == 27,
    }
    ok = ok and all(extra_gates.values())
    return {
        "final_refit_recalculation_status": "passed" if ok else "failed",
        "metrics": rows,
        "extra_gates": extra_gates,
        "actual": actual,
        "source_level_recalculation": True,
    }
