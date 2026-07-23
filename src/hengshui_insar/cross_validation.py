"""Formal cross-validation source-level release checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import EXPECTED_FOLD_RMSE, RELEASE_ROOT
from .io import read_json
from .source_recompute import StreamInputs, recompute_cv_metrics


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


def recalculate_formal_cv(release_root: Path = RELEASE_ROOT, tolerance: float = 1e-6) -> dict[str, Any]:
    metrics = {"fold_rmse": {fold: row["rmse"] for fold, row in recompute_cv_metrics(StreamInputs(release_root=release_root)).items()}}
    rows = []
    ok = True
    for fold, expected in EXPECTED_FOLD_RMSE.items():
        actual = metrics["fold_rmse"].get(fold)
        diff = abs(float(actual) - expected) if actual is not None else float("inf")
        status = "passed" if diff <= tolerance else "failed"
        ok = ok and status == "passed"
        rows.append({"fold_id": fold, "actual_rmse_mm": actual, "expected_rmse_mm": expected, "abs_diff": diff, "status": status})
    return {"formal_cv_recalculation_status": "passed" if ok else "failed", "rows": rows, "source_level_recalculation": True}


def recalculate_final_refit(release_root: Path = RELEASE_ROOT, expected: dict[str, float] | None = None, tolerance: float = 1e-6) -> dict[str, Any]:
    from .constants import EXPECTED_FINAL
    from .source_recompute import recompute_final_metrics

    expected = expected or EXPECTED_FINAL
    actual = recompute_final_metrics(StreamInputs(release_root=release_root))
    rows = {}
    ok = True
    mapping = {"rmse": "rmse", "mae": "mae", "Ske_min": "Ske_min", "Ske_p50": "Ske_p50", "Ske_max": "Ske_max", "Cu_global": "Cu_global"}
    for key, actual_key in mapping.items():
        diff = abs(float(actual[actual_key]) - float(expected[key]))
        rows[key] = {"actual": actual[actual_key], "expected": expected[key], "abs_diff": diff}
        ok = ok and diff <= tolerance
    return {"final_refit_recalculation_status": "passed" if ok else "failed", "metrics": rows, "source_level_recalculation": True}
