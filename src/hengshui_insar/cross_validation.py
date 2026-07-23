"""Formal cross-validation release checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import EXPECTED_FOLD_RMSE, RELEASE_ROOT
from .io import read_json


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


def recalculate_formal_cv(release_root: Path = RELEASE_ROOT, tolerance: float = 1e-12) -> dict[str, Any]:
    metrics = formal_cv_metrics(release_root)
    rows = []
    ok = True
    for fold, expected in EXPECTED_FOLD_RMSE.items():
        actual = metrics["fold_rmse"].get(fold)
        diff = abs(float(actual) - expected) if actual is not None else float("inf")
        status = "passed" if diff <= tolerance else "failed"
        ok = ok and status == "passed"
        rows.append({"fold_id": fold, "actual_rmse_mm": actual, "expected_rmse_mm": expected, "abs_diff": diff, "status": status})
    return {"formal_cv_recalculation_status": "passed" if ok else "failed", "rows": rows}
