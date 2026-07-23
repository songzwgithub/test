#!/usr/bin/env python3
"""Audit whether L01028 complete_results contain real scientific outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REFERENCE_DIR = ROOT / "outputs" / "reference_frames" / "L01028_500m_fixed_quality_median_v1"
COMPLETE = REFERENCE_DIR / "complete_results"
AUDIT = REFERENCE_DIR / "L01028_existing_complete_results_audit.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def expected_parameter_count() -> int:
    # The frozen L01028 formal protocol fixes lag_u=10 days, so lag_u is not a
    # free optimized parameter in the fold0 helper used by the formal runs.
    return 1 + 32 + 1 + 1


def audit_npy(path: Path, expected: int) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        out.update({"valid": False, "reason": "missing"})
        return out
    arr = np.load(path)
    finite = np.isfinite(arr)
    out.update(
        {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": int(arr.size),
            "finite_count": int(finite.sum()),
            "minimum": float(np.nanmin(arr)) if arr.size else None,
            "maximum": float(np.nanmax(arr)) if arr.size else None,
            "sha256": sha256_file(path),
            "all_zero": bool(arr.size > 0 and np.all(arr == 0)),
            "valid": bool(arr.size == expected and finite.all() and arr.size > 0 and not np.all(arr == 0)),
        }
    )
    if not out["valid"]:
        out["reason"] = "parameter_array_invalid_or_placeholder"
    return out


def finite_metric(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def audit_fold(fold_dir: Path, expected: int) -> dict[str, Any]:
    metrics = read_json(fold_dir / "metrics.json") if (fold_dir / "metrics.json").exists() else {}
    conv = read_json(fold_dir / "convergence.json") if (fold_dir / "convergence.json").exists() else {}
    inputs = read_json(fold_dir / "input_hashes.json") if (fold_dir / "input_hashes.json").exists() else {}
    params = audit_npy(fold_dir / "parameters.npy", expected)
    metric_ok = all(finite_metric(metrics.get(k)) for k in ("train_rmse", "validation_rmse", "train_mae", "validation_mae"))
    iterations = conv.get("actual_iterations") or conv.get("iterations") or 0
    conv_ok = isinstance(iterations, (int, float)) and iterations > 0 and conv.get("status") not in {None, "not_refit_yet"}
    valid = bool(params["valid"] and metric_ok and conv_ok and inputs)
    return {
        "fold_dir": str(fold_dir),
        "metrics_exists": (fold_dir / "metrics.json").exists(),
        "convergence_exists": (fold_dir / "convergence.json").exists(),
        "input_hashes_exists": (fold_dir / "input_hashes.json").exists(),
        "parameters": params,
        "finite_metrics": metric_ok,
        "real_convergence": conv_ok,
        "valid": valid,
        "reason": None if valid else "formal_fold_result_invalid_or_incomplete",
    }


def audit_dir(path: Path, required: list[str]) -> dict[str, Any]:
    files = {name: (path / name).exists() for name in required}
    return {"path": str(path), "exists": path.exists(), "required_files": files, "valid": path.exists() and all(files.values())}


def audit_acceptance(path: Path, allowed: set[str] | None = None) -> dict[str, Any]:
    allowed = allowed or {"passed"}
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        out.update({"valid": False, "reason": "missing"})
        return out
    data = read_json(path)
    status = data.get("acceptance_status") or data.get("complete_pipeline_status")
    out.update({"status": status, "valid": status in allowed, "payload": data})
    if not out["valid"]:
        out["reason"] = "acceptance_not_passed"
    return out


def audit_geotiffs(path: Path, names: list[str]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    valid = path.exists()
    reference_shape = None
    reference_crs = None
    reference_transform = None
    for name in names:
        tif = path / name
        item: dict[str, Any] = {"path": str(tif), "exists": tif.exists()}
        if not tif.exists():
            item.update({"valid": False, "reason": "missing"})
            valid = False
        else:
            try:
                finite = 0
                with rasterio.open(tif) as src:
                    if reference_shape is None:
                        reference_shape = (src.height, src.width)
                        reference_crs = src.crs
                        reference_transform = src.transform
                    same_geometry = (src.height, src.width) == reference_shape and src.crs == reference_crs and src.transform == reference_transform
                    for _, window in src.block_windows(1):
                        finite += int(np.count_nonzero(np.isfinite(src.read(1, window=window))))
                item.update({"valid": bool(same_geometry and finite > 0), "finite_count": finite, "sha256": sha256_file(tif)})
                valid = valid and item["valid"]
            except Exception as exc:
                item.update({"valid": False, "reason": str(exc)})
                valid = False
        files[name] = item
    return {"path": str(path), "exists": path.exists(), "files": files, "valid": bool(valid)}


def audit_nonempty_files(path: Path, names: list[str]) -> dict[str, Any]:
    files = {}
    valid = path.exists()
    for name in names:
        f = path / name
        ok = f.exists() and f.stat().st_size > 0
        files[name] = {"path": str(f), "exists": f.exists(), "size_bytes": f.stat().st_size if f.exists() else 0, "valid": ok, "sha256": sha256_file(f) if ok else None}
        valid = valid and ok
    return {"path": str(path), "exists": path.exists(), "files": files, "valid": bool(valid)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default=str(REFERENCE_DIR))
    parser.add_argument("--complete-results", default=None)
    args = parser.parse_args()
    ref = Path(args.reference_dir)
    complete = Path(args.complete_results) if args.complete_results else ref / "complete_results"
    expected = expected_parameter_count()
    folds = {}
    for fold_id in [1, 2, 3, 4]:
        folds[f"fold_{fold_id:02d}"] = audit_fold(complete / "formal_cv" / f"fold_{fold_id:02d}", expected)
    final = audit_dir(
        complete / "final_full_data_refit",
        ["final_parameters.npy", "final_parameters.json", "final_convergence.json", "final_input_hashes.json", "final_fit_metrics.json"],
    )
    if (complete / "final_full_data_refit" / "final_parameters.npy").exists():
        final["parameters"] = audit_npy(complete / "final_full_data_refit" / "final_parameters.npy", expected)
        final["valid"] = bool(final["valid"] and final["parameters"]["valid"])
    parameter_products = audit_geotiffs(
        complete / "parameter_products",
        ["Ske.tif", "logSke_spatial_contribution.tif", "predicted_annual_real_mm.tif", "predicted_annual_imag_mm.tif", "residual_annual_real_mm.tif", "residual_annual_imag_mm.tif", "residual_amplitude_mm.tif"],
    )
    parameter_products["acceptance"] = audit_acceptance(complete / "parameter_products" / "parameter_products_acceptance.json")
    parameter_products["valid"] = bool(parameter_products["valid"] and parameter_products["acceptance"]["valid"])
    products = {
        "parameter_products": parameter_products,
        "storage_products": {
            **audit_geotiffs(complete / "storage_products", ["elastic_storage_index_dimensionless.tif"]),
            "acceptance": audit_acceptance(complete / "storage_products" / "storage_products_acceptance.json", {"passed", "passed_with_limitation"}),
        },
        "uncertainty": {
            **audit_nonempty_files(complete / "uncertainty_and_sensitivity", ["fold_parameter_uncertainty.csv", "uncertainty_acceptance.json"]),
            "acceptance": audit_acceptance(complete / "uncertainty_and_sensitivity" / "uncertainty_acceptance.json"),
        },
        "tables": {
            **audit_nonempty_files(complete / "publication_tables", ["table_formal_cv_metrics.csv", "table_final_refit_metrics.csv", "publication_tables_acceptance.json"]),
            "acceptance": audit_acceptance(complete / "publication_tables" / "publication_tables_acceptance.json"),
        },
        "figures": {
            **audit_nonempty_files(complete / "publication_figures", ["figure_formal_cv_rmse.png", "figure_final_Ske_map.png", "publication_figures_acceptance.json"]),
            "acceptance": audit_acceptance(complete / "publication_figures" / "publication_figures_acceptance.json"),
        },
    }
    for name in ["storage_products", "uncertainty", "tables", "figures"]:
        products[name]["valid"] = bool(products[name]["valid"] and products[name]["acceptance"]["valid"])
    stage_status = {}
    for path in sorted((complete / "stage_status").glob("*.json")):
        data = read_json(path)
        stage_status[path.stem] = data
        if data.get("status") == "completed" and not data.get("acceptance_path"):
            data["semantic_issue"] = "completed_without_acceptance_path"
    failures = []
    if not all(item["valid"] for item in folds.values()):
        failures.append("formal_cv_folds_invalid")
    if not final["valid"]:
        failures.append("final_full_data_refit_invalid")
    for name, item in products.items():
        if not item["valid"]:
            failures.append(f"{name}_invalid")
    if any(data.get("semantic_issue") for data in stage_status.values()):
        failures.append("stage_status_semantics_invalid")
    payload = {
        "audit_status": "passed" if not failures else "failed",
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected_parameter_count": expected,
        "folds": folds,
        "final_full_data_refit": final,
        "derived_products": products,
        "stage_status": stage_status,
        "empty_or_placeholder_detected": bool(failures),
        "failure_reasons": failures,
    }
    write_json(ref / "L01028_existing_complete_results_audit.json", payload)
    print(json.dumps({"audit_status": payload["audit_status"], "failure_reasons": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
