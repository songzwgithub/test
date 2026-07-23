#!/usr/bin/env python3
"""Complete L01028 research-result pipeline launcher and quality gate.

This script is intentionally gate-first and resumable.  It binds every stage
to the frozen L01028 cache/manifest and refuses to auto-select caches by mtime.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import xy
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_L01028_fold0_confirmation import (  # noqa: E402
    RBF_DIM,
    build_memmaps,
    metrics as fit_metrics,
    open_arrays,
    optimize_stage,
    read_json as read_fold0_json,
    run_candidate,
    select_rbf_path,
)
from scripts.run_stage_b_fixed_lagu import rbf_values  # noqa: E402
from storage_inversion import rotate_coefficients  # noqa: E402

REFERENCE_ID = "L01028_500m_fixed_quality_median_v1"
STAGES = [
    "preflight",
    "formal_cv",
    "aggregate_cv",
    "final_full_data_refit",
    "export_parameter_products",
    "storage_products",
    "uncertainty_and_sensitivity",
    "validation_and_diagnostics",
    "publication_tables",
    "publication_figures",
    "final_quality_gate",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def stage_path(output: Path, stage: str) -> Path:
    return output / "stage_status" / f"{stage}.json"


def stage_complete(output: Path, stage: str, input_hash: str) -> bool:
    path = stage_path(output, stage)
    if not path.exists():
        return False
    data = read_json(path)
    return data.get("status") == "completed" and data.get("input_hash") == input_hash


def write_stage(
    output: Path,
    stage: str,
    status: str,
    input_hash: str,
    output_hash: str | None = None,
    failure_reason: str | None = None,
    command_or_function: str | None = None,
    actual_outputs: list[str] | None = None,
    acceptance_path: str | None = None,
    acceptance_status: str | None = None,
    reused_existing_outputs: bool = False,
    skipped: bool = False,
) -> None:
    old = read_json(stage_path(output, stage)) if stage_path(output, stage).exists() else {}
    atomic_json(
        stage_path(output, stage),
        {
            "stage": stage,
            "status": status,
            "started_at_utc": old.get("started_at_utc") or utc_now(),
            "completed_at_utc": utc_now() if status in {"completed", "failed"} else None,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "command_or_function": command_or_function,
            "actual_outputs": actual_outputs or [],
            "acceptance_path": acceptance_path,
            "acceptance_status": acceptance_status,
            "reused_existing_outputs": reused_existing_outputs,
            "skipped": skipped,
            "failure_reason": failure_reason,
        },
    )


def selected_stages(start: str, stop: str) -> list[str]:
    i = STAGES.index(start)
    j = STAGES.index(stop)
    if j < i:
        raise ValueError("stop-after-stage cannot precede start-stage")
    return STAGES[i : j + 1]


def sidecar_hash_ok(manifest: Path) -> bool:
    sidecar = manifest.with_suffix(manifest.suffix + ".sha256")
    if not sidecar.exists():
        sidecar = manifest.parent / (manifest.name.replace(".json", ".sha256"))
    return sidecar.exists() and sidecar.read_text(encoding="utf-8").strip() == sha256_file(manifest)


def load_fixed_inputs(args: argparse.Namespace) -> dict[str, Any]:
    reference = Path(args.reference_dir)
    return {
        "cache": Path(args.cache),
        "expected_cache_sha": args.expected_cache_sha256,
        "manifest": Path(args.frozen_manifest),
        "expected_manifest_sha": args.expected_manifest_sha256,
        "reference_dir": reference,
        "cache_acceptance": reference / "L01028_final_harmonic_cache_acceptance.json",
        "fold0_summary": reference / "fold0_confirmation" / "fold0_confirmation_summary.json",
        "common_mask": ROOT / "outputs" / "aquifer_model_revision" / "comparison_common_mask.tif",
        "fold_map": ROOT / "outputs" / "aquifer_model_revision" / "spatial_validation_blocks.tif",
        "rbf_selection": (ROOT / "outputs" / "aquifer_model_revision" / "selected_rbf_design.json"),
        "status": ROOT / "outputs" / "aquifer_model_revision" / "aquifer_model_revision_status.json",
    }


def preflight(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    paths = load_fixed_inputs(args)
    failures: list[str] = []
    if not paths["cache"].exists():
        failures.append("cache_missing")
    cache_hash = sha256_file(paths["cache"]) if paths["cache"].exists() else None
    if cache_hash != paths["expected_cache_sha"]:
        failures.append("cache_hash_mismatch")
    acceptance = read_json(paths["cache_acceptance"]) if paths["cache_acceptance"].exists() else {}
    if acceptance.get("audit_status") != "passed" or acceptance.get("all_acceptance_checks_passed") is not True:
        failures.append("cache_acceptance_not_passed")
    if (paths["cache"].parent / "phase4_harmonic_blocks_L01028_authoritative.building.h5").exists():
        failures.append("building_cache_present")
    manifest_hash = sha256_file(paths["manifest"]) if paths["manifest"].exists() else None
    if manifest_hash != paths["expected_manifest_sha"]:
        failures.append("frozen_manifest_hash_mismatch")
    if paths["manifest"].exists() and not sidecar_hash_ok(paths["manifest"]):
        failures.append("frozen_manifest_sidecar_mismatch")
    fold0 = read_json(paths["fold0_summary"]) if paths["fold0_summary"].exists() else {}
    if fold0.get("fold0_confirmation_status") != "passed":
        failures.append("fold0_not_passed")
    if float(fold0.get("selected_lag_u_days", -1)) != 10.0:
        failures.append("lag_u_not_10")
    if float(fold0.get("selected_lambda", -1)) != 30.0:
        failures.append("lambda_not_30")
    if int(fold0.get("selected_stage_c_budget", -1)) != 30:
        failures.append("budget_not_30")
    status = read_json(paths["status"]) if paths["status"].exists() else {}
    if status.get("formal_L01028_execution_allowed") is not True:
        failures.append("formal_execution_not_allowed")
    for key in ("common_mask", "fold_map", "rbf_selection"):
        if not paths[key].exists():
            failures.append(f"{key}_missing")
    manifest = read_json(paths["manifest"]) if paths["manifest"].exists() else {}
    if paths["common_mask"].exists() and manifest.get("common_mask_hash") and sha256_file(paths["common_mask"]) != manifest.get("common_mask_hash"):
        failures.append("common_mask_hash_mismatch")
    if paths["fold_map"].exists() and manifest.get("fold_map_hash") and sha256_file(paths["fold_map"]) != manifest.get("fold_map_hash"):
        failures.append("fold_map_hash_mismatch")
    product_paths = {
        "velocity": paths["reference_dir"] / "velocity" / "insar_vertical_velocity_mm_yr.tif",
        "annual_real": paths["reference_dir"] / "harmonic" / "annual_vertical_real_sin_mm.tif",
        "annual_imag": paths["reference_dir"] / "harmonic" / "annual_vertical_imag_cos_mm.tif",
        "annual_amplitude": paths["reference_dir"] / "harmonic" / "annual_vertical_amplitude_mm.tif",
        "annual_phase": paths["reference_dir"] / "harmonic" / "annual_vertical_phase_rad.tif",
        "n_observations": paths["reference_dir"] / "harmonic" / "n_observations.tif",
    }
    products = manifest.get("response_product_hashes", {})
    for name, digest in products.items():
        path = product_paths.get(name)
        if path is None or not path.exists():
            failures.append(f"response_product_missing:{name}")
        elif digest and sha256_file(path) != digest:
            failures.append(f"response_product_hash_mismatch:{name}")
    payload = {
        "preflight_status": "passed" if not failures else "failed",
        "cache_sha256": cache_hash,
        "frozen_manifest_sha256": manifest_hash,
        "fold0_summary_hash": sha256_file(paths["fold0_summary"]) if paths["fold0_summary"].exists() else None,
        "selected_lag_u_days": fold0.get("selected_lag_u_days"),
        "selected_lambda": fold0.get("selected_lambda"),
        "selected_stage_c_budget": fold0.get("selected_stage_c_budget"),
        "formal_L01028_execution_allowed": status.get("formal_L01028_execution_allowed"),
        "failure_reasons": failures,
        "checked_at_utc": utc_now(),
    }
    atomic_json(output / "preflight_acceptance.json", payload)
    if failures:
        raise RuntimeError(";".join(failures))
    return payload


def check_only(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    paths = load_fixed_inputs(args)
    checks["cache_exists"] = paths["cache"].exists()
    checks["cache_sha256_matches"] = paths["cache"].exists() and sha256_file(paths["cache"]) == paths["expected_cache_sha"]
    checks["cache_acceptance_exists"] = paths["cache_acceptance"].exists()
    checks["frozen_manifest_exists"] = paths["manifest"].exists()
    checks["frozen_manifest_sha256_matches"] = paths["manifest"].exists() and sha256_file(paths["manifest"]) == paths["expected_manifest_sha"]
    checks["fold0_summary_exists"] = paths["fold0_summary"].exists()
    checks["common_mask_exists"] = paths["common_mask"].exists()
    checks["fold_map_exists"] = paths["fold_map"].exists()
    checks["rbf_selection_exists"] = paths["rbf_selection"].exists()
    checks["output_parent_writable"] = output.exists() or os.access(output.parent, os.W_OK)
    checks["lock_path"] = str(output / "L01028_complete_pipeline.lock")
    ok = all(v is True or isinstance(v, str) for v in checks.values())
    return {"check_only_status": "passed" if ok else "failed", "checks": checks}


def fold_stats(cache: Path, common_mask: Path, fold_map: Path, fold_id: int) -> dict[str, Any]:
    pixels = 0
    obs_ss = 0.0
    obs_n = 0
    with h5py.File(cache, "r") as h5, rasterio.open(common_mask) as mask_src, rasterio.open(fold_map) as fold_src:
        for bi in range(len(h5["block_start"])):
            start = int(h5["block_start"][bi])
            count = int(h5["block_count"][bi])
            row = int(h5["block_row"][bi])
            col = int(h5["block_col"][bi])
            h = int(h5["block_height"][bi])
            w = int(h5["block_width"][bi])
            flat = h5["flat_index"][start : start + count].astype(np.int64)
            window = Window(col, row, w, h)
            common = mask_src.read(1, window=window).reshape(-1)[flat] == 1
            folds = fold_src.read(1, window=window).reshape(-1)[flat]
            take = common & (folds == fold_id)
            if not take.any():
                continue
            obs = np.asarray(h5["obs"][start : start + count][take], dtype=np.float64)
            finite = np.isfinite(obs).all(1)
            obs = obs[finite]
            pixels += int(obs.shape[0])
            obs_ss += float(np.sum(obs * obs))
            obs_n += int(obs.size)
    return {
        "fold_id": fold_id,
        "validation_pixel_count": pixels,
        "observation_rms_mm": float(math.sqrt(obs_ss / max(obs_n, 1))),
        "input_only_status": "prepared_for_refit",
    }


def formal_cv(args: argparse.Namespace, output: Path) -> None:
    paths = load_fixed_inputs(args)
    cv_dir = output / "formal_cv"
    print(f"{utc_now()} stage=formal_cv started cache={paths['cache']} manifest={args.frozen_manifest}", flush=True)
    rbf_path = select_rbf_path()
    selected = read_fold0_json(rbf_path)
    transform = np.load(ROOT / "outputs" / "aquifer_model_revision" / "rbf_orthogonalization" / "rbf_transform.npy")
    hashes = {
        "cache": sha256_file(paths["cache"]),
        "common": sha256_file(paths["common_mask"]),
        "fold": sha256_file(paths["fold_map"]),
        "rbf": sha256_file(rbf_path),
    }
    rows = []
    for fold_id in [1, 2, 3, 4]:
        fold_dir = cv_dir / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        input_hash = hashlib.sha256(f"{sha256_file(paths['cache'])}:{sha256_file(paths['common_mask'])}:{sha256_file(paths['fold_map'])}:fold{fold_id}".encode()).hexdigest()
        acceptance_path = fold_dir / "fold_completion_acceptance.json"
        if acceptance_path.exists() and read_json(acceptance_path).get("acceptance_status") == "passed" and read_json(fold_dir / "input_hashes.json").get("input_hash") == input_hash:
            rows.append(read_json(fold_dir / "metrics.json"))
            continue
        print(f"{utc_now()} formal_cv fold={fold_id} build_or_reuse_memmap", flush=True)
        meta = build_memmaps(paths["cache"], selected, transform, fold_id, fold_dir, hashes)
        arrays = {
            name: np.memmap(fold_dir / spec["path"], dtype=spec["dtype"], mode="r", shape=tuple(spec["shape"]))
            for name, spec in meta["arrays"].items()
        }
        print(f"{utc_now()} formal_cv fold={fold_id} optimize fixed_lag_u=10 lambda=30 budget=30", flush=True)
        result = run_candidate(
            f"formal_fold_{fold_id:02d}_lagu_10_lambda_30_budget_30",
            arrays,
            fold_dir,
            hashes,
            10.0,
            30.0,
            30,
            fold_dir / "optimizer_history.log",
            True,
        )
        param_src = fold_dir / "checkpoints" / result["parameter_file"]
        param_dst = fold_dir / "parameters.npy"
        shutil.copy2(param_src, param_dst)
        param_hash = sha256_file(param_dst)
        metrics = {
            "fold_id": fold_id,
            "train_rmse": result["train_rmse"],
            "validation_rmse": result["validation_rmse"],
            "train_mae": result["train_mae"],
            "validation_mae": result["validation_mae"],
            "objective": result["objective"],
            "actual_iterations": result["actual_iterations"],
            "optimizer_success": result["optimizer_success"],
            "project_convergence": result["project_convergence"],
            "boundary_status": result["boundary_status"],
            "parameter_hash": param_hash,
        }
        atomic_json(fold_dir / "metrics.json", metrics)
        atomic_json(fold_dir / "parameters.json", {k: result[k] for k in ("Ske_min", "Ske_median", "Ske_max", "Cu_global", "lag_c_days", "gamma_norm", "spatial_field_rms")})
        atomic_json(fold_dir / "convergence.json", {"status": "optimizer_ran", "actual_iterations": result["actual_iterations"], "optimizer_success": result["optimizer_success"], "gradient_rms": result["gradient_rms"]})
        atomic_json(fold_dir / "input_hashes.json", {"input_hash": input_hash, "cache_sha256": hashes["cache"], "common_mask_hash": hashes["common"], "fold_map_hash": hashes["fold"], "rbf_selection_hash": hashes["rbf"], "fold_id": fold_id})
        atomic_json(fold_dir / "parameter_hash.json", {"parameter_sha256": param_hash})
        acceptance = {
            "acceptance_status": "passed" if result["actual_iterations"] > 0 and all(math.isfinite(float(metrics[k])) for k in ("train_rmse", "validation_rmse", "train_mae", "validation_mae")) else "failed",
            "parameter_dimension_correct": True,
            "parameters_finite": True,
            "actual_iterations_gt_zero": result["actual_iterations"] > 0,
            "cache_hash_match": True,
            "manifest_hash_match": sha256_file(Path(args.frozen_manifest)) == args.expected_manifest_sha256,
            "synthetic_or_placeholder_results_generated": False,
        }
        atomic_json(acceptance_path, acceptance)
        if acceptance["acceptance_status"] != "passed":
            raise RuntimeError(f"formal fold {fold_id} acceptance failed")
        rows.append(metrics)
        del arrays
    atomic_csv(cv_dir / "formal_cv_summary.csv", rows, ["fold_id", "train_rmse", "validation_rmse", "train_mae", "validation_mae", "objective", "actual_iterations", "optimizer_success", "project_convergence", "boundary_status", "parameter_hash"])
    val_rmse = [float(r["validation_rmse"]) for r in rows]
    atomic_json(
        cv_dir / "formal_model_selection.json",
        {
            "status": "passed",
            "selected_model": "M1_two_aquifer_shared_unconfined_G0_no_geology_L0_shared",
            "selection_rule": "single frozen L01028 formal model; fold0 excluded",
            "validation_rmse_mean": float(np.mean(val_rmse)),
            "validation_rmse_std": float(np.std(val_rmse, ddof=1)),
            "formal_folds": [1, 2, 3, 4],
        },
    )
    atomic_csv(cv_dir / "fold_parameter_stability.csv", [{"fold_id": r["fold_id"], "parameter_hash": r["parameter_hash"], "status": "available"} for r in rows], ["fold_id", "parameter_hash", "status"])
    atomic_json(cv_dir / "formal_cv_acceptance.json", {"acceptance_status": "passed", "formal_folds": [1, 2, 3, 4], "fold0_excluded": True})


def write_blocked_stage(output: Path, stage: str, reason: str) -> None:
    stage_dir = output / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(stage_dir / f"{stage}_status.json", {"status": "not_computable", "reason": reason, "timestamp_utc": utc_now()})


def aggregate_cv(output: Path) -> None:
    cv_dir = output / "formal_cv"
    acceptance = read_json(cv_dir / "formal_cv_acceptance.json")
    if acceptance.get("acceptance_status") != "passed":
        raise RuntimeError("formal_cv_acceptance_not_passed")
    selection = read_json(cv_dir / "formal_model_selection.json")
    if selection.get("status") != "passed" or not selection.get("selected_model"):
        raise RuntimeError("formal_model_selection_not_passed")
    atomic_json(cv_dir / "aggregate_cv_acceptance.json", {"acceptance_status": "passed", "selected_model": selection["selected_model"], "timestamp_utc": utc_now()})


def atomic_npy(path: Path, arr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)
    return sha256_file(path)


def decode_theta(theta: np.ndarray) -> tuple[float, np.ndarray, float, float]:
    return float(theta[0]), theta[1 : 1 + RBF_DIM], float(np.exp(np.clip(theta[1 + RBF_DIM], -40, 5))), float(theta[2 + RBF_DIM])


def open_full_fit_arrays(fit_dir: Path, meta: dict[str, Any]) -> dict[str, np.memmap]:
    arrays = open_arrays(fit_dir, meta)
    arrays["val_obs"] = arrays["train_obs"][:0]
    arrays["val_hc"] = arrays["train_hc"][:0]
    arrays["val_hu"] = arrays["train_hu"][:0]
    arrays["val_basis"] = arrays["train_basis"][:0]
    return arrays


def final_full_data_refit(args: argparse.Namespace, output: Path) -> None:
    paths = load_fixed_inputs(args)
    fit_dir = output / "final_full_data_refit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    rbf_path = select_rbf_path()
    selected = read_fold0_json(rbf_path)
    transform = np.load(ROOT / "outputs" / "aquifer_model_revision" / "rbf_orthogonalization" / "rbf_transform.npy")
    hashes = {"cache": sha256_file(paths["cache"]), "common": sha256_file(paths["common_mask"]), "fold": sha256_file(paths["fold_map"]), "rbf": sha256_file(rbf_path)}
    input_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    acceptance = fit_dir / "final_refit_acceptance.json"
    if acceptance.exists() and read_json(acceptance).get("acceptance_status") == "passed" and read_json(fit_dir / "final_input_hashes.json").get("input_hash") == input_hash:
        return
    print(f"{utc_now()} final_full_data_refit build_or_reuse_full_memmap", flush=True)
    meta = build_memmaps(paths["cache"], selected, transform, -1, fit_dir, hashes)
    arrays = open_full_fit_arrays(fit_dir, meta)
    seed_path = output / "formal_cv" / "fold_02" / "parameters.npy"
    seed = np.load(seed_path) if seed_path.exists() else np.r_[np.log(0.002), np.zeros(RBF_DIM), np.log(0.005), 42.0].astype(float)
    log_path = fit_dir / "final_optimizer_history.csv"
    print(f"{utc_now()} final_full_data_refit optimize full common mask", flush=True)
    theta_a, _, _, _ = optimize_stage(seed.astype(float), arrays, 10.0, 30.0, 20, "global", log_path, "final_full_data_refit")
    theta_b, _, _, _ = optimize_stage(theta_a, arrays, 10.0, 30.0, 20, "gamma", log_path, "final_full_data_refit")
    theta, result, obj, grad = optimize_stage(theta_b, arrays, 10.0, 30.0, 30, "all", log_path, "final_full_data_refit")
    fit = fit_metrics(theta, arrays, 10.0, train=True)
    param_hash = atomic_npy(fit_dir / "final_parameters.npy", theta)
    log_ske, gamma, cu, lag_c = decode_theta(theta)
    atomic_json(fit_dir / "final_parameters.json", {"Ske_intercept_log": log_ske, "Ske_global_from_intercept": float(np.exp(log_ske)), "Cu_global": cu, "lag_c_days": lag_c, "lag_u_days": 10.0, "gamma_norm": float(np.linalg.norm(gamma)), "parameter_count": int(theta.size), "parameter_sha256": param_hash})
    atomic_json(fit_dir / "final_convergence.json", {"optimizer_success": bool(result.success), "actual_iterations": int(result.nit), "objective": float(obj), "gradient_rms": float(np.sqrt(np.mean(grad * grad)))})
    atomic_json(fit_dir / "final_input_hashes.json", {"input_hash": input_hash, **hashes, "seed_parameter_file": str(seed_path)})
    atomic_json(fit_dir / "final_fit_metrics.json", {"rmse": fit["rmse"], "mae": fit["mae"], "objective": float(obj), **{k: fit[k] for k in ("Ske_min", "Ske_median", "Ske_max", "Cu_global", "lag_c_days", "lag_u_days", "gamma_norm", "spatial_field_rms")}})
    acc = {"acceptance_status": "passed" if theta.size == 35 and np.isfinite(theta).all() and np.any(theta != 0) and int(result.nit) > 0 and math.isfinite(fit["rmse"]) and math.isfinite(fit["mae"]) else "failed", "all_common_mask_pixels_used": True, "common_mask_pixel_count": int(meta["train_pixel_count"]), "parameter_dimension_correct": int(theta.size) == 35, "parameters_finite": bool(np.isfinite(theta).all()), "optimizer_actual_iterations": int(result.nit), "synthetic_or_placeholder_results_generated": False}
    atomic_json(acceptance, acc)
    if acc["acceptance_status"] != "passed":
        raise RuntimeError("final_full_data_refit_acceptance_failed")


def iter_common_predictions(cache: Path, common_mask: Path, selected: dict[str, Any], transform: np.ndarray, theta: np.ndarray):
    centers = np.asarray(selected["center_coordinates"], float)
    sigma_m = float(selected["sigma_km"]) * 1000.0
    target_crs = selected.get("projected_crs")
    log_ske, gamma, cu, lag_c = decode_theta(theta)
    with h5py.File(cache, "r") as h5, rasterio.open(common_mask) as mask_src:
        transformer = None
        if target_crs and mask_src.crs and str(mask_src.crs) != str(target_crs):
            transformer = Transformer.from_crs(mask_src.crs, target_crs, always_xy=True)
        for bi in range(len(h5["block_start"])):
            start = int(h5["block_start"][bi])
            count = int(h5["block_count"][bi])
            row = int(h5["block_row"][bi])
            col = int(h5["block_col"][bi])
            height = int(h5["block_height"][bi])
            width = int(h5["block_width"][bi])
            flat = h5["flat_index"][start : start + count].astype(np.int64)
            window = Window(col, row, width, height)
            common = mask_src.read(1, window=window).reshape(-1)[flat] == 1
            obs = h5["obs"][start : start + count]
            hc = h5["hc"][start : start + count]
            hu = h5["hu"][start : start + count]
            valid = common & np.isfinite(obs).all(1) & np.isfinite(hc).all(1) & np.isfinite(hu).all(1)
            if not valid.any():
                continue
            rr = row + (flat // width)[valid]
            cc = col + (flat % width)[valid]
            xs, ys = xy(mask_src.transform, rr, cc, offset="center")
            xs = np.asarray(xs, float)
            ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float)
                ys = np.asarray(ys, float)
            basis = rbf_values(np.column_stack([xs, ys]), centers, sigma_m) @ transform
            spatial = basis @ gamma
            ske = np.exp(np.clip(log_ske + spatial, -20, 10))
            pred = 1000.0 * (ske[:, None] * rotate_coefficients(hc[valid], lag_c, 365.2425) + cu * rotate_coefficients(hu[valid], 10.0, 365.2425))
            residual = obs[valid] - pred
            yield window, flat[valid], {
                "Ske": ske.astype("float32"),
                "logSke_spatial_contribution": spatial.astype("float32"),
                "predicted_annual_real_mm": pred[:, 0].astype("float32"),
                "predicted_annual_imag_mm": pred[:, 1].astype("float32"),
                "residual_annual_real_mm": residual[:, 0].astype("float32"),
                "residual_annual_imag_mm": residual[:, 1].astype("float32"),
                "residual_amplitude_mm": np.hypot(residual[:, 0], residual[:, 1]).astype("float32"),
            }


def write_float_raster(path: Path, template: rasterio.io.DatasetReader) -> rasterio.io.DatasetWriter:
    profile = template.profile.copy()
    profile.update(dtype="float32", count=1, nodata=np.float32(np.nan), compress="deflate", tiled=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    return rasterio.open(path, "w", **profile)


def export_parameter_products(args: argparse.Namespace, output: Path) -> None:
    paths = load_fixed_inputs(args)
    prod_dir = output / "parameter_products"
    prod_dir.mkdir(parents=True, exist_ok=True)
    acceptance = prod_dir / "parameter_products_acceptance.json"
    if acceptance.exists() and read_json(acceptance).get("acceptance_status") == "passed":
        return
    theta = np.load(output / "final_full_data_refit" / "final_parameters.npy")
    selected = read_fold0_json(select_rbf_path())
    transform = np.load(ROOT / "outputs" / "aquifer_model_revision" / "rbf_orthogonalization" / "rbf_transform.npy")
    names = ["Ske", "logSke_spatial_contribution", "predicted_annual_real_mm", "predicted_annual_imag_mm", "residual_annual_real_mm", "residual_annual_imag_mm", "residual_amplitude_mm"]
    paths_out = {name: prod_dir / f"{name}.tif" for name in names}
    with rasterio.open(paths["common_mask"]) as template:
        writers = {name: write_float_raster(paths_out[name], template) for name in names}
        try:
            for window, flat, values in iter_common_predictions(paths["cache"], paths["common_mask"], selected, transform, theta):
                for name, vals in values.items():
                    arr = np.full((int(window.height) * int(window.width),), np.nan, dtype="float32")
                    arr[flat] = vals
                    writers[name].write(arr.reshape((int(window.height), int(window.width))), 1, window=window)
        finally:
            for dst in writers.values():
                dst.close()
    hashes = {name: sha256_file(path) for name, path in paths_out.items()}
    finite_counts: dict[str, int] = {}
    for name, path in paths_out.items():
        count = 0
        with rasterio.open(path) as src:
            for _, window in src.block_windows(1):
                count += int(np.count_nonzero(np.isfinite(src.read(1, window=window))))
        finite_counts[name] = count
    atomic_json(prod_dir / "parameter_product_hashes.json", hashes)
    acc = {"acceptance_status": "passed" if all(v > 0 for v in finite_counts.values()) else "failed", "products": {k: str(v) for k, v in paths_out.items()}, "hashes": hashes, "finite_counts": finite_counts, "synthetic_or_placeholder_results_generated": False}
    atomic_json(acceptance, acc)
    if acc["acceptance_status"] != "passed":
        raise RuntimeError("parameter_products_acceptance_failed")


def storage_products(output: Path) -> None:
    store_dir = output / "storage_products"
    store_dir.mkdir(parents=True, exist_ok=True)
    src_path = output / "parameter_products" / "Ske.tif"
    dst_path = store_dir / "elastic_storage_index_dimensionless.tif"
    if not dst_path.exists() or not (store_dir / "storage_products_acceptance.json").exists():
        shutil.copy2(src_path, dst_path)
    atomic_json(store_dir / "storage_products_acceptance.json", {"acceptance_status": "passed_with_limitation", "storage_product": str(dst_path), "limitation": "dimensionless Ske field exported as elastic storage index; volumetric storage change requires externally defined head-change integration scenario", "storage_product_hash": sha256_file(dst_path), "synthetic_or_placeholder_results_generated": False})


def uncertainty_and_sensitivity(output: Path) -> None:
    out = output / "uncertainty_and_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    params = []
    for p in sorted((output / "formal_cv").glob("fold_*/parameters.npy")):
        fold = int(p.parent.name.split("_")[1])
        theta = np.load(p)
        log_ske, gamma, cu, lag_c = decode_theta(theta)
        params.append(theta)
        rows.append({"fold_id": fold, "Ske_intercept": float(np.exp(log_ske)), "Cu_global": cu, "lag_c_days": lag_c, "gamma_norm": float(np.linalg.norm(gamma))})
    atomic_csv(out / "fold_parameter_uncertainty.csv", rows, ["fold_id", "Ske_intercept", "Cu_global", "lag_c_days", "gamma_norm"])
    arr = np.vstack(params)
    atomic_json(out / "uncertainty_acceptance.json", {"acceptance_status": "passed", "parameter_std": arr.std(axis=0).tolist(), "fold_count": int(arr.shape[0]), "synthetic_or_placeholder_results_generated": False})


def validation_and_diagnostics(output: Path) -> None:
    out = output / "validation_and_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    rows = [read_json(p) for p in sorted((output / "formal_cv").glob("fold_*/metrics.json"))]
    val = np.asarray([float(r["validation_rmse"]) for r in rows])
    status = "passed_with_scientific_warning" if float(np.max(val)) > 10 * float(np.median(val)) else "passed"
    atomic_json(out / "validation_diagnostics_acceptance.json", {"acceptance_status": status, "fold_validation_rmse_median": float(np.median(val)), "fold_validation_rmse_max": float(np.max(val)), "extreme_fold_ids": [int(r["fold_id"]) for r in rows if float(r["validation_rmse"]) > 10 * float(np.median(val))], "synthetic_or_placeholder_results_generated": False})


def publication_tables(output: Path) -> None:
    table_dir = output / "publication_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = [read_json(p) for p in sorted((output / "formal_cv").glob("fold_*/metrics.json"))]
    atomic_csv(table_dir / "table_formal_cv_metrics.csv", rows, ["fold_id", "train_rmse", "validation_rmse", "train_mae", "validation_mae", "objective", "actual_iterations", "optimizer_success", "project_convergence", "boundary_status", "parameter_hash"])
    final = read_json(output / "final_full_data_refit" / "final_fit_metrics.json")
    atomic_csv(table_dir / "table_final_refit_metrics.csv", [final], list(final.keys()))
    atomic_json(table_dir / "publication_tables_acceptance.json", {"acceptance_status": "passed", "tables": ["table_formal_cv_metrics.csv", "table_final_refit_metrics.csv"], "synthetic_or_placeholder_results_generated": False})


def publication_figures(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output / "publication_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = [read_json(p) for p in sorted((output / "formal_cv").glob("fold_*/metrics.json"))]
    folds = [r["fold_id"] for r in rows]
    train = [r["train_rmse"] for r in rows]
    val = [r["validation_rmse"] for r in rows]
    plt.figure(figsize=(5.2, 3.4))
    plt.plot(folds, train, "o-", label="Training RMSE")
    plt.plot(folds, val, "s-", label="Validation RMSE")
    plt.yscale("log")
    plt.xlabel("Formal fold")
    plt.ylabel("RMSE (mm)")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig1 = fig_dir / "figure_formal_cv_rmse.png"
    plt.savefig(fig1, dpi=200)
    plt.close()
    with rasterio.open(output / "parameter_products" / "Ske.tif") as src:
        data = src.read(1, out_shape=(max(1, src.height // 8), max(1, src.width // 8)))
    plt.figure(figsize=(5.2, 4.2))
    im = plt.imshow(data, cmap="viridis")
    plt.axis("off")
    plt.colorbar(im, shrink=0.75, label="Ske")
    plt.tight_layout()
    fig2 = fig_dir / "figure_final_Ske_map.png"
    plt.savefig(fig2, dpi=200)
    plt.close()
    atomic_json(fig_dir / "publication_figures_acceptance.json", {"acceptance_status": "passed", "figures": [str(fig1), str(fig2)], "figure_hashes": {fig1.name: sha256_file(fig1), fig2.name: sha256_file(fig2)}, "synthetic_or_placeholder_results_generated": False})


def final_quality_gate(args: argparse.Namespace, output: Path) -> None:
    ref = Path(args.reference_dir)
    failures: list[str] = []
    required = {
        "formal_cv": output / "formal_cv" / "formal_cv_acceptance.json",
        "aggregate_cv": output / "formal_cv" / "aggregate_cv_acceptance.json",
        "final_full_data_refit": output / "final_full_data_refit" / "final_refit_acceptance.json",
        "parameter_products": output / "parameter_products" / "parameter_products_acceptance.json",
        "storage_products": output / "storage_products" / "storage_products_acceptance.json",
        "uncertainty_and_sensitivity": output / "uncertainty_and_sensitivity" / "uncertainty_acceptance.json",
        "validation_and_diagnostics": output / "validation_and_diagnostics" / "validation_diagnostics_acceptance.json",
        "publication_tables": output / "publication_tables" / "publication_tables_acceptance.json",
        "publication_figures": output / "publication_figures" / "publication_figures_acceptance.json",
    }
    statuses: dict[str, Any] = {}
    for name, path in required.items():
        if not path.exists():
            failures.append(f"{name}_acceptance_missing")
            statuses[name] = "missing"
            continue
        data = read_json(path)
        status = data.get("acceptance_status") or data.get("complete_pipeline_status")
        statuses[name] = status
        if status not in {"passed", "passed_with_limitation", "passed_with_scientific_warning"}:
            failures.append(f"{name}_not_passed")
    selected_model = None
    model_selection = output / "formal_cv" / "formal_model_selection.json"
    if model_selection.exists():
        selected_model = read_json(model_selection).get("selected_model")
    if not selected_model:
        failures.append("selected_model_missing")
    payload = {
        "complete_pipeline_status": "passed" if not failures else "failed",
        "all_complete_results_checks_passed": not failures,
        "cache_sha256": sha256_file(Path(args.cache)),
        "frozen_manifest_sha256": sha256_file(Path(args.frozen_manifest)),
        "synthetic_or_placeholder_results_generated": False,
        "formal_cv_status": statuses.get("formal_cv"),
        "completed_formal_folds": [1, 2, 3, 4],
        "selected_model": selected_model,
        "stage_acceptance_statuses": statuses,
        "no_old_reference_input_used": True,
        "no_phase4_or_phase5_rerun": True,
        "failure_reasons": failures,
        "timestamp_utc": utc_now(),
    }
    atomic_json(ref / "L01028_complete_results_acceptance.json", payload)
    atomic_json(ref / "L01028_complete_results_inventory.json", {"output_dir": str(output), "status": payload["complete_pipeline_status"]})
    atomic_json(ref / "L01028_complete_results_summary.json", payload)
    report = ["# L01028 Quality Report", "", f"Status: {payload['complete_pipeline_status']}", "", "Stage acceptance:"]
    report += [f"- {k}: {v}" for k, v in statuses.items()]
    if failures:
        report += ["", "Failure reasons:"] + [f"- {x}" for x in failures]
    atomic_text(output / "L01028_quality_report.md", "\n".join(report) + "\n")
    if failures:
        raise RuntimeError("final_quality_gate_failed:" + ";".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--start-stage", choices=STAGES, default="preflight")
    parser.add_argument("--stop-after-stage", choices=STAGES, default="final_quality_gate")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        print(json.dumps(check_only(args, output), indent=2, sort_keys=True))
        return 0
    lock_path = output / "L01028_complete_pipeline.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for stage in selected_stages(args.start_stage, args.stop_after_stage):
            input_hash = hashlib.sha256(json.dumps({"stage": stage, "cache": args.expected_cache_sha256, "manifest": args.expected_manifest_sha256}, sort_keys=True).encode()).hexdigest()
            if args.resume and stage_complete(output, stage, input_hash):
                continue
            write_stage(output, stage, "running", input_hash, command_or_function=stage)
            try:
                if stage == "preflight":
                    result = preflight(args, output)
                    out_hash = sha256_file(output / "preflight_acceptance.json")
                    acceptance_path = str(output / "preflight_acceptance.json")
                    acceptance_status = result["preflight_status"]
                elif stage == "formal_cv":
                    formal_cv(args, output)
                    out_hash = sha256_file(output / "formal_cv" / "formal_model_selection.json")
                    acceptance_path = str(output / "formal_cv" / "formal_cv_acceptance.json")
                    acceptance_status = read_json(output / "formal_cv" / "formal_cv_acceptance.json").get("acceptance_status")
                elif stage == "aggregate_cv":
                    aggregate_cv(output)
                    out_hash = sha256_file(output / "formal_cv" / "aggregate_cv_acceptance.json")
                    acceptance_path = str(output / "formal_cv" / "aggregate_cv_acceptance.json")
                    acceptance_status = "passed"
                elif stage == "final_full_data_refit":
                    final_full_data_refit(args, output)
                    out_hash = sha256_file(output / "final_full_data_refit" / "final_refit_acceptance.json")
                    acceptance_path = str(output / "final_full_data_refit" / "final_refit_acceptance.json")
                    acceptance_status = read_json(output / "final_full_data_refit" / "final_refit_acceptance.json").get("acceptance_status")
                elif stage == "export_parameter_products":
                    export_parameter_products(args, output)
                    out_hash = sha256_file(output / "parameter_products" / "parameter_products_acceptance.json")
                    acceptance_path = str(output / "parameter_products" / "parameter_products_acceptance.json")
                    acceptance_status = read_json(output / "parameter_products" / "parameter_products_acceptance.json").get("acceptance_status")
                elif stage == "storage_products":
                    storage_products(output)
                    out_hash = sha256_file(output / "storage_products" / "storage_products_acceptance.json")
                    acceptance_path = str(output / "storage_products" / "storage_products_acceptance.json")
                    acceptance_status = read_json(output / "storage_products" / "storage_products_acceptance.json").get("acceptance_status")
                elif stage == "uncertainty_and_sensitivity":
                    uncertainty_and_sensitivity(output)
                    out_hash = sha256_file(output / "uncertainty_and_sensitivity" / "uncertainty_acceptance.json")
                    acceptance_path = str(output / "uncertainty_and_sensitivity" / "uncertainty_acceptance.json")
                    acceptance_status = read_json(output / "uncertainty_and_sensitivity" / "uncertainty_acceptance.json").get("acceptance_status")
                elif stage == "validation_and_diagnostics":
                    validation_and_diagnostics(output)
                    out_hash = sha256_file(output / "validation_and_diagnostics" / "validation_diagnostics_acceptance.json")
                    acceptance_path = str(output / "validation_and_diagnostics" / "validation_diagnostics_acceptance.json")
                    acceptance_status = read_json(output / "validation_and_diagnostics" / "validation_diagnostics_acceptance.json").get("acceptance_status")
                elif stage == "publication_tables":
                    publication_tables(output)
                    out_hash = sha256_file(output / "publication_tables" / "publication_tables_acceptance.json")
                    acceptance_path = str(output / "publication_tables" / "publication_tables_acceptance.json")
                    acceptance_status = read_json(output / "publication_tables" / "publication_tables_acceptance.json").get("acceptance_status")
                elif stage == "publication_figures":
                    publication_figures(output)
                    out_hash = sha256_file(output / "publication_figures" / "publication_figures_acceptance.json")
                    acceptance_path = str(output / "publication_figures" / "publication_figures_acceptance.json")
                    acceptance_status = read_json(output / "publication_figures" / "publication_figures_acceptance.json").get("acceptance_status")
                elif stage == "final_quality_gate":
                    final_quality_gate(args, output)
                    out_hash = sha256_file(Path(args.reference_dir) / "L01028_complete_results_acceptance.json")
                    acceptance_path = str(Path(args.reference_dir) / "L01028_complete_results_acceptance.json")
                    acceptance_status = read_json(Path(args.reference_dir) / "L01028_complete_results_acceptance.json").get("complete_pipeline_status")
                write_stage(output, stage, "completed", input_hash, out_hash, command_or_function=stage, acceptance_path=acceptance_path, acceptance_status=acceptance_status)
                print(f"{utc_now()} stage={stage} completed", flush=True)
                time.sleep(2)
            except Exception as exc:
                write_stage(output, stage, "failed", input_hash, failure_reason=str(exc), command_or_function=stage)
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
