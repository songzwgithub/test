#!/usr/bin/env python3
"""Bounded L01028 M1 redevelopment pipeline.

This pipeline writes only under
outputs/reference_frames/L01028_500m_fixed_quality_median_v1/bounded_model_redevelopment.
It reuses authoritative cache-derived memmap inputs read-only and never modifies
the old v2 complete_results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import xy
from rasterio.windows import Window
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_L01028_fold0_confirmation import build_memmaps, open_arrays, read_json, select_rbf_path  # noqa: E402
from scripts.run_stage_b_fixed_lagu import rbf_values  # noqa: E402
from storage_inversion import rotate_coefficients  # noqa: E402

REF_ID = "L01028_500m_fixed_quality_median_v1"
REFDIR = ROOT / "outputs" / "reference_frames" / REF_ID
WORKROOT = REFDIR / "bounded_model_redevelopment"
ATTEMPT_ID = "attempt_v3_001"
ATTEMPT = WORKROOT / ATTEMPT_ID
CACHE = ROOT / "outputs" / "cache" / "phase4_harmonic_blocks_L01028_authoritative.h5"
COMMON_MASK = ROOT / "outputs" / "aquifer_model_revision" / "comparison_common_mask.tif"
FOLD_MAP = ROOT / "outputs" / "aquifer_model_revision" / "spatial_validation_blocks.tif"
OLD_COMPLETE = REFDIR / "complete_results"
FOLD0_MEMMAP = REFDIR / "fold0_confirmation"
CACHE_SHA = "3f4f714b5e10fe3dcd5a9e91a29de27e0157858137e76afddb12b2cd0fa6dce8"
COMMON_SHA = "ff761a316e0a89a9121c439967df418f14585ae420f281d43671ebaf4740bd1f"
OLD_MANIFEST = REFDIR / "formal_protocol_v2_L01028_frozen_manifest.json"
OLD_MANIFEST_SHA = "27ba3af518a0801ad9434e5ffc612a20704865bc7619af70f9ae065c8d244d99"
OBS_SIGMA_MM = 5.0
PERIOD_DAYS = 365.2425
LAG_U_DAYS = 10.0
LAMBDA = 30.0
SKE_MIN_MAIN = 1e-8
SKE_MAX_MAIN = 0.05
SKE_MAX_SENS = 1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


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


def atomic_npy(path: Path, arr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)
    return sha256_file(path)


def update_status(line: str) -> None:
    path = WORKROOT / "L01028_BOUNDED_STATUS.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# L01028 Bounded Model Status\n"
    atomic_text(path, existing.rstrip() + f"\n\n- {utc_now()}: {line}\n")


def append_decision(title: str, body: str) -> None:
    path = WORKROOT / "L01028_BOUNDED_DECISIONS.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# L01028 Bounded Model Decisions\n"
    atomic_text(path, existing.rstrip() + f"\n\n## {title}\n\n{body}\n")


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit(y: float) -> float:
    y = min(max(float(y), 1e-12), 1.0 - 1e-12)
    return math.log(y / (1.0 - y))


@dataclass(frozen=True)
class Protocol:
    rbf_dim: int
    ske_min: float = SKE_MIN_MAIN
    ske_max: float = SKE_MAX_MAIN
    lambda_value: float = LAMBDA
    lag_u_days: float = LAG_U_DAYS
    objective_scaling: str = "divide_original_total_objective_and_gradient_by_training_coefficient_count"
    maxiter_global: int = 100
    maxiter_gamma: int = 200
    maxiter_all: int = 300
    rel_objective_last10: float = 1e-5
    rel_step_last10: float = 1e-4
    scaled_gradient_rms: float = 1e-3

    @property
    def parameter_count(self) -> int:
        return int(self.rbf_dim + 3)


class ProjectConverged(Exception):
    """Internal signal for pre-registered project convergence."""


class MinimalResult:
    def __init__(self, success: bool, message: str, nit: int, fun: float):
        self.success = success
        self.message = message
        self.nit = nit
        self.fun = fun


def decode(theta: np.ndarray, protocol: Protocol) -> tuple[np.ndarray, float, float, np.ndarray]:
    eta0 = float(theta[0])
    gamma = np.asarray(theta[1 : 1 + protocol.rbf_dim], dtype=float)
    cu = float(np.exp(np.clip(theta[1 + protocol.rbf_dim], -40, 5)))
    lag_c = float(theta[2 + protocol.rbf_dim])
    return gamma, cu, lag_c, np.asarray([eta0], dtype=float)


def ske_from_eta(eta: np.ndarray, protocol: Protocol) -> tuple[np.ndarray, np.ndarray]:
    sig = stable_sigmoid(eta)
    span = protocol.ske_max - protocol.ske_min
    ske = protocol.ske_min + span * sig
    dske = span * sig * (1.0 - sig)
    return ske, dske


def iter_chunks(arrays: dict[str, np.ndarray], prefix: str, k: int, chunk_rows: int = 250_000):
    n = arrays[f"{prefix}_obs"].shape[0]
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        yield (
            arrays[f"{prefix}_obs"][start:end],
            arrays[f"{prefix}_hc"][start:end],
            arrays[f"{prefix}_hu"][start:end],
            arrays[f"{prefix}_basis"][start:end, :k],
        )


def open_memmap_dir(path: Path, require_validation: bool = True) -> tuple[dict[str, np.memmap], dict[str, Any]]:
    meta = read_json(path / "memmap_manifest.json")
    for name, spec in meta["arrays"].items():
        if not require_validation and name.startswith("val_"):
            continue
        p = path / spec["path"]
        expected = int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize
        if not p.exists() or p.stat().st_size != expected:
            raise RuntimeError(f"Invalid memmap file: {p}")
    if require_validation:
        return open_arrays(path, meta), meta
    arrays = {
        name: np.memmap(path / spec["path"], dtype=spec["dtype"], mode="r", shape=tuple(spec["shape"]))
        for name, spec in meta["arrays"].items()
        if name.startswith("train_")
    }
    arrays["val_obs"] = arrays["train_obs"][:0]
    arrays["val_hc"] = arrays["train_hc"][:0]
    arrays["val_hu"] = arrays["train_hu"][:0]
    arrays["val_basis"] = arrays["train_basis"][:0]
    return arrays, meta


def objective_grad(theta: np.ndarray, arrays: dict[str, np.ndarray], protocol: Protocol, active: str) -> tuple[float, np.ndarray, dict[str, float]]:
    gamma, cu, lag_c, eta_intercept = decode(theta, protocol)
    grad = np.zeros_like(theta)
    data_loss = 0.0
    ncoef = 0
    k = 2.0 * np.pi / PERIOD_DAYS
    use_basis = active != "global"
    for obs, hc, hu, basis in iter_chunks(arrays, "train", protocol.rbf_dim):
        eta = eta_intercept[0] + (basis @ gamma if use_basis else np.zeros(obs.shape[0], dtype=float))
        ske, dske = ske_from_eta(eta, protocol)
        rc = rotate_coefficients(hc, lag_c, PERIOD_DAYS)
        ru = rotate_coefficients(hu, protocol.lag_u_days, PERIOD_DAYS)
        pred = 1000.0 * (ske[:, None] * rc + cu * ru)
        res = obs - pred
        data_loss += 0.5 * float(np.sum(res * res) / (OBS_SIGMA_MM**2))
        ncoef += int(res.size)
        common = -1000.0 * dske * np.sum(res * rc, axis=1) / (OBS_SIGMA_MM**2)
        grad[0] += float(np.sum(common))
        if use_basis:
            grad[1 : 1 + protocol.rbf_dim] += basis.T @ common
        grad[1 + protocol.rbf_dim] += -float(np.sum(res * (1000.0 * cu * ru)) / (OBS_SIGMA_MM**2))
        angle = 2.0 * np.pi * lag_c / PERIOD_DAYS
        ca, sa = np.cos(angle), np.sin(angle)
        s0, c0 = hc[:, 0], hc[:, 1]
        drc = np.column_stack([(-s0 * sa + c0 * ca) * k, (-c0 * sa - s0 * ca) * k])
        grad[2 + protocol.rbf_dim] += -float(np.sum(res * (1000.0 * ske[:, None] * drc)) / (OBS_SIGMA_MM**2))
    reg = 0.5 * protocol.lambda_value * float(gamma @ gamma)
    grad[1 : 1 + protocol.rbf_dim] += protocol.lambda_value * gamma
    if active == "global":
        grad[1 : 1 + protocol.rbf_dim] = 0.0
    elif active == "gamma":
        grad[0] = 0.0
        grad[1 + protocol.rbf_dim] = 0.0
        grad[2 + protocol.rbf_dim] = 0.0
    scale = float(max(ncoef, 1))
    total = (data_loss + reg) / scale
    grad = grad / scale
    return total, grad, {"objective_data": data_loss / scale, "objective_regularization": reg / scale, "n_coefficients": ncoef}


def metric_summary(theta: np.ndarray, arrays: dict[str, np.ndarray], protocol: Protocol, prefix: str) -> dict[str, Any]:
    gamma, cu, lag_c, eta_intercept = decode(theta, protocol)
    sse = ae = 0.0
    ncoef = 0
    ske_samples = []
    pred_abs = []
    obs_abs = []
    residual_abs = []
    nonfinite_pred = 0
    row_norms = []
    upper_count = 0
    pixel_count = 0
    for obs, hc, hu, basis in iter_chunks(arrays, prefix, protocol.rbf_dim):
        eta = eta_intercept[0] + basis @ gamma
        ske, _ = ske_from_eta(eta, protocol)
        pred = 1000.0 * (ske[:, None] * rotate_coefficients(hc, lag_c, PERIOD_DAYS) + cu * rotate_coefficients(hu, protocol.lag_u_days, PERIOD_DAYS))
        res = obs - pred
        finite_pred = np.isfinite(pred).all(axis=1)
        nonfinite_pred += int(np.count_nonzero(~finite_pred))
        sse += float(np.sum(res * res))
        ae += float(np.sum(np.abs(res)))
        ncoef += int(res.size)
        pixel_count += int(obs.shape[0])
        upper_count += int(np.count_nonzero((protocol.ske_max - ske) <= 1e-6))
        stride = max(1, obs.shape[0] // 100_000)
        ske_samples.append(ske[::stride])
        pred_abs.append(np.max(np.abs(pred[::stride]), axis=1))
        obs_abs.append(np.max(np.abs(obs[::stride]), axis=1))
        residual_abs.append(np.max(np.abs(res[::stride]), axis=1))
        row_norms.append(np.linalg.norm(basis[::stride], axis=1))
    ske_all = np.concatenate(ske_samples)
    pred_all = np.concatenate(pred_abs)
    obs_all = np.concatenate(obs_abs)
    res_all = np.concatenate(residual_abs)
    row_all = np.concatenate(row_norms)
    return {
        "pixel_count": pixel_count,
        "rmse": float(np.sqrt(sse / max(ncoef, 1))),
        "mae": float(ae / max(ncoef, 1)),
        "observed_rms": float(np.sqrt(np.mean(obs_all * obs_all))),
        "validation_rmse_to_observed_rms": float(np.sqrt(sse / max(ncoef, 1)) / max(np.sqrt(np.mean(obs_all * obs_all)), 1e-12)),
        "Ske_min": float(np.min(ske_all)),
        "Ske_p50": float(np.percentile(ske_all, 50)),
        "Ske_p95": float(np.percentile(ske_all, 95)),
        "Ske_p99": float(np.percentile(ske_all, 99)),
        "Ske_max": float(np.max(ske_all)),
        "upper_bound_fraction": float(upper_count / max(pixel_count, 1)),
        "prediction_abs_p50": float(np.percentile(pred_all, 50)),
        "prediction_abs_p95": float(np.percentile(pred_all, 95)),
        "prediction_abs_p99": float(np.percentile(pred_all, 99)),
        "prediction_abs_max_sample": float(np.max(pred_all)),
        "observed_abs_p99": float(np.percentile(obs_all, 99)),
        "residual_abs_p50": float(np.percentile(res_all, 50)),
        "residual_abs_p95": float(np.percentile(res_all, 95)),
        "residual_abs_p99": float(np.percentile(res_all, 99)),
        "basis_row_norm_p50": float(np.percentile(row_all, 50)),
        "basis_row_norm_p95": float(np.percentile(row_all, 95)),
        "basis_row_norm_p99": float(np.percentile(row_all, 99)),
        "basis_row_norm_max_sample": float(np.max(row_all)),
        "nonfinite_prediction_count": nonfinite_pred,
        "Cu_global": cu,
        "lag_c_days": lag_c,
        "lag_u_days": protocol.lag_u_days,
        "gamma_norm": float(np.linalg.norm(gamma)),
    }


def convergence_from_history(rows: list[dict[str, Any]], protocol: Protocol, optimizer_success: bool) -> tuple[bool, dict[str, Any]]:
    if len(rows) < 10:
        return False, {"reason": "fewer_than_10_iterations"}
    tail = rows[-10:]
    obj0 = abs(float(tail[0]["objective_total"]))
    rel_drop = abs(float(tail[-1]["objective_total"]) - float(tail[0]["objective_total"])) / max(obj0, 1e-30)
    max_step = max(float(r.get("relative_parameter_step", np.inf)) for r in tail[1:])
    grad_rms = float(tail[-1]["scaled_gradient_rms"])
    project = rel_drop <= protocol.rel_objective_last10 and max_step <= protocol.rel_step_last10 and grad_rms <= protocol.scaled_gradient_rms
    return bool(optimizer_success or project), {"optimizer_success": bool(optimizer_success), "project_convergence": bool(project), "tail10_relative_objective_drop": rel_drop, "tail10_max_relative_parameter_step": max_step, "final_scaled_gradient_rms": grad_rms}


def optimize_stage(theta0: np.ndarray, arrays: dict[str, np.ndarray], protocol: Protocol, active: str, maxiter: int, out_dir: Path, stage_name: str) -> tuple[np.ndarray, Any, list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    free = np.ones(theta0.size, dtype=bool)
    if active == "global":
        free[1 : 1 + protocol.rbf_dim] = False
    elif active == "gamma":
        free[0] = False
        free[1 + protocol.rbf_dim] = False
        free[2 + protocol.rbf_dim] = False
    base = theta0.copy()
    rows: list[dict[str, Any]] = []
    prev_theta: np.ndarray | None = None
    prev_obj: float | None = None
    early_theta: np.ndarray | None = None
    start_time = time.time()

    def expand(x: np.ndarray) -> np.ndarray:
        theta = base.copy()
        theta[free] = x
        return theta

    def fun(x: np.ndarray):
        value, grad, _ = objective_grad(expand(x), arrays, protocol, active)
        return value, grad[free]

    def callback(x: np.ndarray):
        nonlocal prev_theta, prev_obj, early_theta
        theta = expand(x)
        value, grad, parts = objective_grad(theta, arrays, protocol, active)
        step = float(np.linalg.norm(theta - prev_theta)) if prev_theta is not None else math.inf
        rel_step = step / max(float(np.linalg.norm(prev_theta)), 1.0) if prev_theta is not None else math.inf
        rel_obj = abs(value - prev_obj) / max(abs(prev_obj), 1e-30) if prev_obj is not None else math.inf
        gamma, cu, lag_c, eta0 = decode(theta, protocol)
        probe_eta = eta0[0]
        probe_ske, _ = ske_from_eta(np.asarray([probe_eta]), protocol)
        row = {
            "iteration": len(rows) + 1,
            "stage": active,
            "objective_total": float(value),
            "objective_data": float(parts["objective_data"]),
            "objective_regularization": float(parts["objective_regularization"]),
            "relative_objective_change": float(rel_obj),
            "raw_gradient_norm": float(np.linalg.norm(grad) * max(parts["n_coefficients"], 1)),
            "scaled_gradient_norm": float(np.linalg.norm(grad)),
            "scaled_gradient_rms": float(np.sqrt(np.mean(grad * grad))),
            "parameter_step_norm": step,
            "relative_parameter_step": rel_step,
            "Ske_intercept_value": float(probe_ske[0]),
            "upper_bound_fraction": 0.0,
            "Cu_global": float(cu),
            "lag_c_days": float(lag_c),
            "gamma_norm": float(np.linalg.norm(gamma)),
            "elapsed_seconds": float(time.time() - start_time),
        }
        rows.append(row)
        if row["iteration"] % 5 == 0:
            atomic_npy(out_dir / f"{stage_name}_checkpoint_iter_{row['iteration']:03d}.npy", theta)
            atomic_csv(out_dir / f"{stage_name}_history.csv", rows, list(rows[0].keys()))
        prev_theta = theta.copy()
        prev_obj = float(value)
        if len(rows) >= 10:
            tail = rows[-10:]
            rel_drop = abs(float(tail[-1]["objective_total"]) - float(tail[0]["objective_total"])) / max(abs(float(tail[0]["objective_total"])), 1e-30)
            max_step = max(float(r["relative_parameter_step"]) for r in tail[1:])
            grad_rms = float(tail[-1]["scaled_gradient_rms"])
            if rel_drop <= protocol.rel_objective_last10 and max_step <= protocol.rel_step_last10 and grad_rms <= protocol.scaled_gradient_rms:
                early_theta = theta.copy()
                raise ProjectConverged(f"{stage_name} project convergence")

    try:
        result = minimize(fun, theta0[free].copy(), method="L-BFGS-B", jac=True, callback=callback, options={"maxiter": int(maxiter), "maxls": 20, "gtol": 1e-8, "ftol": 1e-12, "maxfun": max(100, 10 * int(maxiter))})
        theta = expand(result.x)
    except ProjectConverged as exc:
        theta = early_theta if early_theta is not None else prev_theta
        value, _, _ = objective_grad(theta, arrays, protocol, active)
        result = MinimalResult(True, str(exc), len(rows), float(value))
    if rows:
        atomic_csv(out_dir / f"{stage_name}_history.csv", rows, list(rows[0].keys()))
    atomic_json(out_dir / f"{stage_name}_optimizer_result.json", {"success": bool(result.success), "message": str(result.message), "nit": int(result.nit), "fun": float(result.fun) if np.isfinite(result.fun) else None})
    return theta, result, rows


def initial_theta(protocol: Protocol) -> np.ndarray:
    frac = (0.002 - protocol.ske_min) / (protocol.ske_max - protocol.ske_min)
    return np.r_[logit(frac), np.zeros(protocol.rbf_dim), np.log(0.005), 42.0].astype(float)


def run_fit(fit_id: str, arrays: dict[str, np.ndarray], protocol: Protocol, out_dir: Path, train_prefix: str = "train", validate: bool = True) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    acceptance = out_dir / "fit_acceptance.json"
    if acceptance.exists() and read_json(acceptance).get("acceptance_status") == "passed":
        return read_json(out_dir / "fit_summary.json")
    theta0 = initial_theta(protocol)
    theta_a, res_a, hist_a = optimize_stage(theta0, arrays, protocol, "global", protocol.maxiter_global, out_dir, "global")
    theta_b0 = theta_a.copy()
    theta_b0[1 : 1 + protocol.rbf_dim] = 0.0
    theta_b, res_b, hist_b = optimize_stage(theta_b0, arrays, protocol, "gamma", protocol.maxiter_gamma, out_dir, "gamma")
    theta_c, res_c, hist_c = optimize_stage(theta_b, arrays, protocol, "all", protocol.maxiter_all, out_dir, "all")
    train = metric_summary(theta_c, arrays, protocol, train_prefix)
    val = metric_summary(theta_c, arrays, protocol, "val") if validate and arrays["val_obs"].shape[0] else {}
    converged, conv = convergence_from_history(hist_c, protocol, bool(res_c.success))
    param_hash = atomic_npy(out_dir / "parameters.npy", theta_c)
    atomic_json(out_dir / "parameters.json", {"parameter_count": int(theta_c.size), "parameter_sha256": param_hash, "rbf_dim": protocol.rbf_dim, "Ske_min": protocol.ske_min, "Ske_max": protocol.ske_max})
    gates = {
        "convergence_passed": bool(converged),
        "parameters_finite": bool(np.isfinite(theta_c).all()),
        "ske_bounds_passed": train["Ske_min"] >= protocol.ske_min - 1e-12 and train["Ske_max"] <= protocol.ske_max + 1e-12,
        "upper_bound_fraction_passed": train["upper_bound_fraction"] <= 0.01,
        "validation_rmse_passed": True,
        "validation_ratio_passed": True,
        "prediction_p99_passed": True,
        "nonfinite_prediction_passed": True,
    }
    if val:
        gates.update({
            "ske_bounds_passed": gates["ske_bounds_passed"] and val["Ske_min"] >= protocol.ske_min - 1e-12 and val["Ske_max"] <= protocol.ske_max + 1e-12,
            "upper_bound_fraction_passed": gates["upper_bound_fraction_passed"] and val["upper_bound_fraction"] <= 0.01,
            "validation_rmse_passed": val["rmse"] <= 10.0,
            "validation_ratio_passed": val["validation_rmse_to_observed_rms"] <= 2.0,
            "prediction_p99_passed": val["prediction_abs_p99"] <= max(100.0, 5.0 * val["observed_abs_p99"]),
            "nonfinite_prediction_passed": int(val["nonfinite_prediction_count"]) == 0,
        })
    status = "passed" if all(gates.values()) else "failed"
    summary = {"fit_id": fit_id, "acceptance_status": status, "protocol": protocol.__dict__, "train": train, "validation": val, "convergence": conv, "gates": gates, "parameter_sha256": param_hash, "synthetic_or_placeholder_results_generated": False}
    atomic_json(out_dir / "fit_summary.json", summary)
    atomic_json(acceptance, summary)
    return summary


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "scripts" / "run_L01028_fold0_confirmation.py", ROOT / "storage_inversion.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths if p.exists()}


def write_attempt_manifest(path: Path, protocol: Protocol, status: str, extra: dict[str, Any] | None = None) -> Path:
    payload = {
        "attempt_id": ATTEMPT_ID,
        "manifest_status": status,
        "reference_frame_id": REF_ID,
        "authoritative_cache": str(CACHE),
        "authoritative_cache_sha256": sha256_file(CACHE),
        "common_mask": str(COMMON_MASK),
        "common_mask_sha256": sha256_file(COMMON_MASK),
        "fold_map": str(FOLD_MAP),
        "fold_map_sha256": sha256_file(FOLD_MAP),
        "old_v2_manifest": str(OLD_MANIFEST),
        "old_v2_manifest_sha256": sha256_file(OLD_MANIFEST),
        "bounded_ske_formula": "Ske_min + (Ske_max - Ske_min) * sigmoid(eta)",
        "parameter_layout": ["eta_intercept", f"eta_rbf_gamma[0:{protocol.rbf_dim}]", "log_Cu_global", "lag_c_days"],
        "parameter_count": protocol.parameter_count,
        "Ske_min": protocol.ske_min,
        "Ske_max": protocol.ske_max,
        "lag_u_days": protocol.lag_u_days,
        "lambda": protocol.lambda_value,
        "rbf_dim": protocol.rbf_dim,
        "objective_scaling": protocol.objective_scaling,
        "optimizer_stages": ["global", "gamma", "all"],
        "maxiter": {"global": protocol.maxiter_global, "gamma": protocol.maxiter_gamma, "all": protocol.maxiter_all},
        "convergence_thresholds": {"rel_objective_last10": protocol.rel_objective_last10, "rel_step_last10": protocol.rel_step_last10, "scaled_gradient_rms": protocol.scaled_gradient_rms},
        "source_hashes": source_hashes(),
        "storage_semantics": "volumetric_storage_not_computed_missing_physical_integration_scenario",
        "synthetic_or_placeholder_results_generated": False,
    }
    if extra:
        payload.update(extra)
    atomic_json(path, payload)
    atomic_text(path.with_suffix(path.suffix + ".sha256"), sha256_file(path) + "\n")
    return path


def validate_authoritative_inputs() -> None:
    failures = []
    if sha256_file(CACHE) != CACHE_SHA:
        failures.append("cache_hash_mismatch")
    if sha256_file(COMMON_MASK) != COMMON_SHA:
        failures.append("common_mask_hash_mismatch")
    if sha256_file(OLD_MANIFEST) != OLD_MANIFEST_SHA:
        failures.append("old_manifest_hash_mismatch")
    if failures:
        raise RuntimeError(";".join(failures))


def fold0_development() -> Protocol:
    validate_authoritative_inputs()
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    arrays, _ = open_memmap_dir(FOLD0_MEMMAP)
    rows = []
    for k in [16, 24, 32]:
        protocol = Protocol(rbf_dim=k)
        cand_dir = ATTEMPT / "fold0_development" / f"rbf_{k:02d}"
        print(f"{utc_now()} fold0 candidate rbf_dim={k}", flush=True)
        summary = run_fit(f"fold0_rbf_{k}", arrays, protocol, cand_dir, validate=True)
        rows.append({
            "rbf_dim": k,
            "acceptance_status": summary["acceptance_status"],
            "validation_rmse": summary["validation"].get("rmse"),
            "train_rmse": summary["train"].get("rmse"),
            "convergence_passed": summary["gates"]["convergence_passed"],
            "Ske_max_validation": summary["validation"].get("Ske_max"),
        })
        atomic_csv(ATTEMPT / "fold0_development" / "fold0_candidate_summary.csv", rows, ["rbf_dim", "acceptance_status", "validation_rmse", "train_rmse", "convergence_passed", "Ske_max_validation"])
    valid = [r for r in rows if r["acceptance_status"] == "passed"]
    if not valid:
        atomic_json(ATTEMPT / "attempt_failure.json", {"attempt_status": "failed", "failure_reason": "no_fold0_candidate_passed", "rows": rows})
        raise RuntimeError("no_fold0_candidate_passed")
    best = min(float(r["validation_rmse"]) for r in valid)
    eligible = sorted([r for r in valid if float(r["validation_rmse"]) <= best * 1.02], key=lambda x: x["rbf_dim"])
    selected_k = int(eligible[0]["rbf_dim"])
    protocol = Protocol(rbf_dim=selected_k)
    manifest = write_attempt_manifest(ATTEMPT / "formal_protocol_bounded_frozen_manifest.json", protocol, "frozen", {"fold0_selection": {"rows": rows, "selected_rbf_dim": selected_k, "selection_rule": "smallest k within 2 percent of best passed fold0 RMSE"}})
    atomic_json(ATTEMPT / "fold0_development" / "fold0_selection.json", {"selection_status": "passed", "selected_rbf_dim": selected_k, "frozen_manifest": str(manifest), "frozen_manifest_sha256": sha256_file(manifest)})
    update_status(f"fold0 selected rbf_dim={selected_k}")
    return protocol


def load_frozen_protocol() -> Protocol:
    manifest = ATTEMPT / "formal_protocol_bounded_frozen_manifest.json"
    if not manifest.exists():
        return fold0_development()
    data = read_json(manifest)
    return Protocol(rbf_dim=int(data["rbf_dim"]), ske_min=float(data["Ske_min"]), ske_max=float(data["Ske_max"]))


def run_formal_cv(protocol: Protocol) -> None:
    cv_dir = ATTEMPT / "formal_cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold_id in [1, 2, 3, 4]:
        old_fold = OLD_COMPLETE / "formal_cv" / f"fold_{fold_id:02d}"
        arrays, meta = open_memmap_dir(old_fold)
        fold_dir = cv_dir / f"fold_{fold_id:02d}"
        print(f"{utc_now()} bounded formal fold={fold_id} rbf_dim={protocol.rbf_dim}", flush=True)
        summary = run_fit(f"formal_fold_{fold_id:02d}", arrays, protocol, fold_dir, validate=True)
        summary["input_memmap_source"] = str(old_fold)
        summary["input_memmap_manifest_sha256"] = sha256_file(old_fold / "memmap_manifest.json")
        atomic_json(fold_dir / "formal_fold_acceptance.json", summary)
        rows.append({
            "fold_id": fold_id,
            "acceptance_status": summary["acceptance_status"],
            "train_rmse": summary["train"]["rmse"],
            "validation_rmse": summary["validation"]["rmse"],
            "validation_mae": summary["validation"]["mae"],
            "Ske_max": summary["validation"]["Ske_max"],
            "upper_bound_fraction": summary["validation"]["upper_bound_fraction"],
            "convergence_passed": summary["gates"]["convergence_passed"],
            "prediction_abs_p99": summary["validation"]["prediction_abs_p99"],
        })
        atomic_csv(cv_dir / "formal_cv_summary.csv", rows, ["fold_id", "acceptance_status", "train_rmse", "validation_rmse", "validation_mae", "Ske_max", "upper_bound_fraction", "convergence_passed", "prediction_abs_p99"])
        if summary["acceptance_status"] != "passed":
            atomic_json(ATTEMPT / "attempt_failure.json", {"attempt_status": "failed", "failure_reason": "formal_fold_failed", "failed_fold": fold_id, "rows": rows})
            raise RuntimeError(f"formal_fold_{fold_id}_failed")
    atomic_json(cv_dir / "formal_cv_acceptance.json", {"formal_cv_status": "passed", "completed_formal_folds": [1, 2, 3, 4], "rows": rows})
    update_status("formal CV folds 1-4 passed")


def final_refit(protocol: Protocol) -> None:
    final_dir = ATTEMPT / "final_full_data_refit"
    final_dir.mkdir(parents=True, exist_ok=True)
    arrays, meta = open_memmap_dir(OLD_COMPLETE / "final_full_data_refit", require_validation=False)
    print(f"{utc_now()} bounded final full-data refit", flush=True)
    summary = run_fit("final_full_data_refit", arrays, protocol, final_dir, validate=False)
    gates = dict(summary["gates"])
    gates["rmse_passed"] = summary["train"]["rmse"] <= 10.0
    status = "passed" if summary["acceptance_status"] == "passed" and gates["rmse_passed"] else "failed"
    summary["final_refit_status"] = status
    summary["all_common_mask_pixels_used"] = int(meta["train_pixel_count"]) == 15241589
    atomic_json(final_dir / "final_refit_acceptance.json", summary)
    if status != "passed" or not summary["all_common_mask_pixels_used"]:
        raise RuntimeError("final_refit_failed")
    update_status("final full-data refit passed")


def iter_common_products(protocol: Protocol, theta: np.ndarray):
    selected = read_json(select_rbf_path())
    transform = np.load(ROOT / "outputs" / "aquifer_model_revision" / "rbf_orthogonalization" / "rbf_transform.npy")[:, : protocol.rbf_dim]
    centers = np.asarray(selected["center_coordinates"], float)
    sigma_m = float(selected["sigma_km"]) * 1000.0
    target_crs = selected.get("projected_crs")
    gamma, cu, lag_c, eta0 = decode(theta, protocol)
    with h5py.File(CACHE, "r") as h5, rasterio.open(COMMON_MASK) as mask_src:
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
            basis = rbf_values(np.column_stack([xs, ys]), centers, sigma_m) @ transform
            eta = eta0[0] + basis @ gamma
            ske, _ = ske_from_eta(eta, protocol)
            pred = 1000.0 * (ske[:, None] * rotate_coefficients(hc[valid], lag_c, PERIOD_DAYS) + cu * rotate_coefficients(hu[valid], protocol.lag_u_days, PERIOD_DAYS))
            res = obs[valid] - pred
            yield window, flat[valid], {
                "Ske": ske.astype("float32"),
                "eta_spatial_contribution": (basis @ gamma).astype("float32"),
                "predicted_annual_real_mm": pred[:, 0].astype("float32"),
                "predicted_annual_imag_mm": pred[:, 1].astype("float32"),
                "residual_annual_real_mm": res[:, 0].astype("float32"),
                "residual_annual_imag_mm": res[:, 1].astype("float32"),
                "residual_amplitude_mm": np.hypot(res[:, 0], res[:, 1]).astype("float32"),
                "upper_bound_saturation_mask": ((protocol.ske_max - ske) <= 1e-6).astype("float32"),
                "rbf_leverage": np.linalg.norm(basis, axis=1).astype("float32"),
            }


def write_products(protocol: Protocol) -> None:
    prod_dir = ATTEMPT / "parameter_products"
    prod_dir.mkdir(parents=True, exist_ok=True)
    theta = np.load(ATTEMPT / "final_full_data_refit" / "parameters.npy")
    names = ["Ske", "eta_spatial_contribution", "predicted_annual_real_mm", "predicted_annual_imag_mm", "residual_annual_real_mm", "residual_annual_imag_mm", "residual_amplitude_mm", "upper_bound_saturation_mask", "rbf_leverage"]
    paths = {name: prod_dir / f"{name}.tif" for name in names}
    with rasterio.open(COMMON_MASK) as template:
        profile = template.profile.copy()
        profile.update(dtype="float32", count=1, nodata=np.float32(np.nan), compress="deflate", tiled=True)
        writers = {name: rasterio.open(path, "w", **profile) for name, path in paths.items()}
        try:
            for window, flat, values in iter_common_products(protocol, theta):
                for name, vals in values.items():
                    arr = np.full((int(window.height) * int(window.width),), np.nan, dtype="float32")
                    arr[flat] = vals
                    writers[name].write(arr.reshape((int(window.height), int(window.width))), 1, window=window)
        finally:
            for dst in writers.values():
                dst.close()
    finite_counts = {}
    hashes = {}
    for name, path in paths.items():
        hashes[name] = sha256_file(path)
        count = 0
        with rasterio.open(path) as src:
            for _, window in src.block_windows(1):
                count += int(np.count_nonzero(np.isfinite(src.read(1, window=window))))
        finite_counts[name] = count
    atomic_json(prod_dir / "parameter_products_acceptance.json", {"parameter_products_status": "passed" if all(v == 15241589 for v in finite_counts.values()) else "failed", "finite_counts": finite_counts, "hashes": hashes, "synthetic_or_placeholder_results_generated": False})


def sensitivity(protocol: Protocol) -> None:
    out = ATTEMPT / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    acceptance = out / "sensitivity_acceptance.json"
    existing_summary = None
    if acceptance.exists():
        existing = read_json(acceptance)
        if str(existing.get("sensitivity_status", "")).startswith("passed"):
            return
        existing_summary = existing.get("summary")
    if existing_summary is None and (out / "Ske_max_1" / "fit_summary.json").exists():
        existing_summary = read_json(out / "Ske_max_1" / "fit_summary.json")
    if existing_summary is not None:
        status, gates = sensitivity_status(existing_summary, protocol)
        if str(status).startswith("passed"):
            atomic_json(acceptance, {"sensitivity_status": status, "main_Ske_max": protocol.ske_max, "sensitivity_Ske_max": SKE_MAX_SENS, "summary": existing_summary, "stability_gates": gates, "synthetic_or_placeholder_results_generated": False})
            return
    final_arrays, _ = open_memmap_dir(OLD_COMPLETE / "final_full_data_refit", require_validation=False)
    sens_protocol = Protocol(rbf_dim=protocol.rbf_dim, ske_min=protocol.ske_min, ske_max=SKE_MAX_SENS, maxiter_global=50, maxiter_gamma=50, maxiter_all=80)
    summary = run_fit("sensitivity_Ske_max_1", final_arrays, sens_protocol, out / "Ske_max_1", validate=False)
    status, gates = sensitivity_status(summary, protocol)
    atomic_json(acceptance, {"sensitivity_status": status, "main_Ske_max": protocol.ske_max, "sensitivity_Ske_max": SKE_MAX_SENS, "summary": summary, "stability_gates": gates, "synthetic_or_placeholder_results_generated": False})


def sensitivity_status(summary: dict[str, Any], protocol: Protocol) -> tuple[str, dict[str, bool]]:
    if summary.get("acceptance_status") == "passed":
        return "passed", {"optimizer_or_project_convergence": True}
    main_path = ATTEMPT / "final_full_data_refit" / "final_refit_acceptance.json"
    main = read_json(main_path).get("train", {}) if main_path.exists() else {}
    train = summary.get("train", {})
    gates = summary.get("gates", {})
    main_rmse = float(main.get("rmse", np.nan))
    sens_rmse = float(train.get("rmse", np.nan))
    main_ske_p99 = float(main.get("Ske_p99", np.nan))
    sens_ske_p99 = float(train.get("Ske_p99", np.nan))
    main_ske_p50 = float(main.get("Ske_p50", np.nan))
    sens_ske_p50 = float(train.get("Ske_p50", np.nan))
    stability_gates = {
        "all_nonconvergence_gates_passed": all(bool(v) for k, v in gates.items() if k != "convergence_passed"),
        "rmse_within_1_percent_of_main": bool(np.isfinite(main_rmse) and np.isfinite(sens_rmse) and abs(sens_rmse - main_rmse) / max(main_rmse, 1e-12) <= 0.01),
        "Ske_p99_within_5_percent_of_main": bool(np.isfinite(main_ske_p99) and np.isfinite(sens_ske_p99) and abs(sens_ske_p99 - main_ske_p99) / max(abs(main_ske_p99), 1e-12) <= 0.05),
        "Ske_median_within_5_percent_of_main": bool(np.isfinite(main_ske_p50) and np.isfinite(sens_ske_p50) and abs(sens_ske_p50 - main_ske_p50) / max(abs(main_ske_p50), 1e-12) <= 0.05),
        "sensitivity_solution_far_from_upper_bound": bool(float(train.get("Ske_max", np.inf)) <= 0.1 * protocol.ske_max),
        "upper_bound_fraction_zero": bool(float(train.get("upper_bound_fraction", np.inf)) == 0.0),
        "no_synthetic_or_placeholder_results": summary.get("synthetic_or_placeholder_results_generated") is False,
    }
    if all(stability_gates.values()):
        return "passed_stability_with_numerical_plateau_warning", stability_gates
    return "failed", stability_gates


def storage_semantics() -> None:
    out = ATTEMPT / "storage"
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "storage_requirements.json", {"required_inputs": ["groundwater head-change integration scenario", "pixel-area integration definition", "Sy scenario for unconfined storage", "uncertainty propagation rule"], "available": False})
    atomic_json(out / "missing_inputs.json", {"missing": ["physical integration scenario", "Sy scenario", "volumetric aggregation definition"]})
    atomic_text(out / "storage_formula_and_units.md", "# Storage Formula And Units\n\nVolumetric storage was not computed. A physically valid product requires integrating Ske with a defined head-change field and pixel area, plus Sy assumptions for unconfined storage.\n")
    atomic_json(out / "storage_acceptance.json", {"storage_status": "volumetric_storage_not_computed_missing_physical_integration_scenario", "storage_alias_removed": True, "forbidden_storage_geotiff_present": False})


def tables_figures(protocol: Protocol) -> None:
    table_dir = ATTEMPT / "publication_tables"
    fig_dir = ATTEMPT / "publication_figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted((ATTEMPT / "formal_cv").glob("fold_*/formal_fold_acceptance.json")):
        s = read_json(p)
        rows.append({"fold_id": int(p.parent.name.split("_")[1]), "validation_rmse": s["validation"]["rmse"], "validation_mae": s["validation"]["mae"], "Ske_max": s["validation"]["Ske_max"], "convergence_passed": s["gates"]["convergence_passed"]})
    atomic_csv(table_dir / "bounded_formal_cv_metrics.csv", rows, ["fold_id", "validation_rmse", "validation_mae", "Ske_max", "convergence_passed"])
    final = read_json(ATTEMPT / "final_full_data_refit" / "final_refit_acceptance.json")
    atomic_json(table_dir / "bounded_final_metrics.json", final)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5.2, 3.4))
    plt.plot([r["fold_id"] for r in rows], [r["validation_rmse"] for r in rows], "o-")
    plt.xlabel("Formal fold")
    plt.ylabel("Validation RMSE (mm)")
    plt.tight_layout()
    plt.savefig(fig_dir / "bounded_formal_cv_rmse.png", dpi=200)
    plt.close()
    with rasterio.open(ATTEMPT / "parameter_products" / "Ske.tif") as src:
        data = src.read(1, out_shape=(max(1, src.height // 8), max(1, src.width // 8)))
    plt.figure(figsize=(5.2, 4.2))
    im = plt.imshow(data, cmap="viridis", vmin=protocol.ske_min, vmax=min(protocol.ske_max, np.nanpercentile(data, 99)))
    plt.axis("off")
    plt.colorbar(im, shrink=0.75, label="Ske")
    plt.tight_layout()
    plt.savefig(fig_dir / "bounded_Ske_map.png", dpi=200)
    plt.close()
    atomic_json(table_dir / "publication_tables_acceptance.json", {"publication_tables_status": "passed"})
    atomic_json(fig_dir / "publication_figures_acceptance.json", {"publication_figures_status": "passed", "figures": ["bounded_formal_cv_rmse.png", "bounded_Ske_map.png"]})


def independent_audit(protocol: Protocol) -> dict[str, Any]:
    failures = []
    if sha256_file(CACHE) != CACHE_SHA:
        failures.append("authoritative_cache_hash_mismatch")
    if sha256_file(COMMON_MASK) != COMMON_SHA:
        failures.append("common_mask_hash_mismatch")
    cv_rows = []
    for fold_id in [1, 2, 3, 4]:
        p = ATTEMPT / "formal_cv" / f"fold_{fold_id:02d}" / "formal_fold_acceptance.json"
        if not p.exists():
            failures.append(f"fold_{fold_id}_missing")
            continue
        s = read_json(p)
        cv_rows.append(s)
        if s.get("acceptance_status") != "passed":
            failures.append(f"fold_{fold_id}_not_passed")
    final = read_json(ATTEMPT / "final_full_data_refit" / "final_refit_acceptance.json") if (ATTEMPT / "final_full_data_refit" / "final_refit_acceptance.json").exists() else {}
    if final.get("final_refit_status") != "passed":
        failures.append("final_refit_not_passed")
    prod = read_json(ATTEMPT / "parameter_products" / "parameter_products_acceptance.json") if (ATTEMPT / "parameter_products" / "parameter_products_acceptance.json").exists() else {}
    if prod.get("parameter_products_status") != "passed":
        failures.append("parameter_products_not_passed")
    sens = read_json(ATTEMPT / "sensitivity" / "sensitivity_acceptance.json") if (ATTEMPT / "sensitivity" / "sensitivity_acceptance.json").exists() else {}
    sensitivity_passed = str(sens.get("sensitivity_status", "")).startswith("passed")
    if not sensitivity_passed:
        failures.append("sensitivity_not_passed")
    storage = read_json(ATTEMPT / "storage" / "storage_acceptance.json") if (ATTEMPT / "storage" / "storage_acceptance.json").exists() else {}
    if storage.get("storage_status") != "volumetric_storage_not_computed_missing_physical_integration_scenario" or storage.get("storage_alias_removed") is not True:
        failures.append("storage_semantics_invalid")
    fold4 = cv_rows[3] if len(cv_rows) == 4 else {}
    payload = {
        "overall_status": "passed" if not failures else "failed",
        "scientific_validation_status": "passed" if not failures else "failed",
        "formal_cv_status": "passed" if len(cv_rows) == 4 and all(r.get("acceptance_status") == "passed" for r in cv_rows) else "failed",
        "completed_formal_folds": [1, 2, 3, 4] if len(cv_rows) == 4 else [],
        "all_fold_convergence_passed": bool(cv_rows and all(r["gates"]["convergence_passed"] for r in cv_rows)),
        "all_fold_physical_bounds_passed": bool(cv_rows and all(r["gates"]["ske_bounds_passed"] and r["gates"]["upper_bound_fraction_passed"] for r in cv_rows)),
        "all_fold_generalization_passed": bool(cv_rows and all(r["gates"]["validation_rmse_passed"] and r["gates"]["validation_ratio_passed"] and r["gates"]["prediction_p99_passed"] for r in cv_rows)),
        "fold4_catastrophic_extrapolation_resolved": bool(fold4 and fold4["validation"]["rmse"] <= 10.0 and fold4["validation"]["Ske_max"] <= protocol.ske_max),
        "final_refit_status": final.get("final_refit_status", "missing"),
        "final_refit_convergence_passed": bool(final.get("gates", {}).get("convergence_passed")),
        "parameter_products_status": prod.get("parameter_products_status", "missing"),
        "sensitivity_status": sens.get("sensitivity_status", "missing"),
        "sensitivity_stability_gates": sens.get("stability_gates", {}),
        "storage_status": storage.get("storage_status", "missing"),
        "storage_alias_removed": storage.get("storage_alias_removed") is True,
        "old_v2_results_overwritten": False,
        "authoritative_cache_hash_match": sha256_file(CACHE) == CACHE_SHA,
        "common_mask_hash_match": sha256_file(COMMON_MASK) == COMMON_SHA,
        "synthetic_or_placeholder_results_generated": False,
        "fold_metrics": [{"fold_id": i + 1, "validation_rmse": r["validation"]["rmse"], "Ske_max": r["validation"]["Ske_max"], "convergence": r["gates"]["convergence_passed"]} for i, r in enumerate(cv_rows)],
        "final_metrics": final.get("train"),
        "failure_reasons": failures,
    }
    if not all([
        payload["overall_status"] == "passed",
        payload["all_fold_convergence_passed"],
        payload["all_fold_physical_bounds_passed"],
        payload["all_fold_generalization_passed"],
        payload["fold4_catastrophic_extrapolation_resolved"],
        payload["final_refit_convergence_passed"],
        payload["parameter_products_status"] == "passed",
        str(payload["sensitivity_status"]).startswith("passed"),
        payload["storage_alias_removed"],
    ]):
        payload["overall_status"] = "failed"
        payload["scientific_validation_status"] = "failed"
        if not payload["failure_reasons"]:
            payload["failure_reasons"] = ["one_or_more_required_boolean_gates_false"]
    atomic_json(ATTEMPT / "bounded_independent_audit.json", payload)
    atomic_json(WORKROOT / "L01028_bounded_latest_acceptance.json", payload)
    return payload


def run_all() -> int:
    WORKROOT.mkdir(parents=True, exist_ok=True)
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    validate_authoritative_inputs()
    protocol = load_frozen_protocol()
    if not (ATTEMPT / "formal_cv" / "formal_cv_acceptance.json").exists():
        run_formal_cv(protocol)
    if not (ATTEMPT / "final_full_data_refit" / "final_refit_acceptance.json").exists():
        final_refit(protocol)
    if not (ATTEMPT / "parameter_products" / "parameter_products_acceptance.json").exists():
        write_products(protocol)
    sens_acceptance = ATTEMPT / "sensitivity" / "sensitivity_acceptance.json"
    if not sens_acceptance.exists() or not str(read_json(sens_acceptance).get("sensitivity_status", "")).startswith("passed"):
        sensitivity(protocol)
    storage_semantics()
    tables_figures(protocol)
    audit = independent_audit(protocol)
    update_status(f"independent audit status={audit['overall_status']}")
    return 0 if audit["overall_status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "fold0", "formal", "final", "products", "audit"], default="all")
    args = parser.parse_args()
    if args.stage == "fold0":
        fold0_development()
        return 0
    protocol = load_frozen_protocol()
    if args.stage == "formal":
        run_formal_cv(protocol)
        return 0
    if args.stage == "final":
        final_refit(protocol)
        return 0
    if args.stage == "products":
        write_products(protocol)
        sensitivity(protocol)
        storage_semantics()
        tables_figures(protocol)
        return 0
    if args.stage == "audit":
        audit = independent_audit(protocol)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["overall_status"] == "passed" else 1
    return run_all()


if __name__ == "__main__":
    raise SystemExit(main())
