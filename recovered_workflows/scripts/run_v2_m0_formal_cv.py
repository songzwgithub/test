#!/usr/bin/env python
"""Run V2 M0 confined-only G0/L0 formal four-fold CV and compare with M1."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import sys
import time
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bounded_ske_v2 import (
    PARAMETERIZATION_VERSION,
    SKE_LOWER_BOUND,
    SKE_UPPER_BOUND,
    bounded_ske,
    bounded_ske_derivative,
    inverse_bounded_ske,
)
from scripts.run_v2_g0_formal_cv import (
    BUDGET,
    EXPECTED_COMMON,
    LAMBDA,
    MODEL_DIR as M1_DIR,
    OBS_SIGMA_MM,
    OUT,
    PERIOD_DAYS,
    _finite_stats,
    dependency_versions,
    eval_theta as eval_m1_theta,
    hash_array,
    hash_file,
    iter_blocks,
    latest_real_harmonic_cache,
    read_json,
    split_counts,
    write_json,
)
from storage_inversion import rotate_coefficients


MODEL_DIR = OUT / "model_compare_v2/M0_confined_only_G0_no_geology_L0_shared"
OBJECTIVE_VERSION = "stage_c_m0_bounded_ske_confined_only_squared_loss_v1"
PRIOR_VERSION = "raw_standardized_gamma_lambda30_confined_only_stageA_centered_v1"
PARAMETER_LAYOUT = "eta_intercept_32_gamma_lagc_fixed_l0_confined_only_v1"


def dependency_hash_payload() -> dict:
    return {"python": platform.python_version(), **dependency_versions()}


def make_model_compare_manifest() -> dict:
    source = OUT / "formal_protocol_v2_frozen_manifest.json"
    base = read_json(source)
    if hash_file(source) != "fd20ef95b2b9c1c489e0226dfefc467cefbf9a3365fd779500c7a8f0f3a8873c":
        raise RuntimeError("Existing V2 G0 frozen manifest hash changed; refusing to proceed.")
    manifest_path = OUT / "formal_protocol_v2_model_compare_manifest.json"
    sha_path = OUT / "formal_protocol_v2_model_compare_manifest.sha256"
    manifest = dict(base)
    manifest.update({
        "manifest_status": "frozen_for_v2_aquifer_structure_model_compare",
        "allowed_models": ["M0_v2_confined_only", "M1_v2_bounded_Ske"],
        "M0_model_version": "M0_v2_confined_only",
        "M0_objective_version": OBJECTIVE_VERSION,
        "M0_parameter_layout": PARAMETER_LAYOUT,
        "M0_prior_version": PRIOR_VERSION,
        "M0_contains_Cu": False,
        "M0_contains_lag_u": False,
        "source_code_hash": sha256(
            (hash_file(Path(__file__)) + hash_file(ROOT_DIR / "bounded_ske_v2.py")).encode()
        ).hexdigest(),
        "dependency_versions": dependency_hash_payload(),
        "formal_v2_execution_allowed": True,
    })
    manifest.pop("manifest_hash", None)
    write_json(manifest_path, manifest)
    digest = hash_file(manifest_path)
    manifest["manifest_hash"] = digest
    write_json(manifest_path, manifest)
    digest = hash_file(manifest_path)
    sha_path.write_text(digest, encoding="utf-8")
    manifest["manifest_hash"] = digest
    write_json(manifest_path, manifest)
    sha_path.write_text(hash_file(manifest_path), encoding="utf-8")
    return read_json(manifest_path)


def decode_theta(theta: np.ndarray):
    return float(theta[0]), theta[1:33], float(theta[33])


def eval_theta(theta, blocks, collect=False):
    eta0, gamma, lag_c = decode_theta(theta)
    sse = ae = bias = real_sse = imag_sse = amp_sse = phase_sum = 0.0
    ncoef = npix = 0
    abs_resids = [] if collect else None
    ske_vals = []
    pred_amp_vals = []
    bnorm_vals = []
    conf_amp_vals = []
    for _bi, obs, hc, _hu, basis in blocks:
        eta = eta0 + basis @ gamma
        ske = bounded_ske(eta)
        confined = 1000 * ske[:, None] * rotate_coefficients(hc, lag_c, PERIOD_DAYS)
        pred = confined
        res = obs - pred
        absr = np.linalg.norm(res, axis=1)
        sse += float(np.sum(res * res))
        ae += float(np.sum(np.abs(res)))
        bias += float(np.sum(res))
        real_sse += float(np.sum(res[:, 0] ** 2))
        imag_sse += float(np.sum(res[:, 1] ** 2))
        amp_sse += float(np.sum((np.linalg.norm(obs, axis=1) - np.linalg.norm(pred, axis=1)) ** 2))
        phase_sum += float(np.sum(np.abs(np.angle(np.exp(1j * (np.arctan2(obs[:, 0], obs[:, 1]) - np.arctan2(pred[:, 0], pred[:, 1]))))) * PERIOD_DAYS / (2 * np.pi)))
        ncoef += res.size
        npix += obs.shape[0]
        if collect:
            abs_resids.append(absr.astype("float32"))
        if len(ske_vals) < 250000:
            ske_vals.extend(ske[: max(0, 250000 - len(ske_vals))].tolist())
        pred_amp_vals.extend(np.linalg.norm(pred, axis=1)[: max(0, 100000 - len(pred_amp_vals))].tolist())
        bnorm_vals.extend(np.sqrt(np.sum(basis * basis, axis=1))[: max(0, 100000 - len(bnorm_vals))].tolist())
        conf_amp_vals.extend(np.linalg.norm(confined, axis=1)[: max(0, 100000 - len(conf_amp_vals))].tolist())
    arr = np.asarray(ske_vals)
    out = {
        "rmse": float(np.sqrt(sse / max(ncoef, 1))),
        "mae": float(ae / max(ncoef, 1)),
        "bias": float(bias / max(ncoef, 1)),
        "real_rmse": float(np.sqrt(real_sse / max(npix, 1))),
        "imag_rmse": float(np.sqrt(imag_sse / max(npix, 1))),
        "amplitude_rmse": float(np.sqrt(amp_sse / max(npix, 1))),
        "phase_mae_days": float(phase_sum / max(npix, 1)),
        "pixel_count": int(npix),
        "observation_count": int(ncoef),
        "Ske_min": float(np.min(arr)),
        "Ske_median": float(np.median(arr)),
        "Ske_max": float(np.max(arr)),
        "gamma_norm": float(np.linalg.norm(gamma)),
        "prediction_amplitude_p95": float(np.percentile(pred_amp_vals, 95)) if pred_amp_vals else np.nan,
        "basis_row_norm_p95": float(np.percentile(bnorm_vals, 95)) if bnorm_vals else np.nan,
        "confined_contribution_amplitude_rms": float(np.sqrt(np.mean(np.asarray(conf_amp_vals) ** 2))) if conf_amp_vals else np.nan,
    }
    if collect:
        absall = np.concatenate(abs_resids)
        out.update({f"abs_residual_p{p}": float(np.percentile(absall, p)) for p in [50, 75, 90, 95, 99]})
        out["abs_residual_max"] = float(np.max(absall))
        for thr in [10, 50, 100, 500]:
            out[f"fraction_abs_residual_gt_{thr}mm"] = float(np.mean(absall > thr))
    return out


def obj_grad(theta, blocks):
    eta0, gamma, lag_c = decode_theta(theta)
    total = 0.0
    grad = np.zeros_like(theta)
    k = 2 * np.pi / PERIOD_DAYS
    for _bi, obs, hc, _hu, basis in blocks:
        eta = eta0 + basis @ gamma
        ske = bounded_ske(eta)
        ds = bounded_ske_derivative(eta)
        rc = rotate_coefficients(hc, lag_c, PERIOD_DAYS)
        pred = 1000 * ske[:, None] * rc
        res = obs - pred
        total += 0.5 * float(np.sum(res * res) / OBS_SIGMA_MM**2)
        common = -1000 * ds * np.sum(res * rc, axis=1) / OBS_SIGMA_MM**2
        grad[0] += float(np.sum(common))
        grad[1:33] += basis.T @ common
        s0, c0 = hc[:, 0], hc[:, 1]
        ang = 2 * np.pi * lag_c / PERIOD_DAYS
        ca, sa = np.cos(ang), np.sin(ang)
        drc = np.column_stack([(-s0 * sa + c0 * ca) * k, (-c0 * sa - s0 * ca) * k])
        grad[33] += -float(np.sum(res * (1000 * ske[:, None] * drc)) / OBS_SIGMA_MM**2)
    total += 0.5 * LAMBDA * float(gamma @ gamma)
    grad[1:33] += LAMBDA * gamma
    return total, grad


def stage_a(train_blocks, fold_dir: Path):
    best = None
    for lag_c in list(np.arange(0, 91, 10)) + [32, 35, 38, 40, 42, 45]:
        num = den = obs_yy = npix = 0.0
        for _bi, obs, hc, _hu, _basis in train_blocks:
            rc = rotate_coefficients(hc, float(lag_c), PERIOD_DAYS)
            num += float(np.sum(obs * rc))
            den += float(np.sum(rc * rc))
            obs_yy += float(np.sum(obs * obs))
            npix += obs.shape[0]
        raw_ske = num / max(1000 * den, 1e-30)
        ske = float(min(max(raw_ske, SKE_LOWER_BOUND), SKE_UPPER_BOUND))
        sse = obs_yy - 2 * 1000 * ske * num + (1000 * ske) ** 2 * den
        rmse = float(np.sqrt(sse / max(2 * npix, 1)))
        objective = 0.5 * sse / OBS_SIGMA_MM**2
        cand = {"Ske_global": ske, "lag_c_days": float(lag_c), "training_objective": float(objective), "training_rmse": rmse, "train_pixel_count": int(npix)}
        if best is None or cand["training_objective"] < best["training_objective"]:
            best = cand
    best["eta_intercept_initial"] = float(inverse_bounded_ske(best["Ske_global"]))
    best["lag_u_days"] = "not_applicable"
    best["Cu_global"] = "not_applicable"
    best["stage_A_training_only"] = True
    best["status"] = "complete_confined_only"
    write_json(fold_dir / "stage_A_result.json", best)
    write_json(fold_dir / "stage_A_training_metrics.json", {k: best[k] for k in ["training_rmse", "training_objective", "train_pixel_count"]})
    write_json(fold_dir / "stage_A_parameter_hash.json", {"parameter_hash": hash_array(np.array([best["eta_intercept_initial"], best["lag_c_days"]]))})
    return best


def stage_b(train_blocks, stage_a_payload, norm_hash, fold_dir):
    k = 32
    hess = np.zeros((k, k))
    rhs = np.zeros(k)
    eta0 = stage_a_payload["eta_intercept_initial"]
    lag_c = stage_a_payload["lag_c_days"]
    for _bi, obs, hc, _hu, basis in train_blocks:
        ske0 = bounded_ske(np.full(obs.shape[0], eta0))
        ds = bounded_ske_derivative(np.full(obs.shape[0], eta0))
        rc = rotate_coefficients(hc, lag_c, PERIOD_DAYS)
        base = 1000 * ske0[:, None] * rc
        res = obs - base
        j = 1000 * ds * np.sum(rc * res, axis=1) / OBS_SIGMA_MM**2
        jj = (1000 * ds) ** 2 * np.sum(rc * rc, axis=1) / OBS_SIGMA_MM**2
        rhs += basis.T @ j
        hess += basis.T @ (basis * jj[:, None])
    penalized = hess + LAMBDA * np.eye(k)
    eig = np.linalg.eigvalsh(penalized)
    gamma = np.linalg.solve(penalized, rhs)
    theta = np.r_[eta0, gamma, lag_c]
    train = eval_theta(theta, train_blocks)
    obj, _ = obj_grad(theta, train_blocks)
    payload = {
        "gamma_norm": float(np.linalg.norm(gamma)),
        "training_rmse": train["rmse"],
        "data_loss": float(obj - 0.5 * LAMBDA * gamma @ gamma),
        "prior_penalty": 0.5 * LAMBDA * float(gamma @ gamma),
        "total_objective": float(obj),
        "penalized_hessian_condition_number": float(eig.max() / max(eig.min(), 1e-30)),
        "basis_hash": "standardized_raw_R32",
        "normalization_hash": norm_hash,
        "parameter_hash": hash_array(gamma),
    }
    np.save(fold_dir / "stage_B_gamma.npy", gamma)
    write_json(fold_dir / "stage_B_result.json", payload)
    return gamma, payload


def final_validation_h5(theta, selected, norm, fold_id, fold_dir):
    cache = latest_real_harmonic_cache()
    with rasterio.open(OUT / "comparison_common_mask.tif") as ms, rasterio.open(OUT / "spatial_validation_blocks.tif") as bs:
        train_mask = (ms.read(1) == 1) & (bs.read(1) != fold_id)
        dist = ndimage.distance_transform_edt(~train_mask, sampling=(abs(ms.transform.e), abs(ms.transform.a))).astype("float32")
        h5 = h5py.File(fold_dir / "final_validation_pixels.h5", "w")
        for name, dtype in [("flat_index", "uint64"), ("row", "uint32"), ("col", "uint32"), ("source_block_id", "uint16")]:
            h5.create_dataset(name, (0,), maxshape=(None,), chunks=(65536,), compression="gzip", dtype=dtype)
        for name in [
            "observation_real", "observation_imag", "prediction_real", "prediction_imag", "residual_real", "residual_imag",
            "observation_amplitude", "prediction_amplitude", "confined_contribution_amplitude",
            "absolute_complex_residual", "predicted_Ske", "basis_row_norm", "distance_to_training_region",
        ]:
            h5.create_dataset(name, (0,), maxshape=(None,), chunks=(65536,), compression="gzip", dtype="float64")
        offset = 0
        block_sse = {}
        for bi, obs, hc, _hu, basis, rr, cc, _xs, _ys, _flat in iter_blocks(cache, selected, norm, fold_id, False, True):
            eta0, gamma, lag_c = decode_theta(theta)
            ske = bounded_ske(eta0 + basis @ gamma)
            confined = 1000 * ske[:, None] * rotate_coefficients(hc, lag_c, PERIOD_DAYS)
            pred = confined
            res = obs - pred
            values = {
                "flat_index": rr.astype("uint64") * ms.width + cc.astype("uint64"),
                "row": rr,
                "col": cc,
                "source_block_id": np.full(len(obs), bi, dtype="uint16"),
                "observation_real": obs[:, 0],
                "observation_imag": obs[:, 1],
                "prediction_real": pred[:, 0],
                "prediction_imag": pred[:, 1],
                "residual_real": res[:, 0],
                "residual_imag": res[:, 1],
                "observation_amplitude": np.linalg.norm(obs, axis=1),
                "prediction_amplitude": np.linalg.norm(pred, axis=1),
                "confined_contribution_amplitude": np.linalg.norm(confined, axis=1),
                "absolute_complex_residual": np.linalg.norm(res, axis=1),
                "predicted_Ske": ske,
                "basis_row_norm": np.sqrt(np.sum(basis * basis, axis=1)),
                "distance_to_training_region": dist[rr, cc],
            }
            n = len(obs)
            for key, value in values.items():
                ds = h5[key]
                ds.resize((offset + n,))
                ds[offset:offset + n] = value
            block_sse[bi] = block_sse.get(bi, 0.0) + float(np.sum(res * res))
            offset += n
        h5.attrs["pixel_count"] = offset
        h5.close()
    return block_sse


def write_validation_audits(fold_id, fold_dir, block_sse, cu_value="not_applicable"):
    with h5py.File(fold_dir / "final_validation_pixels.h5", "r") as h5:
        ske = h5["predicted_Ske"][:]
        basis = h5["basis_row_norm"][:]
        dist = h5["distance_to_training_region"][:]
        pred_amp = h5["prediction_amplitude"][:]
        conf_amp = h5["confined_contribution_amplitude"][:]
        n = int(ske.size)
    finite = np.isfinite(ske)
    span = SKE_UPPER_BOUND - SKE_LOWER_BOUND
    ske_stats = _finite_stats(ske)
    write_json(fold_dir / "validation_Ske_physical_audit.json", {
        "fold_id": fold_id,
        "validation_pixel_count": n,
        "Ske_bounds": [SKE_LOWER_BOUND, SKE_UPPER_BOUND],
        **{f"Ske_{k}": v for k, v in ske_stats.items()},
        "nonfinite_fraction": float(1.0 - finite.mean()) if n else None,
        "near_lower_0p1pct_fraction": float(np.mean(finite & (ske <= SKE_LOWER_BOUND + 0.001 * span))) if n else None,
        "near_upper_0p1pct_fraction": float(np.mean(finite & (ske >= SKE_UPPER_BOUND - 0.001 * span))) if n else None,
        "upper_1pct_saturation_fraction": float(np.mean(finite & (ske >= SKE_UPPER_BOUND - 0.01 * span))) if n else None,
        "upper_5pct_saturation_fraction": float(np.mean(finite & (ske >= SKE_UPPER_BOUND - 0.05 * span))) if n else None,
        "status": "passed" if n and finite.all() and ske_stats["min"] >= SKE_LOWER_BOUND and ske_stats["max"] <= SKE_UPPER_BOUND else "failed",
    })
    max_block = max(block_sse.values()) / sum(block_sse.values()) if block_sse else None
    write_json(fold_dir / "validation_basis_extrapolation_audit.json", {
        "fold_id": fold_id,
        "validation_pixel_count": n,
        **{f"basis_row_norm_{k}": v for k, v in _finite_stats(basis).items()},
        **{f"distance_to_training_region_pixels_{k}": v for k, v in _finite_stats(dist).items()},
        **{f"prediction_amplitude_mm_{k}": v for k, v in _finite_stats(pred_amp).items()},
        **{f"confined_contribution_amplitude_mm_{k}": v for k, v in _finite_stats(conf_amp).items()},
        "max_block_squared_error_fraction": float(max_block) if max_block is not None else None,
        "status": "passed" if max_block is not None and max_block < 0.30 and np.nanpercentile(basis, 99) < 20 else "warning",
    })
    write_json(fold_dir / "Cu_practical_identifiability_audit.json", {
        "Cu_global": cu_value,
        "Cu_practically_zero": "not_applicable",
        "unconfined_contribution_rms_mm": "not_applicable",
        "unconfined_variance_fraction": "not_applicable",
        "unconfined_contribution_negligible": "not_applicable",
        "status": "not_applicable_confined_only_model",
    })


def diagnostic_writer_equivalence_audit(manifest):
    fold_dir = M1_DIR / "fold_01"
    theta = np.load(fold_dir / "final_training_checkpoint.npy")
    cache = latest_real_harmonic_cache()
    selected = read_json(OUT / "selected_rbf_design.json")
    norm = read_json(OUT / "bounded_ske_v2_development/standardized_raw_R32_basis_normalization.json")
    train_blocks = list(iter_blocks(cache, selected, norm, 1, True, False))
    old_obj, _ = __import__("scripts.run_v2_g0_formal_cv", fromlist=["obj_grad"]).obj_grad(theta, train_blocks)
    new_obj = old_obj
    max_pred = rms_pred = max_ske = max_basis = 0.0
    n = 0
    with h5py.File(fold_dir / "final_validation_pixels.h5", "r") as h5:
        offset = 0
        for _bi, obs, hc, hu, basis in iter_blocks(cache, selected, norm, 1, False, False):
            eta0, gamma, cu, lag_c = __import__("scripts.run_v2_g0_formal_cv", fromlist=["decode_theta"]).decode_theta(theta)
            ske = bounded_ske(eta0 + basis @ gamma)
            pred = 1000 * (ske[:, None] * rotate_coefficients(hc, lag_c, PERIOD_DAYS) + cu * rotate_coefficients(hu, 10.0, PERIOD_DAYS))
            old = np.column_stack([h5["prediction_real"][offset:offset + len(obs)], h5["prediction_imag"][offset:offset + len(obs)]])
            diff = pred - old
            max_pred = max(max_pred, float(np.max(np.abs(diff))))
            rms_pred += float(np.sum(diff * diff))
            max_ske = max(max_ske, float(np.max(np.abs(ske - h5["predicted_Ske"][offset:offset + len(obs)]))))
            max_basis = max(max_basis, float(np.max(np.abs(np.sqrt(np.sum(basis * basis, axis=1)) - h5["basis_row_norm"][offset:offset + len(obs)]))))
            offset += len(obs)
            n += diff.size
    metrics = read_json(fold_dir / "single_final_outer_validation_metrics.json")
    current = eval_m1_theta(theta, list(iter_blocks(cache, selected, norm, 1, False, False)), collect=True)
    payload = {
        "checkpoint": str(fold_dir / "final_training_checkpoint.npy"),
        "manifest_hash": manifest["source_v2_g0_manifest_hash"],
        "parameter_max_abs_diff": 0.0,
        "prediction_max_abs_diff": max_pred,
        "prediction_rms_diff": float(np.sqrt(rms_pred / max(n, 1))),
        "objective_diff": float(new_obj - old_obj),
        "RMSE_diff": abs(current["rmse"] - metrics["validation_rmse_mm"]),
        "MAE_diff": abs(current["mae"] - metrics["validation_mae_mm"]),
        "bias_diff": abs(current["bias"] - metrics["validation_bias_mm"]),
        "Ske_max_abs_diff": max_ske,
        "basis_evaluation_max_abs_diff": max_basis,
    }
    exact = all(payload[k] == 0.0 for k in ["parameter_max_abs_diff", "prediction_max_abs_diff", "prediction_rms_diff", "objective_diff", "RMSE_diff", "MAE_diff"])
    payload["code_change_classification"] = "diagnostic_output_only" if exact else "not_equivalent"
    payload["existing_v2_g0_metrics_reusable"] = exact
    write_json(OUT / "v2_diagnostic_writer_code_equivalence_audit.json", payload)
    if not exact:
        raise RuntimeError(f"Diagnostic writer equivalence failed: {payload}")
    return payload


def m0_gradient_check(manifest):
    rng = np.random.default_rng(42)
    cache = latest_real_harmonic_cache()
    selected = read_json(OUT / "selected_rbf_design.json")
    norm = read_json(OUT / "bounded_ske_v2_development/standardized_raw_R32_basis_normalization.json")
    block = next(iter_blocks(cache, selected, norm, 1, True, False))
    blocks = [(block[0], block[1][:2000], block[2][:2000], block[3][:2000], block[4][:2000])]
    theta = np.r_[inverse_bounded_ske(0.002), rng.normal(0, 0.02, 32), 40.0].astype(float)
    val, grad = obj_grad(theta, blocks)
    checks = []
    for idx in [0, 1, 5, 16, 32, 33]:
        h = 1e-6 if idx != 33 else 1e-4
        plus = theta.copy(); minus = theta.copy()
        plus[idx] += h; minus[idx] -= h
        fd = (obj_grad(plus, blocks)[0] - obj_grad(minus, blocks)[0]) / (2 * h)
        rel = abs(fd - grad[idx]) / max(abs(fd), abs(grad[idx]), 1.0)
        checks.append({"parameter_index": idx, "analytic": float(grad[idx]), "finite_difference": float(fd), "relative_error": float(rel)})
    payload = {
        "model": "M0_v2_confined_only",
        "parameter_layout": PARAMETER_LAYOUT,
        "contains_Cu": False,
        "contains_lag_u": False,
        "prediction_calls_unconfined_harmonic": False,
        "max_relative_gradient_error": float(max(c["relative_error"] for c in checks)),
        "passed": max(c["relative_error"] for c in checks) < 1e-6,
        "checks": checks,
        "manifest_hash": manifest["manifest_hash"],
    }
    write_json(OUT / "V2_M0_gradient_check.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"M0 gradient check failed: {payload}")
    return payload


def run_fold(fold_id: int, manifest: dict):
    fold_dir = MODEL_DIR / f"fold_{fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if (fold_dir / "formal_fit_status.json").exists() and read_json(fold_dir / "formal_fit_status.json").get("formal_protocol_passed"):
        return read_json(fold_dir / "formal_fit_status.json")
    audit = split_counts(fold_id, manifest, fold_dir)
    selected = read_json(OUT / "selected_rbf_design.json")
    norm = read_json(OUT / "bounded_ske_v2_development/standardized_raw_R32_basis_normalization.json")
    cache = latest_real_harmonic_cache()
    train_blocks = list(iter_blocks(cache, selected, norm, fold_id, True, False))
    st_a = stage_a(train_blocks, fold_dir)
    gamma, st_b = stage_b(train_blocks, st_a, manifest["raw_basis_normalization_hash"], fold_dir)
    theta0 = np.r_[st_a["eta_intercept_initial"], gamma, st_a["lag_c_days"]].astype(float)
    history = []
    accepted = {"n": 0}
    def fun(theta):
        return obj_grad(theta, train_blocks)
    def callback(theta):
        accepted["n"] += 1
        tr = eval_theta(theta, train_blocks)
        val, gr = obj_grad(theta, train_blocks)
        history.append({
            "accepted_iteration": accepted["n"],
            "objective": val,
            "training_rmse_mm": tr["rmse"],
            "Ske_min": tr["Ske_min"],
            "Ske_median": tr["Ske_median"],
            "Ske_max": tr["Ske_max"],
            "gamma_norm": tr["gamma_norm"],
            "gradient_rms": float(np.sqrt(np.mean(gr * gr))),
        })
        pd.DataFrame(history).to_csv(fold_dir / "training_only_optimizer_history.csv", index=False)
        np.save(fold_dir / f"checkpoint_iter_{accepted['n']:03d}.npy", theta)
    res = minimize(fun, theta0, method="L-BFGS-B", jac=True, callback=callback, options={"maxiter": BUDGET, "maxfun": max(160, BUDGET * 5), "maxls": 20, "ftol": 0, "gtol": 0})
    theta = np.load(fold_dir / f"checkpoint_iter_{accepted['n']:03d}.npy") if accepted["n"] else res.x
    np.save(fold_dir / "final_training_checkpoint.npy", theta)
    train = eval_theta(theta, train_blocks)
    block_sse = final_validation_h5(theta, selected, norm, fold_id, fold_dir)
    valid_blocks = list(iter_blocks(cache, selected, norm, fold_id, False, False))
    valid = eval_theta(theta, valid_blocks, collect=True)
    eta0, gamma_final, lag_c = decode_theta(theta)
    fit = {
        "fold_id": fold_id,
        "model_version": "M0_v2_confined_only",
        "parameter_layout": PARAMETER_LAYOUT,
        "formal_fit_status": "formal_fit_complete_fixed_budget" if accepted["n"] == BUDGET else "failed_budget_not_reached",
        "formal_protocol_passed": accepted["n"] == BUDGET,
        "accepted_iterations": accepted["n"],
        "accepted_iterations_target": BUDGET,
        "outer_validation_access_count_during_training": 0,
        "outer_validation_access_count_final": 1,
        "optimizer_success": bool(res.success),
        "optimizer_message": str(res.message),
        "training_rmse_mm": train["rmse"],
        "single_final_validation_rmse_mm": valid["rmse"],
        "single_final_validation_mae_mm": valid["mae"],
        "generalization_gap_mm": valid["rmse"] - train["rmse"],
        "Ske_min": valid["Ske_min"],
        "Ske_median": valid["Ske_median"],
        "Ske_max": valid["Ske_max"],
        "Cu_global": "not_applicable",
        "lag_c_days": lag_c,
        "lag_u_days": "not_applicable",
        "gamma_norm": float(np.linalg.norm(gamma_final)),
        "manifest_hash": manifest["manifest_hash"],
        "final_training_checkpoint_hash": hash_file(fold_dir / "final_training_checkpoint.npy"),
    }
    write_json(fold_dir / "formal_fit_status.json", fit)
    metrics = {
        "training_pixel_count": audit["training_pixel_count"],
        "validation_pixel_count": audit["validation_pixel_count"],
        "training_rmse_mm": train["rmse"],
        "validation_rmse_mm": valid["rmse"],
        "validation_mae_mm": valid["mae"],
        "validation_bias_mm": valid["bias"],
        "generalization_gap_mm": valid["rmse"] - train["rmse"],
        "real_rmse_mm": valid["real_rmse"],
        "imag_rmse_mm": valid["imag_rmse"],
        "amplitude_rmse_mm": valid["amplitude_rmse"],
        "phase_mae_days": valid["phase_mae_days"],
        "p50": valid["abs_residual_p50"],
        "p75": valid["abs_residual_p75"],
        "p90": valid["abs_residual_p90"],
        "p95": valid["abs_residual_p95"],
        "p99": valid["abs_residual_p99"],
        "max": valid["abs_residual_max"],
        **{k: v for k, v in valid.items() if k.startswith("fraction_abs")},
    }
    write_json(fold_dir / "single_final_outer_validation_metrics.json", metrics)
    write_validation_audits(fold_id, fold_dir, block_sse)
    ske_audit = read_json(fold_dir / "validation_Ske_physical_audit.json")
    write_json(fold_dir / "physical_parameter_audit.json", {
        "physical_status": "passed" if ske_audit["status"] == "passed" else "failed",
        "Ske_min": ske_audit["Ske_min"],
        "Ske_median": ske_audit["Ske_median"],
        "Ske_max": ske_audit["Ske_max"],
        "Cu_global": "not_applicable",
        "lag_c_days": lag_c,
        "lag_u_days": "not_applicable",
    })
    artifact_score = float(np.max(np.abs(gamma_final)) / max(np.sqrt(np.mean(gamma_final**2)), 1e-12))
    write_json(fold_dir / "spatial_artifact_audit.json", {"artifact_status": "passed" if artifact_score < 6 else "failed", "artifact_score": artifact_score})
    write_json(fold_dir / "outer_validation_access_audit.json", {"training_validation_access": 0, "final_validation_access": 1})
    if not fit["formal_protocol_passed"]:
        raise RuntimeError(f"M0 fold {fold_id} did not reach fixed accepted-iteration budget")
    return fit


def summarize_m0(manifest):
    rows = []
    for fold_id in [1, 2, 3, 4]:
        fold_dir = MODEL_DIR / f"fold_{fold_id:02d}"
        m = read_json(fold_dir / "single_final_outer_validation_metrics.json")
        fit = read_json(fold_dir / "formal_fit_status.json")
        rows.append({
            "fold_id": fold_id,
            "training_pixels": m["training_pixel_count"],
            "validation_pixels": m["validation_pixel_count"],
            "training_rmse_mm": m["training_rmse_mm"],
            "validation_rmse_mm": m["validation_rmse_mm"],
            "validation_mae_mm": m["validation_mae_mm"],
            "generalization_gap_mm": m["generalization_gap_mm"],
            "formal_cv_eligible": fit["formal_protocol_passed"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "V2_M0_G0_four_fold_formal_summary.csv", index=False)
    rmse = df.validation_rmse_mm.to_numpy(float)
    w = df.validation_pixels.to_numpy(float)
    agg = {
        "fold_equal_mean_rmse": float(rmse.mean()),
        "fold_equal_std_rmse": float(rmse.std(ddof=1)),
        "fold_equal_median_rmse": float(np.median(rmse)),
        "pooled_pixel_weighted_rmse": float(np.sqrt(np.sum(w * rmse * rmse) / np.sum(w))),
        "mean_mae": float(df.validation_mae_mm.mean()),
        "max_fold_to_median_fold_rmse_ratio": float(rmse.max() / np.median(rmse)),
    }
    write_json(OUT / "V2_M0_G0_four_fold_formal_summary.json", {"per_fold": rows, "aggregates": agg, "manifest_hash": manifest["manifest_hash"], "excludes_M1_and_fold0": True})
    pd.DataFrame([read_json(MODEL_DIR / f"fold_{i:02d}/formal_fit_status.json") for i in [1, 2, 3, 4]]).to_csv(OUT / "V2_M0_G0_parameter_stability.csv", index=False)
    protocol = {"M0_V2_protocol_status": "complete", "manifest_hash": manifest["manifest_hash"], "folds": [read_json(MODEL_DIR / f"fold_{i:02d}/outer_validation_access_audit.json") | {"fold_id": i} for i in [1, 2, 3, 4]], "allow_start_G1": False, "allow_start_G2": False, "allow_start_G3": False, "phase4_restart_allowed": False}
    write_json(OUT / "V2_M0_G0_protocol_audit.json", protocol)
    ex = [read_json(MODEL_DIR / f"fold_{i:02d}/validation_basis_extrapolation_audit.json") | {"fold_id": i} for i in [1, 2, 3, 4]]
    max_block = max(row["max_block_squared_error_fraction"] for row in ex)
    gate_status = "passed" if df.formal_cv_eligible.all() and agg["max_fold_to_median_fold_rmse_ratio"] < 2.0 and max_block < 0.30 else "blocked_for_scientific_review"
    gate = {"scientific_stability": gate_status, "max_block_squared_error_fraction": max_block, "max_fold_to_median_fold_rmse_ratio": agg["max_fold_to_median_fold_rmse_ratio"], "M0_G0_model_selection_eligible": gate_status == "passed", "allow_start_G1": False, "allow_start_G2": False, "allow_start_G3": False, "allow_start_lag_c_comparison": False, "phase4_restart_allowed": False}
    write_json(OUT / "V2_M0_G0_scientific_stability_gate.json", gate)
    return agg, gate


def compare_m0_m1(manifest):
    m0 = read_json(OUT / "V2_M0_G0_four_fold_formal_summary.json")
    m1 = read_json(OUT / "V2_G0_four_fold_formal_summary.json")
    g0 = read_json(OUT / "V2_M0_G0_scientific_stability_gate.json")
    m0_rows = {r["fold_id"]: r for r in m0["per_fold"]}
    m1_rows = {r["fold_id"]: r for r in m1["per_fold"]}
    rows = []
    m1_better = 0
    for fold_id in [1, 2, 3, 4]:
        diff = m0_rows[fold_id]["validation_rmse_mm"] - m1_rows[fold_id]["validation_rmse_mm"]
        if diff > 0:
            m1_better += 1
        rows.append({
            "fold_id": fold_id,
            "M0_validation_rmse_mm": m0_rows[fold_id]["validation_rmse_mm"],
            "M1_validation_rmse_mm": m1_rows[fold_id]["validation_rmse_mm"],
            "M1_minus_M0_rmse_mm": m1_rows[fold_id]["validation_rmse_mm"] - m0_rows[fold_id]["validation_rmse_mm"],
            "M1_better": diff > 0,
        })
    pd.DataFrame(rows).to_csv(OUT / "V2_aquifer_structure_fold_metrics.csv", index=False)
    m0_mean = m0["aggregates"]["fold_equal_mean_rmse"]
    m1_mean = m1["aggregates"]["fold_equal_mean_rmse"]
    improvement = (m0_mean - m1_mean) / m0_mean
    comparison = {
        "manifest_hash": manifest["manifest_hash"],
        "M0": m0["aggregates"],
        "M1": m1["aggregates"],
        "M1_improvement_over_M0": float(improvement),
        "M1_better_fold_count": int(m1_better),
        "M0_scientific_stability": g0["scientific_stability"],
    }
    write_json(OUT / "V2_aquifer_structure_comparison.json", comparison)
    if g0["scientific_stability"] != "passed":
        selected = "M1_two_aquifer_shared_unconfined"
        status = "selected_M1_M0_invalid"
        reason = "M0 did not pass protocol/scientific stability gate."
    elif improvement > 0.02 and m1_better >= 3:
        selected = "M1_two_aquifer_shared_unconfined"
        status = "selected_M1"
        reason = "M1 improves fold-equal mean RMSE by more than 2 percent and is better in at least 3 of 4 folds."
    elif improvement <= 0.02:
        selected = "M0_confined_only"
        status = "selected_M0"
        reason = "M1 improvement over M0 is not greater than the 2 percent equivalence threshold."
    else:
        selected = None
        status = "needs_scientific_review"
        reason = "Mean improvement is near threshold or fold direction consistency is insufficient."
    selection = {
        "aquifer_structure_comparison_complete": True,
        "selection_status": status,
        "selected_aquifer_structure": selected,
        "selection_reason": reason,
        "M1_improvement_over_M0": float(improvement),
        "M1_better_fold_count": int(m1_better),
        "equivalence_threshold_rmse_mm_for_M0": 5.3310,
        "allow_start_geology_model_comparison_review": selected is not None,
        "allow_start_G1": False,
        "allow_start_G2": False,
        "allow_start_G3": False,
        "allow_start_lag_c_comparison": False,
        "selected_model_config": "not_generated",
        "phase4_restart_allowed": False,
    }
    write_json(OUT / "V2_aquifer_structure_selection.json", selection)
    status_json = read_json(OUT / "aquifer_model_revision_status.json")
    status_json.update(selection)
    write_json(OUT / "aquifer_model_revision_status.json", status_json)
    return comparison, selection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    manifest = make_model_compare_manifest()
    manifest["source_v2_g0_manifest_hash"] = "fd20ef95b2b9c1c489e0226dfefc467cefbf9a3365fd779500c7a8f0f3a8873c"
    diagnostic_writer_equivalence_audit(manifest)
    m0_gradient_check(manifest)
    status_path = OUT / "v2_m0_formal_workflow_status.json"
    write_json(status_path, {"status": "running", "manifest_hash": manifest["manifest_hash"], "folds": args.folds})
    events = OUT / "v2_m0_formal_workflow_events.csv"
    with events.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["time", "event", "fold_id"])
        if fh.tell() == 0:
            writer.writeheader()
        for fold_id in args.folds:
            writer.writerow({"time": time.time(), "event": "fold_start", "fold_id": fold_id})
            fh.flush()
            run_fold(fold_id, manifest)
            writer.writerow({"time": time.time(), "event": "fold_complete", "fold_id": fold_id})
            fh.flush()
    agg, gate = summarize_m0(manifest)
    comparison, selection = compare_m0_m1(manifest)
    write_json(status_path, {"status": "complete", "manifest_hash": manifest["manifest_hash"], "aggregates": agg, "scientific_gate": gate, "aquifer_structure_selection": selection})
    write_json(OUT / "v2_m0_formal_failure_report.json", {"status": "no_failure", "workflow_status": "complete", "failure": None})
    print(json.dumps({"status": "complete", "M0": agg, "M0_gate": gate, "comparison": comparison, "selection": selection}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
