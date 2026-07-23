#!/usr/bin/env python
"""Develop bounded-Ske V2 candidates on fold0 only."""
from __future__ import annotations

import csv
import json
import sys
import time
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import xy
from rasterio.windows import Window
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bounded_ske_v2 import (
    OBJECTIVE_VERSION,
    PARAMETERIZATION_VERSION,
    SKE_LOWER_BOUND,
    SKE_UPPER_BOUND,
    bounded_ske,
    bounded_ske_derivative,
    inverse_bounded_ske,
    saturation_fractions,
)
from profiled_stage_a import latest_real_harmonic_cache
from scripts.run_stage_b_fixed_lagu import rbf_values
from scripts.run_stage_c_fixed_lagu import LAG_U_FIXED_DAYS
from storage_inversion import rotate_coefficients


OUT = Path("outputs/aquifer_model_revision")
DEV = OUT / "bounded_ske_v2_development"
OBS_SIGMA_MM = 5.0
PERIOD_DAYS = 365.2425
LAMBDA = 30.0
FOLD_ID = 0
BUDGET = 40


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def hash_array(arr: np.ndarray) -> str:
    return sha256(np.asarray(arr, dtype="float64").tobytes()).hexdigest()


def hash_json(payload: dict) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def basis_raw_standardization(selected: dict) -> dict:
    mask_path = OUT / "comparison_common_mask.tif"
    centers = np.asarray(selected["center_coordinates"], float)
    sigma_m = float(selected["sigma_km"]) * 1000.0
    sums = np.zeros(centers.shape[0]); sumsq = np.zeros(centers.shape[0]); n = 0
    with rasterio.open(mask_path) as src:
        transformer = None
        if selected.get("projected_crs") and src.crs and str(src.crs) != selected["projected_crs"]:
            transformer = Transformer.from_crs(src.crs, selected["projected_crs"], always_xy=True)
        for _, window in src.block_windows(1):
            mask = src.read(1, window=window) == 1
            if not mask.any():
                continue
            rr, cc = np.nonzero(mask)
            rr = rr + int(window.row_off); cc = cc + int(window.col_off)
            xs, ys = xy(src.transform, rr, cc, offset="center")
            xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            phi = rbf_values(np.column_stack([xs, ys]), centers, sigma_m)
            sums += phi.sum(axis=0); sumsq += (phi * phi).sum(axis=0); n += phi.shape[0]
    mean = sums / max(n, 1)
    rms = np.sqrt(np.maximum(sumsq / max(n, 1) - mean * mean, 1e-30))
    payload = {
        "basis_type": "standardized_raw_R32_gaussian",
        "uses_response_variables": False,
        "common_mask_pixel_count": int(n),
        "raw_basis_mean": mean.tolist(),
        "raw_basis_rms": rms.tolist(),
        "normalization_hash": hash_json({"mean": mean.tolist(), "rms": rms.tolist(), "sigma_km": selected["sigma_km"], "centers": selected["center_coordinates"]}),
        "rbf_centers_hash": hash_array(centers),
        "sigma_km": selected["sigma_km"],
    }
    write_json(DEV / "standardized_raw_R32_basis_normalization.json", payload)
    return payload


def iter_fold_blocks(cache_path: Path, selected: dict, basis_kind: str, transform=None, raw_norm=None, train=True):
    centers = np.asarray(selected["center_coordinates"], float)
    sigma_m = float(selected["sigma_km"]) * 1000.0
    mask_path = OUT / "comparison_common_mask.tif"
    blocks_path = OUT / "spatial_validation_blocks.tif"
    with h5py.File(cache_path, "r") as h5, rasterio.open(mask_path) as mask_src, rasterio.open(blocks_path) as block_src:
        transformer = None
        if selected.get("projected_crs") and mask_src.crs and str(mask_src.crs) != selected["projected_crs"]:
            transformer = Transformer.from_crs(mask_src.crs, selected["projected_crs"], always_xy=True)
        for bi, start in enumerate(h5["block_start"][:]):
            count = int(h5["block_count"][bi])
            if count == 0:
                continue
            start = int(start); r = int(h5["block_row"][bi]); c = int(h5["block_col"][bi])
            h = int(h5["block_height"][bi]); w = int(h5["block_width"][bi])
            window = Window(c, r, w, h)
            flat = h5["flat_index"][start:start + count].astype(int)
            rows = flat // w; cols = flat % w
            valid = mask_src.read(1, window=window).ravel()[flat] == 1
            folds = block_src.read(1, window=window).ravel()[flat]
            take = folds != FOLD_ID if train else folds == FOLD_ID
            obs = h5["obs"][start:start + count]
            hc = h5["hc"][start:start + count]
            hu = h5["hu"][start:start + count]
            common = valid & take & np.isfinite(obs).all(1) & np.isfinite(hc).all(1) & np.isfinite(hu).all(1)
            if not common.any():
                continue
            rr = r + rows[common]; cc = c + cols[common]
            xs, ys = xy(mask_src.transform, rr, cc, offset="center")
            xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            phi = rbf_values(np.column_stack([xs, ys]), centers, sigma_m)
            if basis_kind == "orthogonal":
                basis = phi @ transform
            else:
                mean = np.asarray(raw_norm["raw_basis_mean"], float)
                rms = np.asarray(raw_norm["raw_basis_rms"], float)
                basis = (phi - mean) / rms
            yield int(bi), obs[common].astype(float), hc[common].astype(float), hu[common].astype(float), basis.astype(float), xs, ys


def decode(theta):
    return theta[0], theta[1:33], float(np.exp(theta[33])), float(theta[34])


def objective_grad(theta, blocks, return_parts=False):
    eta0, gamma, cu, lag_c = decode(theta)
    total = 0.0
    grad = np.zeros_like(theta)
    k = 2.0 * np.pi / PERIOD_DAYS
    sse = ae = ncoef = 0.0
    ske_sample = []
    gamma_prior = 0.5 * LAMBDA * float(gamma @ gamma)
    for _bi, obs, hc, hu, basis, _x, _y in blocks:
        eta = eta0 + basis @ gamma
        ske = bounded_ske(eta)
        ds = bounded_ske_derivative(eta)
        rc = rotate_coefficients(hc, lag_c, PERIOD_DAYS)
        ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS, PERIOD_DAYS)
        pred = 1000.0 * (ske[:, None] * rc + cu * ru)
        res = obs - pred
        total += 0.5 * float(np.sum(res * res) / OBS_SIGMA_MM**2)
        sse += float(np.sum(res * res)); ae += float(np.sum(np.abs(res))); ncoef += res.size
        common = -1000.0 * ds * np.sum(res * rc, axis=1) / OBS_SIGMA_MM**2
        grad[0] += float(np.sum(common))
        grad[1:33] += basis.T @ common
        grad[33] += -float(np.sum(res * (1000.0 * cu * ru)) / OBS_SIGMA_MM**2)
        s0, c0 = hc[:, 0], hc[:, 1]
        angle = 2.0 * np.pi * lag_c / PERIOD_DAYS
        ca, sa = np.cos(angle), np.sin(angle)
        drc = np.column_stack([(-s0 * sa + c0 * ca) * k, (-c0 * sa - s0 * ca) * k])
        grad[34] += -float(np.sum(res * (1000.0 * ske[:, None] * drc)) / OBS_SIGMA_MM**2)
        if len(ske_sample) < 200000:
            ske_sample.extend(ske[: 200000 - len(ske_sample)].tolist())
    total += gamma_prior
    grad[1:33] += LAMBDA * gamma
    if return_parts:
        arr = np.asarray(ske_sample, float)
        return total, grad, {
            "rmse": float(np.sqrt(sse / max(ncoef, 1))),
            "mae": float(ae / max(ncoef, 1)),
            "Ske_min": float(np.min(arr)),
            "Ske_median": float(np.median(arr)),
            "Ske_max": float(np.max(arr)),
            "gamma_norm": float(np.linalg.norm(gamma)),
        }
    return total, grad


def finite_difference_check(theta, blocks, name: str) -> dict:
    val, grad = objective_grad(theta, blocks)
    idxs = [0, 1, 5, 16, 32, 33, 34]
    rows = []
    max_rel = 0.0
    for idx in idxs:
        step = 1e-6 * max(abs(theta[idx]), 1.0)
        tp = theta.copy(); tm = theta.copy()
        tp[idx] += step; tm[idx] -= step
        vp, _ = objective_grad(tp, blocks)
        vm, _ = objective_grad(tm, blocks)
        fd = (vp - vm) / (2 * step)
        rel = abs(fd - grad[idx]) / max(abs(fd), abs(grad[idx]), 1.0)
        max_rel = max(max_rel, rel)
        rows.append({"parameter_index": idx, "analytic": float(grad[idx]), "finite_difference": float(fd), "relative_error": float(rel)})
    payload = {"candidate": name, "max_relative_gradient_error": float(max_rel), "passed": bool(max_rel < 1e-6), "checks": rows}
    write_json(DEV / f"{name}_gradient_check.json", payload)
    return payload


def evaluate(theta, blocks):
    _val, _grad, parts = objective_grad(theta, blocks, return_parts=True)
    return parts


def run_candidate(name: str, basis_kind: str, selected: dict, transform=None, raw_norm=None) -> dict:
    cache = latest_real_harmonic_cache()
    train_blocks = list(iter_fold_blocks(cache, selected, basis_kind, transform, raw_norm, train=True))
    valid_blocks = list(iter_fold_blocks(cache, selected, basis_kind, transform, raw_norm, train=False))
    stage_a = json.loads((OUT / "model_compare/G0_no_geology_L0_shared/fold_00/stage_A/stage_A_fixed_lag_u_10d_result.json").read_text())
    eta0 = float(inverse_bounded_ske(stage_a["Ske_global"]))
    theta0 = np.r_[eta0, np.zeros(32), np.log(stage_a["Cu_global"]), stage_a["lag_c_days"]].astype(float)
    grad = finite_difference_check(theta0, train_blocks[:2], name)
    if not grad["passed"]:
        raise RuntimeError(f"{name} gradient check failed: {grad['max_relative_gradient_error']}")
    hist = []
    prev = theta0.copy()
    accepted = {"n": 0}
    def fun(theta):
        val, g = objective_grad(theta, train_blocks)
        return val, g
    def cb(theta):
        accepted["n"] += 1
        _v, g, train = objective_grad(theta, train_blocks, return_parts=True)
        hist.append({"accepted_iteration": accepted["n"], **train, "gradient_rms": float(np.sqrt(np.mean(g*g)))})
        pd.DataFrame(hist).to_csv(DEV / f"{name}_stageC_history.csv", index=False)
    result = minimize(fun, theta0, method="L-BFGS-B", jac=True, callback=cb, options={"maxiter": BUDGET, "maxfun": max(160, BUDGET*5), "maxls": 20, "ftol": 0, "gtol": 0})
    theta = result.x.astype(float)
    np.save(DEV / f"{name}_stageC_theta.npy", theta)
    train = evaluate(theta, train_blocks)
    valid = evaluate(theta, valid_blocks)
    # Approximate Hessian condition for gamma block via basis norm proxy.
    block_rows = pressure_blocks(name, selected, basis_kind, theta, transform, raw_norm)
    max_basis = max(r["basis_row_norm_p95"] for r in block_rows)
    artifact_score = max(r["Ske_max"] for r in block_rows) / max(np.median([r["Ske_median"] for r in block_rows]), 1e-30)
    payload = {
        "candidate": name,
        "model": "M1_v2_bounded_Ske",
        "basis": basis_kind,
        "ske_parameterization": "bounded_logistic",
        "ske_lower_bound": SKE_LOWER_BOUND,
        "ske_upper_bound": SKE_UPPER_BOUND,
        "objective_version": OBJECTIVE_VERSION,
        "parameterization_version": PARAMETERIZATION_VERSION,
        "gradient_check": grad,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "accepted_iterations": int(accepted["n"]),
        "training_RMSE": train["rmse"],
        "development_validation_RMSE": valid["rmse"],
        "Ske_min": train["Ske_min"],
        "Ske_median": train["Ske_median"],
        "Ske_max": train["Ske_max"],
        "gamma_norm": train["gamma_norm"],
        "Hessian_condition_proxy": float(max_basis),
        "artifact_score": float(artifact_score),
    }
    write_json(DEV / f"{name}_development_result.json", payload)
    return payload


def pressure_blocks(name, selected, basis_kind, theta, transform=None, raw_norm=None):
    cache = latest_real_harmonic_cache()
    rows = []
    for bi, _obs, _hc, _hu, basis, _x, _y in iter_fold_blocks(cache, selected, basis_kind, transform, raw_norm, train=True):
        # train=True covers all non-fold0; add validation later by a second pass below? Pressure should be full common mask.
        pass
    rows_by_block = {}
    for train in (True, False):
        for bi, _obs, _hc, _hu, basis, x, y in iter_fold_blocks(cache, selected, basis_kind, transform, raw_norm, train=train):
            eta0, gamma, _cu, _lag = decode(theta)
            eta = eta0 + basis @ gamma
            ske = bounded_ske(eta)
            bnorm = np.sqrt(np.sum(basis*basis, axis=1))
            lev = bnorm * bnorm
            if bi not in rows_by_block:
                rows_by_block[bi] = {"eta": [], "ske": [], "bnorm": [], "lev": [], "n": 0}
            rows_by_block[bi]["eta"].append(eta.astype("float32"))
            rows_by_block[bi]["ske"].append(ske.astype("float32"))
            rows_by_block[bi]["bnorm"].append(bnorm.astype("float32"))
            rows_by_block[bi]["lev"].append(lev.astype("float32"))
            rows_by_block[bi]["n"] += len(eta)
    for bi, d in rows_by_block.items():
        eta = np.concatenate(d["eta"]); ske = np.concatenate(d["ske"]); bnorm = np.concatenate(d["bnorm"]); lev = np.concatenate(d["lev"])
        sat = saturation_fractions(ske)
        rows.append({
            "candidate": name,
            "block_id": int(bi),
            "pixel_count": int(d["n"]),
            "eta_min": float(np.min(eta)),
            "eta_median": float(np.median(eta)),
            "eta_p95": float(np.percentile(eta, 95)),
            "eta_p99": float(np.percentile(eta, 99)),
            "eta_max": float(np.max(eta)),
            "Ske_min": float(np.min(ske)),
            "Ske_median": float(np.median(ske)),
            "Ske_p95": float(np.percentile(ske, 95)),
            "Ske_p99": float(np.percentile(ske, 99)),
            "Ske_max": float(np.max(ske)),
            "basis_row_norm_median": float(np.median(bnorm)),
            "basis_row_norm_p95": float(np.percentile(bnorm, 95)),
            "basis_row_norm_max": float(np.max(bnorm)),
            "basis_leverage_p95": float(np.percentile(lev, 95)),
            "nonfinite_fraction": float(np.mean(~np.isfinite(ske))),
            **sat,
        })
    pd.DataFrame(rows).to_csv(DEV / f"{name}_full_domain_pressure_by_block.csv", index=False)
    return rows


def select_model(results):
    rows = []
    for res in results:
        pressure = pd.read_csv(DEV / f"{res['candidate']}_full_domain_pressure_by_block.csv")
        max_sat = float(pressure["fraction_Ske_within_5pct_of_upper_bound"].max())
        max_ske = float(pressure["Ske_max"].max())
        max_basis = float(pressure["basis_row_norm_p95"].max())
        failure = bool(max_sat > 0.25 or pressure["nonfinite_fraction"].max() > 0 or max_ske > SKE_UPPER_BOUND + 1e-12)
        rows.append({**res, "max_upper_5pct_saturation": max_sat, "max_full_domain_Ske": max_ske, "max_block_basis_row_norm_p95": max_basis, "bounded_parameter_saturation_failure": failure})
    pd.DataFrame(rows).to_csv(DEV / "bounded_ske_development_comparison.csv", index=False)
    v2a, v2b = rows
    selected = None
    reason = []
    if v2a["bounded_parameter_saturation_failure"] and not v2b["bounded_parameter_saturation_failure"]:
        selected = v2b; reason.append("V2a has saturation or full-domain stability failure while V2b does not.")
    elif (not v2b["bounded_parameter_saturation_failure"]) and v2b["development_validation_RMSE"] <= 1.02 * v2a["development_validation_RMSE"]:
        selected = v2b; reason.append("V2b is stable and development RMSE is within 2 percent of V2a.")
    elif not v2a["bounded_parameter_saturation_failure"]:
        selected = v2a; reason.append("V2a has lower development RMSE without triggering pressure-test failure.")
    payload = {
        "model_selection_status": "selected_stable_candidate" if selected else "no_stable_candidate",
        "selected_candidate": None if selected is None else selected["candidate"],
        "selection_reason": reason,
        "formal_v2_execution_allowed": False,
        "candidate_rows": rows,
    }
    write_json(DEV / "bounded_ske_model_selection.json", payload)
    write_json(DEV / "bounded_ske_basis_stability_audit.json", {"candidate_rows": rows})
    if selected:
        manifest = {
            "manifest_status": "draft_not_authorized_for_formal_execution",
            "new_model_version": "M1_v2_bounded_Ske",
            "bounded_logistic_parameterization": PARAMETERIZATION_VERSION,
            "Ske_bounds": [SKE_LOWER_BOUND, SKE_UPPER_BOUND],
            "basis_type": selected["basis"],
            "basis_normalization_hash": json.loads((DEV / "standardized_raw_R32_basis_normalization.json").read_text()).get("normalization_hash") if selected["basis"] == "raw_standardized" else "weighted_orthogonal_basis_existing",
            "RBF_centers_hash": json.loads((DEV / "standardized_raw_R32_basis_normalization.json").read_text()).get("rbf_centers_hash"),
            "sigma_km": json.loads((OUT / "selected_rbf_design.json").read_text())["sigma_km"],
            "lambda": LAMBDA,
            "lag_u": LAG_U_FIXED_DAYS,
            "Stage_C_budget": BUDGET,
            "objective_version": OBJECTIVE_VERSION,
            "gradient_test_result": selected["gradient_check"],
            "development_selection_result": payload,
            "formal_v2_execution_allowed": False,
        }
        write_json(OUT / "formal_protocol_v2_draft_manifest.json", manifest)
    return payload


def main():
    DEV.mkdir(parents=True, exist_ok=True)
    selected = json.loads((OUT / "selected_rbf_design.json").read_text())
    transform = np.load(OUT / "rbf_orthogonalization/rbf_transform.npy")
    raw_norm = basis_raw_standardization(selected)
    results = [
        run_candidate("V2a_bounded_orthogonal_R32", "orthogonal", selected, transform=transform),
        run_candidate("V2b_bounded_standardized_raw_R32", "raw_standardized", selected, raw_norm=raw_norm),
    ]
    selection = select_model(results)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
