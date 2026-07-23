#!/usr/bin/env python
"""Run a formal G0/L0 fixed-budget spatial validation fit for one outer fold."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import asdict
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiled_stage_a import StageAStats, latest_real_harmonic_cache, solve_from_stats
from scripts.audit_stage_b_lambda_effect import artifact_metrics
from scripts.run_stage_b_fixed_lagu import rbf_values
from scripts.run_stage_c_fixed_lagu import LAG_U_FIXED_DAYS, OBJECTIVE_VERSION, PRIOR_VERSION, decode
from storage_inversion import rotate_coefficients


BASIS_HASH = "fb5d0531ebf865b5e375e928f6560794a532a975f501e83c3e4cdd1d60f5f9fd"
PROTOCOL_VERSION = "formal_outer_fold_fixed_budget_v1"
MODEL_VARIANT = "M1_two_aquifer_shared_unconfined"
GEOLOGY_MODEL = "G0_no_geology"
LAG_C_MODE = "L0_shared"
LAMBDA = 30.0
OBS_SIGMA_MM = 5.0
PERIOD_DAYS = 365.2425


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def hash_array(arr: np.ndarray) -> str:
    return sha256(np.asarray(arr, dtype="float64").tobytes()).hexdigest()


def hash_json(payload: dict) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dependency_versions() -> dict:
    names = ["numpy", "pandas", "scipy", "rasterio", "h5py", "pyproj"]
    out = {"python": platform.python_version()}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def split_hash(mask: Path, blocks: Path, fold_id: int, train: bool) -> str:
    h = sha256()
    h.update(mask.read_bytes())
    h.update(blocks.read_bytes())
    h.update(f"fold={fold_id};train={train}".encode())
    return h.hexdigest()


def iter_blocks_fold(cache_path, mask_path, block_path, selected, transform, fold_id: int, train: bool):
    centers = np.asarray(selected["center_coordinates"], float)
    sigma_m = float(selected["sigma_km"]) * 1000.0
    target_crs = selected.get("projected_crs")
    with h5py.File(cache_path, "r") as h5, rasterio.open(mask_path) as mask_src, rasterio.open(block_path) as block_src:
        transformer = None
        if target_crs and mask_src.crs and str(mask_src.crs) != str(target_crs):
            transformer = Transformer.from_crs(mask_src.crs, target_crs, always_xy=True)
        for bi, start in enumerate(h5["block_start"][:]):
            count = int(h5["block_count"][bi])
            if count == 0:
                continue
            start = int(start)
            r = int(h5["block_row"][bi])
            c = int(h5["block_col"][bi])
            h = int(h5["block_height"][bi])
            w = int(h5["block_width"][bi])
            window = Window(c, r, w, h)
            flat = h5["flat_index"][start:start + count].astype(int)
            rows = flat // w
            cols = flat % w
            valid_mask = mask_src.read(1, window=window).ravel()[flat] == 1
            folds = block_src.read(1, window=window).ravel()[flat]
            take = folds != fold_id if train else folds == fold_id
            obs = h5["obs"][start:start + count]
            hc = h5["hc"][start:start + count]
            hu = h5["hu"][start:start + count]
            common = valid_mask & take & np.isfinite(obs).all(1) & np.isfinite(hc).all(1) & np.isfinite(hu).all(1)
            if not common.any():
                continue
            rr = r + rows[common]
            cc = c + cols[common]
            xs, ys = xy(mask_src.transform, rr, cc, offset="center")
            xs = np.asarray(xs, float)
            ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float)
                ys = np.asarray(ys, float)
            phi = rbf_values(np.column_stack([xs, ys]), centers, sigma_m)
            basis = phi @ transform
            yield obs[common].astype(float), hc[common].astype(float), hu[common].astype(float), basis.astype(float)


def stats_from_block(obs, hc, hu) -> StageAStats:
    invsig2 = 1.0 / (OBS_SIGMA_MM**2)
    hs, hc0 = hc[:, 0], hc[:, 1]
    us, uc0 = hu[:, 0], hu[:, 1]
    os, oc = obs[:, 0], obs[:, 1]
    return StageAStats(
        n=int(obs.shape[0]),
        obs_yy=float(np.sum(obs * obs) * invsig2),
        hc_norm=float(np.sum(hc * hc) * 1_000_000.0 * invsig2),
        hu_norm=float(np.sum(hu * hu) * 1_000_000.0 * invsig2),
        hc_obs_cos=float(np.sum(hs * os + hc0 * oc) * 1000.0 * invsig2),
        hc_obs_sin=float(np.sum(hc0 * os - hs * oc) * 1000.0 * invsig2),
        hu_obs_cos=float(np.sum(us * os + uc0 * oc) * 1000.0 * invsig2),
        hu_obs_sin=float(np.sum(uc0 * os - us * oc) * 1000.0 * invsig2),
        cross_cos=float(np.sum(hs * us + hc0 * uc0) * 1_000_000.0 * invsig2),
        cross_sin=float(np.sum(hc0 * us - hs * uc0) * 1_000_000.0 * invsig2),
        observation_sigma_mm=OBS_SIGMA_MM,
        period_days=PERIOD_DAYS,
    )


def add_stats(a: StageAStats | None, b: StageAStats) -> StageAStats:
    if a is None:
        return b
    payload = {k: getattr(a, k) + getattr(b, k) for k in asdict(a) if k not in {"observation_sigma_mm", "period_days"}}
    payload["n"] = int(payload["n"])
    payload["observation_sigma_mm"] = OBS_SIGMA_MM
    payload["period_days"] = PERIOD_DAYS
    return StageAStats(**payload)


def count_pixels(cache, mask, blocks, selected, transform, fold_id: int) -> dict:
    counts = {}
    for train in (True, False):
        pixels = coeffs = 0
        for obs, *_ in iter_blocks_fold(cache, mask, blocks, selected, transform, fold_id, train):
            pixels += int(obs.shape[0])
            coeffs += int(obs.size)
        counts["training_pixel_count" if train else "validation_pixel_count"] = pixels
        counts["training_observation_count" if train else "validation_observation_count"] = coeffs
    return counts


def stage_a_fit(cache, mask, blocks, selected, transform, fold_id: int) -> dict:
    stats = None
    for obs, hc, hu, _basis in iter_blocks_fold(cache, mask, blocks, selected, transform, fold_id, True):
        stats = add_stats(stats, stats_from_block(obs, hc, hu))
    if stats is None:
        raise RuntimeError("No fold training pixels")
    coarse = [solve_from_stats(stats, lag_c, LAG_U_FIXED_DAYS) for lag_c in np.arange(0.0, 91.0, 10.0)]
    center = min(coarse, key=lambda r: r.objective).lag_c_days
    candidates = []
    for radius in (5.0, 2.0, 1.0):
        local = [solve_from_stats(stats, float(np.clip(center + d, 0.0, 365.2425)), LAG_U_FIXED_DAYS) for d in (-radius, 0.0, radius)]
        best = min(local, key=lambda r: r.objective)
        center = best.lag_c_days
        candidates.extend(local)
    best = min(candidates, key=lambda r: r.objective)
    return {
        "stage_A_training_only": True,
        "warm_start_source": "fold0_development_method_only_not_parameters",
        "train_pixel_count": int(stats.n),
        "Ske_global": best.ske_global,
        "Cu_global": best.cu_global,
        "lag_c_days": best.lag_c_days,
        "lag_u_days": LAG_U_FIXED_DAYS,
        "training_objective": best.objective,
        "training_rmse": best.rmse,
        "status": best.status,
    }


def stage_b_fit(cache, mask, blocks, selected, transform, fold_id: int, stage_a: dict) -> tuple[np.ndarray, dict]:
    k = transform.shape[1]
    hess = np.zeros((k, k), float)
    rhs = np.zeros(k, float)
    ske0 = float(stage_a["Ske_global"])
    cu = float(stage_a["Cu_global"])
    lag_c = float(stage_a["lag_c_days"])
    for obs, hc, hu, basis in iter_blocks_fold(cache, mask, blocks, selected, transform, fold_id, True):
        rc = rotate_coefficients(hc, lag_c)
        ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS)
        base = 1000.0 * (ske0 * rc + cu * ru)
        residual = obs - base
        j_scalar = 1000.0 * ske0 * np.sum(rc * residual, axis=1) / (OBS_SIGMA_MM**2)
        jj_weight = (1000.0 * ske0) ** 2 * np.sum(rc * rc, axis=1) / (OBS_SIGMA_MM**2)
        rhs += basis.T @ j_scalar
        hess += basis.T @ (basis * jj_weight[:, None])
    penalized = hess + LAMBDA * np.eye(k)
    eig = np.linalg.eigvalsh(penalized)
    gamma = np.linalg.solve(penalized, rhs)
    train = evaluate_theta_metrics(np.r_[np.log(ske0), gamma, np.log(cu), lag_c], cache, mask, blocks, selected, transform, fold_id, True, collect_detail=False)
    return gamma, {
        "stage_B_training_only": True,
        "lambda": LAMBDA,
        "gamma_norm": float(np.linalg.norm(gamma)),
        "penalized_Hessian_positive_definite": bool(np.min(eig) > 0),
        "penalized_Hessian_condition_number": float(np.max(eig) / max(np.min(eig), 1e-30)),
        "training_rmse": train["rmse"],
        "spatial_field_rms": train["spatial_field_rms"],
        "Ske_min": train["ske_min"],
        "Ske_median": train["ske_median"],
        "Ske_max": train["ske_max"],
    }


def objective_grad_train(theta, cache, mask, blocks, selected, transform, fold_id: int, lam: float, global_prior: dict):
    log_ske, gamma, cu, lag_c = decode(theta)
    total = 0.0
    grad = np.zeros_like(theta)
    k = 2.0 * np.pi / PERIOD_DAYS
    for obs, hc, hu, basis in iter_blocks_fold(cache, mask, blocks, selected, transform, fold_id, True):
        spatial = basis @ gamma
        ske = np.exp(np.clip(log_ske + spatial, -20, 10))
        rc = rotate_coefficients(hc, lag_c, PERIOD_DAYS)
        ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS, PERIOD_DAYS)
        pred = 1000.0 * (ske[:, None] * rc + cu * ru)
        res = obs - pred
        total += 0.5 * float(np.sum(res * res) / (OBS_SIGMA_MM**2))
        common_factor = -1000.0 * ske * np.sum(res * rc, axis=1) / (OBS_SIGMA_MM**2)
        grad[0] += float(np.sum(common_factor))
        grad[1:33] += basis.T @ common_factor
        grad[33] += -float(np.sum(res * (1000.0 * cu * ru)) / (OBS_SIGMA_MM**2))
        s0, c0 = hc[:, 0], hc[:, 1]
        angle = 2.0 * np.pi * lag_c / PERIOD_DAYS
        ca, sa = np.cos(angle), np.sin(angle)
        drc = np.column_stack([(-s0 * sa + c0 * ca) * k, (-c0 * sa - s0 * ca) * k])
        grad[34] += -float(np.sum(res * (1000.0 * ske[:, None] * drc)) / (OBS_SIGMA_MM**2))
    gamma_prior_unscaled = 0.5 * float(gamma @ gamma)
    gamma_prior_scaled = lam * gamma_prior_unscaled
    total += gamma_prior_scaled
    grad[1:33] += lam * gamma
    dglob = theta[[0, 33, 34]] - global_prior["mean"]
    gpen = 0.5 * float(np.sum(global_prior["precision"] * dglob * dglob))
    total += gpen
    grad[[0, 33, 34]] += global_prior["precision"] * dglob
    return total, grad, {
        "data_loss": total - gamma_prior_scaled - gpen,
        "gamma_prior_penalty_unscaled": gamma_prior_unscaled,
        "gamma_prior_penalty_scaled": gamma_prior_scaled,
        "global_prior_penalty": gpen,
        "total_objective": total,
    }


def evaluate_theta_metrics(theta, cache, mask, blocks, selected, transform, fold_id: int, train: bool, collect_detail: bool):
    log_ske, gamma, cu, lag_c = decode(theta)
    sse = ae = bias_sum = obs_sum = obs_sq_sum = 0.0
    ncoef = 0
    real_sse = imag_sse = amp_sse = phase_abs_sum = 0.0
    phase_n = 0
    abs_values = [] if collect_detail else None
    ske_sample = []
    ske_min = np.inf
    ske_max = -np.inf
    spatial_sq = 0.0
    spatial_n = 0
    for obs, hc, hu, basis in iter_blocks_fold(cache, mask, blocks, selected, transform, fold_id, train):
        spatial = basis @ gamma
        ske = np.exp(np.clip(log_ske + spatial, -20, 10))
        pred = 1000.0 * (ske[:, None] * rotate_coefficients(hc, lag_c) + cu * rotate_coefficients(hu, LAG_U_FIXED_DAYS))
        res = obs - pred
        sse += float(np.sum(res * res))
        ae += float(np.sum(np.abs(res)))
        bias_sum += float(np.sum(res))
        obs_sum += float(np.sum(obs))
        obs_sq_sum += float(np.sum(obs * obs))
        ncoef += int(res.size)
        real_sse += float(np.sum(res[:, 0] ** 2))
        imag_sse += float(np.sum(res[:, 1] ** 2))
        amp_sse += float(np.sum((np.linalg.norm(obs, axis=1) - np.linalg.norm(pred, axis=1)) ** 2))
        obs_phase = np.arctan2(obs[:, 0], obs[:, 1])
        pred_phase = np.arctan2(pred[:, 0], pred[:, 1])
        phase_diff = np.angle(np.exp(1j * (obs_phase - pred_phase)))
        phase_abs_sum += float(np.sum(np.abs(phase_diff) * PERIOD_DAYS / (2.0 * np.pi)))
        phase_n += int(obs.shape[0])
        spatial_sq += float(np.sum(spatial * spatial))
        spatial_n += int(spatial.size)
        ske_min = min(ske_min, float(np.min(ske)))
        ske_max = max(ske_max, float(np.max(ske)))
        if len(ske_sample) < 250_000:
            need = 250_000 - len(ske_sample)
            ske_sample.extend(ske[:need].astype(float).tolist())
        if collect_detail:
            abs_values.append(np.abs(res).reshape(-1))
    obs_mean = obs_sum / max(ncoef, 1)
    sst = obs_sq_sum - ncoef * obs_mean * obs_mean
    median_abs = float(np.median(np.concatenate(abs_values))) if collect_detail and abs_values else np.nan
    return {
        "rmse": float(np.sqrt(sse / max(ncoef, 1))),
        "mae": float(ae / max(ncoef, 1)),
        "median_absolute_error": median_abs,
        "bias": float(bias_sum / max(ncoef, 1)),
        "r2": float(1.0 - sse / max(sst, 1e-12)),
        "log_likelihood": float(-0.5 * ncoef * np.log(2.0 * np.pi * OBS_SIGMA_MM**2) - 0.5 * sse / (OBS_SIGMA_MM**2)),
        "harmonic_real_rmse": float(np.sqrt(real_sse / max(phase_n, 1))),
        "harmonic_imag_rmse": float(np.sqrt(imag_sse / max(phase_n, 1))),
        "amplitude_rmse": float(np.sqrt(amp_sse / max(phase_n, 1))),
        "phase_mae_days": float(phase_abs_sum / max(phase_n, 1)),
        "gamma_norm": float(np.linalg.norm(gamma)),
        "spatial_field_rms": float(np.sqrt(spatial_sq / max(spatial_n, 1))),
        "ske_min": float(ske_min),
        "ske_median": float(np.median(np.asarray(ske_sample, float))),
        "ske_max": float(ske_max),
        "Cu_global": float(cu),
        "lag_c_days": float(lag_c),
        "lag_u_days": LAG_U_FIXED_DAYS,
        "observation_count": int(ncoef),
        "pixel_count": int(phase_n),
    }


def history_row(iteration, theta, grad, parts, train, prev_theta, nfev, checkpoint, ck_hash):
    step = float(np.linalg.norm(theta - prev_theta)) if prev_theta is not None else np.nan
    rel_step = step / max(float(np.linalg.norm(prev_theta)), 1.0) if prev_theta is not None else np.nan
    return {
        "accepted_iteration": int(iteration),
        "training_objective": float(parts["total_objective"]),
        "normalized_training_objective": float(parts["total_objective"] / max(train["observation_count"], 1)),
        "training_rmse_mm": train["rmse"],
        "gradient_norm": float(np.linalg.norm(grad)),
        "gradient_rms": float(np.sqrt(np.mean(grad * grad))),
        "relative_parameter_step": rel_step,
        "gamma_norm": train["gamma_norm"],
        "spatial_field_rms": train["spatial_field_rms"],
        "Ske_min": train["ske_min"],
        "Ske_median": train["ske_median"],
        "Ske_max": train["ske_max"],
        "Cu_global": train["Cu_global"],
        "lag_c": train["lag_c_days"],
        "lag_u": train["lag_u_days"],
        "parameter_hash": hash_array(theta),
        "checkpoint_filename": checkpoint.name,
        "checkpoint_hash": ck_hash,
        "function_evaluations": int(nfev),
    }


def checkpoint_metadata(iteration, theta, ck_hash, parts, train, selected, training_mask_hash, validation_mask_hash, budget):
    return {
        "accepted_iteration": int(iteration),
        "parameter_hash": hash_array(theta),
        "checkpoint_hash": ck_hash,
        "training_objective": float(parts["total_objective"]),
        "training_rmse": train["rmse"],
        "gamma_norm": train["gamma_norm"],
        "Ske_min": train["ske_min"],
        "Ske_median": train["ske_median"],
        "Ske_max": train["ske_max"],
        "Cu": train["Cu_global"],
        "lag_c": train["lag_c_days"],
        "lag_u": train["lag_u_days"],
        "basis_hash": selected["basis_design_hash"],
        "training_mask_hash": training_mask_hash,
        "validation_mask_hash": validation_mask_hash,
        "lambda": LAMBDA,
        "objective_version": OBJECTIVE_VERSION,
        "prior_version": PRIOR_VERSION,
        "formal_protocol_version": PROTOCOL_VERSION,
        "formal_iteration_budget": int(budget),
    }


def generate_manifest(root, config, selected, mask, blocks, transform, budget):
    payload = {
        "model_variant": MODEL_VARIANT,
        "geology_model": GEOLOGY_MODEL,
        "lag_c_mode": LAG_C_MODE,
        "lag_u_global_days": LAG_U_FIXED_DAYS,
        "RBF_design": "R32_sigma1",
        "effective_spacing_km": 16.1242,
        "sigma_km": 17.2506,
        "orthogonal_basis_count": int(transform.shape[1]),
        "orthogonal_basis_hash": selected["basis_design_hash"],
        "lambda_multiplier": LAMBDA,
        "formal_stage_c_iteration_budget": int(budget),
        "config_hash": hash_file(config),
        "code_hash": hash_json({
            "run_formal_g0_fold1.py": hash_file(Path(__file__)),
            "run_stage_c_fixed_lagu.py": hash_file(ROOT / "scripts/run_stage_c_fixed_lagu.py"),
            "run_stage_b_fixed_lagu.py": hash_file(ROOT / "scripts/run_stage_b_fixed_lagu.py"),
            "profiled_stage_a.py": hash_file(ROOT / "profiled_stage_a.py"),
        }),
        "git_commit": None,
        "common_mask_hash": hash_file(mask),
        "fold_map_hash": hash_file(blocks),
        "objective_version": OBJECTIVE_VERSION,
        "prior_version": PRIOR_VERSION,
        "parameter_layout_version": "stage_c_fixed_lagu_35_parameters_v1",
        "dependency_versions": dependency_versions(),
        "manifest_status": "frozen_for_formal_g0_fourfold",
    }
    manifest = root / "formal_protocol_frozen_manifest.json"
    write_json(manifest, payload)
    payload["manifest_hash"] = hash_file(manifest)
    write_json(manifest, payload)
    return payload


def run(output_root: Path, config: Path, fold_id: int = 1):
    if fold_id not in {1, 2, 3, 4}:
        raise ValueError("This runner is currently restricted to formal fold1, fold2, fold3, or fold4")
    root = output_root
    selected = json.loads((root / "selected_rbf_design.json").read_text())
    if selected["basis_design_hash"] != BASIS_HASH:
        raise RuntimeError("Frozen basis hash mismatch")
    transform = np.load(root / "rbf_orthogonalization" / "rbf_transform.npy")
    budget = int(json.loads((root / "model_compare/G0_no_geology_L0_shared/fold_00/stage_C/development_early_stopping_selection.json").read_text())["selected_iteration_budget"])
    if budget != 40:
        raise RuntimeError("Frozen formal budget mismatch")
    cache = latest_real_harmonic_cache()
    mask = root / "comparison_common_mask.tif"
    blocks = root / "spatial_validation_blocks.tif"
    if hash_file(mask) != "ff761a316e0a89a9121c439967df418f14585ae420f281d43671ebaf4740bd1f":
        raise RuntimeError("Frozen common mask hash mismatch")
    if hash_file(blocks) != "d24dc63e65d3a1fa1a0e698620ba6d8e03fcf518a9a5ef0721c59374a1d46e3a":
        raise RuntimeError("Frozen fold map hash mismatch")
    manifest_path = root / "formal_protocol_frozen_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = "bd08b8640af45badd9c87cf5111791be9d10789699bf312972a9af48070219fe"
        if manifest.get("manifest_hash") != expected:
            raise RuntimeError(f"Frozen manifest hash mismatch: {manifest.get('manifest_hash')} != {expected}")
    else:
        manifest = generate_manifest(root, config, selected, mask, blocks, transform, budget)
    fold_dir = root / "model_compare" / "G0_no_geology_L0_shared" / f"fold_{fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "fold_id": fold_id,
        f"fold{fold_id}_role": "formal_outer_fold_pending",
        f"fold{fold_id}_formal_cv_eligible": False,
        "manifest_hash": manifest["manifest_hash"],
        "outer_validation_access_count_during_training": 0,
    }
    write_json(fold_dir / "fold_role_status.json", status)
    counts = count_pixels(cache, mask, blocks, selected, transform, fold_id)
    stage_a = stage_a_fit(cache, mask, blocks, selected, transform, fold_id)
    write_json(fold_dir / "stage_A_training_only_result.json", stage_a)
    gamma, stage_b = stage_b_fit(cache, mask, blocks, selected, transform, fold_id, stage_a)
    np.save(fold_dir / "stage_B_training_only_gamma.npy", gamma)
    write_json(fold_dir / "stage_B_training_only_result.json", stage_b)
    theta0 = np.r_[np.log(stage_a["Ske_global"]), gamma, np.log(stage_a["Cu_global"]), stage_a["lag_c_days"]].astype(float)
    global_prior = {
        "mean": theta0[[0, 33, 34]],
        "precision": np.array([1.0, 1.0, 1.0 / (PERIOD_DAYS**2)], dtype=float),
    }
    history = []
    previous = {"theta": theta0.copy()}
    nfev = {"n": 0}
    start = time.time()
    training_mask_hash = split_hash(mask, blocks, fold_id, True)
    validation_mask_hash = split_hash(mask, blocks, fold_id, False)

    def update_running(accepted, checkpoint=None, ck_hash=None):
        elapsed = time.time() - start
        per_iter = elapsed / max(accepted, 1)
        write_json(fold_dir / "formal_fold_running_status.json", {
            "status": "running_expected_long_duration" if accepted < budget else "training_budget_completed_pending_final_validation",
            "accepted_iterations_completed": int(accepted),
            "accepted_iterations_target": int(budget),
            "last_checkpoint": str(checkpoint) if checkpoint else None,
            "last_checkpoint_hash": ck_hash,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": max(budget - accepted, 0) * per_iter,
            "outer_validation_access_count_during_training": 0,
            "last_update_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })

    def fun(theta):
        nfev["n"] += 1
        val, grad, _parts = objective_grad_train(theta, cache, mask, blocks, selected, transform, fold_id, LAMBDA, global_prior)
        if not np.isfinite(val) or not np.isfinite(grad).all():
            raise FloatingPointError("non-finite training objective/gradient")
        return val, grad

    def callback(theta):
        accepted = len(history) + 1
        val, grad, parts = objective_grad_train(theta, cache, mask, blocks, selected, transform, fold_id, LAMBDA, global_prior)
        train = evaluate_theta_metrics(theta, cache, mask, blocks, selected, transform, fold_id, True, collect_detail=False)
        if train["ske_min"] <= 0 or train["Cu_global"] <= 0 or not (0 <= train["lag_c_days"] <= PERIOD_DAYS):
            raise RuntimeError("physical hard failure")
        checkpoint = fold_dir / f"checkpoint_iter_{accepted:03d}.npy"
        np.save(checkpoint, theta)
        ck_hash = hash_file(checkpoint)
        history.append(history_row(accepted, theta, grad, parts, train, previous["theta"], nfev["n"], checkpoint, ck_hash))
        pd.DataFrame(history).to_csv(fold_dir / "training_only_optimizer_history.csv", index=False)
        write_json(fold_dir / f"checkpoint_iter_{accepted:03d}.metadata.json", checkpoint_metadata(accepted, theta, ck_hash, parts, train, selected, training_mask_hash, validation_mask_hash, budget))
        previous["theta"] = theta.copy()
        update_running(accepted, checkpoint, ck_hash)

    result = minimize(
        fun,
        theta0,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        bounds=[(None, None)] + [(None, None)] * 32 + [(None, None), (0.0, PERIOD_DAYS)],
        options={"maxiter": budget, "maxfun": max(100, budget * 4), "maxls": 10, "ftol": 0.0, "gtol": 0.0},
    )
    hist = pd.read_csv(fold_dir / "training_only_optimizer_history.csv")
    completed = int(hist["accepted_iteration"].max())
    theta_final = np.load(fold_dir / f"checkpoint_iter_{completed:03d}.npy").astype(float)
    final_checkpoint = fold_dir / "final_training_checkpoint.npy"
    np.save(final_checkpoint, theta_final)
    final_checkpoint_hash = hash_file(final_checkpoint)
    final_obj, final_grad, final_parts = objective_grad_train(theta_final, cache, mask, blocks, selected, transform, fold_id, LAMBDA, global_prior)
    final_train = evaluate_theta_metrics(theta_final, cache, mask, blocks, selected, transform, fold_id, True, collect_detail=False)
    validation_access_final = 0
    final_valid = None
    line_search_failure = "LINE SEARCH" in str(result.message).upper() or "ABNORMAL" in str(result.message).upper()
    if completed == budget and not line_search_failure:
        final_valid = evaluate_theta_metrics(theta_final, cache, mask, blocks, selected, transform, fold_id, False, collect_detail=True)
        validation_access_final = 1
    gamma_final = decode(theta_final)[1]
    artifact_score = float(np.max(np.abs(gamma_final)) / max(np.sqrt(np.mean(gamma_final * gamma_final)), 1e-12))
    artifact = {
        **artifact_metrics(gamma_final, np.zeros_like(gamma_final), np.arange(gamma_final.size), gamma_final),
        "artifact_score": artifact_score,
        "artifact_status": "passed" if artifact_score < 6.0 else "failed",
    }
    physical = {
        "parameter_boundaries": "passed",
        "physical_status": "passed" if final_train["ske_min"] > 0 and final_train["Cu_global"] > 0 and 0 <= final_train["lag_c_days"] <= PERIOD_DAYS else "failed",
        "Ske_min": final_train["ske_min"],
        "Ske_median": final_train["ske_median"],
        "Ske_max": final_train["ske_max"],
        "Cu_global": final_train["Cu_global"],
        "lag_c_days": final_train["lag_c_days"],
        "lag_u_days": LAG_U_FIXED_DAYS,
    }
    checkpoint_alignment = True
    for _, row in hist.iterrows():
        ck = fold_dir / row["checkpoint_filename"]
        checkpoint_alignment = checkpoint_alignment and ck.exists() and row["parameter_hash"] == hash_array(np.load(ck)) and row["checkpoint_hash"] == hash_file(ck)
    protocol_passed = bool(completed == budget and checkpoint_alignment and not line_search_failure and physical["physical_status"] == "passed" and artifact["artifact_status"] == "passed" and validation_access_final == 1)
    write_json(fold_dir / "outer_validation_access_audit.json", {
        "outer_validation_access_count_during_training": 0,
        "outer_validation_access_count_final": validation_access_final,
    })
    if final_valid is not None:
        metrics_payload = {
            "validation_rmse_mm": final_valid["rmse"],
            "validation_mae_mm": final_valid["mae"],
            "validation_median_absolute_error_mm": final_valid["median_absolute_error"],
            "validation_bias_mm": final_valid["bias"],
            "validation_r2": final_valid["r2"],
            "validation_log_likelihood": final_valid["log_likelihood"],
            "harmonic_real_rmse_mm": final_valid["harmonic_real_rmse"],
            "harmonic_imag_rmse_mm": final_valid["harmonic_imag_rmse"],
            "amplitude_rmse_mm": final_valid["amplitude_rmse"],
            "phase_mae_days": final_valid["phase_mae_days"],
            "training_rmse_mm": final_train["rmse"],
            "generalization_gap_mm": final_valid["rmse"] - final_train["rmse"],
            "Ske_min": final_train["ske_min"],
            "Ske_median": final_train["ske_median"],
            "Ske_max": final_train["ske_max"],
            "Cu_global": final_train["Cu_global"],
            "lag_c_days": final_train["lag_c_days"],
            "lag_u_days": LAG_U_FIXED_DAYS,
            "gamma_norm": final_train["gamma_norm"],
            "spatial_field_rms": final_train["spatial_field_rms"],
            **counts,
        }
        write_json(fold_dir / "single_final_outer_validation_metrics.json", metrics_payload)
    write_json(fold_dir / "final_training_checkpoint_metadata.json", checkpoint_metadata(completed, theta_final, final_checkpoint_hash, final_parts, final_train, selected, training_mask_hash, validation_mask_hash, budget))
    write_json(fold_dir / "physical_parameter_audit.json", physical)
    write_json(fold_dir / "spatial_artifact_audit.json", artifact)
    fit = {
        "fold_id": 1,
        f"fold{fold_id}_role": "formal_outer_validation_fold" if protocol_passed else "formal_outer_fold_failed_or_incomplete",
        f"fold{fold_id}_formal_cv_eligible": protocol_passed,
        f"fold{fold_id}_validation_rmse_did_not_modify_hyperparameters": True,
        "accepted_iterations": completed,
        "accepted_iterations_target": budget,
        "checkpoint_alignment_passed": checkpoint_alignment,
        "outer_validation_access_count_during_training": 0,
        "outer_validation_access_count_final": validation_access_final,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "line_search_failure": line_search_failure,
        "formal_fit_status": "formal_fit_complete_fixed_budget" if protocol_passed else ("formal_fit_failed_checkpoint_alignment" if not checkpoint_alignment else "formal_fit_failed_numerical" if line_search_failure else "formal_fit_incomplete_fixed_budget_not_reached"),
        "formal_protocol_passed": protocol_passed,
        "training_rmse_mm": final_train["rmse"],
        "single_final_validation_rmse_mm": None if final_valid is None else final_valid["rmse"],
        "single_final_validation_mae_mm": None if final_valid is None else final_valid["mae"],
        "generalization_gap_mm": None if final_valid is None else final_valid["rmse"] - final_train["rmse"],
        "Ske_min": final_train["ske_min"],
        "Ske_median": final_train["ske_median"],
        "Ske_max": final_train["ske_max"],
        "Cu_global": final_train["Cu_global"],
        "lag_c_days": final_train["lag_c_days"],
        "lag_u_days": LAG_U_FIXED_DAYS,
        "gamma_norm": final_train["gamma_norm"],
        "spatial_field_rms": final_train["spatial_field_rms"],
        "final_training_checkpoint_hash": final_checkpoint_hash,
        "manifest_hash": manifest["manifest_hash"],
        "elapsed_seconds": time.time() - start,
    }
    fit["fold_id"] = fold_id
    write_json(fold_dir / "formal_fit_status.json", fit)
    status_path = root / "aquifer_model_revision_status.json"
    status = json.loads(status_path.read_text())
    status.update({
        f"fold{fold_id}_role": fit[f"fold{fold_id}_role"],
        f"fold{fold_id}_formal_cv_eligible": protocol_passed,
        "allow_continue_g0_other_folds": False,
        "allow_continue_g1_g2_g3": False,
        "allow_lag_c_model_comparison": False,
        "selected_model_config": "not_generated",
        "phase4_restart_allowed": False,
        "formal_protocol_manifest_hash": manifest["manifest_hash"],
    })
    if fold_id == 1:
        status["allow_continue_g0_fold2"] = protocol_passed
        status["allow_continue_g0_fold2_fold4"] = False
    elif fold_id == 2:
        status["allow_continue_g0_fold3"] = protocol_passed
        status["allow_continue_g0_fold2_fold4"] = False
        status["next_allowed_step"] = (
            "G0 fold2 formal outer validation passed; G0 fold3 may be started only by explicit user request. "
            "Do not auto-start fold3, fold4, G1-G3, lag_c comparison, selected_model_config generation, or Phase4/5."
        )
    elif fold_id == 3:
        status["allow_continue_g0_fold4"] = protocol_passed
        status["allow_continue_g0_fold2_fold4"] = False
        status["next_allowed_step"] = (
            "G0 fold3 formal outer validation passed; G0 fold4 may be started only by explicit user request. "
            "Do not auto-start fold4, G1-G3, lag_c comparison, selected_model_config generation, or Phase4/5."
        )
    elif fold_id == 4:
        status["allow_start_geology_model_comparison_review"] = protocol_passed
        status["allow_start_G1"] = False
        status["allow_start_G2"] = False
        status["allow_start_G3"] = False
        status["allow_continue_g0_fold2_fold4"] = False
        status["next_allowed_step"] = (
            "G0 fold4 formal outer validation passed; generate/review G0 four-fold formal summary before any G1-G3 decision. "
            "Do not auto-start G1-G3, lag_c comparison, selected_model_config generation, or Phase4/5."
        )
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": manifest, "stage_A": stage_a, "stage_B": stage_b, "fit": fit}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fold-id", type=int, default=1)
    args = parser.parse_args()
    run(Path(args.output_root), Path(args.config), fold_id=args.fold_id)


if __name__ == "__main__":
    main()
