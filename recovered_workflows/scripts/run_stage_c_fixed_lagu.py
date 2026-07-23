#!/usr/bin/env python
"""Stage C fold0 joint tuning with lag_u fixed at 10 days."""
from __future__ import annotations

import argparse
import json
import shutil
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiled_stage_a import latest_real_harmonic_cache
from scripts.audit_stage_b_lambda_effect import artifact_metrics
from scripts.run_stage_b_fixed_lagu import OBS_SIGMA_MM, rbf_values
from storage_inversion import rotate_coefficients


PERIOD_DAYS = 365.2425
LAG_U_FIXED_DAYS = 10.0
OBJECTIVE_VERSION = "stage_c_fixed_lagu_squared_loss_v1"
PRIOR_VERSION = "gamma_lambda30_global_weak_stageA_centered_v1"


def hash_array(arr):
    return sha256(np.asarray(arr, dtype="float64").tobytes()).hexdigest()


def hash_json(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def iter_blocks(cache_path, mask_path, block_path, selected, transform, fold_id=0, train=True):
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
            r = int(h5["block_row"][bi]); c = int(h5["block_col"][bi])
            h = int(h5["block_height"][bi]); w = int(h5["block_width"][bi])
            window = Window(c, r, w, h)
            flat = h5["flat_index"][start:start + count].astype(int)
            rows = flat // w; cols = flat % w
            mask = mask_src.read(1, window=window).ravel()[flat] == 1
            folds = block_src.read(1, window=window).ravel()[flat]
            take = folds != fold_id if train else folds == fold_id
            obs = h5["obs"][start:start + count]
            hc = h5["hc"][start:start + count]
            hu = h5["hu"][start:start + count]
            common = mask & take & np.isfinite(obs).all(1) & np.isfinite(hc).all(1) & np.isfinite(hu).all(1)
            if not common.any():
                continue
            rr = r + rows[common]; cc = c + cols[common]
            xs, ys = xy(mask_src.transform, rr, cc, offset="center")
            xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            phi = rbf_values(np.column_stack([xs, ys]), centers, sigma_m)
            basis = phi @ transform
            yield obs[common].astype(float), hc[common].astype(float), hu[common].astype(float), basis.astype(float), (rr, cc)


def decode(theta):
    return float(theta[0]), theta[1:33], float(np.exp(theta[33])), float(theta[34])


def objective_grad(theta, cache, mask, blocks, selected, transform, gamma_lambda, global_prior):
    log_ske, gamma, cu, lag_c = decode(theta)
    total = 0.0
    grad = np.zeros_like(theta)
    k = 2.0 * np.pi / PERIOD_DAYS
    ru_cache_lag = None
    for obs, hc, hu, b, _ in iter_blocks(cache, mask, blocks, selected, transform, train=True):
        spatial = b @ gamma
        ske = np.exp(np.clip(log_ske + spatial, -20, 10))
        rc = rotate_coefficients(hc, lag_c, PERIOD_DAYS)
        ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS, PERIOD_DAYS)
        pred = 1000.0 * (ske[:, None] * rc + cu * ru)
        res = obs - pred
        total += 0.5 * float(np.sum(res * res) / (OBS_SIGMA_MM**2))
        common_factor = -1000.0 * ske * np.sum(res * rc, axis=1) / (OBS_SIGMA_MM**2)
        grad[0] += float(np.sum(common_factor))
        grad[1:33] += b.T @ common_factor
        grad[33] += -float(np.sum(res * (1000.0 * cu * ru)) / (OBS_SIGMA_MM**2))
        s0, c0 = hc[:, 0], hc[:, 1]
        angle = 2.0 * np.pi * lag_c / PERIOD_DAYS
        ca, sa = np.cos(angle), np.sin(angle)
        drc = np.column_stack([(-s0 * sa + c0 * ca) * k, (-c0 * sa - s0 * ca) * k])
        grad[34] += -float(np.sum(res * (1000.0 * ske[:, None] * drc)) / (OBS_SIGMA_MM**2))
    gamma_prior_unscaled = 0.5 * float(gamma @ gamma)
    gamma_prior_scaled = gamma_lambda * gamma_prior_unscaled
    total += gamma_prior_scaled
    grad[1:33] += gamma_lambda * gamma
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


def metrics(theta, cache, mask, blocks, selected, transform, train=False, write_maps_dir=None):
    log_ske, gamma, cu, lag_c = decode(theta)
    sse = ae = 0.0
    ncoef = 0
    skes = []
    spatials = []
    with rasterio.open(mask) as src:
        arr_sp = np.full(src.shape, np.nan, dtype="float32") if write_maps_dir else None
        arr_ske = np.full(src.shape, np.nan, dtype="float32") if write_maps_dir else None
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw", tiled=True)
    for obs, hc, hu, b, rc_idx in iter_blocks(cache, mask, blocks, selected, transform, train=train):
        spatial = b @ gamma
        ske = np.exp(np.clip(log_ske + spatial, -20, 10))
        pred = 1000.0 * (ske[:, None] * rotate_coefficients(hc, lag_c) + cu * rotate_coefficients(hu, LAG_U_FIXED_DAYS))
        res = obs - pred
        sse += float(np.sum(res * res)); ae += float(np.sum(np.abs(res))); ncoef += int(res.size)
        skes.append(ske); spatials.append(spatial)
        if write_maps_dir:
            rr, cc = rc_idx
            arr_sp[rr, cc] = spatial.astype("float32")
            arr_ske[rr, cc] = ske.astype("float32")
    ske_all = np.concatenate(skes); sp_all = np.concatenate(spatials)
    out = {
        "rmse": float(np.sqrt(sse / max(ncoef, 1))),
        "mae": float(ae / max(ncoef, 1)),
        "gamma_norm": float(np.linalg.norm(gamma)),
        "gamma_max_abs": float(np.max(np.abs(gamma))),
        "spatial_field_rms": float(np.sqrt(np.mean(sp_all * sp_all))),
        "spatial_field_max_abs": float(np.max(np.abs(sp_all))),
        "ske_min": float(np.min(ske_all)),
        "ske_median": float(np.median(ske_all)),
        "ske_max": float(np.max(ske_all)),
        "ske_spatial_cv": float((np.max(ske_all) - np.min(ske_all)) / max(np.median(ske_all), 1e-12)),
        "Cu_global": cu,
        "lag_c_days": lag_c,
        "lag_u_days": LAG_U_FIXED_DAYS,
    }
    if write_maps_dir:
        write_maps_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(write_maps_dir / "spatial_effect.tif", "w", **profile) as dst:
            dst.write(arr_sp, 1)
        with rasterio.open(write_maps_dir / "Ske_MAP_fold0.tif", "w", **profile) as dst:
            dst.write(arr_ske, 1)
    return out


def project_convergence(rows):
    stable = 0
    for a, b in zip(rows[:-1], rows[1:]):
        rel_obj = abs(a["total_objective"] - b["total_objective"]) / max(abs(a["total_objective"]), 1.0)
        if rel_obj < 1e-8 and b["gradient_rms"] < 1e-5 and b["relative_parameter_step"] < 1e-6:
            stable += 1
        else:
            stable = 0
    return stable >= 3, stable


def make_png_from_tif(tif, png):
    try:
        import matplotlib.pyplot as plt
        with rasterio.open(tif) as src:
            arr = src.read(1)
        finite = np.isfinite(arr)
        vmin, vmax = np.nanpercentile(arr[finite], [2, 98]) if finite.any() else (0, 1)
        plt.figure(figsize=(6, 5))
        plt.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.axis("off"); plt.colorbar(shrink=0.75)
        plt.tight_layout()
        plt.savefig(png, dpi=180)
        plt.close()
    except Exception as exc:
        png.with_suffix(".error.txt").write_text(str(exc), encoding="utf-8")


def run(output_root, resume_from_checkpoint=False, maxiter_total=30):
    root = Path(output_root)
    fold_dir = root / "model_compare" / "G0_no_geology_L0_shared" / "fold_00"
    stage_c = fold_dir / "stage_C"
    stage_c.mkdir(parents=True, exist_ok=True)
    selected = json.loads((root / "selected_rbf_design.json").read_text())
    stage_a = json.loads((fold_dir / "stage_A" / "stage_A_fixed_lag_u_10d_result.json").read_text())
    stage_b_sel = json.loads((fold_dir / "stage_B" / "stage_B_lambda_selection_verified.json").read_text())
    gamma0 = np.load(fold_dir / "stage_B" / "selected_gamma.npy")
    transform = np.load(root / "rbf_orthogonalization" / "rbf_transform.npy")
    cache = latest_real_harmonic_cache()
    mask = root / "comparison_common_mask.tif"; blocks = root / "spatial_validation_blocks.tif"
    theta0 = np.r_[np.log(stage_a["Ske_global"]), gamma0, np.log(stage_a["Cu_global"]), stage_a["lag_c_days"]].astype(float)
    history_seed = []
    resume_archive = None
    if resume_from_checkpoint:
        checkpoint = stage_c / "checkpoint_iter_030.npy"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Required resume checkpoint is missing: {checkpoint}")
        previous_key = json.loads((stage_c / "stage_C_cache_key.json").read_text())["cache_key"]
        old_hist_path = stage_c / "optimizer_iteration_history.csv"
        if not old_hist_path.exists():
            raise FileNotFoundError("Cannot resume without existing optimizer_iteration_history.csv")
        history_seed = pd.read_csv(old_hist_path).to_dict(orient="records")
        if len(history_seed) < 30:
            raise RuntimeError("Resume requires the existing 30-iteration Stage C history")
        resume_archive = stage_c / "archive" / f"iteration_limit_30_before_resume_{int(time.time())}"
        resume_archive.mkdir(parents=True, exist_ok=True)
        for name in [
            "optimizer_result.json", "optimizer_iteration_history.csv", "fold_metrics.json",
            "stage_B_stage_C_comparison.json", "stage_C_spatial_artifact_audit.json",
            "stage_C_physical_parameter_audit.json", "spatial_effect.tif", "Ske_MAP_fold0.tif",
            "spatial_effect_preview.png", "Ske_preview.png", "stage_C_cache_key.json",
        ]:
            p = stage_c / name
            if p.exists():
                shutil.copy2(p, resume_archive / name)
        theta0 = np.load(checkpoint).astype(float)
    global_prior = {
        "mean": np.array([theta0[0], theta0[33], theta0[34]], float),
        "precision": np.array([1.0, 1.0, 1.0 / (365.2425**2)], float),
    }
    gamma_lambda = float(stage_b_sel["verified_selected_lambda"])
    cache_key = hash_json({
        "fold_id": 0,
        "orthogonal_basis_hash": selected["basis_design_hash"],
        "lambda": gamma_lambda,
        "stage_A_parameter_hash": hash_json(stage_a),
        "stage_B_gamma_hash": hash_array(gamma0),
        "lag_u_fixed_days": LAG_U_FIXED_DAYS,
        "objective_function_version": OBJECTIVE_VERSION,
        "prior_version": PRIOR_VERSION,
    })
    if resume_from_checkpoint and previous_key != cache_key:
        raise RuntimeError("Checkpoint cache key does not match current Stage C configuration")
    (stage_c / "stage_C_cache_key.json").write_text(json.dumps({"cache_key": cache_key}, indent=2), encoding="utf-8")
    history = list(history_seed)
    last_obj = float(history[-1]["total_objective"]) if history else None
    last = {"theta": theta0.copy(), "objective": last_obj}
    nfev = {"n": 0}

    def fun(theta):
        nfev["n"] += 1
        t_eval = time.time()
        print(f"stage_C_objective_eval_start n={nfev['n']}", flush=True)
        val, grad, parts = objective_grad(theta, cache, mask, blocks, selected, transform, gamma_lambda, global_prior)
        print(f"stage_C_objective_eval_done n={nfev['n']} objective={val:.6g} grad_rms={np.sqrt(np.mean(grad*grad)):.6g} elapsed_s={time.time()-t_eval:.1f}", flush=True)
        return val, grad

    def callback(theta):
        print(f"stage_C_callback_start iteration={len(history)}", flush=True)
        val, grad, parts = objective_grad(theta, cache, mask, blocks, selected, transform, gamma_lambda, global_prior)
        train = metrics(theta, cache, mask, blocks, selected, transform, train=True)
        next_iter = len(history)
        do_val = (next_iter % 5 == 0) or (next_iter >= int(maxiter_total) - 5)
        valid = metrics(theta, cache, mask, blocks, selected, transform, train=False) if do_val else {}
        step = float(np.linalg.norm(theta - last["theta"]))
        rel_step = step / max(float(np.linalg.norm(last["theta"])), 1.0)
        rel_obj = np.nan if last["objective"] is None else abs(last["objective"] - val) / max(abs(last["objective"]), 1.0)
        row = {
            "iteration": len(history), **parts,
            "relative_objective_change": rel_obj,
            "training_rmse_mm": train["rmse"],
            "validation_rmse_mm": valid.get("rmse", np.nan),
            "validation_mae_mm": valid.get("mae", np.nan),
            "generalization_gap_mm": valid.get("rmse", np.nan) - train["rmse"] if valid else np.nan,
            "gradient_norm": float(np.linalg.norm(grad)),
            "gradient_rms": float(np.sqrt(np.mean(grad * grad))),
            "relative_parameter_step": rel_step,
            "gamma_norm": train["gamma_norm"],
            "gamma_max_abs": train["gamma_max_abs"],
            "spatial_field_rms": train["spatial_field_rms"],
            "spatial_field_max_abs": train["spatial_field_max_abs"],
            "ske_min": train["ske_min"],
            "ske_median": train["ske_median"],
            "ske_max": train["ske_max"],
            "ske_spatial_cv": train["ske_spatial_cv"],
            "Cu_global": train["Cu_global"],
            "lag_c_days": train["lag_c_days"],
            "lag_u_days": LAG_U_FIXED_DAYS,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(stage_c / "optimizer_iteration_history.csv", index=False)
        if len(history) % 5 == 0:
            np.save(stage_c / f"checkpoint_iter_{len(history):03d}.npy", theta)
        last["theta"] = theta.copy(); last["objective"] = val
        print(f"stage_C_callback_done iteration={row['iteration']} train_rmse={row['training_rmse_mm']:.6f} val_rmse={row['validation_rmse_mm']} gamma_norm={row['gamma_norm']:.6f}", flush=True)

    initial, g0, p0 = objective_grad(theta0, cache, mask, blocks, selected, transform, gamma_lambda, global_prior)
    start = time.time()
    remaining_iter = max(0, int(maxiter_total) - len(history))
    if remaining_iter <= 0:
        raise RuntimeError("No remaining iterations requested for Stage C resume")
    res = minimize(fun, theta0, method="L-BFGS-B", jac=True, callback=callback,
                   bounds=[(None, None)] + [(None, None)] * 32 + [(None, None), (0.0, 365.2425)],
                   options={"maxiter": remaining_iter, "maxfun": max(80, remaining_iter * 3), "maxls": 5, "ftol": 1e-8, "gtol": 1e-5})
    theta = res.x
    final, gf, pf = objective_grad(theta, cache, mask, blocks, selected, transform, gamma_lambda, global_prior)
    # Ensure last five validation metrics exist by appending final validation row if needed.
    train = metrics(theta, cache, mask, blocks, selected, transform, train=True, write_maps_dir=stage_c)
    valid = metrics(theta, cache, mask, blocks, selected, transform, train=False)
    conv, stable = project_convergence(history)
    gamma_recent = [r["gamma_norm"] for r in history[-5:]]
    gamma_growth = len(gamma_recent) >= 5 and all(b > a for a, b in zip(gamma_recent[:-1], gamma_recent[1:]))
    stage_b_df = pd.read_csv(fold_dir / "stage_B" / "stage_B_rbf_regularization_sensitivity.csv")
    stage_b_rmse = float(stage_b_df.loc[stage_b_df["lambda"].eq(30.0), "validation_RMSE"].iloc[0])
    val_change = (valid["rmse"] - stage_b_rmse) / max(stage_b_rmse, 1e-12) * 100
    fold_status = "refit_complete" if res.success else ("refit_complete_project_convergence" if conv else "refit_failed")
    if val_change > 1.0:
        fold_status = "refit_overfit_warning"
    optimizer_payload = {
        "success": bool(res.success), "message": str(res.message), "nit": int(res.nit), "nfev": int(res.nfev),
        "resume_from_checkpoint": bool(resume_from_checkpoint),
        "resume_archive": str(resume_archive) if resume_archive else None,
        "resume_initial_iteration": int(len(history_seed)),
        "final_cumulative_iteration": int(len(history)),
        "initial_objective": float(initial), "final_objective": float(final),
        "initial_parts": p0, "final_parts": pf,
        "project_convergence": bool(conv), "project_stable_iteration_count": int(stable),
        "lag_u_fixed_due_to_weak_identifiability": True,
        "lag_u_fixed_days": LAG_U_FIXED_DAYS,
        "free_parameter_count": 35,
        "stage_A_parameter_hash": hash_json(stage_a),
        "stage_B_gamma_hash": hash_array(gamma0),
        "orthogonal_basis_hash": selected["basis_design_hash"],
        "lambda_selection_hash": hash_json(stage_b_sel),
        "stage_C_initial_parameter_hash": hash_array(theta0),
        "stage_C_final_parameter_hash": hash_array(theta),
    }
    (stage_c / "optimizer_result.json").write_text(json.dumps(optimizer_payload, indent=2), encoding="utf-8")
    comparison = {
        "stage_B_validation_RMSE_mm": stage_b_rmse,
        "stage_C_training_RMSE_mm": train["rmse"],
        "stage_C_validation_RMSE_mm": valid["rmse"],
        "stage_C_validation_MAE_mm": valid["mae"],
        "stage_C_generalization_gap_mm": valid["rmse"] - train["rmse"],
        "stage_C_gamma_norm": train["gamma_norm"],
        "stage_C_spatial_field_RMS": train["spatial_field_rms"],
        "stage_C_Ske_min": train["ske_min"],
        "stage_C_Ske_median": train["ske_median"],
        "stage_C_Ske_max": train["ske_max"],
        "stage_C_Cu_global": train["Cu_global"],
        "stage_C_lag_c_days": train["lag_c_days"],
        "validation_rmse_change_percent": val_change,
        "training_rmse_change_percent": np.nan,
        "gamma_norm_change_percent": (train["gamma_norm"] - float(np.linalg.norm(gamma0))) / max(float(np.linalg.norm(gamma0)), 1e-12) * 100,
        "Cu_change_percent": (train["Cu_global"] - stage_a["Cu_global"]) / stage_a["Cu_global"] * 100,
        "lag_c_change_days": train["lag_c_days"] - stage_a["lag_c_days"],
        "stage_C_generalization_degradation": bool(val_change > 1.0),
        "stage_C_overfit_warning": bool(val_change > 1.0 and train["rmse"] < stage_b_df["training_RMSE"].min()),
    }
    (stage_c / "stage_B_stage_C_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    # Artifact audit on final spatial map.
    with rasterio.open(stage_c / "spatial_effect.tif") as src:
        sp = src.read(1)
    finite = np.isfinite(sp)
    vals = sp[finite].astype(float)
    artifact = {
        **artifact_metrics(vals, np.zeros_like(vals), np.arange(vals.size, dtype=float), theta[1:33]),
        "edge_amplification_score": float(np.nanpercentile(np.abs(sp[finite]), 95) / max(np.nanmedian(np.abs(sp[finite])), 1e-12)),
        "spatial_laplacian_outlier_fraction": float(np.mean(np.abs(vals - np.nanmedian(vals)) > 5 * np.nanstd(vals))),
    }
    (stage_c / "stage_C_spatial_artifact_audit.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    physical = {
        "Ske_min_gt_0": bool(train["ske_min"] > 0), "Ske_min": train["ske_min"], "Ske_median": train["ske_median"], "Ske_max": train["ske_max"],
        "Cu_global_gt_0": bool(train["Cu_global"] > 0), "Cu_global": train["Cu_global"],
        "lag_c_within_bounds": bool(0 <= train["lag_c_days"] <= 365.2425), "lag_c_days": train["lag_c_days"],
        "lag_u_days": LAG_U_FIXED_DAYS, "lag_u_fixed": True,
    }
    (stage_c / "stage_C_physical_parameter_audit.json").write_text(json.dumps(physical, indent=2), encoding="utf-8")
    make_png_from_tif(stage_c / "spatial_effect.tif", stage_c / "spatial_effect_preview.png")
    make_png_from_tif(stage_c / "Ske_MAP_fold0.tif", stage_c / "Ske_preview.png")
    metrics_payload = {
        "fold_status": fold_status, "stage_C_status": "complete_validated_fold0" if fold_status in {"refit_complete", "refit_complete_project_convergence"} else fold_status,
        "training_RMSE_mm": train["rmse"], "validation_RMSE_mm": valid["rmse"], "validation_MAE_mm": valid["mae"],
        "generalization_gap_mm": valid["rmse"] - train["rmse"], "gamma_norm": train["gamma_norm"], "Ske_min": train["ske_min"], "Ske_median": train["ske_median"], "Ske_max": train["ske_max"],
        "Cu_global": train["Cu_global"], "lag_c_days": train["lag_c_days"], "lag_u_days": LAG_U_FIXED_DAYS,
        "elapsed_seconds": time.time() - start,
    }
    (stage_c / "fold_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    status_path = root / "aquifer_model_revision_status.json"
    status = json.loads(status_path.read_text())
    status.update({
        "g0_fold0_status": fold_status,
        "stage_C_status": metrics_payload["stage_C_status"],
        "geology_model_comparison": "partial",
        "selected_geology_model": None,
        "selected_model_config": "not_generated",
        "phase4_restart_allowed": False,
        "allow_continue_g0_other_folds": bool(fold_status in {"refit_complete", "refit_complete_project_convergence"}),
    })
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({"optimizer": optimizer_payload, "metrics": metrics_payload, "comparison": comparison}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--maxiter-total", type=int, default=30)
    args = parser.parse_args()
    run(args.output_root, resume_from_checkpoint=args.resume_from_checkpoint, maxiter_total=args.maxiter_total)


if __name__ == "__main__":
    main()
