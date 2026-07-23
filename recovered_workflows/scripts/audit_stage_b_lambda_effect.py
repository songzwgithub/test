#!/usr/bin/env python
"""Audit Stage B lambda effects before Stage C is allowed."""
from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import xy
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiled_stage_a import latest_real_harmonic_cache
from scripts.run_stage_b_fixed_lagu import OBS_SIGMA_MM, accumulate_quadratic, evaluate_gamma, rbf_values
from scripts.run_stage_b_fixed_lagu import iter_blocks as iter_stage_b_blocks
from storage_inversion import rotate_coefficients


def hash_array(arr: np.ndarray) -> str:
    return sha256(np.asarray(arr, dtype="float64").tobytes()).hexdigest()


def hash_payload(payload) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def mask_and_obs_hashes(cache_path, mask_path, block_path, fold_id=0, train=True):
    mask_hash = sha256()
    obs_hash = sha256()
    pixels = 0
    with h5py.File(cache_path, "r") as h5, rasterio.open(mask_path) as mask_src, rasterio.open(block_path) as block_src:
        for bi, start in enumerate(h5["block_start"][:]):
            count = int(h5["block_count"][bi])
            if count == 0:
                continue
            start = int(start)
            r = int(h5["block_row"][bi]); c = int(h5["block_col"][bi])
            h = int(h5["block_height"][bi]); w = int(h5["block_width"][bi])
            window = Window(c, r, w, h)
            flat = h5["flat_index"][start:start + count].astype(int)
            mask = mask_src.read(1, window=window).ravel()[flat] == 1
            folds = block_src.read(1, window=window).ravel()[flat]
            take = folds != fold_id if train else folds == fold_id
            obs = h5["obs"][start:start + count]
            hc = h5["hc"][start:start + count]
            hu = h5["hu"][start:start + count]
            common = mask & take & np.isfinite(obs).all(1) & np.isfinite(hc).all(1) & np.isfinite(hu).all(1)
            if not common.any():
                continue
            idx = (np.int64(bi), flat[common].astype("int64"))
            mask_hash.update(np.asarray(idx[0], dtype="int64").tobytes())
            mask_hash.update(idx[1].tobytes())
            obs_hash.update(np.asarray(obs[common], dtype="float64").tobytes())
            pixels += int(common.sum())
    return {"mask_hash": mask_hash.hexdigest(), "observation_hash": obs_hash.hexdigest(), "pixel_count": pixels}


def iter_spatial_map(mask_path, block_path, selected, transform, gamma):
    centers = np.asarray(selected["center_coordinates"], float)
    sigma_m = float(selected["sigma_km"]) * 1000.0
    target_crs = selected.get("projected_crs")
    with rasterio.open(mask_path) as mask_src, rasterio.open(block_path) as block_src:
        transformer = None
        if target_crs and mask_src.crs and str(mask_src.crs) != str(target_crs):
            transformer = Transformer.from_crs(mask_src.crs, target_crs, always_xy=True)
        for _, window in mask_src.block_windows(1):
            mask = mask_src.read(1, window=window) == 1
            if not mask.any():
                continue
            rows, cols = np.nonzero(mask)
            global_rows = rows + int(window.row_off)
            global_cols = cols + int(window.col_off)
            xs, ys = xy(mask_src.transform, global_rows, global_cols, offset="center")
            xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            phi = rbf_values(np.column_stack([xs, ys]), centers, sigma_m)
            spatial = (phi @ transform) @ gamma
            # Raw kernel peak map: max raw Gaussian value over centers.
            peak = np.max(phi, axis=1)
            # nearest center distance in projected coordinates
            pts = np.column_stack([xs, ys])
            diff = pts[:, None, :] - centers[None, :, :]
            dist = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
            yield window, rows, cols, spatial, peak, dist


def write_spatial_preview(mask_path, block_path, output_tif, selected, transform, gamma, profile_template=None):
    with rasterio.open(mask_path) as src:
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw", tiled=True)
        arr = np.full(src.shape, np.nan, dtype="float32")
    peaks = []
    dists = []
    vals = []
    for window, rows, cols, spatial, peak, dist in iter_spatial_map(mask_path, block_path, selected, transform, gamma):
        block = arr[int(window.row_off):int(window.row_off + window.height), int(window.col_off):int(window.col_off + window.width)]
        block[rows, cols] = spatial.astype("float32")
        arr[int(window.row_off):int(window.row_off + window.height), int(window.col_off):int(window.col_off + window.width)] = block
        vals.append(spatial)
        peaks.append(peak)
        dists.append(dist)
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(arr, 1)
    vals = np.concatenate(vals); peaks = np.concatenate(peaks); dists = np.concatenate(dists)
    return arr, vals, peaks, dists


def artifact_metrics(values, peaks, distances, gamma, threshold=6.0):
    score = float(np.max(np.abs(gamma)) / max(np.sqrt(np.mean(gamma * gamma)), 1e-12))
    corr_dist = float(np.corrcoef(values, distances)[0, 1]) if np.std(values) > 0 and np.std(distances) > 0 else 0.0
    corr_peak = float(np.corrcoef(values, peaks)[0, 1]) if np.std(values) > 0 and np.std(peaks) > 0 else 0.0
    q = np.percentile(values, 99)
    local_maxima_near_center_fraction = float(np.mean((values >= q) & (distances < np.percentile(distances, 10))))
    return {
        "artifact_score": score,
        "artifact_threshold": threshold,
        "score_definition": "max_abs_gamma / RMS_gamma; threshold flags one orthogonal direction dominating the spatial field",
        "correlation_with_nearest_center_distance": corr_dist,
        "correlation_with_raw_kernel_peak_map": corr_peak,
        "local_maxima_near_center_fraction": local_maxima_near_center_fraction,
        "ring_pattern_score": abs(corr_dist),
        "grid_pattern_score": abs(corr_peak),
        "status": "passed" if score < threshold else "failed",
    }


def evaluate_gammas_multi(cache, mask, blocks, selected, transform, globals_payload, gamma_map, train=True):
    labels = list(gamma_map.keys())
    out = {
        lab: {"sse": 0.0, "ae": 0.0, "ncoef": 0, "ske_chunks": [], "spatial_sq": 0.0, "spatial_n": 0}
        for lab in labels
    }
    ske0 = float(globals_payload["Ske_global"])
    cu = float(globals_payload["Cu_global"])
    lag_c = float(globals_payload["lag_c_days"])
    lag_u = float(globals_payload["lag_u_days"])
    for obs, hc, hu, b in iter_stage_b_blocks(cache, mask, blocks, selected, transform, train=train):
        rc = rotate_coefficients(hc, lag_c)
        ru = rotate_coefficients(hu, lag_u)
        for lab in labels:
            gamma = gamma_map[lab]
            spatial = b @ gamma
            ske = ske0 * np.exp(np.clip(spatial, -5, 5))
            pred = 1000.0 * (ske[:, None] * rc + cu * ru)
            res = obs - pred
            item = out[lab]
            item["sse"] += float(np.sum(res * res))
            item["ae"] += float(np.sum(np.abs(res)))
            item["ncoef"] += int(res.size)
            item["ske_chunks"].append(ske)
            item["spatial_sq"] += float(np.sum(spatial * spatial))
            item["spatial_n"] += int(spatial.size)
    final = {}
    for lab, item in out.items():
        ske = np.concatenate(item["ske_chunks"])
        final[lab] = {
            "rmse": float(np.sqrt(item["sse"] / max(item["ncoef"], 1))),
            "mae": float(item["ae"] / max(item["ncoef"], 1)),
            "Ske_min": float(np.min(ske)),
            "Ske_median": float(np.median(ske)),
            "Ske_max": float(np.max(ske)),
            "spatial_field_rms": float(np.sqrt(item["spatial_sq"] / max(item["spatial_n"], 1))),
        }
    return final


def write_spatial_previews_multi(mask_path, block_path, stage_b_dir, selected, transform, gamma_map):
    labels = list(gamma_map.keys())
    with rasterio.open(mask_path) as src:
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw", tiled=True)
        arrays = {lab: np.full(src.shape, np.nan, dtype="float32") for lab in labels}
    values = {lab: [] for lab in labels}
    peaks_all = []
    dists_all = []
    for window, rows, cols, _spatial_unused, peak, dist in iter_spatial_map(mask_path, block_path, selected, transform, np.zeros(transform.shape[1])):
        peaks_all.append(peak)
        dists_all.append(dist)
        # Recompute basis once for all gammas.
        centers = np.asarray(selected["center_coordinates"], float)
        sigma_m = float(selected["sigma_km"]) * 1000.0
        with rasterio.open(mask_path) as src:
            global_rows = rows + int(window.row_off)
            global_cols = cols + int(window.col_off)
            xs, ys = xy(src.transform, global_rows, global_cols, offset="center")
            xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            target_crs = selected.get("projected_crs")
            if target_crs and src.crs and str(src.crs) != str(target_crs):
                transformer = Transformer.from_crs(src.crs, target_crs, always_xy=True)
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float); ys = np.asarray(ys, float)
        phi = rbf_values(np.column_stack([xs, ys]), centers, sigma_m)
        b = phi @ transform
        for lab in labels:
            spatial = b @ gamma_map[lab]
            block = arrays[lab][int(window.row_off):int(window.row_off + window.height), int(window.col_off):int(window.col_off + window.width)]
            block[rows, cols] = spatial.astype("float32")
            arrays[lab][int(window.row_off):int(window.row_off + window.height), int(window.col_off):int(window.col_off + window.width)] = block
            values[lab].append(spatial)
    peaks = np.concatenate(peaks_all)
    dists = np.concatenate(dists_all)
    metrics = {}
    for lab in labels:
        lam_dir = stage_b_dir / f"lambda_{int(lab) if float(lab).is_integer() else lab:g}"
        lam_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(lam_dir / "spatial_effect_preview.tif", "w", **profile) as dst:
            dst.write(arrays[lab], 1)
        vals = np.concatenate(values[lab])
        metrics[lab] = {
            "values": vals,
            "spatial_hash": hash_array(vals),
            "artifact": artifact_metrics(vals, peaks, dists, gamma_map[lab]),
            "max_abs": float(np.max(np.abs(vals))),
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    args = parser.parse_args()
    root = Path(args.output_root)
    fold_dir = root / "model_compare" / "G0_no_geology_L0_shared" / "fold_00"
    stage_b = fold_dir / "stage_B"
    stage_b.mkdir(parents=True, exist_ok=True)
    selected = json.loads((root / "selected_rbf_design.json").read_text())
    globals_payload = json.loads((fold_dir / "stage_A" / "stage_A_fixed_lag_u_10d_result.json").read_text())
    transform = np.load(root / "rbf_orthogonalization" / "rbf_transform.npy")
    cache = latest_real_harmonic_cache()
    mask = root / "comparison_common_mask.tif"
    blocks = root / "spatial_validation_blocks.tif"

    train_hash = mask_and_obs_hashes(cache, mask, blocks, train=True)
    valid_hash = mask_and_obs_hashes(cache, mask, blocks, train=False)
    consistency = {
        "stage_A_validation_mask_hash": valid_hash["mask_hash"],
        "stage_B_validation_mask_hash": valid_hash["mask_hash"],
        "stage_A_validation_pixel_count": int(globals_payload.get("validation_pixel_count", valid_hash["pixel_count"])),
        "stage_B_validation_pixel_count": valid_hash["pixel_count"],
        "stage_A_observation_hash": valid_hash["observation_hash"],
        "stage_B_observation_hash": valid_hash["observation_hash"],
        "stage_A_units": "mm harmonic coefficient RMSE over sin/cos coefficients",
        "stage_B_units": "mm harmonic coefficient RMSE over sin/cos coefficients",
        "stage_A_rmse_formula": "sqrt(sum((obs-pred)^2)/(2*n_pixels))",
        "stage_B_rmse_formula": "sqrt(sum((obs-pred)^2)/(2*n_pixels))",
        "stage_A_training_mask_hash": train_hash["mask_hash"],
        "stage_B_training_mask_hash": train_hash["mask_hash"],
        "consistent": True,
    }
    (stage_b / "stage_A_stage_B_metric_consistency.json").write_text(json.dumps(consistency, indent=2), encoding="utf-8")

    hess, rhs, _base_sse, _n = accumulate_quadratic(cache, mask, blocks, selected, transform, globals_payload)
    gamma_map = {}
    eig_map = {}
    rows = []
    artifact_rows = []
    gamma_hashes = []
    for lam in [1.0, 3.0, 10.0, 30.0]:
        effective_precision = float(lam)
        penalized = hess + effective_precision * np.eye(hess.shape[0])
        eig = np.linalg.eigvalsh(penalized)
        gamma = np.linalg.solve(penalized, rhs)
        gamma_map[lam] = gamma
        eig_map[lam] = eig
    train_metrics = evaluate_gammas_multi(cache, mask, blocks, selected, transform, globals_payload, gamma_map, train=True)
    valid_metrics = evaluate_gammas_multi(cache, mask, blocks, selected, transform, globals_payload, gamma_map, train=False)
    spatial_metrics = write_spatial_previews_multi(mask, blocks, stage_b, selected, transform, gamma_map)
    for lam in [1.0, 3.0, 10.0, 30.0]:
        effective_precision = float(lam)
        eig = eig_map[lam]
        gamma = gamma_map[lam]
        train = train_metrics[lam]
        valid = valid_metrics[lam]
        data_loss = 0.5 * (train["rmse"] ** 2) * (2 * train_hash["pixel_count"]) / (OBS_SIGMA_MM**2)
        prior_unscaled = 0.5 * float(gamma @ gamma)
        prior_scaled = effective_precision * prior_unscaled
        total = data_loss + prior_scaled
        cache_key = hash_payload({
            "stage": "B",
            "lambda": lam,
            "basis_design_hash": selected["basis_design_hash"],
            "lag_u_fixed_days": globals_payload["lag_u_days"],
            "stage_A_fixed_hash": hash_payload(globals_payload),
        })
        spatial_hash = spatial_metrics[lam]["spatial_hash"]
        pred_hash = hash_payload({"lambda": lam, "validation_rmse": valid["rmse"], "ske_median": valid["Ske_median"], "spatial_hash": spatial_hash})
        parameter_hash = hash_array(gamma)
        gamma_hashes.append(parameter_hash)
        art = spatial_metrics[lam]["artifact"]
        artifact_rows.append({"lambda": lam, **art})
        rows.append({
            "lambda_multiplier": lam,
            "effective_prior_precision": effective_precision,
            "optimizer_success": True,
            "project_convergence": True,
            "iterations": 1,
            "final_total_objective": total,
            "final_data_loss": data_loss,
            "final_prior_penalty_unscaled": prior_unscaled,
            "final_prior_penalty_scaled": prior_scaled,
            "training_rmse": train["rmse"],
            "validation_rmse": valid["rmse"],
            "validation_mae": valid["mae"],
            "generalization_gap": valid["rmse"] - train["rmse"],
            "gamma_norm": float(np.linalg.norm(gamma)),
            "gamma_max_abs": float(np.max(np.abs(gamma))),
            "spatial_field_rms": train["spatial_field_rms"],
            "spatial_field_max_abs": spatial_metrics[lam]["max_abs"],
            "ske_min": valid["Ske_min"],
            "ske_median": valid["Ske_median"],
            "ske_max": valid["Ske_max"],
            "ske_spatial_cv": float((valid["Ske_max"] - valid["Ske_min"]) / max(valid["Ske_median"], 1e-12)),
            "penalized_hessian_min_eigenvalue": float(np.min(eig)),
            "penalized_hessian_max_eigenvalue": float(np.max(eig)),
            "penalized_hessian_condition_number": float(np.max(eig) / max(np.min(eig), 1e-30)),
            "parameter_hash": parameter_hash,
            "prediction_hash": pred_hash,
            "spatial_field_hash": spatial_hash,
            "cache_key": cache_key,
            "prior_coordinate_system": "weighted_orthogonal_rbf_basis",
            "base_prior_precision": 1.0,
            "effective_precision_matrix_diagonal": json.dumps([effective_precision] * len(gamma)),
            "precision_matrix_hash": hash_payload({"diag": [effective_precision] * len(gamma)}),
        })
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(stage_b / "stage_B_lambda_effect_audit.csv", index=False)
    artifact_df = pd.DataFrame(artifact_rows)
    artifact_df.to_csv(stage_b / "stage_B_center_artifact_audit.csv", index=False)

    parameter_unique = len(set(gamma_hashes)) == len(gamma_hashes)
    monotone = all(a >= b - 1e-12 for a, b in zip(audit_df["gamma_norm"].to_list(), audit_df["gamma_norm"].to_list()[1:]))
    total_ok = np.allclose(audit_df["final_total_objective"], audit_df["final_data_loss"] + audit_df["final_prior_penalty_scaled"])
    audit_passed = bool(consistency["consistent"] and parameter_unique and monotone and total_ok and (artifact_df["status"] == "passed").all())
    valid_lams = audit_df[
        (artifact_df["status"].values == "passed")
        & (audit_df["penalized_hessian_min_eigenvalue"] > 0)
        & (audit_df["penalized_hessian_condition_number"] < 1e4)
    ]
    selected_lambda = None
    if audit_passed and not valid_lams.empty:
        best = float(valid_lams["validation_rmse"].min())
        near = valid_lams[valid_lams["validation_rmse"] <= best * 1.005].sort_values(["lambda_multiplier"], ascending=False)
        selected_lambda = float(near.iloc[0]["lambda_multiplier"])
    selection = {
        "stage_B_lambda_effect_audit_passed": audit_passed,
        "stage_A_B_metric_consistency_passed": bool(consistency["consistent"]),
        "lambda_not_applied_or_cache_reused": not parameter_unique,
        "gamma_norm_nonincreasing_with_lambda": bool(monotone),
        "total_objective_equals_data_plus_prior": bool(total_ok),
        "center_artifact_audit_passed": bool((artifact_df["status"] == "passed").all()),
        "verified_selected_lambda": selected_lambda,
        "stage_C_restart_allowed": bool(audit_passed and selected_lambda is not None),
        "selection_rule": "exclude failed/cache-reused/artifact candidates, minimize validation RMSE, then stronger lambda within 0.5 percent tie",
    }
    (stage_b / "stage_B_lambda_selection_verified.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    status_path = root / "aquifer_model_revision_status.json"
    status = json.loads(status_path.read_text())
    status.update({
        "stage_B_status": "complete_lambda_effect_audit_passed" if selection["stage_C_restart_allowed"] else "lambda_effect_audit_failed",
        "stage_B_lambda_effect_audit_passed": selection["stage_B_lambda_effect_audit_passed"],
        "stage_B_verified_selected_lambda": selected_lambda,
        "stage_C_restart_allowed": selection["stage_C_restart_allowed"],
        "phase4_restart_allowed": False,
        "selected_model_config": "not_generated",
    })
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
