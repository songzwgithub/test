#!/usr/bin/env python
"""Stage B fold0 sensitivity with fixed lag_u and orthogonal RBF gamma only."""
from __future__ import annotations

import argparse
import json
import sys
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
from storage_inversion import rotate_coefficients


OBS_SIGMA_MM = 5.0


def rbf_values(points: np.ndarray, centers: np.ndarray, sigma_m: float) -> np.ndarray:
    diff = points[:, None, :] - centers[None, :, :]
    return np.exp(-0.5 * np.sum(diff * diff, axis=2) / max(sigma_m * sigma_m, 1e-30))


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
            r = int(h5["block_row"][bi])
            c = int(h5["block_col"][bi])
            h = int(h5["block_height"][bi])
            w = int(h5["block_width"][bi])
            window = Window(c, r, w, h)
            flat = h5["flat_index"][start:start + count].astype(int)
            rows = flat // w
            cols = flat % w
            mask = mask_src.read(1, window=window).ravel()[flat] == 1
            folds = block_src.read(1, window=window).ravel()[flat]
            take = folds != fold_id if train else folds == fold_id
            obs = h5["obs"][start:start + count]
            hc = h5["hc"][start:start + count]
            hu = h5["hu"][start:start + count]
            common = mask & take & np.isfinite(obs).all(1) & np.isfinite(hc).all(1) & np.isfinite(hu).all(1)
            if not common.any():
                continue
            global_rows = r + rows[common]
            global_cols = c + cols[common]
            xs, ys = xy(mask_src.transform, global_rows, global_cols, offset="center")
            xs = np.asarray(xs, float)
            ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float)
                ys = np.asarray(ys, float)
            phi = rbf_values(np.column_stack([xs, ys]), centers, sigma_m)
            basis = phi @ transform
            yield obs[common].astype(float), hc[common].astype(float), hu[common].astype(float), basis.astype(float)


def accumulate_quadratic(cache, mask, blocks, selected, transform, globals_payload):
    k = transform.shape[1]
    hess = np.zeros((k, k), float)
    rhs = np.zeros(k, float)
    base_sse = 0.0
    n = 0
    ske0 = float(globals_payload["Ske_global"])
    cu = float(globals_payload["Cu_global"])
    lag_c = float(globals_payload["lag_c_days"])
    lag_u = float(globals_payload["lag_u_days"])
    for obs, hc, hu, b in iter_blocks(cache, mask, blocks, selected, transform, train=True):
        rc = rotate_coefficients(hc, lag_c)
        ru = rotate_coefficients(hu, lag_u)
        base = 1000.0 * (ske0 * rc + cu * ru)
        residual = obs - base
        j_scalar = 1000.0 * ske0 * np.sum(rc * residual, axis=1) / (OBS_SIGMA_MM**2)
        jj_weight = (1000.0 * ske0) ** 2 * np.sum(rc * rc, axis=1) / (OBS_SIGMA_MM**2)
        rhs += b.T @ j_scalar
        hess += b.T @ (b * jj_weight[:, None])
        base_sse += float(np.sum(residual * residual))
        n += int(obs.shape[0])
    return hess, rhs, base_sse, n


def evaluate_gamma(cache, mask, blocks, selected, transform, globals_payload, gamma, train=True):
    ske0 = float(globals_payload["Ske_global"])
    cu = float(globals_payload["Cu_global"])
    lag_c = float(globals_payload["lag_c_days"])
    lag_u = float(globals_payload["lag_u_days"])
    sse = 0.0
    ae = 0.0
    ncoef = 0
    ske_values = []
    spatial_values = []
    for obs, hc, hu, b in iter_blocks(cache, mask, blocks, selected, transform, train=train):
        spatial = b @ gamma
        ske = ske0 * np.exp(np.clip(spatial, -5, 5))
        rc = rotate_coefficients(hc, lag_c)
        ru = rotate_coefficients(hu, lag_u)
        pred = 1000.0 * (ske[:, None] * rc + cu * ru)
        res = obs - pred
        sse += float(np.sum(res * res))
        ae += float(np.sum(np.abs(res)))
        ncoef += int(res.size)
        ske_values.append(ske)
        spatial_values.append(spatial)
    ske_all = np.concatenate(ske_values)
    spatial_all = np.concatenate(spatial_values)
    rmse = float(np.sqrt(sse / max(ncoef, 1)))
    return {
        "rmse": rmse,
        "mae": float(ae / max(ncoef, 1)),
        "Ske_min": float(np.min(ske_all)),
        "Ske_median": float(np.median(ske_all)),
        "Ske_max": float(np.max(ske_all)),
        "spatial_field_rms": float(np.sqrt(np.mean(spatial_all * spatial_all))),
        "center_pattern_artifact_score": float(np.max(np.abs(gamma)) / max(np.sqrt(np.mean(gamma * gamma)), 1e-12)),
    }


def run_stage_b(output_root: Path, lambdas=(1, 3, 10, 30)):
    output_root = Path(output_root)
    fold_dir = output_root / "model_compare" / "G0_no_geology_L0_shared" / "fold_00"
    stage_b = fold_dir / "stage_B"
    stage_b.mkdir(parents=True, exist_ok=True)
    selected = json.loads((output_root / "selected_rbf_design.json").read_text())
    globals_payload = json.loads((fold_dir / "stage_A" / "stage_A_fixed_lag_u_10d_result.json").read_text())
    transform = np.load(output_root / "rbf_orthogonalization" / "rbf_transform.npy")
    cache = latest_real_harmonic_cache()
    mask = output_root / "comparison_common_mask.tif"
    blocks = output_root / "spatial_validation_blocks.tif"
    hess, rhs, _base_sse, _n = accumulate_quadratic(cache, mask, blocks, selected, transform, globals_payload)
    rows = []
    gammas = {}
    for lam in lambdas:
        penalized = hess + float(lam) * np.eye(hess.shape[0])
        eig = np.linalg.eigvalsh(penalized)
        positive = bool(np.min(eig) > 0)
        cond = float(np.max(eig) / max(np.min(eig), 1e-30))
        gamma = np.linalg.solve(penalized, rhs)
        train = evaluate_gamma(cache, mask, blocks, selected, transform, globals_payload, gamma, train=True)
        valid = evaluate_gamma(cache, mask, blocks, selected, transform, globals_payload, gamma, train=False)
        row = {
            "lambda": float(lam),
            "optimizer_status": "closed_form_penalized_gauss_newton_gamma_only",
            "converged": True,
            "training_RMSE": train["rmse"],
            "validation_RMSE": valid["rmse"],
            "generalization_gap": valid["rmse"] - train["rmse"],
            "gamma_norm": float(np.linalg.norm(gamma)),
            "spatial_field_RMS": train["spatial_field_rms"],
            "Ske_min": valid["Ske_min"],
            "Ske_median": valid["Ske_median"],
            "Ske_max": valid["Ske_max"],
            "penalized_Hessian_condition_number": cond,
            "penalized_Hessian_positive_definite": positive,
            "center_pattern_artifact_score": valid["center_pattern_artifact_score"],
            "lag_u_fixed_days": float(globals_payload["lag_u_days"]),
            "basis_design_hash": selected["basis_design_hash"],
        }
        rows.append(row)
        gammas[float(lam)] = gamma
    df = pd.DataFrame(rows)
    df.to_csv(stage_b / "stage_B_rbf_regularization_sensitivity.csv", index=False)
    valid = df[(df["converged"]) & (df["penalized_Hessian_positive_definite"]) & (df["penalized_Hessian_condition_number"] < 1e4) & (df["center_pattern_artifact_score"] < 6.0)]
    selected_lambda = None
    reason = "no lambda passed Stage B gates"
    if not valid.empty:
        best = float(valid["validation_RMSE"].min())
        near = valid[valid["validation_RMSE"] <= best * 1.005].sort_values("lambda", ascending=False)
        selected_lambda = float(near.iloc[0]["lambda"])
        reason = "minimum validation RMSE with 0.5 percent stronger-regularization tie rule"
        np.save(stage_b / "selected_gamma.npy", gammas[selected_lambda])
    selection = {
        "stage_B_status": "complete_lambda_selected" if selected_lambda is not None else "failed_no_valid_lambda",
        "selected_lambda": selected_lambda,
        "selection_reason": reason,
        "lag_u_fixed_days": float(globals_payload["lag_u_days"]),
        "stage_B_fixed_lag_u": True,
        "stage_B_free_parameters": ["orthogonal_ske_gamma_00_to_31"],
        "stage_B_parameter_layout_contains_free_lag_u": False,
        "basis_design_hash": selected["basis_design_hash"],
    }
    (stage_b / "stage_B_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selection, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    args = parser.parse_args()
    selection, rows = run_stage_b(Path(args.output_root))
    print(json.dumps({"selection": selection, "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
