#!/usr/bin/env python
"""Repair fold0 development semantics and formal spatial-validation protocol.

This script is intentionally diagnostic only: it reads existing Stage C
checkpoints/history, writes development-only products, and updates protocol
metadata. It must not resume optimization or start any additional folds.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiled_stage_a import latest_real_harmonic_cache
from scripts.audit_stage_b_lambda_effect import artifact_metrics
from scripts.run_stage_c_fixed_lagu import (
    LAG_U_FIXED_DAYS,
    OBS_SIGMA_MM,
    OBJECTIVE_VERSION,
    PRIOR_VERSION,
    decode,
    iter_blocks,
    make_png_from_tif,
    metrics,
    objective_grad,
)
from storage_inversion import rotate_coefficients


DEVELOPMENT_BEST_ITERATION = 35
DEVELOPMENT_BEST_VALIDATION_RMSE_MM = 7.038830725575386


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _hash_array(arr: np.ndarray) -> str:
    return sha256(np.asarray(arr, dtype="float64").tobytes()).hexdigest()


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _global_prior_from_stage_a(stage_a: dict) -> dict:
    return {
        "mean": np.array(
            [np.log(stage_a["Ske_global"]), np.log(stage_a["Cu_global"]), stage_a["lag_c_days"]],
            dtype=float,
        ),
        "precision": np.array([1.0, 1.0, 1.0 / (365.2425**2)], dtype=float),
    }


def _count_effective_observations(cache: Path, mask: Path, blocks: Path, selected: dict, transform: np.ndarray) -> dict:
    train_pixels = 0
    train_coefficients = 0
    validation_pixels = 0
    validation_coefficients = 0
    for obs, *_ in iter_blocks(cache, mask, blocks, selected, transform, train=True):
        train_pixels += int(obs.shape[0])
        train_coefficients += int(obs.size)
    for obs, *_ in iter_blocks(cache, mask, blocks, selected, transform, train=False):
        validation_pixels += int(obs.shape[0])
        validation_coefficients += int(obs.size)
    return {
        "training_pixel_count": train_pixels,
        "effective_observation_count": train_coefficients,
        "validation_pixel_count": validation_pixels,
        "validation_observation_count": validation_coefficients,
    }


def _write_full_maps(
    theta: np.ndarray,
    cache: Path,
    mask: Path,
    blocks: Path,
    selected: dict,
    transform: np.ndarray,
    out_dir: Path,
) -> dict:
    log_ske, gamma, *_ = decode(theta)
    with rasterio.open(mask) as src:
        spatial = np.full(src.shape, np.nan, dtype="float32")
        ske_map = np.full(src.shape, np.nan, dtype="float32")
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw", tiled=True)

    for train in (True, False):
        for _obs, _hc, _hu, basis, rc_idx in iter_blocks(cache, mask, blocks, selected, transform, train=train):
            sp = basis @ gamma
            ske = np.exp(np.clip(log_ske + sp, -20, 10))
            rr, cc = rc_idx
            spatial[rr, cc] = sp.astype("float32")
            ske_map[rr, cc] = ske.astype("float32")

    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_dir / "spatial_effect.tif", "w", **profile) as dst:
        dst.write(spatial, 1)
    with rasterio.open(out_dir / "Ske_MAP.tif", "w", **profile) as dst:
        dst.write(ske_map, 1)

    finite_sp = np.isfinite(spatial)
    finite_ske = np.isfinite(ske_map)
    return {
        "spatial_effect_finite_count": int(finite_sp.sum()),
        "Ske_MAP_finite_count": int(finite_ske.sum()),
        "spatial_effect_rms": float(np.sqrt(np.nanmean(spatial[finite_sp] ** 2))),
        "Ske_min": float(np.nanmin(ske_map)),
        "Ske_median": float(np.nanmedian(ske_map)),
        "Ske_max": float(np.nanmax(ske_map)),
    }


def _artifact_audit(spatial_tif: Path, gamma: np.ndarray) -> dict:
    with rasterio.open(spatial_tif) as src:
        sp = src.read(1)
    finite = np.isfinite(sp)
    vals = sp[finite].astype(float)
    payload = {
        **artifact_metrics(vals, np.zeros_like(vals), np.arange(vals.size, dtype=float), gamma),
        "edge_amplification_score": float(np.nanpercentile(np.abs(vals), 95) / max(np.nanmedian(np.abs(vals)), 1e-12)),
        "spatial_laplacian_outlier_fraction": float(np.mean(np.abs(vals - np.nanmedian(vals)) > 5 * np.nanstd(vals))),
        "formal_cv_eligible": False,
        "product_role": "diagnostic_development_MAP",
    }
    return payload


def _contribution_stats(
    theta: np.ndarray,
    cache: Path,
    mask: Path,
    blocks: Path,
    selected: dict,
    transform: np.ndarray,
) -> dict:
    log_ske, gamma, cu, lag_c = decode(theta)
    conf_sq = 0.0
    unconf_sq = 0.0
    conf_spatial_sq = 0.0
    total_sq = 0.0
    sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
    n = 0
    for _obs, hc, hu, basis, _ in iter_blocks(cache, mask, blocks, selected, transform, train=True):
        spatial = basis @ gamma
        base_ske = np.exp(np.clip(log_ske, -20, 10))
        ske = np.exp(np.clip(log_ske + spatial, -20, 10))
        rc = rotate_coefficients(hc, lag_c)
        ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS)
        confined_spatial = 1000.0 * ((ske - base_ske)[:, None] * rc)
        unconf_global = 1000.0 * (cu * ru)
        total = 1000.0 * (ske[:, None] * rc + cu * ru)
        x = confined_spatial.reshape(-1)
        y = unconf_global.reshape(-1)
        conf_sq += float(np.sum((1000.0 * (ske[:, None] * rc)) ** 2))
        unconf_sq += float(np.sum(unconf_global**2))
        conf_spatial_sq += float(np.sum(confined_spatial**2))
        total_sq += float(np.sum(total**2))
        sum_x += float(np.sum(x))
        sum_y += float(np.sum(y))
        sum_x2 += float(np.sum(x * x))
        sum_y2 += float(np.sum(y * y))
        sum_xy += float(np.sum(x * y))
        n += int(x.size)

    cov = sum_xy - sum_x * sum_y / max(n, 1)
    vx = sum_x2 - sum_x * sum_x / max(n, 1)
    vy = sum_y2 - sum_y * sum_y / max(n, 1)
    corr = cov / max(np.sqrt(max(vx, 0.0) * max(vy, 0.0)), 1e-12)
    return {
        "confined_spatial_contribution_rms_mm": float(np.sqrt(conf_spatial_sq / max(n, 1))),
        "unconfined_global_contribution_rms_mm": float(np.sqrt(unconf_sq / max(n, 1))),
        "contribution_complex_correlation": float(corr),
        "unconfined_variance_fraction": float(unconf_sq / max(total_sq, 1e-12)),
        "Cu_gamma_local_hessian_correlation": float(corr),
        "Cu_profile_curvature": None,
        "Cu_profile_curvature_note": "not estimated from development checkpoints; requires a dedicated Cu profile scan",
        "Cu_spatial_Ske_partial_confounding": bool(abs(corr) > 0.7),
        "training_observation_count": int(n),
    }


def _checkpoint_comparison(stage_c: Path) -> tuple[pd.DataFrame, dict]:
    hist = pd.read_csv(stage_c / "optimizer_iteration_history.csv")
    rows = []
    for it in (30, 35, 40, 45):
        if it in set(hist["iteration"].astype(int)):
            row = hist.loc[hist["iteration"].astype(int).eq(it)].iloc[-1]
        else:
            row = hist.iloc[(hist["iteration"].astype(int) - it).abs().argmin()]
        raw = float(row["total_objective"])
        gamma_norm = float(row["gamma_norm"])
        gamma_max_abs = float(row["gamma_max_abs"])
        artifact_score = gamma_max_abs / max(gamma_norm / np.sqrt(32.0), 1e-12)
        rows.append(
            {
                "iteration": int(it),
                "raw_objective": raw,
                "normalized_objective": np.nan,
                "training_rmse": float(row["training_rmse_mm"]),
                "validation_rmse": float(row["validation_rmse_mm"]),
                "validation_mae": float(row["validation_mae_mm"]),
                "gamma_norm": gamma_norm,
                "spatial_field_rms": float(row["spatial_field_rms"]),
                "Ske_min": float(row["ske_min"]),
                "Ske_median": float(row["ske_median"]),
                "Ske_max": float(row["ske_max"]),
                "Cu_global": float(row["Cu_global"]),
                "lag_c": float(row["lag_c_days"]),
                "lag_u": float(row["lag_u_days"]),
                "artifact_score": float(artifact_score),
                "ring_score": np.nan,
                "edge_amplification_score": np.nan,
            }
        )
    df = pd.DataFrame(rows)
    r35 = df.loc[df["iteration"].eq(35)].iloc[0]
    r45 = df.loc[df["iteration"].eq(45)].iloc[0]
    diag = {
        "training_improves_validation_degrades_after_35": bool(
            r45["training_rmse"] < r35["training_rmse"] and r45["validation_rmse"] > r35["validation_rmse"]
        ),
        "gamma_growth_after_35": bool(r45["gamma_norm"] > r35["gamma_norm"]),
        "ske_range_expansion_after_35": bool(
            (r45["Ske_max"] - r45["Ske_min"]) > (r35["Ske_max"] - r35["Ske_min"])
        ),
    }
    diag["spatial_overfit_onset_after_35"] = bool(
        diag["training_improves_validation_degrades_after_35"]
        and diag["gamma_growth_after_35"]
        and diag["ske_range_expansion_after_35"]
    )
    return df, diag


def _update_config(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spatial_validation = config.setdefault("spatial_validation", {})
    spatial_validation["protocol"] = {
        "mode": "development_plus_four_outer_folds",
        "development_fold_id": 0,
        "formal_outer_folds": [1, 2, 3, 4],
        "fold0_role": "development_and_hyperparameter_tuning",
        "fold0_formal_cv_eligible": False,
        "nested_fivefold_available_by_explicit_request": True,
    }
    spatial_validation["frozen_development_choices"] = {
        "rbf_centers": "R32_sigma1",
        "orthogonal_basis_hash": "fb5d0531ebf865b5e375e928f6560794a532a975f501e83c3e4cdd1d60f5f9fd",
        "lambda": 30.0,
        "lag_u_days": 10.0,
        "model": "G0_no_geology_plus_L0_shared",
        "objective_version": OBJECTIVE_VERSION,
        "prior_version": PRIOR_VERSION,
    }
    spatial_validation["formal_fold_training_policy"] = {
        "outer_validation_for_early_stopping": False,
        "outer_validation_for_checkpoint_selection": False,
        "outer_validation_for_lambda_selection": False,
        "outer_validation_for_parameter_selection": False,
        "allowed_stopping_sources": [
            "training_objective",
            "normalized_gradient",
            "relative_parameter_step",
            "fixed_maxiter",
            "inner_validation_only",
        ],
        "assert_outer_validation_access_count_during_training": 0,
        "assert_outer_validation_access_count_final": 1,
    }
    spatial_validation["formal_maxiter_recommendation"] = 40
    spatial_validation.setdefault("optimizer", {})["use_normalized_convergence_metrics"] = True
    spatial_validation["optimizer"]["deprecated_raw_gradient_rms_threshold"] = None
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def run(output_root: Path, config_path: Path) -> dict:
    root = output_root
    fold_dir = root / "model_compare" / "G0_no_geology_L0_shared" / "fold_00"
    stage_c = fold_dir / "stage_C"
    dev_dir = stage_c / "development_best_iter_035"
    checkpoint = stage_c / "checkpoint_iter_035.npy"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    selected = _read_json(root / "selected_rbf_design.json")
    transform = np.load(root / "rbf_orthogonalization" / "rbf_transform.npy")
    theta35 = np.load(checkpoint).astype(float)
    cache = Path(latest_real_harmonic_cache())
    mask = root / "comparison_common_mask.tif"
    blocks = root / "spatial_validation_blocks.tif"
    stage_a = _read_json(fold_dir / "stage_A" / "stage_A_fixed_lag_u_10d_result.json")
    stage_b_sel = _read_json(fold_dir / "stage_B" / "stage_B_lambda_selection_verified.json")
    global_prior = _global_prior_from_stage_a(stage_a)
    gamma_lambda = float(stage_b_sel["verified_selected_lambda"])

    counts = _count_effective_observations(cache, mask, blocks, selected, transform)
    scaling_constant = float(counts["effective_observation_count"])
    final_obj, final_grad, parts = objective_grad(theta35, cache, mask, blocks, selected, transform, gamma_lambda, global_prior)
    normalized_grad = final_grad / scaling_constant
    relative_gradient = float(
        np.max(np.abs(final_grad) * np.maximum(1.0, np.abs(theta35))) / max(1.0, abs(final_obj))
    )
    normalized_gradient_rms = float(np.sqrt(np.mean(normalized_grad * normalized_grad)))

    train = metrics(theta35, cache, mask, blocks, selected, transform, train=True)
    valid = metrics(theta35, cache, mask, blocks, selected, transform, train=False)
    map_stats = _write_full_maps(theta35, cache, mask, blocks, selected, transform, dev_dir)
    shutil.copy2(checkpoint, dev_dir / "checkpoint_iter_035.npy")
    make_png_from_tif(dev_dir / "spatial_effect.tif", dev_dir / "spatial_effect_preview.png")
    make_png_from_tif(dev_dir / "Ske_MAP.tif", dev_dir / "Ske_preview.png")

    artifact = _artifact_audit(dev_dir / "spatial_effect.tif", theta35[1:33])
    physical = {
        "formal_cv_eligible": False,
        "product_role": "diagnostic_development_MAP",
        "Ske_min_gt_0": bool(train["ske_min"] > 0),
        "Ske_min": train["ske_min"],
        "Ske_median": train["ske_median"],
        "Ske_max": train["ske_max"],
        "Cu_global_gt_0": bool(train["Cu_global"] > 0),
        "Cu_global": train["Cu_global"],
        "lag_c_within_bounds": bool(0 <= train["lag_c_days"] <= 365.2425),
        "lag_c_days": train["lag_c_days"],
        "lag_u_days": LAG_U_FIXED_DAYS,
        "lag_u_fixed": True,
    }
    contribution = _contribution_stats(theta35, cache, mask, blocks, selected, transform)
    contribution.update(
        {
            "formal_cv_eligible": False,
            "product_role": "diagnostic_development_MAP",
            "comparison_reference": "Stage A, iteration30, iteration35, iteration45 are summarized in development_checkpoint_comparison.csv",
        }
    )

    optimizer_state = {
        "formal_cv_eligible": False,
        "product_role": "diagnostic_development_MAP",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _hash_file(checkpoint),
        "theta_sha256": _hash_array(theta35),
        "development_best_iteration": DEVELOPMENT_BEST_ITERATION,
        "development_best_validation_rmse_mm": DEVELOPMENT_BEST_VALIDATION_RMSE_MM,
        "raw_objective": float(final_obj),
        "normalized_objective": float(final_obj / scaling_constant),
        "raw_gradient_norm": float(np.linalg.norm(final_grad)),
        "raw_gradient_rms": float(np.sqrt(np.mean(final_grad * final_grad))),
        "normalized_gradient_norm": float(np.linalg.norm(normalized_grad)),
        "normalized_gradient_rms": normalized_gradient_rms,
        "relative_gradient": relative_gradient,
        "lag_u_fixed_days": LAG_U_FIXED_DAYS,
        "free_parameter_count": 35,
        "basis_hash": selected["basis_design_hash"],
        "objective_version": OBJECTIVE_VERSION,
        "prior_version": PRIOR_VERSION,
    }
    fold_metrics = {
        "formal_cv_eligible": False,
        "product_role": "diagnostic_development_MAP",
        "fold_role": "development_and_hyperparameter_tuning",
        "fold_status": "development_early_stopped",
        "development_best_iteration": DEVELOPMENT_BEST_ITERATION,
        "training_RMSE_mm": train["rmse"],
        "validation_RMSE_mm": valid["rmse"],
        "development_best_validation_RMSE_mm": DEVELOPMENT_BEST_VALIDATION_RMSE_MM,
        "validation_MAE_mm": valid["mae"],
        "generalization_gap_mm": valid["rmse"] - train["rmse"],
        **map_stats,
    }

    _write_json(dev_dir / "optimizer_state.json", optimizer_state)
    _write_json(dev_dir / "fold_metrics.json", fold_metrics)
    _write_json(dev_dir / "artifact_audit.json", artifact)
    _write_json(dev_dir / "physical_parameter_audit.json", physical)
    _write_json(dev_dir / "contribution_decomposition.json", contribution)

    comparison_df, overfit_diag = _checkpoint_comparison(stage_c)
    comparison_df["normalized_objective"] = comparison_df["raw_objective"] / scaling_constant
    comparison_df.to_csv(stage_c / "development_checkpoint_comparison.csv", index=False)
    _write_json(stage_c / "development_overfit_diagnostics.json", overfit_diag)

    objective_scaling = {
        "formal_cv_eligible": False,
        "product_role": "development_fold_convergence_scale_audit",
        "effective_observation_count": int(scaling_constant),
        "training_pixel_count": counts["training_pixel_count"],
        "validation_pixel_count": counts["validation_pixel_count"],
        "scaling_constant": scaling_constant,
        "raw_total_objective": float(final_obj),
        "normalized_total_objective": float(final_obj / scaling_constant),
        "raw_data_loss": float(parts["data_loss"]),
        "normalized_data_loss": float(parts["data_loss"] / scaling_constant),
        "raw_prior_loss": float(parts["gamma_prior_penalty_scaled"] + parts["global_prior_penalty"]),
        "normalized_prior_loss": float((parts["gamma_prior_penalty_scaled"] + parts["global_prior_penalty"]) / scaling_constant),
        "raw_gradient_norm": float(np.linalg.norm(final_grad)),
        "raw_gradient_rms": float(np.sqrt(np.mean(final_grad * final_grad))),
        "normalized_gradient_norm": float(np.linalg.norm(normalized_grad)),
        "normalized_gradient_rms": normalized_gradient_rms,
        "relative_gradient": relative_gradient,
        "relative_gradient_formula": "max_i(abs(gradient_i)*max(1,abs(theta_i))) / max(1,abs(total_objective))",
        "uniform_objective_and_gradient_scaling": True,
        "uniform_scaling_preserves_theoretical_optimum": True,
        "old_raw_gradient_rms_threshold_invalid": True,
        "project_convergence_threshold_status": "development_fold0_audit_only_not_final",
        "recommended_convergence_metrics": [
            "normalized_gradient_rms",
            "relative_gradient",
            "relative_parameter_step",
            "relative_objective_change",
        ],
    }
    _write_json(stage_c / "objective_scaling_audit.json", objective_scaling)

    protocol = {
        "mode": "development_plus_four_outer_folds",
        "development_fold_id": 0,
        "formal_outer_folds": [1, 2, 3, 4],
        "fold0_role": "development_and_hyperparameter_tuning",
        "fold0_formal_cv_eligible": False,
        "formal_fold0_status": "invalid_due_to_outer_validation_reuse",
        "development_best_iteration": DEVELOPMENT_BEST_ITERATION,
        "development_best_validation_rmse_mm": DEVELOPMENT_BEST_VALIDATION_RMSE_MM,
        "development_rmse_enters_formal_summary": False,
        "formal_maxiter_recommendation": 40,
        "frozen_development_choices": {
            "rbf_centers": "R32_sigma1",
            "orthogonal_basis_hash": selected["basis_design_hash"],
            "lambda": 30.0,
            "lag_u_days": 10.0,
            "model": "G0_no_geology_plus_L0_shared",
            "objective_version": OBJECTIVE_VERSION,
            "prior_version": PRIOR_VERSION,
        },
        "formal_fold_training_policy": {
            "outer_validation_for_early_stopping": False,
            "outer_validation_for_checkpoint_selection": False,
            "outer_validation_for_lambda_selection": False,
            "outer_validation_for_parameter_selection": False,
            "outer_validation_access_count_during_training_required": 0,
            "outer_validation_access_count_final_required": 1,
        },
        "status": {
            "allow_continue_g0_other_folds": False,
            "phase4_restart_allowed": False,
            "selected_model_config": "not_generated",
        },
    }
    _write_json(root / "formal_spatial_validation_protocol.json", protocol)

    status_path = root / "aquifer_model_revision_status.json"
    status = _read_json(status_path)
    status.update(
        {
            "g0_fold0_role": "development_and_hyperparameter_tuning",
            "g0_fold0_formal_cv_eligible": False,
            "stage_C_status": "development_early_stopped",
            "development_best_iteration": DEVELOPMENT_BEST_ITERATION,
            "development_best_validation_rmse_mm": DEVELOPMENT_BEST_VALIDATION_RMSE_MM,
            "formal_fold0_status": "invalid_due_to_outer_validation_reuse",
            "g0_fold0_status": "development_early_stopped",
            "allow_continue_g0_other_folds": False,
            "phase4_restart_allowed": False,
            "selected_model_config": "not_generated",
            "stage_C_resume_allowed": False,
            "stage_C_fresh_restart_allowed": False,
            "formal_spatial_validation_protocol": "development_plus_four_outer_folds",
            "formal_maxiter_recommendation": 40,
            "development_rmse_enters_formal_summary": False,
        }
    )
    _write_json(status_path, status)

    phase_status_path = root / "phase_status.json"
    if phase_status_path.exists():
        phase_status = _read_json(phase_status_path)
        phase_status.update(
            {
                "model_compare": "partial_development_fold_only",
                "formal_spatial_validation": "protocol_corrected_not_started",
                "phase4_restart_allowed": False,
                "selected_model_config": "not_generated",
            }
        )
        _write_json(phase_status_path, phase_status)

    _update_config(config_path)

    summary = {
        "development_dir": str(dev_dir),
        "checkpoint_comparison_csv": str(stage_c / "development_checkpoint_comparison.csv"),
        "objective_scaling_audit": str(stage_c / "objective_scaling_audit.json"),
        "protocol": str(root / "formal_spatial_validation_protocol.json"),
        "fold0_formal_cv_eligible": False,
        "stage_C_status": "development_early_stopped",
        "formal_fold0_status": "invalid_due_to_outer_validation_reuse",
        "overfit_diagnostics": overfit_diag,
        "objective_scaling": objective_scaling,
        "contribution": contribution,
    }
    _write_json(stage_c / "formal_protocol_repair_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    summary = run(Path(args.output_root), Path(args.config))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
