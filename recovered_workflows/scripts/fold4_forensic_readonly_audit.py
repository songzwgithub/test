#!/usr/bin/env python
"""Read-only forensic audit for the fold4 extreme validation error.

This script deliberately does not iterate over the validation fold or recompute
predictions. It only inspects artifacts already written by the single final
outer-validation pass.
"""
from __future__ import annotations

import csv
import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np


ROOT = Path("outputs/aquifer_model_revision")
FOLD = ROOT / "model_compare/G0_no_geology_L0_shared/fold_04"
REQUIRED_FORENSIC_PRODUCTS = {
    "observations": FOLD / "single_final_validation_observations.npy",
    "predictions": FOLD / "single_final_validation_predictions.npy",
    "residuals": FOLD / "single_final_validation_residuals.npy",
    "pixel_indices": FOLD / "single_final_validation_pixel_indices.npy",
    "coordinates": FOLD / "single_final_validation_coordinates.npy",
    "source_block_ids": FOLD / "single_final_validation_source_block_ids.npy",
    "dataset_metadata": FOLD / "single_final_validation_dataset_metadata.json",
}
RATIO_CANDIDATES = [1000.0, 100.0, 365.25, 2.0 * math.pi, 0.05546576 / (4.0 * math.pi), -1.0]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def quantiles(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    labels = {
        "p0.1": 0.1,
        "p1": 1.0,
        "p5": 5.0,
        "p25": 25.0,
        "median": 50.0,
        "p75": 75.0,
        "p95": 95.0,
        "p99": 99.0,
        "p99.9": 99.9,
    }
    if finite.size == 0:
        return {
            "count": int(arr.size),
            "finite_count": 0,
            "min": None,
            **{key: None for key in labels},
            "max": None,
            "mean": None,
            "std": None,
            "RMS": None,
            "MAE": None,
        }
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        **{key: float(np.percentile(finite, pct)) for key, pct in labels.items()},
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "RMS": float(np.sqrt(np.mean(finite * finite))),
        "MAE": float(np.mean(np.abs(finite))),
    }


def detect_fixed_ratio(observation: np.ndarray, prediction: np.ndarray, candidates=RATIO_CANDIDATES) -> dict:
    obs = np.asarray(observation, dtype=float).reshape(-1)
    pred = np.asarray(prediction, dtype=float).reshape(-1)
    good = np.isfinite(obs) & np.isfinite(pred) & (np.abs(pred) > 1e-12)
    if not good.any():
        return {"unit_or_sign_mismatch_suspected": False, "best_candidate_ratio": None, "candidate_errors": {}}
    ratios = obs[good] / pred[good]
    candidate_errors = {str(c): float(np.median(np.abs(ratios - c) / max(abs(c), 1e-30))) for c in candidates}
    best_key = min(candidate_errors, key=candidate_errors.get)
    best_ratio = float(best_key)
    return {
        "unit_or_sign_mismatch_suspected": bool(candidate_errors[best_key] < 0.05),
        "best_candidate_ratio": best_ratio,
        "best_relative_median_error": candidate_errors[best_key],
        "candidate_errors": candidate_errors,
    }


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def harmonic_transform_diagnostics(observation: np.ndarray, prediction: np.ndarray) -> dict:
    obs = np.asarray(observation, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    transforms = {
        "pred": pred,
        "conj_pred": np.column_stack([pred[:, 0], -pred[:, 1]]),
        "neg_real_plus_i_imag": np.column_stack([-pred[:, 0], pred[:, 1]]),
        "real_minus_i_imag": np.column_stack([pred[:, 0], -pred[:, 1]]),
        "imag_plus_i_real": np.column_stack([pred[:, 1], pred[:, 0]]),
        "pred_times_1000": pred * 1000.0,
        "pred_div_1000": pred / 1000.0,
    }
    scores = {name: rmse(obs, value) for name, value in transforms.items()}
    best = min(scores, key=scores.get)
    base = scores["pred"]
    return {
        "transform_rmse": scores,
        "best_transform": best,
        "best_transform_significantly_lowers_rmse": bool(scores[best] < 0.5 * base),
    }


def block_error_concentration(block_ids: np.ndarray, residual: np.ndarray, top_counts=(1, 3, 5)) -> dict:
    block_ids = np.asarray(block_ids)
    residual = np.asarray(residual, dtype=float).reshape(block_ids.shape[0], -1)
    total_sse = float(np.nansum(residual * residual))
    rows = []
    for block_id in np.unique(block_ids):
        take = block_ids == block_id
        res = residual[take]
        sse = float(np.nansum(res * res))
        rows.append({
            "block_id": int(block_id) if np.issubdtype(type(block_id), np.integer) else str(block_id),
            "pixel_count": int(take.sum()),
            "residual_RMSE": float(np.sqrt(np.nanmean(res * res))),
            "error_sse": sse,
        })
    rows.sort(key=lambda row: row["error_sse"], reverse=True)
    fractions = {}
    for n in top_counts:
        fractions[f"top_{n}_block_error_fraction"] = float(sum(row["error_sse"] for row in rows[:n]) / max(total_sse, 1e-30))
    return {"rows": rows, **fractions}


def reference_offset_diagnostics(residual: np.ndarray) -> dict:
    residual = np.asarray(residual, dtype=float)
    base = float(np.sqrt(np.nanmean(residual * residual)))
    offset = float(np.nanmean(residual))
    corrected = residual - offset
    corrected_rmse = float(np.sqrt(np.nanmean(corrected * corrected)))
    reduction = (base - corrected_rmse) / max(base, 1e-30)
    return {
        "global_mean": offset,
        "raw_RMSE": base,
        "constant_offset_removed_RMSE": corrected_rmse,
        "relative_RMSE_reduction_after_global_offset": float(reduction),
        "reference_offset_failure_suspected": bool(reduction > 0.5),
    }


def missing_forensic_products() -> dict:
    return {name: str(path) for name, path in REQUIRED_FORENSIC_PRODUCTS.items() if not path.exists()}


def unavailable_stats(label: str, reason: str) -> dict:
    return {
        "quantity": label,
        "status": "unavailable",
        "reason": reason,
        "count": None,
        "finite_count": None,
        "min": None,
        "p0.1": None,
        "p1": None,
        "p5": None,
        "p25": None,
        "median": None,
        "p75": None,
        "p95": None,
        "p99": None,
        "p99.9": None,
        "max": None,
        "mean": None,
        "std": None,
        "RMS": None,
        "MAE": None,
    }


def write_quantile_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_incomplete_block_csv(path: Path, reason: str) -> None:
    fieldnames = [
        "block_id",
        "source_file",
        "dataset_name",
        "dataset_shape",
        "dtype",
        "scale_factor",
        "offset",
        "units",
        "pixel_count",
        "observation_RMS",
        "prediction_RMS",
        "residual_RMSE",
        "residual_MAE",
        "residual_bias",
        "maximum_abs_residual",
        "finite_fraction",
        "coordinate_bounds",
        "status",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: None for key in fieldnames} | {"status": "unavailable", "reason": reason})


def update_status_files(root: Path, fold4_formal_cv_eligible: bool) -> None:
    status_path = root / "aquifer_model_revision_status.json"
    status = read_json(status_path)
    status.update({
        "G0_four_fold_protocol_status": "complete",
        "G0_four_fold_scientific_status": "blocked_by_fold4_extreme_validation_error",
        "G0_model_selection_eligible": False,
        "allow_start_geology_model_comparison_review": False,
        "allow_start_G1": False,
        "allow_start_G2": False,
        "allow_start_G3": False,
        "allow_continue_g1_g2_g3": False,
        "allow_lag_c_model_comparison": False,
        "phase4_restart_allowed": False,
        "selected_model_config": "not_generated",
        "fold4_formal_cv_eligible": fold4_formal_cv_eligible,
        "G0_four_fold_summary_status": "provisional_pending_fold4_root_cause",
    })
    write_json(status_path, status)
    for name in ["G0_four_fold_formal_summary.json", "G0_four_fold_protocol_audit.json"]:
        path = root / name
        if not path.exists():
            continue
        payload = read_json(path)
        payload.update({
            "G0_four_fold_protocol_status": "complete",
            "G0_four_fold_scientific_status": "blocked_by_fold4_extreme_validation_error",
            "G0_model_selection_eligible": False,
            "G0_four_fold_summary_status": "provisional_pending_fold4_root_cause",
            "allow_start_geology_model_comparison_review": False,
            "allow_start_G1": False,
            "allow_start_G2": False,
            "allow_start_G3": False,
            "allow_lag_c_model_comparison": False,
            "phase4_restart_allowed": False,
            "selected_model_config": "not_generated",
            "fold4_formal_cv_eligible": fold4_formal_cv_eligible,
        })
        write_json(path, payload)


def main() -> None:
    metrics = read_json(FOLD / "single_final_outer_validation_metrics.json")
    fit = read_json(FOLD / "formal_fit_status.json")
    manifest = read_json(ROOT / "formal_protocol_frozen_manifest.json")
    missing = missing_forensic_products()
    incomplete = bool(missing)
    reason = (
        "single_final_validation_pixel_level_observations_predictions_residuals_indices_coordinates_and_source_blocks_not_saved"
        if incomplete
        else "pixel_level_forensic_products_available"
    )
    update_status_files(ROOT, bool(fit.get("fold4_formal_cv_eligible")))
    rows = [
        unavailable_stats("validation_observation", reason),
        unavailable_stats("validation_prediction", reason),
        unavailable_stats("validation_residual", reason),
    ]
    write_quantile_csv(FOLD / "fold4_extreme_error_quantiles.csv", rows)
    distribution = {
        "audit_mode": "read_only_existing_final_validation_artifacts",
        "fold4_forensic_data_incomplete": incomplete,
        "missing_forensic_products": missing,
        "validation_metric_summary": {
            "validation_pixel_count": metrics.get("validation_pixel_count"),
            "validation_observation_count": metrics.get("validation_observation_count"),
            "validation_rmse_mm": metrics.get("validation_rmse_mm"),
            "validation_mae_mm": metrics.get("validation_mae_mm"),
            "validation_bias_mm": metrics.get("validation_bias_mm"),
            "harmonic_real_rmse_mm": metrics.get("harmonic_real_rmse_mm"),
            "harmonic_imag_rmse_mm": metrics.get("harmonic_imag_rmse_mm"),
            "amplitude_rmse_mm": metrics.get("amplitude_rmse_mm"),
            "phase_mae_days": metrics.get("phase_mae_days"),
        },
        "validation_observation": rows[0],
        "validation_prediction": rows[1],
        "validation_residual": rows[2],
        "fraction_abs_residual_gt_10mm": None,
        "fraction_abs_residual_gt_50mm": None,
        "fraction_abs_residual_gt_100mm": None,
        "fraction_abs_residual_gt_500mm": None,
        "fraction_abs_residual_gt_1000mm": None,
        "bias": metrics.get("validation_bias_mm"),
        "Pearson_correlation": None,
        "Spearman_correlation": None,
        "regression_slope": None,
        "regression_intercept": None,
        "status": "incomplete_pixel_level_forensic_data",
    }
    write_json(FOLD / "fold4_extreme_error_distribution_audit.json", distribution)
    write_incomplete_block_csv(FOLD / "fold4_error_by_source_block.csv", reason)
    write_json(FOLD / "fold4_units_and_metadata_audit.json", {
        "audit_mode": "read_only_existing_final_validation_artifacts",
        "fold4_forensic_data_incomplete": incomplete,
        "missing_forensic_products": missing,
        "manifest_hash": manifest.get("manifest_hash"),
        "common_mask_hash": manifest.get("common_mask_hash"),
        "fold_map_hash": manifest.get("fold_map_hash"),
        "basis_hash": manifest.get("orthogonal_basis_hash"),
        "lag_u_days": manifest.get("lag_u_global_days"),
        "lambda_multiplier": manifest.get("lambda_multiplier"),
        "observation_sigma_mm": 5.0,
        "dataset_units": None,
        "scale_factor": None,
        "offset": None,
        "dtype": None,
        "harmonic_convention": None,
        "mm_metre_conversion": None,
        "radian_displacement_conversion": None,
        "complex_component_order": None,
        "nodata_value": None,
        "reference_point": None,
        "candidate_ratio_tests": {str(c): "not_evaluable_without_pixel_observation_prediction_pairs" for c in RATIO_CANDIDATES},
        "unit_or_sign_mismatch_suspected": None,
        "status": "incomplete_pixel_level_forensic_data",
    })
    write_json(FOLD / "fold4_spatial_alignment_audit.json", {
        "audit_mode": "read_only_existing_final_validation_artifacts",
        "fold4_forensic_data_incomplete": incomplete,
        "validation_flat_indices_unique": None,
        "index_roundtrip_failure_count": None,
        "coordinate_roundtrip_failure_count": None,
        "duplicate_index_count": None,
        "ordering_hash_observation": None,
        "ordering_hash_prediction": None,
        "fold4_prediction_observation_misalignment": None,
        "reason": reason,
        "status": "incomplete_pixel_level_forensic_data",
    })
    write_json(FOLD / "fold4_harmonic_component_error_audit.json", {
        "audit_mode": "read_only_existing_final_validation_artifacts",
        "fold4_forensic_data_incomplete": incomplete,
        "real_RMSE": metrics.get("harmonic_real_rmse_mm"),
        "imag_RMSE": metrics.get("harmonic_imag_rmse_mm"),
        "amplitude_RMSE": metrics.get("amplitude_rmse_mm"),
        "phase_MAE_days": metrics.get("phase_mae_days"),
        "observation_amplitude_quantiles": None,
        "prediction_amplitude_quantiles": None,
        "phase_difference_quantiles": None,
        "candidate_transform_tests": {
            name: "not_evaluable_without_pixel_observation_prediction_pairs"
            for name in ["pred", "conj(pred)", "-real+i*imag", "real-i*imag", "imag+i*real", "pred*1000", "pred/1000"]
        },
        "real_imag_or_phase_convention_suspected": None,
        "status": "incomplete_pixel_level_forensic_data",
    })
    write_json(FOLD / "fold4_reference_consistency_audit.json", {
        "audit_mode": "read_only_existing_final_validation_artifacts",
        "fold4_forensic_data_incomplete": incomplete,
        "global_mean_residual_mm": metrics.get("validation_bias_mm"),
        "per_block_mean": None,
        "spatially_smoothed_mean": None,
        "constant_offset_removed_RMSE": None,
        "relative_RMSE_reduction_after_global_offset": None,
        "reference_offset_failure_suspected": None,
        "status": "incomplete_pixel_level_forensic_data",
    })
    write_json(FOLD / "fold4_extrapolation_support_audit.json", {
        "audit_mode": "read_only_existing_final_validation_artifacts",
        "fold4_forensic_data_incomplete": incomplete,
        "nearest_active_center_distance": None,
        "RBF_basis_row_norm": None,
        "RBF_leverage": None,
        "distance_to_nearest_training_pixel": None,
        "distance_to_training_convex_hull": None,
        "training_support_density": None,
        "orthogonal_basis_magnitude": None,
        "predicted_Ske": None,
        "prediction_amplitude": None,
        "residual_magnitude": None,
        "correlations_with_residual": None,
        "fold4_out_of_domain_generalization_failure": None,
        "status": "incomplete_pixel_level_forensic_data",
    })
    write_json(FOLD / "fold4_spatial_residual_products_status.json", {
        "requested_products": [
            "fold4_validation_observation.tif",
            "fold4_validation_prediction.tif",
            "fold4_validation_residual.tif",
            "fold4_abs_residual.tif",
            "fold4_residual_preview.png",
            "normal_range_preview",
            "full_range_preview",
            "log_abs_residual_preview",
        ],
        "generated": False,
        "reason": reason,
        "validation_area_was_not_reaccessed": True,
        "status": "not_generated_incomplete_pixel_level_forensic_data",
    })
    write_json(FOLD / "fold4_extreme_error_root_cause_assessment.json", {
        "root_cause": "inconclusive",
        "evidence": [
            "fold4 formal protocol passed and fold4_formal_cv_eligible remains true",
            "fold4 aggregate validation RMSE is 766.5605 mm, far above folds 1-3",
            "single final validation did not save pixel-level observations, predictions, residuals, indices, coordinates, or source block IDs",
            "read-only constraints prohibit recomputing validation predictions for forensic reconstruction in this round",
        ],
        "affected_files": [str(FOLD / "single_final_outer_validation_metrics.json")],
        "affected_folds": ["fold4_extreme_error_confirmed_by_aggregate_metrics", "fold1_fold3_not_shown_affected_by_current_evidence"],
        "affected_pipeline_version": manifest.get("objective_version"),
        "whether_generic_bug": "unknown",
        "whether_retraining_required": "unknown_pending_pixel_level_forensic_data",
        "validation_recompute_performed": False,
        "fold4_deleted_as_outlier": False,
        "old_fold_new_fold_mixing_allowed_if_input_hash_changes": False,
        "frozen_parameter_validation_recompute_allowed_only_for_confirmed_validation_assembly_bug": True,
        "fold4_final_training_checkpoint_hash": fit.get("final_training_checkpoint_hash"),
        "G0_four_fold_protocol_status": "complete",
        "G0_four_fold_scientific_status": "blocked_by_fold4_extreme_validation_error",
        "G0_model_selection_eligible": False,
        "allow_start_G1_G2_G3": False,
        "status": "blocked_pending_fold4_root_cause",
    })
    print(json.dumps({
        "fold4_forensic_data_incomplete": incomplete,
        "missing_forensic_product_count": len(missing),
        "G0_four_fold_scientific_status": "blocked_by_fold4_extreme_validation_error",
        "G0_model_selection_eligible": False,
        "allow_start_G1_G2_G3": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
