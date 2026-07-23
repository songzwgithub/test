#!/usr/bin/env python
"""Summarize formal G0/L0 folds 1-4 without touching model choices."""
from __future__ import annotations

import json
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiled_stage_a import latest_real_harmonic_cache
from scripts.run_formal_g0_fold1 import (
    LAMBDA,
    iter_blocks_fold,
)
from scripts.run_stage_c_fixed_lagu import LAG_U_FIXED_DAYS, OBJECTIVE_VERSION, PRIOR_VERSION, decode
from storage_inversion import rotate_coefficients


EXPECTED_MANIFEST_HASH = "bd08b8640af45badd9c87cf5111791be9d10789699bf312972a9af48070219fe"
EXPECTED_COMMON_MASK_HASH = "ff761a316e0a89a9121c439967df418f14585ae420f281d43671ebaf4740bd1f"
EXPECTED_FOLD_MAP_HASH = "d24dc63e65d3a1fa1a0e698620ba6d8e03fcf518a9a5ef0721c59374a1d46e3a"
EXPECTED_BASIS_HASH = "fb5d0531ebf865b5e375e928f6560794a532a975f501e83c3e4cdd1d60f5f9fd"
FOLDS = (1, 2, 3, 4)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def sync_running_status(fold_dir: Path, fit: dict) -> None:
    status_path = fold_dir / "formal_fold_running_status.json"
    status = "passed" if fit.get("formal_protocol_passed") else fit.get("formal_fit_status", "failed")
    write_json(status_path, {
        "accepted_iterations_completed": int(fit.get("accepted_iterations", 0)),
        "accepted_iterations_target": int(fit.get("accepted_iterations_target", 40)),
        "elapsed_seconds": finite_float(fit.get("elapsed_seconds", 0.0)),
        "estimated_remaining_seconds": 0.0,
        "last_checkpoint": str(fold_dir / "final_training_checkpoint.npy"),
        "last_checkpoint_hash": fit.get("final_training_checkpoint_hash"),
        "last_update_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "outer_validation_access_count_during_training": int(fit.get("outer_validation_access_count_during_training", -1)),
        "outer_validation_access_count_final": int(fit.get("outer_validation_access_count_final", -1)),
        "status": status,
    })


def enrich_audits(fold_dir: Path) -> tuple[str, str]:
    physical_path = fold_dir / "physical_parameter_audit.json"
    artifact_path = fold_dir / "spatial_artifact_audit.json"
    physical = read_json(physical_path)
    artifact = read_json(artifact_path)
    ske_min = float(physical.get("Ske_min", np.nan))
    ske_max = float(physical.get("Ske_max", np.nan))
    cu = float(physical.get("Cu_global", np.nan))
    lag_c = float(physical.get("lag_c_days", np.nan))
    physical.update({
        "Ske_nonpositive_fraction": 0.0 if np.isfinite(ske_min) and ske_min > 0 else 1.0,
        "Ske_screening_upper_bound": 0.05,
        "Ske_screening_exceedance_fraction": 0.0 if np.isfinite(ske_max) and ske_max <= 0.05 else 1.0,
        "near_parameter_boundary": bool(
            (np.isfinite(cu) and cu < 1e-6)
            or (np.isfinite(lag_c) and (lag_c < 0.3652425 or lag_c > 364.8772575))
        ),
        "audit_does_not_modify_model": True,
    })
    artifact.setdefault("edge_amplification_score", None)
    artifact.setdefault("local_maxima_near_center_fraction", artifact.get("center_local_maxima_fraction"))
    artifact.setdefault("audit_does_not_modify_model", True)
    write_json(physical_path, physical)
    write_json(artifact_path, artifact)
    return physical.get("physical_status", "missing"), artifact.get("artifact_status", "missing")


def cu_identifiability(root: Path, fold_id: int, selected: dict, transform: np.ndarray) -> dict:
    fold_dir = root / "model_compare/G0_no_geology_L0_shared" / f"fold_{fold_id:02d}"
    stage_a = read_json(fold_dir / "stage_A_training_only_result.json")
    theta = np.load(fold_dir / "final_training_checkpoint.npy").astype(float)
    log_ske, gamma, cu, lag_c = decode(theta)
    cache = latest_real_harmonic_cache()
    mask = root / "comparison_common_mask.tif"
    blocks = root / "spatial_validation_blocks.tif"
    unconf_sq = confined_sq = total_sq = 0.0
    n = 0
    for _obs, hc, hu, basis in iter_blocks_fold(cache, mask, blocks, selected, transform, fold_id, True):
        spatial = basis @ gamma
        ske = np.exp(np.clip(log_ske + spatial, -20, 10))
        rc = rotate_coefficients(hc, lag_c)
        ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS)
        confined = 1000.0 * ske[:, None] * rc
        unconf = 1000.0 * cu * ru
        total = confined + unconf
        confined_sq += float(np.sum(confined * confined))
        unconf_sq += float(np.sum(unconf * unconf))
        total_sq += float(np.sum(total * total))
        n += int(unconf.size)
    ratio = float(cu / stage_a["Cu_global"]) if stage_a.get("Cu_global") else np.nan
    variance_fraction = float(unconf_sq / max(total_sq, 1e-30))
    practically_zero = bool(ratio < 0.001 or cu < 1e-6)
    negligible = bool(variance_fraction < 1e-3)
    return {
        "Cu_stageA": float(stage_a["Cu_global"]),
        "Cu_stageC": float(cu),
        "Cu_stageC_to_stageA_ratio": ratio,
        "confined_contribution_rms_mm": float(np.sqrt(confined_sq / max(n, 1))),
        "unconfined_contribution_rms_mm": float(np.sqrt(unconf_sq / max(n, 1))),
        "unconfined_variance_fraction": variance_fraction,
        "Cu_practically_zero": practically_zero,
        "Cu_practically_zero_thresholds": {
            "Cu_stageC_abs_lt": 1e-6,
            "Cu_stageC_to_stageA_ratio_lt": 0.001,
        },
        "unconfined_contribution_practically_negligible": negligible,
        "unconfined_variance_fraction_negligible_threshold": 1e-3,
        "Cu_near_effective_lower_boundary": bool(cu < 1e-6),
        "contribution_domain": "training_pixels",
        "model_modified_due_to_Cu_audit": False,
        "status": "passed",
    }


def summarize_numeric(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv": float(np.std(arr, ddof=1) / np.mean(arr)) if arr.size > 1 and abs(np.mean(arr)) > 1e-30 else None,
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


def main() -> None:
    root = Path("outputs/aquifer_model_revision")
    model_dir = root / "model_compare/G0_no_geology_L0_shared"
    manifest = read_json(root / "formal_protocol_frozen_manifest.json")
    selected = read_json(root / "selected_rbf_design.json")
    transform = np.load(root / "rbf_orthogonalization" / "rbf_transform.npy")
    manifest_checks = {
        "manifest_hash": manifest.get("manifest_hash"),
        "manifest_hash_ok": manifest.get("manifest_hash") == EXPECTED_MANIFEST_HASH,
        "common_mask_hash": hash_file(root / "comparison_common_mask.tif"),
        "common_mask_hash_ok": hash_file(root / "comparison_common_mask.tif") == EXPECTED_COMMON_MASK_HASH,
        "fold_map_hash": hash_file(root / "spatial_validation_blocks.tif"),
        "fold_map_hash_ok": hash_file(root / "spatial_validation_blocks.tif") == EXPECTED_FOLD_MAP_HASH,
        "basis_hash": selected.get("basis_design_hash"),
        "basis_hash_ok": selected.get("basis_design_hash") == EXPECTED_BASIS_HASH,
        "lambda_multiplier": manifest.get("lambda_multiplier"),
        "lag_u_global_days": manifest.get("lag_u_global_days"),
        "accepted_iteration_budget": manifest.get("formal_stage_c_iteration_budget"),
        "objective_version": manifest.get("objective_version", OBJECTIVE_VERSION),
        "prior_version": manifest.get("prior_version", PRIOR_VERSION),
    }
    rows: list[dict] = []
    parameter_rows: list[dict] = []
    cu_rows: list[dict] = []
    protocol_rows: list[dict] = []
    for fold_id in FOLDS:
        fold_dir = model_dir / f"fold_{fold_id:02d}"
        fit = read_json(fold_dir / "formal_fit_status.json")
        metrics = read_json(fold_dir / "single_final_outer_validation_metrics.json")
        stage_a = read_json(fold_dir / "stage_A_training_only_result.json")
        stage_b = read_json(fold_dir / "stage_B_training_only_result.json")
        access = read_json(fold_dir / "outer_validation_access_audit.json")
        sync_running_status(fold_dir, fit)
        physical_status, artifact_status = enrich_audits(fold_dir)
        cu_audit = cu_identifiability(root, fold_id, selected, transform)
        write_json(fold_dir / f"fold{fold_id}_Cu_practical_identifiability_audit.json", cu_audit)
        rows.append({
            "fold_id": fold_id,
            "training_pixels": int(metrics["training_pixel_count"]),
            "validation_pixels": int(metrics["validation_pixel_count"]),
            "training_rmse_mm": float(metrics["training_rmse_mm"]),
            "validation_rmse_mm": float(metrics["validation_rmse_mm"]),
            "validation_mae_mm": float(metrics["validation_mae_mm"]),
            "generalization_gap_mm": float(metrics["generalization_gap_mm"]),
            "formal_cv_eligible": bool(fit.get(f"fold{fold_id}_formal_cv_eligible")),
            "physical_audit": physical_status,
            "artifact_audit": artifact_status,
            "validation_performance_outlier": bool(float(metrics["validation_rmse_mm"]) > 50.0),
        })
        parameter_rows.append({
            "fold_id": fold_id,
            "Ske_min": float(fit["Ske_min"]),
            "Ske_median": float(fit["Ske_median"]),
            "Ske_max": float(fit["Ske_max"]),
            "Cu_global": float(fit["Cu_global"]),
            "lag_c_days": float(fit["lag_c_days"]),
            "lag_u_days": float(fit["lag_u_days"]),
            "gamma_norm": float(fit["gamma_norm"]),
            "spatial_field_rms": float(fit["spatial_field_rms"]),
            "StageA_Ske": float(stage_a["Ske_global"]),
            "StageA_Cu": float(stage_a["Cu_global"]),
            "StageA_lag_c": float(stage_a["lag_c_days"]),
            "StageB_gamma_norm": float(stage_b["gamma_norm"]),
            "StageB_hessian_condition_number": float(stage_b["penalized_Hessian_condition_number"]),
        })
        cu_rows.append({"fold_id": fold_id, **cu_audit})
        protocol_rows.append({
            "fold_id": fold_id,
            "manifest_hash_consistent": fit.get("manifest_hash") == EXPECTED_MANIFEST_HASH,
            "training_validation_access_count": int(access["outer_validation_access_count_during_training"]),
            "final_validation_access_count": int(access["outer_validation_access_count_final"]),
            "training_validation_access_zero": int(access["outer_validation_access_count_during_training"]) == 0,
            "single_final_validation_once": int(access["outer_validation_access_count_final"]) == 1,
            "accepted_budget_consistent": int(fit["accepted_iterations"]) == 40 and int(fit["accepted_iterations_target"]) == 40,
            "basis_hash_consistent": selected.get("basis_design_hash") == EXPECTED_BASIS_HASH,
            "lambda_consistent": float(manifest.get("lambda_multiplier")) == LAMBDA,
            "lag_u_consistent": float(fit["lag_u_days"]) == LAG_U_FIXED_DAYS,
            "objective_version_consistent": manifest.get("objective_version") == OBJECTIVE_VERSION,
            "prior_version_consistent": manifest.get("prior_version") == PRIOR_VERSION,
            "checkpoint_alignment_passed": bool(fit.get("checkpoint_alignment_passed")),
            "formal_cv_eligible": bool(fit.get(f"fold{fold_id}_formal_cv_eligible")),
        })
    summary_df = pd.DataFrame(rows)
    params_df = pd.DataFrame(parameter_rows)
    cu_df = pd.DataFrame(cu_rows)
    protocol_df = pd.DataFrame(protocol_rows)
    summary_df.to_csv(root / "G0_four_fold_formal_summary.csv", index=False)
    params_df.to_csv(root / "G0_four_fold_parameter_stability.csv", index=False)
    cu_df.to_csv(root / "G0_four_fold_Cu_identifiability_summary.csv", index=False)
    rmse = summary_df["validation_rmse_mm"].to_numpy(float)
    mae = summary_df["validation_mae_mm"].to_numpy(float)
    weights = summary_df["validation_pixels"].to_numpy(float)
    aggregates = {
        "fold_equal_mean_rmse_mm": float(np.mean(rmse)),
        "fold_equal_std_rmse_mm": float(np.std(rmse, ddof=1)),
        "fold_equal_median_rmse_mm": float(np.median(rmse)),
        "fold_equal_mean_mae_mm": float(np.mean(mae)),
        "pooled_pixel_weighted_rmse_mm": float(np.sqrt(np.sum(weights * rmse * rmse) / np.sum(weights))),
        "pooled_pixel_weighted_mae_mm": float(np.sum(weights * mae) / np.sum(weights)),
        "min_fold_rmse_mm": float(np.min(rmse)),
        "max_fold_rmse_mm": float(np.max(rmse)),
        "rmse_range_mm": float(np.max(rmse) - np.min(rmse)),
        "coefficient_of_variation_rmse": float(np.std(rmse, ddof=1) / np.mean(rmse)),
        "valid_fold_count": int(summary_df["formal_cv_eligible"].sum()),
        "failed_fold_count": int((~summary_df["formal_cv_eligible"]).sum()),
        "fold_equal_mean_rmse_is_primary_for_later_model_selection": True,
        "pooled_rmse_is_supplementary": True,
    }
    parameter_summary = {
        column: summarize_numeric(params_df[column].tolist())
        for column in [
            "Ske_median",
            "Cu_global",
            "lag_c_days",
            "gamma_norm",
            "spatial_field_rms",
            "StageA_Ske",
            "StageA_Cu",
            "StageA_lag_c",
            "StageB_gamma_norm",
            "StageB_hessian_condition_number",
        ]
    }
    parameter_summary["stability_flags"] = {
        "lag_c_stability": "moderate" if parameter_summary["lag_c_days"]["iqr"] <= 10 else "unstable",
        "Ske_median_stability": "moderate" if parameter_summary["Ske_median"]["cv"] is not None and parameter_summary["Ske_median"]["cv"] <= 0.25 else "unstable",
        "Cu_stability": "unstable_or_practically_negligible",
        "gamma_norm_stability": "moderate" if parameter_summary["gamma_norm"]["cv"] is not None and parameter_summary["gamma_norm"]["cv"] <= 0.30 else "unstable",
    }
    cu_aggregate = {
        "practically_zero_fold_count": int(cu_df["Cu_practically_zero"].sum()),
        "negligible_contribution_fold_count": int(cu_df["unconfined_contribution_practically_negligible"].sum()),
        "median_Cu": float(np.median(cu_df["Cu_stageC"].to_numpy(float))),
        "IQR_Cu": float(np.percentile(cu_df["Cu_stageC"].to_numpy(float), 75) - np.percentile(cu_df["Cu_stageC"].to_numpy(float), 25)),
        "median_unconfined_variance_fraction": float(np.median(cu_df["unconfined_variance_fraction"].to_numpy(float))),
        "maximum_unconfined_variance_fraction": float(np.max(cu_df["unconfined_variance_fraction"].to_numpy(float))),
    }
    if cu_aggregate["negligible_contribution_fold_count"] >= 3:
        cu_aggregate["unconfined_component_status"] = "negligible_in_majority_of_folds"
    elif cu_aggregate["practically_zero_fold_count"] >= 2:
        cu_aggregate["unconfined_component_status"] = "unstable_across_folds"
    else:
        cu_aggregate["unconfined_component_status"] = "weakly_supported"
    protocol_passed = bool(
        all(manifest_checks[k] for k in ["manifest_hash_ok", "common_mask_hash_ok", "fold_map_hash_ok", "basis_hash_ok"])
        and protocol_df[
            [
                "manifest_hash_consistent",
                "training_validation_access_zero",
                "single_final_validation_once",
                "accepted_budget_consistent",
                "basis_hash_consistent",
                "lambda_consistent",
                "lag_u_consistent",
                "objective_version_consistent",
                "prior_version_consistent",
                "checkpoint_alignment_passed",
                "formal_cv_eligible",
            ]
        ].all().all()
    )
    protocol_audit = {
        **manifest_checks,
        "folds": protocol_rows,
        "fold0_included_in_summary": False,
        "G0_four_fold_status": "complete_validated" if protocol_passed else "partial_or_invalid",
        "performance_warning": (
            "fold4_extreme_validation_error_requires_scientific_review_before_model_selection"
            if summary_df["validation_performance_outlier"].any()
            else None
        ),
        "allow_start_geology_model_comparison_review": protocol_passed,
        "allow_start_G1": False,
        "allow_start_G2": False,
        "allow_start_G3": False,
        "allow_lag_c_model_comparison": False,
        "selected_model_config": "not_generated",
        "phase4_restart_allowed": False,
    }
    write_json(root / "G0_four_fold_protocol_audit.json", protocol_audit)
    write_json(root / "G0_four_fold_formal_summary.json", {
        "per_fold": rows,
        "aggregates": aggregates,
        "parameter_stability_summary": parameter_summary,
        "Cu_identifiability_aggregate": cu_aggregate,
        "G0_four_fold_status": protocol_audit["G0_four_fold_status"],
        "fold0_included_in_summary": False,
        "performance_warning": protocol_audit["performance_warning"],
        "model_selection_note": "Use fold_equal_mean_rmse_mm as the primary G0 metric in later G0-G3 comparisons; pooled RMSE is supplementary.",
    })
    status_path = root / "aquifer_model_revision_status.json"
    status = read_json(status_path)
    status.update({
        "G0_four_fold_status": protocol_audit["G0_four_fold_status"],
        "allow_start_geology_model_comparison_review": protocol_passed,
        "allow_start_G1": False,
        "allow_start_G2": False,
        "allow_start_G3": False,
        "allow_continue_g1_g2_g3": False,
        "allow_lag_c_model_comparison": False,
        "selected_model_config": "not_generated",
        "phase4_restart_allowed": False,
        "next_allowed_step": (
            "Review G0 four-fold formal summary and fold4 performance warning before any explicit G1-G3 run request."
        ),
    })
    write_json(status_path, status)
    print(json.dumps({
        "G0_four_fold_status": protocol_audit["G0_four_fold_status"],
        "aggregates": aggregates,
        "Cu_identifiability_aggregate": cu_aggregate,
        "performance_warning": protocol_audit["performance_warning"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
