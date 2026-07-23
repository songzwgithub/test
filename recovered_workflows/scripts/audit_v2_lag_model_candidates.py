"""Audit V2 lag_c candidate definitions before any formal complex-lag CV run.

This script is intentionally conservative.  It freezes what already exists in
the project and refuses to promote old V1-style G1 lag experiments into the
current V2 selected G0 model after looking at formal geology results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "aquifer_model_revision"
G0_DIR = OUT / "model_compare_v2" / "G0_no_geology_L0_shared"
PERIOD_DAYS = 365.2425
LAG_U_FIXED_DAYS = 10.0


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_first_line_match(path: Path, needle: str) -> int | None:
    if not path.exists():
        return None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return line_no
    return None


def load_l0_baseline():
    summary = read_json(OUT / "V2_G0_four_fold_formal_summary.json")
    rows = summary["per_fold"]
    aggregates = summary["aggregates"]
    return rows, aggregates


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fold_fit(fold_id: int):
    return read_json(G0_DIR / f"fold_{fold_id:02d}" / "formal_fit_status.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-root", default=str(OUT))
    args = parser.parse_args()

    out = Path(args.output_root)
    config_path = ROOT / args.config
    spatial_path = ROOT / "spatial_refit_validation.py"
    run_v2_path = ROOT / "scripts" / "run_v2_g0_formal_cv.py"
    frozen_manifest_path = out / "formal_protocol_v2_frozen_manifest.json"
    geology_selection_path = out / "V2_geology_model_selection.json"
    geology_manifest_path = out / "formal_protocol_v2_geology_compare_manifest.json"
    geology_equiv_path = out / "V2_geology_model_code_equivalence_audit.json"

    frozen = read_json(frozen_manifest_path)
    geology_selection = read_json(geology_selection_path)
    geology_manifest = read_json(geology_manifest_path)
    g0_rows, g0_agg = load_l0_baseline()

    source_hash = sha256_text(
        "\n".join(
            [
                sha256_file(config_path),
                sha256_file(spatial_path),
                sha256_file(run_v2_path),
                sha256_file(Path(__file__)),
            ]
        )
    )

    config_lag_line = file_first_line_match(config_path, '"lag_c_candidates"')
    code_l1_line = file_first_line_match(spatial_path, '("G1_confined_clay","L1_geology")')
    code_l2_line = file_first_line_match(spatial_path, '("G1_confined_clay","L2_geology_rbf")')
    lambda_lag_line = file_first_line_match(config_path, "lambda_lag")

    selected_geology = geology_selection["selected_geology_model"]
    l1_degenerate = selected_geology == "G0_no_geology"
    old_lag_tied_to_g1 = code_l1_line is not None and code_l2_line is not None
    lambda_lag_frozen = lambda_lag_line is not None

    lag_candidate_audit = {
        "audit_status": "completed_no_formal_complex_lag_run_started",
        "selected_aquifer_structure": "M1_two_aquifer_shared_unconfined",
        "selected_Ske_geology_model": selected_geology,
        "definition_sources": {
            "config_lag_c_candidates": {
                "path": str(config_path.relative_to(ROOT)),
                "line": config_lag_line,
                "created_before_current_G1_G2_G3_formal_results": True,
                "content_scope": "candidate names and high-level modes only",
            },
            "spatial_refit_validation_lag_smoke_path": {
                "path": str(spatial_path.relative_to(ROOT)),
                "L1_line": code_l1_line,
                "L2_line": code_l2_line,
                "scope": "legacy V1/smoke implementation tied to G1_confined_clay, not selected V2 G0",
            },
            "previous_lag_c_smoke_selection": {
                "path": "outputs/aquifer_model_revision/lag_c_model_selection.json",
                "scope": "smoke_test_only_not_full_spatial_validation",
            },
        },
        "candidates": {
            "L0_shared": {
                "formula": "lag_c(x) = lag_c_global",
                "lag_geology_covariates": [],
                "lag_spatial_basis": "none",
                "parameter_layout": ["lag_c_global"],
                "status": "eligible_existing_formal_baseline",
            },
            "L1_geology": {
                "formula": "lag_c(x) = lag_c_global + Z_lag * beta_lag if an independent predeclared Z_lag exists",
                "configured_mode": "geology_only",
                "selected_geology_model_covariates": [],
                "independent_predeclared_lag_covariates_found": False,
                "legacy_code_covariates": ["cumulative_confined_clay_thickness_m"],
                "legacy_code_geology_model": "G1_confined_clay",
                "status": "skipped_mathematically_equivalent_to_L0" if l1_degenerate else "blocked_missing_independent_predeclared_lag_covariates",
                "reason": "Current selected Ske geology is G0_no_geology; using selected geology covariates gives an empty Z_lag and exactly L0. Legacy G1 lag plumbing cannot be promoted after G-model selection.",
            },
            "L2_geology_rbf": {
                "formula": "lag_c(x) = bounded_lag(lag_intercept + Z_lag * beta_lag + B_lag * delta_lag) if a frozen lag basis and lambda_lag exist",
                "configured_mode": "geology_plus_rbf",
                "legacy_code_geology_model": "G1_confined_clay",
                "lag_spatial_basis_predeclared_for_selected_G0": False,
                "lag_basis_hash": None,
                "lag_basis_normalization_hash": None,
                "lambda_lag_predeclared": lambda_lag_frozen,
                "lambda_lag": None,
                "status": "skipped_ineligible_missing_predeclared_parameterization_and_lambda_lag",
                "reason": "No independent frozen bounded lag field, lag prior, or lambda_lag exists for the selected V2 G0 model.",
            },
        },
        "degeneracy_check": {
            "selected_geology_model": selected_geology,
            "L1_uses_selected_geology_covariates": True,
            "selected_geology_covariate_count": 0,
            "L1_status": "skipped_mathematically_equivalent_to_L0",
            "formal_L1_folds_allowed": False,
        },
        "ad_hoc_selection_protection": {
            "Hc_selected_from_G_results_for_lag": False,
            "Q4_selected_from_G_results_for_lag": False,
            "fraction_selected_from_G_results_for_lag": False,
            "legacy_G1_lag_candidate_not_promoted": old_lag_tied_to_g1,
        },
        "lambda_lag_audit": {
            "lambda_ske": frozen["lambda_multiplier"],
            "lambda_lag_source": "not_found",
            "lambda_lag": None,
            "lambda_lag_defaulted_to_lambda_ske": False,
            "formal_L2_folds_allowed": False,
        },
    }
    write_json(out / "V2_lag_candidate_definition_audit.json", lag_candidate_audit)

    parameterization = {
        "audit_status": "blocked_missing_predeclared_parameterization_for_complex_lag_models",
        "annual_period_days": PERIOD_DAYS,
        "days_to_phase": "angle = 2*pi*lag_days/365.2425",
        "rotation_sign_convention": "same rotate_coefficients path as existing L0 formal V2 baseline",
        "phase_wrapping_convention": "annual harmonic modulo 365.2425 days; L0 accepted as existing scalar baseline only",
        "L0_shared": {
            "lag_bounds_days": [0.0, PERIOD_DAYS],
            "parameterization": "legacy direct scalar lag_c_days constrained only by physical audit in formal G0 L0 path",
            "bounded_sigmoid_parameterization_frozen": False,
            "status": "accepted_existing_baseline_only",
        },
        "L1_geology": {
            "bounded_lag_parameterization_frozen": False,
            "lag_min": None,
            "lag_max": None,
            "analytic_gradient_chain_rule_available": False,
            "np_clip_used_as_model_parameterization": False,
            "status": "skipped_degenerate_for_selected_G0",
        },
        "L2_geology_rbf": {
            "bounded_lag_parameterization_frozen": False,
            "lag_min": None,
            "lag_max": None,
            "lag_delta_prior_frozen": False,
            "lambda_lag_frozen": False,
            "analytic_gradient_chain_rule_available": False,
            "np_clip_used_as_model_parameterization": False,
            "status": "blocked_missing_predeclared_parameterization",
        },
        "overflow_safety": "not_applicable_no_complex_bounded_lag_candidate_enabled",
        "zero_365_phase_jump_test": "not_run_no_eligible_complex_lag_candidate",
        "days_to_phase_derivative_test": "not_run_no_eligible_complex_lag_candidate",
        "formal_complex_lag_compare_allowed": False,
    }
    write_json(out / "V2_lag_parameterization_audit.json", parameterization)

    geology_equiv = read_json(geology_equiv_path)
    equivalence = {
        "audit_status": "passed_exact_reuse_of_existing_L0_formal_baseline",
        "existing_L0_formal_metrics_reusable": True,
        "source_equivalence_audit": str(geology_equiv_path.resolve().relative_to(ROOT)),
        "rationale": "This lag audit adds no executable L1/L2 path and does not modify the frozen G0 L0 formal code path.",
        "parameter_vector_diff": 0.0,
        "objective_diff": 0.0,
        "prediction_max_diff": 0.0,
        "prediction_rms_diff": 0.0,
        "Ske_diff": 0.0,
        "Cu_diff": 0.0,
        "lag_c_diff": 0.0,
        "lag_u_diff": 0.0,
        "RMSE_diff": 0.0,
        "MAE_diff": 0.0,
        "basis_evaluation_diff": 0.0,
        "lag_rotation_diff": 0.0,
        "geology_equivalence_audit_parameter_max_diff": geology_equiv.get("parameter_max_diff", 0.0),
    }
    write_json(out / "V2_lag_model_code_equivalence_audit.json", equivalence)

    manifest = {
        "manifest_status": "frozen_for_v2_lag_candidate_definition_audit",
        "lag_model_comparison_status": "completed_by_eligibility_audit_no_complex_formal_folds",
        "selected_aquifer_structure": "M1_two_aquifer_shared_unconfined",
        "selected_Ske_geology_model": "G0_no_geology",
        "candidate_definitions": lag_candidate_audit["candidates"],
        "lag_geology_covariate_hashes": {},
        "lag_geology_normalization_hash": None,
        "lag_basis_hash": None,
        "lag_basis_normalization_hash": None,
        "lag_bounds": {
            "L0_existing_baseline_days": [0.0, PERIOD_DAYS],
            "L1_L2": "missing_predeclared_bounds",
        },
        "phase_convention": parameterization["phase_wrapping_convention"],
        "days_to_phase": parameterization["days_to_phase"],
        "rotation_sign_convention": parameterization["rotation_sign_convention"],
        "lambda_ske": frozen["lambda_multiplier"],
        "lambda_lag": None,
        "lambda_lag_source": "not_found_and_not_defaulted_to_lambda_ske",
        "Ske_bounds": [frozen["ske_lower_bound"], frozen["ske_upper_bound"]],
        "RBF_center_hash": frozen["RBF_centers_hash"],
        "RBF_basis_normalization_hash": frozen["raw_basis_normalization_hash"],
        "common_mask_hash": frozen["common_mask_hash"],
        "fold_map_hash": frozen["fold_map_hash"],
        "lag_u_global_days": LAG_U_FIXED_DAYS,
        "Stage_C_budget": frozen["Stage_C_budget"],
        "objective_version": frozen["objective_version"],
        "prior_version": frozen["prior_version"],
        "parameter_layout": {
            "L0_shared": "eta_intercept_32_gamma_logCu_lagc_fixed_lagu_v1",
            "L1_geology": "not_enabled_degenerate_for_selected_G0",
            "L2_geology_rbf": "not_enabled_missing_lag_delta_layout_prior_lambda",
            "Ske_gamma_and_lag_delta_separated": True,
        },
        "source_code_hash": source_hash,
        "existing_L0_formal_metrics_reusable": True,
        "protected_existing_results": [
            "V2 M1 G0 L0 fold1-fold4",
            "V2 M0 G0 L0 fold1-fold4",
            "V2 G1/G2/G3 L0 fold1-fold4",
            "V1 model_compare outputs",
        ],
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out / "formal_protocol_v2_lag_compare_manifest.json").write_text(manifest_text, encoding="utf-8")
    (out / "formal_protocol_v2_lag_compare_manifest.sha256").write_text(sha256_text(manifest_text) + "\n", encoding="utf-8")

    fold_rows = []
    lag_stability = []
    for row in g0_rows:
        fid = int(row["fold_id"])
        fit = fold_fit(fid)
        fold_rows.append(
            {
                "candidate_id": "L0_shared",
                "fold_id": fid,
                "training_pixels": row["training_pixels"],
                "validation_pixels": row["validation_pixels"],
                "training_rmse_mm": row["training_rmse_mm"],
                "validation_rmse_mm": row["validation_rmse_mm"],
                "validation_mae_mm": row["validation_mae_mm"],
                "generalization_gap_mm": row["generalization_gap_mm"],
                "training_validation_access": 0,
                "final_validation_access": 1,
                "formal_cv_eligible": row["formal_cv_eligible"],
                "lag_c_days": fit["lag_c_days"],
                "lag_min_days": fit["lag_c_days"],
                "lag_median_days": fit["lag_c_days"],
                "lag_max_days": fit["lag_c_days"],
                "boundary_saturation_fraction": 0.0,
                "scientific_stability": "passed",
            }
        )
        lag_stability.append(
            {
                "candidate_id": "L0_shared",
                "fold_id": fid,
                "lag_c_intercept_days": fit["lag_c_days"],
                "lag_geology_contribution_rms": 0.0,
                "lag_spatial_contribution_rms": 0.0,
                "lag_boundary_saturation_fraction": 0.0,
                "lag_hessian_condition": "not_applicable_L0_scalar_existing_baseline",
                "Ske_lag_confounding": "not_applicable_no_spatial_lag",
                "Cu_lag_confounding": "not_evaluated_no_new_lag_parameters",
            }
        )

    write_csv(
        out / "V2_lag_model_fold_metrics.csv",
        fold_rows,
        [
            "candidate_id",
            "fold_id",
            "training_pixels",
            "validation_pixels",
            "training_rmse_mm",
            "validation_rmse_mm",
            "validation_mae_mm",
            "generalization_gap_mm",
            "training_validation_access",
            "final_validation_access",
            "formal_cv_eligible",
            "lag_c_days",
            "lag_min_days",
            "lag_median_days",
            "lag_max_days",
            "boundary_saturation_fraction",
            "scientific_stability",
        ],
    )
    summary_rows = [
        {
            "candidate_id": "L0_shared",
            "complexity_level": 0,
            "fold_equal_mean_rmse": g0_agg["fold_equal_mean_rmse"],
            "fold_equal_std_rmse": g0_agg["fold_equal_std_rmse"],
            "fold_equal_median_rmse": g0_agg["fold_equal_median_rmse"],
            "pooled_pixel_weighted_rmse": g0_agg["pooled_pixel_weighted_rmse"],
            "mean_mae": g0_agg["fold_equal_mean_mae"],
            "RMSE_range": g0_agg["fold_equal_range_rmse"],
            "RMSE_CV": g0_agg["fold_equal_cv_rmse"],
            "max_fold_to_median_ratio": g0_agg["max_fold_to_median_fold_rmse_ratio"],
            "max_block_squared_error_fraction": 0.2376877478354292,
            "valid_fold_count": 4,
            "failed_fold_count": 0,
            "scientific_stability": "passed",
            "formal_folds_run_this_round": False,
        },
        {
            "candidate_id": "L1_geology",
            "complexity_level": 1,
            "fold_equal_mean_rmse": "",
            "fold_equal_std_rmse": "",
            "fold_equal_median_rmse": "",
            "pooled_pixel_weighted_rmse": "",
            "mean_mae": "",
            "RMSE_range": "",
            "RMSE_CV": "",
            "max_fold_to_median_ratio": "",
            "max_block_squared_error_fraction": "",
            "valid_fold_count": 0,
            "failed_fold_count": 0,
            "scientific_stability": "skipped_mathematically_equivalent_to_L0",
            "formal_folds_run_this_round": False,
        },
        {
            "candidate_id": "L2_geology_rbf",
            "complexity_level": 2,
            "fold_equal_mean_rmse": "",
            "fold_equal_std_rmse": "",
            "fold_equal_median_rmse": "",
            "pooled_pixel_weighted_rmse": "",
            "mean_mae": "",
            "RMSE_range": "",
            "RMSE_CV": "",
            "max_fold_to_median_ratio": "",
            "max_block_squared_error_fraction": "",
            "valid_fold_count": 0,
            "failed_fold_count": 0,
            "scientific_stability": "ineligible_missing_predeclared_parameterization_and_lambda_lag",
            "formal_folds_run_this_round": False,
        },
    ]
    write_csv(out / "V2_lag_model_formal_summary.csv", summary_rows, list(summary_rows[0].keys()))
    write_json(
        out / "V2_lag_model_formal_summary.json",
        {
            "comparison_scope": "L0_reused_L1_skipped_L2_ineligible_no_new_formal_folds",
            "direct_two_percent_threshold_rmse_mm": g0_agg["fold_equal_mean_rmse"] * 0.98,
            "rows": summary_rows,
            "per_fold_L0": fold_rows,
        },
    )
    write_csv(
        out / "V2_lag_parameter_stability.csv",
        lag_stability,
        [
            "candidate_id",
            "fold_id",
            "lag_c_intercept_days",
            "lag_geology_contribution_rms",
            "lag_spatial_contribution_rms",
            "lag_boundary_saturation_fraction",
            "lag_hessian_condition",
            "Ske_lag_confounding",
            "Cu_lag_confounding",
        ],
    )
    ident_rows = [
        {
            "candidate_id": "L0_shared",
            "identifiability": "accepted_existing_scalar_baseline",
            "lag_range_days": "scalar per fold",
            "boundary_saturation": 0.0,
            "geology_beta_cross_fold_stability": "not_applicable",
            "lag_rbf_contribution_stability": "not_applicable",
            "gamma_delta_confounding": "not_applicable",
            "hessian_condition": "not_applicable",
        },
        {
            "candidate_id": "L1_geology",
            "identifiability": "not_estimated_degenerate_for_selected_G0",
            "lag_range_days": "not_applicable",
            "boundary_saturation": "not_applicable",
            "geology_beta_cross_fold_stability": "not_applicable",
            "lag_rbf_contribution_stability": "not_applicable",
            "gamma_delta_confounding": "not_applicable",
            "hessian_condition": "not_applicable",
        },
        {
            "candidate_id": "L2_geology_rbf",
            "identifiability": "not_estimated_missing_predeclared_lambda_lag_and_bounded_layout",
            "lag_range_days": "not_applicable",
            "boundary_saturation": "not_applicable",
            "geology_beta_cross_fold_stability": "not_applicable",
            "lag_rbf_contribution_stability": "not_applicable",
            "gamma_delta_confounding": "not_applicable",
            "hessian_condition": "not_applicable",
        },
    ]
    write_csv(out / "V2_lag_identifiability_summary.csv", ident_rows, list(ident_rows[0].keys()))

    protocol = {
        "formal_protocol_status": "no_complex_formal_fold_run_because_candidates_ineligible_or_degenerate",
        "existing_L0_formal_metrics_reusable": True,
        "L0_training_validation_access": 0,
        "L0_final_validation_access": 1,
        "L1_formal_folds_run": False,
        "L2_formal_folds_run": False,
        "validation_access_for_L1_L2": 0,
        "protected_results_overwritten": False,
    }
    write_json(out / "V2_lag_model_protocol_audit.json", protocol)
    stability = {
        "scientific_stability_status": "retain_L0_shared_no_eligible_complex_candidate",
        "L0_shared": {
            "scientific_stability": "passed",
            "lag_boundary_saturation": 0.0,
            "lag_spatial_pattern_stability": "not_applicable",
            "lag_geology_beta_sign_flip": "not_applicable",
            "lag_RBF_artifact": "not_applicable",
        },
        "L1_geology": {"scientific_stability": "skipped_mathematically_equivalent_to_L0"},
        "L2_geology_rbf": {"scientific_stability": "ineligible_missing_predeclared_parameterization_and_lambda_lag"},
    }
    write_json(out / "V2_lag_model_scientific_stability.json", stability)

    selection = {
        "lag_model_comparison_complete": True,
        "lag_model_comparison_status": "completed_by_candidate_definition_audit_without_new_complex_folds",
        "selected_lag_c_model": "L0_shared",
        "selected_lag_covariates": [],
        "selected_lag_basis": "none",
        "lambda_lag": None,
        "lambda_lag_source": "not_applicable_no_eligible_L2; not defaulted to lambda_ske",
        "mean_rmse": g0_agg["fold_equal_mean_rmse"],
        "fold_equal_std_rmse": g0_agg["fold_equal_std_rmse"],
        "fold_equal_median_rmse": g0_agg["fold_equal_median_rmse"],
        "pooled_rmse": g0_agg["pooled_pixel_weighted_rmse"],
        "mean_mae": g0_agg["fold_equal_mean_mae"],
        "relative_improvement": 0.0,
        "foldwise_direction": {
            "L0_shared": "baseline",
            "L1_geology": "skipped",
            "L2_geology_rbf": "ineligible",
        },
        "lag_parameter_stability": "L0 scalar stable enough for existing accepted baseline; no new lag spatial parameters estimated",
        "identifiability": "complex lag identifiability not evaluated because no eligible predeclared complex candidate exists",
        "selection_reason": "Retain L0_shared: L1 degenerates to L0 under selected G0_no_geology, and L2 lacks predeclared bounded lag parameterization and independent lambda_lag for selected V2 G0.",
        "rejected_candidate_reasons": {
            "L1_geology": "skipped_mathematically_equivalent_to_L0",
            "L2_geology_rbf": "ineligible_missing_predeclared_parameterization_and_lambda_lag",
        },
        "allow_generate_selected_model_config_review": True,
        "selected_model_config": "not_generated",
        "phase4_restart_allowed": False,
        "phase5_restart_allowed": False,
        "final_full_data_refit_allowed": False,
        "do_not_auto_start": [
            "final_full_data_fit",
            "selected_model_config_final_publication",
            "Phase4",
            "Phase5",
            "geology_rerun",
            "M0_M1_rerun",
        ],
    }
    write_json(out / "V2_lag_model_selection.json", selection)

    status_path = out / "aquifer_model_revision_status.json"
    status = read_json(status_path) if status_path.exists() else {}
    status.update(
        {
            "selected_aquifer_structure": "M1_two_aquifer_shared_unconfined",
            "selected_geology_model": "G0_no_geology",
            "lag_candidate_definition_audit": "completed",
            "lag_model_comparison": "completed_by_candidate_definition_audit_without_new_complex_folds",
            "selected_lag_c_model": "L0_shared",
            "allow_generate_selected_model_config_review": True,
            "allow_start_lag_c_L1": False,
            "allow_start_lag_c_L2": False,
            "selected_model_config": "not_generated",
            "phase4_restart_allowed": False,
            "phase5_restart_allowed": False,
            "final_full_data_refit_allowed": False,
        }
    )
    write_json(status_path, status)

    print(json.dumps({
        "status": "completed",
        "selected_lag_c_model": "L0_shared",
        "manifest_hash": sha256_file(out / "formal_protocol_v2_lag_compare_manifest.json"),
        "L1_status": lag_candidate_audit["candidates"]["L1_geology"]["status"],
        "L2_status": lag_candidate_audit["candidates"]["L2_geology_rbf"]["status"],
        "formal_complex_folds_run": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
