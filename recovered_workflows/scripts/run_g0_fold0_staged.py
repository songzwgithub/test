#!/usr/bin/env python
"""Run only G0 fold0 staged M1 refit with the reduced active RBF basis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from m1_inversion import M1ParameterLayout, parameter_hash
from spatial_refit_validation import (
    _active_mask_for_stage,
    _parameter_boundary_proximity,
    _prior_precision_for_layout,
    _rms,
    _run_scaled_optimizer,
    _streaming_metrics,
    latest_real_harmonic_cache,
    load_global_rbf_selection,
    RealM1Dataset,
)


def logit(p):
    p = np.clip(float(p), 1e-9, 1 - 1e-9)
    return float(np.log(p / (1 - p)))


def theta_with_lags(layout, lag_c_days, lag_u_days, ske_intercept=np.log(0.0015), cu_raw=-7.0):
    theta = np.zeros(layout.total_parameters, float)
    theta[layout.slices["ske"].start] = ske_intercept
    theta[layout.slices["lag_c"].start] = logit(float(lag_c_days) / 365.2425)
    theta[layout.slices["cu_global"]] = cu_raw
    theta[layout.slices["lag_u_global"]] = logit(float(lag_u_days) / 182.62125) if lag_u_days > 0 else -30.0
    return theta


def stage_row(label, run, theta, dataset, fold_id, layout, transform):
    train = _streaming_metrics(theta, dataset, "G0_no_geology", "L0_shared", fold_id, layout, train=True, design_transform=transform)
    val = _streaming_metrics(theta, dataset, "G0_no_geology", "L0_shared", fold_id, layout, train=False, design_transform=transform)
    decoded = _parameter_boundary_proximity(theta, layout)
    return {
        "start_id": label,
        "initial_objective": float(run["initial_objective"]),
        "final_objective": float(run["final_objective"]),
        "optimizer_success": bool(run["result"].success),
        "optimizer_status": str(run["result"].message),
        "project_convergence": bool(run["converged_by_project"]),
        "iterations": int(run["result"].nit),
        "function_evaluations": int(run["result"].nfev),
        "gradient_rms": _rms(run["final_grad"]),
        "relative_step": float(run["history"]["relative_parameter_step_history"][-1]) if run["history"]["relative_parameter_step_history"] else np.nan,
        "Ske_global": float(theta[layout.slices["ske"].start]),
        "Cu_global": val.get("Cu_global"),
        "lag_c_days": val.get("lag_c_median_days"),
        "lag_u_days": val.get("lag_u_global_days"),
        "training_rmse": train.get("rmse"),
        "validation_rmse": val.get("rmse"),
        "validation_mae": val.get("mae"),
        "generalization_gap": val.get("rmse") - train.get("rmse") if val and train else np.nan,
        "near_parameter_boundary": bool(decoded["near_parameter_boundary"]),
    }


def close_stage_a(rows):
    ok = rows[(rows["optimizer_success"]) | (rows["project_convergence"])]
    if ok.empty:
        return "failed", False
    best = rows.sort_values("final_objective").head(3)
    rel_span = (best["final_objective"].max() - best["final_objective"].min()) / max(abs(best["final_objective"].min()), 1.0)
    lag_span = max(best["lag_c_days"].max() - best["lag_c_days"].min(), best["lag_u_days"].max() - best["lag_u_days"].min())
    multimodal = bool(rel_span > 1e-3 or lag_span > 10)
    if multimodal:
        return "complete_multimodal_warning", True
    return "complete_converged", False


def choose_lambda(df):
    valid = df[(df["optimizer_success"]) | (df["project_convergence"])].copy()
    if valid.empty:
        return None, "no converged lambda candidates"
    best_rmse = float(valid["validation_rmse"].min())
    candidates = valid[valid["validation_rmse"] <= best_rmse * 1.005].sort_values(["lambda_multiplier"], ascending=False)
    row = candidates.iloc[0]
    return float(row["lambda_multiplier"]), "minimum validation RMSE with 0.5% stronger-regularization tie rule"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--stage-a-screen-maxiter", type=int, default=15)
    parser.add_argument("--stage-a-final-maxiter", type=int, default=100)
    parser.add_argument("--stage-b-maxiter", type=int, default=50)
    parser.add_argument("--stage-c-maxiter", type=int, default=100)
    args = parser.parse_args()
    if args.fold_id != 0:
        raise ValueError("This script is intentionally restricted to G0 fold0")
    output = Path(args.output_root)
    fold_dir = output / "model_compare" / "G0_no_geology_L0_shared" / "fold_00"
    fold_dir.mkdir(parents=True, exist_ok=True)
    selection = load_global_rbf_selection(output)
    if selection is None:
        raise RuntimeError("rbf_global_basis_selection.json is required before staged optimization")
    cache = latest_real_harmonic_cache()
    rasters = {
        "cumulative_confined_clay_thickness_m": Path("data/geology_rasters/cumulative_confined_clay_thickness_m.tif"),
        "quaternary_thickness_m": Path("data/geology_rasters/quaternary_thickness_m.tif"),
        "confined_clay_fraction": Path("data/geology_rasters/confined_clay_fraction.tif"),
    }
    dataset = RealM1Dataset(cache, rasters, rasters["cumulative_confined_clay_thickness_m"], output, n_folds=5, rbf_selection=selection)
    layout = M1ParameterLayout(0, len(selection["active_column_indices"]), 0, 0, "L0_shared")
    transform, rbf_diag = dataset.compute_rbf_design_transform("G0_no_geology", 0, fold_dir)
    if not rbf_diag["rbf_fold_condition_passed"]:
        raise RuntimeError("RBF fold condition did not pass; optimization is prohibited")
    project_ftol, project_gtol, project_xtol, stable = 1e-8, 1e-5, 1e-6, 3
    maxfun = 500
    maxls = 10

    stage_a = fold_dir / "stage_A"
    stage_a.mkdir(parents=True, exist_ok=True)
    screen_rows = []
    screen_runs = {}
    active_a = _active_mask_for_stage(layout, "A_global_no_rbf")
    for lag_c in [15, 30, 45, 60]:
        for lag_u in [0, 15, 30, 45, 60]:
            sid = f"lagc_{lag_c:g}_lagu_{lag_u:g}"
            theta0 = theta_with_lags(layout, lag_c, lag_u)
            run = _run_scaled_optimizer(
                dataset, "G0_no_geology", "L0_shared", 0, layout, theta0,
                args.stage_a_screen_maxiter, project_ftol, project_gtol, project_xtol, stable,
                transform, _prior_precision_for_layout(layout, 1.0), stage_a / "screen" / sid,
                stage_name=f"stage_A_screen_{sid}", active_mask=active_a, write_iteration_history=False,
                maxfun=max(5 * args.stage_a_screen_maxiter, args.stage_a_screen_maxiter), maxls=maxls,
            )
            theta = run["theta_final"]
            row = stage_row(sid, run, theta, dataset, 0, layout, transform)
            row["lag_c_initial"] = lag_c
            row["lag_u_initial"] = lag_u
            screen_rows.append(row)
            screen_runs[sid] = theta
            pd.DataFrame(screen_rows).to_csv(stage_a / "multistart_summary.csv", index=False)
    summary = pd.DataFrame(screen_rows).sort_values("final_objective")
    best_ids = summary.head(3)["start_id"].tolist()
    final_rows = []
    final_runs = {}
    for sid in best_ids:
        theta0 = screen_runs[sid]
        run = _run_scaled_optimizer(
            dataset, "G0_no_geology", "L0_shared", 0, layout, theta0,
            args.stage_a_final_maxiter, project_ftol, project_gtol, project_xtol, stable,
            transform, _prior_precision_for_layout(layout, 1.0), stage_a / sid,
            stage_name=f"stage_A_final_{sid}", active_mask=active_a, write_iteration_history=True,
            maxfun=maxfun, maxls=maxls,
        )
        row = stage_row(sid, run, run["theta_final"], dataset, 0, layout, transform)
        final_rows.append(row)
        final_runs[sid] = run
    all_stage_a = pd.concat([summary, pd.DataFrame(final_rows)], ignore_index=True)
    all_stage_a.to_csv(stage_a / "multistart_summary.csv", index=False)
    best_row = pd.DataFrame(final_rows).sort_values("final_objective").iloc[0]
    best_run = final_runs[best_row["start_id"]]
    best_dir = stage_a / "best_start"
    best_dir.mkdir(parents=True, exist_ok=True)
    np.save(best_dir / "initial_parameters.npy", best_run["theta_initial"])
    np.save(best_dir / "optimized_parameters.npy", best_run["theta_final"])
    (best_dir / "optimizer_result.json").write_text(json.dumps({
        "stage": "A",
        "stage_A_status": None,
        "best_start_id": best_row["start_id"],
        "optimizer_success": bool(best_run["result"].success),
        "optimizer_message": str(best_run["result"].message),
        "project_convergence": bool(best_run["converged_by_project"]),
        "initial_objective": float(best_run["initial_objective"]),
        "final_objective": float(best_run["final_objective"]),
        "gradient_rms": _rms(best_run["final_grad"]),
        "stop_reasons": best_run["stop_reasons"],
    }, indent=2), encoding="utf-8")
    hist = pd.DataFrame(best_run["iteration_rows"])
    if not hist.empty:
        hist.to_csv(best_dir / "optimizer_iteration_history.csv", index=False)
    stage_a_status, multimodal = close_stage_a(pd.DataFrame(final_rows))
    payload = json.loads((best_dir / "optimizer_result.json").read_text(encoding="utf-8"))
    payload["stage_A_status"] = stage_a_status
    payload["stage_A_multimodal"] = multimodal
    (best_dir / "optimizer_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if stage_a_status != "complete_converged":
        (fold_dir / "fold_metrics.json").write_text(json.dumps({
            "fold_status": "refit_failed",
            "stage_A_status": stage_a_status,
            "stage_A_multimodal": multimodal,
            "phase4_restart_allowed": False,
            "rbf_basis_selection_hash": selection["selection_mask_hash"],
        }, indent=2), encoding="utf-8")
        print(json.dumps({"stage_A_status": stage_a_status, "continue_to_stage_B": False}, indent=2))
        return

    stage_b = fold_dir / "stage_B"
    stage_b.mkdir(parents=True, exist_ok=True)
    b_rows = []
    b_thetas = {}
    active_b = _active_mask_for_stage(layout, "B_rbf_only")
    base_theta = best_run["theta_final"].copy()
    rbf_start = layout.slices["ske"].start + 1 + layout.n_ske_geology
    base_theta[rbf_start:layout.slices["ske"].stop] = 0.0
    for mult in [1, 3, 10, 30]:
        run = _run_scaled_optimizer(
            dataset, "G0_no_geology", "L0_shared", 0, layout, base_theta,
            args.stage_b_maxiter, project_ftol, project_gtol, project_xtol, stable,
            transform, _prior_precision_for_layout(layout, mult), stage_b / f"lambda_{mult}",
            stage_name=f"stage_B_lambda_{mult}", active_mask=active_b, write_iteration_history=True,
            maxfun=max(5 * args.stage_b_maxiter, args.stage_b_maxiter), maxls=maxls,
        )
        theta = run["theta_final"]
        row = stage_row(f"lambda_{mult}", run, theta, dataset, 0, layout, transform)
        train = _streaming_metrics(theta, dataset, "G0_no_geology", "L0_shared", 0, layout, train=True, design_transform=transform)
        val = _streaming_metrics(theta, dataset, "G0_no_geology", "L0_shared", 0, layout, train=False, design_transform=transform)
        row.update({
            "lambda_multiplier": mult,
            "optimizer_success": bool(run["result"].success),
            "project_convergence": bool(run["converged_by_project"]),
            "rbf_coefficient_norm": float(np.linalg.norm(theta[rbf_start:layout.slices["ske"].stop])),
            "ske_min": val.get("Ske_min"),
            "ske_median": val.get("Ske_median"),
            "ske_max": val.get("Ske_max"),
            "ske_spatial_cv": float((val.get("Ske_max") - val.get("Ske_min")) / max(val.get("Ske_median"), 1e-12)),
        })
        b_rows.append(row)
        b_thetas[mult] = theta
        pd.DataFrame(b_rows).to_csv(stage_b / "G0_fold0_rbf_regularization_sensitivity.csv", index=False)
    bdf = pd.DataFrame(b_rows)
    selected_lambda, reason = choose_lambda(bdf)
    (stage_b / "rbf_regularization_selection.json").write_text(json.dumps({
        "selected_lambda_multiplier": selected_lambda,
        "selection_reason": reason,
        "rbf_basis_selection_hash": selection["selection_mask_hash"],
    }, indent=2), encoding="utf-8")
    if selected_lambda is None:
        (fold_dir / "fold_metrics.json").write_text(json.dumps({"fold_status": "refit_failed", "stage_B_status": "failed_no_converged_lambda"}, indent=2), encoding="utf-8")
        print(json.dumps({"stage_B_status": "failed_no_converged_lambda"}, indent=2))
        return

    stage_c = fold_dir / "stage_C"
    stage_c.mkdir(parents=True, exist_ok=True)
    theta_c0 = b_thetas[selected_lambda].copy()
    np.save(stage_c / "initial_parameters.npy", theta_c0)
    run_c = _run_scaled_optimizer(
        dataset, "G0_no_geology", "L0_shared", 0, layout, theta_c0,
        args.stage_c_maxiter, 1e-7, 1e-4, 1e-5, 5,
        transform, _prior_precision_for_layout(layout, selected_lambda), stage_c,
        stage_name="stage_C_joint", active_mask=_active_mask_for_stage(layout, "C_joint"), write_iteration_history=True,
        maxfun=maxfun, maxls=maxls,
    )
    theta_c = run_c["theta_final"]
    np.save(stage_c / "optimized_parameters.npy", theta_c)
    train_c = _streaming_metrics(theta_c, dataset, "G0_no_geology", "L0_shared", 0, layout, train=True, design_transform=transform)
    val_c = _streaming_metrics(theta_c, dataset, "G0_no_geology", "L0_shared", 0, layout, train=False, design_transform=transform)
    boundary = _parameter_boundary_proximity(theta_c, layout)
    converged = bool(run_c["result"].success or run_c["converged_by_project"])
    fold_status = "refit_complete" if run_c["result"].success else ("refit_complete_project_convergence" if run_c["converged_by_project"] else "refit_failed")
    metrics = {
        "fold_status": fold_status,
        "optimizer_success": bool(run_c["result"].success),
        "project_convergence": bool(run_c["converged_by_project"]),
        "training_rmse_mm": train_c.get("rmse"),
        "validation_rmse_mm": val_c.get("rmse"),
        "validation_mae_mm": val_c.get("mae"),
        "generalization_gap_mm": val_c.get("rmse") - train_c.get("rmse"),
        "rbf_coefficient_norm": float(np.linalg.norm(theta_c[rbf_start:layout.slices["ske"].stop])),
        "Ske_min": val_c.get("Ske_min"),
        "Ske_median": val_c.get("Ske_median"),
        "Ske_max": val_c.get("Ske_max"),
        "Cu_global": val_c.get("Cu_global"),
        "lag_c_global_days": val_c.get("lag_c_median_days"),
        "lag_u_global_days": val_c.get("lag_u_global_days"),
        "near_parameter_boundary": bool(boundary["near_parameter_boundary"]),
        "selected_lambda_multiplier": selected_lambda,
        "parameter_hash": parameter_hash(theta_c),
        "rbf_basis_selection_hash": selection["selection_mask_hash"],
        "training_fold_transform_hash": transform["training_fold_transform_hash"],
        "phase4_restart_allowed": False,
    }
    (stage_c / "optimizer_result.json").write_text(json.dumps({
        "optimizer_success": bool(run_c["result"].success),
        "optimizer_message": str(run_c["result"].message),
        "project_convergence": bool(run_c["converged_by_project"]),
        "initial_objective": float(run_c["initial_objective"]),
        "final_objective": float(run_c["final_objective"]),
        "stop_reasons": run_c["stop_reasons"],
    }, indent=2), encoding="utf-8")
    pd.DataFrame(run_c["iteration_rows"]).to_csv(stage_c / "optimizer_iteration_history.csv", index=False)
    (stage_c / "fold_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (fold_dir / "fold_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
