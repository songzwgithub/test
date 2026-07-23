#!/usr/bin/env python
"""Recompute development checkpoints and run fold0 formal-protocol dry run.

The dry run is restricted to fold0 and is not a formal CV result. It verifies
the no-outer-validation-during-training contract before fold1 can be piloted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiled_stage_a import StageAStats, solve_from_stats
from scripts.audit_stage_b_lambda_effect import artifact_metrics
from scripts.run_stage_b_fixed_lagu import accumulate_quadratic, evaluate_gamma
from scripts.run_stage_c_fixed_lagu import (
    LAG_U_FIXED_DAYS,
    OBJECTIVE_VERSION,
    PRIOR_VERSION,
    decode,
    iter_blocks,
    metrics,
    objective_grad,
)
from storage_inversion import rotate_coefficients


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def hash_array(arr: np.ndarray) -> str:
    return sha256(np.asarray(arr, dtype="float64").tobytes()).hexdigest()


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def add_stats(a: StageAStats | None, b: StageAStats) -> StageAStats:
    if a is None:
        return b
    payload = {k: getattr(a, k) + getattr(b, k) for k in asdict(a) if k not in {"observation_sigma_mm", "period_days"}}
    payload["n"] = int(payload["n"])
    payload["observation_sigma_mm"] = a.observation_sigma_mm
    payload["period_days"] = a.period_days
    return StageAStats(**payload)


def stats_from_block(obs: np.ndarray, hc: np.ndarray, hu: np.ndarray, sigma=5.0, period=365.2425) -> StageAStats:
    invsig2 = 1.0 / max(sigma**2, 1e-30)
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
        observation_sigma_mm=float(sigma),
        period_days=float(period),
    )


def stream_stage_a_stats(cache, mask, blocks, selected, transform) -> StageAStats:
    stats = None
    for obs, hc, hu, _basis, _rc in iter_blocks(cache, mask, blocks, selected, transform, train=True):
        stats = add_stats(stats, stats_from_block(obs, hc, hu))
    if stats is None:
        raise RuntimeError("No training pixels available for Stage A dry run")
    return stats


def fixed_lagu_stage_a(stats: StageAStats) -> dict:
    coarse = []
    for lag_c in np.arange(0.0, 91.0, 10.0):
        coarse.append(solve_from_stats(stats, lag_c, LAG_U_FIXED_DAYS))
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
        "Ske_global": best.ske_global,
        "Cu_global": best.cu_global,
        "lag_c_days": best.lag_c_days,
        "lag_u_days": LAG_U_FIXED_DAYS,
        "training_objective": best.objective,
        "training_rmse": best.rmse,
        "status": best.status,
    }


def parameter_row(iteration, theta, objective, grad, parts, train, previous_theta, nfev, checkpoint, checkpoint_hash):
    step = float(np.linalg.norm(theta - previous_theta)) if previous_theta is not None else np.nan
    rel_step = step / max(float(np.linalg.norm(previous_theta)), 1.0) if previous_theta is not None else np.nan
    return {
        "accepted_iteration": int(iteration),
        "callback_sequence": int(iteration),
        "scipy_nit": int(iteration),
        "function_evaluations": int(nfev),
        "parameter_hash": hash_array(theta),
        "checkpoint_filename": checkpoint.name,
        "checkpoint_hash": checkpoint_hash,
        **parts,
        "training_rmse_mm": train["rmse"],
        "validation_rmse_mm": np.nan,
        "validation_mae_mm": np.nan,
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
    }


def checkpoint_metadata_payload(iteration, theta, objective, train, validation, selected, mask_hash, lam):
    return {
        "accepted_iteration": int(iteration),
        "parameter_hash": hash_array(theta),
        "objective": float(objective),
        "training_rmse": train["rmse"],
        "validation_rmse_if_development": None if validation is None else validation["rmse"],
        "basis_hash": selected["basis_design_hash"],
        "mask_hash": mask_hash,
        "lambda": float(lam),
        "lag_u": LAG_U_FIXED_DAYS,
    }


def _init_checkpoint_accumulators(checkpoints):
    out = []
    for item in checkpoints:
        log_ske, gamma, cu, lag_c = decode(item["theta"])
        dglob = item["theta"][[0, 33, 34]]
        gpen = 0.5 * float(np.sum(np.array([1.0, 1.0, 1.0 / (365.2425**2)]) * 0.0 * dglob))
        out.append(
            {
                **{k: item[k] for k in ("checkpoint_iteration", "parameter_hash", "checkpoint_hash")},
                "theta": item["theta"],
                "log_ske": log_ske,
                "gamma": gamma,
                "Cu_global": cu,
                "lag_c": lag_c,
                "lag_u": LAG_U_FIXED_DAYS,
                "train_sse": 0.0,
                "train_abs": 0.0,
                "train_ncoef": 0,
                "validation_sse": 0.0,
                "validation_abs": 0.0,
                "validation_ncoef": 0,
                "data_loss": 0.0,
                "spatial_sum_sq": 0.0,
                "spatial_n": 0,
                "Ske_min": np.inf,
                "Ske_max": -np.inf,
                "Ske_sample": [],
                "gamma_norm": float(np.linalg.norm(gamma)),
                "artifact_score": float(np.max(np.abs(gamma)) / max(np.sqrt(np.mean(gamma * gamma)), 1e-12)),
                "_unused_global_prior_placeholder": gpen,
            }
        )
    return out


def _update_checkpoint_metric_acc(acc, obs, hc, hu, basis, train, lam, global_prior):
    log_ske = acc["log_ske"]
    gamma = acc["gamma"]
    cu = acc["Cu_global"]
    lag_c = acc["lag_c"]
    spatial = basis @ gamma
    ske = np.exp(np.clip(log_ske + spatial, -20, 10))
    pred = 1000.0 * (ske[:, None] * rotate_coefficients(hc, lag_c) + cu * rotate_coefficients(hu, LAG_U_FIXED_DAYS))
    res = obs - pred
    sse = float(np.sum(res * res))
    ae = float(np.sum(np.abs(res)))
    ncoef = int(res.size)
    if train:
        acc["train_sse"] += sse
        acc["train_abs"] += ae
        acc["train_ncoef"] += ncoef
        acc["data_loss"] += 0.5 * sse / (5.0**2)
        acc["spatial_sum_sq"] += float(np.sum(spatial * spatial))
        acc["spatial_n"] += int(spatial.size)
        acc["Ske_min"] = min(acc["Ske_min"], float(np.min(ske)))
        acc["Ske_max"] = max(acc["Ske_max"], float(np.max(ske)))
        if len(acc["Ske_sample"]) < 200_000:
            need = 200_000 - len(acc["Ske_sample"])
            acc["Ske_sample"].extend(ske[:need].astype(float).tolist())
    else:
        acc["validation_sse"] += sse
        acc["validation_abs"] += ae
        acc["validation_ncoef"] += ncoef


def _fast_recompute_checkpoint_metrics(checkpoints, cache, mask, blocks, selected, transform, lam, global_prior):
    accs = _init_checkpoint_accumulators(checkpoints)
    for train in (True, False):
        for obs, hc, hu, basis, _ in iter_blocks(cache, mask, blocks, selected, transform, train=train):
            for acc in accs:
                _update_checkpoint_metric_acc(acc, obs, hc, hu, basis, train, lam, global_prior)
    rows = []
    precision = global_prior["precision"]
    mean = global_prior["mean"]
    for acc in accs:
        theta = acc["theta"]
        gamma = acc["gamma"]
        gamma_prior_scaled = float(lam * 0.5 * gamma @ gamma)
        dglob = theta[[0, 33, 34]] - mean
        global_prior_penalty = 0.5 * float(np.sum(precision * dglob * dglob))
        objective = acc["data_loss"] + gamma_prior_scaled + global_prior_penalty
        ske_sample = np.asarray(acc["Ske_sample"], dtype=float)
        physical = bool(np.isfinite(objective) and acc["Ske_min"] > 0 and acc["Cu_global"] > 0 and 0 <= acc["lag_c"] <= 365.2425)
        artifact = bool(acc["artifact_score"] < 6.0)
        rows.append(
            {
                "checkpoint_iteration": acc["checkpoint_iteration"],
                "parameter_hash": acc["parameter_hash"],
                "raw_objective": float(objective),
                "normalized_objective": np.nan,
                "training_rmse": float(np.sqrt(acc["train_sse"] / max(acc["train_ncoef"], 1))),
                "validation_rmse": float(np.sqrt(acc["validation_sse"] / max(acc["validation_ncoef"], 1))),
                "validation_mae": float(acc["validation_abs"] / max(acc["validation_ncoef"], 1)),
                "gamma_norm": acc["gamma_norm"],
                "spatial_field_rms": float(np.sqrt(acc["spatial_sum_sq"] / max(acc["spatial_n"], 1))),
                "Ske_min": float(acc["Ske_min"]),
                "Ske_median": float(np.median(ske_sample)) if ske_sample.size else np.nan,
                "Ske_max": float(acc["Ske_max"]),
                "Cu_global": float(acc["Cu_global"]),
                "lag_c": float(acc["lag_c"]),
                "lag_u": float(acc["lag_u"]),
                "artifact_status": "passed" if artifact else "failed",
                "physical_status": "passed" if physical else "failed",
                "checkpoint_hash": acc["checkpoint_hash"],
                "artifact_score": acc["artifact_score"],
            }
        )
    return rows


def recompute_development_checkpoints(root: Path, selected, transform, cache, mask, blocks) -> dict:
    stage_c = root / "model_compare" / "G0_no_geology_L0_shared" / "fold_00" / "stage_C"
    stage_a = json.loads((root / "model_compare/G0_no_geology_L0_shared/fold_00/stage_A/stage_A_fixed_lag_u_10d_result.json").read_text())
    stage_b_sel = json.loads((root / "model_compare/G0_no_geology_L0_shared/fold_00/stage_B/stage_B_lambda_selection_verified.json").read_text())
    global_prior = {
        "mean": np.array([np.log(stage_a["Ske_global"]), np.log(stage_a["Cu_global"]), stage_a["lag_c_days"]], dtype=float),
        "precision": np.array([1.0, 1.0, 1.0 / (365.2425**2)], dtype=float),
    }
    lam = float(stage_b_sel["verified_selected_lambda"])
    mask_hash = hash_file(mask)
    checkpoints = []
    meta_dir = stage_c / "checkpoint_metadata"
    for checkpoint in sorted(stage_c.glob("checkpoint_iter_*.npy")):
        iteration = int(checkpoint.stem.split("_")[-1])
        theta = np.load(checkpoint).astype(float)
        checkpoints.append({
            "checkpoint_iteration": iteration,
            "checkpoint": checkpoint,
            "theta": theta,
            "parameter_hash": hash_array(theta),
            "checkpoint_hash": hash_file(checkpoint),
        })
    rows = _fast_recompute_checkpoint_metrics(checkpoints, cache, mask, blocks, selected, transform, lam, global_prior)
    for row, item in zip(rows, checkpoints):
        write_json(
            meta_dir / f"{item['checkpoint'].stem}.metadata.json",
            checkpoint_metadata_payload(
                row["checkpoint_iteration"],
                item["theta"],
                row["raw_objective"],
                {"rmse": row["training_rmse"]},
                {"rmse": row["validation_rmse"]},
                selected,
                mask_hash,
                lam,
            ),
        )
    df = pd.DataFrame(rows).sort_values("checkpoint_iteration")
    nobs = int((stage_c / "objective_scaling_audit.json").exists() and json.loads((stage_c / "objective_scaling_audit.json").read_text()).get("effective_observation_count", 1)) or 1
    df["normalized_objective"] = df["raw_objective"] / float(nobs)
    df.to_csv(stage_c / "development_checkpoint_recomputed_metrics.csv", index=False)
    valid = df[(df["artifact_status"].eq("passed")) & (df["physical_status"].eq("passed"))].copy()
    best_rmse = float(valid["validation_rmse"].min())
    near = valid[valid["validation_rmse"] <= best_rmse * 1.001].sort_values("checkpoint_iteration")
    selected_row = near.iloc[0]
    selection = {
        "selected_iteration_budget": int(selected_row["checkpoint_iteration"]),
        "selected_checkpoint_hash": str(selected_row["checkpoint_hash"]),
        "selection_reason": "minimum recomputed checkpoint validation RMSE with 0.1 percent earlier-iteration tie rule; artifact and physical audits passed",
        "validation_rmse": float(selected_row["validation_rmse"]),
        "training_rmse": float(selected_row["training_rmse"]),
        "gamma_norm": float(selected_row["gamma_norm"]),
        "Ske_range": [float(selected_row["Ske_min"]), float(selected_row["Ske_max"])],
        "Cu": float(selected_row["Cu_global"]),
        "lag_c": float(selected_row["lag_c"]),
        "history_rmse_deprecated_due_to_off_by_one": True,
    }
    write_json(stage_c / "development_early_stopping_selection.json", selection)
    return selection


def dry_run(root: Path, selected, transform, cache, mask, blocks, budget: int) -> dict:
    fold_dir = root / "model_compare" / "G0_no_geology_L0_shared" / "fold_00"
    dry = fold_dir / "formal_protocol_dry_run"
    dry.mkdir(parents=True, exist_ok=True)
    counters = {"during_training": 0, "final": 0}
    t0 = time.time()
    stage_a = fixed_lagu_stage_a(stream_stage_a_stats(cache, mask, blocks, selected, transform))
    write_json(dry / "stage_A_training_only_result.json", stage_a)
    hess, rhs, _base_sse, _n = accumulate_quadratic(cache, mask, blocks, selected, transform, stage_a)
    lam = 30.0
    gamma = np.linalg.solve(hess + lam * np.eye(hess.shape[0]), rhs)
    np.save(dry / "stage_B_training_only_gamma.npy", gamma)
    stage_b_train = evaluate_gamma(cache, mask, blocks, selected, transform, stage_a, gamma, train=True)
    write_json(dry / "stage_B_training_only_result.json", {"stage_B_training_only": True, "lambda": lam, "gamma_norm": float(np.linalg.norm(gamma)), **stage_b_train})
    theta0 = np.r_[np.log(stage_a["Ske_global"]), gamma, np.log(stage_a["Cu_global"]), stage_a["lag_c_days"]].astype(float)
    global_prior = {
        "mean": np.array([theta0[0], theta0[33], theta0[34]], dtype=float),
        "precision": np.array([1.0, 1.0, 1.0 / (365.2425**2)], dtype=float),
    }
    mask_hash = hash_file(mask)
    history = []
    nfev = {"n": 0}
    previous = {"theta": theta0.copy()}

    def fun(theta):
        nfev["n"] += 1
        val, grad, _parts = objective_grad(theta, cache, mask, blocks, selected, transform, lam, global_prior)
        if not np.isfinite(val) or not np.isfinite(grad).all():
            raise FloatingPointError("non-finite objective or gradient")
        return val, grad

    def callback(theta):
        accepted_iteration = len(history) + 1
        val, grad, parts = objective_grad(theta, cache, mask, blocks, selected, transform, lam, global_prior)
        train = metrics(theta, cache, mask, blocks, selected, transform, train=True)
        checkpoint = dry / f"checkpoint_iter_{accepted_iteration:03d}.npy"
        np.save(checkpoint, theta)
        checkpoint_hash = hash_file(checkpoint)
        write_json(dry / f"checkpoint_iter_{accepted_iteration:03d}.metadata.json", checkpoint_metadata_payload(accepted_iteration, theta, val, train, None, selected, mask_hash, lam))
        row = parameter_row(accepted_iteration, theta, val, grad, parts, train, previous["theta"], nfev["n"], checkpoint, checkpoint_hash)
        history.append(row)
        pd.DataFrame(history).to_csv(dry / "training_only_optimizer_history.csv", index=False)
        previous["theta"] = theta.copy()

    result = minimize(
        fun,
        theta0,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        bounds=[(None, None)] + [(None, None)] * 32 + [(None, None), (0.0, 365.2425)],
        options={"maxiter": int(budget), "maxfun": max(80, int(budget) * 3), "maxls": 5, "ftol": 1e-8, "gtol": 1e-5},
    )
    theta = result.x.astype(float)
    if len(history) < int(budget):
        final_iteration = len(history) + 1
        val, grad, parts = objective_grad(theta, cache, mask, blocks, selected, transform, lam, global_prior)
        train = metrics(theta, cache, mask, blocks, selected, transform, train=True)
        checkpoint = dry / f"checkpoint_iter_{final_iteration:03d}.npy"
        np.save(checkpoint, theta)
        checkpoint_hash = hash_file(checkpoint)
        write_json(dry / f"checkpoint_iter_{final_iteration:03d}.metadata.json", checkpoint_metadata_payload(final_iteration, theta, val, train, None, selected, mask_hash, lam))
        history.append(parameter_row(final_iteration, theta, val, grad, parts, train, previous["theta"], nfev["n"], checkpoint, checkpoint_hash))
        pd.DataFrame(history).to_csv(dry / "training_only_optimizer_history.csv", index=False)
    final_checkpoint = dry / "final_training_checkpoint.npy"
    np.save(final_checkpoint, theta)
    counters["final"] += 1
    final_train = metrics(theta, cache, mask, blocks, selected, transform, train=True)
    final_valid = metrics(theta, cache, mask, blocks, selected, transform, train=False)
    log_ske, gamma_final, cu, lag_c = decode(theta)
    artifact = {
        **artifact_metrics(gamma_final, np.zeros_like(gamma_final), np.arange(gamma_final.size), gamma_final),
        "artifact_status": "passed" if np.max(np.abs(gamma_final)) / max(np.sqrt(np.mean(gamma_final * gamma_final)), 1e-12) < 6.0 else "failed",
    }
    physical = {
        "physical_status": "passed" if final_train["ske_min"] > 0 and cu > 0 and 0 <= lag_c <= 365.2425 else "failed",
        "Ske_min": final_train["ske_min"],
        "Ske_median": final_train["ske_median"],
        "Ske_max": final_train["ske_max"],
        "Cu_global": cu,
        "lag_c_days": lag_c,
        "lag_u_days": LAG_U_FIXED_DAYS,
    }
    checkpoint_alignment_passed = bool(pd.read_csv(dry / "training_only_optimizer_history.csv").apply(lambda r: (dry / r["checkpoint_filename"]).exists(), axis=1).all())
    fixed_budget_reached = len(history) == int(budget)
    no_numeric_failure = bool(np.isfinite(result.fun) and np.isfinite(theta).all())
    formal_passed = bool(
        checkpoint_alignment_passed
        and fixed_budget_reached
        and no_numeric_failure
        and physical["physical_status"] == "passed"
        and artifact["artifact_status"] == "passed"
        and counters["during_training"] == 0
        and counters["final"] == 1
    )
    audit = {
        "protocol_dry_run_on_development_fold": True,
        "checkpoint_alignment_passed": checkpoint_alignment_passed,
        "selected_iteration_budget": int(budget),
        "stage_A_training_only": True,
        "stage_B_training_only": True,
        "stage_C_training_only": True,
        "outer_validation_access_count_during_training": counters["during_training"],
        "outer_validation_access_count_final": counters["final"],
        "fixed_budget_reached": fixed_budget_reached,
        "optimizer_converged": bool(result.success),
        "optimizer_message": str(result.message),
        "final_parameter_hash": hash_array(theta),
        "final_training_checkpoint_hash": hash_file(final_checkpoint),
        "training_rmse": final_train["rmse"],
        "single_final_validation_rmse": final_valid["rmse"],
        "single_final_validation_mae": final_valid["mae"],
        "physical_audit": physical,
        "artifact_audit": artifact,
        "formal_fit_status": "formal_fit_complete_fixed_budget" if formal_passed else "formal_fit_failed",
        "formal_protocol_passed": formal_passed,
        "elapsed_seconds": time.time() - t0,
    }
    write_json(dry / "outer_validation_access_audit.json", {
        "outer_validation_access_count_during_training": counters["during_training"],
        "outer_validation_access_count_final": counters["final"],
    })
    write_json(dry / "single_final_outer_validation_metrics.json", final_valid)
    write_json(dry / "formal_fit_status.json", audit)
    write_json(root / "formal_protocol_dry_run_audit.json", audit)
    return audit


def update_status_and_config(root: Path, config: Path, selection: dict, dry_audit: dict | None) -> None:
    cfg = json.loads(config.read_text())
    sv = cfg.setdefault("spatial_validation", {})
    sv["formal_stage_c_iteration_budget"] = int(selection["selected_iteration_budget"])
    sv.pop("formal_maxiter_recommendation", None)
    config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    status_path = root / "aquifer_model_revision_status.json"
    status = json.loads(status_path.read_text())
    status.update({
        "formal_stage_c_iteration_budget": int(selection["selected_iteration_budget"]),
        "formal_maxiter_recommendation": None,
        "checkpoint_history_alignment": "recomputed_checkpoint_metrics_available",
        "stage_C_status": "development_early_stopped",
        "g0_fold0_status": "development_early_stopped",
        "allow_continue_g0_other_folds": False,
        "allow_continue_g0_fold1_pilot": bool(dry_audit and dry_audit.get("formal_protocol_passed")),
        "allow_continue_g0_fold2_fold4": False,
        "allow_continue_g1_g2_g3": False,
        "phase4_restart_allowed": False,
        "selected_model_config": "not_generated",
    })
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root)
    selected = json.loads((root / "selected_rbf_design.json").read_text())
    transform = np.load(root / "rbf_orthogonalization" / "rbf_transform.npy")
    from profiled_stage_a import latest_real_harmonic_cache

    cache = latest_real_harmonic_cache()
    mask = root / "comparison_common_mask.tif"
    blocks = root / "spatial_validation_blocks.tif"
    selection = recompute_development_checkpoints(root, selected, transform, cache, mask, blocks)
    dry_audit = None
    if args.run_dry_run:
        dry_audit = dry_run(root, selected, transform, cache, mask, blocks, int(selection["selected_iteration_budget"]))
    update_status_and_config(root, Path(args.config), selection, dry_audit)
    print(json.dumps({"selection": selection, "dry_run": dry_audit}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
