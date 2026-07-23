#!/usr/bin/env python
"""Resume the fold0 formal-protocol dry run from its latest dry-run checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiled_stage_a import latest_real_harmonic_cache
from scripts.audit_stage_b_lambda_effect import artifact_metrics
from scripts.formal_protocol_checkpoint_and_dry_run import hash_file
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


BASIS_HASH = "fb5d0531ebf865b5e375e928f6560794a532a975f501e83c3e4cdd1d60f5f9fd"
FORMAL_PROTOCOL_VERSION = "development_plus_four_outer_folds_fixed_budget_v1"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def hash_array(arr: np.ndarray) -> str:
    return sha256(np.asarray(arr, dtype="float64").tobytes()).hexdigest()


def checkpoint_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def split_hash(mask: Path, blocks: Path, fold_id: int, train: bool) -> str:
    h = sha256()
    h.update(mask.read_bytes())
    h.update(blocks.read_bytes())
    h.update(f"fold={fold_id};train={train}".encode())
    return h.hexdigest()


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "accepted_iteration" not in df.columns and "iteration" in df.columns:
        df = df.rename(columns={"iteration": "accepted_iteration"})
    drop = [c for c in ("validation_rmse_mm", "validation_mae_mm") if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
        df.to_csv(path, index=False)
    return df


def training_objective_grad_metrics(theta, cache, mask, blocks, selected, transform, lam, global_prior):
    log_ske, gamma, cu, lag_c = decode(theta)
    total = 0.0
    grad = np.zeros_like(theta)
    sse = 0.0
    ae = 0.0
    ncoef = 0
    spatial_sq = 0.0
    spatial_n = 0
    ske_values = []
    k = 2.0 * np.pi / 365.2425
    for obs, hc, hu, b, _ in iter_blocks(cache, mask, blocks, selected, transform, train=True):
        spatial = b @ gamma
        ske = np.exp(np.clip(log_ske + spatial, -20, 10))
        rc = rotate_coefficients(hc, lag_c, 365.2425)
        ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS, 365.2425)
        pred = 1000.0 * (ske[:, None] * rc + cu * ru)
        res = obs - pred
        sse_block = float(np.sum(res * res))
        sse += sse_block
        ae += float(np.sum(np.abs(res)))
        ncoef += int(res.size)
        total += 0.5 * sse_block / (5.0**2)
        common_factor = -1000.0 * ske * np.sum(res * rc, axis=1) / (5.0**2)
        grad[0] += float(np.sum(common_factor))
        grad[1:33] += b.T @ common_factor
        grad[33] += -float(np.sum(res * (1000.0 * cu * ru)) / (5.0**2))
        s0, c0 = hc[:, 0], hc[:, 1]
        angle = 2.0 * np.pi * lag_c / 365.2425
        ca, sa = np.cos(angle), np.sin(angle)
        drc = np.column_stack([(-s0 * sa + c0 * ca) * k, (-c0 * sa - s0 * ca) * k])
        grad[34] += -float(np.sum(res * (1000.0 * ske[:, None] * drc)) / (5.0**2))
        spatial_sq += float(np.sum(spatial * spatial))
        spatial_n += int(spatial.size)
        if sum(len(x) for x in ske_values) < 250_000:
            ske_values.append(ske[: max(0, 250_000 - sum(len(x) for x in ske_values))].astype(float))
    gamma_prior_unscaled = 0.5 * float(gamma @ gamma)
    gamma_prior_scaled = float(lam) * gamma_prior_unscaled
    total += gamma_prior_scaled
    grad[1:33] += float(lam) * gamma
    dglob = theta[[0, 33, 34]] - global_prior["mean"]
    global_prior_penalty = 0.5 * float(np.sum(global_prior["precision"] * dglob * dglob))
    total += global_prior_penalty
    grad[[0, 33, 34]] += global_prior["precision"] * dglob
    ske_sample = np.concatenate(ske_values) if ske_values else np.array([np.nan])
    parts = {
        "data_loss": float(total - gamma_prior_scaled - global_prior_penalty),
        "gamma_prior_penalty_unscaled": gamma_prior_unscaled,
        "gamma_prior_penalty_scaled": gamma_prior_scaled,
        "global_prior_penalty": global_prior_penalty,
        "total_objective": float(total),
    }
    train = {
        "rmse": float(np.sqrt(sse / max(ncoef, 1))),
        "mae": float(ae / max(ncoef, 1)),
        "gamma_norm": float(np.linalg.norm(gamma)),
        "spatial_field_rms": float(np.sqrt(spatial_sq / max(spatial_n, 1))),
        "ske_min": float(np.nanmin(ske_sample)),
        "ske_median": float(np.nanmedian(ske_sample)),
        "ske_max": float(np.nanmax(ske_sample)),
        "Cu_global": float(cu),
        "lag_c_days": float(lag_c),
        "lag_u_days": LAG_U_FIXED_DAYS,
        "effective_observation_count": int(ncoef),
    }
    return float(total), grad, parts, train


def build_history_row(iteration, theta, grad, parts, train, prev_theta, nfev, checkpoint, ck_hash):
    step = float(np.linalg.norm(theta - prev_theta)) if prev_theta is not None else np.nan
    rel_step = step / max(float(np.linalg.norm(prev_theta)), 1.0) if prev_theta is not None else np.nan
    norm_obj = parts["total_objective"] / max(train["effective_observation_count"], 1)
    return {
        "accepted_iteration": int(iteration),
        "training_objective": parts["total_objective"],
        "normalized_training_objective": norm_obj,
        "data_loss": parts["data_loss"],
        "gamma_prior_penalty_scaled": parts["gamma_prior_penalty_scaled"],
        "global_prior_penalty": parts["global_prior_penalty"],
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
        "callback_sequence": int(iteration),
        "scipy_nit": int(iteration),
        "function_evaluations": int(nfev),
    }


def checkpoint_metadata(iteration, theta, ck_hash, parts, train, selected, training_mask_hash, validation_mask_hash, lam, budget):
    return {
        "accepted_iteration": int(iteration),
        "parameter_hash": hash_array(theta),
        "checkpoint_hash": ck_hash,
        "training_objective": parts["total_objective"],
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
        "lambda": float(lam),
        "objective_version": OBJECTIVE_VERSION,
        "prior_version": PRIOR_VERSION,
        "formal_protocol_version": FORMAL_PROTOCOL_VERSION,
        "formal_iteration_budget": int(budget),
    }


def update_status(dry, status, start_time, completed, target, last_checkpoint, last_hash, train_access=0):
    elapsed = time.time() - start_time
    remaining = max(target - completed, 0)
    per_iter = elapsed / max(completed - status.get("_resume_start_iteration", 0), 1)
    payload = {
        "status": "running_expected_long_duration" if completed < target else "training_budget_completed_pending_final_validation",
        "accepted_iterations_completed": int(completed),
        "accepted_iterations_target": int(target),
        "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
        "last_checkpoint_hash": last_hash,
        "elapsed_seconds": float(elapsed),
        "estimated_remaining_seconds": float(per_iter * remaining),
        "outer_validation_access_count_during_training": int(train_access),
        "last_update_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(dry / "formal_protocol_dry_run_status.json", payload)


def write_final_status(dry, start_time, audit):
    write_json(
        dry / "formal_protocol_dry_run_status.json",
        {
            "status": "passed" if audit["formal_protocol_passed"] else audit["formal_fit_status"],
            "accepted_iterations_completed": int(audit["accepted_iterations_completed"]),
            "accepted_iterations_target": int(audit["accepted_iterations_target"]),
            "fixed_budget_completed": bool(audit["fixed_budget_completed"]),
            "last_checkpoint": str(dry / "final_training_checkpoint.npy"),
            "last_checkpoint_hash": audit["final_training_checkpoint_hash"],
            "elapsed_seconds": float(audit["total_elapsed_seconds"]),
            "estimated_remaining_seconds": 0.0,
            "outer_validation_access_count_during_training": int(audit["outer_validation_access_count_during_training"]),
            "outer_validation_access_count_final": int(audit["outer_validation_access_count_final"]),
            "last_update_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    args = parser.parse_args()
    root = Path(args.output_root)
    fold_dir = root / "model_compare" / "G0_no_geology_L0_shared" / "fold_00"
    dry = fold_dir / "formal_protocol_dry_run"
    history_path = dry / "training_only_optimizer_history.csv"
    selected = json.loads((root / "selected_rbf_design.json").read_text())
    if selected["basis_design_hash"] != BASIS_HASH:
        raise RuntimeError("RBF basis hash mismatch")
    transform = np.load(root / "rbf_orthogonalization" / "rbf_transform.npy")
    stage_a = json.loads((dry / "stage_A_training_only_result.json").read_text())
    gamma = np.load(dry / "stage_B_training_only_gamma.npy")
    budget = int(json.loads((fold_dir / "stage_C" / "development_early_stopping_selection.json").read_text())["selected_iteration_budget"])
    if budget != 40:
        raise RuntimeError(f"Unexpected formal iteration budget: {budget}")
    lam = 30.0
    cache = latest_real_harmonic_cache()
    mask = root / "comparison_common_mask.tif"
    blocks = root / "spatial_validation_blocks.tif"
    training_mask_hash = split_hash(mask, blocks, 0, True)
    validation_mask_hash = split_hash(mask, blocks, 0, False)
    mask_hash = hash_file(mask)
    history = load_history(history_path)
    completed = int(history["accepted_iteration"].max())
    checkpoint = dry / f"checkpoint_iter_{completed:03d}.npy"
    metadata_path = dry / f"checkpoint_iter_{completed:03d}.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    theta0 = np.load(checkpoint).astype(float)
    if metadata["accepted_iteration"] != completed:
        raise RuntimeError("Checkpoint metadata accepted_iteration mismatch")
    if metadata["parameter_hash"] != hash_array(theta0):
        raise RuntimeError("Checkpoint parameter hash mismatch")
    if str(history.iloc[-1]["parameter_hash"]) != hash_array(theta0):
        raise RuntimeError("History parameter hash mismatch")
    if str(history.iloc[-1]["checkpoint_hash"]) != checkpoint_hash(checkpoint):
        raise RuntimeError("History checkpoint hash mismatch")
    if metadata["basis_hash"] != BASIS_HASH or metadata["lambda"] != lam or metadata["lag_u"] != LAG_U_FIXED_DAYS:
        raise RuntimeError("Frozen dry-run metadata mismatch")
    if metadata.get("mask_hash") not in {None, mask_hash}:
        raise RuntimeError("Mask hash mismatch")
    theta_initial = np.r_[np.log(stage_a["Ske_global"]), gamma, np.log(stage_a["Cu_global"]), stage_a["lag_c_days"]].astype(float)
    global_prior = {
        "mean": theta_initial[[0, 33, 34]],
        "precision": np.array([1.0, 1.0, 1.0 / (365.2425**2)], dtype=float),
    }
    precheck = {
        "resumed_from_correct_checkpoint": True,
        "accepted_iteration": completed,
        "parameter_hash": hash_array(theta0),
        "checkpoint_hash": checkpoint_hash(checkpoint),
        "training_mask_hash": training_mask_hash,
        "validation_mask_hash": validation_mask_hash,
        "basis_hash": BASIS_HASH,
        "lambda": lam,
        "lag_u": LAG_U_FIXED_DAYS,
        "objective_version": OBJECTIVE_VERSION,
        "prior_version": PRIOR_VERSION,
        "formal_iteration_budget": budget,
    }
    write_json(dry / "resume_precheck.json", precheck)
    outer_validation_access_count_during_training = 0
    nfev = {"n": 0}

    def fun(theta):
        nfev["n"] += 1
        val, grad, _parts = objective_grad(theta, cache, mask, blocks, selected, transform, lam, global_prior)
        if not np.isfinite(val) or not np.isfinite(grad).all():
            raise FloatingPointError("non-finite training objective/gradient")
        return val, grad

    rows = history.to_dict(orient="records")
    previous = {"theta": theta0.copy()}
    start_time = time.time()
    status_hint = {"_resume_start_iteration": completed}

    def callback(theta):
        accepted = len(rows) + 1
        val, grad, parts, train = training_objective_grad_metrics(theta, cache, mask, blocks, selected, transform, lam, global_prior)
        if not np.isfinite(val) or not np.isfinite(theta).all():
            raise FloatingPointError("non-finite accepted parameter state")
        if train["ske_min"] <= 0 or train["Cu_global"] <= 0 or not (0 <= train["lag_c_days"] <= 365.2425):
            raise RuntimeError("physical hard failure")
        checkpoint_i = dry / f"checkpoint_iter_{accepted:03d}.npy"
        np.save(checkpoint_i, theta)
        ck_hash = checkpoint_hash(checkpoint_i)
        row = build_history_row(accepted, theta, grad, parts, train, previous["theta"], nfev["n"], checkpoint_i, ck_hash)
        rows.append(row)
        pd.DataFrame(rows).to_csv(history_path, index=False)
        write_json(
            dry / f"checkpoint_iter_{accepted:03d}.metadata.json",
            checkpoint_metadata(accepted, theta, ck_hash, parts, train, selected, training_mask_hash, validation_mask_hash, lam, budget),
        )
        previous["theta"] = theta.copy()
        update_status(dry, status_hint, start_time, accepted, budget, checkpoint_i, ck_hash, outer_validation_access_count_during_training)

    remaining = budget - completed
    result = minimize(
        fun,
        theta0,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        bounds=[(None, None)] + [(None, None)] * 32 + [(None, None), (0.0, 365.2425)],
        options={"maxiter": remaining, "maxfun": max(80, remaining * 4), "maxls": 10, "ftol": 0.0, "gtol": 0.0},
    )
    final_history = pd.read_csv(history_path)
    completed = int(final_history["accepted_iteration"].max())
    theta_final = np.load(dry / f"checkpoint_iter_{completed:03d}.npy").astype(float)
    final_checkpoint = dry / "final_training_checkpoint.npy"
    np.save(final_checkpoint, theta_final)
    final_checkpoint_hash = checkpoint_hash(final_checkpoint)
    final_obj, final_grad, final_parts, final_train = training_objective_grad_metrics(theta_final, cache, mask, blocks, selected, transform, lam, global_prior)
    fixed_budget_completed = completed == budget
    optimizer_message = str(result.message)
    line_search_failure = "LINE SEARCH" in optimizer_message.upper() or "ABNORMAL" in optimizer_message.upper()
    outer_validation_access_count_final = 0
    final_valid = None
    if fixed_budget_completed and not line_search_failure:
        final_valid = metrics(theta_final, cache, mask, blocks, selected, transform, train=False)
        outer_validation_access_count_final = 1
    gamma_final = decode(theta_final)[1]
    artifact_score = float(np.max(np.abs(gamma_final)) / max(np.sqrt(np.mean(gamma_final * gamma_final)), 1e-12))
    artifact = {
        **artifact_metrics(gamma_final, np.zeros_like(gamma_final), np.arange(gamma_final.size), gamma_final),
        "artifact_score": artifact_score,
        "artifact_status": "passed" if artifact_score < 6.0 else "failed",
    }
    physical = {
        "physical_status": "passed" if final_train["ske_min"] > 0 and final_train["Cu_global"] > 0 and 0 <= final_train["lag_c_days"] <= 365.2425 else "failed",
        "Ske_min": final_train["ske_min"],
        "Ske_median": final_train["ske_median"],
        "Ske_max": final_train["ske_max"],
        "Cu_global": final_train["Cu_global"],
        "lag_c_days": final_train["lag_c_days"],
        "lag_u_days": LAG_U_FIXED_DAYS,
    }
    checkpoint_alignment_passed = True
    for _, row in final_history.iterrows():
        ck = dry / row["checkpoint_filename"]
        checkpoint_alignment_passed = checkpoint_alignment_passed and ck.exists() and row["parameter_hash"] == hash_array(np.load(ck)) and row["checkpoint_hash"] == checkpoint_hash(ck)
    formal_passed = bool(
        fixed_budget_completed
        and not line_search_failure
        and checkpoint_alignment_passed
        and physical["physical_status"] == "passed"
        and artifact["artifact_status"] == "passed"
        and outer_validation_access_count_during_training == 0
        and outer_validation_access_count_final == 1
    )
    write_json(
        dry / "final_training_checkpoint_metadata.json",
        checkpoint_metadata(budget, theta_final, final_checkpoint_hash, final_parts, final_train, selected, training_mask_hash, validation_mask_hash, lam, budget),
    )
    write_json(
        dry / "outer_validation_access_audit.json",
        {
            "outer_validation_access_count_during_training": outer_validation_access_count_during_training,
            "outer_validation_access_count_final": outer_validation_access_count_final,
        },
    )
    if final_valid is not None:
        write_json(dry / "single_final_outer_validation_metrics.json", final_valid)
    audit = {
        "checkpoint_alignment_passed": checkpoint_alignment_passed,
        "accepted_iterations_completed": completed,
        "accepted_iterations_target": budget,
        "fixed_budget_completed": fixed_budget_completed,
        "outer_validation_access_count_during_training": outer_validation_access_count_during_training,
        "outer_validation_access_count_final": outer_validation_access_count_final,
        "training_rmse": final_train["rmse"],
        "single_final_validation_rmse": None if final_valid is None else final_valid["rmse"],
        "single_final_validation_mae": None if final_valid is None else final_valid["mae"],
        "optimizer_success": bool(result.success),
        "optimizer_message": optimizer_message,
        "line_search_failure": line_search_failure,
        "physical_audit": physical,
        "artifact_audit": artifact,
        "formal_fit_status": "formal_fit_complete_fixed_budget" if formal_passed else ("formal_fit_failed_checkpoint_alignment" if not checkpoint_alignment_passed else "formal_fit_failed_numerical" if line_search_failure else "formal_fit_incomplete_fixed_budget_not_reached"),
        "formal_protocol_passed": formal_passed,
        "resumed_from_correct_checkpoint": True,
        "resume_accepted_iteration": int(precheck["accepted_iteration"]),
        "total_elapsed_seconds": time.time() - start_time,
        "final_training_checkpoint_hash": final_checkpoint_hash,
    }
    write_json(dry / "formal_fit_status.json", audit)
    write_json(root / "formal_protocol_dry_run_audit.json", audit)
    status_path = root / "aquifer_model_revision_status.json"
    status = json.loads(status_path.read_text())
    status.update(
        {
            "formal_protocol_dry_run": "passed" if formal_passed else audit["formal_fit_status"],
            "allow_continue_g0_fold1_pilot": formal_passed,
            "allow_continue_g0_other_folds": False,
            "allow_continue_g0_fold2_fold4": False,
            "allow_continue_g1_g2_g3": False,
            "allow_lag_c_model_comparison": False,
            "selected_model_config": "not_generated",
            "phase4_restart_allowed": False,
        }
    )
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    write_final_status(dry, start_time, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
