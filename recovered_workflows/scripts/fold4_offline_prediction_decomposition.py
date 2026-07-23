#!/usr/bin/env python
"""Offline decomposition of fold4 prediction explosion using forensic replay HDF5 only."""
from __future__ import annotations

import csv
import json
import sys
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_stage_b_fixed_lagu import rbf_values
from scripts.run_stage_c_fixed_lagu import LAG_U_FIXED_DAYS, decode


ROOT = Path("outputs/aquifer_model_revision")
FOLD = ROOT / "model_compare/G0_no_geology_L0_shared/fold_04"
REPLAY = FOLD / "forensic_replay_01"
PIXELS = REPLAY / "fold4_forensic_pixels.h5"
TARGET_BLOCKS = {24, 34, 23, 33}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def q(arr, qs=(50, 95, 99, 100)) -> dict:
    arr = np.asarray(arr, dtype=float)
    return {f"p{int(v)}" if v != 50 else "median": float(np.percentile(arr, v)) for v in qs}


def stat_pack(arr, prefix, include_min=False, include_rms=False) -> dict:
    arr = np.asarray(arr, dtype=float)
    out = {}
    if include_min:
        out[f"{prefix}_min"] = float(np.min(arr))
    out[f"{prefix}_median"] = float(np.median(arr))
    out[f"{prefix}_p95"] = float(np.percentile(arr, 95))
    if prefix not in {"confined_head_amplitude", "unconfined_contribution_amplitude", "distance_to_training_region"}:
        out[f"{prefix}_p99"] = float(np.percentile(arr, 99))
    out[f"{prefix}_max"] = float(np.max(arr))
    if include_rms:
        out[f"{prefix}_RMS"] = float(np.sqrt(np.mean(arr * arr)))
    return out


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def infer_normal_blocks(block_table: pd.DataFrame, n: int = 4) -> set[int]:
    good = block_table[~block_table["block_id"].isin(TARGET_BLOCKS)].sort_values("residual_RMSE")
    return set(good["block_id"].head(n).astype(int).tolist())


def main() -> None:
    status = read_json(ROOT / "aquifer_model_revision_status.json")
    status.update({
        "fold4_root_cause_status": "offline_prediction_decomposition_running",
        "root_cause_current": "probable_block_local_prediction_basis_explosion",
        "G0_model_selection_eligible": False,
        "allow_start_G1_G2_G3": False,
        "allow_start_G1": False,
        "allow_start_G2": False,
        "allow_start_G3": False,
        "phase4_restart_allowed": False,
    })
    write_json(ROOT / "aquifer_model_revision_status.json", status)
    selected = read_json(ROOT / "selected_rbf_design.json")
    centers = np.asarray(selected["center_coordinates"], dtype=float)
    sigma_m = float(selected["sigma_km"]) * 1000.0
    transform = np.load(ROOT / "rbf_orthogonalization/rbf_transform.npy")
    theta = np.load(FOLD / "final_training_checkpoint.npy").astype(float)
    log_ske, gamma, cu, lag_c = decode(theta)
    eig = pd.read_csv(ROOT / "rbf_orthogonalization/rbf_eigenvalues.csv")
    block_table = pd.read_csv(REPLAY / "fold4_error_by_source_block_forensic.csv")
    normal_blocks = infer_normal_blocks(block_table)
    keep_blocks = TARGET_BLOCKS | normal_blocks
    rows_component = []
    rows_raw = []
    rows_basis = []
    rows_log = []
    basis_acc = {
        "all": {"sum_b2": np.zeros(32), "max_abs_b": np.zeros(32), "sum_bg2": np.zeros(32), "max_abs_bg": np.zeros(32), "n": 0},
    }
    for block in keep_blocks | set(block_table["block_id"].astype(int).tolist()):
        basis_acc[int(block)] = {"sum_b2": np.zeros(32), "max_abs_b": np.zeros(32), "sum_bg2": np.zeros(32), "max_abs_bg": np.zeros(32), "n": 0}
    direct_diff_sse = direct_diff_max = stored_pred_sse = 0.0
    ncoef = 0
    ske_diff_max = ske_diff_sse = 0.0
    phi_bad = False
    block_payload = {}
    with h5py.File(PIXELS, "r") as h5:
        block_ids = h5["source_block_id"][:].astype(int)
        unique_blocks = np.unique(block_ids)
        ordering_hash_observation = h5.attrs.get("ordering_hash_observation")
        ordering_hash_prediction = h5.attrs.get("ordering_hash_prediction")
        for block_id in unique_blocks:
            idx = np.where(block_ids == block_id)[0]
            x = h5["x"][idx]
            y = h5["y"][idx]
            points = np.column_stack([x, y])
            phi = rbf_values(points, centers, sigma_m)
            b = phi @ transform
            eta = log_ske + b @ gamma
            ske_direct = np.exp(np.clip(eta, -20, 10))
            ske_stored = h5["predicted_Ske"][idx]
            confined = np.column_stack([h5["confined_contribution_real"][idx], h5["confined_contribution_imag"][idx]])
            unconf = np.column_stack([h5["unconfined_contribution_real"][idx], h5["unconfined_contribution_imag"][idx]])
            pred_stored = np.column_stack([h5["prediction_real"][idx], h5["prediction_imag"][idx]])
            pred_direct = confined + unconf
            obs_amp = h5["observation_amplitude"][idx]
            pred_amp = h5["prediction_amplitude"][idx]
            conf_amp = np.linalg.norm(confined, axis=1)
            unconf_amp = np.linalg.norm(unconf, axis=1)
            confined_head_amp = conf_amp / np.maximum(1000.0 * ske_stored, 1e-30)
            basis_norm = np.sqrt(np.sum(b * b, axis=1))
            raw_norm = np.sqrt(np.sum(phi * phi, axis=1))
            amp_ratio = basis_norm / np.maximum(raw_norm, 1e-30)
            dist_train = h5["distance_to_training_region"][idx]
            stored_ske_diff = ske_direct - ske_stored
            ske_diff_max = max(ske_diff_max, float(np.max(np.abs(stored_ske_diff))))
            ske_diff_sse += float(np.sum(stored_ske_diff * stored_ske_diff))
            pdiff = pred_direct - pred_stored
            direct_diff_max = max(direct_diff_max, float(np.max(np.abs(pdiff))))
            direct_diff_sse += float(np.sum(pdiff * pdiff))
            stored_pred_sse += float(np.sum(pred_stored * pred_stored))
            ncoef += int(pdiff.size)
            phi_bad = phi_bad or bool((phi < -1e-12).any() or (phi > 1 + 1e-12).any() or ~np.isfinite(phi).all())
            if block_id in keep_blocks:
                rows_component.append({
                    "block_id": int(block_id),
                    "block_role": "target_extreme" if block_id in TARGET_BLOCKS else "normal_control",
                    "pixel_count": int(len(idx)),
                    **stat_pack(obs_amp, "observation_amplitude", include_rms=True),
                    **stat_pack(pred_amp, "prediction_amplitude", include_rms=True),
                    **stat_pack(ske_stored, "predicted_Ske", include_min=True),
                    **stat_pack(confined_head_amp, "confined_head_amplitude"),
                    **stat_pack(conf_amp, "confined_contribution_amplitude", include_rms=True),
                    **stat_pack(unconf_amp, "unconfined_contribution_amplitude", include_rms=True),
                    **stat_pack(basis_norm, "orthogonal_basis_row_norm"),
                    **stat_pack(dist_train, "distance_to_training_region"),
                })
                rows_raw.append({
                    "block_id": int(block_id),
                    "block_role": "target_extreme" if block_id in TARGET_BLOCKS else "normal_control",
                    "pixel_count": int(len(idx)),
                    "raw_phi_row_norm_median": float(np.median(raw_norm)),
                    "raw_phi_row_norm_p95": float(np.percentile(raw_norm, 95)),
                    "raw_phi_row_norm_max": float(np.max(raw_norm)),
                    "raw_phi_max": float(np.max(phi)),
                    "raw_phi_min": float(np.min(phi)),
                    "raw_phi_sum_median": float(np.median(np.sum(phi, axis=1))),
                    "raw_phi_sum_p95": float(np.percentile(np.sum(phi, axis=1), 95)),
                    "nearest_center_distance_m_median": float(np.median(h5["nearest_rbf_center_distance"][idx])),
                    "nearest_center_distance_m_p95": float(np.percentile(h5["nearest_rbf_center_distance"][idx], 95)),
                    "coordinate_x_median": float(np.median(x)),
                    "coordinate_y_median": float(np.median(y)),
                    "sigma_m": sigma_m,
                    "raw_phi_valid_0_1_finite": bool((phi >= -1e-12).all() and (phi <= 1 + 1e-12).all() and np.isfinite(phi).all()),
                })
                rows_basis.append({
                    "block_id": int(block_id),
                    "block_role": "target_extreme" if block_id in TARGET_BLOCKS else "normal_control",
                    "pixel_count": int(len(idx)),
                    "B_row_norm_median": float(np.median(basis_norm)),
                    "B_row_norm_p95": float(np.percentile(basis_norm, 95)),
                    "B_row_norm_p99": float(np.percentile(basis_norm, 99)),
                    "B_row_norm_max": float(np.max(basis_norm)),
                    "B_max_abs": float(np.max(np.abs(b))),
                    "B_p95_max_abs_per_row": float(np.percentile(np.max(np.abs(b), axis=1), 95)),
                    "B_leverage_median": float(np.median(basis_norm * basis_norm)),
                    "B_leverage_p95": float(np.percentile(basis_norm * basis_norm, 95)),
                    "orthogonal_amplification_ratio_median": float(np.median(amp_ratio)),
                    "orthogonal_amplification_ratio_p95": float(np.percentile(amp_ratio, 95)),
                    "orthogonal_amplification_ratio_max": float(np.max(amp_ratio)),
                })
                rows_log.append({
                    "block_id": int(block_id),
                    "block_role": "target_extreme" if block_id in TARGET_BLOCKS else "normal_control",
                    "pixel_count": int(len(idx)),
                    "eta_min": float(np.min(eta)),
                    "eta_median": float(np.median(eta)),
                    "eta_p95": float(np.percentile(eta, 95)),
                    "eta_p99": float(np.percentile(eta, 99)),
                    "eta_max": float(np.max(eta)),
                    "Ske_min": float(np.min(ske_stored)),
                    "Ske_median": float(np.median(ske_stored)),
                    "Ske_p95": float(np.percentile(ske_stored, 95)),
                    "Ske_p99": float(np.percentile(ske_stored, 99)),
                    "Ske_max": float(np.max(ske_stored)),
                    "Ske_direct_stored_max_abs_diff": float(np.max(np.abs(stored_ske_diff))),
                    "logSke_out_of_domain_extrapolation": bool(np.percentile(ske_stored, 95) > 0.05 or np.max(eta) > 0.0),
                })
            accs = [basis_acc["all"], basis_acc[int(block_id)]]
            for acc in accs:
                acc["sum_b2"] += np.sum(b * b, axis=0)
                acc["max_abs_b"] = np.maximum(acc["max_abs_b"], np.max(np.abs(b), axis=0))
                bg = b * gamma[None, :]
                acc["sum_bg2"] += np.sum(bg * bg, axis=0)
                acc["max_abs_bg"] = np.maximum(acc["max_abs_bg"], np.max(np.abs(bg), axis=0))
                acc["n"] += int(len(idx))
            block_payload[int(block_id)] = {
                "eta_median": float(np.median(eta)),
                "eta_max": float(np.max(eta)),
                "Ske_p95": float(np.percentile(ske_stored, 95)),
                "Ske_max": float(np.max(ske_stored)),
                "B_norm_p95": float(np.percentile(basis_norm, 95)),
                "prediction_amp_p95": float(np.percentile(pred_amp, 95)),
            }
        ske_all = h5["predicted_Ske"][:]
        validation_ske_physical = {
            "Ske_min": float(np.min(ske_all)),
            "Ske_median": float(np.median(ske_all)),
            "Ske_p95": float(np.percentile(ske_all, 95)),
            "Ske_p99": float(np.percentile(ske_all, 99)),
            "Ske_max": float(np.max(ske_all)),
            "fraction_gt_0_01": float(np.mean(ske_all > 0.01)),
            "fraction_gt_0_05": float(np.mean(ske_all > 0.05)),
            "fraction_gt_0_1": float(np.mean(ske_all > 0.1)),
            "fraction_gt_1": float(np.mean(ske_all > 1.0)),
            "fraction_nonfinite": float(np.mean(~np.isfinite(ske_all))),
            "by_block": block_payload,
            "validation_side_physical_extrapolation_failure": bool(np.mean(ske_all > 0.05) > 0 or max(v["Ske_max"] for v in block_payload.values()) > 0.1),
            "no_clipping_applied": True,
        }
    append_csv(REPLAY / "fold4_prediction_component_by_block.csv", rows_component)
    append_csv(REPLAY / "fold4_raw_rbf_by_block.csv", rows_raw)
    append_csv(REPLAY / "fold4_orthogonal_basis_by_block.csv", rows_basis)
    append_csv(REPLAY / "fold4_logSke_extrapolation_by_block.csv", rows_log)
    top_rows = []
    for key, acc in basis_acc.items():
        if key not in {"all", 24}:
            continue
        n = max(acc["n"], 1)
        for j in range(32):
            ev = float(eig.loc[eig["direction"] == j, "eigenvalue"].iloc[0])
            top_rows.append({
                "scope": str(key),
                "basis_index": j,
                "eigenvalue": ev,
                "inverse_sqrt_eigenvalue": float(1.0 / np.sqrt(ev)),
                "transform_column_norm": float(np.linalg.norm(transform[:, j])),
                "gamma": float(gamma[j]),
                "B_RMS": float(np.sqrt(acc["sum_b2"][j] / n)),
                "B_max_abs": float(acc["max_abs_b"][j]),
                "B_gamma_RMS": float(np.sqrt(acc["sum_bg2"][j] / n)),
                "B_gamma_max": float(acc["max_abs_bg"][j]),
            })
    top_rows.sort(key=lambda r: (r["scope"] != "24", -r["B_gamma_RMS"]))
    append_csv(REPLAY / "fold4_top_explosive_basis_directions.csv", top_rows)
    write_json(REPLAY / "fold4_validation_Ske_physical_audit.json", validation_ske_physical)
    pred_recon = {
        "max_absolute_difference": direct_diff_max,
        "RMS_difference": float(np.sqrt(direct_diff_sse / max(ncoef, 1))),
        "relative_RMS_difference": float(np.sqrt(direct_diff_sse / max(stored_pred_sse, 1e-30))),
        "parameter_hash": read_json(FOLD / "final_training_checkpoint_metadata.json")["parameter_hash"],
        "basis_hash": selected["basis_design_hash"],
        "direct_prediction_reproduces_stored_prediction": bool(direct_diff_max < 1e-10),
        "used_saved_contributions_only": True,
        "validation_source_reaccessed": False,
    }
    write_json(REPLAY / "fold4_prediction_independent_reconstruction.json", pred_recon)
    log_check = {
        "Ske_direct_from_eta_vs_stored_max_abs_difference": ske_diff_max,
        "Ske_direct_from_eta_vs_stored_RMS_difference": float(np.sqrt(ske_diff_sse / max(sum(r["pixel_count"] for r in rows_log), 1))),
        "prediction_parameter_transform_bug": bool(ske_diff_max > 1e-10),
        "logSke_out_of_domain_extrapolation": bool(any(r["logSke_out_of_domain_extrapolation"] for r in rows_log if r["block_id"] in TARGET_BLOCKS)),
    }
    train_meta = read_json(FOLD / "final_training_checkpoint_metadata.json")
    block24 = block_payload[24]
    train_compare = {
        "training_support_source": "final_training_checkpoint_metadata_only_no_training_cache_reaccess",
        "eta_training_quantiles": "unavailable_without_rereading_training_pixels",
        "B_row_norm_training_quantiles": "unavailable_without_rereading_training_pixels",
        "Ske_training_min": train_meta["Ske_min"],
        "Ske_training_median": train_meta["Ske_median"],
        "Ske_training_max": train_meta["Ske_max"],
        "block24_eta_max": block24["eta_max"],
        "block24_Ske_max": block24["Ske_max"],
        "Ske_extrapolation_ratio_vs_training_max": float(block24["Ske_max"] / max(train_meta["Ske_max"], 1e-30)),
        "prediction_amplitude_extrapolation_ratio_vs_observation_scale": float(block24["prediction_amp_p95"] / 18.677954806769126),
        "basis_norm_extrapolation_ratio": "unavailable_without_training_basis_norm_quantiles",
        "training_cache_reaccessed": False,
    }
    write_json(REPLAY / "fold4_training_vs_block24_extrapolation.json", train_compare)
    # Root-cause logic.
    raw_phi_ok = not phi_bad and all(r["raw_phi_valid_0_1_finite"] for r in rows_raw)
    block24_basis = next(r for r in rows_basis if r["block_id"] == 24)
    normal_basis_p95 = np.median([r["B_row_norm_p95"] for r in rows_basis if r["block_role"] == "normal_control"])
    block24_log = next(r for r in rows_log if r["block_id"] == 24)
    comp24 = next(r for r in rows_component if r["block_id"] == 24)
    head_normal = comp24["confined_head_amplitude_p95"] < 50.0
    if not pred_recon["direct_prediction_reproduces_stored_prediction"]:
        root = "confirmed_prediction_implementation_error"
    elif not raw_phi_ok:
        root = "confirmed_raw_rbf_coordinate_scale_error"
    elif block24_basis["orthogonal_amplification_ratio_p95"] > 50 and block24_basis["B_row_norm_p95"] > 5 * max(normal_basis_p95, 1e-30):
        root = "confirmed_orthogonal_basis_out_of_support_amplification"
    elif block24_log["Ske_p95"] > 0.05 and head_normal:
        root = "confirmed_logSke_out_of_domain_extrapolation"
    elif not head_normal:
        root = "confirmed_confined_head_block_scaling_error"
    else:
        root = "probable_basis_extrapolation_failure"
    recommendation = {
        "confirmed_prediction_implementation_error": "fix_predictor_recompute_fold1_to_fold4_metrics_from_frozen_checkpoints_no_retraining",
        "confirmed_raw_rbf_coordinate_scale_error": "fix_generic_prediction_pipeline_invalidate_all_affected_formal_metrics_determine_training_impact",
        "confirmed_orthogonal_basis_out_of_support_amplification": "new_basis_version_new_manifest_retrain_all_formal_folds_if_training_used_same_basis",
        "confirmed_logSke_out_of_domain_extrapolation": "retain_fold4_result_mark_G0_spatially_unstable_allow_candidate_model_comparison_after_user_review",
        "confirmed_confined_head_block_scaling_error": "repair_input_create_new_manifest_do_not_mix_old_new_results",
        "probable_basis_extrapolation_failure": "continue_blocking_until_user_review_or_training_support_audit",
    }[root]
    write_json(REPLAY / "fold4_prediction_explosion_root_cause.json", {
        "root_cause": root,
        "recommended_action": recommendation,
        "recommended_action_executed": False,
        "evidence": {
            "raw_phi_ok": raw_phi_ok,
            "block24_B_row_norm_p95": block24_basis["B_row_norm_p95"],
            "normal_control_B_row_norm_p95_median": float(normal_basis_p95),
            "block24_orthogonal_amplification_ratio_p95": block24_basis["orthogonal_amplification_ratio_p95"],
            "block24_eta_range": {k: block24_log[k] for k in ["eta_min", "eta_median", "eta_p95", "eta_p99", "eta_max"]},
            "block24_Ske_range": {k: block24_log[k] for k in ["Ske_min", "Ske_median", "Ske_p95", "Ske_p99", "Ske_max"]},
            "direct_prediction_reproduces_stored_prediction": pred_recon["direct_prediction_reproduces_stored_prediction"],
            "confined_head_amplitude_p95": comp24["confined_head_amplitude_p95"],
            "validation_source_reaccessed": False,
            "optimizer_called": False,
        },
        "whether_training_was_affected": "unknown_without_training_support_detail",
        "whether_retraining_is_required": "depends_on_new_basis_or_training_impact_review",
        "whether_only_metric_recomputation_is_required": root == "confirmed_prediction_implementation_error",
        "allow_candidate_model_comparison": False,
    })
    status = read_json(ROOT / "aquifer_model_revision_status.json")
    status.update({
        "fold4_root_cause_status": "offline_prediction_decomposition_complete",
        "root_cause_current": root,
        "G0_model_selection_eligible": False,
        "allow_start_G1_G2_G3": False,
        "allow_start_G1": False,
        "allow_start_G2": False,
        "allow_start_G3": False,
        "phase4_restart_allowed": False,
    })
    write_json(ROOT / "aquifer_model_revision_status.json", status)
    print(json.dumps({
        "root_cause": root,
        "raw_phi_ok": raw_phi_ok,
        "block24_B_row_norm_p95": block24_basis["B_row_norm_p95"],
        "block24_amplification_ratio_p95": block24_basis["orthogonal_amplification_ratio_p95"],
        "block24_Ske_p95": block24_log["Ske_p95"],
        "direct_prediction_reproduces_stored_prediction": pred_recon["direct_prediction_reproduces_stored_prediction"],
        "allow_candidate_model_comparison": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
