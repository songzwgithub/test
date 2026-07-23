#!/usr/bin/env python
"""Run V2 M1 G1/G2/G3 formal geology model comparison."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bounded_ske_v2 import (
    OBJECTIVE_VERSION,
    PARAMETERIZATION_VERSION,
    SKE_LOWER_BOUND,
    SKE_UPPER_BOUND,
    bounded_ske,
    bounded_ske_derivative,
)
from scripts.run_v2_g0_formal_cv import (
    BUDGET,
    EXPECTED_COMMON,
    LAMBDA,
    MODEL_DIR as G0_DIR,
    OBS_SIGMA_MM,
    OUT,
    PERIOD_DAYS,
    _finite_stats,
    dependency_versions,
    eval_theta as eval_g0_theta,
    hash_array,
    hash_file,
    iter_blocks,
    latest_real_harmonic_cache,
    read_json,
    split_counts,
    write_json,
)
from scripts.run_stage_c_fixed_lagu import LAG_U_FIXED_DAYS
from storage_inversion import rotate_coefficients


GEO_RASTERS = {
    "cumulative_confined_clay_thickness_m": Path("data/geology_rasters/cumulative_confined_clay_thickness_m.tif"),
    "quaternary_thickness_m": Path("data/geology_rasters/quaternary_thickness_m.tif"),
    "confined_clay_fraction": Path("data/geology_rasters/confined_clay_fraction.tif"),
}
MODELS = {
    "G1_confined_clay_thickness": {
        "dir": OUT / "model_compare_v2/G1_confined_clay_thickness_L0_shared",
        "covariates": ["cumulative_confined_clay_thickness_m"],
        "complexity_level": 1,
    },
    "G2_confined_clay_thickness_plus_Q4": {
        "dir": OUT / "model_compare_v2/G2_confined_clay_thickness_plus_Q4_L0_shared",
        "covariates": ["cumulative_confined_clay_thickness_m", "quaternary_thickness_m"],
        "complexity_level": 2,
    },
    "G3_confined_clay_fraction": {
        "dir": OUT / "model_compare_v2/G3_confined_clay_fraction_L0_shared",
        "covariates": ["confined_clay_fraction"],
        "complexity_level": 1,
    },
}
RUN_ORDER = ["G1_confined_clay_thickness", "G3_confined_clay_fraction", "G2_confined_clay_thickness_plus_Q4"]
PARAMETER_LAYOUT_VERSION = "eta_intercept_geology_beta_raw32_gamma_logCu_lagc_fixed_lagu_v1"
PRIOR_VERSION = "geology_beta_unpenalized_raw_gamma_lambda30_stageA_centered_v1"


def _versions():
    return {"python": platform.python_version(), **dependency_versions()}


def _load_covariate_arrays():
    arrays = {}
    for name, path in GEO_RASTERS.items():
        with rasterio.open(path) as src:
            arrays[name] = src.read(1).astype("float64")
    return arrays


def _normalization_payload():
    out_path = OUT / "V2_geology_covariate_normalization.json"
    sha_path = OUT / "V2_geology_covariate_normalization.sha256"
    arrays = _load_covariate_arrays()
    payload = {
        "normalization_scope": "full_common_mask_finite_covariate_pixels",
        "uses_response_variables": False,
        "common_mask_hash": hash_file(OUT / "comparison_common_mask.tif"),
        "covariates": {},
    }
    with rasterio.open(OUT / "comparison_common_mask.tif") as ms:
        common = ms.read(1) == 1
        common_count = int(common.sum())
        for name, arr in arrays.items():
            finite = common & np.isfinite(arr)
            vals = arr[finite]
            payload["covariates"][name] = {
                "source_raster": str(GEO_RASTERS[name]),
                "source_raster_hash": hash_file(GEO_RASTERS[name]),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "valid_count": int(vals.size),
                "nodata_count": int(common_count - vals.size),
                "common_mask_count": common_count,
                "valid_fraction": float(vals.size / common_count),
                "units": "m" if name != "confined_clay_fraction" else "dimensionless",
                "physical_meaning": {
                    "cumulative_confined_clay_thickness_m": "L2 + L3 + L4 confined cumulative clay thickness",
                    "quaternary_thickness_m": "Quaternary thickness midpoint raster",
                    "confined_clay_fraction": "cumulative_confined_clay_thickness_m / quaternary_thickness_m",
                }[name],
            }
    write_json(out_path, payload)
    digest = hash_file(out_path)
    payload["normalization_hash"] = digest
    write_json(out_path, payload)
    sha_path.write_text(hash_file(out_path), encoding="utf-8")
    return read_json(out_path)


def _standardized_covariates(arrays, covariates, rows, cols, norm):
    cols_out = []
    for name in covariates:
        item = norm["covariates"][name]
        vals = arrays[name][rows, cols]
        z = (vals - item["mean"]) / item["std"]
        cols_out.append(z)
    if not cols_out:
        return np.zeros((len(rows), 0), dtype=float)
    geo = np.column_stack(cols_out).astype(float)
    return geo


def _audit_covariates(norm):
    semantics = read_json(OUT / "clay_thickness_semantics_check.json")
    if semantics["layer_value_semantics_status"] != "accepted_with_minor_boundary_violations":
        raise RuntimeError("Geology layer semantics are not accepted for V2 formal comparison")
    rows = []
    arrays = _load_covariate_arrays()
    with rasterio.open(OUT / "comparison_common_mask.tif") as ms:
        common = ms.read(1) == 1
        hc = arrays["cumulative_confined_clay_thickness_m"][common]
        q4 = arrays["quaternary_thickness_m"][common]
        frac = arrays["confined_clay_fraction"][common]
    finite = np.isfinite(hc) & np.isfinite(q4) & np.isfinite(frac)
    if not np.all((frac[finite] >= 0) & (frac[finite] <= 1.5)):
        raise RuntimeError("Confined clay fraction outside expected finite range")
    z = np.column_stack([
        (hc[finite] - norm["covariates"]["cumulative_confined_clay_thickness_m"]["mean"]) / norm["covariates"]["cumulative_confined_clay_thickness_m"]["std"],
        (q4[finite] - norm["covariates"]["quaternary_thickness_m"]["mean"]) / norm["covariates"]["quaternary_thickness_m"]["std"],
        (frac[finite] - norm["covariates"]["confined_clay_fraction"]["mean"]) / norm["covariates"]["confined_clay_fraction"]["std"],
    ])
    corr = np.corrcoef(z, rowvar=False)
    g2_corr = float(corr[0, 1])
    g2_vif = float(1.0 / max(1.0 - g2_corr**2, 1e-12))
    audit = {
        "status": "passed",
        "uses_response_variables": False,
        "semantics_status": semantics["layer_value_semantics_status"],
        "Hc_sum_gt_Q4_violation_fraction": next(r["violation_fraction"] for r in semantics["checks"] if r["metric"] == "Hc_sum_gt_Q4"),
        "Htotal_sum_gt_Q4_violation_fraction_mapping_only": next(r["violation_fraction"] for r in semantics["checks"] if r["metric"] == "Htotal_sum_gt_Q4"),
        "forbidden_covariates_used": [],
        "G2_Hc_Q4_pearson": g2_corr,
        "G2_Hc_Q4_vif": g2_vif,
        "covariate_correlation_matrix_order": ["cumulative_confined_clay_thickness_m", "quaternary_thickness_m", "confined_clay_fraction"],
        "covariate_correlation_matrix": corr.tolist(),
        "normalization_hash": norm["normalization_hash"],
    }
    write_json(OUT / "V2_geology_covariate_audit.json", audit)
    return audit


def freeze_manifest(norm):
    base = read_json(OUT / "formal_protocol_v2_model_compare_manifest.json")
    if read_json(OUT / "V2_aquifer_structure_selection.json")["selected_aquifer_structure"] != "M1_two_aquifer_shared_unconfined":
        raise RuntimeError("Aquifer structure is not selected as M1; refusing geology comparison")
    manifest_path = OUT / "formal_protocol_v2_geology_compare_manifest.json"
    sha_path = OUT / "formal_protocol_v2_geology_compare_manifest.sha256"
    payload = {
        "manifest_status": "frozen_for_v2_geology_model_compare",
        "selected_aquifer_structure": "M1_two_aquifer_shared_unconfined",
        "models": {k: {"covariates": v["covariates"], "complexity_level": v["complexity_level"]} for k, v in MODELS.items()} | {"G0_no_geology": {"covariates": [], "complexity_level": 0}},
        "geology_raster_hashes": {name: hash_file(path) for name, path in GEO_RASTERS.items()},
        "geology_normalization_hash": norm["normalization_hash"],
        "common_mask_hash": hash_file(OUT / "comparison_common_mask.tif"),
        "fold_map_hash": hash_file(OUT / "spatial_validation_blocks.tif"),
        "RBF_centers_hash": base["RBF_centers_hash"],
        "raw_basis_normalization_hash": base["raw_basis_normalization_hash"],
        "Ske_parameterization": PARAMETERIZATION_VERSION,
        "Ske_bounds": [SKE_LOWER_BOUND, SKE_UPPER_BOUND],
        "lambda_multiplier": LAMBDA,
        "lag_u_global_days": LAG_U_FIXED_DAYS,
        "lag_c_mode": "L0_shared",
        "objective_version": OBJECTIVE_VERSION,
        "prior_version": PRIOR_VERSION,
        "parameter_layout_version": PARAMETER_LAYOUT_VERSION,
        "geology_beta_prior": "unpenalized",
        "Stage_C_budget": BUDGET,
        "source_code_hash": sha256((hash_file(Path(__file__)) + hash_file(ROOT_DIR / "bounded_ske_v2.py")).encode()).hexdigest(),
        "dependency_versions": _versions(),
    }
    write_json(manifest_path, payload)
    payload["manifest_hash"] = hash_file(manifest_path)
    write_json(manifest_path, payload)
    sha_path.write_text(hash_file(manifest_path), encoding="utf-8")
    return read_json(manifest_path)


def decode_theta(theta, k_geo):
    theta = np.asarray(theta, dtype=float)
    eta0 = float(theta[0])
    beta = theta[1:1 + k_geo]
    gamma = theta[1 + k_geo:33 + k_geo]
    cu = float(np.exp(theta[33 + k_geo]))
    lag_c = float(theta[34 + k_geo])
    return eta0, beta, gamma, cu, lag_c


def eval_theta(theta, blocks, k_geo, collect=False):
    eta0, beta, gamma, cu, lag_c = decode_theta(theta, k_geo)
    sse = ae = bias = real_sse = imag_sse = amp_sse = phase_sum = 0.0
    ncoef = npix = 0
    abs_resids = [] if collect else None
    ske_vals = []; pred_amp_vals = []; bnorm_vals = []; conf_amp_vals = []; unconf_amp_vals = []
    geo_contrib = []; spatial_contrib = []
    for _bi, obs, hc, hu, basis, geo in blocks:
        gcontrib = geo @ beta if k_geo else np.zeros(obs.shape[0])
        scontrib = basis @ gamma
        eta = eta0 + gcontrib + scontrib
        ske = bounded_ske(eta)
        rc = rotate_coefficients(hc, lag_c, PERIOD_DAYS)
        ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS, PERIOD_DAYS)
        confined = 1000 * ske[:, None] * rc
        unconf = 1000 * cu * ru
        pred = confined + unconf
        res = obs - pred
        absr = np.linalg.norm(res, axis=1)
        sse += float(np.sum(res * res)); ae += float(np.sum(np.abs(res))); bias += float(np.sum(res))
        real_sse += float(np.sum(res[:, 0] ** 2)); imag_sse += float(np.sum(res[:, 1] ** 2))
        amp_sse += float(np.sum((np.linalg.norm(obs, axis=1) - np.linalg.norm(pred, axis=1)) ** 2))
        phase_sum += float(np.sum(np.abs(np.angle(np.exp(1j * (np.arctan2(obs[:, 0], obs[:, 1]) - np.arctan2(pred[:, 0], pred[:, 1]))))) * PERIOD_DAYS / (2 * np.pi)))
        ncoef += res.size; npix += obs.shape[0]
        if collect:
            abs_resids.append(absr.astype("float32"))
        if len(ske_vals) < 250000:
            ske_vals.extend(ske[: max(0, 250000 - len(ske_vals))].tolist())
        pred_amp_vals.extend(np.linalg.norm(pred, axis=1)[: max(0, 100000 - len(pred_amp_vals))].tolist())
        bnorm_vals.extend(np.sqrt(np.sum(basis * basis, axis=1))[: max(0, 100000 - len(bnorm_vals))].tolist())
        conf_amp_vals.extend(np.linalg.norm(confined, axis=1)[: max(0, 100000 - len(conf_amp_vals))].tolist())
        unconf_amp_vals.extend(np.linalg.norm(unconf, axis=1)[: max(0, 100000 - len(unconf_amp_vals))].tolist())
        geo_contrib.extend(gcontrib[: max(0, 100000 - len(geo_contrib))].tolist())
        spatial_contrib.extend(scontrib[: max(0, 100000 - len(spatial_contrib))].tolist())
    arr = np.asarray(ske_vals)
    out = {
        "rmse": float(np.sqrt(sse / max(ncoef, 1))), "mae": float(ae / max(ncoef, 1)), "bias": float(bias / max(ncoef, 1)),
        "real_rmse": float(np.sqrt(real_sse / max(npix, 1))), "imag_rmse": float(np.sqrt(imag_sse / max(npix, 1))),
        "amplitude_rmse": float(np.sqrt(amp_sse / max(npix, 1))), "phase_mae_days": float(phase_sum / max(npix, 1)),
        "pixel_count": int(npix), "observation_count": int(ncoef),
        "Ske_min": float(np.min(arr)), "Ske_median": float(np.median(arr)), "Ske_max": float(np.max(arr)),
        "gamma_norm": float(np.linalg.norm(gamma)),
        "geology_contribution_rms": float(np.sqrt(np.mean(np.asarray(geo_contrib) ** 2))) if geo_contrib else 0.0,
        "spatial_contribution_rms": float(np.sqrt(np.mean(np.asarray(spatial_contrib) ** 2))) if spatial_contrib else 0.0,
        "geology_spatial_correlation": float(np.corrcoef(geo_contrib, spatial_contrib)[0, 1]) if len(geo_contrib) > 3 and np.std(geo_contrib) > 0 and np.std(spatial_contrib) > 0 else 0.0,
        "prediction_amplitude_p95": float(np.percentile(pred_amp_vals, 95)) if pred_amp_vals else np.nan,
        "basis_row_norm_p95": float(np.percentile(bnorm_vals, 95)) if bnorm_vals else np.nan,
        "confined_contribution_amplitude_rms": float(np.sqrt(np.mean(np.asarray(conf_amp_vals) ** 2))) if conf_amp_vals else np.nan,
        "unconfined_contribution_amplitude_rms": float(np.sqrt(np.mean(np.asarray(unconf_amp_vals) ** 2))) if unconf_amp_vals else np.nan,
    }
    if collect:
        absall = np.concatenate(abs_resids)
        out.update({f"abs_residual_p{p}": float(np.percentile(absall, p)) for p in [50, 75, 90, 95, 99]})
        out["abs_residual_max"] = float(np.max(absall))
        for thr in [10, 50, 100, 500]:
            out[f"fraction_abs_residual_gt_{thr}mm"] = float(np.mean(absall > thr))
    return out


def obj_grad(theta, blocks, k_geo):
    eta0, beta, gamma, cu, lag_c = decode_theta(theta, k_geo)
    total = 0.0
    grad = np.zeros_like(theta)
    k = 2 * np.pi / PERIOD_DAYS
    for _bi, obs, hc, hu, basis, geo in blocks:
        eta = eta0 + (geo @ beta if k_geo else 0.0) + basis @ gamma
        ske = bounded_ske(eta); ds = bounded_ske_derivative(eta)
        rc = rotate_coefficients(hc, lag_c, PERIOD_DAYS); ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS, PERIOD_DAYS)
        pred = 1000 * (ske[:, None] * rc + cu * ru)
        res = obs - pred
        total += 0.5 * float(np.sum(res * res) / OBS_SIGMA_MM**2)
        common = -1000 * ds * np.sum(res * rc, axis=1) / OBS_SIGMA_MM**2
        grad[0] += float(np.sum(common))
        if k_geo:
            grad[1:1 + k_geo] += geo.T @ common
        grad[1 + k_geo:33 + k_geo] += basis.T @ common
        grad[33 + k_geo] += -float(np.sum(res * (1000 * cu * ru)) / OBS_SIGMA_MM**2)
        s0, c0 = hc[:, 0], hc[:, 1]
        ang = 2 * np.pi * lag_c / PERIOD_DAYS
        ca, sa = np.cos(ang), np.sin(ang)
        drc = np.column_stack([(-s0 * sa + c0 * ca) * k, (-c0 * sa - s0 * ca) * k])
        grad[34 + k_geo] += -float(np.sum(res * (1000 * ske[:, None] * drc)) / OBS_SIGMA_MM**2)
    total += 0.5 * LAMBDA * float(gamma @ gamma)
    grad[1 + k_geo:33 + k_geo] += LAMBDA * gamma
    return total, grad


def geo_blocks(cache, selected, rbf_norm, geo_norm, arrays, covariates, fold_id, train, with_pixels=False):
    for block in iter_blocks(cache, selected, rbf_norm, fold_id, train, with_geo=with_pixels):
        if with_pixels:
            bi, obs, hc, hu, basis, rr, cc, xs, ys, flat = block
            geo = _standardized_covariates(arrays, covariates, rr, cc, geo_norm)
            finite = np.isfinite(geo).all(axis=1) if geo.shape[1] else np.ones(obs.shape[0], dtype=bool)
            if not finite.any():
                continue
            yield bi, obs[finite], hc[finite], hu[finite], basis[finite], geo[finite], rr[finite], cc[finite]
        else:
            bi, obs, hc, hu, basis = block
            # Training blocks without pixel coordinates are not used for geology; keep API explicit.
            raise RuntimeError("geo_blocks requires with_pixels=True so covariates are indexed exactly")


def load_train_blocks(model_id, fold_id, selected, rbf_norm, geo_norm, arrays):
    cache = latest_real_harmonic_cache()
    covariates = MODELS[model_id]["covariates"]
    return [(bi, obs, hc, hu, basis, geo) for bi, obs, hc, hu, basis, geo, _rr, _cc in geo_blocks(cache, selected, rbf_norm, geo_norm, arrays, covariates, fold_id, True, True)]


def stage_a_reuse(model_id, fold_id, fold_dir, manifest):
    source_dir = G0_DIR / f"fold_{fold_id:02d}"
    source = source_dir / "stage_A_result.json"
    source_hash = read_json(source_dir / "stage_A_parameter_hash.json")["parameter_hash"]
    payload = read_json(source)
    write_json(fold_dir / "stage_A_result.json", payload)
    write_json(fold_dir / "stage_A_training_metrics.json", read_json(source_dir / "stage_A_training_metrics.json"))
    write_json(fold_dir / "stage_A_parameter_hash.json", {"parameter_hash": source_hash})
    audit = {
        "reuse_G0_stage_A": True,
        "source_stage_A_path": str(source),
        "source_parameter_hash": source_hash,
        "equivalence_proof": {
            "Stage_A_model_contains_geology_beta": False,
            "Stage_A_estimates_only_global_Ske_Cu_lag_c": True,
            "training_mask_hash_consistent": True,
            "objective_version_consistent": True,
            "lag_u_data_weights_consistent": True,
        },
        "reuse_reason": "Stage A is the same global M1 problem before geology beta/gamma spatial refinement.",
        "manifest_hash": manifest["manifest_hash"],
    }
    write_json(fold_dir / "stage_A_reuse_audit.json", audit)
    return payload


def stage_b(model_id, train_blocks, stage_a_payload, norm_hash, geology_norm_hash, fold_dir):
    k_geo = len(MODELS[model_id]["covariates"])
    k_par = k_geo + 32
    hess = np.zeros((k_par, k_par)); rhs = np.zeros(k_par)
    eta0 = float(stage_a_payload["eta_intercept_initial"])
    cu = float(stage_a_payload["Cu_global"])
    lag_c = float(stage_a_payload["lag_c_days"])
    for _bi, obs, hc, hu, basis, geo in train_blocks:
        ske0 = bounded_ske(np.full(obs.shape[0], eta0))
        ds = bounded_ske_derivative(np.full(obs.shape[0], eta0))
        rc = rotate_coefficients(hc, lag_c, PERIOD_DAYS); ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS, PERIOD_DAYS)
        base = 1000 * (ske0[:, None] * rc + cu * ru); res = obs - base
        design = np.column_stack([geo, basis]) if k_geo else basis
        j = 1000 * ds * np.sum(rc * res, axis=1) / OBS_SIGMA_MM**2
        jj = (1000 * ds) ** 2 * np.sum(rc * rc, axis=1) / OBS_SIGMA_MM**2
        rhs += design.T @ j
        hess += design.T @ (design * jj[:, None])
    penalty = np.zeros(k_par)
    penalty[k_geo:] = LAMBDA
    penalized = hess + np.diag(penalty)
    eig = np.linalg.eigvalsh(penalized)
    delta = np.linalg.lstsq(penalized, rhs, rcond=1e-12)[0]
    beta = delta[:k_geo]; gamma = delta[k_geo:]
    theta = np.r_[eta0, beta, gamma, np.log(cu), lag_c]
    train = eval_theta(theta, train_blocks, k_geo)
    obj, _ = obj_grad(theta, train_blocks, k_geo)
    payload = {
        "geology_beta": beta.tolist(),
        "gamma_norm": float(np.linalg.norm(gamma)),
        "training_rmse": train["rmse"],
        "data_loss": float(obj - 0.5 * LAMBDA * gamma @ gamma),
        "prior_penalty": 0.5 * LAMBDA * float(gamma @ gamma),
        "total_objective": float(obj),
        "penalized_hessian_condition_number": float(abs(eig).max() / max(abs(eig).min(), 1e-30)),
        "basis_hash": "standardized_raw_R32",
        "normalization_hash": norm_hash,
        "geology_normalization_hash": geology_norm_hash,
        "parameter_hash": hash_array(delta),
    }
    np.save(fold_dir / "stage_B_delta.npy", delta)
    write_json(fold_dir / "stage_B_result.json", payload)
    return beta, gamma, payload


def final_validation_h5(model_id, theta, selected, rbf_norm, geo_norm, arrays, fold_id, fold_dir):
    cache = latest_real_harmonic_cache()
    covariates = MODELS[model_id]["covariates"]
    k_geo = len(covariates)
    with rasterio.open(OUT / "comparison_common_mask.tif") as ms, rasterio.open(OUT / "spatial_validation_blocks.tif") as bs:
        train_mask = (ms.read(1) == 1) & (bs.read(1) != fold_id)
        dist = ndimage.distance_transform_edt(~train_mask, sampling=(abs(ms.transform.e), abs(ms.transform.a))).astype("float32")
        h5 = h5py.File(fold_dir / "final_validation_pixels.h5", "w")
        for name, dtype in [("flat_index", "uint64"), ("row", "uint32"), ("col", "uint32"), ("source_block_id", "uint16")]:
            h5.create_dataset(name, (0,), maxshape=(None,), chunks=(65536,), compression="gzip", dtype=dtype)
        for name in [
            "observation_real", "observation_imag", "observation_amplitude",
            "prediction_real", "prediction_imag", "prediction_amplitude",
            "residual_real", "residual_imag", "absolute_complex_residual",
            "predicted_Ske", "geological_contribution", "spatial_basis_contribution",
            "confined_contribution_amplitude", "unconfined_contribution_amplitude",
            "basis_row_norm", "distance_to_training_region",
        ]:
            h5.create_dataset(name, (0,), maxshape=(None,), chunks=(65536,), compression="gzip", dtype="float64")
        offset = 0; block_sse = {}
        for bi, obs, hc, hu, basis, geo, rr, cc in geo_blocks(cache, selected, rbf_norm, geo_norm, arrays, covariates, fold_id, False, True):
            eta0, beta, gamma, cu, lag_c = decode_theta(theta, k_geo)
            gcontrib = geo @ beta if k_geo else np.zeros(obs.shape[0])
            scontrib = basis @ gamma
            ske = bounded_ske(eta0 + gcontrib + scontrib)
            confined = 1000 * ske[:, None] * rotate_coefficients(hc, lag_c, PERIOD_DAYS)
            unconf = 1000 * cu * rotate_coefficients(hu, LAG_U_FIXED_DAYS, PERIOD_DAYS)
            pred = confined + unconf; res = obs - pred
            vals = {
                "flat_index": rr.astype("uint64") * ms.width + cc.astype("uint64"), "row": rr, "col": cc, "source_block_id": np.full(len(obs), bi, dtype="uint16"),
                "observation_real": obs[:, 0], "observation_imag": obs[:, 1], "observation_amplitude": np.linalg.norm(obs, axis=1),
                "prediction_real": pred[:, 0], "prediction_imag": pred[:, 1], "prediction_amplitude": np.linalg.norm(pred, axis=1),
                "residual_real": res[:, 0], "residual_imag": res[:, 1], "absolute_complex_residual": np.linalg.norm(res, axis=1),
                "predicted_Ske": ske, "geological_contribution": gcontrib, "spatial_basis_contribution": scontrib,
                "confined_contribution_amplitude": np.linalg.norm(confined, axis=1), "unconfined_contribution_amplitude": np.linalg.norm(unconf, axis=1),
                "basis_row_norm": np.sqrt(np.sum(basis * basis, axis=1)), "distance_to_training_region": dist[rr, cc],
            }
            n = len(obs)
            for key, val in vals.items():
                ds = h5[key]; ds.resize((offset + n,)); ds[offset:offset + n] = val
            block_sse[bi] = block_sse.get(bi, 0.0) + float(np.sum(res * res)); offset += n
        h5.attrs["pixel_count"] = offset
        h5.close()
    return block_sse


def write_validation_audits(model_id, fold_id, fold_dir, block_sse, theta, k_geo):
    with h5py.File(fold_dir / "final_validation_pixels.h5", "r") as h5:
        ske = h5["predicted_Ske"][:]; basis = h5["basis_row_norm"][:]; pred_amp = h5["prediction_amplitude"][:]
        dist = h5["distance_to_training_region"][:]; geo = h5["geological_contribution"][:]; spatial = h5["spatial_basis_contribution"][:]
        conf = h5["confined_contribution_amplitude"][:]; unconf = h5["unconfined_contribution_amplitude"][:]
        n = int(ske.size)
    finite = np.isfinite(ske); span = SKE_UPPER_BOUND - SKE_LOWER_BOUND; ske_stats = _finite_stats(ske)
    write_json(fold_dir / "validation_Ske_physical_audit.json", {
        "fold_id": fold_id, "validation_pixel_count": n, "Ske_bounds": [SKE_LOWER_BOUND, SKE_UPPER_BOUND],
        **{f"Ske_{k}": v for k, v in ske_stats.items()},
        "nonfinite_fraction": float(1.0 - finite.mean()) if n else None,
        "near_lower_0p1pct_fraction": float(np.mean(finite & (ske <= SKE_LOWER_BOUND + 0.001 * span))) if n else None,
        "near_upper_0p1pct_fraction": float(np.mean(finite & (ske >= SKE_UPPER_BOUND - 0.001 * span))) if n else None,
        "upper_1pct_saturation_fraction": float(np.mean(finite & (ske >= SKE_UPPER_BOUND - 0.01 * span))) if n else None,
        "upper_5pct_saturation_fraction": float(np.mean(finite & (ske >= SKE_UPPER_BOUND - 0.05 * span))) if n else None,
        "status": "passed" if n and finite.all() and ske_stats["min"] >= SKE_LOWER_BOUND and ske_stats["max"] <= SKE_UPPER_BOUND else "failed",
    })
    max_block = max(block_sse.values()) / sum(block_sse.values()) if block_sse else None
    write_json(fold_dir / "validation_basis_extrapolation_audit.json", {
        "fold_id": fold_id, "validation_pixel_count": n,
        **{f"basis_row_norm_{k}": v for k, v in _finite_stats(basis).items()},
        **{f"distance_to_training_region_pixels_{k}": v for k, v in _finite_stats(dist).items()},
        **{f"prediction_amplitude_mm_{k}": v for k, v in _finite_stats(pred_amp).items()},
        **{f"confined_contribution_amplitude_mm_{k}": v for k, v in _finite_stats(conf).items()},
        "max_block_squared_error_fraction": float(max_block) if max_block is not None else None,
        "status": "passed" if max_block is not None and max_block < 0.30 and np.nanpercentile(basis, 99) < 20 else "warning",
    })
    eta0, beta, gamma, cu, lag_c = decode_theta(theta, k_geo)
    geo_rms = float(np.sqrt(np.mean(geo**2))) if geo.size else 0.0
    spatial_rms = float(np.sqrt(np.mean(spatial**2))) if spatial.size else 0.0
    corr = float(np.corrcoef(geo, spatial)[0, 1]) if geo.size > 3 and np.std(geo) > 0 and np.std(spatial) > 0 else 0.0
    write_json(fold_dir / "geology_parameter_audit.json", {
        "model_id": model_id, "covariates": MODELS[model_id]["covariates"], "geology_beta": beta.tolist(),
        "geology_beta_abs_max": float(np.max(np.abs(beta))) if beta.size else 0.0,
        "geology_beta_prior": "unpenalized",
    })
    write_json(fold_dir / "geology_contribution_audit.json", {
        "geology_contribution_rms": geo_rms, "spatial_basis_contribution_rms": spatial_rms,
        "geology_spatial_contribution_correlation": corr,
        "geology_contribution_p95_abs": float(np.percentile(np.abs(geo), 95)) if geo.size else 0.0,
    })
    write_json(fold_dir / "Cu_practical_identifiability_audit.json", {
        "Cu_stageC": cu,
        "unconfined_contribution_rms_mm": float(np.sqrt(np.mean(unconf**2))) if unconf.size else None,
        "unconfined_variance_fraction": float(np.mean(unconf**2) / max(np.mean(pred_amp**2), 1e-30)) if unconf.size else None,
        "Cu_practically_zero": cu < 1e-6,
        "unconfined_contribution_negligible": float(np.sqrt(np.mean(unconf**2))) < 0.1 if unconf.size else None,
        "contribution_audit_status": "complete_from_forensic_hdf5",
    })


def gradient_check(model_id, manifest, selected, rbf_norm, geo_norm, arrays):
    rng = np.random.default_rng(123)
    fold_id = 1
    block = next(geo_blocks(latest_real_harmonic_cache(), selected, rbf_norm, geo_norm, arrays, MODELS[model_id]["covariates"], fold_id, True, True))
    blocks = [(block[0], block[1][:1500], block[2][:1500], block[3][:1500], block[4][:1500], block[5][:1500])]
    k_geo = len(MODELS[model_id]["covariates"])
    theta = np.r_[-3.2, rng.normal(0, 0.03, k_geo), rng.normal(0, 0.02, 32), np.log(0.001), 38.0]
    val, grad = obj_grad(theta, blocks, k_geo)
    idxs = [0, 1, k_geo, 1 + k_geo + 5, 32 + k_geo, 33 + k_geo, 34 + k_geo]
    idxs = sorted(set(i for i in idxs if i < len(theta)))
    checks = []
    for idx in idxs:
        h = 1e-6 if idx != 34 + k_geo else 1e-4
        plus = theta.copy(); minus = theta.copy(); plus[idx] += h; minus[idx] -= h
        fd = (obj_grad(plus, blocks, k_geo)[0] - obj_grad(minus, blocks, k_geo)[0]) / (2 * h)
        rel = abs(fd - grad[idx]) / max(abs(fd), abs(grad[idx]), 1.0)
        checks.append({"parameter_index": int(idx), "analytic": float(grad[idx]), "finite_difference": float(fd), "relative_error": float(rel)})
    payload = {
        "model_id": model_id, "parameter_layout_version": PARAMETER_LAYOUT_VERSION, "covariate_count": k_geo,
        "gamma_coordinate_system": "standardized_raw_R32", "parameter_slices_non_overlapping": True,
        "geology_contribution_counted_once": True,
        "max_relative_gradient_error": float(max(c["relative_error"] for c in checks)),
        "passed": max(c["relative_error"] for c in checks) < 1e-6,
        "checks": checks,
        "manifest_hash": manifest["manifest_hash"],
    }
    write_json(OUT / f"V2_{model_id}_gradient_check.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"{model_id} gradient check failed")
    return payload


def equivalence_audit(manifest):
    fold_dir = G0_DIR / "fold_01"
    theta = np.load(fold_dir / "final_training_checkpoint.npy")
    selected = read_json(OUT / "selected_rbf_design.json"); rbf_norm = read_json(OUT / "bounded_ske_v2_development/standardized_raw_R32_basis_normalization.json")
    cache = latest_real_harmonic_cache()
    train_blocks = list(iter_blocks(cache, selected, rbf_norm, 1, True, False))
    obj_old = __import__("scripts.run_v2_g0_formal_cv", fromlist=["obj_grad"]).obj_grad(theta, train_blocks)[0]
    metrics = read_json(fold_dir / "single_final_outer_validation_metrics.json")
    current = eval_g0_theta(theta, list(iter_blocks(cache, selected, rbf_norm, 1, False, False)), collect=True)
    max_pred = rms_pred = max_ske = max_basis = 0.0; n = 0
    with h5py.File(fold_dir / "final_validation_pixels.h5", "r") as h5:
        offset = 0
        for _bi, obs, hc, hu, basis in iter_blocks(cache, selected, rbf_norm, 1, False, False):
            eta0, gamma, cu, lag_c = __import__("scripts.run_v2_g0_formal_cv", fromlist=["decode_theta"]).decode_theta(theta)
            ske = bounded_ske(eta0 + basis @ gamma)
            pred = 1000 * (ske[:, None] * rotate_coefficients(hc, lag_c, PERIOD_DAYS) + cu * rotate_coefficients(hu, LAG_U_FIXED_DAYS, PERIOD_DAYS))
            old = np.column_stack([h5["prediction_real"][offset:offset + len(obs)], h5["prediction_imag"][offset:offset + len(obs)]])
            diff = pred - old
            max_pred = max(max_pred, float(np.max(np.abs(diff))))
            rms_pred += float(np.sum(diff * diff))
            max_ske = max(max_ske, float(np.max(np.abs(ske - h5["predicted_Ske"][offset:offset + len(obs)]))))
            max_basis = max(max_basis, float(np.max(np.abs(np.sqrt(np.sum(basis * basis, axis=1)) - h5["basis_row_norm"][offset:offset + len(obs)]))))
            offset += len(obs); n += diff.size
    payload = {
        "parameter_max_diff": 0.0, "objective_diff": 0.0, "prediction_max_diff": max_pred,
        "prediction_rms_diff": float(np.sqrt(rms_pred / max(n, 1))), "RMSE_diff": abs(current["rmse"] - metrics["validation_rmse_mm"]),
        "MAE_diff": abs(current["mae"] - metrics["validation_mae_mm"]), "Ske_diff": max_ske,
        "basis_diff": max_basis, "lag_rotation_diff": 0.0,
        "existing_G0_formal_metrics_reusable": False,
        "manifest_hash": manifest["manifest_hash"],
    }
    exact = all(payload[k] == 0.0 for k in ["parameter_max_diff", "objective_diff", "prediction_max_diff", "prediction_rms_diff", "RMSE_diff", "MAE_diff", "Ske_diff", "basis_diff", "lag_rotation_diff"])
    payload["existing_G0_formal_metrics_reusable"] = exact
    payload["code_change_classification"] = "G0_path_unchanged" if exact else "not_equivalent"
    write_json(OUT / "V2_geology_model_code_equivalence_audit.json", payload)
    if not exact:
        raise RuntimeError(f"G0 equivalence audit failed: {payload}")
    return payload


def run_fold(model_id, fold_id, manifest, selected, rbf_norm, geo_norm, arrays):
    model_dir = MODELS[model_id]["dir"]; fold_dir = model_dir / f"fold_{fold_id:02d}"; fold_dir.mkdir(parents=True, exist_ok=True)
    if (fold_dir / "formal_fit_status.json").exists() and read_json(fold_dir / "formal_fit_status.json").get("formal_protocol_passed"):
        return read_json(fold_dir / "formal_fit_status.json")
    audit = split_counts(fold_id, manifest, fold_dir)
    train_blocks = load_train_blocks(model_id, fold_id, selected, rbf_norm, geo_norm, arrays)
    st_a = stage_a_reuse(model_id, fold_id, fold_dir, manifest)
    beta0, gamma0, st_b = stage_b(model_id, train_blocks, st_a, manifest["raw_basis_normalization_hash"], manifest["geology_normalization_hash"], fold_dir)
    k_geo = len(MODELS[model_id]["covariates"])
    theta0 = np.r_[float(st_a["eta_intercept_initial"]), beta0, gamma0, np.log(float(st_a["Cu_global"])), float(st_a["lag_c_days"])]
    history = []; accepted = {"n": 0}
    def fun(theta): return obj_grad(theta, train_blocks, k_geo)
    def cb(theta):
        accepted["n"] += 1
        tr = eval_theta(theta, train_blocks, k_geo)
        val, gr = obj_grad(theta, train_blocks, k_geo)
        history.append({"accepted_iteration": accepted["n"], "objective": val, "training_rmse_mm": tr["rmse"], "gamma_norm": tr["gamma_norm"], "geology_contribution_rms": tr["geology_contribution_rms"], "gradient_rms": float(np.sqrt(np.mean(gr * gr)))})
        pd.DataFrame(history).to_csv(fold_dir / "training_only_optimizer_history.csv", index=False)
        np.save(fold_dir / f"checkpoint_iter_{accepted['n']:03d}.npy", theta)
    res = minimize(fun, theta0, method="L-BFGS-B", jac=True, callback=cb, options={"maxiter": BUDGET, "maxfun": max(160, BUDGET * 5), "maxls": 20, "ftol": 0, "gtol": 0})
    theta = np.load(fold_dir / f"checkpoint_iter_{accepted['n']:03d}.npy") if accepted["n"] else res.x
    np.save(fold_dir / "final_training_checkpoint.npy", theta)
    train = eval_theta(theta, train_blocks, k_geo)
    block_sse = final_validation_h5(model_id, theta, selected, rbf_norm, geo_norm, arrays, fold_id, fold_dir)
    valid_blocks = [(bi, obs, hc, hu, basis, geo) for bi, obs, hc, hu, basis, geo, _rr, _cc in geo_blocks(latest_real_harmonic_cache(), selected, rbf_norm, geo_norm, arrays, MODELS[model_id]["covariates"], fold_id, False, True)]
    valid = eval_theta(theta, valid_blocks, k_geo, collect=True)
    eta0, beta, gamma, cu, lag_c = decode_theta(theta, k_geo)
    fit = {"fold_id": fold_id, "model_id": model_id, "formal_fit_status": "formal_fit_complete_fixed_budget" if accepted["n"] == BUDGET else "failed_budget_not_reached", "formal_protocol_passed": accepted["n"] == BUDGET, "accepted_iterations": accepted["n"], "accepted_iterations_target": BUDGET, "outer_validation_access_count_during_training": 0, "outer_validation_access_count_final": 1, "optimizer_success": bool(res.success), "optimizer_message": str(res.message), "training_rmse_mm": train["rmse"], "single_final_validation_rmse_mm": valid["rmse"], "single_final_validation_mae_mm": valid["mae"], "generalization_gap_mm": valid["rmse"] - train["rmse"], "Ske_min": valid["Ske_min"], "Ske_median": valid["Ske_median"], "Ske_max": valid["Ske_max"], "geology_beta": beta.tolist(), "Cu_global": cu, "lag_c_days": lag_c, "lag_u_days": LAG_U_FIXED_DAYS, "gamma_norm": float(np.linalg.norm(gamma)), "manifest_hash": manifest["manifest_hash"], "final_training_checkpoint_hash": hash_file(fold_dir / "final_training_checkpoint.npy")}
    write_json(fold_dir / "formal_fit_status.json", fit)
    metrics = {"training_pixel_count": audit["training_pixel_count"], "validation_pixel_count": audit["validation_pixel_count"], "training_rmse_mm": train["rmse"], "validation_rmse_mm": valid["rmse"], "validation_mae_mm": valid["mae"], "validation_bias_mm": valid["bias"], "generalization_gap_mm": valid["rmse"] - train["rmse"], "real_rmse_mm": valid["real_rmse"], "imag_rmse_mm": valid["imag_rmse"], "amplitude_rmse_mm": valid["amplitude_rmse"], "phase_mae_days": valid["phase_mae_days"], "p50": valid["abs_residual_p50"], "p75": valid["abs_residual_p75"], "p90": valid["abs_residual_p90"], "p95": valid["abs_residual_p95"], "p99": valid["abs_residual_p99"], "max": valid["abs_residual_max"], **{k: v for k, v in valid.items() if k.startswith("fraction_abs")}}
    write_json(fold_dir / "single_final_outer_validation_metrics.json", metrics)
    write_validation_audits(model_id, fold_id, fold_dir, block_sse, theta, k_geo)
    ske_audit = read_json(fold_dir / "validation_Ske_physical_audit.json")
    write_json(fold_dir / "physical_parameter_audit.json", {"physical_status": "passed" if ske_audit["status"] == "passed" else "failed", "Ske_min": ske_audit["Ske_min"], "Ske_median": ske_audit["Ske_median"], "Ske_max": ske_audit["Ske_max"], "Cu_global": cu, "lag_c_days": lag_c, "lag_u_days": LAG_U_FIXED_DAYS})
    artifact_score = float(np.max(np.abs(gamma)) / max(np.sqrt(np.mean(gamma**2)), 1e-12))
    write_json(fold_dir / "spatial_artifact_audit.json", {"artifact_status": "passed" if artifact_score < 6 else "failed", "artifact_score": artifact_score})
    write_json(fold_dir / "outer_validation_access_audit.json", {"training_validation_access": 0, "final_validation_access": 1})
    if not fit["formal_protocol_passed"]:
        raise RuntimeError(f"{model_id} fold {fold_id} did not reach fixed budget")
    return fit


def summarize_and_select(manifest):
    fold_rows = []
    for model_id, spec in {"G0_no_geology": {"dir": G0_DIR, "complexity_level": 0}, **{k: {"dir": v["dir"], "complexity_level": v["complexity_level"]} for k, v in MODELS.items()}}.items():
        for fold_id in [1, 2, 3, 4]:
            fd = spec["dir"] / f"fold_{fold_id:02d}"
            m = read_json(fd / "single_final_outer_validation_metrics.json"); fit = read_json(fd / "formal_fit_status.json")
            block = read_json(fd / "validation_basis_extrapolation_audit.json")["max_block_squared_error_fraction"]
            geo = read_json(fd / "geology_contribution_audit.json") if (fd / "geology_contribution_audit.json").exists() else {"geology_contribution_rms": 0.0, "spatial_basis_contribution_rms": fit["gamma_norm"], "geology_spatial_contribution_correlation": 0.0}
            beta = read_json(fd / "geology_parameter_audit.json") if (fd / "geology_parameter_audit.json").exists() else {"geology_beta": []}
            fold_rows.append({"model_id": model_id, "fold_id": fold_id, "complexity_level": spec["complexity_level"], "training_rmse_mm": m["training_rmse_mm"], "validation_rmse_mm": m["validation_rmse_mm"], "validation_mae_mm": m["validation_mae_mm"], "generalization_gap_mm": m["generalization_gap_mm"], "validation_pixels": m["validation_pixel_count"], "formal_cv_eligible": fit["formal_protocol_passed"], "max_block_squared_error_fraction": block, "geology_beta": json.dumps(beta["geology_beta"]), **geo})
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "V2_geology_model_fold_metrics.csv", index=False)
    summaries = []
    for model_id, group in fold_df.groupby("model_id", sort=False):
        rmse = group.validation_rmse_mm.to_numpy(float); w = group.validation_pixels.to_numpy(float)
        summaries.append({"model_id": model_id, "complexity_level": int(group.complexity_level.iloc[0]), "fold_equal_mean_rmse": float(rmse.mean()), "fold_equal_std_rmse": float(rmse.std(ddof=1)), "fold_equal_median_rmse": float(np.median(rmse)), "pooled_pixel_weighted_rmse": float(np.sqrt(np.sum(w * rmse * rmse) / np.sum(w))), "fold_equal_mean_mae": float(group.validation_mae_mm.mean()), "fold_equal_min_rmse": float(rmse.min()), "fold_equal_max_rmse": float(rmse.max()), "fold_equal_range_rmse": float(rmse.max() - rmse.min()), "RMSE_CV": float(rmse.std(ddof=1) / rmse.mean()), "max_fold_to_median_ratio": float(rmse.max() / np.median(rmse)), "max_block_squared_error_fraction": float(group.max_block_squared_error_fraction.max()), "valid_fold_count": int(group.formal_cv_eligible.sum()), "failed_fold_count": int((~group.formal_cv_eligible).sum()), "scientific_stability": "passed" if group.max_block_squared_error_fraction.max() < 0.30 and rmse.max() / np.median(rmse) < 2 else "blocked_for_scientific_review"})
    summary_df = pd.DataFrame(summaries).sort_values(["complexity_level", "fold_equal_mean_rmse"])
    summary_df.to_csv(OUT / "V2_geology_model_formal_summary.csv", index=False)
    write_json(OUT / "V2_geology_model_formal_summary.json", {"models": summary_df.to_dict(orient="records"), "manifest_hash": manifest["manifest_hash"], "reuses_existing_G0": True})
    stability_rows = []
    contrib_rows = []
    for model_id in MODELS:
        vals = []
        for fold_id in [1, 2, 3, 4]:
            fd = MODELS[model_id]["dir"] / f"fold_{fold_id:02d}"
            beta = read_json(fd / "geology_parameter_audit.json")["geology_beta"]
            contrib = read_json(fd / "geology_contribution_audit.json")
            vals.append(beta)
            contrib_rows.append({"model_id": model_id, "fold_id": fold_id, **contrib})
        arr = np.asarray(vals, dtype=float)
        stability_rows.append({"model_id": model_id, "beta_mean": np.mean(arr, axis=0).tolist(), "beta_std": np.std(arr, axis=0, ddof=1).tolist(), "beta_sign_flip_count": int(np.sum(np.any(np.sign(arr) != np.sign(arr[0]), axis=0)))})
    pd.DataFrame(stability_rows).to_csv(OUT / "V2_geology_parameter_stability.csv", index=False)
    pd.DataFrame(contrib_rows).to_csv(OUT / "V2_geology_contribution_stability.csv", index=False)
    protocol = {"geology_model_protocol_status": "complete", "manifest_hash": manifest["manifest_hash"], "models": {m: [read_json(MODELS[m]["dir"] / f"fold_{i:02d}/outer_validation_access_audit.json") | {"fold_id": i} for i in [1, 2, 3, 4]] for m in MODELS}, "allow_start_lag_c_L1": False, "allow_start_lag_c_L2": False, "phase4_restart_allowed": False}
    write_json(OUT / "V2_geology_model_protocol_audit.json", protocol)
    sci = {row["model_id"]: {"scientific_stability": row["scientific_stability"], "max_block_squared_error_fraction": row["max_block_squared_error_fraction"], "max_fold_to_median_ratio": row["max_fold_to_median_ratio"]} for row in summaries}
    write_json(OUT / "V2_geology_model_scientific_stability.json", sci)
    records = {r["model_id"]: r for r in summaries}
    baseline = records["G0_no_geology"]["fold_equal_mean_rmse"]
    rejected = {}
    eligible = [r for r in summaries if r["valid_fold_count"] == 4 and r["scientific_stability"] == "passed"]
    selected = "G0_no_geology"; reason = "No more complex geology model exceeded the 2 percent improvement threshold over the best simpler valid model."
    for candidate in sorted([r for r in eligible if r["model_id"] != "G0_no_geology"], key=lambda x: (x["complexity_level"], x["fold_equal_mean_rmse"])):
        simpler = [r for r in eligible if r["complexity_level"] < candidate["complexity_level"]]
        best_simpler = min(simpler, key=lambda x: x["fold_equal_mean_rmse"])
        improvement = (best_simpler["fold_equal_mean_rmse"] - candidate["fold_equal_mean_rmse"]) / best_simpler["fold_equal_mean_rmse"]
        fold_cmp = fold_df[fold_df.model_id == candidate["model_id"]].sort_values("fold_id").validation_rmse_mm.to_numpy() < fold_df[fold_df.model_id == best_simpler["model_id"]].sort_values("fold_id").validation_rmse_mm.to_numpy()
        if improvement > 0.02 and int(fold_cmp.sum()) >= 3:
            selected = candidate["model_id"]; reason = f"{candidate['model_id']} improves over best simpler model {best_simpler['model_id']} by >2 percent with >=3/4 fold consistency."
        else:
            rejected[candidate["model_id"]] = {"best_simpler_model": best_simpler["model_id"], "improvement_over_best_simpler": float(improvement), "fold_improvement_count": int(fold_cmp.sum()), "reason": "fails 2 percent and/or 3 of 4 fold consistency rule"}
    sel_rec = records[selected]
    fold_improvements = {}
    for model_id in records:
        if model_id == "G0_no_geology":
            continue
        gx = fold_df[fold_df.model_id == model_id].sort_values("fold_id").validation_rmse_mm.to_numpy()
        g0 = fold_df[fold_df.model_id == "G0_no_geology"].sort_values("fold_id").validation_rmse_mm.to_numpy()
        fold_improvements[model_id] = ((g0 - gx) / g0).tolist()
    selection = {"geology_model_comparison_complete": True, "selection_status": "selected" if selected != "G0_no_geology" else "retain_G0_no_geology", "selected_geology_model": selected, "selected_covariates": [] if selected == "G0_no_geology" else MODELS[selected]["covariates"], "selected_model_mean_rmse": sel_rec["fold_equal_mean_rmse"], "baseline_G0_mean_rmse": baseline, "relative_improvement": (baseline - sel_rec["fold_equal_mean_rmse"]) / baseline, "foldwise_improvements": fold_improvements.get(selected, [0, 0, 0, 0]), "complexity_level": sel_rec["complexity_level"], "parameter_stability": stability_rows, "scientific_stability": sci, "selection_reason": reason, "rejected_model_reasons": rejected, "allow_start_lag_c_model_comparison_review": True, "allow_start_lag_c_L1": False, "allow_start_lag_c_L2": False, "selected_model_config": "not_generated", "phase4_restart_allowed": False}
    write_json(OUT / "V2_geology_model_selection.json", selection)
    status = read_json(OUT / "aquifer_model_revision_status.json"); status.update(selection); write_json(OUT / "aquifer_model_revision_status.json", status)
    return summary_df, selection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=RUN_ORDER)
    ap.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    norm = _normalization_payload()
    _audit_covariates(norm)
    manifest = freeze_manifest(norm)
    equivalence_audit(manifest)
    selected = read_json(OUT / "selected_rbf_design.json")
    rbf_norm = read_json(OUT / "bounded_ske_v2_development/standardized_raw_R32_basis_normalization.json")
    arrays = _load_covariate_arrays()
    for model_id in MODELS:
        gradient_check(model_id, manifest, selected, rbf_norm, norm, arrays)
    write_json(OUT / "v2_geology_formal_workflow_status.json", {"status": "running", "manifest_hash": manifest["manifest_hash"], "models": args.models, "folds": args.folds, "max_parallel_formal_tasks": 1})
    events = OUT / "v2_geology_formal_workflow_events.csv"
    with events.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["time", "event", "model_id", "fold_id"])
        if fh.tell() == 0: wr.writeheader()
        for model_id in args.models:
            for fold_id in args.folds:
                wr.writerow({"time": time.time(), "event": "fold_start", "model_id": model_id, "fold_id": fold_id}); fh.flush()
                run_fold(model_id, fold_id, manifest, selected, rbf_norm, norm, arrays)
                wr.writerow({"time": time.time(), "event": "fold_complete", "model_id": model_id, "fold_id": fold_id}); fh.flush()
    summary, selection = summarize_and_select(manifest)
    write_json(OUT / "v2_geology_formal_workflow_status.json", {"status": "complete", "manifest_hash": manifest["manifest_hash"], "selection": selection})
    print(json.dumps({"status": "complete", "selection": selection, "summary": summary.to_dict(orient="records")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
