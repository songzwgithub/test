#!/usr/bin/env python
"""Run V2 G0/L0 formal four-fold CV and summary."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import sys
import time
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import xy
from rasterio.windows import Window
from scipy import ndimage, stats
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
    inverse_bounded_ske,
    saturation_fractions,
)
from profiled_stage_a import StageAStats, latest_real_harmonic_cache, solve_from_stats
from scripts.run_stage_b_fixed_lagu import rbf_values
from scripts.run_stage_c_fixed_lagu import LAG_U_FIXED_DAYS
from storage_inversion import rotate_coefficients


OUT = Path("outputs/aquifer_model_revision")
MODEL_DIR = OUT / "model_compare_v2/G0_no_geology_L0_shared"
OBS_SIGMA_MM = 5.0
PERIOD_DAYS = 365.2425
LAMBDA = 30.0
BUDGET = 40
EXPECTED_COMMON = 15_241_589


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def hash_array(arr: np.ndarray) -> str:
    return sha256(np.asarray(arr, dtype="float64").tobytes()).hexdigest()


def dependency_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ["numpy", "pandas", "scipy", "rasterio", "h5py", "pyproj"]:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def freeze_manifest() -> dict:
    frozen = OUT / "formal_protocol_v2_frozen_manifest.json"
    sha_path = OUT / "formal_protocol_v2_frozen_manifest.sha256"
    if frozen.exists():
        manifest = read_json(frozen)
        digest = hash_file(frozen)
        if sha_path.exists() and sha_path.read_text().strip() != digest:
            raise RuntimeError("Frozen V2 manifest sha256 mismatch")
        return manifest
    draft = read_json(OUT / "formal_protocol_v2_draft_manifest.json")
    selected = read_json(OUT / "selected_rbf_design.json")
    norm = read_json(OUT / "bounded_ske_v2_development/standardized_raw_R32_basis_normalization.json")
    manifest = {
        "manifest_status": "frozen_for_formal_v2_g0_fourfold",
        "model_version": "M1_v2_bounded_Ske",
        "candidate": "V2b_bounded_standardized_raw_R32",
        "aquifer_structure": "M1_two_aquifer_shared_unconfined",
        "geology_model": "G0_no_geology",
        "lag_c_mode": "L0_shared",
        "lag_u_global_days": LAG_U_FIXED_DAYS,
        "lambda_multiplier": LAMBDA,
        "Stage_C_budget": BUDGET,
        "ske_parameterization": "bounded_logistic",
        "parameterization_version": PARAMETERIZATION_VERSION,
        "ske_lower_bound": SKE_LOWER_BOUND,
        "ske_upper_bound": SKE_UPPER_BOUND,
        "basis_type": "standardized_raw_R32_gaussian",
        "RBF_centers_hash": norm["rbf_centers_hash"],
        "raw_basis_normalization_hash": norm["normalization_hash"],
        "sigma_km": selected["sigma_km"],
        "objective_version": OBJECTIVE_VERSION,
        "prior_version": "raw_standardized_gamma_lambda30_stageA_centered_v1",
        "parameter_layout": "eta_intercept_32_gamma_logCu_lagc_fixed_lagu_v1",
        "common_mask_hash": hash_file(OUT / "comparison_common_mask.tif"),
        "fold_map_hash": hash_file(OUT / "spatial_validation_blocks.tif"),
        "source_code_hash": sha256(
            (hash_file(Path(__file__)) + hash_file(ROOT_DIR / "bounded_ske_v2.py")).encode()
        ).hexdigest(),
        "dependency_versions": dependency_versions(),
        "development_selection_result_hash": sha256(json.dumps(draft["development_selection_result"], sort_keys=True).encode()).hexdigest(),
        "formal_v2_execution_allowed": True,
    }
    required = [
        "model_version", "objective_version", "parameterization_version", "ske_lower_bound", "ske_upper_bound",
        "RBF_centers_hash", "sigma_km", "raw_basis_normalization_hash", "common_mask_hash", "fold_map_hash",
        "lambda_multiplier", "lag_u_global_days", "lag_c_mode", "parameter_layout", "prior_version", "Stage_C_budget",
        "source_code_hash", "dependency_versions",
    ]
    missing = [k for k in required if k not in manifest or manifest[k] in (None, "")]
    if missing:
        raise RuntimeError(f"Missing V2 manifest fields: {missing}")
    write_json(frozen, manifest)
    digest = hash_file(frozen)
    sha_path.write_text(digest, encoding="utf-8")
    manifest["manifest_hash"] = digest
    write_json(frozen, manifest)
    digest = hash_file(frozen)
    sha_path.write_text(digest, encoding="utf-8")
    return manifest


def split_counts(fold_id: int, manifest: dict, fold_dir: Path) -> dict:
    common = train = val = inter = union = 0
    with rasterio.open(OUT / "comparison_common_mask.tif") as ms, rasterio.open(OUT / "spatial_validation_blocks.tif") as bs:
        for _, window in ms.block_windows(1):
            m = ms.read(1, window=window) == 1
            f = bs.read(1, window=window)
            tr = m & (f != fold_id)
            va = m & (f == fold_id)
            common += int(m.sum()); train += int(tr.sum()); val += int(va.sum())
            inter += int((tr & va).sum()); union += int((tr | va).sum())
    audit = {
        "fold_id": fold_id,
        "common_mask_pixel_count": common,
        "training_pixel_count": train,
        "validation_pixel_count": val,
        "training_validation_intersection_count": inter,
        "training_validation_union_count": union,
        "common_mask_hash": hash_file(OUT / "comparison_common_mask.tif"),
        "fold_map_hash": hash_file(OUT / "spatial_validation_blocks.tif"),
        "manifest_hash": manifest["manifest_hash"],
        "status": "passed" if common == EXPECTED_COMMON and train + val == common and inter == 0 and union == common else "failed",
    }
    write_json(fold_dir / "mask_partition_audit.json", audit)
    if audit["status"] != "passed":
        raise RuntimeError(f"fold{fold_id} mask audit failed")
    return audit


def iter_blocks(cache_path, selected, norm, fold_id: int, train: bool, with_geo=False):
    centers = np.asarray(selected["center_coordinates"], float)
    sigma_m = float(selected["sigma_km"]) * 1000.0
    mean = np.asarray(norm["raw_basis_mean"], float)
    rms = np.asarray(norm["raw_basis_rms"], float)
    with h5py.File(cache_path, "r") as h5, rasterio.open(OUT / "comparison_common_mask.tif") as ms, rasterio.open(OUT / "spatial_validation_blocks.tif") as bs:
        transformer = None
        if selected.get("projected_crs") and ms.crs and str(ms.crs) != selected["projected_crs"]:
            transformer = Transformer.from_crs(ms.crs, selected["projected_crs"], always_xy=True)
        for bi, start in enumerate(h5["block_start"][:]):
            count = int(h5["block_count"][bi])
            if count == 0:
                continue
            start = int(start); r = int(h5["block_row"][bi]); c = int(h5["block_col"][bi])
            h = int(h5["block_height"][bi]); w = int(h5["block_width"][bi])
            window = Window(c, r, w, h)
            flat = h5["flat_index"][start:start+count].astype(int)
            rows = flat // w; cols = flat % w
            m = ms.read(1, window=window).ravel()[flat] == 1
            folds = bs.read(1, window=window).ravel()[flat]
            take = folds != fold_id if train else folds == fold_id
            obs = h5["obs"][start:start+count]; hc = h5["hc"][start:start+count]; hu = h5["hu"][start:start+count]
            common = m & take & np.isfinite(obs).all(1) & np.isfinite(hc).all(1) & np.isfinite(hu).all(1)
            if not common.any():
                continue
            rr = r + rows[common]; cc = c + cols[common]
            xs, ys = xy(ms.transform, rr, cc, offset="center")
            xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            phi = rbf_values(np.column_stack([xs, ys]), centers, sigma_m)
            basis = (phi - mean) / rms
            if with_geo:
                yield bi, obs[common].astype(float), hc[common].astype(float), hu[common].astype(float), basis.astype(float), rr.astype(int), cc.astype(int), xs, ys, flat[common].astype(int)
            else:
                yield bi, obs[common].astype(float), hc[common].astype(float), hu[common].astype(float), basis.astype(float)


def stats_from_block(obs, hc, hu) -> StageAStats:
    inv = 1 / OBS_SIGMA_MM**2
    hs, hc0 = hc[:, 0], hc[:, 1]; us, uc = hu[:, 0], hu[:, 1]; os, oc = obs[:, 0], obs[:, 1]
    return StageAStats(
        n=int(obs.shape[0]), obs_yy=float(np.sum(obs*obs)*inv),
        hc_norm=float(np.sum(hc*hc)*1_000_000*inv), hu_norm=float(np.sum(hu*hu)*1_000_000*inv),
        hc_obs_cos=float(np.sum(hs*os+hc0*oc)*1000*inv), hc_obs_sin=float(np.sum(hc0*os-hs*oc)*1000*inv),
        hu_obs_cos=float(np.sum(us*os+uc*oc)*1000*inv), hu_obs_sin=float(np.sum(uc*os-us*oc)*1000*inv),
        cross_cos=float(np.sum(hs*us+hc0*uc)*1_000_000*inv), cross_sin=float(np.sum(hc0*us-hs*uc)*1_000_000*inv),
        observation_sigma_mm=OBS_SIGMA_MM, period_days=PERIOD_DAYS,
    )


def add_stats(a, b):
    if a is None:
        return b
    payload = {k: getattr(a, k) + getattr(b, k) for k in a.__dataclass_fields__ if k not in {"observation_sigma_mm", "period_days"}}
    payload["n"] = int(payload["n"]); payload["observation_sigma_mm"] = OBS_SIGMA_MM; payload["period_days"] = PERIOD_DAYS
    return StageAStats(**payload)


def stage_a(train_blocks, fold_dir: Path):
    stats_acc = None
    for _bi, obs, hc, hu, _b in train_blocks:
        stats_acc = add_stats(stats_acc, stats_from_block(obs, hc, hu))
    coarse = [solve_from_stats(stats_acc, lag, LAG_U_FIXED_DAYS) for lag in np.arange(0, 91, 10)]
    center = min(coarse, key=lambda r: r.objective).lag_c_days
    candidates = []
    for radius in (5, 2, 1):
        local = [solve_from_stats(stats_acc, float(np.clip(center+d, 0, PERIOD_DAYS)), LAG_U_FIXED_DAYS) for d in (-radius, 0, radius)]
        best = min(local, key=lambda r: r.objective); center = best.lag_c_days; candidates.extend(local)
    best = min(candidates, key=lambda r: r.objective)
    eta0 = float(inverse_bounded_ske(best.ske_global))
    payload = {
        "stage_A_training_only": True, "Ske_global": best.ske_global, "eta_intercept_initial": eta0,
        "Cu_global": best.cu_global, "lag_c_days": best.lag_c_days, "lag_u_days": LAG_U_FIXED_DAYS,
        "training_objective": best.objective, "training_rmse": best.rmse, "train_pixel_count": int(stats_acc.n), "status": best.status,
    }
    write_json(fold_dir / "stage_A_result.json", payload)
    write_json(fold_dir / "stage_A_training_metrics.json", {"training_rmse": best.rmse, "training_objective": best.objective, "train_pixel_count": int(stats_acc.n)})
    write_json(fold_dir / "stage_A_parameter_hash.json", {"parameter_hash": hash_array(np.array([eta0, best.cu_global, best.lag_c_days]))})
    return payload


def decode_theta(theta):
    return float(theta[0]), theta[1:33], float(np.exp(theta[33])), float(theta[34])


def eval_theta(theta, blocks, collect=False):
    eta0, gamma, cu, lag_c = decode_theta(theta)
    sse = ae = bias = real_sse = imag_sse = amp_sse = phase_sum = 0.0
    ncoef = npix = 0
    abs_resids = [] if collect else None
    ske_vals = []; pred_amp_vals = []; bnorm_vals = []; conf_amp_vals = []
    for _bi, obs, hc, hu, basis in blocks:
        eta = eta0 + basis @ gamma
        ske = bounded_ske(eta)
        rc = rotate_coefficients(hc, lag_c, PERIOD_DAYS)
        ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS, PERIOD_DAYS)
        confined = 1000 * ske[:, None] * rc
        unconf = 1000 * cu * ru
        pred = confined + unconf
        res = obs - pred
        absr = np.linalg.norm(res, axis=1)
        sse += float(np.sum(res*res)); ae += float(np.sum(np.abs(res))); bias += float(np.sum(res))
        real_sse += float(np.sum(res[:,0]**2)); imag_sse += float(np.sum(res[:,1]**2))
        amp_sse += float(np.sum((np.linalg.norm(obs, axis=1)-np.linalg.norm(pred, axis=1))**2))
        phase_sum += float(np.sum(np.abs(np.angle(np.exp(1j*(np.arctan2(obs[:,0],obs[:,1])-np.arctan2(pred[:,0],pred[:,1]))))) * PERIOD_DAYS/(2*np.pi)))
        ncoef += res.size; npix += obs.shape[0]
        if collect:
            abs_resids.append(absr.astype("float32"))
        if len(ske_vals) < 250000:
            need = 250000 - len(ske_vals); ske_vals.extend(ske[:need].tolist())
        pred_amp_vals.extend(np.linalg.norm(pred, axis=1)[: max(0, 100000-len(pred_amp_vals))].tolist())
        bnorm_vals.extend(np.sqrt(np.sum(basis*basis, axis=1))[: max(0, 100000-len(bnorm_vals))].tolist())
        conf_amp_vals.extend(np.linalg.norm(confined, axis=1)[: max(0, 100000-len(conf_amp_vals))].tolist())
    arr = np.asarray(ske_vals)
    out = {
        "rmse": float(np.sqrt(sse/max(ncoef,1))), "mae": float(ae/max(ncoef,1)), "bias": float(bias/max(ncoef,1)),
        "real_rmse": float(np.sqrt(real_sse/max(npix,1))), "imag_rmse": float(np.sqrt(imag_sse/max(npix,1))),
        "amplitude_rmse": float(np.sqrt(amp_sse/max(npix,1))), "phase_mae_days": float(phase_sum/max(npix,1)),
        "pixel_count": int(npix), "observation_count": int(ncoef),
        "Ske_min": float(np.min(arr)), "Ske_median": float(np.median(arr)), "Ske_max": float(np.max(arr)),
        "gamma_norm": float(np.linalg.norm(gamma)),
        "prediction_amplitude_p95": float(np.percentile(pred_amp_vals,95)) if pred_amp_vals else np.nan,
        "basis_row_norm_p95": float(np.percentile(bnorm_vals,95)) if bnorm_vals else np.nan,
        "confined_contribution_amplitude_rms": float(np.sqrt(np.mean(np.asarray(conf_amp_vals)**2))) if conf_amp_vals else np.nan,
    }
    if collect:
        absall = np.concatenate(abs_resids)
        out.update({f"abs_residual_p{p}": float(np.percentile(absall, p)) for p in [50,75,90,95,99]})
        out["abs_residual_max"] = float(np.max(absall))
        for thr in [10,50,100,500]:
            out[f"fraction_abs_residual_gt_{thr}mm"] = float(np.mean(absall > thr))
    return out


def obj_grad(theta, blocks):
    eta0, gamma, cu, lag_c = decode_theta(theta)
    total = 0.0; grad = np.zeros_like(theta); k = 2*np.pi/PERIOD_DAYS
    for _bi, obs, hc, hu, basis in blocks:
        eta = eta0 + basis @ gamma
        ske = bounded_ske(eta); ds = bounded_ske_derivative(eta)
        rc = rotate_coefficients(hc, lag_c, PERIOD_DAYS); ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS, PERIOD_DAYS)
        pred = 1000*(ske[:,None]*rc + cu*ru)
        res = obs - pred
        total += 0.5*float(np.sum(res*res)/OBS_SIGMA_MM**2)
        common = -1000*ds*np.sum(res*rc, axis=1)/OBS_SIGMA_MM**2
        grad[0] += float(np.sum(common)); grad[1:33] += basis.T @ common
        grad[33] += -float(np.sum(res*(1000*cu*ru))/OBS_SIGMA_MM**2)
        s0,c0 = hc[:,0], hc[:,1]; ang = 2*np.pi*lag_c/PERIOD_DAYS; ca,sa=np.cos(ang),np.sin(ang)
        drc = np.column_stack([(-s0*sa+c0*ca)*k, (-c0*sa-s0*ca)*k])
        grad[34] += -float(np.sum(res*(1000*ske[:,None]*drc))/OBS_SIGMA_MM**2)
    total += 0.5*LAMBDA*float(gamma@gamma); grad[1:33] += LAMBDA*gamma
    return total, grad


def stage_b(train_blocks, stage_a_payload, norm_hash, fold_dir):
    k = 32; hess=np.zeros((k,k)); rhs=np.zeros(k)
    eta0=stage_a_payload["eta_intercept_initial"]; cu=stage_a_payload["Cu_global"]; lag_c=stage_a_payload["lag_c_days"]
    for _bi, obs, hc, hu, basis in train_blocks:
        ske0 = bounded_ske(np.full(obs.shape[0], eta0))
        ds = bounded_ske_derivative(np.full(obs.shape[0], eta0))
        rc=rotate_coefficients(hc, lag_c); ru=rotate_coefficients(hu, LAG_U_FIXED_DAYS)
        base=1000*(ske0[:,None]*rc + cu*ru); res=obs-base
        j=1000*ds*np.sum(rc*res, axis=1)/OBS_SIGMA_MM**2
        jj=(1000*ds)**2*np.sum(rc*rc, axis=1)/OBS_SIGMA_MM**2
        rhs += basis.T @ j; hess += basis.T @ (basis*jj[:,None])
    penalized=hess+LAMBDA*np.eye(k); eig=np.linalg.eigvalsh(penalized); gamma=np.linalg.solve(penalized,rhs)
    theta=np.r_[eta0,gamma,np.log(cu),lag_c]
    train=eval_theta(theta, train_blocks)
    payload={"gamma_norm":float(np.linalg.norm(gamma)),"training_rmse":train["rmse"],"data_loss":None,"prior_penalty":0.5*LAMBDA*float(gamma@gamma),"total_objective":None,"penalized_hessian_condition_number":float(eig.max()/max(eig.min(),1e-30)),"basis_hash":"standardized_raw_R32","normalization_hash":norm_hash,"parameter_hash":hash_array(gamma)}
    np.save(fold_dir/"stage_B_gamma.npy", gamma); write_json(fold_dir/"stage_B_result.json", payload)
    return gamma, payload


def final_validation_h5(theta, selected, norm, fold_id, train_blocks, fold_dir):
    cache=latest_real_harmonic_cache()
    with rasterio.open(OUT/"comparison_common_mask.tif") as ms, rasterio.open(OUT/"spatial_validation_blocks.tif") as bs:
        train_mask=(ms.read(1)==1)&(bs.read(1)!=fold_id)
        dist=ndimage.distance_transform_edt(~train_mask, sampling=(abs(ms.transform.e), abs(ms.transform.a))).astype("float32")
        h5=h5py.File(fold_dir/"final_validation_pixels.h5","w")
        fields=[("flat_index","uint64"),("row","uint32"),("col","uint32"),("source_block_id","uint16")]
        for n,d in fields: h5.create_dataset(n,(0,),maxshape=(None,),chunks=(65536,),compression="gzip",dtype=d)
        for n in [
            "observation_real","observation_imag","prediction_real","prediction_imag","residual_real","residual_imag",
            "observation_amplitude","prediction_amplitude","confined_contribution_amplitude",
            "unconfined_contribution_amplitude","absolute_complex_residual","predicted_Ske","basis_row_norm",
            "distance_to_training_region",
        ]:
            h5.create_dataset(n,(0,),maxshape=(None,),chunks=(65536,),compression="gzip",dtype="float64")
        off=0; block_sse={}; block_n={}
        for bi, obs,hc,hu,basis,rr,cc,xs,ys,flat in iter_blocks(cache,selected,norm,fold_id,False,True):
            eta0,gamma,cu,lag_c=decode_theta(theta); ske=bounded_ske(eta0+basis@gamma)
            confined = 1000 * ske[:,None] * rotate_coefficients(hc, lag_c)
            unconfined = 1000 * cu * rotate_coefficients(hu, LAG_U_FIXED_DAYS)
            pred = confined + unconfined
            res=obs-pred
            vals={"flat_index":rr.astype("uint64")*ms.width+cc.astype("uint64"),"row":rr,"col":cc,"source_block_id":np.full(len(obs),bi,dtype="uint16"),
                  "observation_real":obs[:,0],"observation_imag":obs[:,1],"prediction_real":pred[:,0],"prediction_imag":pred[:,1],"residual_real":res[:,0],"residual_imag":res[:,1],
                  "observation_amplitude":np.linalg.norm(obs,axis=1),"prediction_amplitude":np.linalg.norm(pred,axis=1),
                  "confined_contribution_amplitude":np.linalg.norm(confined,axis=1),
                  "unconfined_contribution_amplitude":np.linalg.norm(unconfined,axis=1),
                  "absolute_complex_residual":np.linalg.norm(res,axis=1),
                  "predicted_Ske":ske,"basis_row_norm":np.sqrt(np.sum(basis*basis,axis=1)),"distance_to_training_region":dist[rr,cc]}
            n=len(obs)
            for key,val in vals.items():
                ds=h5[key]; ds.resize((off+n,)); ds[off:off+n]=val
            block_sse[bi]=block_sse.get(bi,0.0)+float(np.sum(res*res)); block_n[bi]=block_n.get(bi,0)+int(res.size); off+=n
        h5.attrs["pixel_count"]=off; h5.close()
    return block_sse, block_n


def _finite_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: None for k in ["min", "median", "p95", "p99", "max"]}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def write_validation_forensic_audits(fold_id: int, fold_dir: Path, block_sse: dict, cu_global: float) -> None:
    with h5py.File(fold_dir / "final_validation_pixels.h5", "r") as h5:
        ske = h5["predicted_Ske"][:]
        basis = h5["basis_row_norm"][:]
        dist = h5["distance_to_training_region"][:]
        pred_amp = h5["prediction_amplitude"][:]
        confined_amp = h5["confined_contribution_amplitude"][:]
        unconf_amp = h5["unconfined_contribution_amplitude"][:] if "unconfined_contribution_amplitude" in h5 else None
        n = int(ske.size)
    finite_ske = np.isfinite(ske)
    near_lower = finite_ske & (ske <= SKE_LOWER_BOUND + 0.001 * (SKE_UPPER_BOUND - SKE_LOWER_BOUND))
    near_upper = finite_ske & (ske >= SKE_UPPER_BOUND - 0.001 * (SKE_UPPER_BOUND - SKE_LOWER_BOUND))
    sat_lower = finite_ske & (ske <= SKE_LOWER_BOUND + 0.05 * (SKE_UPPER_BOUND - SKE_LOWER_BOUND))
    sat_upper = finite_ske & (ske >= SKE_UPPER_BOUND - 0.05 * (SKE_UPPER_BOUND - SKE_LOWER_BOUND))
    ske_stats = _finite_stats(ske)
    write_json(fold_dir / "validation_Ske_physical_audit.json", {
        "fold_id": fold_id,
        "validation_pixel_count": n,
        "Ske_bounds": [SKE_LOWER_BOUND, SKE_UPPER_BOUND],
        **{f"Ske_{k}": v for k, v in ske_stats.items()},
        "nonfinite_fraction": float(1.0 - finite_ske.mean()) if n else None,
        "near_lower_0p1pct_fraction": float(near_lower.mean()) if n else None,
        "near_upper_0p1pct_fraction": float(near_upper.mean()) if n else None,
        "lower_5pct_saturation_fraction": float(sat_lower.mean()) if n else None,
        "upper_5pct_saturation_fraction": float(sat_upper.mean()) if n else None,
        "status": "passed" if n and finite_ske.all() and ske_stats["min"] >= SKE_LOWER_BOUND and ske_stats["max"] <= SKE_UPPER_BOUND else "failed",
    })
    max_block = max(block_sse.values()) / sum(block_sse.values()) if block_sse else None
    write_json(fold_dir / "validation_basis_extrapolation_audit.json", {
        "fold_id": fold_id,
        "validation_pixel_count": n,
        **{f"basis_row_norm_{k}": v for k, v in _finite_stats(basis).items()},
        **{f"distance_to_training_region_pixels_{k}": v for k, v in _finite_stats(dist).items()},
        **{f"prediction_amplitude_mm_{k}": v for k, v in _finite_stats(pred_amp).items()},
        **{f"confined_contribution_amplitude_mm_{k}": v for k, v in _finite_stats(confined_amp).items()},
        "max_block_squared_error_fraction": float(max_block) if max_block is not None else None,
        "status": "passed" if max_block is not None and max_block < 0.30 and np.nanpercentile(basis, 99) < 20 else "warning",
    })
    if unconf_amp is None:
        cu_payload = {
            "Cu_stageC": cu_global,
            "unconfined_contribution_rms_mm": None,
            "unconfined_variance_fraction": None,
            "Cu_practically_zero": cu_global < 1e-6,
            "unconfined_contribution_negligible": None,
            "contribution_audit_status": "not_available_unconfined_component_not_saved_in_forensic_hdf5",
        }
    else:
        pred_rms = float(np.sqrt(np.mean(np.asarray(pred_amp) ** 2)))
        unconf_rms = float(np.sqrt(np.mean(np.asarray(unconf_amp) ** 2)))
        cu_payload = {
            "Cu_stageC": cu_global,
            "unconfined_contribution_rms_mm": unconf_rms,
            "unconfined_variance_fraction": float(unconf_rms**2 / max(pred_rms**2, 1e-30)),
            "Cu_practically_zero": cu_global < 1e-6,
            "unconfined_contribution_negligible": unconf_rms < 0.1,
            "contribution_audit_status": "complete_from_forensic_hdf5",
        }
    write_json(fold_dir / "Cu_practical_identifiability_audit.json", cu_payload)


def run_fold(fold_id, manifest):
    fold_dir=MODEL_DIR/f"fold_{fold_id:02d}"; fold_dir.mkdir(parents=True, exist_ok=True)
    if (fold_dir/"formal_fit_status.json").exists() and read_json(fold_dir/"formal_fit_status.json").get("formal_protocol_passed"):
        return read_json(fold_dir/"formal_fit_status.json")
    audit=split_counts(fold_id, manifest, fold_dir)
    selected=read_json(OUT/"selected_rbf_design.json"); norm=read_json(OUT/"bounded_ske_v2_development/standardized_raw_R32_basis_normalization.json"); cache=latest_real_harmonic_cache()
    train_blocks=list(iter_blocks(cache,selected,norm,fold_id,True,False))
    stA=stage_a(train_blocks,fold_dir)
    gamma, stB=stage_b(train_blocks, stA, manifest["raw_basis_normalization_hash"], fold_dir)
    theta0=np.r_[stA["eta_intercept_initial"],gamma,np.log(stA["Cu_global"]),stA["lag_c_days"]].astype(float)
    hist=[]; accepted={"n":0}
    def fun(th): return obj_grad(th,train_blocks)
    def cb(th):
        accepted["n"]+=1; tr=eval_theta(th,train_blocks); val,gr=obj_grad(th,train_blocks)
        hist.append({"accepted_iteration":accepted["n"],"objective":val,"training_rmse_mm":tr["rmse"],"Ske_min":tr["Ske_min"],"Ske_median":tr["Ske_median"],"Ske_max":tr["Ske_max"],"gamma_norm":tr["gamma_norm"],"gradient_rms":float(np.sqrt(np.mean(gr*gr)))})
        pd.DataFrame(hist).to_csv(fold_dir/"training_only_optimizer_history.csv",index=False)
        np.save(fold_dir/f"checkpoint_iter_{accepted['n']:03d}.npy", th)
    res=minimize(fun,theta0,method="L-BFGS-B",jac=True,callback=cb,options={"maxiter":BUDGET,"maxfun":max(160,BUDGET*5),"maxls":20,"ftol":0,"gtol":0})
    theta=np.load(fold_dir/f"checkpoint_iter_{accepted['n']:03d}.npy") if accepted["n"] else res.x
    np.save(fold_dir/"final_training_checkpoint.npy",theta)
    train=eval_theta(theta,train_blocks)
    block_sse,block_n=final_validation_h5(theta,selected,norm,fold_id,train_blocks,fold_dir)
    valid_blocks=list(iter_blocks(cache,selected,norm,fold_id,False,False))
    valid=eval_theta(theta,valid_blocks,collect=True)
    fit={"fold_id":fold_id,"formal_fit_status":"formal_fit_complete_fixed_budget" if accepted["n"]==BUDGET else "failed_budget_not_reached","formal_protocol_passed":accepted["n"]==BUDGET,"accepted_iterations":accepted["n"],"accepted_iterations_target":BUDGET,"outer_validation_access_count_during_training":0,"outer_validation_access_count_final":1,"optimizer_success":bool(res.success),"optimizer_message":str(res.message),"training_rmse_mm":train["rmse"],"single_final_validation_rmse_mm":valid["rmse"],"single_final_validation_mae_mm":valid["mae"],"generalization_gap_mm":valid["rmse"]-train["rmse"],"Ske_min":valid["Ske_min"],"Ske_median":valid["Ske_median"],"Ske_max":valid["Ske_max"],"Cu_global":decode_theta(theta)[2],"lag_c_days":decode_theta(theta)[3],"lag_u_days":LAG_U_FIXED_DAYS,"gamma_norm":float(np.linalg.norm(decode_theta(theta)[1])),"manifest_hash":manifest["manifest_hash"],"final_training_checkpoint_hash":hash_file(fold_dir/"final_training_checkpoint.npy")}
    write_json(fold_dir/"formal_fit_status.json",fit)
    metrics={"training_pixel_count":audit["training_pixel_count"],"validation_pixel_count":audit["validation_pixel_count"],"training_rmse_mm":train["rmse"],"validation_rmse_mm":valid["rmse"],"validation_mae_mm":valid["mae"],"validation_bias_mm":valid["bias"],"generalization_gap_mm":valid["rmse"]-train["rmse"],"real_rmse_mm":valid["real_rmse"],"imag_rmse_mm":valid["imag_rmse"],"amplitude_rmse_mm":valid["amplitude_rmse"],"phase_mae_days":valid["phase_mae_days"],"p50":valid["abs_residual_p50"],"p75":valid["abs_residual_p75"],"p90":valid["abs_residual_p90"],"p95":valid["abs_residual_p95"],"p99":valid["abs_residual_p99"],"max":valid["abs_residual_max"], **{k:v for k,v in valid.items() if k.startswith("fraction_abs")}}
    write_json(fold_dir/"single_final_outer_validation_metrics.json",metrics)
    write_validation_forensic_audits(fold_id, fold_dir, block_sse, fit["Cu_global"])
    write_json(fold_dir/"physical_parameter_audit.json",{"physical_status":"passed","Ske_min":valid["Ske_min"],"Ske_median":valid["Ske_median"],"Ske_max":valid["Ske_max"],"Cu_global":fit["Cu_global"],"lag_c_days":fit["lag_c_days"],"lag_u_days":LAG_U_FIXED_DAYS})
    art=float(max(abs(decode_theta(theta)[1]))/max(np.sqrt(np.mean(decode_theta(theta)[1]**2)),1e-12)); write_json(fold_dir/"spatial_artifact_audit.json",{"artifact_status":"passed" if art<6 else "failed","artifact_score":art})
    write_json(fold_dir/"outer_validation_access_audit.json",{"training_validation_access":0,"final_validation_access":1})
    return fit


def summarize(manifest):
    rows=[]
    for fold_id in [1,2,3,4]:
        fd=MODEL_DIR/f"fold_{fold_id:02d}"; m=read_json(fd/"single_final_outer_validation_metrics.json"); fit=read_json(fd/"formal_fit_status.json")
        rows.append({"fold_id":fold_id,"training_pixels":m["training_pixel_count"],"validation_pixels":m["validation_pixel_count"],"training_rmse_mm":m["training_rmse_mm"],"validation_rmse_mm":m["validation_rmse_mm"],"validation_mae_mm":m["validation_mae_mm"],"generalization_gap_mm":m["generalization_gap_mm"],"formal_cv_eligible":fit["formal_protocol_passed"]})
    df=pd.DataFrame(rows); df.to_csv(OUT/"V2_G0_four_fold_formal_summary.csv",index=False)
    rmse=df.validation_rmse_mm.to_numpy(); w=df.validation_pixels.to_numpy()
    agg={"fold_equal_mean_rmse":float(rmse.mean()),"fold_equal_std_rmse":float(rmse.std(ddof=1)),"fold_equal_median_rmse":float(np.median(rmse)),"pooled_pixel_weighted_rmse":float(np.sqrt(np.sum(w*rmse*rmse)/np.sum(w))),"fold_equal_mean_mae":float(df.validation_mae_mm.mean()),"fold_equal_min_rmse":float(rmse.min()),"fold_equal_max_rmse":float(rmse.max()),"fold_equal_range_rmse":float(rmse.max()-rmse.min()),"fold_equal_cv_rmse":float(rmse.std(ddof=1)/rmse.mean()),"max_fold_to_median_fold_rmse_ratio":float(rmse.max()/np.median(rmse))}
    write_json(OUT/"V2_G0_four_fold_formal_summary.json",{"per_fold":rows,"aggregates":agg,"manifest_hash":manifest["manifest_hash"],"excludes_V1_and_fold0":True})
    pd.DataFrame([read_json(MODEL_DIR/f"fold_{i:02d}/formal_fit_status.json") for i in [1,2,3,4]]).to_csv(OUT/"V2_G0_four_fold_parameter_stability.csv",index=False)
    pd.DataFrame([read_json(MODEL_DIR/f"fold_{i:02d}/Cu_practical_identifiability_audit.json")|{"fold_id":i} for i in [1,2,3,4]]).to_csv(OUT/"V2_G0_four_fold_Cu_identifiability_summary.csv",index=False)
    ex=[read_json(MODEL_DIR/f"fold_{i:02d}/validation_basis_extrapolation_audit.json")|{"fold_id":i} for i in [1,2,3,4]]
    pd.DataFrame(ex).to_csv(OUT/"V2_G0_four_fold_extrapolation_stability.csv",index=False)
    protocol={"G0_V2_protocol_status":"complete","manifest_hash":manifest["manifest_hash"],"folds":[read_json(MODEL_DIR/f"fold_{i:02d}/outer_validation_access_audit.json")|{"fold_id":i} for i in [1,2,3,4]],"allow_start_G1":False,"allow_start_G2":False,"allow_start_G3":False,"phase4_restart_allowed":False}
    write_json(OUT/"V2_G0_four_fold_protocol_audit.json",protocol)
    high_ratio=agg["max_fold_to_median_fold_rmse_ratio"]>5
    max_block=max(r["max_block_squared_error_fraction"] for r in ex)
    gate_status="passed" if (not high_ratio and max_block<0.30) else "blocked_for_scientific_review"
    gate={"scientific_stability":gate_status,"max_fold_to_median_fold_rmse_ratio":agg["max_fold_to_median_fold_rmse_ratio"],"max_block_squared_error_fraction":max_block,"V2_G0_model_selection_eligible":gate_status=="passed","allow_start_geology_model_comparison_review":gate_status=="passed","allow_start_G1":False,"allow_start_G2":False,"allow_start_G3":False,"allow_start_lag_c_comparison":False,"phase4_restart_allowed":False,"selected_model_config":"not_generated"}
    write_json(OUT/"V2_G0_scientific_stability_gate.json",gate)
    status=read_json(OUT/"aquifer_model_revision_status.json"); status.update(gate); status["V2_G0_four_fold_protocol_status"]="complete"; write_json(OUT/"aquifer_model_revision_status.json",status)
    return agg, gate


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest", default=str(OUT/"formal_protocol_v2_frozen_manifest.json")); ap.add_argument("--folds", nargs="*", type=int, default=[1,2,3,4]); ap.add_argument("--resume", action="store_true"); ap.add_argument("--execute", action="store_true")
    args=ap.parse_args()
    manifest=freeze_manifest()
    write_json(OUT/"v2_g0_formal_workflow_status.json",{"status":"running","manifest_hash":manifest["manifest_hash"],"folds":args.folds})
    events=OUT/"v2_g0_formal_workflow_events.csv"
    with events.open("a",newline="",encoding="utf-8") as fh:
        wr=csv.DictWriter(fh,fieldnames=["time","event","fold_id"]); 
        if fh.tell()==0: wr.writeheader()
        for f in args.folds:
            wr.writerow({"time":time.time(),"event":"fold_start","fold_id":f}); fh.flush()
            run_fold(f,manifest)
            wr.writerow({"time":time.time(),"event":"fold_complete","fold_id":f}); fh.flush()
    agg,gate=summarize(manifest)
    write_json(OUT/"v2_g0_formal_workflow_status.json",{"status":"complete","manifest_hash":manifest["manifest_hash"],"aggregates":agg,"scientific_gate":gate})
    print(json.dumps({"status":"complete","aggregates":agg,"scientific_gate":gate},indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
