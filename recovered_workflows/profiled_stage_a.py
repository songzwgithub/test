"""Profiled Stage A solver for the two-aquifer global model.

For fixed confined and unconfined lags, the global amplitudes are solved by a
2x2 non-negative weighted least-squares problem. The outer search therefore has
only two variables: lag_c_days and lag_u_days.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.optimize import minimize

from storage_inversion import rotate_coefficients


@dataclass(frozen=True)
class ProfiledStageAResult:
    lag_c_days: float
    lag_u_days: float
    ske_global: float
    cu_global: float
    objective: float
    rmse: float
    status: str
    irls_iterations: int = 0


@dataclass(frozen=True)
class StageAStats:
    n: int
    obs_yy: float
    hc_norm: float
    hu_norm: float
    hc_obs_cos: float
    hc_obs_sin: float
    hu_obs_cos: float
    hu_obs_sin: float
    cross_cos: float
    cross_sin: float
    observation_sigma_mm: float = 5.0
    period_days: float = 365.2425


def stage_a_sufficient_stats(obs, hc, hu, observation_sigma_mm=5.0, period_days=365.2425):
    obs = np.asarray(obs, float)
    hc = np.asarray(hc, float)
    hu = np.asarray(hu, float)
    invsig2 = 1.0 / max(observation_sigma_mm**2, 1e-30)
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
        observation_sigma_mm=float(observation_sigma_mm),
        period_days=float(period_days),
    )


def solve_from_stats(stats: StageAStats, lag_c_days, lag_u_days):
    ac = 2.0 * np.pi * float(lag_c_days) / stats.period_days
    au = 2.0 * np.pi * float(lag_u_days) / stats.period_days
    aty0 = np.cos(ac) * stats.hc_obs_cos + np.sin(ac) * stats.hc_obs_sin
    aty1 = np.cos(au) * stats.hu_obs_cos + np.sin(au) * stats.hu_obs_sin
    diff = ac - au
    ata01 = np.cos(diff) * stats.cross_cos + np.sin(diff) * stats.cross_sin
    ata = np.array([[stats.hc_norm, ata01], [ata01, stats.hu_norm]], float)
    aty = np.array([aty0, aty1], float)
    ske, cu = solve_nonnegative_2x2(ata, aty)
    amp = np.array([ske, cu])
    objective = 0.5 * float(stats.obs_yy - 2.0 * amp @ aty + amp @ ata @ amp)
    rmse = float(np.sqrt(max(2.0 * objective * stats.observation_sigma_mm**2, 0.0) / max(stats.n, 1)))
    return ProfiledStageAResult(float(lag_c_days), float(lag_u_days), float(ske), float(cu), objective, rmse, "ok", 0)


def coarse_profile_grid_from_stats(train_stats: StageAStats, validation_stats: StageAStats | None = None, lag_c_grid=None, lag_u_grid=None):
    lag_c_grid = np.arange(0.0, 91.0, 10.0) if lag_c_grid is None else np.asarray(lag_c_grid, float)
    lag_u_grid = np.arange(0.0, 91.0, 10.0) if lag_u_grid is None else np.asarray(lag_u_grid, float)
    t0 = perf_counter()
    rows = []
    for lag_c in lag_c_grid:
        for lag_u in lag_u_grid:
            r = solve_from_stats(train_stats, lag_c, lag_u)
            validation_rmse = np.nan
            if validation_stats is not None:
                vr = score_stats_with_amplitudes(validation_stats, lag_c, lag_u, r.ske_global, r.cu_global)
                validation_rmse = vr.rmse
            rows.append({
                "lag_c_days": r.lag_c_days,
                "lag_u_days": r.lag_u_days,
                "Ske_global": r.ske_global,
                "Cu_global": r.cu_global,
                "training_objective": r.objective,
                "training_rmse": r.rmse,
                "validation_rmse": validation_rmse,
                "status": r.status,
                "irls_iterations": 0,
            })
    df = pd.DataFrame(rows).sort_values("training_objective").reset_index(drop=True)
    df.attrs["grid_elapsed_seconds"] = perf_counter() - t0
    return df


def score_stats_with_amplitudes(stats: StageAStats, lag_c_days, lag_u_days, ske, cu):
    ac = 2.0 * np.pi * float(lag_c_days) / stats.period_days
    au = 2.0 * np.pi * float(lag_u_days) / stats.period_days
    aty0 = np.cos(ac) * stats.hc_obs_cos + np.sin(ac) * stats.hc_obs_sin
    aty1 = np.cos(au) * stats.hu_obs_cos + np.sin(au) * stats.hu_obs_sin
    diff = ac - au
    ata01 = np.cos(diff) * stats.cross_cos + np.sin(diff) * stats.cross_sin
    ata = np.array([[stats.hc_norm, ata01], [ata01, stats.hu_norm]], float)
    aty = np.array([aty0, aty1], float)
    amp = np.array([float(ske), float(cu)])
    objective = 0.5 * float(stats.obs_yy - 2.0 * amp @ aty + amp @ ata @ amp)
    rmse = float(np.sqrt(max(2.0 * objective * stats.observation_sigma_mm**2, 0.0) / max(stats.n, 1)))
    return ProfiledStageAResult(float(lag_c_days), float(lag_u_days), float(ske), float(cu), objective, rmse, "ok", 0)


def refine_profiled_lags_from_stats(train_stats: StageAStats, starts, bounds=((0.0, 365.2425), (0.0, 182.62125))):
    rows = []
    t0 = perf_counter()
    for i, start in enumerate(starts):
        x0 = np.array([float(start[0]), float(start[1])])
        center = x0.copy()
        best = None
        evals = 0
        for radius in (5.0, 2.0, 1.0):
            candidates = []
            for dc in (-radius, 0.0, radius):
                for du in (-radius, 0.0, radius):
                    lag_c = float(np.clip(center[0] + dc, bounds[0][0], bounds[0][1]))
                    lag_u = float(np.clip(center[1] + du, bounds[1][0], bounds[1][1]))
                    candidates.append(solve_from_stats(train_stats, lag_c, lag_u))
                    evals += 1
            best = min(candidates, key=lambda r: r.objective)
            center = np.array([best.lag_c_days, best.lag_u_days])
        rows.append({
            "start_id": int(i),
            "start_lag_c_days": float(x0[0]),
            "start_lag_u_days": float(x0[1]),
            "lag_c_days": best.lag_c_days,
            "lag_u_days": best.lag_u_days,
            "Ske_global": best.ske_global,
            "Cu_global": best.cu_global,
            "training_objective": best.objective,
            "training_rmse": best.rmse,
            "optimizer_success": True,
            "optimizer_message": "sufficient_statistics_three_radius_local_grid",
            "function_evaluations": int(evals),
            "status": best.status,
        })
    df = pd.DataFrame(rows).sort_values("training_objective").reset_index(drop=True)
    df.attrs["local_refinement_elapsed_seconds"] = perf_counter() - t0
    return df


def joint_confirm_from_stats(train_stats: StageAStats, best, maxiter=25):
    best = dict(best)
    x0 = np.array([
        np.log(max(float(best["Ske_global"]), 1e-12)),
        np.log(max(float(best["Cu_global"]), 1e-12)),
        float(best["lag_c_days"]),
        float(best["lag_u_days"]),
    ])

    def objective(x):
        return score_stats_with_amplitudes(train_stats, x[2], x[3], np.exp(np.clip(x[0], -30, 5)), np.exp(np.clip(x[1], -30, 5))).objective

    result = minimize(objective, x0, method="L-BFGS-B", bounds=[(-30, 5), (-30, 5), (0.0, 365.2425), (0.0, 182.62125)], options={"maxiter": int(maxiter), "ftol": 1e-8, "gtol": 1e-6})
    final = {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "initial_objective": float(objective(x0)),
        "final_objective": float(result.fun),
        "Ske_global": float(np.exp(result.x[0])),
        "Cu_global": float(np.exp(result.x[1])),
        "lag_c_days": float(result.x[2]),
        "lag_u_days": float(result.x[3]),
    }
    final["relative_objective_improvement"] = float((final["initial_objective"] - final["final_objective"]) / max(abs(final["initial_objective"]), 1.0))
    final["parameter_relative_change"] = float(np.linalg.norm(result.x - x0) / max(np.linalg.norm(x0), 1.0))
    return final


def _as_weight_vector(weights, n):
    if weights is None:
        return np.ones(n, float)
    weights = np.asarray(weights, float).reshape(-1)
    if weights.size != n:
        raise ValueError("weights length must match number of pixels")
    return weights


def _normal_equations(obs, hc_rot, hu_rot, weights=None):
    obs = np.asarray(obs, float)
    hc_rot = np.asarray(hc_rot, float)
    hu_rot = np.asarray(hu_rot, float)
    if obs.shape != hc_rot.shape or obs.shape != hu_rot.shape or obs.ndim != 2 or obs.shape[1] != 2:
        raise ValueError("obs, hc_rot, and hu_rot must be n x 2 arrays")
    w = _as_weight_vector(weights, obs.shape[0])
    a = 1000.0 * hc_rot
    b = 1000.0 * hu_rot
    ata00 = float(np.sum(w * np.sum(a * a, axis=1)))
    ata01 = float(np.sum(w * np.sum(a * b, axis=1)))
    ata11 = float(np.sum(w * np.sum(b * b, axis=1)))
    aty0 = float(np.sum(w * np.sum(a * obs, axis=1)))
    aty1 = float(np.sum(w * np.sum(b * obs, axis=1)))
    return np.array([[ata00, ata01], [ata01, ata11]], float), np.array([aty0, aty1], float)


def solve_nonnegative_2x2(ata, aty):
    ata = np.asarray(ata, float).reshape(2, 2)
    aty = np.asarray(aty, float).reshape(2)
    candidates = []
    try:
        x = np.linalg.solve(ata + np.eye(2) * 1e-18, aty)
        if np.all(x >= 0):
            candidates.append(x)
    except np.linalg.LinAlgError:
        pass
    for idx in (0, 1):
        x = np.zeros(2, float)
        denom = max(float(ata[idx, idx]), 1e-30)
        x[idx] = max(0.0, float(aty[idx] / denom))
        candidates.append(x)
    candidates.append(np.zeros(2, float))
    best = min(candidates, key=lambda v: 0.5 * float(v @ ata @ v - 2.0 * v @ aty))
    return np.maximum(best, 0.0)


def _loss_and_rmse(obs, hc_rot, hu_rot, ske, cu, weights=None, observation_sigma_mm=5.0, huber_delta=None):
    pred = 1000.0 * (float(ske) * hc_rot + float(cu) * hu_rot)
    residual = np.asarray(obs, float) - pred
    w = _as_weight_vector(weights, residual.shape[0])
    sq = np.sum(residual * residual, axis=1)
    rmse = float(np.sqrt(np.mean(sq)))
    if huber_delta is None:
        return 0.5 * float(np.sum(w * sq / max(observation_sigma_mm**2, 1e-30))), rmse
    mahal = np.sqrt(sq) / max(observation_sigma_mm, 1e-30)
    loss = np.where(mahal <= huber_delta, 0.5 * mahal * mahal, huber_delta * (mahal - 0.5 * huber_delta))
    return float(np.sum(w * loss)), rmse


def solve_fixed_lags(obs, hc, hu, lag_c_days, lag_u_days, weights=None, period_days=365.2425, observation_sigma_mm=5.0, huber_delta=None, max_irls=5, irls_tol=1e-4):
    hc_rot = rotate_coefficients(np.asarray(hc, float), lag_c_days, period_days)
    hu_rot = rotate_coefficients(np.asarray(hu, float), lag_u_days, period_days)
    base_weights = _as_weight_vector(weights, len(obs))
    if huber_delta is None:
        ata, aty = _normal_equations(obs, hc_rot, hu_rot, base_weights)
        ske, cu = solve_nonnegative_2x2(ata, aty)
        objective, rmse = _loss_and_rmse(obs, hc_rot, hu_rot, ske, cu, base_weights, observation_sigma_mm)
        return ProfiledStageAResult(lag_c_days, lag_u_days, float(ske), float(cu), objective, rmse, "ok", 0)
    irls_weights = base_weights.copy()
    previous = None
    ske = cu = 0.0
    iterations = 0
    for iterations in range(1, max_irls + 1):
        ata, aty = _normal_equations(obs, hc_rot, hu_rot, irls_weights)
        ske, cu = solve_nonnegative_2x2(ata, aty)
        pred = 1000.0 * (ske * hc_rot + cu * hu_rot)
        residual = np.asarray(obs, float) - pred
        mahal = np.sqrt(np.sum(residual * residual, axis=1)) / max(observation_sigma_mm, 1e-30)
        robust = np.ones_like(mahal)
        large = mahal > huber_delta
        robust[large] = huber_delta / np.maximum(mahal[large], 1e-12)
        irls_weights = base_weights * robust
        current = np.array([ske, cu])
        if previous is not None and np.linalg.norm(current - previous) / max(np.linalg.norm(previous), 1.0) < irls_tol:
            break
        previous = current
    objective, rmse = _loss_and_rmse(obs, hc_rot, hu_rot, ske, cu, base_weights, observation_sigma_mm, huber_delta)
    return ProfiledStageAResult(lag_c_days, lag_u_days, float(ske), float(cu), objective, rmse, "ok", iterations)


def coarse_profile_grid(obs, hc, hu, lag_c_grid=None, lag_u_grid=None, weights=None, validation=None, **kwargs):
    lag_c_grid = np.arange(0.0, 91.0, 10.0) if lag_c_grid is None else np.asarray(lag_c_grid, float)
    lag_u_grid = np.arange(0.0, 91.0, 10.0) if lag_u_grid is None else np.asarray(lag_u_grid, float)
    if kwargs.get("huber_delta") is None and weights is None:
        return coarse_profile_grid_squared_fast(obs, hc, hu, lag_c_grid, lag_u_grid, validation=validation, observation_sigma_mm=kwargs.get("observation_sigma_mm", 5.0))
    rows = []
    start = perf_counter()
    for lag_c in lag_c_grid:
        for lag_u in lag_u_grid:
            result = solve_fixed_lags(obs, hc, hu, lag_c, lag_u, weights=weights, **kwargs)
            row = {
                "lag_c_days": result.lag_c_days,
                "lag_u_days": result.lag_u_days,
                "Ske_global": result.ske_global,
                "Cu_global": result.cu_global,
                "training_objective": result.objective,
                "training_rmse": result.rmse,
                "validation_rmse": np.nan,
                "status": result.status,
                "irls_iterations": result.irls_iterations,
            }
            if validation is not None:
                vobs, vhc, vhu = validation
                vhc_rot = rotate_coefficients(np.asarray(vhc, float), result.lag_c_days, kwargs.get("period_days", 365.2425))
                vhu_rot = rotate_coefficients(np.asarray(vhu, float), result.lag_u_days, kwargs.get("period_days", 365.2425))
                _obj, vrmse = _loss_and_rmse(vobs, vhc_rot, vhu_rot, result.ske_global, result.cu_global, observation_sigma_mm=kwargs.get("observation_sigma_mm", 5.0))
                row["validation_rmse"] = vrmse
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("training_objective").reset_index(drop=True)
    df.attrs["grid_elapsed_seconds"] = perf_counter() - start
    return df


def _lag_terms(obs, h, lag_grid, observation_sigma_mm=5.0):
    obs = np.asarray(obs, float)
    out = []
    for lag in lag_grid:
        r = 1000.0 * rotate_coefficients(h, float(lag))
        out.append({
            "lag": float(lag),
            "rot": r,
            "ata": float(np.sum(r * r) / max(observation_sigma_mm**2, 1e-30)),
            "aty": float(np.sum(r * obs) / max(observation_sigma_mm**2, 1e-30)),
        })
    return out


def _profile_from_terms(obs_yy, cterm, uterm, observation_sigma_mm=5.0):
    ata01 = float(np.sum(cterm["rot"] * uterm["rot"]) / max(observation_sigma_mm**2, 1e-30))
    ata = np.array([[cterm["ata"], ata01], [ata01, uterm["ata"]]], float)
    aty = np.array([cterm["aty"], uterm["aty"]], float)
    ske, cu = solve_nonnegative_2x2(ata, aty)
    objective = 0.5 * float(obs_yy - 2.0 * np.dot([ske, cu], aty) + np.array([ske, cu]) @ ata @ np.array([ske, cu]))
    return float(ske), float(cu), objective


def coarse_profile_grid_squared_fast(obs, hc, hu, lag_c_grid, lag_u_grid, validation=None, observation_sigma_mm=5.0):
    start = perf_counter()
    obs = np.asarray(obs, float)
    obs_yy = float(np.sum(obs * obs) / max(observation_sigma_mm**2, 1e-30))
    cterms = _lag_terms(obs, hc, lag_c_grid, observation_sigma_mm)
    uterms = _lag_terms(obs, hu, lag_u_grid, observation_sigma_mm)
    vobs_yy = None
    vcterms = vuterms = None
    if validation is not None:
        vobs, vhc, vhu = validation
        vobs = np.asarray(vobs, float)
        vobs_yy = float(np.sum(vobs * vobs) / max(observation_sigma_mm**2, 1e-30))
        vcterms = _lag_terms(vobs, vhc, lag_c_grid, observation_sigma_mm)
        vuterms = _lag_terms(vobs, vhu, lag_u_grid, observation_sigma_mm)
    rows = []
    n = len(obs)
    vn = len(validation[0]) if validation is not None else 0
    for i, cterm in enumerate(cterms):
        for j, uterm in enumerate(uterms):
            ske, cu, objective = _profile_from_terms(obs_yy, cterm, uterm, observation_sigma_mm)
            train_sse = max(2.0 * objective * observation_sigma_mm**2, 0.0)
            validation_rmse = np.nan
            if validation is not None:
                _vske, _vcu, vobjective = _profile_from_terms(vobs_yy, vcterms[i], vuterms[j], observation_sigma_mm)
                # Score validation using training amplitudes, not validation-fitted amplitudes.
                vata01 = float(np.sum(vcterms[i]["rot"] * vuterms[j]["rot"]) / max(observation_sigma_mm**2, 1e-30))
                vata = np.array([[vcterms[i]["ata"], vata01], [vata01, vuterms[j]["ata"]]], float)
                vaty = np.array([vcterms[i]["aty"], vuterms[j]["aty"]], float)
                amp = np.array([ske, cu])
                vobj = 0.5 * float(vobs_yy - 2.0 * amp @ vaty + amp @ vata @ amp)
                validation_rmse = float(np.sqrt(max(2.0 * vobj * observation_sigma_mm**2, 0.0) / max(vn, 1)))
            rows.append({
                "lag_c_days": float(cterm["lag"]),
                "lag_u_days": float(uterm["lag"]),
                "Ske_global": ske,
                "Cu_global": cu,
                "training_objective": objective,
                "training_rmse": float(np.sqrt(train_sse / max(n, 1))),
                "validation_rmse": validation_rmse,
                "status": "ok",
                "irls_iterations": 0,
            })
    df = pd.DataFrame(rows).sort_values("training_objective").reset_index(drop=True)
    df.attrs["grid_elapsed_seconds"] = perf_counter() - start
    return df


def refine_profiled_lags(obs, hc, hu, starts, bounds=((0.0, 365.2425), (0.0, 182.62125)), weights=None, **kwargs):
    rows = []
    start_time = perf_counter()
    for i, start in enumerate(starts):
        x0 = np.array([float(start[0]), float(start[1])])
        best = None
        evals = 0
        center = x0.copy()
        for radius in (5.0, 2.0):
            candidates = []
            for dc in (-radius, 0.0, radius):
                for du in (-radius, 0.0, radius):
                    lag_c = float(np.clip(center[0] + dc, bounds[0][0], bounds[0][1]))
                    lag_u = float(np.clip(center[1] + du, bounds[1][0], bounds[1][1]))
                    result = solve_fixed_lags(obs, hc, hu, lag_c, lag_u, weights=weights, **kwargs)
                    candidates.append(result)
                    evals += 1
            best = min(candidates, key=lambda r: r.objective)
            center = np.array([best.lag_c_days, best.lag_u_days])
        result = best
        rows.append({
            "start_id": int(i),
            "start_lag_c_days": float(x0[0]),
            "start_lag_u_days": float(x0[1]),
            "lag_c_days": result.lag_c_days,
            "lag_u_days": result.lag_u_days,
            "Ske_global": result.ske_global,
            "Cu_global": result.cu_global,
            "training_objective": result.objective,
            "training_rmse": result.rmse,
            "optimizer_success": True,
            "optimizer_message": "deterministic_two_radius_local_grid",
            "function_evaluations": int(evals),
            "status": result.status,
        })
    df = pd.DataFrame(rows).sort_values("training_objective").reset_index(drop=True)
    df.attrs["local_refinement_elapsed_seconds"] = perf_counter() - start_time
    return df


def latest_real_harmonic_cache(root=Path("outputs/cache")):
    complete = sorted(Path(root).glob("phase4_harmonic_blocks_*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in complete:
        if path.name.endswith(".tmp"):
            continue
        return path
    raise FileNotFoundError("No phase4 harmonic block cache found")


def load_real_fold_arrays(cache_path, mask_path, block_path, fold_id=0, train=True, dtype="float32", progress_label=None):
    obs_parts = []
    hc_parts = []
    hu_parts = []
    with h5py.File(cache_path, "r") as h5, rasterio.open(mask_path) as mask_src, rasterio.open(block_path) as block_src:
        nblocks = len(h5["block_start"])
        pixels = 0
        t0 = perf_counter()
        for bi, start in enumerate(h5["block_start"][:]):
            if progress_label and (bi == 0 or (bi + 1) % 10 == 0):
                print(f"{progress_label} load_block={bi+1}/{nblocks} kept_pixels={pixels} elapsed_s={perf_counter()-t0:.1f}", flush=True)
            count = int(h5["block_count"][bi])
            if count == 0:
                continue
            start = int(start)
            r = int(h5["block_row"][bi])
            c = int(h5["block_col"][bi])
            h = int(h5["block_height"][bi])
            w = int(h5["block_width"][bi])
            window = Window(c, r, w, h)
            flat = h5["flat_index"][start:start + count].astype(int)
            mask = mask_src.read(1, window=window).ravel()[flat] == 1
            folds = block_src.read(1, window=window).ravel()[flat]
            take = folds != fold_id if train else folds == fold_id
            obs = h5["obs"][start:start + count]
            hc = h5["hc"][start:start + count]
            hu = h5["hu"][start:start + count]
            common = mask & take & np.isfinite(obs).all(1) & np.isfinite(hc).all(1) & np.isfinite(hu).all(1)
            if not common.any():
                continue
            pixels += int(common.sum())
            obs_parts.append(obs[common].astype(dtype, copy=False))
            hc_parts.append(hc[common].astype(dtype, copy=False))
            hu_parts.append(hu[common].astype(dtype, copy=False))
    if not obs_parts:
        raise RuntimeError("No real Stage A pixels found for requested fold/train split")
    obs = np.vstack(obs_parts)
    hc = np.vstack(hc_parts)
    hu = np.vstack(hu_parts)
    if progress_label:
        print(f"{progress_label} load_done pixels={len(obs)} elapsed_s={perf_counter()-t0:.1f}", flush=True)
    return obs, hc, hu


def joint_confirm(obs, hc, hu, best, maxiter=25, observation_sigma_mm=5.0):
    best = dict(best)
    x0 = np.array([
        np.log(max(float(best["Ske_global"]), 1e-12)),
        np.log(max(float(best["Cu_global"]), 1e-12)),
        float(best["lag_c_days"]),
        float(best["lag_u_days"]),
    ])

    def objective(x):
        ske = float(np.exp(np.clip(x[0], -30, 5)))
        cu = float(np.exp(np.clip(x[1], -30, 5)))
        lag_c = float(x[2])
        lag_u = float(x[3])
        hc_rot = rotate_coefficients(hc, lag_c)
        hu_rot = rotate_coefficients(hu, lag_u)
        obj, _rmse = _loss_and_rmse(obs, hc_rot, hu_rot, ske, cu, observation_sigma_mm=observation_sigma_mm)
        return obj

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=[(-30, 5), (-30, 5), (0.0, 365.2425), (0.0, 182.62125)],
        options={"maxiter": int(maxiter), "ftol": 1e-8, "gtol": 1e-6},
    )
    final = {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "initial_objective": float(objective(x0)),
        "final_objective": float(result.fun),
        "Ske_global": float(np.exp(result.x[0])),
        "Cu_global": float(np.exp(result.x[1])),
        "lag_c_days": float(result.x[2]),
        "lag_u_days": float(result.x[3]),
    }
    final["relative_objective_improvement"] = float((final["initial_objective"] - final["final_objective"]) / max(abs(final["initial_objective"]), 1.0))
    final["parameter_relative_change"] = float(np.linalg.norm(result.x - x0) / max(np.linalg.norm(x0), 1.0))
    return final


def run_real_fold0_stage_a(output_root="outputs/aquifer_model_revision", cache_path=None, huber_delta=None):
    output_root = Path(output_root)
    fold_dir = output_root / "model_compare" / "G0_no_geology_L0_shared" / "fold_00" / "stage_A"
    fold_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_path) if cache_path else latest_real_harmonic_cache()
    mask_path = output_root / "comparison_common_mask.tif"
    block_path = output_root / "spatial_validation_blocks.tif"
    t0 = perf_counter()
    obs, hc, hu = load_real_fold_arrays(cache_path, mask_path, block_path, fold_id=0, train=True, progress_label="stage_A_train")
    vobs, vhc, vhu = load_real_fold_arrays(cache_path, mask_path, block_path, fold_id=0, train=False, progress_label="stage_A_validation")
    load_seconds = perf_counter() - t0
    print("stage_A_sufficient_stats_start", flush=True)
    train_stats = stage_a_sufficient_stats(obs, hc, hu)
    validation_stats = stage_a_sufficient_stats(vobs, vhc, vhu)
    print("stage_A_sufficient_stats_done", flush=True)

    print("stage_A_profile_grid_start", flush=True)
    if huber_delta is not None:
        print("stage_A_huber_requested_falling_back_to_array_grid_for_irls", flush=True)
        grid = coarse_profile_grid(
            obs,
            hc,
            hu,
            lag_c_grid=np.arange(0, 91, 10),
            lag_u_grid=np.arange(0, 91, 10),
            validation=(vobs, vhc, vhu),
            huber_delta=huber_delta,
        )
    else:
        grid = coarse_profile_grid_from_stats(train_stats, validation_stats, lag_c_grid=np.arange(0, 91, 10), lag_u_grid=np.arange(0, 91, 10))
    grid_seconds = float(grid.attrs.get("grid_elapsed_seconds", np.nan))
    grid.to_csv(fold_dir / "stage_A_profile_grid.csv", index=False)
    print(f"stage_A_profile_grid_done rows={len(grid)} elapsed_s={grid_seconds:.1f}", flush=True)

    starts = grid[["lag_c_days", "lag_u_days"]].head(5).to_numpy()
    print("stage_A_local_refinement_start", flush=True)
    refined = refine_profiled_lags(obs, hc, hu, starts=starts, huber_delta=huber_delta) if huber_delta is not None else refine_profiled_lags_from_stats(train_stats, starts=starts)
    refine_seconds = float(refined.attrs.get("local_refinement_elapsed_seconds", np.nan))
    refined.to_csv(fold_dir / "stage_A_profiled_multistart_summary.csv", index=False)
    print(f"stage_A_local_refinement_done rows={len(refined)} elapsed_s={refine_seconds:.1f}", flush=True)
    best = refined.iloc[0].to_dict()
    print("stage_A_joint_confirmation_start", flush=True)
    joint = joint_confirm(obs, hc, hu, best) if huber_delta is not None else joint_confirm_from_stats(train_stats, best)
    print("stage_A_joint_confirmation_done", flush=True)
    rel = float(joint["relative_objective_improvement"])
    lag_near_boundary = min(best["lag_c_days"], 365.2425 - best["lag_c_days"]) / 365.2425 < 0.001 or min(best["lag_u_days"], 182.62125 - best["lag_u_days"]) / 182.62125 < 0.001
    positive = best["Ske_global"] > 0 and best["Cu_global"] > 0
    top = refined.head(5)
    objective_span = (top["training_objective"].max() - top["training_objective"].min()) / max(abs(top["training_objective"].min()), 1.0)
    lag_span = max(top["lag_c_days"].max() - top["lag_c_days"].min(), top["lag_u_days"].max() - top["lag_u_days"].min())
    multimodal = bool(objective_span > 1e-3 or lag_span > 10)
    disagreement = bool(rel > 0.001 and joint["parameter_relative_change"] > 0.01)
    if disagreement:
        status = "profile_joint_disagreement"
    elif not positive or lag_near_boundary:
        status = "failed"
    elif multimodal:
        status = "complete_multimodal_warning"
    else:
        status = "complete_converged"
    payload = {
        "stage_A_status": status,
        "stage_A_uses_rbf_design": False,
        "objective_loss": "huber_irls" if huber_delta is not None else "squared_loss",
        "huber_delta": huber_delta,
        "train_pixel_count": int(obs.shape[0]),
        "validation_pixel_count": int(vobs.shape[0]),
        "load_seconds": float(load_seconds),
        "profile_grid_evaluation_seconds": grid_seconds,
        "local_refinement_seconds": refine_seconds,
        "joint_confirmation": joint,
        "joint_confirmation_relative_improvement": rel,
        "stage_A_profile_joint_disagreement": disagreement,
        "multimodal_warning": multimodal,
        "best_profiled_result": best,
        "Ske_global": float(best["Ske_global"]),
        "Cu_global": float(best["Cu_global"]),
        "lag_c_days": float(best["lag_c_days"]),
        "lag_u_days": float(best["lag_u_days"]),
        "total_elapsed_seconds": float(perf_counter() - t0),
    }
    (fold_dir / "stage_A_best_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-real-fold0", action="store_true")
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    args = parser.parse_args()
    if not args.run_real_fold0:
        parser.error("Use --run-real-fold0 to execute the real fold0 profiled Stage A")
    payload = run_real_fold0_stage_a(args.output_root, args.cache, args.huber_delta)
    print(json.dumps({
        "stage_A_status": payload["stage_A_status"],
        "Ske_global": payload["Ske_global"],
        "Cu_global": payload["Cu_global"],
        "lag_c_days": payload["lag_c_days"],
        "lag_u_days": payload["lag_u_days"],
        "joint_confirmation_relative_improvement": payload["joint_confirmation_relative_improvement"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
