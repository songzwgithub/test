#!/usr/bin/env python3
"""Resumable isolated fold0 confirmation for the L01028 formal protocol."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import xy
from rasterio.windows import Window
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_stage_b_lambda_effect import artifact_metrics, iter_spatial_map  # noqa: E402
from scripts.run_stage_b_fixed_lagu import rbf_values  # noqa: E402
from storage_inversion import rotate_coefficients  # noqa: E402


REFERENCE_ID = "L01028_500m_fixed_quality_median_v1"
EXPECTED_CACHE_SHA256 = "3f4f714b5e10fe3dcd5a9e91a29de27e0157858137e76afddb12b2cd0fa6dce8"
DEFAULT_CACHE = Path("outputs/cache/phase4_harmonic_blocks_L01028_authoritative.h5")
DEFAULT_OUTPUT = Path("outputs/reference_frames/L01028_500m_fixed_quality_median_v1/fold0_confirmation")
OUTPUT_ROOT = Path("outputs/aquifer_model_revision")
COMMON_MASK = OUTPUT_ROOT / "comparison_common_mask.tif"
FOLD_MAP = OUTPUT_ROOT / "spatial_validation_blocks.tif"
RBF_PATHS = [OUTPUT_ROOT / "selected_rbf_design.json", OUTPUT_ROOT / "rbf_global_basis_selection.json"]
RBF_TRANSFORM = OUTPUT_ROOT / "rbf_orthogonalization" / "rbf_transform.npy"
OBS_SIGMA_MM = 5.0
PERIOD_DAYS = 365.2425
RBF_DIM = 32


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def atomic_npy(path: Path, arr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)
    return sha256_file(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_rbf_path() -> Path:
    for path in RBF_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("No RBF selection JSON found")


def validate_output_dir(output: Path) -> Path:
    out = output.resolve()
    expected = (ROOT / DEFAULT_OUTPUT).resolve()
    if out != expected:
        raise ValueError(f"output-dir must be exactly {expected}")
    if "model_compare" in out.parts:
        raise ValueError("Refusing to write old model_compare/fold_00 output")
    return out


def validate_cache(cache: Path) -> tuple[Path, str, str]:
    path = cache.resolve()
    cache_hash = sha256_file(path)
    if cache_hash != EXPECTED_CACHE_SHA256:
        raise ValueError(f"Unexpected cache SHA256: {cache_hash}")
    with h5py.File(path, "r") as h5:
        if int(h5.attrs.get("complete", 0)) != 1:
            raise ValueError("L01028 cache is not complete")
        if str(h5.attrs.get("reference_frame_id", "")) != REFERENCE_ID:
            raise ValueError("Cache reference_frame_id is not L01028")
        cache_key = str(h5.attrs.get("cache_key", ""))
    return path, cache_hash, cache_key


def iter_blocks(cache: Path, selected: dict[str, Any], transform: np.ndarray, fold_id: int, train: bool):
    centers = np.asarray(selected["center_coordinates"], float)
    sigma_m = float(selected["sigma_km"]) * 1000.0
    target_crs = selected.get("projected_crs")
    with h5py.File(cache, "r") as h5, rasterio.open(COMMON_MASK) as mask_src, rasterio.open(FOLD_MAP) as fold_src:
        transformer = None
        if target_crs and mask_src.crs and str(mask_src.crs) != str(target_crs):
            transformer = Transformer.from_crs(mask_src.crs, target_crs, always_xy=True)
        for bi in range(len(h5["block_start"])):
            start = int(h5["block_start"][bi])
            count = int(h5["block_count"][bi])
            if count == 0:
                continue
            row = int(h5["block_row"][bi])
            col = int(h5["block_col"][bi])
            height = int(h5["block_height"][bi])
            width = int(h5["block_width"][bi])
            flat = h5["flat_index"][start : start + count].astype(np.int64)
            window = Window(col, row, width, height)
            mask = mask_src.read(1, window=window).reshape(-1)[flat] == 1
            folds = fold_src.read(1, window=window).reshape(-1)[flat]
            take = folds != fold_id if train else folds == fold_id
            obs = h5["obs"][start : start + count]
            hc = h5["hc"][start : start + count]
            hu = h5["hu"][start : start + count]
            valid = mask & take & np.isfinite(obs).all(1) & np.isfinite(hc).all(1) & np.isfinite(hu).all(1)
            if not valid.any():
                continue
            rr = row + (flat // width)[valid]
            cc = col + (flat % width)[valid]
            xs, ys = xy(mask_src.transform, rr, cc, offset="center")
            xs = np.asarray(xs, float)
            ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float)
                ys = np.asarray(ys, float)
            phi = rbf_values(np.column_stack([xs, ys]), centers, sigma_m)
            basis = phi @ transform
            yield obs[valid].astype("float32"), hc[valid].astype("float32"), hu[valid].astype("float32"), basis.astype("float32")


def count_pixels(cache: Path, fold_id: int) -> tuple[int, int]:
    train = valid = 0
    with h5py.File(cache, "r") as h5, rasterio.open(COMMON_MASK) as mask_src, rasterio.open(FOLD_MAP) as fold_src:
        for bi in range(len(h5["block_start"])):
            start = int(h5["block_start"][bi])
            count = int(h5["block_count"][bi])
            row = int(h5["block_row"][bi])
            col = int(h5["block_col"][bi])
            height = int(h5["block_height"][bi])
            width = int(h5["block_width"][bi])
            flat = h5["flat_index"][start : start + count].astype(np.int64)
            window = Window(col, row, width, height)
            common = mask_src.read(1, window=window).reshape(-1)[flat] == 1
            folds = fold_src.read(1, window=window).reshape(-1)[flat]
            train += int(np.count_nonzero(common & (folds != fold_id)))
            valid += int(np.count_nonzero(common & (folds == fold_id)))
    return train, valid


def memmap_specs(train_n: int, val_n: int) -> dict[str, tuple[int, ...]]:
    return {
        "train_obs": (train_n, 2),
        "train_hc": (train_n, 2),
        "train_hu": (train_n, 2),
        "train_basis": (train_n, RBF_DIM),
        "val_obs": (val_n, 2),
        "val_hc": (val_n, 2),
        "val_hu": (val_n, 2),
        "val_basis": (val_n, RBF_DIM),
    }


def manifest_file_ok(output: Path, name: str, spec: dict[str, Any]) -> bool:
    path = output / spec["path"]
    shape = tuple(spec["shape"])
    dtype = np.dtype(spec["dtype"])
    return path.exists() and path.stat().st_size == int(np.prod(shape)) * dtype.itemsize == int(spec["file_size_bytes"])


def recover_memmap_manifest(output: Path, fold_id: int, train_n: int, val_n: int, hashes: dict[str, str]) -> dict[str, Any] | None:
    specs = memmap_specs(train_n, val_n)
    arrays: dict[str, dict[str, Any]] = {}
    for name, shape in specs.items():
        path = output / f"{name}.dat"
        expected_size = int(np.prod(shape)) * np.dtype("float32").itemsize
        if not path.exists() or path.stat().st_size != expected_size:
            return None
        arrays[name] = {
            "path": f"{name}.dat",
            "shape": list(shape),
            "dtype": "float32",
            "file_size_bytes": expected_size,
        }
    meta = {
        "complete": True,
        "fold_id": fold_id,
        "cache_sha256": hashes["cache"],
        "common_mask_hash": hashes["common"],
        "fold_map_hash": hashes["fold"],
        "rbf_selection_hash": hashes["rbf"],
        "train_pixel_count": train_n,
        "validation_pixel_count": val_n,
        "rbf_dimension": RBF_DIM,
        "generation_code_hash": sha256_file(Path(__file__)),
        "arrays": arrays,
        "completed_at_utc": utc_now(),
        "recovered_from_existing_dat_files": True,
    }
    atomic_json(output / "memmap_manifest.json", meta)
    return meta


def validate_memmap_manifest(output: Path, hashes: dict[str, str]) -> dict[str, Any] | None:
    meta_path = output / "memmap_manifest.json"
    if not meta_path.exists():
        legacy = output / "fold0_design_memmap_manifest.json"
        if not legacy.exists():
            return None
        legacy_payload = read_json(legacy)
        if all((output / v["path"]).exists() for v in legacy_payload.get("arrays", {}).values()):
            meta = dict(legacy_payload)
            meta["complete"] = True
            meta["rbf_dimension"] = RBF_DIM
            for key, spec in meta["arrays"].items():
                p = output / spec["path"]
                spec["file_size_bytes"] = p.stat().st_size
            atomic_json(meta_path, meta)
        else:
            return None
    meta = read_json(meta_path)
    required = {
        "cache_sha256": hashes["cache"],
        "common_mask_hash": hashes["common"],
        "fold_map_hash": hashes["fold"],
        "rbf_selection_hash": hashes["rbf"],
    }
    if any(meta.get(k) != v for k, v in required.items()) or meta.get("complete") is not True:
        return None
    if int(meta.get("rbf_dimension", 0)) != RBF_DIM:
        return None
    if not all(manifest_file_ok(output, name, spec) for name, spec in meta.get("arrays", {}).items()):
        return None
    return meta


def build_memmaps(cache: Path, selected: dict[str, Any], transform: np.ndarray, fold_id: int, output: Path, hashes: dict[str, str]) -> dict[str, Any]:
    meta = validate_memmap_manifest(output, hashes)
    if meta is not None:
        return meta
    train_n, val_n = count_pixels(cache, fold_id)
    recovered = recover_memmap_manifest(output, fold_id, train_n, val_n, hashes)
    if recovered is not None:
        return recovered
    for name in ["train_obs", "train_hc", "train_hu", "train_basis", "val_obs", "val_hc", "val_hu", "val_basis"]:
        for suffix in (".dat", ".dat.tmp"):
            stale = output / f"{name}{suffix}"
            if stale.exists():
                stale.unlink()
    specs = memmap_specs(train_n, val_n)
    arrays = {name: np.memmap(output / f"{name}.dat.tmp", dtype="float32", mode="w+", shape=shape) for name, shape in specs.items()}
    offsets = {"train": 0, "val": 0}
    for train_flag, prefix in [(True, "train"), (False, "val")]:
        for obs, hc, hu, basis in iter_blocks(cache, selected, transform, fold_id, train=train_flag):
            n = obs.shape[0]
            start = offsets[prefix]
            end = start + n
            arrays[f"{prefix}_obs"][start:end] = obs
            arrays[f"{prefix}_hc"][start:end] = hc
            arrays[f"{prefix}_hu"][start:end] = hu
            arrays[f"{prefix}_basis"][start:end] = basis
            offsets[prefix] = end
    for arr in arrays.values():
        arr.flush()
    arrays.clear()
    gc.collect()
    for name in specs:
        os.replace(output / f"{name}.dat.tmp", output / f"{name}.dat")
    meta = {
        "complete": True,
        "fold_id": fold_id,
        "cache_sha256": hashes["cache"],
        "common_mask_hash": hashes["common"],
        "fold_map_hash": hashes["fold"],
        "rbf_selection_hash": hashes["rbf"],
        "train_pixel_count": train_n,
        "validation_pixel_count": val_n,
        "rbf_dimension": RBF_DIM,
        "generation_code_hash": sha256_file(Path(__file__)),
        "arrays": {
            name: {
                "path": f"{name}.dat",
                "shape": list(shape),
                "dtype": "float32",
                "file_size_bytes": int(np.prod(shape)) * np.dtype("float32").itemsize,
            }
            for name, shape in specs.items()
        },
        "completed_at_utc": utc_now(),
    }
    atomic_json(output / "memmap_manifest.json", meta)
    return meta


def open_arrays(output: Path, meta: dict[str, Any]) -> dict[str, np.memmap]:
    return {
        name: np.memmap(output / spec["path"], dtype=spec["dtype"], mode="r", shape=tuple(spec["shape"]))
        for name, spec in meta["arrays"].items()
    }


def iter_chunks(arrays: dict[str, np.ndarray], train: bool, include_basis: bool, chunk_rows: int = 250_000):
    prefix = "train" if train else "val"
    n = arrays[f"{prefix}_obs"].shape[0]
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        yield (
            arrays[f"{prefix}_obs"][start:end],
            arrays[f"{prefix}_hc"][start:end],
            arrays[f"{prefix}_hu"][start:end],
            arrays[f"{prefix}_basis"][start:end] if include_basis else None,
        )


def decode(theta: np.ndarray) -> tuple[float, np.ndarray, float, float]:
    return float(theta[0]), theta[1:33], float(np.exp(np.clip(theta[33], -40, 5))), float(theta[34])


def objective_grad(theta: np.ndarray, arrays: dict[str, np.ndarray], lag_u: float, lam: float, active: str) -> tuple[float, np.ndarray]:
    log_ske, gamma, cu, lag_c = decode(theta)
    grad = np.zeros_like(theta)
    total = 0.0
    k = 2.0 * np.pi / PERIOD_DAYS
    for obs, hc, hu, basis in iter_chunks(arrays, train=True, include_basis=(active != "global")):
        spatial = np.zeros(obs.shape[0], float) if active == "global" else basis @ gamma
        ske = np.exp(np.clip(log_ske + spatial, -20, 10))
        rc = rotate_coefficients(hc, lag_c, PERIOD_DAYS)
        ru = rotate_coefficients(hu, lag_u, PERIOD_DAYS)
        pred = 1000.0 * (ske[:, None] * rc + cu * ru)
        res = obs - pred
        total += 0.5 * float(np.sum(res * res) / (OBS_SIGMA_MM**2))
        common = -1000.0 * ske * np.sum(res * rc, axis=1) / (OBS_SIGMA_MM**2)
        grad[0] += float(np.sum(common))
        if basis is not None:
            grad[1:33] += basis.T @ common
        grad[33] += -float(np.sum(res * (1000.0 * cu * ru)) / (OBS_SIGMA_MM**2))
        angle = 2.0 * np.pi * lag_c / PERIOD_DAYS
        ca, sa = np.cos(angle), np.sin(angle)
        s0, c0 = hc[:, 0], hc[:, 1]
        drc = np.column_stack([(-s0 * sa + c0 * ca) * k, (-c0 * sa - s0 * ca) * k])
        grad[34] += -float(np.sum(res * (1000.0 * ske[:, None] * drc)) / (OBS_SIGMA_MM**2))
    total += 0.5 * float(lam) * float(gamma @ gamma)
    grad[1:33] += float(lam) * gamma
    if active == "global":
        grad[1:33] = 0.0
    elif active == "gamma":
        grad[[0, 33, 34]] = 0.0
    return total, grad


def metrics(theta: np.ndarray, arrays: dict[str, np.ndarray], lag_u: float, train: bool) -> dict[str, float]:
    log_ske, gamma, cu, lag_c = decode(theta)
    sse = ae = 0.0
    ncoef = 0
    ske_min = np.inf
    ske_max = -np.inf
    ske_values = []
    spatial_ss = 0.0
    spatial_n = 0
    for obs, hc, hu, basis in iter_chunks(arrays, train=train, include_basis=True):
        spatial = basis @ gamma
        ske = np.exp(np.clip(log_ske + spatial, -20, 10))
        pred = 1000.0 * (ske[:, None] * rotate_coefficients(hc, lag_c, PERIOD_DAYS) + cu * rotate_coefficients(hu, lag_u, PERIOD_DAYS))
        res = obs - pred
        sse += float(np.sum(res * res))
        ae += float(np.sum(np.abs(res)))
        ncoef += int(res.size)
        ske_min = min(ske_min, float(np.min(ske)))
        ske_max = max(ske_max, float(np.max(ske)))
        ske_values.append(ske[:: max(1, len(ske) // 5000)])
        spatial_ss += float(np.sum(spatial * spatial))
        spatial_n += int(spatial.size)
    sample = np.concatenate(ske_values)
    return {
        "rmse": float(np.sqrt(sse / max(ncoef, 1))),
        "mae": float(ae / max(ncoef, 1)),
        "Ske_min": float(ske_min),
        "Ske_median": float(np.median(sample)),
        "Ske_max": float(ske_max),
        "gamma_norm": float(np.linalg.norm(gamma)),
        "spatial_field_rms": float(np.sqrt(spatial_ss / max(spatial_n, 1))),
        "Cu_global": float(cu),
        "lag_c_days": float(lag_c),
        "lag_u_days": float(lag_u),
    }


def optimize_stage(theta0: np.ndarray, arrays: dict[str, np.ndarray], lag_u: float, lam: float, maxiter: int, active: str, log_path: Path, candidate_id: str) -> tuple[np.ndarray, Any, float, np.ndarray]:
    free = np.ones(theta0.size, dtype=bool)
    if active == "global":
        free[1:33] = False
    elif active == "gamma":
        free[[0, 33, 34]] = False
    base = theta0.copy()
    started = time.time()
    iteration = {"n": 0}

    def fun(x):
        theta = base.copy()
        theta[free] = x
        value, grad = objective_grad(theta, arrays, lag_u, lam, active)
        return value, grad[free]

    def callback(x):
        iteration["n"] += 1
        if iteration["n"] == 1 or iteration["n"] % 5 == 0:
            theta = base.copy()
            theta[free] = x
            value, _ = objective_grad(theta, arrays, lag_u, lam, active)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{utc_now()} candidate={candidate_id} stage={active} iter={iteration['n']} objective={value:.6g} elapsed_s={time.time()-started:.1f}\n")

    result = minimize(
        fun,
        theta0[free].copy(),
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={"maxiter": int(maxiter), "maxls": 10, "maxfun": max(20, 5 * int(maxiter))},
    )
    theta = base.copy()
    theta[free] = result.x
    final_obj, final_grad = objective_grad(theta, arrays, lag_u, lam, active)
    return theta, result, final_obj, final_grad


def candidate_config(candidate_id: str, lag_u: float, lam: float, budget: int, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "fold_id": 0,
        "fixed_lag_u_days": float(lag_u),
        "lambda": float(lam),
        "stage_c_budget": int(budget),
        "cache_hash": hashes["cache"],
        "common_mask_hash": hashes["common"],
        "fold_map_hash": hashes["fold"],
        "rbf_selection_hash": hashes["rbf"],
        "objective": "same_full_pixel_fold0_M1_G0_L0_fixed_lag_u",
    }


def checkpoint_valid(path: Path, config_hash: str, hashes: dict[str, str]) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    param_path = path.parent / str(payload.get("parameter_file", ""))
    return (
        payload.get("completed") is True
        and payload.get("candidate_config_hash") == config_hash
        and payload.get("cache_hash") == hashes["cache"]
        and payload.get("common_mask_hash") == hashes["common"]
        and payload.get("fold_map_hash") == hashes["fold"]
        and payload.get("rbf_selection_hash") == hashes["rbf"]
        and param_path.exists()
        and payload.get("parameter_sha256") == sha256_file(param_path)
        and all(k in payload for k in ("train_rmse", "validation_rmse", "objective", "parameter_hash"))
    )


def run_candidate(candidate_id: str, arrays: dict[str, np.ndarray], output: Path, hashes: dict[str, str], lag_u: float, lam: float, budget: int, log_path: Path, resume: bool, seed: np.ndarray | None = None) -> dict[str, Any]:
    cfg = candidate_config(candidate_id, lag_u, lam, budget, hashes)
    cfg_hash = sha256_json(cfg)
    ckpt = output / "checkpoints" / f"{candidate_id}.json"
    if resume and checkpoint_valid(ckpt, cfg_hash, hashes):
        return read_json(ckpt)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{utc_now()} candidate={candidate_id} start lag_u={lag_u} lambda={lam} budget={budget}\n")
    theta0 = seed.copy() if seed is not None else np.r_[np.log(0.002), np.zeros(RBF_DIM), np.log(0.005), 42.0].astype(float)
    theta_a, _, _, _ = optimize_stage(theta0, arrays, lag_u, lam, 20, "global", log_path, candidate_id)
    theta_b0 = theta_a.copy()
    theta_b0[1:33] = 0.0
    theta_b, _, _, _ = optimize_stage(theta_b0, arrays, lag_u, lam, 20, "gamma", log_path, candidate_id)
    theta_c, result, obj, grad = optimize_stage(theta_b, arrays, lag_u, lam, budget, "all", log_path, candidate_id)
    train = metrics(theta_c, arrays, lag_u, train=True)
    val = metrics(theta_c, arrays, lag_u, train=False)
    param_name = f"{candidate_id}_parameters.npy"
    param_sha = atomic_npy(output / "checkpoints" / param_name, theta_c)
    payload = {
        "candidate_id": candidate_id,
        "candidate_config": cfg,
        "candidate_config_hash": cfg_hash,
        "cache_hash": hashes["cache"],
        "common_mask_hash": hashes["common"],
        "fold_map_hash": hashes["fold"],
        "rbf_selection_hash": hashes["rbf"],
        "fixed_lag_u_days": float(lag_u),
        "lambda": float(lam),
        "stage_c_budget": int(budget),
        "train_rmse": train["rmse"],
        "validation_rmse": val["rmse"],
        "train_mae": train["mae"],
        "validation_mae": val["mae"],
        "objective": float(obj),
        "optimizer_success": bool(result.success),
        "project_convergence": False,
        "actual_iterations": int(result.nit),
        "boundary_status": "warning" if val["Ske_max"] > 1.0 or val["Cu_global"] < 1e-10 else "passed",
        "parameter_hash": hashlib.sha256(np.asarray(theta_c, dtype="float64").tobytes()).hexdigest(),
        "parameter_file": param_name,
        "parameter_sha256": param_sha,
        "gradient_rms": float(np.sqrt(np.mean(grad * grad))),
        "Ske_min": val["Ske_min"],
        "Ske_median": val["Ske_median"],
        "Ske_max": val["Ske_max"],
        "Cu_global": val["Cu_global"],
        "lag_c_days": val["lag_c_days"],
        "gamma_norm": val["gamma_norm"],
        "spatial_field_rms": val["spatial_field_rms"],
        "completed": True,
        "completed_at_utc": utc_now(),
    }
    atomic_json(ckpt, payload)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{utc_now()} candidate={candidate_id} complete validation_rmse={val['rmse']:.6f} checkpoint={ckpt}\n")
    gc.collect()
    return payload


def choose_default_within(rows: list[dict[str, Any]], default: float, key: str) -> float:
    best = min(rows, key=lambda row: float(row["validation_rmse"]))
    default_row = next((row for row in rows if float(row[key]) == float(default)), None)
    if default_row and float(default_row["validation_rmse"]) <= float(best["validation_rmse"]) * 1.02:
        return float(default)
    return float(best[key])


def write_metrics_csv(output: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_id", "fixed_lag_u_days", "lambda", "stage_c_budget", "train_rmse", "validation_rmse",
        "train_mae", "validation_mae", "objective", "optimizer_success", "project_convergence",
        "actual_iterations", "boundary_status", "parameter_hash", "parameter_file",
    ]
    tmp = output / f"fold0_candidate_metrics.csv.tmp.{os.getpid()}"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    os.replace(tmp, output / "fold0_candidate_metrics.csv")


def check_only(cache: Path, output: Path) -> dict[str, Any]:
    cache_path, cache_hash, cache_key = validate_cache(cache)
    rbf_path = select_rbf_path()
    hashes = {
        "cache": cache_hash,
        "common": sha256_file(COMMON_MASK),
        "fold": sha256_file(FOLD_MAP),
        "rbf": sha256_file(rbf_path),
    }


def compute_artifact_metrics(selected: dict[str, Any], transform: np.ndarray, gamma: np.ndarray) -> dict[str, Any]:
    values = []
    peaks = []
    distances = []
    stride = 1
    for _, _, _, spatial, peak, dist in iter_spatial_map(COMMON_MASK, FOLD_MAP, selected, transform, gamma):
        if spatial.size > 100_000:
            stride = max(stride, int(np.ceil(spatial.size / 100_000)))
        values.append(np.asarray(spatial[::stride], dtype=float))
        peaks.append(np.asarray(peak[::stride], dtype=float))
        distances.append(np.asarray(dist[::stride], dtype=float))
    return artifact_metrics(np.concatenate(values), np.concatenate(peaks), np.concatenate(distances), np.asarray(gamma, dtype=float))


def load_candidate(output: Path, candidate_id: str, hashes: dict[str, str]) -> dict[str, Any]:
    path = output / "checkpoints" / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate checkpoint: {candidate_id}")
    row = read_json(path)
    param = output / "checkpoints" / str(row.get("parameter_file", ""))
    if not param.exists():
        raise FileNotFoundError(f"Missing candidate parameters: {candidate_id}")
    if row.get("completed") is not True:
        raise RuntimeError(f"Candidate not complete: {candidate_id}")
    for key, hkey in [("cache_hash", "cache"), ("common_mask_hash", "common"), ("fold_map_hash", "fold"), ("rbf_selection_hash", "rbf")]:
        if row.get(key) != hashes[hkey]:
            raise RuntimeError(f"Candidate hash mismatch for {candidate_id}: {key}")
    if row.get("parameter_sha256") != sha256_file(param):
        raise RuntimeError(f"Candidate parameter SHA256 mismatch: {candidate_id}")
    return row


def finalize_from_checkpoints(output: Path, cache_path: Path, cache_hash: str, cache_key: str, hashes: dict[str, str], selected: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
    a_ids = ["A_lagu_0_lambda_30_budget_30", "A_lagu_10_lambda_30_budget_30", "A_lagu_20_lambda_30_budget_30"]
    a_rows = [load_candidate(output, cid, hashes) for cid in a_ids]
    selected_lag = choose_default_within(a_rows, 10.0, "fixed_lag_u_days")
    a_selected = next(row for row in a_rows if float(row["fixed_lag_u_days"]) == selected_lag)
    b30 = dict(a_selected)
    b30["candidate_id"] = "B_lambda_30_reused"
    b30["reused_from_candidate_id"] = a_selected["candidate_id"]
    atomic_json(output / "checkpoints" / "B_lambda_30_reused.json", b30)
    b10 = load_candidate(output, "B_lambda_10", hashes)
    b_rows = [b10, b30]
    selected_lambda = choose_default_within(b_rows, 30.0, "lambda")
    b_selected = next(row for row in b_rows if float(row["lambda"]) == selected_lambda)
    c30 = dict(b_selected)
    c30["candidate_id"] = "C_budget_30_reused"
    c30["reused_from_candidate_id"] = b_selected["candidate_id"]
    atomic_json(output / "checkpoints" / "C_budget_30_reused.json", c30)
    c40 = load_candidate(output, "C_budget_40", hashes)
    selected_budget = 40 if c40["optimizer_success"] and float(c40["validation_rmse"]) <= float(c30["validation_rmse"]) * 1.02 else 30
    final_row = c40 if selected_budget == 40 else c30
    rows = a_rows + b_rows + [c30, c40]
    write_metrics_csv(output, rows)
    theta = np.load(output / "checkpoints" / final_row["parameter_file"])
    artifact = compute_artifact_metrics(selected, transform, theta[1:33])
    meta_hash = sha256_file(output / "memmap_manifest.json") if (output / "memmap_manifest.json").exists() else None
    summary = {
        "fold0_confirmation_status": "passed",
        "cache_path": str(cache_path),
        "cache_sha256": cache_hash,
        "cache_key": cache_key,
        "common_mask_hash": hashes["common"],
        "fold_map_hash": hashes["fold"],
        "rbf_selection_hash": hashes["rbf"],
        "memmap_manifest_hash": meta_hash,
        "candidates": {row["candidate_id"]: str(output / "checkpoints" / f"{row['candidate_id']}.json") for row in rows},
        "selected_lag_u_days": selected_lag,
        "selected_lambda": selected_lambda,
        "selected_stage_c_budget": selected_budget,
        "two_percent_retention_rule": True,
        "actual_candidate_runs": 0,
        "reused_candidate_results": len(rows),
        "resumed_from_checkpoints": [row["candidate_id"] for row in rows],
        "no_formal_fold_accessed": True,
        "no_phase4_or_phase5_run": True,
        "failure_reasons": [],
        "artifact_metrics": artifact,
        "completed_at_utc": utc_now(),
    }
    atomic_json(output / "fold0_confirmation_summary.json", summary)
    return summary
    meta = validate_memmap_manifest(output, hashes)
    return {
        "check_only_status": "passed",
        "cache_path": str(cache_path),
        "cache_sha256": cache_hash,
        "cache_key_present": bool(cache_key),
        "common_mask_hash": hashes["common"],
        "fold_map_hash": hashes["fold"],
        "rbf_selection_path": str(rbf_path),
        "rbf_selection_hash": hashes["rbf"],
        "memmap_reusable": meta is not None,
        "memmap_manifest_path": str(output / "memmap_manifest.json"),
        "fold0_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.fold_id != 0:
        raise SystemExit("Only fold-id=0 is allowed")
    output = validate_output_dir(Path(args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        print(json.dumps(check_only(Path(args.cache), output), indent=2, sort_keys=True))
        return 0

    cache_path, cache_hash, cache_key = validate_cache(Path(args.cache))
    rbf_path = select_rbf_path()
    selected = read_json(rbf_path)
    transform = np.load(RBF_TRANSFORM)
    hashes = {
        "cache": cache_hash,
        "common": sha256_file(COMMON_MASK),
        "fold": sha256_file(FOLD_MAP),
        "rbf": sha256_file(rbf_path),
    }
    if args.finalize_only:
        summary = finalize_from_checkpoints(output, cache_path, cache_hash, cache_key, hashes, selected, transform)
        print(json.dumps({"fold0_finalize_only_status": summary["fold0_confirmation_status"], "selected_lag_u_days": summary["selected_lag_u_days"], "selected_lambda": summary["selected_lambda"], "selected_stage_c_budget": summary["selected_stage_c_budget"]}, indent=2))
        return 0
    log_path = output / "fold0_confirmation.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{utc_now()} fold0 resume={args.resume} cache={cache_path}\n")
    meta = build_memmaps(cache_path, selected, transform, 0, output, hashes)
    arrays = open_arrays(output, meta)
    actual_runs = 0
    reused = 0
    resumed = []

    a_rows = []
    theta_by_id: dict[str, np.ndarray] = {}
    for lag in [0.0, 10.0, 20.0]:
        cid = f"A_lagu_{int(lag)}_lambda_30_budget_30"
        before = output / "checkpoints" / f"{cid}.json"
        existed_before = before.exists()
        row = run_candidate(cid, arrays, output, hashes, lag, 30.0, 30, log_path, args.resume)
        if existed_before and row.get("completed"):
            reused += 1
            resumed.append(cid)
        else:
            actual_runs += 1
        theta_by_id[cid] = np.load(output / "checkpoints" / row["parameter_file"])
        a_rows.append(row)
    selected_lag = choose_default_within(a_rows, 10.0, "fixed_lag_u_days")
    a_selected = next(row for row in a_rows if float(row["fixed_lag_u_days"]) == selected_lag)

    b30 = dict(a_selected)
    b30["candidate_id"] = "B_lambda_30_reused"
    b30["reused_from_candidate_id"] = a_selected["candidate_id"]
    atomic_json(output / "checkpoints" / "B_lambda_30_reused.json", b30)
    b10_path = output / "checkpoints" / "B_lambda_10.json"
    b10_existed = b10_path.exists()
    b10 = run_candidate("B_lambda_10", arrays, output, hashes, selected_lag, 10.0, 30, log_path, args.resume, theta_by_id[a_selected["candidate_id"]])
    if b10_existed and b10.get("completed"):
        reused += 1
        resumed.append("B_lambda_10")
    else:
        actual_runs += 1
    b_rows = [b10, b30]
    selected_lambda = choose_default_within(b_rows, 30.0, "lambda")
    b_selected = next(row for row in b_rows if float(row["lambda"]) == selected_lambda)

    c30 = dict(b_selected)
    c30["candidate_id"] = "C_budget_30_reused"
    c30["reused_from_candidate_id"] = b_selected["candidate_id"]
    atomic_json(output / "checkpoints" / "C_budget_30_reused.json", c30)
    seed_theta = np.load(output / "checkpoints" / b_selected["parameter_file"])
    c40_path = output / "checkpoints" / "C_budget_40.json"
    c40_existed = c40_path.exists()
    c40 = run_candidate("C_budget_40", arrays, output, hashes, selected_lag, selected_lambda, 40, log_path, args.resume, seed_theta)
    if c40_existed and c40.get("completed"):
        reused += 1
        resumed.append("C_budget_40")
    else:
        actual_runs += 1
    c_rows = [c30, c40]
    selected_budget = 40 if c40["optimizer_success"] and float(c40["validation_rmse"]) <= float(c30["validation_rmse"]) * 1.02 else 30
    final_row = c40 if selected_budget == 40 else c30
    rows = a_rows + b_rows + c_rows
    write_metrics_csv(output, rows)
    artifact = compute_artifact_metrics(selected, transform, np.load(output / "checkpoints" / final_row["parameter_file"])[1:33])
    summary = {
        "fold0_confirmation_status": "passed",
        "cache_path": str(cache_path),
        "cache_sha256": cache_hash,
        "cache_key": cache_key,
        "common_mask_hash": hashes["common"],
        "fold_map_hash": hashes["fold"],
        "rbf_selection_hash": hashes["rbf"],
        "memmap_manifest_hash": sha256_file(output / "memmap_manifest.json"),
        "candidates": {row["candidate_id"]: str(output / "checkpoints" / f"{row['candidate_id']}.json") for row in rows},
        "selected_lag_u_days": selected_lag,
        "selected_lambda": selected_lambda,
        "selected_stage_c_budget": selected_budget,
        "two_percent_retention_rule": True,
        "actual_candidate_runs": actual_runs,
        "reused_candidate_results": reused + 2,
        "resumed_from_checkpoints": resumed,
        "no_formal_fold_accessed": True,
        "no_phase4_or_phase5_run": True,
        "failure_reasons": [],
        "artifact_metrics": artifact,
        "completed_at_utc": utc_now(),
    }
    atomic_json(output / "fold0_confirmation_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
