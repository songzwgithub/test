#!/usr/bin/env python3
"""Rebuild the Phase-4 harmonic cache from authoritative L01028 time series.

The script intentionally reuses the old cache spatial partition and static
datasets, and only recomputes harmonic observations and weights.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio
import yaml


DEFAULT_OLD_CACHE = Path("outputs/cache/phase4_harmonic_blocks_d0283cfacbadc767.h5")
DEFAULT_TIMESERIES_DIR = Path("../../geo_timeseries_gacos_filtered_L01028")
DEFAULT_CONFIG = Path("config.yaml")
DEFAULT_REFERENCE_DIR = Path("outputs/reference_frames/L01028_500m_fixed_quality_median_v1")
DEFAULT_COMMON_MASK = Path("outputs/aquifer_model_revision/comparison_common_mask.tif")
DEFAULT_OUTPUT_CACHE = Path("outputs/cache/phase4_harmonic_blocks_L01028_authoritative.h5")
DEFAULT_AUDIT_JSON = Path("outputs/cache/phase4_harmonic_blocks_L01028_authoritative_audit.json")

DATE_RE = re.compile(r"(?:geo_\d{8}_(\d{8})|(\d{8})(?=\.tif(?:f)?$))", re.IGNORECASE)
STATIC_DATASETS = (
    "hc",
    "hu",
    "z",
    "flat_index",
    "block_start",
    "block_count",
    "block_row",
    "block_col",
    "block_height",
    "block_width",
)
REQUIRED_DATASETS = STATIC_DATASETS + ("obs", "weights")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def h5_dataset_hash(dataset: h5py.Dataset, chunk_rows: int = 1_000_000) -> str:
    digest = hashlib.sha256()
    if dataset.shape == ():
        digest.update(np.asarray(dataset[()]).tobytes())
        return digest.hexdigest()
    n = int(dataset.shape[0])
    for start in range(0, n, chunk_rows):
        arr = np.asarray(dataset[start : min(start + chunk_rows, n)])
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def parse_date(path: Path) -> pd.Timestamp | None:
    if "velocity" in path.name.lower():
        return None
    match = DATE_RE.search(path.name)
    if not match:
        return None
    token = match.group(1) or match.group(2)
    return pd.Timestamp(pd.to_datetime(token, format="%Y%m%d"))


def list_timeseries(paths_root: Path) -> tuple[list[Path], pd.DatetimeIndex]:
    candidates = sorted(paths_root.glob("*.tif")) + sorted(paths_root.glob("*.tiff"))
    dated: list[tuple[pd.Timestamp, Path]] = []
    for path in candidates:
        date = parse_date(path)
        if date is not None:
            dated.append((date, path))
    dated.sort(key=lambda item: item[0])
    dates = pd.DatetimeIndex([item[0] for item in dated])
    paths = [item[1] for item in dated]
    if len(paths) != len(set(dates)):
        raise RuntimeError("Duplicate dates detected in L01028 time-series directory.")
    return paths, dates


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_path(config_path: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def displacement_scale_to_mm(unit: str) -> float:
    unit_l = str(unit).lower()
    if unit_l in {"mm", "millimeter", "millimeters"}:
        return 1.0
    if unit_l in {"m", "meter", "meters"}:
        return 1000.0
    raise ValueError(f"Unsupported displacement unit: {unit!r}")


def harmonic_coefficients(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    origin: str,
    period_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    output = np.full((values.shape[0], 2), np.nan, dtype=float)
    covariances = np.full((values.shape[0], 2, 2), np.nan, dtype=float)
    t = (pd.DatetimeIndex(pd.to_datetime(dates)) - pd.Timestamp(origin)).days.to_numpy(dtype=float)
    X = np.column_stack(
        [
            np.ones(len(t)),
            t / 365.2425,
            np.sin(2 * np.pi * t / float(period_days)),
            np.cos(2 * np.pi * t / float(period_days)),
        ]
    )
    finite = np.isfinite(values)
    nobs = finite.sum(axis=1)
    good = nobs >= 24
    if good.any():
        Y = np.where(finite[good], values[good], 0.0)
        W = finite[good].astype(float)
        xtwx = np.einsum("ti,nt,tj->nij", X, W, X, optimize=True)
        xtwy = np.einsum("ti,nt->ni", X, Y, optimize=True)
        xtwx += np.eye(X.shape[1])[None, :, :] * 1e-8
        beta = np.linalg.solve(xtwx, xtwy[..., None])[..., 0]
        output[good] = beta[:, 2:4]
        fitted = np.einsum("ti,ni->nt", X, beta, optimize=True)
        resid = np.where(finite[good], values[good] - fitted, 0.0)
        dof = np.maximum(1, nobs[good] - X.shape[1])
        sigma2 = np.sum(W * resid * resid, axis=1) / dof
        inv = np.linalg.pinv(xtwx)
        covariances[good] = inv[:, 2:4, 2:4] * sigma2[:, None, None]
    return output, covariances


def validate_old_cache(h5: h5py.File) -> dict[str, int]:
    missing = [name for name in REQUIRED_DATASETS if name not in h5]
    if missing:
        raise RuntimeError(f"Old cache missing datasets: {missing}")
    n_blocks = int(len(h5["block_start"]))
    n_pixels = int(len(h5["flat_index"]))
    if n_blocks != int(len(h5["block_count"])):
        raise RuntimeError("Old cache block_start/block_count length mismatch.")
    if n_pixels != int(h5["obs"].shape[0]) or n_pixels != int(h5["weights"].shape[0]):
        raise RuntimeError("Old cache obs/weights/flat_index length mismatch.")
    return {"block_count": n_blocks, "record_count": n_pixels}


def read_common_mask_count(path: Path) -> tuple[int, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": tuple(src.transform),
        }
    return int(np.count_nonzero(arr)), profile


def check_raster_geometry(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise RuntimeError("No rasters supplied for geometry check.")
    with rasterio.open(paths[0]) as ref:
        profile = {
            "width": ref.width,
            "height": ref.height,
            "crs": str(ref.crs),
            "transform": tuple(ref.transform),
            "dtype": ref.dtypes[0],
        }
    for path in paths[1:]:
        with rasterio.open(path) as src:
            if src.width != profile["width"] or src.height != profile["height"]:
                raise RuntimeError(f"Raster size mismatch: {path}")
            if str(src.crs) != profile["crs"] or tuple(src.transform) != profile["transform"]:
                raise RuntimeError(f"Raster georeference mismatch: {path}")
    return profile


def copy_static_datasets(src: h5py.File, dst: h5py.File) -> None:
    for name in STATIC_DATASETS:
        src.copy(name, dst)


def create_like(src: h5py.Dataset, dst: h5py.File, name: str) -> h5py.Dataset:
    kwargs: dict[str, Any] = {}
    for attr in ("compression", "compression_opts", "shuffle", "fletcher32", "chunks"):
        value = getattr(src, attr, None)
        if value is not None:
            kwargs[attr] = value
    return dst.create_dataset(name, shape=src.shape, dtype=src.dtype, **kwargs)


def copy_attrs(src: h5py.File, dst: h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def read_stack_window(
    datasets: list[rasterio.io.DatasetReader],
    row: int,
    col: int,
    height: int,
    width: int,
    scale_to_mm: np.float32,
    incidence: np.ndarray,
) -> np.ndarray:
    window = rasterio.windows.Window(col_off=col, row_off=row, width=width, height=height)
    stack = np.empty((len(datasets), height, width), dtype=np.float32)
    for i, src in enumerate(datasets):
        stack[i] = src.read(1, window=window).astype(np.float32, copy=False)
    stack *= scale_to_mm
    inc = np.asarray(incidence[row : row + height, col : col + width], dtype=np.float32)
    cosine = np.cos(np.deg2rad(inc, dtype=np.float32))
    cosine[np.abs(cosine) < np.float32(1e-6)] = np.nan
    stack /= cosine[None, :, :]
    return stack


def block_response_values(path: Path, row: int, col: int, height: int, width: int, flat_index: np.ndarray) -> np.ndarray:
    window = rasterio.windows.Window(col_off=col, row_off=row, width=width, height=height)
    with rasterio.open(path) as src:
        arr = src.read(1, window=window)
    return np.asarray(arr.reshape(-1)[flat_index], dtype=float)


def load_reference_frame_id(manifest_path: Path) -> str:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return str(
        manifest.get("reference_frame_id")
        or manifest.get("candidate_id")
        or manifest.get("id")
        or "L01028_500m_fixed_quality_median_v1"
    )


def build_cache_key_payload(args: argparse.Namespace, inputs: dict[str, str], config_values: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": "rebuild_L01028_phase4_harmonic_cache",
        "algorithm_version": 1,
        "old_cache_hash": inputs["old_cache_hash"],
        "reference_manifest_hash": inputs["reference_manifest_hash"],
        "annual_real_hash": inputs["annual_real_hash"],
        "annual_imag_hash": inputs["annual_imag_hash"],
        "incidence_grid_hash": inputs["incidence_grid_hash"],
        "common_mask_hash": inputs["common_mask_hash"],
        "timeseries_directory": str(Path(args.timeseries_dir).resolve()),
        "timeseries_dates_hash": inputs["timeseries_dates_hash"],
        "harmonic_origin": config_values["harmonic_origin"],
        "annual_period_days": config_values["annual_period_days"],
        "observation_sigma_mm": config_values["observation_sigma_mm"],
        "weight_formula": "obs_var=nanmean(diag(cov)); invalid_or_nonpositive_to_sigma2; weights=1/max(obs_var,sigma2)",
        "reference_application_count": 1,
    }


def collect_inputs(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    config = load_config(config_path)
    temporal = config.get("temporal", {})
    phase4 = config.get("phase4", {})
    insar = config.get("insar", {})
    observation_sigma_mm = float(phase4.get("observation_sigma_mm", 5.0))
    harmonic_origin = temporal.get("harmonic_origin")
    annual_period_days = float(temporal.get("annual_period_days", 365.2425))
    if not harmonic_origin:
        raise RuntimeError("config temporal.harmonic_origin is required.")
    incidence_path = resolve_config_path(config_path, insar["incidence_grid"])
    scale_to_mm = np.float32(displacement_scale_to_mm(insar.get("displacement_unit", "m")))
    ts_paths, dates = list_timeseries(Path(args.timeseries_dir))
    if len(ts_paths) != 245:
        raise RuntimeError(f"Expected 245 L01028 scenes, found {len(ts_paths)}.")
    reference_dir = Path(args.reference_dir)
    annual_real_path = reference_dir / "harmonic" / "annual_vertical_real_sin_mm.tif"
    annual_imag_path = reference_dir / "harmonic" / "annual_vertical_imag_cos_mm.tif"
    manifest_path = reference_dir / "reference_frame_manifest.json"
    for path in (incidence_path, annual_real_path, annual_imag_path, manifest_path, Path(args.old_cache), Path(args.common_mask)):
        if not path.exists():
            raise FileNotFoundError(path)
    dates_hash = hashlib.sha256("\n".join(d.strftime("%Y-%m-%d") for d in dates).encode()).hexdigest()
    hashes = {
        "old_cache_hash": sha256_file(Path(args.old_cache)),
        "reference_manifest_hash": sha256_file(manifest_path),
        "annual_real_hash": sha256_file(annual_real_path),
        "annual_imag_hash": sha256_file(annual_imag_path),
        "incidence_grid_hash": sha256_file(incidence_path),
        "common_mask_hash": sha256_file(Path(args.common_mask)),
        "timeseries_dates_hash": dates_hash,
    }
    return {
        "config": config,
        "config_values": {
            "harmonic_origin": harmonic_origin,
            "annual_period_days": annual_period_days,
            "observation_sigma_mm": observation_sigma_mm,
            "scale_to_mm": float(scale_to_mm),
        },
        "incidence_path": incidence_path,
        "scale_to_mm": scale_to_mm,
        "timeseries_paths": ts_paths,
        "dates": dates,
        "annual_real_path": annual_real_path,
        "annual_imag_path": annual_imag_path,
        "manifest_path": manifest_path,
        "reference_frame_id": load_reference_frame_id(manifest_path),
        "hashes": hashes,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def dry_run(args: argparse.Namespace) -> None:
    inputs = collect_inputs(args)
    ts_profile = check_raster_geometry(inputs["timeseries_paths"])
    response_profile = check_raster_geometry([inputs["annual_real_path"], inputs["annual_imag_path"], Path(args.common_mask)])
    if ts_profile["width"] != response_profile["width"] or ts_profile["height"] != response_profile["height"]:
        raise RuntimeError("Time-series and response raster dimensions differ.")
    incidence = np.load(inputs["incidence_path"], mmap_mode="r")
    if tuple(incidence.shape) != (ts_profile["height"], ts_profile["width"]):
        raise RuntimeError("Incidence grid shape does not match L01028 rasters.")
    common_count, _ = read_common_mask_count(Path(args.common_mask))
    with h5py.File(args.old_cache, "r") as old:
        cache_info = validate_old_cache(old)
        extra_pixels = int(cache_info["record_count"] - common_count)
    audit = {
        "audit_status": "dry_run_passed",
        "date_count": int(len(inputs["dates"])),
        "block_count": cache_info["block_count"],
        "record_count": cache_info["record_count"],
        "common_mask_pixel_count": common_count,
        "extra_pixels_vs_common_mask": extra_pixels,
        "input_hashes": inputs["hashes"],
        "config_values": inputs["config_values"],
        "source_timeseries_directory": str(Path(args.timeseries_dir).resolve()),
        "reference_frame_id": inputs["reference_frame_id"],
        "dry_run_no_hdf5_written": True,
    }
    write_json(Path(args.audit_json), audit)


def rebuild(args: argparse.Namespace) -> None:
    output = Path(args.output_cache)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it.")
    building = output.with_name(output.stem + ".building.h5")
    if building.exists():
        if not args.overwrite:
            raise FileExistsError(f"{building} exists; pass --overwrite to replace it.")
        building.unlink()
    inputs = collect_inputs(args)
    common_count, _ = read_common_mask_count(Path(args.common_mask))
    ts_profile = check_raster_geometry(inputs["timeseries_paths"])
    response_profile = check_raster_geometry([inputs["annual_real_path"], inputs["annual_imag_path"]])
    if ts_profile["width"] != response_profile["width"] or ts_profile["height"] != response_profile["height"]:
        raise RuntimeError("Time-series and response raster dimensions differ.")
    incidence = np.load(inputs["incidence_path"], mmap_mode="r")
    if tuple(incidence.shape) != (ts_profile["height"], ts_profile["width"]):
        raise RuntimeError("Incidence grid shape does not match L01028 rasters.")
    sigma = float(inputs["config_values"]["observation_sigma_mm"])
    max_weight = 1.0 / (sigma * sigma)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_weight = 0.0
    obs_max_abs_diff = 0.0
    obs_ssq_diff = 0.0
    obs_diff_count = 0
    weights_min = math.inf
    weights_max = -math.inf
    weight_samples: list[np.ndarray] = []
    low_weight_count = 0
    with contextlib.ExitStack() as stack:
        rasters = [stack.enter_context(rasterio.open(path)) for path in inputs["timeseries_paths"]]
        old = stack.enter_context(h5py.File(args.old_cache, "r"))
        new = stack.enter_context(h5py.File(building, "w"))
        cache_info = validate_old_cache(old)
        copy_static_datasets(old, new)
        obs_ds = create_like(old["obs"], new, "obs")
        weights_ds = create_like(old["weights"], new, "weights")
        copy_attrs(old, new)
        new.attrs["complete"] = 0
        n_blocks = cache_info["block_count"]
        for block_id in range(n_blocks):
            start = int(old["block_start"][block_id])
            count = int(old["block_count"][block_id])
            end = start + count
            row = int(old["block_row"][block_id])
            col = int(old["block_col"][block_id])
            height = int(old["block_height"][block_id])
            width = int(old["block_width"][block_id])
            flat_index = np.asarray(old["flat_index"][start:end], dtype=np.int64)
            stack_mm = read_stack_window(
                rasters,
                row,
                col,
                height,
                width,
                inputs["scale_to_mm"],
                incidence,
            )
            values = stack_mm.reshape(len(rasters), -1).T[flat_index].astype(np.float64, copy=False)
            obs, cov = harmonic_coefficients(
                inputs["dates"],
                values,
                str(inputs["config_values"]["harmonic_origin"]),
                float(inputs["config_values"]["annual_period_days"]),
            )
            obs_var = np.nanmean(np.diagonal(cov, axis1=1, axis2=2), axis=1)
            obs_var = np.where(np.isfinite(obs_var) & (obs_var > 0), obs_var, sigma * sigma)
            weights = 1.0 / np.maximum(obs_var, sigma * sigma)
            if not np.all(np.isfinite(obs)):
                raise RuntimeError(f"Non-finite obs in block {block_id}.")
            if not (np.all(np.isfinite(weights)) and np.all(weights > 0) and np.all(weights <= max_weight + 1e-12)):
                raise RuntimeError(f"Invalid weights in block {block_id}.")
            real_ref = block_response_values(inputs["annual_real_path"], row, col, height, width, flat_index)
            imag_ref = block_response_values(inputs["annual_imag_path"], row, col, height, width, flat_index)
            diff = obs - np.column_stack([real_ref, imag_ref])
            block_max = float(np.nanmax(np.abs(diff)))
            if block_max > 1e-4:
                raise RuntimeError(f"Block {block_id} obs mismatch with L01028 response: max_abs_diff={block_max:.6g} mm")
            obs_max_abs_diff = max(obs_max_abs_diff, block_max)
            obs_ssq_diff += float(np.nansum(diff * diff))
            obs_diff_count += int(diff.size)
            weights_min = min(weights_min, float(np.min(weights)))
            weights_max = max(weights_max, float(np.max(weights)))
            low_weight_count += int(np.count_nonzero(weights < max_weight))
            if len(weight_samples) < 100:
                weight_samples.append(weights[:: max(1, len(weights) // 1000)].astype(np.float64, copy=True))
            total_weight += float(np.sum(weights))
            obs_ds[start:end] = obs.astype(obs_ds.dtype, copy=False)
            weights_ds[start:end] = weights.astype(weights_ds.dtype, copy=False)
            if (block_id + 1) % 10 == 0 or block_id + 1 == n_blocks:
                print(f"processed {block_id + 1}/{n_blocks} blocks", flush=True)
        static_ok = True
        for name in STATIC_DATASETS:
            if h5_dataset_hash(old[name]) != h5_dataset_hash(new[name]):
                static_ok = False
                break
        cache_key_payload = build_cache_key_payload(args, inputs["hashes"], inputs["config_values"])
        cache_key = sha256_json(cache_key_payload)
        new.attrs["total_weight"] = total_weight
        new.attrs["cache_key"] = cache_key
        new.attrs["reference_frame_id"] = inputs["reference_frame_id"]
        new.attrs["reference_manifest_hash"] = inputs["hashes"]["reference_manifest_hash"]
        new.attrs["annual_real_hash"] = inputs["hashes"]["annual_real_hash"]
        new.attrs["annual_imag_hash"] = inputs["hashes"]["annual_imag_hash"]
        new.attrs["source_timeseries_directory"] = str(Path(args.timeseries_dir).resolve())
        new.attrs["incidence_grid_hash"] = inputs["hashes"]["incidence_grid_hash"]
        new.attrs["common_mask_hash"] = inputs["hashes"]["common_mask_hash"]
        new.attrs["observation_sigma_mm"] = sigma
        new.attrs["weight_formula"] = "obs_var=nanmean(diag(cov)); invalid_or_nonpositive_to_sigma2; weights=1/max(obs_var,sigma2)"
        new.attrs["source_cache_hash"] = inputs["hashes"]["old_cache_hash"]
        new.attrs["cache_provenance"] = json.dumps(cache_key_payload, sort_keys=True, separators=(",", ":"))
        new.attrs["complete"] = 1
        new.flush()
    os.replace(building, output)
    new_hash = sha256_file(output)
    weights_concat = np.concatenate(weight_samples) if weight_samples else np.array([], dtype=float)
    audit = {
        "audit_status": "passed",
        "date_count": int(len(inputs["dates"])),
        "block_count": int(cache_info["block_count"]),
        "record_count": int(cache_info["record_count"]),
        "common_mask_pixel_count": int(common_count),
        "extra_pixels_vs_common_mask": int(cache_info["record_count"] - common_count),
        "obs_max_abs_diff_mm": obs_max_abs_diff,
        "obs_rms_diff_mm": float(math.sqrt(obs_ssq_diff / max(1, obs_diff_count))),
        "weights_min": weights_min,
        "weights_median_sample": float(np.median(weights_concat)) if weights_concat.size else None,
        "weights_max": weights_max,
        "weights_below_0_04_count": low_weight_count,
        "old_cache_hash": inputs["hashes"]["old_cache_hash"],
        "new_cache_hash": new_hash,
        "input_hashes": inputs["hashes"],
        "static_datasets_identical": bool(static_ok),
        "cache_matches_L01028_response": bool(obs_max_abs_diff <= 1e-4),
        "cache_key": cache_key,
    }
    write_json(Path(args.audit_json), audit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-cache", default=str(DEFAULT_OLD_CACHE))
    parser.add_argument("--timeseries-dir", default=str(DEFAULT_TIMESERIES_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--common-mask", default=str(DEFAULT_COMMON_MASK))
    parser.add_argument("--output-cache", default=str(DEFAULT_OUTPUT_CACHE))
    parser.add_argument("--audit-json", default=str(DEFAULT_AUDIT_JSON))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.dry_run:
            dry_run(args)
        else:
            rebuild(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
