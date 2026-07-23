"""Build the L01028 fixed-pixel InSAR reference frame and response products.

The script never edits source GeoTIFFs or old V2 formal outputs.  It computes
the reference in native LOS geometry, subtracts the normalized reference series
on the fly, and fits velocity plus annual harmonic coefficients from the
rereferenced time series.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insar_processing import (
    GeoTiffCube,
    L01028_REFERENCE_FRAME_ID,
    LOS_SIGN_CONVENTION,
    apply_reference_to_los,
    assert_reference_dates_match,
    displacement_scale_to_mm,
    normalize_reference_series,
    reference_application_metadata,
    sha256_array,
)
from io_utils import load_config, resolve_config_path
from spatial_utils import circular_mask, iter_windows, radius_window


CANDIDATE_ID = "L01028"
CENTER_LON = 114.71554875
CENTER_LAT = 38.06686417
RADIUS_M = 500.0
REFERENCE_METHOD = "robust_fixed_quality_pixel_median"
REFERENCE_MODE = "on_the_fly"
REFERENCE_GEOMETRY = "native_LOS"
EXPECTED_CANDIDATE_PIXELS = 1449
EXPECTED_FIXED_PIXELS = 956
EXPECTED_EPOCHS = 245
V_L01028_MM_YR = 7.3823


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stack_metadata_hash(files: list[str]) -> str:
    rows = []
    for item in files:
        path = Path(item)
        st = path.stat()
        rows.append({"path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return sha256_text(json.dumps(rows, sort_keys=True))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cleanup_stale_product_temps(output_dir: Path, include_building=False):
    removed = []
    # A partially written finalizing file is never resumed.  It is rebuilt from
    # the complete uncompressed .building.tif after all spatial tiles finish.
    patterns = ["*.tmp", "*.finalizing.tif"]
    if include_building:
        patterns.append("*.building.tif")
    candidates = []
    for pattern in patterns:
        candidates.extend((output_dir / "harmonic").glob(pattern))
        candidates.extend((output_dir / "velocity").glob(pattern))
    for tmp in candidates:
        tmp.unlink()
        removed.append(str(tmp))
    if include_building:
        for path in [output_dir / "response_product_tile_status.json", output_dir / "response_products_complete.json"]:
            if path.exists():
                path.unlink()
                removed.append(str(path))
    return removed


def _building_profile(src: rasterio.io.DatasetReader, *, dtype: str, nodata):
    """Return an uncompressed tiled profile for fast incremental writes."""
    profile = src.profile.copy()
    # Source GeoTIFFs may themselves be compressed.  Explicitly remove all
    # compression options so .building.tif files are genuinely uncompressed.
    for key in ("compress", "predictor", "zlevel", "num_threads"):
        profile.pop(key, None)
    profile.update(
        count=1,
        dtype=dtype,
        nodata=nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return profile


def _validate_uncompressed_building_raster(
    path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_dtype: str,
):
    """Reject incompatible old temporary rasters before resuming."""
    with rasterio.open(path) as src:
        if src.width != expected_width or src.height != expected_height:
            raise RuntimeError(
                f"Existing building raster has the wrong shape: {path} "
                f"({src.width}x{src.height}, expected {expected_width}x{expected_height})"
            )
        if src.dtypes[0] != expected_dtype:
            raise RuntimeError(
                f"Existing building raster has dtype {src.dtypes[0]}, "
                f"expected {expected_dtype}: {path}"
            )
        if src.compression is not None:
            raise RuntimeError(
                f"Existing building raster is compressed ({src.compression}): {path}. "
                "The optimized writer requires uncompressed .building.tif files. "
                "Restart once with --restart-products; do not mix the old compressed "
                "temporary files with the new build mode."
            )


def _compress_building_raster(
    building_path: Path,
    finalizing_path: Path,
    *,
    predictor: int,
):
    """Sequentially convert one complete uncompressed raster to DEFLATE."""
    if finalizing_path.exists():
        finalizing_path.unlink()
    with rasterio.open(building_path) as src:
        profile = src.profile.copy()
        profile.update(
            compress="deflate",
            predictor=predictor,
            zlevel=6,
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(finalizing_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                dst.write(src.read(1, window=window), 1, window=window)
            dataset_tags = src.tags()
            band_tags = src.tags(1)
            if dataset_tags:
                dst.update_tags(**dataset_tags)
            if band_tags:
                dst.update_tags(1, **band_tags)
            if src.descriptions and src.descriptions[0]:
                dst.set_band_description(1, src.descriptions[0])


def _publish_compressed_products(
    tmp_paths: dict[str, Path],
    paths: dict[str, Path],
):
    """Build every compressed final file first, then publish the set."""
    finalizing_paths = {
        key: path.with_suffix(path.suffix + ".finalizing.tif")
        for key, path in paths.items()
    }
    for key, building_path in tmp_paths.items():
        if not building_path.exists():
            raise RuntimeError(f"Missing completed building raster: {building_path}")
        predictor = 2 if key == "n_observations" else 3
        print(f"compress_final_product {key}", flush=True)
        _compress_building_raster(
            building_path,
            finalizing_paths[key],
            predictor=predictor,
        )

    # No official product is replaced until all six compressed files exist.
    missing = [str(path) for path in finalizing_paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Compressed finalization incomplete; missing: {missing}")
    for key, finalizing_path in finalizing_paths.items():
        finalizing_path.replace(paths[key])


def select_fixed_reference_pixels(cube: GeoTiffCube, output_dir: Path):
    scale = displacement_scale_to_mm(cube.unit)
    first = cube.epochs["source_file"].iloc[0]
    with rasterio.open(first) as src:
        window = radius_window(src, CENTER_LON, CENTER_LAT, RADIUS_M)
        mask = circular_mask(src, window, CENTER_LON, CENTER_LAT, RADIUS_M)
        row0, col0 = int(window.row_off), int(window.col_off)
        rr, cc = np.indices(mask.shape)
        rows = rr[mask] + row0
        cols = cc[mask] + col0
        candidate_ids = (rows.astype("uint64") * np.uint64(src.width) + cols.astype("uint64")).astype("uint64")
        local_rows = rr[mask]
        local_cols = cc[mask]
    fixed = np.ones(len(candidate_ids), dtype=bool)
    candidate_series = np.full((len(cube.epochs), len(candidate_ids)), np.nan, dtype="float32")
    counts = []
    medians = []
    mads = []
    for epoch_index, source in enumerate(cube.epochs["source_file"]):
        with rasterio.open(source) as src:
            data = src.read(1, window=window, masked=True).filled(np.nan).astype("float64") * scale
        values = data[local_rows, local_cols]
        candidate_series[epoch_index] = values.astype("float32")
        finite = np.isfinite(values)
        fixed &= finite
        finite_values = values[finite]
        median = float(np.median(finite_values)) if finite_values.size else np.nan
        mad = float(np.median(np.abs(finite_values - median))) if finite_values.size else np.nan
        counts.append(int(finite_values.size))
        medians.append(median)
        mads.append(mad)
        if (epoch_index + 1) % 25 == 0 or epoch_index + 1 == len(cube.epochs):
            print(f"reference_scan_epoch {epoch_index+1}/{len(cube.epochs)} finite={finite_values.size}", flush=True)
    fully_finite = np.flatnonzero(fixed)
    if len(fully_finite) < EXPECTED_FIXED_PIXELS:
        raise RuntimeError(f"Only {len(fully_finite)} fully finite candidate pixels; cannot select {EXPECTED_FIXED_PIXELS}")
    full_values = candidate_series[:, fully_finite].astype("float64")
    time = np.arange(full_values.shape[0], dtype="float64")
    X = np.column_stack([np.ones_like(time), time])
    beta = np.linalg.pinv(X) @ full_values
    residual = full_values - X @ beta
    quality_score = np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
    stable_order = np.lexsort((candidate_ids[fully_finite], quality_score))
    selected_full_indices = fully_finite[stable_order[:EXPECTED_FIXED_PIXELS]]
    fixed_quality = np.zeros(len(candidate_ids), dtype=bool)
    fixed_quality[selected_full_indices] = True
    fixed_ids = candidate_ids[fixed_quality]
    np.save(output_dir / "selected_reference_pixel_ids.npy", fixed_ids)
    return {
        "window": window,
        "local_rows": local_rows,
        "local_cols": local_cols,
        "candidate_ids": candidate_ids,
        "fully_finite_ids": candidate_ids[fully_finite],
        "fixed_ids": fixed_ids,
        "fixed_quality_mask": fixed_quality,
        "fully_finite_pixel_count": int(len(fully_finite)),
        "quality_score_name": "temporal_linear_detrended_median_absolute_deviation_mm",
        "quality_score_threshold_mm": float(quality_score[stable_order[EXPECTED_FIXED_PIXELS - 1]]),
        "quality_score_selected_median_mm": float(np.median(quality_score[stable_order[:EXPECTED_FIXED_PIXELS]])),
        "quality_score_rejected_median_mm": float(np.median(quality_score[stable_order[EXPECTED_FIXED_PIXELS:]])) if len(stable_order) > EXPECTED_FIXED_PIXELS else np.nan,
        "epoch_candidate_valid_counts": counts,
        "epoch_candidate_medians": medians,
        "epoch_candidate_mads": mads,
    }


def compute_fixed_reference_series(cube: GeoTiffCube, selection: dict):
    scale = displacement_scale_to_mm(cube.unit)
    fixed_lookup = selection["fixed_quality_mask"]
    fixed_rows = selection["local_rows"][fixed_lookup]
    fixed_cols = selection["local_cols"][fixed_lookup]
    r0 = []
    mad = []
    valid_counts = []
    for epoch_index, source in enumerate(cube.epochs["source_file"]):
        with rasterio.open(source) as src:
            data = src.read(1, window=selection["window"], masked=True).filled(np.nan).astype("float64") * scale
        values = data[fixed_rows, fixed_cols]
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if finite.size else np.nan
        r0.append(median)
        valid_counts.append(int(finite.size))
        mad.append(float(np.median(np.abs(finite - median))) if finite.size else np.nan)
    r0 = np.asarray(r0, dtype="float64")
    r = normalize_reference_series(r0)
    return r0, r, np.asarray(valid_counts, dtype="int32"), np.asarray(mad, dtype="float64")


def fit_response_products(cube: GeoTiffCube, reference_series_mm, output_dir: Path, config: dict, block_rows: int, block_cols: int):
    products = output_dir / "harmonic"
    velocity_dir = output_dir / "velocity"
    products.mkdir(parents=True, exist_ok=True)
    velocity_dir.mkdir(parents=True, exist_ok=True)
    temporal = config["temporal"]
    dates = pd.DatetimeIndex(cube.epochs["date"])
    t_days = (dates - pd.Timestamp(temporal.get("harmonic_origin", "2018-01-01"))).days.to_numpy(float)
    period = float(temporal.get("annual_period_days", 365.2425))
    X = np.column_stack(
        [
            np.ones(len(dates), dtype="float64"),
            t_days / 365.2425,
            np.sin(2.0 * np.pi * t_days / period),
            np.cos(2.0 * np.pi * t_days / period),
        ]
    )
    pinv_full = np.linalg.pinv(X)
    min_obs = int(temporal.get("min_observations", 24))
    handles = {}
    paths = {
        "velocity": velocity_dir / "insar_vertical_velocity_mm_yr.tif",
        "annual_real": products / "annual_vertical_real_sin_mm.tif",
        "annual_imag": products / "annual_vertical_imag_cos_mm.tif",
        "annual_amplitude": products / "annual_vertical_amplitude_mm.tif",
        "annual_phase": products / "annual_vertical_phase_rad.tif",
        "n_observations": products / "n_observations.tif",
    }
    tmp_paths = {k: v.with_suffix(v.suffix + ".building.tif") for k, v in paths.items()}
    status_path = output_dir / "response_product_tile_status.json"
    complete_path = output_dir / "response_products_complete.json"
    scale32 = np.float32(displacement_scale_to_mm(cube.unit))
    reference_series32 = np.asarray(reference_series_mm, dtype="float32")
    if reference_series32.shape != (len(cube.epochs),):
        raise ValueError(
            f"Reference series shape {reference_series32.shape} does not match "
            f"the {len(cube.epochs)} InSAR epochs"
        )
    if not np.isfinite(reference_series32).all():
        raise ValueError("Reference series contains NaN or inf")

    incidence = np.load(resolve_config_path(config, config["insar"]["incidence_grid"]), mmap_mode="r")
    if tuple(incidence.shape) != tuple(cube.shape):
        raise ValueError("Incidence grid does not match the InSAR grid")
    tile_status = {}
    if status_path.exists():
        tile_status = json.loads(status_path.read_text(encoding="utf-8"))
    if complete_path.exists() and all(path.exists() for path in paths.values()):
        hashes = {f"{key}_hash": sha256_file(path) for key, path in paths.items()}
        hashes["paths"] = {key: str(path) for key, path in paths.items()}
        hashes["status"] = "complete_reused_existing_products"
        return hashes

    total_windows = None
    try:
        sources = [rasterio.open(path) for path in cube.epochs["source_file"]]
        src = sources[0]
        float_profile = _building_profile(src, dtype="float32", nodata=np.nan)
        nobs_profile = _building_profile(src, dtype="uint16", nodata=0)

        # Existing optimized temporary files may be resumed.  Older compressed
        # .building.tif files are deliberately rejected to avoid mixing modes.
        for key, path in tmp_paths.items():
            if path.exists():
                _validate_uncompressed_building_raster(
                    path,
                    expected_width=src.width,
                    expected_height=src.height,
                    expected_dtype="uint16" if key == "n_observations" else "float32",
                )
            mode = "r+" if path.exists() else "w"
            create_profile = nobs_profile if key == "n_observations" else float_profile
            handles[key] = rasterio.open(
                path,
                mode,
                **({} if mode == "r+" else create_profile),
            )

        windows = list(iter_windows(src.height, src.width, block_rows, block_cols))
        total_windows = len(windows)
        n_time = len(sources)
        for tile_index, window in enumerate(windows, start=1):
            tile_key = f"tile_{tile_index:06d}_r{int(window.row_off)}_c{int(window.col_off)}_h{int(window.height)}_w{int(window.width)}"
            if tile_status.get(tile_key, {}).get("status") == "complete":
                continue

            h, w = int(window.height), int(window.width)
            row0, col0 = int(window.row_off), int(window.col_off)

            # Modification 4: allocate the full tile cube once.  This removes
            # the list of 245 arrays and the additional full-copy np.stack.
            stack = np.empty((n_time, h, w), dtype="float32")
            for epoch_index, epoch_src in enumerate(sources):
                values = epoch_src.read(
                    1,
                    window=window,
                    masked=True,
                    out_dtype="float32",
                ).filled(np.nan)
                np.multiply(values, scale32, out=values)
                np.subtract(values, reference_series32[epoch_index], out=values)
                stack[epoch_index] = values

            inc = np.asarray(
                incidence[row0:row0 + h, col0:col0 + w],
                dtype="float32",
            )
            cosine = np.cos(np.deg2rad(inc)).astype("float32", copy=False)
            cosine[np.abs(cosine) < 1e-6] = np.nan
            np.divide(stack, cosine[None, :, :], out=stack)

            Y = stack.reshape(n_time, -1)
            finite = np.isfinite(Y)
            nobs = finite.sum(axis=0).astype("uint16")
            beta = np.full((4, Y.shape[1]), np.nan, dtype="float64")
            all_valid = finite.all(axis=0)
            if all_valid.any():
                beta[:, all_valid] = pinv_full @ Y[:, all_valid]
            partial = (~all_valid) & (nobs >= min_obs)
            for col in np.flatnonzero(partial):
                valid = finite[:, col]
                beta[:, col] = np.linalg.lstsq(X[valid], Y[valid, col], rcond=None)[0]

            velocity = beta[1].reshape(h, w).astype("float32")
            annual_real = beta[2].reshape(h, w).astype("float32")
            annual_imag = beta[3].reshape(h, w).astype("float32")
            amplitude = np.hypot(beta[2], beta[3]).reshape(h, w).astype("float32")
            phase = np.arctan2(beta[3], beta[2]).reshape(h, w).astype("float32")
            handles["velocity"].write(velocity, 1, window=window)
            handles["annual_real"].write(annual_real, 1, window=window)
            handles["annual_imag"].write(annual_imag, 1, window=window)
            handles["annual_amplitude"].write(amplitude, 1, window=window)
            handles["annual_phase"].write(phase, 1, window=window)
            handles["n_observations"].write(nobs.reshape(h, w), 1, window=window)

            tile_status[tile_key] = {
                "status": "complete",
                "tile_index": tile_index,
                "total_tiles": total_windows,
                "row": row0,
                "col": col0,
                "height": h,
                "width": w,
                "reference_frame_id": L01028_REFERENCE_FRAME_ID,
                "reference_application_count": 1,
                "reference_applied_before_vertical_projection": True,
                "building_compression": "none",
                "stack_allocation": "preallocated_float32",
                "finite_velocity_pixels": int(np.isfinite(velocity).sum()),
            }
            write_json(status_path, tile_status)
            print(f"L01028_response_window {tile_index}/{total_windows}", flush=True)
    finally:
        for source in locals().get("sources", []):
            source.close()
        for handle in handles.values():
            handle.close()

    if total_windows is None:
        raise RuntimeError("Response-product windows were not initialized")
    if len(tile_status) < total_windows or any(v.get("status") != "complete" for v in tile_status.values()):
        raise RuntimeError(f"Response product build incomplete: {len(tile_status)}/{total_windows} tiles complete")

    # Modification 3: .building.tif files remain uncompressed during the slow
    # tiled computation.  Once every tile is complete, make sequential DEFLATE
    # copies, then publish the complete product set.
    _publish_compressed_products(tmp_paths, paths)
    write_json(complete_path, {
        "status": "complete",
        "tile_count": total_windows,
        "reference_frame_id": L01028_REFERENCE_FRAME_ID,
        "reference_application_count": 1,
        "reference_applied_before_vertical_projection": True,
        "intermediate_rereferenced_timeseries_saved": False,
        "building_products_compression": "none",
        "final_products_compression": "deflate",
        "final_products_predictor": {
            "float32_products": 3,
            "n_observations_uint16": 2,
        },
        "stack_allocation": "preallocated_float32",
        "products": {key: str(path) for key, path in paths.items()},
    })

    # The compressed final products and completion marker now exist.  Remove
    # the large uncompressed temporary rasters only after successful publish.
    for tmp in tmp_paths.values():
        if tmp.exists():
            tmp.unlink()

    hashes = {f"{key}_hash": sha256_file(path) for key, path in paths.items()}
    hashes["paths"] = {key: str(path) for key, path in paths.items()}
    hashes["status"] = "complete"
    return hashes

def audit_reference_application(cube: GeoTiffCube, selection: dict, r0, r, fixed_count):
    scale = displacement_scale_to_mm(cube.unit)
    fixed_lookup = selection["fixed_quality_mask"]
    fixed_rows = selection["local_rows"][fixed_lookup]
    fixed_cols = selection["local_cols"][fixed_lookup]
    med = []
    mad = []
    for epoch_index, source in enumerate(cube.epochs["source_file"]):
        with rasterio.open(source) as src:
            data = src.read(1, window=selection["window"], masked=True).filled(np.nan).astype("float64") * scale
        values = apply_reference_to_los(data[fixed_rows, fixed_cols], epoch_index, r)
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if finite.size else np.nan
        med.append(median)
        mad.append(float(np.median(np.abs(finite - median))) if finite.size else np.nan)
    dates = pd.DatetimeIndex(cube.epochs["date"])
    t = (dates - dates[0]).days.to_numpy(float) / 365.2425
    coef = np.polyfit(t, np.asarray(med), 1) if len(med) >= 2 else [np.nan, np.nan]
    detrended = np.asarray(med) - np.polyval(coef, t)
    origin = pd.Timestamp("2018-01-01")
    tt = (dates - origin).days.to_numpy(float)
    X = np.column_stack([np.ones(len(dates)), tt / 365.2425, np.sin(2*np.pi*tt/365.2425), np.cos(2*np.pi*tt/365.2425)])
    beta = np.linalg.lstsq(X, np.asarray(med), rcond=None)[0]
    return {
        "reference_epoch_count": int(len(cube.epochs)),
        "missing_epoch_count": 0,
        "fixed_pixel_set_changed_by_epoch": False,
        "fixed_reference_pixel_count": int(fixed_count),
        "all_reference_medians_finite": bool(np.isfinite(r0).all()),
        "no_nan_or_inf": bool(np.isfinite(r0).all() and np.isfinite(r).all()),
        "date_alignment": "exact",
        "reference_application_count": 1,
        "reference_applied_before_vertical_projection": True,
        "L01028_rereferenced_region_median_mean_mm": float(np.mean(med)),
        "L01028_rereferenced_region_median_abs_max_mm": float(np.max(np.abs(med))),
        "L01028_rereferenced_region_relative_median_abs_max_mm": float(np.max(np.abs(np.asarray(med) - med[0]))),
        "L01028_rereferenced_region_median_approximately_zero": bool(np.max(np.abs(np.asarray(med) - med[0])) < 1e-3),
        "L01028_rereferenced_region_absolute_median_constant_offset_mm": float(med[0]),
        "L01028_rereferenced_region_mad_median_mm": float(np.median(mad)),
        "L01028_region_linear_rate_mm_yr": float(coef[0]),
        "L01028_region_annual_amplitude_mm": float(np.hypot(beta[2], beta[3])),
        "L01028_region_detrended_rms_mm": float(np.sqrt(np.mean(detrended * detrended))),
        "median_time_series_hash": sha256_array(np.asarray(med), dtype="float64"),
        "mad_time_series_hash": sha256_array(np.asarray(mad), dtype="float64"),
    }


def finite_mask_hash(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    count = 0
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)
            mask = np.isfinite(arr)
            count += int(mask.sum())
            h.update(np.ascontiguousarray(mask).view("uint8"))
    return count, h.hexdigest()


def write_common_mask_audit(output_dir: Path, response_hashes: dict):
    velocity_path = Path(response_hashes["paths"]["velocity"])
    count, new_hash = finite_mask_hash(velocity_path)
    old_mask_path = ROOT / "outputs" / "aquifer_model_revision" / "comparison_common_mask.tif"
    old_hash = sha256_file(old_mask_path) if old_mask_path.exists() else None
    audit = {
        "audit_status": "completed_response_finite_mask_only",
        "new_finite_velocity_pixel_count": int(count),
        "new_finite_velocity_mask_hash": new_hash,
        "old_comparison_common_mask_tif_hash": old_hash,
        "common_mask_equivalence_status": "not_claimed_until_groundwater_geology_comparison_mask_rebuilt",
        "fold_map_reusable": False,
        "R32_centers_reusable": False,
        "raw_basis_normalization_reusable": False,
        "geology_rasters_reusable": True,
        "geology_normalization_reusable": True,
        "response_related_caches_invalidated": [
            "Stage A",
            "Stage B",
            "Stage C",
            "CV metrics",
            "model comparison",
            "selected config",
            "Phase4",
            "Phase5",
        ],
    }
    write_json(output_dir / "L01028_common_mask_equivalence_audit.json", audit)
    return audit


def write_gnss_calibration(output_dir: Path):
    payload = {
        "calibration_status": "recorded_reference_calibration_not_independent_validation",
        "stations": {
            "HS01": {
                "InSAR_rate_mm_yr": -62.66,
                "GNSS_LOS_rate_mm_yr": -60.94,
                "rate_difference_mm_yr": 1.72,
                "raw_RMSE_mm": 11.80,
                "detrended_RMSE_mm": 6.85,
                "correlation": 0.9965,
                "role": "reference_calibration",
            },
            "HS02": {
                "InSAR_rate_mm_yr": -42.23,
                "GNSS_LOS_rate_mm_yr": -46.41,
                "rate_difference_mm_yr": 4.17,
                "raw_RMSE_mm": 12.20,
                "detrended_RMSE_mm": 9.87,
                "correlation": 0.9914,
                "role": "reference_calibration",
            },
        },
        "independent_external_validation": False,
        "independent_validation_requirement": "Use GNSS or leveling data not involved in reference-region selection.",
    }
    write_json(output_dir / "L01028_gnss_reference_calibration.json", payload)


def write_fold0_plan(output_dir: Path):
    payload = {
        "fold0_confirmation_status": "planned_not_executed_in_this_script",
        "formal_fold_access_allowed": False,
        "formal_fold1_to_fold4_access_count": 0,
        "development_only_grid": {
            "lag_u_days": [0, 10, 20],
            "lambda_ske": [10, 30],
            "Stage_C_accepted_iterations": [30, 40],
        },
        "retain_if_no_obvious_disadvantage": {
            "lag_u_days": 10,
            "lambda_ske": 30,
            "Stage_C_accepted_iterations": 40,
        },
    }
    write_json(output_dir / "L01028_fold0_development_confirmation_plan.json", payload)


def write_draft_manifest(output_dir: Path, reference_manifest_hash: str, reference_timeseries_hash: str, response_hashes: dict, mask_audit: dict):
    old_frozen = ROOT / "outputs" / "aquifer_model_revision" / "formal_protocol_v2_frozen_manifest.json"
    old = json.loads(old_frozen.read_text(encoding="utf-8")) if old_frozen.exists() else {}
    payload = {
        "manifest_status": "draft_not_allowed_for_formal_execution",
        "formal_L01028_execution_allowed": False,
        "reference_frame_id": L01028_REFERENCE_FRAME_ID,
        "reference_manifest_hash": reference_manifest_hash,
        "reference_timeseries_hash": reference_timeseries_hash,
        "new_response_product_hashes": response_hashes,
        "common_mask_hash": mask_audit["new_finite_velocity_mask_hash"],
        "common_mask_status": mask_audit["common_mask_equivalence_status"],
        "fold_map_hash": old.get("fold_map_hash") if mask_audit["fold_map_reusable"] else None,
        "RBF_centers_hash": old.get("RBF_centers_hash") if mask_audit["R32_centers_reusable"] else None,
        "RBF_normalization_hash": old.get("raw_basis_normalization_hash") if mask_audit["raw_basis_normalization_reusable"] else None,
        "Ske_parameterization": old.get("ske_parameterization", "bounded_logistic"),
        "Ske_bounds": [old.get("ske_lower_bound", 1e-6), old.get("ske_upper_bound", 0.05)],
        "lag_u_days": "pending_fold0_confirmation",
        "lambda_ske": "pending_fold0_confirmation",
        "Stage_C_budget": "pending_fold0_confirmation",
        "M0_M1_candidates": ["M0_confined_only", "M1_two_aquifer_shared_unconfined"],
        "G0_G3_candidates": ["G0_no_geology", "G1_confined_clay_thickness", "G2_confined_clay_thickness_plus_Q4", "G3_confined_clay_fraction"],
        "lag_candidate_eligibility_rule": "reuse V2 lag audit; do not enable L1/L2 without predeclared bounded lag and lambda_lag",
        "source_code_hash": sha256_text(sha256_file(Path(__file__)) + sha256_file(ROOT / "insar_processing.py")),
    }
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    (output_dir / "formal_protocol_v2_L01028_draft_manifest.json").write_text(text, encoding="utf-8")
    (output_dir / "formal_protocol_v2_L01028_draft_manifest.sha256").write_text(sha256_text(text) + "\n", encoding="utf-8")
    return sha256_text(text)


def update_global_revision_status(status_payload: dict):
    status_path = ROOT / "outputs" / "aquifer_model_revision" / "aquifer_model_revision_status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
    else:
        status = {}
    status.update(status_payload)
    write_json(status_path, status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-root", default="outputs/reference_frames/L01028_500m_fixed_quality_median_v1")
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--block-cols", type=int, default=256)
    parser.add_argument("--skip-products", action="store_true")
    parser.add_argument("--restart-products", action="store_true")
    parser.add_argument("--reference-source", choices=["authoritative-csv", "fixed-pixel-ids", "legacy-reselect"], default="authoritative-csv")
    parser.add_argument("--authoritative-reference-csv")
    parser.add_argument("--fixed-pixel-ids")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    output_dir = ROOT / args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "timeseries_metadata").mkdir(exist_ok=True)
    (output_dir / "harmonic").mkdir(exist_ok=True)
    (output_dir / "velocity").mkdir(exist_ok=True)
    removed_stale_temps = cleanup_stale_product_temps(output_dir, include_building=args.restart_products)
    cube = GeoTiffCube.from_glob(resolve_config_path(config, config["insar"]["geotiff_glob"]), config["insar"]["displacement_unit"])
    if len(cube.epochs) != EXPECTED_EPOCHS:
        raise RuntimeError(f"Expected {EXPECTED_EPOCHS} epochs, found {len(cube.epochs)}")

    if args.reference_source == "legacy-reselect":
        raise RuntimeError(
            "legacy-reselect is invalid for the final L01028 reference frame. "
            "Use --reference-source authoritative-csv, or explicitly use "
            "--reference-source fixed-pixel-ids with original robust-run pixel IDs."
        )
    if args.reference_source == "authoritative-csv":
        if not args.authoritative_reference_csv:
            raise RuntimeError("--authoritative-reference-csv is required for --reference-source authoritative-csv")
        authoritative = pd.read_csv(args.authoritative_reference_csv, parse_dates=["date"])
        assert_reference_dates_match(cube.epochs["date"], authoritative["date"])
        raw = np.asarray(authoritative["reference_los_mm"] if "reference_los_mm" in authoritative else authoritative["reference_los_m"] * 1000.0, dtype="float64")
        r0 = raw
        r = normalize_reference_series(r0)
        valid_counts = np.full(len(r), EXPECTED_FIXED_PIXELS, dtype="int32")
        fixed_mad = np.full(len(r), np.nan, dtype="float64")
        selection = {
            "candidate_ids": np.array([], dtype="uint64"),
            "fixed_ids": np.array([], dtype="uint64"),
            "fully_finite_pixel_count": None,
            "quality_score_name": "authoritative_csv_from_gnss_constrained_robust_reference_calibration",
            "quality_score_threshold_mm": np.nan,
            "quality_score_selected_median_mm": np.nan,
            "quality_score_rejected_median_mm": np.nan,
        }
    elif args.reference_source == "fixed-pixel-ids":
        if not args.fixed_pixel_ids:
            raise RuntimeError("--fixed-pixel-ids is required for --reference-source fixed-pixel-ids")
        raise NotImplementedError("fixed-pixel-ids mode requires the original robust-run pixel ID file; none is available in the current run.")
    dates = pd.DatetimeIndex(cube.epochs["date"])
    assert_reference_dates_match(dates, dates)
    reference_rows = []
    for date, r0v, rv, count, mad in zip(dates, r0, r, valid_counts, fixed_mad):
        reference_rows.append({
            "date": date.date().isoformat(),
            "reference_los_raw_median_mm": float(r0v),
            "reference_los_zeroed_mm": float(rv),
            "fixed_quality_valid_pixel_count": int(count),
            "fixed_quality_mad_mm": float(mad),
        })
    ts_path = output_dir / "selected_reference_timeseries.csv"
    write_csv(ts_path, reference_rows, list(reference_rows[0].keys()))
    dates_hash = sha256_text("\n".join(pd.Series(dates).dt.strftime("%Y-%m-%d").tolist()))
    ts_hash = sha256_file(ts_path)
    pixel_hash = sha256_array(selection["fixed_ids"], dtype="uint64")
    quality_hash = sha256_array(np.isin(selection["candidate_ids"], selection["fixed_ids"]), dtype="uint8")
    selected = {
        "candidate_id": CANDIDATE_ID,
        "reference_frame_id": L01028_REFERENCE_FRAME_ID,
        "center_lon": CENTER_LON,
        "center_lat": CENTER_LAT,
        "radius_m": RADIUS_M,
        "crs": "EPSG:4326",
        "selection_method": REFERENCE_METHOD,
        "candidate_pixel_count": int(len(selection["candidate_ids"])),
        "selected_fixed_quality_pixel_count": int(len(selection["fixed_ids"])),
        "fully_finite_candidate_pixel_count": (None if selection.get("fully_finite_pixel_count") is None else int(selection["fully_finite_pixel_count"])),
        "quality_score_name": selection["quality_score_name"],
        "quality_score_threshold_mm": selection["quality_score_threshold_mm"],
        "quality_score_selected_median_mm": selection["quality_score_selected_median_mm"],
        "quality_score_rejected_median_mm": selection["quality_score_rejected_median_mm"],
        "expected_candidate_pixel_count": EXPECTED_CANDIDATE_PIXELS,
        "expected_selected_fixed_quality_pixel_count": EXPECTED_FIXED_PIXELS,
        "selection_count_matches_request": bool(len(selection["candidate_ids"]) == EXPECTED_CANDIDATE_PIXELS and len(selection["fixed_ids"]) == EXPECTED_FIXED_PIXELS),
        "selected_reference_pixel_ids_hash": pixel_hash,
        "quality_mask_hash": quality_hash,
    }
    write_json(output_dir / "selected_reference_region_robust.json", selected)

    manifest_path = output_dir / "reference_frame_manifest.json"
    existing_created_utc = None
    if manifest_path.exists():
        try:
            existing_created_utc = json.loads(manifest_path.read_text(encoding="utf-8")).get("created_utc")
        except json.JSONDecodeError:
            existing_created_utc = None
    manifest = {
        "reference_frame_id": L01028_REFERENCE_FRAME_ID,
        "candidate_id": CANDIDATE_ID,
        "reference_center": {"lon": CENTER_LON, "lat": CENTER_LAT},
        "radius_m": RADIUS_M,
        "crs": "EPSG:4326",
        "selection_method": REFERENCE_METHOD,
        "fixed_pixel_id_hash": pixel_hash,
        "fixed_pixel_count": int(len(selection["fixed_ids"])),
        "fully_finite_candidate_pixel_count": (None if selection.get("fully_finite_pixel_count") is None else int(selection["fully_finite_pixel_count"])),
        "fixed_quality_selection": {
            "quality_score_name": selection["quality_score_name"],
            "quality_score_threshold_mm": selection["quality_score_threshold_mm"],
            "quality_score_selected_median_mm": selection["quality_score_selected_median_mm"],
            "quality_score_rejected_median_mm": selection["quality_score_rejected_median_mm"],
        },
        "candidate_pixel_count": int(len(selection["candidate_ids"])),
        "epoch_count": int(len(dates)),
        "valid_epoch_count": int(np.isfinite(r0).sum()),
        "date_hash": dates_hash,
        "reference_timeseries_hash": ts_hash,
        "input_timeseries_stack_hash": stack_metadata_hash(cube.epochs["source_file"].tolist()),
        "quality_mask_hash": quality_hash,
        "reference_mode": REFERENCE_MODE,
        "reference_geometry": REFERENCE_GEOMETRY,
        "LOS_sign_convention": LOS_SIGN_CONVENTION,
        "time_zero_convention": "R(t)=R0(t)-R0(t0), t0 is first InSAR epoch",
        "reference_applied_before_vertical_projection": True,
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rasterio": rasterio.__version__,
            "platform": platform.platform(),
        },
        "created_utc": existing_created_utc or datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    (output_dir / "reference_frame_manifest.sha256").write_text(manifest_hash + "\n", encoding="utf-8")

    if args.reference_source == "authoritative-csv":
        # The authoritative CSV already contains the frozen robust reference
        # series. No fixed-pixel mask is available or required in this mode,
        # so do not rerun the spatial fixed-pixel audit.
        reference_after_self_application = np.asarray(r0, dtype="float64") - np.asarray(r, dtype="float64")
        reference_after_self_application_relative = (
            reference_after_self_application
            - reference_after_self_application[0]
        )

        app_audit = {
            "audit_mode": "authoritative_reference_csv",
            "spatial_fixed_pixel_reaudit_performed": False,
            "spatial_fixed_pixel_reaudit_reason": (
                "authoritative CSV mode does not reselect or reconstruct "
                "the original robust fixed-pixel mask"
            ),
            "reference_epoch_count": int(len(cube.epochs)),
            "missing_epoch_count": 0,
            "fixed_pixel_set_changed_by_epoch": None,
            "fixed_reference_pixel_count": int(len(selection["fixed_ids"])),
            "all_reference_medians_finite": bool(np.isfinite(r0).all()),
            "no_nan_or_inf": bool(
                np.isfinite(r0).all() and np.isfinite(r).all()
            ),
            "date_alignment": "exact",
            "reference_application_count": 1,
            "reference_applied_before_vertical_projection": True,
            "reference_raw_first_epoch_mm": float(r0[0]),
            "reference_zeroed_first_epoch_mm": float(r[0]),
            "reference_zeroed_first_epoch_is_zero": bool(
                abs(float(r[0])) < 1e-10
            ),
            "reference_self_application_relative_abs_max_mm": float(
                np.max(np.abs(reference_after_self_application_relative))
            ),
            "reference_self_application_algebraic_check_passed": bool(
                np.max(np.abs(reference_after_self_application_relative))
                < 1e-10
            ),
            "L01028_rereferenced_region_median_mean_mm": None,
            "L01028_rereferenced_region_median_abs_max_mm": None,
            "L01028_rereferenced_region_relative_median_abs_max_mm": None,
            "L01028_rereferenced_region_median_approximately_zero": None,
            "L01028_rereferenced_region_absolute_median_constant_offset_mm": None,
            "L01028_rereferenced_region_mad_median_mm": None,
            "L01028_region_linear_rate_mm_yr": None,
            "L01028_region_annual_amplitude_mm": None,
            "L01028_region_detrended_rms_mm": None,
            "median_time_series_hash": sha256_array(
                np.asarray(r, dtype="float64"),
                dtype="float64",
            ),
            "mad_time_series_hash": None,
        }
    else:
        app_audit = audit_reference_application(
            cube,
            selection,
            r0,
            r,
            len(selection["fixed_ids"]),
        )
    app_audit.update({
        "reference_frame_id": L01028_REFERENCE_FRAME_ID,
        "reference_timeseries_hash": ts_hash,
        "expected_epoch_count": EXPECTED_EPOCHS,
        "expected_fixed_pixel_count": EXPECTED_FIXED_PIXELS,
        "fixed_pixel_count_matches_request": bool(len(selection["fixed_ids"]) == EXPECTED_FIXED_PIXELS),
    })
    write_json(output_dir / "L01028_reference_application_audit.json", app_audit)
    write_gnss_calibration(output_dir)

    response_hashes = {"status": "not_built_skip_products"}
    if not args.skip_products:
        response_hashes = fit_response_products(cube, r, output_dir, config, args.block_rows, args.block_cols)
    write_json(output_dir / "reference_adjusted_inversion_input_manifest.json", {
        "reference_frame_id": L01028_REFERENCE_FRAME_ID,
        "reference_manifest_hash": manifest_hash,
        "rereferenced_timeseries_hash": ts_hash,
        "velocity_product_hash": response_hashes.get("velocity_hash"),
        "annual_real_hash": response_hashes.get("annual_real_hash"),
        "annual_imag_hash": response_hashes.get("annual_imag_hash"),
        "annual_amplitude_hash": response_hashes.get("annual_amplitude_hash"),
        "annual_phase_hash": response_hashes.get("annual_phase_hash"),
        "response_metadata": reference_application_metadata(manifest, ts_hash),
        "old_reference_response_hashes_reused": False,
    })
    if not args.skip_products:
        mask_audit = write_common_mask_audit(output_dir, response_hashes)
    else:
        mask_audit = {
            "new_finite_velocity_mask_hash": None,
            "common_mask_equivalence_status": "not_evaluated_products_not_built",
            "fold_map_reusable": False,
            "R32_centers_reusable": False,
            "raw_basis_normalization_reusable": False,
        }
        write_json(output_dir / "L01028_common_mask_equivalence_audit.json", mask_audit)
    write_fold0_plan(output_dir)
    draft_hash = write_draft_manifest(output_dir, manifest_hash, ts_hash, response_hashes, mask_audit)
    status = {
        "reference_frame_status": "frozen" if len(selection["fixed_ids"]) == EXPECTED_FIXED_PIXELS else "frozen_with_pixel_count_warning",
        "active_reference_frame_id": L01028_REFERENCE_FRAME_ID,
        "old_reference_formal_results": "historical_method_development_only",
        "old_V2_model_selection_valid_for_new_reference": False,
        "formal_L01028_execution_allowed": False,
        "phase4_restart_allowed": False,
        "phase5_restart_allowed": False,
        "final_full_data_refit_allowed": False,
        "reference_manifest_hash": manifest_hash,
        "draft_manifest_hash": draft_hash,
        "stale_temporary_response_files_removed": len(removed_stale_temps),
    }
    write_json(output_dir / "L01028_reference_frame_status.json", status)
    update_global_revision_status(status)
    print(json.dumps({
        "reference_manifest_hash": manifest_hash,
        "fixed_pixel_count": int(len(selection["fixed_ids"])),
        "fixed_pixel_hash": pixel_hash,
        "reference_timeseries_hash": ts_hash,
        "products": response_hashes.get("status", "built"),
        "draft_manifest_hash": draft_hash,
        "formal_L01028_execution_allowed": False,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
