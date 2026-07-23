"""Audit completed L01028 response products and freeze the formal manifest.

The script is deliberately gate-first: source-reference equivalence must pass
before product, mask, fold0, and formal-manifest stages are allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insar_processing import (  # noqa: E402
    GeoTiffCube,
    L01028_REFERENCE_FRAME_ID,
    apply_reference_to_los,
    assert_reference_dates_match,
    displacement_scale_to_mm,
)
from io_utils import load_config, resolve_config_path  # noqa: E402


REF = ROOT / "outputs" / "reference_frames" / L01028_REFERENCE_FRAME_ID
REV = ROOT / "outputs" / "aquifer_model_revision"
PRODUCTS = {
    "velocity": REF / "velocity" / "insar_vertical_velocity_mm_yr.tif",
    "annual_real": REF / "harmonic" / "annual_vertical_real_sin_mm.tif",
    "annual_imag": REF / "harmonic" / "annual_vertical_imag_cos_mm.tif",
    "annual_amplitude": REF / "harmonic" / "annual_vertical_amplitude_mm.tif",
    "annual_phase": REF / "harmonic" / "annual_vertical_phase_rad.tif",
    "n_observations": REF / "harmonic" / "n_observations.tif",
}


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_array(values, dtype="float64") -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(arr.view("uint8")).hexdigest()


def update_status(payload):
    path = REV / "aquifer_model_revision_status.json"
    status = read_json(path) if path.exists() else {}
    status.update(payload)
    write_json(path, status)


def find_auto_reference_source(explicit_path: str | None) -> Path | None:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            ROOT / "outputs" / "auto_reference_selection_robust" / "selected_reference_timeseries.csv",
            ROOT.parent / "outputs" / "auto_reference_selection_robust" / "selected_reference_timeseries.csv",
            ROOT.parent.parent / "outputs" / "auto_reference_selection_robust" / "selected_reference_timeseries.csv",
        ]
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def reference_source_equivalence(auto_path: Path | None):
    out_path = REF / "L01028_reference_source_equivalence_audit.json"
    local_path = REF / "selected_reference_timeseries.csv"
    manifest = read_json(REF / "reference_frame_manifest.json")
    selected = read_json(REF / "selected_reference_region_robust.json")
    if auto_path is None:
        payload = {
            "status": "blocked_missing_auto_reference_source",
            "required_source": "outputs/auto_reference_selection_robust/selected_reference_timeseries.csv",
            "local_reference_timeseries": str(local_path),
            "reference_frame_id": manifest["reference_frame_id"],
            "fixed_pixel_hash": selected["selected_reference_pixel_ids_hash"],
            "continue_allowed": False,
        }
        write_json(out_path, payload)
        update_status({"formal_L01028_execution_allowed": False, "L01028_block_reason": payload["status"]})
        return payload
    auto = pd.read_csv(auto_path, parse_dates=["date"])
    local = pd.read_csv(local_path, parse_dates=["date"])
    region_path = auto_path.parent / "selected_reference_region_robust.json"
    auto_region = read_json(region_path) if region_path.exists() else {}
    raw_col = "reference_los_raw_median_mm"
    zero_col = "reference_los_zeroed_mm"
    if raw_col not in local or zero_col not in local:
        raise ValueError("Local L01028 reference timeseries must include raw and zeroed reference columns")
    if raw_col in auto:
        auto_raw = np.asarray(auto[raw_col], float)
    elif "reference_los_mm" in auto:
        auto_raw = np.asarray(auto["reference_los_mm"], float)
    elif "reference_los_m" in auto:
        auto_raw = np.asarray(auto["reference_los_m"], float) * 1000.0
    else:
        auto_raw = np.full(len(auto), np.nan)
    if zero_col in auto:
        auto_zero = np.asarray(auto[zero_col], float)
    else:
        auto_zero = auto_raw - float(auto_raw[0]) if len(auto_raw) else auto_raw
    local_raw = np.asarray(local[raw_col], float)
    local_zero = np.asarray(local[zero_col], float)
    dates_match = len(auto) == len(local) == 245 and np.all(pd.DatetimeIndex(auto["date"]) == pd.DatetimeIndex(local["date"]))
    raw_diff = auto_raw - local_raw
    zero_diff = auto_zero - local_zero
    numeric_hash_auto = sha256_array(np.column_stack([auto_raw, auto_zero]))
    numeric_hash_local = sha256_array(np.column_stack([local_raw, local_zero]))
    auto_frame_id = auto_region.get("reference_frame_id") or (
        f"{auto_region.get('candidate_id', '')}_500m_fixed_quality_median_v1"
        if auto_region.get("candidate_id") == "L01028" and float(auto_region.get("radius_m", 0)) == 500.0
        else None
    )
    auto_fixed_hash = auto_region.get("fixed_pixel_hash") or auto_region.get("selected_reference_pixel_ids_hash")
    auto_fixed_count = auto_region.get("n_quality_pixels")
    local_fixed_count = selected.get("selected_fixed_quality_pixel_count")
    raw_ok = bool(np.isfinite(raw_diff).all() and np.nanmax(np.abs(raw_diff)) == 0)
    zero_ok = bool(np.isfinite(zero_diff).all() and np.nanmax(np.abs(zero_diff)) == 0)
    hash_ok = numeric_hash_auto == numeric_hash_local
    frame_ok = auto_frame_id == manifest["reference_frame_id"]
    fixed_hash_ok = auto_fixed_hash == selected["selected_reference_pixel_ids_hash"] if auto_fixed_hash else False
    fixed_count_ok = int(auto_fixed_count) == int(local_fixed_count) if auto_fixed_count is not None and local_fixed_count is not None else False
    payload = {
        "status": "passed" if dates_match and raw_ok and zero_ok and hash_ok and frame_ok and fixed_hash_ok else "failed",
        "auto_reference_timeseries": str(auto_path),
        "local_reference_timeseries": str(local_path),
        "date_count_auto": int(len(auto)),
        "date_count_local": int(len(local)),
        "dates_exactly_match": bool(dates_match),
        "raw_median_max_abs_diff_mm": float(np.nanmax(np.abs(raw_diff))),
        "zeroed_reference_max_abs_diff_mm": float(np.nanmax(np.abs(zero_diff))),
        "numeric_array_hash_auto": numeric_hash_auto,
        "numeric_array_hash_local": numeric_hash_local,
        "numeric_array_hash_match": hash_ok,
        "auto_reference_frame_id": auto_frame_id,
        "local_reference_frame_id": manifest["reference_frame_id"],
        "reference_frame_id_match": frame_ok,
        "auto_fixed_pixel_hash": auto_fixed_hash,
        "local_fixed_pixel_hash": selected["selected_reference_pixel_ids_hash"],
        "fixed_pixel_hash_match": fixed_hash_ok,
        "auto_fixed_pixel_count": auto_fixed_count,
        "local_fixed_pixel_count": local_fixed_count,
        "fixed_pixel_count_match": fixed_count_ok,
        "continue_allowed": bool(dates_match and raw_ok and zero_ok and hash_ok and frame_ok and fixed_hash_ok),
        "stop_reason": None if dates_match and raw_ok and zero_ok and hash_ok and frame_ok and fixed_hash_ok else "reference_source_equivalence_failed",
    }
    write_json(out_path, payload)
    if not payload["continue_allowed"]:
        update_status({"formal_L01028_execution_allowed": False, "L01028_block_reason": "reference_source_equivalence_failed"})
    return payload


def write_directory_reference_metadata(original_dir: Path, corrected_dir: Path, allow_external_writes: bool):
    payloads = {
        original_dir / "reference_state_metadata.json": {
            "input_reference_state": "original_reference",
            "dynamic_reference_allowed": True,
            "reference_frame_id": None,
        },
        corrected_dir / "reference_state_metadata.json": {
            "input_reference_state": "already_rereferenced",
            "dynamic_reference_allowed": False,
            "reference_frame_id": L01028_REFERENCE_FRAME_ID,
            "reference_application_count": 1,
        },
    }
    results = {}
    for path, payload in payloads.items():
        try:
            if not allow_external_writes and not str(path.resolve()).startswith(str(ROOT.resolve())):
                raise PermissionError("external metadata write requires --allow-external-metadata-writes")
            write_json(path, payload)
            results[str(path)] = "written"
        except PermissionError as exc:
            results[str(path)] = f"not_written_permission_required: {exc}"
    write_json(REF / "L01028_reference_state_metadata_write_audit.json", results)
    return results


def update_product_tags():
    manifest = read_json(REF / "reference_frame_manifest.json")
    tags = {
        "reference_applied": "true",
        "reference_frame_id": L01028_REFERENCE_FRAME_ID,
        "reference_application_count": "1",
        "reference_timeseries_hash": manifest["reference_timeseries_hash"],
        "reference_mode": "on_the_fly",
        "reference_geometry": "native_LOS",
        "reference_applied_before_vertical_projection": "true",
    }
    for path in PRODUCTS.values():
        with rasterio.open(path, "r+") as dst:
            dst.update_tags(**tags)


def product_integrity_audit():
    update_product_tags()
    leftover = list(REF.glob("**/*.building.tif")) + list(REF.glob("**/*.tmp")) + list(REF.glob("**/*.tmp*"))
    complete = read_json(REF / "response_products_complete.json")
    tile_status = read_json(REF / "response_product_tile_status.json")
    hashes = {f"{name}_hash": sha256_file(path) for name, path in PRODUCTS.items()}
    profiles = {}
    for name, path in PRODUCTS.items():
        with rasterio.open(path) as src:
            profiles[name] = {
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),
                "transform": tuple(src.transform),
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
                "tags": src.tags(),
            }
    base = profiles["velocity"]
    aligned = all((p["width"], p["height"], p["crs"], p["transform"]) == (base["width"], base["height"], base["crs"], base["transform"]) for p in profiles.values())
    max_amp_diff = 0.0
    max_phase_diff = 0.0
    finite_counts = {}
    nobs_min = None
    nobs_max = None
    abnormal = {}
    with rasterio.open(PRODUCTS["annual_real"]) as real, rasterio.open(PRODUCTS["annual_imag"]) as imag, rasterio.open(PRODUCTS["annual_amplitude"]) as amp, rasterio.open(PRODUCTS["annual_phase"]) as phase, rasterio.open(PRODUCTS["n_observations"]) as nobs:
        for _, window in real.block_windows(1):
            r = real.read(1, window=window)
            i = imag.read(1, window=window)
            a = amp.read(1, window=window)
            p = phase.read(1, window=window)
            n = nobs.read(1, window=window)
            calc_amp = np.hypot(r, i)
            calc_phase = np.arctan2(i, r)
            valid_amp = np.isfinite(a) & np.isfinite(calc_amp)
            valid_phase = np.isfinite(p) & np.isfinite(calc_phase)
            if valid_amp.any():
                max_amp_diff = max(max_amp_diff, float(np.nanmax(np.abs(a[valid_amp] - calc_amp[valid_amp]))))
            if valid_phase.any():
                max_phase_diff = max(max_phase_diff, float(np.nanmax(np.abs(p[valid_phase] - calc_phase[valid_phase]))))
            nobs_min = int(np.nanmin(n)) if nobs_min is None else min(nobs_min, int(np.nanmin(n)))
            nobs_max = int(np.nanmax(n)) if nobs_max is None else max(nobs_max, int(np.nanmax(n)))
    for name, path in PRODUCTS.items():
        count = 0
        bad = 0
        with rasterio.open(path) as src:
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window)
                if name == "n_observations":
                    finite = arr > 0
                    bad += int(((arr < 0) | (arr > 245)).sum())
                else:
                    finite = np.isfinite(arr)
                    bad += int(np.isinf(arr).sum())
                count += int(finite.sum())
        finite_counts[name] = count
        abnormal[name] = bad
    metadata_ok = all(p["tags"].get("reference_frame_id") == L01028_REFERENCE_FRAME_ID for p in profiles.values())
    payload = {
        "status": "passed" if not leftover and aligned and metadata_ok and max_amp_diff <= 1e-5 and max_phase_diff <= 1e-6 and all(v == 0 for v in abnormal.values()) and complete.get("status") == "complete" else "failed",
        "leftover_building_or_tmp_files": [str(p) for p in leftover],
        "grid_alignment_ok": bool(aligned),
        "reference_metadata_ok": bool(metadata_ok),
        "all_tile_count": complete.get("tile_count"),
        "completed_tile_count": sum(1 for v in tile_status.values() if v.get("status") == "complete"),
        "all_tiles_complete": bool(sum(1 for v in tile_status.values() if v.get("status") == "complete") == complete.get("tile_count")),
        "amplitude_hypot_max_abs_diff": max_amp_diff,
        "phase_atan2_max_abs_diff": max_phase_diff,
        "finite_pixel_counts": finite_counts,
        "abnormal_value_counts": abnormal,
        "n_observations_min": nobs_min,
        "n_observations_max": nobs_max,
        "n_observations_range_legal": bool(nobs_min is not None and nobs_min >= 0 and nobs_max <= 245),
        "product_hashes": hashes,
    }
    write_json(REF / "L01028_product_integrity_audit.json", payload)
    return payload


def fit_coefficients_from_stack(stack, dates, config):
    temporal = config["temporal"]
    t_days = (pd.DatetimeIndex(dates) - pd.Timestamp(temporal.get("harmonic_origin", "2018-01-01"))).days.to_numpy(float)
    period = float(temporal.get("annual_period_days", 365.2425))
    X = np.column_stack([np.ones(len(dates)), t_days / 365.2425, np.sin(2*np.pi*t_days/period), np.cos(2*np.pi*t_days/period)])
    pinv = np.linalg.pinv(X)
    n, h, w = stack.shape
    Y = stack.reshape(n, -1)
    finite = np.isfinite(Y)
    beta = np.full((4, Y.shape[1]), np.nan, dtype="float64")
    all_valid = finite.all(axis=0)
    if all_valid.any():
        beta[:, all_valid] = pinv @ Y[:, all_valid]
    return beta.reshape(4, h, w)


def explicit_vs_onthefly_audit(config, corrected_dir: Path):
    original_cube = GeoTiffCube.from_glob(resolve_config_path(config, config["insar"]["geotiff_glob"]), config["insar"]["displacement_unit"])
    corrected_cube = GeoTiffCube.from_glob(corrected_dir / "geo_*.tif", config["insar"]["displacement_unit"])
    assert_reference_dates_match(original_cube.epochs["date"], corrected_cube.epochs["date"])
    ts = pd.read_csv(REF / "selected_reference_timeseries.csv", parse_dates=["date"])
    assert_reference_dates_match(original_cube.epochs["date"], ts["date"])
    ref = ts["reference_los_zeroed_mm"].to_numpy(float)
    incidence = np.load(resolve_config_path(config, config["insar"]["incidence_grid"]), mmap_mode="r")
    scale = displacement_scale_to_mm(config["insar"]["displacement_unit"])
    windows = []
    with rasterio.open(original_cube.epochs["source_file"].iloc[0]) as src:
        for row in np.linspace(0, src.height - 64, 5).astype(int):
            for col in np.linspace(0, src.width - 64, 5).astype(int):
                windows.append(Window(int(col), int(row), 64, 64))
    rows = []
    max_diffs = {"velocity": 0.0, "annual_real": 0.0, "annual_imag": 0.0, "annual_amplitude": 0.0, "annual_phase": 0.0}
    sampled_pixels = 0
    for wi, window in enumerate(windows[:25], start=1):
        a_stack = []
        b_stack = []
        for ei, (orig, corr) in enumerate(zip(original_cube.epochs["source_file"], corrected_cube.epochs["source_file"])):
            with rasterio.open(orig) as src:
                old = src.read(1, window=window, masked=True).filled(np.nan).astype("float32") * scale
            with rasterio.open(corr) as src:
                cor = src.read(1, window=window, masked=True).filled(np.nan).astype("float32") * scale
            a_stack.append(apply_reference_to_los(old, ei, ref).astype("float32"))
            b_stack.append(cor)
        a_stack = np.stack(a_stack)
        b_stack = np.stack(b_stack)
        row0, col0 = int(window.row_off), int(window.col_off)
        inc = np.asarray(incidence[row0:row0+64, col0:col0+64], dtype="float32")
        cos = np.cos(np.deg2rad(inc)); cos[np.abs(cos) < 1e-6] = np.nan
        coef_a = fit_coefficients_from_stack(a_stack / cos[None, :, :], original_cube.epochs["date"], config)
        coef_b = fit_coefficients_from_stack(b_stack / cos[None, :, :], corrected_cube.epochs["date"], config)
        products_a = {
            "velocity": coef_a[1],
            "annual_real": coef_a[2],
            "annual_imag": coef_a[3],
            "annual_amplitude": np.hypot(coef_a[2], coef_a[3]),
            "annual_phase": np.arctan2(coef_a[3], coef_a[2]),
        }
        products_b = {
            "velocity": coef_b[1],
            "annual_real": coef_b[2],
            "annual_imag": coef_b[3],
            "annual_amplitude": np.hypot(coef_b[2], coef_b[3]),
            "annual_phase": np.arctan2(coef_b[3], coef_b[2]),
        }
        sampled_pixels += 64 * 64
        row = {"tile_index": wi, "pixel_count": 64 * 64}
        for key in max_diffs:
            diff = np.abs(products_a[key] - products_b[key])
            val = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
            max_diffs[key] = max(max_diffs[key], val)
            row[f"{key}_max_abs_diff"] = val
        rows.append(row)
        print(f"explicit_vs_onthefly_tile {wi}/{len(windows[:25])}", flush=True)
    passed = sampled_pixels >= 100000 and max(max_diffs.values()) <= 1e-4
    payload = {
        "status": "passed" if passed else "failed",
        "sampled_tile_count": len(rows),
        "sampled_pixel_count": sampled_pixels,
        "corrected_timeseries_dir": str(corrected_dir),
        "dynamic_path": "original -> subtract L01028 zeroed LOS -> vertical -> fit",
        "explicit_path": "geo_timeseries_gacos_filtered_L01028 -> no subtract -> vertical -> fit",
        "geo_velocity_note": "geo_velocity.tif is not mixed into formal response products; equivalence uses time series only.",
        "max_abs_diffs": max_diffs,
        "tile_rows": rows,
    }
    write_json(REF / "L01028_explicit_vs_onthefly_equivalence_audit.json", payload)
    return payload


def build_common_mask_and_reuse_audit():
    old_mask = REV / "comparison_common_mask.tif"
    old_blocks = REV / "spatial_validation_blocks.tif"
    old_rbf = REV / "selected_rbf_design.json"
    out_mask = REF / "comparison_common_mask_L01028.tif"
    with rasterio.open(PRODUCTS["annual_real"]) as real, rasterio.open(PRODUCTS["annual_imag"]) as imag, rasterio.open(ROOT / "outputs" / "geological_model_covariates.tif") as geo:
        profile = real.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
        old_count = old_intersection = old_union = added = removed = 0
        new_count = 0
        h = hashlib.sha256()
        with rasterio.open(old_mask) as old, rasterio.open(out_mask, "w", **profile) as dst:
            for _, window in real.block_windows(1):
                rr = real.read(1, window=window)
                ii = imag.read(1, window=window)
                gg = geo.read(window=window)
                common = np.isfinite(rr) & np.isfinite(ii) & np.isfinite(gg).all(axis=0)
                arr = common.astype("uint8")
                dst.write(arr, 1, window=window)
                old_arr = old.read(1, window=window).astype(bool)
                new_count += int(common.sum())
                old_count += int(old_arr.sum())
                old_intersection += int((common & old_arr).sum())
                old_union += int((common | old_arr).sum())
                added += int((common & ~old_arr).sum())
                removed += int((~common & old_arr).sum())
                h.update(np.ascontiguousarray(arr).view("uint8"))
    new_hash = sha256_file(out_mask)
    old_hash = sha256_file(old_mask)
    identical = new_hash == old_hash and added == 0 and removed == 0
    payload = {
        "status": "passed",
        "new_common_mask_path": str(out_mask),
        "new_common_mask_hash": new_hash,
        "new_common_mask_array_hash": h.hexdigest(),
        "old_common_mask_hash": old_hash,
        "old_pixel_count": old_count,
        "new_pixel_count": new_count,
        "intersection_pixel_count": old_intersection,
        "union_pixel_count": old_union,
        "added_pixel_count": added,
        "removed_pixel_count": removed,
        "masks_identical": bool(identical),
        "fold_map_reusable": bool(identical),
        "fold_map_hash": sha256_file(old_blocks) if identical and old_blocks.exists() else None,
        "R32_centers_reusable": bool(identical),
        "RBF_centers_hash": read_json(old_rbf).get("RBF_centers_hash") or read_json(REV / "formal_protocol_v2_frozen_manifest.json").get("RBF_centers_hash") if identical and old_rbf.exists() else None,
        "raw_basis_normalization_reusable": bool(identical),
        "RBF_normalization_hash": read_json(REV / "formal_protocol_v2_frozen_manifest.json").get("raw_basis_normalization_hash") if identical else None,
        "geology_normalization_reusable": True,
    }
    write_json(REF / "L01028_common_mask_equivalence_audit.json", payload)
    return payload


def write_fold0_confirmation():
    payload = {
        "status": "not_run_pending_user_approval",
        "formal_fold1_to_fold4_access_count": 0,
        "development_only_grid": {
            "lag_u_days": [0, 10, 20],
            "lambda_ske": [10, 30],
            "Stage_C_accepted_iterations": [30, 40],
        },
        "selection_if_no_clear_disadvantage": {
            "lag_u_days": 10,
            "lambda_ske": 30,
            "Stage_C_accepted_iterations": 40,
        },
        "retain_previous_configuration": None,
        "note": "Fold0 confirmation requires a new L01028 harmonic cache/training pipeline and is not executed by this audit script.",
    }
    write_json(REF / "L01028_fold0_minimal_confirmation.json", payload)
    return payload


def freeze_manifest(product_audit, mask_audit, fold0):
    allowed = product_audit["status"] == "passed" and mask_audit["status"] == "passed" and fold0["status"] == "passed"
    response = read_json(REF / "reference_adjusted_inversion_input_manifest.json")
    payload = {
        "manifest_status": "frozen" if allowed else "not_frozen_blocked_pending_fold0_confirmation",
        "formal_L01028_execution_allowed": bool(allowed),
        "reference_manifest_hash": response["reference_manifest_hash"],
        "reference_timeseries_hash": response["rereferenced_timeseries_hash"],
        "response_product_hashes": product_audit["product_hashes"],
        "new_common_mask_hash": mask_audit["new_common_mask_hash"],
        "fold_map_hash": mask_audit.get("fold_map_hash"),
        "RBF_centers_hash": mask_audit.get("RBF_centers_hash"),
        "RBF_normalization_hash": mask_audit.get("RBF_normalization_hash"),
        "fold0_confirmation": fold0,
        "double_reference_protection_version": "insar_processing_apply_reference_to_los_v2",
        "source_code_hash": sha256_text(sha256_file(ROOT / "insar_processing.py") + sha256_file(Path(__file__))),
        "phase4_restart_allowed": False,
        "phase5_restart_allowed": False,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path = REF / "formal_protocol_v2_L01028_frozen_manifest.json"
    path.write_text(text, encoding="utf-8")
    (REF / "formal_protocol_v2_L01028_frozen_manifest.sha256").write_text(sha256_text(text) + "\n", encoding="utf-8")
    update_status({
        "formal_L01028_execution_allowed": bool(allowed),
        "phase4_restart_allowed": False,
        "phase5_restart_allowed": False,
        "L01028_formal_manifest_status": payload["manifest_status"],
    })
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--auto-reference-timeseries")
    parser.add_argument("--original-dir", default="../../geo_timeseries_gacos_filtered")
    parser.add_argument("--corrected-dir", default="../../geo_timeseries_gacos_filtered_L01028")
    parser.add_argument("--allow-external-metadata-writes", action="store_true")
    parser.add_argument("--continue-if-source-missing", action="store_true")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    auto = find_auto_reference_source(args.auto_reference_timeseries)
    src_audit = reference_source_equivalence(auto)
    if not src_audit["continue_allowed"] and not args.continue_if_source_missing:
        print(json.dumps(src_audit, indent=2), flush=True)
        return 2
    write_directory_reference_metadata((ROOT / args.original_dir).resolve(), (ROOT / args.corrected_dir).resolve(), args.allow_external_metadata_writes)
    product = product_integrity_audit()
    equiv = explicit_vs_onthefly_audit(config, (ROOT / args.corrected_dir).resolve())
    mask = build_common_mask_and_reuse_audit()
    fold0 = write_fold0_confirmation()
    frozen = freeze_manifest(product, mask, fold0)
    print(json.dumps({
        "source_equivalence": src_audit["status"],
        "product_integrity": product["status"],
        "explicit_vs_onthefly": equiv["status"],
        "common_mask": mask["status"],
        "fold0_confirmation": fold0["status"],
        "formal_L01028_execution_allowed": frozen["formal_L01028_execution_allowed"],
    }, indent=2), flush=True)
    return 0 if frozen["formal_L01028_execution_allowed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
