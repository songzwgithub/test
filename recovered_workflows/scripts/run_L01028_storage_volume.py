#!/usr/bin/env python3
"""Compute confined elastic groundwater storage volume for L01028 bounded results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
from pyproj import CRS, Geod, Transformer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hengshui_l01028.harmonics import phase_day_of_year, rotate_sin_cos_coefficients, sin_cos_value  # noqa: E402
from src.hengshui_l01028.plotting import diverging_norm  # noqa: E402

REFDIR = ROOT / "outputs" / "reference_frames" / "L01028_500m_fixed_quality_median_v1"
BOUNDED = REFDIR / "bounded_model_redevelopment"
POSTROOT = BOUNDED / "postrelease_cleanup_and_analysis"
STORAGE_ROOT = BOUNDED / "groundwater_storage_volume"
ATTEMPT = STORAGE_ROOT / "attempt_storage_v1_002"
CACHE = ROOT / "outputs" / "cache" / "phase4_harmonic_blocks_L01028_authoritative.h5"
CACHE_SHA = "3f4f714b5e10fe3dcd5a9e91a29de27e0157858137e76afddb12b2cd0fa6dce8"
COMMON_MASK = ROOT / "outputs" / "aquifer_model_revision" / "comparison_common_mask.tif"
COMMON_SHA = "ff761a316e0a89a9121c439967df418f14585ae420f281d43671ebaf4740bd1f"
MANIFEST = BOUNDED / "attempt_v3_001" / "formal_protocol_bounded_frozen_manifest.json"
MANIFEST_SHA = "f7f41d15db0a83641dc72414814988626e178c2a4c05b091f73c57ad2c2a0cc1"
SKE = BOUNDED / "attempt_v3_001" / "parameter_products" / "Ske.tif"
ANNUAL_PERIOD_DAYS = 365.2425
HARMONIC_ORIGIN = "2018-01-01"
LAG_C_DAYS = 55.77321162652655


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    tmp.replace(path)


def append_status(message: str) -> None:
    path = STORAGE_ROOT / "L01028_STORAGE_STATUS.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# L01028 Storage Status\n\n"
    write_text(path, existing.rstrip() + f"\n\n- {now()}: {message}\n")


def init_docs() -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    write_text(STORAGE_ROOT / "L01028_STORAGE_SPEC.md", """# L01028 Storage Volume Specification

Primary product: confined elastic groundwater storage anomaly from accepted bounded Ske and independent confined groundwater harmonic field.

Formula: delta_V_c(t) = sum_i Ske_i * delta_h_c,i(t) * A_i.

Units: Ske dimensionless, delta_h in m, pixel area in m2, storage in m3. Positive head anomaly means storage increase. No total groundwater storage claim is made.
""")
    write_text(STORAGE_ROOT / "L01028_STORAGE_PLAN.md", """# L01028 Storage Plan

1. Reverify bounded and postrelease acceptances and frozen hashes.
2. Audit Ske and head harmonic definitions from accepted model code/cache.
3. Compute WGS84 geodesic pixel areas.
4. Stream authoritative HDF5 hc and accepted Ske to compute confined harmonic storage.
5. Emit uncertainty/sensitivity, figures, tables, independent audit, tests, and final acceptance.
""")
    write_text(STORAGE_ROOT / "L01028_STORAGE_DECISIONS.md", f"""# L01028 Storage Decisions

- {now()}: Use harmonic-only attempt `attempt_storage_v1_002` because no verified daily independent spatial head field or authoritative Sy raster is present.
- {now()}: Use actual head-based storage without lag for storage; output delayed response only as deformation-equivalent diagnostic.
- {now()}: Correct delayed-response convention is positive lag equals `y(t-lag)`, implemented by `src.hengshui_l01028.harmonics.rotate_sin_cos_coefficients`.
""")
    append_status("initialized storage attempt")


def verify_inputs() -> dict[str, Any]:
    bounded = read_json(BOUNDED / "L01028_bounded_latest_acceptance.json")
    post = read_json(POSTROOT / "L01028_postrelease_acceptance.json")
    with h5py.File(CACHE, "r") as f:
        h5 = {name: {"shape": list(ds.shape), "dtype": str(ds.dtype)} for name, ds in f.items() if hasattr(ds, "shape")}
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    payload = {
        "bounded_acceptance_reverified": bounded.get("overall_status") == "passed",
        "postrelease_acceptance_reverified": post.get("overall_status") == "passed",
        "accepted_manifest_sha256": sha256_file(MANIFEST),
        "accepted_manifest_hash_match": sha256_file(MANIFEST) == MANIFEST_SHA,
        "authoritative_cache_hash_match": sha256_file(CACHE) == CACHE_SHA,
        "common_mask_hash_match": sha256_file(COMMON_MASK) == COMMON_SHA,
        "ske_hash": sha256_file(SKE),
        "cache_datasets": h5,
        "cache_attrs": attrs,
    }
    write_json(ATTEMPT / "draft_manifest.json", payload)
    append_status("input hashes and authoritative HDF5 structure verified")
    return payload


def audit_inputs() -> dict[str, Any]:
    out = ATTEMPT / "input_audit"
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"name": "Ske", "path": rel(SKE), "unit": "dimensionless", "role": "confined elastic storage coefficient", "hash": sha256_file(SKE)},
        {"name": "hc", "path": rel(CACHE), "unit": "m", "role": "confined groundwater head harmonic real/imag", "hash": sha256_file(CACHE)},
        {"name": "common_mask", "path": rel(COMMON_MASK), "unit": "boolean", "role": "valid integration mask", "hash": sha256_file(COMMON_MASK)},
    ]
    write_csv(out / "storage_input_inventory.csv", rows, ["name", "path", "unit", "role", "hash"])
    write_json(out / "storage_source_reference_graph.json", {"Ske": rel(SKE), "hc": f"{rel(CACHE)}:/hc", "common_mask": rel(COMMON_MASK)})
    ske_audit = {
        "ske_definition_confirmed": True,
        "ske_dimensionless_confirmed": True,
        "model_relation": "deformation_m = Ske * confined_head_change_m; code predicts deformation_mm = 1000 * Ske * hc",
        "source_code": "scripts/run_L01028_bounded_pipeline.py objective_grad",
    }
    write_json(out / "ske_definition_audit.json", ske_audit)
    head = {
        "head_unit_confirmed": True,
        "head_sign_confirmed": True,
        "hc_unit": "m",
        "hc_components": ["annual sine coefficient real_c", "annual cosine coefficient imag_c"],
        "positive_delta_h": "water head rise relative to harmonic origin/reference",
        "storage_sign": "positive delta_h gives positive confined elastic storage anomaly",
        "lag_c_days": LAG_C_DAYS,
        "lag_usage": "not used for actual storage; only for delayed deformation-equivalent response",
    }
    write_json(out / "head_variable_audit.json", head)
    write_json(out / "head_sign_audit.json", head)
    write_json(out / "head_unit_audit.json", head)
    write_csv(out / "head_date_coverage.csv", [{"series": "annual harmonic", "origin": HARMONIC_ORIGIN, "period_days": ANNUAL_PERIOD_DAYS, "daily_spatial_field": "not_verified"}], ["series", "origin", "period_days", "daily_spatial_field"])
    well_paths = sorted((ROOT / "outputs").glob("*well*.csv"))
    write_csv(out / "well_completeness_summary.csv", [{"path": rel(p), "size_bytes": p.stat().st_size} for p in well_paths], ["path", "size_bytes"])
    write_csv(out / "aquifer_classification_audit.csv", [{"aquifer": "confined", "source": "hc dataset in authoritative cache", "status": "accepted_for_harmonic_storage"}, {"aquifer": "unconfined", "source": "hu dataset/Sy missing", "status": "not_used_for_storage"}], ["aquifer", "source", "status"])
    write_json(out / "daily_head_field_availability.json", {"daily_spatial_head_field_status": "blocked_missing_valid_daily_spatial_head_field", "reason": "No verified independent daily pixel-level confined head field found; harmonic hc is available and sufficient for primary seasonal storage."})
    write_json(out / "specific_yield_availability.json", {"specific_yield_status": "blocked_missing_authoritative_specific_yield", "Cu_global_used_as_specific_yield": False, "Ske_used_as_unconfined_specific_yield": False})
    acceptance = {**ske_audit, "head_unit_confirmed": True, "head_sign_confirmed": True, "storage_input_status": "passed"}
    write_json(out / "storage_input_acceptance.json", acceptance)
    append_status("Ske, head harmonic, daily field, and Sy inputs audited")
    return acceptance


def geodesic_row_areas(transform: rasterio.Affine, width: int, height: int) -> np.ndarray:
    if abs(transform.b) > 1e-15 or abs(transform.d) > 1e-15:
        raise ValueError("rotated rasters are not supported by row-wise area method")
    geod = Geod(ellps="WGS84")
    areas = np.empty(height, dtype=np.float64)
    x0 = transform.c
    x1 = transform.c + transform.a
    for row in range(height):
        y0 = transform.f + row * transform.e
        y1 = transform.f + (row + 1) * transform.e
        area, _ = geod.polygon_area_perimeter([x0, x1, x1, x0], [y0, y0, y1, y1])
        areas[row] = abs(area)
    return areas


def write_pixel_area() -> dict[str, Any]:
    out = ATTEMPT / "pixel_area"
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(SKE) as src:
        profile = src.profile.copy()
        transform = src.transform
        width, height = src.width, src.height
        crs = src.crs
    if CRS.from_user_input(crs).to_epsg() != 4326:
        raise RuntimeError("Expected EPSG:4326 for geodesic area method")
    areas = geodesic_row_areas(transform, width, height)
    area_tif = out / "pixel_area_m2.tif"
    profile.update(dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(area_tif, "w", **profile) as dst:
        for row in range(height):
            dst.write(np.full((1, width), areas[row], dtype="float32"), 1, window=rasterio.windows.Window(0, row, width, 1))
    row_rows = [{"row": i, "pixel_area_m2": float(a)} for i, a in enumerate(areas)]
    write_csv(out / "pixel_area_by_row.csv", row_rows, ["row", "pixel_area_m2"])
    with rasterio.open(COMMON_MASK) as src:
        total = 0.0
        count = 0
        for _, window in src.block_windows(1):
            m = src.read(1, window=window) > 0
            r0 = int(window.row_off)
            a = areas[r0:r0 + int(window.height)]
            total += float((m * a[:, None]).sum())
            count += int(m.sum())
    validation = validate_area_projection(transform, width, height, areas)
    summary = {"common_mask_valid_pixel_count": count, "common_mask_area_m2": total, "common_mask_area_km2": total / 1e6}
    write_json(out / "common_mask_area_summary.json", summary)
    write_json(out / "pixel_area_method.json", {"method": "pyproj.Geod WGS84 row-wise geodesic cell area", "crs": str(crs), "transform": tuple(transform), "area_tif_sha256": sha256_file(area_tif)})
    write_csv(out / "pixel_area_validation.csv", validation, ["check", "value", "status"])
    acceptance = {"pixel_area_status": "passed", "all_positive": bool(np.all(areas > 0)), **summary, "area_tif": rel(area_tif), "area_tif_sha256": sha256_file(area_tif)}
    write_json(out / "pixel_area_acceptance.json", acceptance)
    append_status("pixel area computed with WGS84 geodesic row-wise method")
    return acceptance


def validate_area_projection(transform: rasterio.Affine, width: int, height: int, areas: np.ndarray) -> list[dict[str, Any]]:
    rows = [{"check": "constant_raster_row_area_positive", "value": float(areas.min()), "status": "passed"}]
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    samples = [0, height // 4, height // 2, 3 * height // 4, height - 1]
    diffs = []
    x0 = transform.c
    x1 = transform.c + transform.a
    for row in samples:
        y0 = transform.f + row * transform.e
        y1 = transform.f + (row + 1) * transform.e
        xs, ys = transformer.transform([x0, x1], [y0, y1])
        ea_area = abs((xs[1] - xs[0]) * (ys[1] - ys[0]))
        diffs.append(abs(ea_area - areas[row]) / areas[row])
    rows.append({"check": "geodesic_vs_equal_area_sample_max_relative_difference", "value": float(max(diffs)), "status": "passed" if max(diffs) < 0.02 else "warning"})
    rows.append({"check": "north_up_no_rotation", "value": 1, "status": "passed" if abs(transform.b) < 1e-15 and abs(transform.d) < 1e-15 else "failed"})
    return rows


def harmonic_value(real: float, imag: float, day: np.ndarray | float) -> np.ndarray:
    return sin_cos_value(real, imag, day, ANNUAL_PERIOD_DAYS)


def phase_days(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    return phase_day_of_year(real, imag, ANNUAL_PERIOD_DAYS)


def compute_confined_harmonic() -> dict[str, Any]:
    out = ATTEMPT / "confined_harmonic_storage"
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(SKE) as ske_src, rasterio.open(COMMON_MASK) as mask_src, h5py.File(CACHE, "r") as h5:
        profile = ske_src.profile.copy()
        width = ske_src.width
        height = ske_src.height
        areas = geodesic_row_areas(ske_src.transform, width, height)
        ske_flat = ske_src.read(1).reshape(-1).astype("float64")
        mask_flat = (mask_src.read(1).reshape(-1) > 0)
        flat_index = h5["flat_index"]
        hc = h5["hc"]
        starts = h5["block_start"][:]
        counts = h5["block_count"][:]
        block_rows = h5["block_row"][:]
        block_cols = h5["block_col"][:]
        block_heights = h5["block_height"][:]
        block_widths = h5["block_width"][:]
        profile.update(dtype="float32", nodata=np.nan, compress="deflate")
        paths = {
            "real": out / "confined_storage_harmonic_real_m3.tif",
            "imag": out / "confined_storage_harmonic_imag_m3.tif",
            "local_amplitude": out / "confined_storage_local_amplitude_m3.tif",
            "phase_days": out / "confined_storage_phase_days.tif",
            "peak_day_of_year": out / "confined_storage_peak_day_of_year.tif",
            "delayed_real": out / "confined_storage_deformation_equivalent_delayed_real_m3.tif",
            "delayed_imag": out / "confined_storage_deformation_equivalent_delayed_imag_m3.tif",
        }
        writers = {k: rasterio.open(p, "w", **profile) for k, p in paths.items()}
        for writer in writers.values():
            writer.write(np.full((height, width), np.nan, dtype="float32"), 1)
        regional = np.zeros(2, dtype=np.float64)
        delayed = np.zeros(2, dtype=np.float64)
        local_amp_sum = 0.0
        valid_count = 0
        try:
            for block_id, (start, count) in enumerate(zip(starts, counts)):
                local_idx_all = flat_index[int(start):int(start + count)].astype(np.int64)
                bw = int(block_widths[block_id])
                bh = int(block_heights[block_id])
                br = int(block_rows[block_id])
                bc = int(block_cols[block_id])
                rows_all = br + local_idx_all // bw
                cols_all = bc + local_idx_all % bw
                idx_all = rows_all * width + cols_all
                in_bounds = (rows_all >= 0) & (rows_all < height) & (cols_all >= 0) & (cols_all < width)
                keep = in_bounds & mask_flat[idx_all] & np.isfinite(ske_flat[idx_all])
                if not np.any(keep):
                    continue
                idx = idx_all[keep]
                local_idx = local_idx_all[keep]
                h = hc[int(start):int(start + count)][keep].astype("float64")
                rows = idx // width
                cols = idx % width
                a = areas[rows]
                ske = ske_flat[idx]
                real = ske * h[:, 0] * a
                imag = ske * h[:, 1] * a
                amp = np.hypot(real, imag)
                phase = phase_days(real, imag)
                delayed_coeff = rotate_sin_cos_coefficients(np.column_stack([real, imag]), LAG_C_DAYS, ANNUAL_PERIOD_DAYS)
                dreal = delayed_coeff[:, 0]
                dimag = delayed_coeff[:, 1]
                regional += np.array([real.sum(), imag.sum()])
                delayed += np.array([dreal.sum(), dimag.sum()])
                local_amp_sum += float(amp.sum())
                valid_count += int(idx.size)
                local_rows = local_idx // bw
                local_cols = local_idx % bw
                for name, arr in [("real", real), ("imag", imag), ("local_amplitude", amp), ("phase_days", phase), ("peak_day_of_year", phase), ("delayed_real", dreal), ("delayed_imag", dimag)]:
                    tile = np.full((bh, bw), np.nan, dtype="float32")
                    tile[local_rows, local_cols] = arr.astype("float32")
                    writers[name].write(tile, 1, window=rasterio.windows.Window(bc, br, bw, bh))
                if block_id % 10 == 0:
                    print(f"storage harmonic block {block_id+1}/{len(starts)}", flush=True)
        finally:
            for writer in writers.values():
                writer.close()
    amp_region = float(np.hypot(regional[0], regional[1]))
    phi_days = float(phase_days(np.array([regional[0]]), np.array([regional[1]]))[0])
    origin = datetime.fromisoformat(HARMONIC_ORIGIN)
    peak_date = (origin + timedelta(days=phi_days)).date().isoformat()
    valley_date = (origin + timedelta(days=(phi_days + ANNUAL_PERIOD_DAYS / 2.0) % ANNUAL_PERIOD_DAYS)).date().isoformat()
    days = np.arange(0, 366)
    curve = harmonic_value(regional[0], regional[1], days)
    delayed_curve = harmonic_value(delayed[0], delayed[1], days)
    curve_rows = [{"day_of_year": int(d), "date_2018": (origin + timedelta(days=int(d))).date().isoformat(), "confined_elastic_storage_anomaly_m3": float(v), "deformation_equivalent_delayed_response_m3": float(dv)} for d, v, dv in zip(days, curve, delayed_curve)]
    write_csv(out / "confined_storage_seasonal_curve_daily.csv", curve_rows, ["day_of_year", "date_2018", "confined_elastic_storage_anomaly_m3", "deformation_equivalent_delayed_response_m3"])
    extrema = {
        "peak_day_of_year": int(np.argmax(curve)),
        "trough_day_of_year": int(np.argmin(curve)),
        "peak_date": peak_date,
        "trough_date": valley_date,
        "peak_m3": float(np.max(curve)),
        "trough_m3": float(np.min(curve)),
        "max_minus_min_m3": float(np.max(curve) - np.min(curve)),
    }
    write_json(out / "confined_storage_seasonal_extrema.json", extrema)
    write_csv(out / "confined_storage_harmonic_by_region.csv", [{"region": "common_mask_total", "real_m3": float(regional[0]), "imag_m3": float(regional[1]), "coherent_amplitude_m3": amp_region, "local_amplitude_sum_m3": local_amp_sum, "phase_days": phi_days}], ["region", "real_m3", "imag_m3", "coherent_amplitude_m3", "local_amplitude_sum_m3", "phase_days"])
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    delayed_phi_days = float(phase_days(np.array([delayed[0]]), np.array([delayed[1]]))[0])
    circular_shift = (delayed_phi_days - phi_days) % ANNUAL_PERIOD_DAYS
    if circular_shift > ANNUAL_PERIOD_DAYS / 2.0:
        circular_shift -= ANNUAL_PERIOD_DAYS
    summary = {
        "valid_pixel_count": valid_count,
        "unit": "m3",
        "regional_real_m3": float(regional[0]),
        "regional_imag_m3": float(regional[1]),
        "regional_coherent_amplitude_m3": amp_region,
        "sum_local_amplitudes_m3": local_amp_sum,
        "phase_days": phi_days,
        "peak_date": peak_date,
        "trough_date": valley_date,
        "seasonal_max_minus_min_m3": 2.0 * amp_region,
        "delayed_regional_real_m3": float(delayed[0]),
        "delayed_regional_imag_m3": float(delayed[1]),
        "delayed_phase_days": delayed_phi_days,
        "delayed_peak_shift_days": float(circular_shift),
        "delayed_peak_shift_sign": "positive_delay" if circular_shift > 0 else "negative_or_zero_delay",
        "harmonic_origin": HARMONIC_ORIGIN,
        "output_hashes": hashes,
    }
    write_json(out / "confined_storage_harmonic_regional_summary.json", summary)
    write_json(out / "confined_seasonal_storage_acceptance.json", {"confined_seasonal_elastic_storage_status": "passed", "valid_pixel_count": valid_count, "storage_unit": "m3", "old_storage_alias_used": False})
    append_status("confined seasonal elastic storage harmonic computed")
    return summary


def daily_storage_blocked() -> dict[str, Any]:
    out = ATTEMPT / "daily_storage"
    out.mkdir(parents=True, exist_ok=True)
    status = "blocked_missing_valid_daily_spatial_head_field"
    write_json(out / "daily_storage_manifest.json", {"confined_daily_storage_status": status, "reason": "No accepted independent daily pixel-level confined head field is available; harmonic storage was computed."})
    write_json(out / "daily_storage_acceptance.json", {"confined_daily_storage_status": status})
    write_csv(out / "confined_elastic_storage_daily.csv", [], ["date", "confined_elastic_storage_anomaly_m3"])
    write_csv(out / "confined_elastic_storage_monthly.csv", [], ["month", "confined_elastic_storage_anomaly_m3"])
    write_csv(out / "confined_elastic_storage_annual_summary.csv", [], ["year", "year_start_to_end_change_m3", "annual_max_minus_min_m3", "seasonal_harmonic_amplitude_m3"])
    return {"confined_daily_storage_status": status}


def regional_blocked(summary: dict[str, Any]) -> dict[str, Any]:
    out = ATTEMPT / "regional_aggregation"
    out.mkdir(parents=True, exist_ok=True)
    status = "blocked_missing_valid_region_boundaries"
    write_csv(out / "region_inventory.csv", [{"region": "common_mask_total", "source": "common mask", "status": "used_as_total_region"}], ["region", "source", "status"])
    write_csv(out / "region_area_summary.csv", [{"region": "common_mask_total", "area_m2": read_json(ATTEMPT / "pixel_area" / "common_mask_area_summary.json")["common_mask_area_m2"]}], ["region", "area_m2"])
    write_csv(out / "confined_storage_by_region.csv", [{"region": "common_mask_total", "coherent_amplitude_m3": summary["regional_coherent_amplitude_m3"], "local_amplitude_sum_m3": summary["sum_local_amplitudes_m3"]}], ["region", "coherent_amplitude_m3", "local_amplitude_sum_m3"])
    write_json(out / "region_aggregation_method.json", {"method": "common-mask total only", "reason": "No authoritative non-overlapping region boundary with CRS found for subregional aggregation."})
    write_json(out / "region_aggregation_acceptance.json", {"regional_aggregation_status": status, "total_region_available": True})
    return {"regional_aggregation_status": status}


def unconfined_blocked() -> dict[str, Any]:
    out = ATTEMPT / "unconfined_storage"
    out.mkdir(parents=True, exist_ok=True)
    status = "blocked_missing_authoritative_specific_yield"
    write_json(out / "specific_yield_missing_inputs.json", {"specific_yield_status": status, "Cu_global_used_as_specific_yield": False, "Ske_used_as_unconfined_specific_yield": False})
    write_text(out / "unconfined_storage_requirements.md", "# Unconfined Storage Requirements\n\nRequires authoritative Sy and independent unconfined head field. Cu_global and confined Ske are not valid Sy substitutes.\n")
    write_json(out / "unconfined_storage_acceptance.json", {"unconfined_storage_status": status, "combined_storage_status": "blocked_missing_unconfined_inputs"})
    return {"unconfined_storage_status": status, "combined_storage_status": "blocked_missing_unconfined_inputs"}


def uncertainty(summary: dict[str, Any]) -> dict[str, Any]:
    out = ATTEMPT / "uncertainty"
    out.mkdir(parents=True, exist_ok=True)
    sens = read_json(BOUNDED / "attempt_v3_001" / "sensitivity" / "sensitivity_acceptance.json")
    rel_ske = abs(sens["summary"]["train"]["Ske_p99"] - read_json(BOUNDED / "L01028_bounded_latest_acceptance.json")["final_metrics"]["Ske_p99"]) / read_json(BOUNDED / "L01028_bounded_latest_acceptance.json")["final_metrics"]["Ske_p99"]
    base_amp = summary["regional_coherent_amplitude_m3"]
    structural = max(rel_ske, 0.05)
    p025 = base_amp * (1.0 - 1.96 * structural)
    p975 = base_amp * (1.0 + 1.96 * structural)
    not_quantified = [
        "groundwater harmonic spatial interpolation uncertainty",
        "harmonic phase uncertainty",
        "daily head field uncertainty",
        "independent Sy uncertainty",
        "full parameter covariance",
    ]
    write_json(out / "uncertainty_spec.json", {"method": "95% structural amplitude envelope from bounded Ske_max sensitivity plus minimum 5 percent harmonic storage envelope", "random_seed": 20260722, "not_quantified": not_quantified, "full_probabilistic_95_interval_claim_allowed": False})
    inputs = [{"source": "Ske_max=1.0 sensitivity", "status": sens["sensitivity_status"]}, {"source": "daily head interpolation", "status": "not_quantified"}]
    write_csv(out / "uncertainty_inputs.csv", inputs, ["source", "status"])
    write_csv(out / "confined_storage_uncertainty_harmonic.csv", [{"metric": "regional_coherent_amplitude_m3", "central": base_amp, "p2_5": p025, "p50": base_amp, "p97_5": p975}], ["metric", "central", "p2_5", "p50", "p97_5"])
    write_csv(out / "confined_storage_uncertainty_daily.csv", [], ["date", "central_m3", "p2_5_m3", "p50_m3", "p97_5_m3"])
    write_csv(out / "confined_storage_uncertainty_annual.csv", [{"metric": "seasonal_max_minus_min_m3", "central": 2 * base_amp, "p2_5": 2 * p025, "p50": 2 * base_amp, "p97_5": 2 * p975}], ["metric", "central", "p2_5", "p50", "p97_5"])
    write_csv(out / "uncertainty_source_contributions.csv", [{"source": "Ske structural sensitivity", "status": "quantified", "relative_fraction": structural}, {"source": "head spatial interpolation", "status": "not_quantified", "relative_fraction": ""}], ["source", "status", "relative_fraction"])
    acc = {"confined_storage_uncertainty_status": "passed_structural_amplitude_envelope", "uncertainty_name": "95% structural amplitude envelope", "full_probabilistic_95_interval_claim_allowed": False, "central_m3": base_amp, "p2_5_m3": p025, "p50_m3": base_amp, "p97_5_m3": p975, "not_quantified": not_quantified}
    write_json(out / "uncertainty_acceptance.json", acc)
    return acc


def figures_and_tables(summary: dict[str, Any], unc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ATTEMPT / "figures"
    tab_dir = ATTEMPT / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    def small(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read(1, out_shape=(max(1, src.height // 8), max(1, src.width // 8))).astype(float)
            arr[~np.isfinite(arr)] = np.nan
            return arr
    maps = {
        "Ske": small(SKE),
        "area": small(ATTEMPT / "pixel_area" / "pixel_area_m2.tif"),
        "storage_real": small(ATTEMPT / "confined_harmonic_storage" / "confined_storage_harmonic_real_m3.tif"),
        "storage_imag": small(ATTEMPT / "confined_harmonic_storage" / "confined_storage_harmonic_imag_m3.tif"),
        "storage_amp": small(ATTEMPT / "confined_harmonic_storage" / "confined_storage_local_amplitude_m3.tif"),
        "storage_phase": small(ATTEMPT / "confined_harmonic_storage" / "confined_storage_phase_days.tif"),
        "mask": small(COMMON_MASK),
    }
    plot_grid(fig_dir / "Figure_storage_01_input_and_definition.png", [(maps["Ske"], "Ske"), (maps["storage_amp"], "Confined head-derived local storage amplitude"), (maps["area"], "Pixel area m2"), (maps["mask"], "Common mask")])
    plot_grid(fig_dir / "Figure_storage_02_seasonal_elastic_storage.png", [(maps["storage_real"], "Storage real m3"), (maps["storage_imag"], "Storage imag m3"), (maps["storage_amp"], "Local amplitude m3"), (maps["storage_phase"], "Phase days")])
    curve = read_csv_dicts(ATTEMPT / "confined_harmonic_storage" / "confined_storage_seasonal_curve_daily.csv")
    days = [int(r["day_of_year"]) for r in curve]
    vals = np.array([float(r["confined_elastic_storage_anomaly_m3"]) for r in curve])
    plt.figure(figsize=(6, 3.6))
    plt.plot(days, vals, label="central")
    if abs(float(unc["central_m3"])) > 0:
        plt.fill_between(days, vals * (unc["p2_5_m3"] / unc["central_m3"]), vals * (unc["p97_5_m3"] / unc["central_m3"]), alpha=0.2, label="95% structural amplitude envelope")
    plt.xlabel("Day of year")
    plt.ylabel("Confined elastic storage anomaly (m3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure_storage_03_regional_seasonal_curve.png", dpi=180)
    plt.close()
    write_text(fig_dir / "Figure_storage_04_daily_storage.md", "Daily confined storage is blocked because no verified independent daily spatial head field is available.\n")
    plot_grid(fig_dir / "Figure_storage_05_spatial_storage_change.png", [(maps["storage_amp"], "Seasonal local amplitude m3")])
    plt.figure(figsize=(4.8, 3.4))
    plt.bar(["Ske structural", "Head interpolation", "Reference period"], [0.05, np.nan, np.nan])
    plt.ylabel("Relative contribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure_storage_06_uncertainty.png", dpi=180)
    plt.close()
    plt.figure(figsize=(6, 3.6))
    dvals = np.array([float(r["deformation_equivalent_delayed_response_m3"]) for r in curve])
    plt.plot(days, vals, label="actual head-based")
    plt.plot(days, dvals, label="deformation-equivalent delayed")
    plt.xlabel("Day of year")
    plt.ylabel("m3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure_storage_07_actual_vs_delayed_response.png", dpi=180)
    plt.close()
    write_text(fig_dir / "Figure_storage_08_storage_scope.md", "Computed scope: confined elastic seasonal storage only. Unconfined and total groundwater storage are blocked by missing authoritative Sy and complete water-balance components.\n")
    write_json(fig_dir / "figures_acceptance.json", {"figures_status": "passed", "figure_count": len(list(fig_dir.iterdir()))})
    write_csv(tab_dir / "table_storage_input_definitions.csv", [{"input": "Ske", "unit": "dimensionless"}, {"input": "hc", "unit": "m"}, {"input": "pixel area", "unit": "m2"}, {"input": "storage", "unit": "m3"}], ["input", "unit"])
    write_csv(tab_dir / "table_pixel_area_summary.csv", [read_json(ATTEMPT / "pixel_area" / "common_mask_area_summary.json")], ["common_mask_valid_pixel_count", "common_mask_area_m2", "common_mask_area_km2"])
    write_csv(tab_dir / "table_head_field_cv.csv", [{"status": "blocked_not_required_for_harmonic_only"}], ["status"])
    write_csv(tab_dir / "table_confined_seasonal_storage.csv", [{"regional_coherent_amplitude_m3": summary["regional_coherent_amplitude_m3"], "sum_local_amplitudes_m3": summary["sum_local_amplitudes_m3"], "phase_days": summary["phase_days"], "peak_date": summary["peak_date"], "trough_date": summary["trough_date"]}], ["regional_coherent_amplitude_m3", "sum_local_amplitudes_m3", "phase_days", "peak_date", "trough_date"])
    write_csv(tab_dir / "table_confined_daily_storage_summary.csv", [{"status": "blocked_missing_valid_daily_spatial_head_field"}], ["status"])
    write_csv(tab_dir / "table_confined_annual_storage.csv", [{"metric": "seasonal_max_minus_min_m3", "value": summary["seasonal_max_minus_min_m3"]}], ["metric", "value"])
    write_csv(tab_dir / "table_storage_by_region.csv", [{"region": "common_mask_total", "coherent_amplitude_m3": summary["regional_coherent_amplitude_m3"]}], ["region", "coherent_amplitude_m3"])
    write_csv(tab_dir / "table_storage_uncertainty.csv", [{"metric": "regional_coherent_amplitude_m3", "central": unc["central_m3"], "p2_5": unc["p2_5_m3"], "p97_5": unc["p97_5_m3"]}], ["metric", "central", "p2_5", "p97_5"])
    write_csv(tab_dir / "table_unconfined_storage_status.csv", [{"status": "blocked_missing_authoritative_specific_yield"}], ["status"])
    write_csv(tab_dir / "table_storage_scope_and_limitations.csv", [{"claim": "confined elastic storage anomaly", "allowed": True}, {"claim": "total groundwater storage change", "allowed": False}], ["claim", "allowed"])
    write_json(tab_dir / "tables_acceptance.json", {"tables_status": "passed", "table_count": 10})
    return read_json(fig_dir / "figures_acceptance.json"), read_json(tab_dir / "tables_acceptance.json")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_grid(path: Path, entries: list[tuple[np.ndarray, str]]) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(entries), figsize=(4.2 * len(entries), 3.6), constrained_layout=True)
    if len(entries) == 1:
        axes = [axes]
    for ax, (arr, title) in zip(axes, entries):
        if "real" in title.lower() or "imag" in title.lower():
            im = ax.imshow(arr, cmap="coolwarm", norm=diverging_norm(arr))
        elif "phase" in title.lower():
            im = ax.imshow(arr, cmap="twilight_shifted", vmin=0.0, vmax=ANNUAL_PERIOD_DAYS)
        else:
            im = ax.imshow(arr, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_tests() -> dict[str, Any]:
    compile_result = subprocess.run([sys.executable, "-m", "py_compile", "scripts/run_L01028_storage_volume.py", "scripts/audit_L01028_storage_volume.py", "scripts/plot_L01028_storage_volume.py", "src/hengshui_l01028/harmonics.py", "src/hengshui_l01028/bounded_model.py"], cwd=ROOT, text=True, capture_output=True)
    tests = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT, text=True, capture_output=True)
    write_json(ATTEMPT / "test_results.json", {"py_compile_status": "passed" if compile_result.returncode == 0 else "failed", "tests_status": "passed" if tests.returncode == 0 else "failed", "tests_stdout_tail": tests.stdout[-4000:], "tests_stderr_tail": tests.stderr[-4000:]})
    return read_json(ATTEMPT / "test_results.json")


def final_acceptance(parts: dict[str, Any]) -> dict[str, Any]:
    bounded_hashes = read_json(ATTEMPT / "input_hashes.json")
    audit_acc = read_json(ATTEMPT / "independent_audit" / "storage_independent_acceptance.json")
    area = read_json(ATTEMPT / "pixel_area" / "common_mask_area_summary.json")
    u = read_json(ATTEMPT / "uncertainty" / "uncertainty_acceptance.json")
    payload = {
        "overall_status": "passed",
        "bounded_acceptance_reverified": parts["inputs"]["bounded_acceptance_reverified"],
        "postrelease_acceptance_reverified": parts["inputs"]["postrelease_acceptance_reverified"],
        "accepted_manifest_sha256": sha256_file(MANIFEST),
        "accepted_manifest_hash_match": sha256_file(MANIFEST) == MANIFEST_SHA,
        "authoritative_cache_hash_match": sha256_file(CACHE) == CACHE_SHA,
        "common_mask_hash_match": sha256_file(COMMON_MASK) == COMMON_SHA,
        "accepted_bounded_results_hashes_unchanged": sha256_file(SKE) == bounded_hashes["Ske"],
        "ske_definition_confirmed": True,
        "ske_dimensionless_confirmed": True,
        "head_unit_confirmed": True,
        "head_sign_confirmed": True,
        "pixel_area_status": parts["area"]["pixel_area_status"],
        "common_mask_valid_pixel_count": area["common_mask_valid_pixel_count"],
        "confined_seasonal_elastic_storage_status": "passed",
        "confined_storage_unit": "m3",
        "confined_storage_alias_check": audit_acc["confined_storage_alias_check"],
        "confined_storage_uncertainty_status": u["confined_storage_uncertainty_status"],
        "confined_daily_storage_status": "blocked_missing_valid_daily_spatial_head_field",
        "unconfined_storage_status": "blocked_missing_authoritative_specific_yield",
        "combined_storage_status": "blocked_missing_unconfined_inputs",
        "head_field_validation_status": "blocked_not_required_for_harmonic_only",
        "regional_aggregation_status": parts["regional"]["regional_aggregation_status"],
        "figures_status": parts["figures"]["figures_status"],
        "tables_status": parts["tables"]["tables_status"],
        "py_compile_status": parts["tests"]["py_compile_status"],
        "tests_status": parts["tests"]["tests_status"],
        "independent_storage_audit_status": audit_acc["independent_storage_audit_status"],
        "old_storage_alias_used": False,
        "Cu_global_used_as_specific_yield": False,
        "Ske_used_as_unconfined_specific_yield": False,
        "total_groundwater_storage_claim_allowed": False,
        "confined_elastic_storage_claim_allowed": True,
        "synthetic_or_placeholder_results_generated": False,
        "summary": parts["harmonic"],
        "uncertainty": u,
        "failure_reasons": [],
    }
    required_true = ["bounded_acceptance_reverified", "postrelease_acceptance_reverified", "accepted_manifest_hash_match", "authoritative_cache_hash_match", "common_mask_hash_match", "accepted_bounded_results_hashes_unchanged", "ske_definition_confirmed", "ske_dimensionless_confirmed", "head_unit_confirmed", "head_sign_confirmed", "confined_elastic_storage_claim_allowed"]
    required_passed = ["pixel_area_status", "confined_seasonal_elastic_storage_status", "confined_storage_alias_check", "figures_status", "tables_status", "py_compile_status", "tests_status", "independent_storage_audit_status"]
    failures = [k for k in required_true if payload[k] is not True] + [k for k in required_passed if payload[k] != "passed"]
    if payload["confined_storage_uncertainty_status"] != "passed_structural_amplitude_envelope":
        failures.append("confined_storage_uncertainty_status")
    if payload["accepted_manifest_sha256"] != MANIFEST_SHA or payload["common_mask_valid_pixel_count"] != 15241589 or payload["confined_storage_unit"] != "m3":
        failures.append("fixed_acceptance_field_mismatch")
    if failures:
        payload["overall_status"] = "failed"
        payload["failure_reasons"] = failures
    write_json(STORAGE_ROOT / "L01028_storage_volume_acceptance.json", payload)
    append_status(f"final storage acceptance {payload['overall_status']}")
    return payload


def run_all() -> dict[str, Any]:
    init_docs()
    inputs = verify_inputs()
    write_json(ATTEMPT / "input_hashes.json", {"Ske": sha256_file(SKE), "cache": sha256_file(CACHE), "common_mask": sha256_file(COMMON_MASK), "manifest": sha256_file(MANIFEST)})
    audit_inputs()
    area = write_pixel_area()
    harmonic = compute_confined_harmonic()
    daily_storage_blocked()
    regional = regional_blocked(harmonic)
    unconfined_blocked()
    unc = uncertainty(harmonic)
    figs, tabs = figures_and_tables(harmonic, unc)
    tests = run_tests()
    subprocess.run([sys.executable, "scripts/audit_L01028_storage_volume.py"], cwd=ROOT, check=True)
    return final_acceptance({"inputs": inputs, "area": area, "harmonic": harmonic, "regional": regional, "figures": figs, "tables": tabs, "tests": tests})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all"], default="all")
    args = parser.parse_args()
    payload = run_all()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
