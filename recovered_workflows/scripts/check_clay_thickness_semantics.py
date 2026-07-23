#!/usr/bin/env python3
"""Check whether clay-group rasters can be treated as cumulative thickness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from geology_preprocessing import stack_all_valid, write_geotiff
from io_utils import ROOT, load_config, resolve_config_path


DERIVED = {
    "clay_unconfined_m": "clay_group_1_m",
    "cumulative_confined_clay_thickness_m": "Hc_sum",
    "cumulative_total_clay_thickness_m": "Htotal_sum",
    "confined_clay_fraction": "Hc_fraction",
    "total_clay_fraction": "Htotal_fraction",
}


def _pixel_area_km2(src):
    if src.crs and src.crs.is_projected:
        return abs(src.transform.a * src.transform.e) / 1e6
    if src.crs and src.crs.to_epsg() == 4326:
        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        area, _ = geod.polygon_area_perimeter(
            [src.bounds.left, src.bounds.right, src.bounds.right, src.bounds.left],
            [src.bounds.bottom, src.bounds.bottom, src.bounds.top, src.bounds.top],
        )
        return abs(area) / (src.width * src.height) / 1e6
    return abs(src.transform.a * src.transform.e) / 1e6


def _band_map(src):
    return {src.descriptions[i]: src.read(i + 1).astype("float32") for i in range(src.count)}


def _violation_row(metric, valid, excess, pixel_area):
    violation = valid & np.isfinite(excess) & (excess > 0)
    values = excess[violation].astype(float)
    return {
        "metric": metric,
        "valid_pixel_count": int(valid.sum()),
        "violation_pixel_count": int(violation.sum()),
        "violation_fraction": float(violation.sum() / max(1, valid.sum())),
        "affected_area_km2": float(violation.sum() * pixel_area),
        "maximum_excess_m": float(np.max(values)) if values.size else 0.0,
        "p50_excess_m": float(np.percentile(values, 50)) if values.size else 0.0,
        "p95_excess_m": float(np.percentile(values, 95)) if values.size else 0.0,
        "p99_excess_m": float(np.percentile(values, 99)) if values.size else 0.0,
    }


def _boundary_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask & ~eroded


def _boundary_diagnostics(src, bands, Htotal, Q4, valid_htotal_q4, violation, output, pixel_area):
    q4_boundary = _boundary_mask(np.isfinite(Q4))
    clay_boundary = np.zeros(src.shape, dtype=bool)
    for i in range(1, 5):
        clay_boundary |= _boundary_mask(np.isfinite(bands[f"clay_group_{i}_m"]))
    any_boundary = q4_boundary | clay_boundary
    distance_to_q4 = ndimage.distance_transform_edt(~q4_boundary).astype("float32")
    distance_to_any = ndimage.distance_transform_edt(~any_boundary).astype("float32")
    violation_distance = np.full(src.shape, np.nan, "float32")
    violation_distance[violation] = distance_to_any[violation]
    labels, n_components = ndimage.label(violation, structure=np.ones((3, 3), dtype=bool))
    component_sizes = np.bincount(labels.ravel())[1:] if n_components else np.array([], dtype=int)
    largest_area = float(component_sizes.max() * pixel_area) if component_sizes.size else 0.0
    n = int(violation.sum())
    within_1 = float((violation & (distance_to_any <= 1)).sum() / n) if n else 0.0
    within_2 = float((violation & (distance_to_any <= 2)).sum() / n) if n else 0.0
    interior = float((violation & (distance_to_any > 2)).sum() / n) if n else 0.0
    origin = "rasterized_polygon_boundary_mismatch" if within_2 >= 0.90 else "mixed_boundary_and_interior_mismatch"
    write_geotiff(output / "clay_thickness_violation_boundary_distance.tif", src, [violation_distance], ["distance_to_any_boundary_pixels"])
    diagnostics = {
        "violation_pixel_count": n,
        "distance_to_q4_polygon_boundary_pixels": {
            "p50": float(np.nanpercentile(distance_to_q4[violation], 50)) if n else 0.0,
            "p95": float(np.nanpercentile(distance_to_q4[violation], 95)) if n else 0.0,
            "maximum": float(np.nanmax(distance_to_q4[violation])) if n else 0.0,
        },
        "distance_to_any_clay_polygon_boundary_pixels": {
            "p50": float(np.nanpercentile(distance_to_any[violation], 50)) if n else 0.0,
            "p95": float(np.nanpercentile(distance_to_any[violation], 95)) if n else 0.0,
            "maximum": float(np.nanmax(distance_to_any[violation])) if n else 0.0,
        },
        "within_1_pixel_boundary_fraction": within_1,
        "within_2_pixel_boundary_fraction": within_2,
        "interior_violation_fraction": interior,
        "connected_component_count": int(n_components),
        "largest_component_area_km2": largest_area,
        "violation_origin": origin,
        "note": "Boundary diagnostics are for Htotal_sum > Q4; formal confined model uses Hc_sum, which has zero violations.",
    }
    (output / "clay_thickness_violation_boundary_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return diagnostics


def run(config_path="config.yaml", output_root="outputs/aquifer_model_revision"):
    config = load_config(config_path)
    output = ROOT / output_root
    output.mkdir(parents=True, exist_ok=True)
    raw_stack = resolve_config_path(config, config["geology"]["raw_rasters"]["stack"])
    raw_dir = ROOT / config["geology"].get("raw_raster_dir", "data/geology_rasters")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if config["geology"].get("layer_value_semantics") not in {"cumulative_clay_thickness", "layer_bottom_depth", "unknown"}:
        raise ValueError("Invalid geology.layer_value_semantics")
    with rasterio.open(raw_stack) as src:
        bands = _band_map(src)
        pixel_area = _pixel_area_km2(src)
        required = [f"clay_group_{i}_m" for i in range(1, 5)] + ["quaternary_thickness_m"]
        missing = [name for name in required if name not in bands]
        if missing:
            raise ValueError(f"Missing raw bands: {missing}")
        H1 = bands["clay_group_1_m"]
        Hc = stack_all_valid([bands[f"clay_group_{i}_m"] for i in config["geology"]["aquifer_group_definition"]["confined_groups"]])
        Htotal = stack_all_valid([bands[f"clay_group_{i}_m"] for i in range(1, 5)])
        Q4 = bands["quaternary_thickness_m"]
        valid_h1_q4 = np.isfinite(H1) & np.isfinite(Q4)
        valid_hc_q4 = np.isfinite(Hc) & np.isfinite(Q4)
        valid_htotal_q4 = np.isfinite(Htotal) & np.isfinite(Q4)
        rows = []
        for name in [f"clay_group_{i}_m" for i in range(1, 5)]:
            arr = bands[name]
            valid = np.isfinite(arr)
            rows.append(_violation_row(f"{name}_negative", valid, -arr, pixel_area))
        rows.append(_violation_row("H1_gt_Q4", valid_h1_q4, H1 - Q4, pixel_area))
        rows.append(_violation_row("Hc_sum_gt_Q4", valid_hc_q4, Hc - Q4, pixel_area))
        rows.append(_violation_row("Htotal_sum_gt_Q4", valid_htotal_q4, Htotal - Q4, pixel_area))
        rows_df = pd.DataFrame(rows)
        violation_map = np.zeros(src.shape, "uint8")
        violation_map[valid_h1_q4 & ((H1 - Q4) > 0)] |= 1
        violation_map[valid_hc_q4 & ((Hc - Q4) > 0)] |= 2
        violation_map[valid_htotal_q4 & ((Htotal - Q4) > 0)] |= 4
        write_geotiff(output / "clay_thickness_violation_map.tif", src, [violation_map], ["violation_bitmask"], dtype="uint8", nodata=0)
        htotal_fraction = np.divide(Htotal, Q4, out=np.full_like(Htotal, np.nan), where=valid_htotal_q4 & (Q4 > 0))
        hc_fraction = np.divide(Hc, Q4, out=np.full_like(Hc, np.nan), where=valid_hc_q4 & (Q4 > 0))
        tolerance = 1.0e-6
        htotal_violation_fraction = float(rows_df.loc[rows_df.metric == "Htotal_sum_gt_Q4", "violation_fraction"].iloc[0])
        hc_violation_fraction = float(rows_df.loc[rows_df.metric == "Hc_sum_gt_Q4", "violation_fraction"].iloc[0])
        block = htotal_violation_fraction > 0.001 or hc_violation_fraction > 0.001
        minor_boundary_warning = (not block) and htotal_violation_fraction > 0
        semantics_status = "unresolved" if block else (
            "accepted_with_minor_boundary_violations" if minor_boundary_warning else "accepted_cumulative_clay_thickness"
        )
        derived_arrays = {
            "clay_unconfined_m": H1,
            "cumulative_confined_clay_thickness_m": Hc,
            "cumulative_total_clay_thickness_m": Htotal,
            "confined_clay_fraction": hc_fraction.astype("float32"),
            "total_clay_fraction": htotal_fraction.astype("float32"),
        }
        for name, arr in derived_arrays.items():
            write_geotiff(raw_dir / f"{name}.tif", src, [arr], [name])
        violation = valid_htotal_q4 & ((Htotal - Q4) > 0)
        boundary_diagnostics = _boundary_diagnostics(src, bands, Htotal, Q4, valid_htotal_q4, violation, output, pixel_area)
        rows_df.to_csv(output / "clay_thickness_semantics_check.csv", index=False)
        summary = {
            "status": "blocked" if block else "passed_with_minor_boundary_warning" if minor_boundary_warning else "passed",
            "layer_value_semantics_configured": config["geology"].get("layer_value_semantics"),
            "layer_value_semantics_status": semantics_status,
            "semantics_warning": "minor_total_clay_vs_q4_boundary_mismatch" if minor_boundary_warning else None,
            "aquifer_group_definition": config["geology"].get("aquifer_group_definition"),
            "formula": {
                "clay_unconfined_m": "clay_group_1_m",
                "cumulative_confined_clay_thickness_m": "clay_group_2_m + clay_group_3_m + clay_group_4_m",
                "cumulative_total_clay_thickness_m": "clay_group_1_m + clay_group_2_m + clay_group_3_m + clay_group_4_m",
                "confined_clay_fraction": "cumulative_confined_clay_thickness_m / quaternary_thickness_m",
                "total_clay_fraction": "cumulative_total_clay_thickness_m / quaternary_thickness_m",
            },
            "derived_outputs_written": [str(raw_dir / f"{name}.tif") for name in DERIVED],
            "checks": rows_df.to_dict(orient="records"),
            "boundary_diagnostics": boundary_diagnostics,
            "notes": [
                "Cumulative clay thickness is not maximum burial depth.",
                "No clipping to Q4 thickness was applied.",
                "All sums use strict finite propagation, not nansum.",
                "Formal confined models use cumulative_confined_clay_thickness_m; cumulative_total_clay_thickness_m is mapping/audit/supplementary only.",
            ],
        }
        (output / "clay_thickness_semantics_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(rows_df.to_string(index=False))
    if block:
        raise RuntimeError("Clay thickness semantics check failed; Phase 4 remains blocked")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    args = parser.parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    main()
