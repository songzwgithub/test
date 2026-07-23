#!/usr/bin/env python3
"""Diagnose real polygon overlaps before atomic geology raster construction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd

from geology_preprocessing import geology_output_root, repaired_layer_values, resolve_layer_specs
from io_utils import ROOT, load_config


LAYER_LABELS = {
    "clay_group_1": "L1",
    "clay_group_2": "L2",
    "clay_group_3": "L3",
    "clay_group_4": "L4",
    "quaternary_thickness": "Q4",
}


def _diagnose_layer(spec, out_dir):
    frame = gpd.read_file(spec.file)
    values, repairs = repaired_layer_values(frame, spec)
    try:
        projected_crs = frame.estimate_utm_crs() or "EPSG:3857"
    except Exception:
        projected_crs = "EPSG:3857"
    gdf = frame.to_crs(projected_crs).copy()
    gdf["midpoint"] = values
    gdf["source_area_m2"] = gdf.geometry.area
    rows = []
    intersections = []
    for i in range(len(gdf)):
        gi = gdf.geometry.iloc[i]
        if gi is None or gi.is_empty:
            continue
        for j in range(i + 1, len(gdf)):
            gj = gdf.geometry.iloc[j]
            if gj is None or gj.is_empty or not gi.intersects(gj):
                continue
            inter = gi.intersection(gj)
            area = float(inter.area) if not inter.is_empty else 0.0
            if area <= 0:
                continue
            same = bool(gdf.midpoint.iloc[i] == gdf.midpoint.iloc[j])
            geometry_equal = bool(gi.equals(gj))
            contains = bool(gi.contains(gj) or gj.contains(gi))
            frac_i = area / float(gdf.source_area_m2.iloc[i]) if gdf.source_area_m2.iloc[i] else 0.0
            frac_j = area / float(gdf.source_area_m2.iloc[j]) if gdf.source_area_m2.iloc[j] else 0.0
            row = {
                "feature_i": int(i),
                "feature_j": int(j),
                "interval_i": str(frame.iloc[i].get(spec.label_field, "")) if spec.label_field else str(values[i]),
                "interval_j": str(frame.iloc[j].get(spec.label_field, "")) if spec.label_field else str(values[j]),
                "midpoint_i": float(values[i]),
                "midpoint_j": float(values[j]),
                "intersection_area_m2": area,
                "intersection_fraction_i": frac_i,
                "intersection_fraction_j": frac_j,
                "same_interval": same,
                "geometry_equal": geometry_equal,
                "contains_relation": contains,
                "touches_only": False,
                "likely_duplicate": bool(geometry_equal or (same and frac_i > 0.98 and frac_j > 0.98)),
                "likely_topology_sliver": bool(area < 1000 and min(frac_i, frac_j) < 0.001),
            }
            rows.append(row)
            intersections.append({**row, "geometry": inter})
    label = LAYER_LABELS.get(spec.name, spec.name)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"{label}_overlap_pairs.csv", index=False)
    if label == "L4" and intersections:
        gpd.GeoDataFrame(intersections, geometry="geometry", crs=projected_crs).to_file(out_dir / "L4_overlap_map.gpkg", driver="GPKG")
    return {
        "layer": label,
        "pair_count": int(len(table)),
        "same_interval_count": int(table["same_interval"].sum()) if not table.empty else 0,
        "different_interval_count": int((~table["same_interval"]).sum()) if not table.empty else 0,
        "duplicate_geometry_count": int(table["geometry_equal"].sum()) if not table.empty else 0,
        "nested_count": int(table["contains_relation"].sum()) if not table.empty else 0,
        "overlap_area_km2_sum_pairs": float(table["intersection_area_m2"].sum() / 1e6) if not table.empty else 0.0,
        "repairs": repairs,
    }


def run(config_path="config.yaml"):
    config = load_config(config_path)
    out_dir = ROOT / geology_output_root(config) / "polygon_overlap_details"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in resolve_layer_specs(config, ROOT):
        if spec.name not in LAYER_LABELS:
            continue
        summaries.append(_diagnose_layer(spec, out_dir))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "polygon_overlap_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
