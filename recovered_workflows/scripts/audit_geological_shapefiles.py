#!/usr/bin/env python3
"""Audit configured geology shapefiles before raster preprocessing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd

from geology_preprocessing import (
    geology_output_root,
    layer_values,
    polygon_overlap_fraction,
    resolve_layer_specs,
    sha256_file,
    sidecar_files,
    write_json,
)
from io_utils import ROOT, load_config


def run(config_path="config.yaml"):
    config = load_config(config_path)
    output = ROOT / geology_output_root(config)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    samples = {}
    for spec in resolve_layer_specs(config, ROOT):
        frame = gpd.read_file(spec.file)
        if frame.crs is None:
            raise ValueError(f"{spec.name}: missing CRS")
        if frame.empty:
            raise ValueError(f"{spec.name}: empty shapefile")
        required = [field for field in (spec.lower_field, spec.upper_field, spec.interval_field, spec.code_field, spec.label_field) if field]
        missing = [field for field in required if field not in frame.columns]
        if missing:
            raise ValueError(f"{spec.name}: missing fields {missing}")
        parse_error = None
        try:
            values = layer_values(frame, spec)
            valid_values = values
        except Exception as exc:
            parse_error = str(exc)
            valid_values = pd.Series(dtype=float)
        overlap = polygon_overlap_fraction(frame)
        sidecars = sidecar_files(spec.file)
        rows.append(
            {
                "layer": spec.name,
                "file": str(spec.file),
                "crs": str(frame.crs),
                "feature_count": int(len(frame)),
                "geometry_types": ",".join(sorted(frame.geometry.geom_type.unique())),
                "required_fields": ",".join(required),
                "value_min": float(pd.Series(valid_values).min()) if len(valid_values) else float("nan"),
                "value_median": float(pd.Series(valid_values).median()) if len(valid_values) else float("nan"),
                "value_max": float(pd.Series(valid_values).max()) if len(valid_values) else float("nan"),
                "unique_values": int(pd.Series(valid_values).nunique()) if len(valid_values) else 0,
                "parse_status": "failed" if parse_error else "ok",
                "parse_error": parse_error,
                "overlap_fraction": float(overlap),
                "overlap_status": "fail_overlap_gt_0.001" if overlap > 0.001 else "ok",
                "sidecar_count": len(sidecars),
                "shp_sha256": sha256_file(spec.file),
            }
        )
        sample = frame[required].head(10).copy()
        if parse_error is None:
            sample["parsed_value"] = values[: min(10, len(values))]
        samples[spec.name] = sample.to_dict(orient="records")
    audit = pd.DataFrame(rows)
    audit_path = output / "geological_shapefile_audit.csv"
    audit.to_csv(audit_path, index=False)
    write_json(
        output / "geological_shapefile_audit.json",
        {
            "status": "invalid_input" if (audit["parse_status"] == "failed").any() else ("requires_topology_review" if (audit["overlap_fraction"] > 0.001).any() else "complete"),
            "overlap_policy": "overlap_fraction > 0.001 is flagged and not silently hidden; rasterization still uses all_touched=False after explicit audit",
            "layers": rows,
            "samples": samples,
        },
    )
    print(audit.to_string(index=False))
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
