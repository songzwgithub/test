#!/usr/bin/env python3
"""Validate raw geology rasters and build formal model covariates."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio

from geology_preprocessing import (
    MODEL_BANDS,
    RAW_BANDS,
    categorical_dummy,
    check_raster_alignment,
    geology_output_root,
    raster_stats,
    sha256_file,
    standardize_continuous,
    write_geotiff,
    write_json,
)
from io_utils import ROOT, load_config, resolve_config_path


def run(config_path="config.yaml"):
    config = load_config(config_path)
    geo = config["geology"]
    output = ROOT / geology_output_root(config)
    reference = resolve_config_path(config, geo["preprocessing"]["reference_raster"])
    raw_stack = ROOT / geo["raw_rasters"]["stack"]
    model_path = output / "geological_model_covariates.tif"
    with rasterio.open(reference) as ref, rasterio.open(raw_stack) as src:
        check_raster_alignment(src, ref)
        raw = {src.descriptions[i]: src.read(i + 1).astype("float32") for i in range(src.count)}
        missing = [name for name in RAW_BANDS if name not in raw]
        if missing:
            raise ValueError(f"Missing raw raster bands: {missing}")
        zone = raw["extraction_layer_zone_code"]
        zone_values = set(np.unique(zone[np.isfinite(zone)].astype(int)).tolist())
        invalid_zone = sorted(zone_values - {1, 2})
        if invalid_zone:
            raise ValueError(f"Extraction zone has unexpected codes: {invalid_zone}")
        q4 = raw["quaternary_thickness_m"]
        if np.isfinite(q4).sum() == 0 or np.nanmin(q4) < 0:
            raise ValueError("Q4 thickness raster is invalid or all NoData")
        valid_mask = np.isfinite(raw["clay_total_m"]) & np.isfinite(raw["clay_confined_m"]) & np.isfinite(q4) & np.isfinite(zone)
        model_arrays = []
        metadata = {"status": "complete", "raw_stack": str(raw_stack), "raw_stack_sha256": sha256_file(raw_stack), "covariates": {}}
        for source, target in [
            ("clay_total_m", "clay_total_z"),
            ("clay_confined_m", "clay_confined_z"),
            ("quaternary_thickness_m", "quaternary_thickness_z"),
        ]:
            z, meta = standardize_continuous(source, raw[source], valid_mask)
            model_arrays.append(z)
            metadata["covariates"][target] = meta
        dummy, meta = categorical_dummy("extraction_layer_zone_code", zone, 2, valid_mask)
        model_arrays.append(dummy)
        metadata["covariates"]["extraction_layer_zone_2"] = meta
        try:
            write_geotiff(model_path, ref, model_arrays, list(MODEL_BANDS))
            model_write_status = "canonical_written"
        except PermissionError as exc:
            model_path = output / "geological_model_covariates_atomic.tif"
            write_geotiff(model_path, ref, model_arrays, list(MODEL_BANDS))
            model_write_status = f"canonical_locked_fallback_written: {exc}"
    rows = []
    with rasterio.open(model_path) as model:
        with rasterio.open(reference) as ref:
            check_raster_alignment(model, ref)
        for idx, name in enumerate(model.descriptions, 1):
            rows.append(raster_stats(model.read(idx), name, "standardized" if name.endswith("_z") else "dummy"))
    metadata["model_stack"] = str(model_path)
    metadata["model_write_status"] = model_write_status
    metadata["model_stack_sha256"] = sha256_file(model_path)
    metadata["bands"] = list(MODEL_BANDS)
    metadata["standardization_checks"] = "mean_z < 1e-3, 0.95 < std_z < 1.05, p99(abs(z)) < 10 over joint valid mask"
    write_json(output / "geological_covariate_metadata.json", metadata)
    write_json(
        output / "geology_anomaly_trace.json",
        {
            "old_anomaly_cause": [
                "formal inversion previously rasterized/interpreted shapefiles inside Phase 4",
                "string intervals and categorical fields were inferred too permissively",
                "derived clay sums allowed partial finite components and could hide NoData",
                "standardization did not enforce strict finite-mask distribution checks",
            ],
            "new_fix": "Phase 4 reads only pre-rasterized aligned GeoTIFF model covariates with explicit continuous and categorical definitions.",
        },
    )
    pd.DataFrame(rows).to_csv(output / "geology_model_raster_audit.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
