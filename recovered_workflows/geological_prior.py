"""Formal geology covariate reader for pre-rasterized audited GeoTIFF inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from geology_preprocessing import (
    MODEL_BANDS,
    RAW_BANDS,
    categorical_dummy,
    check_raster_alignment,
    raster_stats,
    sha256_file,
    standardize_continuous,
    write_geotiff,
    write_json,
)


def inspect_vector(path, field):
    """Compatibility audit helper; formal inversion does not call this."""
    import geopandas as gpd

    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"Vector CRS missing: {path}")
    if frame.empty:
        raise ValueError(f"Empty vector: {path}")
    if field not in frame:
        raise ValueError(f"Required attribute {field} missing: {path}")
    types = set(frame.geometry.geom_type)
    if not all("Polygon" in x or "LineString" in x for x in types):
        raise ValueError(f"Unsupported geometry {types}")
    return types, frame.crs


def build_geological_products(config, reference_raster, model_output, quality_output, stats_output):
    """Build formal model covariates from audited raw rasters.

    The formal inversion path deliberately does not read or interpret shapefiles.
    It only consumes a pre-rasterized raw physical-unit GeoTIFF stack whose grid
    exactly matches the InSAR reference raster.
    """
    import pandas as pd
    import rasterio

    if config.get("input_mode") != "pre_rasterized":
        raise ValueError("Formal geology input_mode must be 'pre_rasterized'; direct SHP input is disabled")
    raw_stack = Path(config["raw_rasters"]["stack"])
    if not raw_stack.exists():
        raise FileNotFoundError(f"Pre-rasterized raw geology stack is missing: {raw_stack}")
    model_output = Path(model_output)
    quality_output = Path(quality_output)
    stats_output = Path(stats_output)
    metadata_path = model_output.parent / "geological_covariate_metadata.json"
    audit_path = model_output.parent / "geology_model_raster_audit.csv"
    with rasterio.open(reference_raster) as ref, rasterio.open(raw_stack) as raw_src:
        check_raster_alignment(raw_src, ref)
        raw = {raw_src.descriptions[i]: raw_src.read(i + 1).astype("float32") for i in range(raw_src.count)}
        missing = [name for name in RAW_BANDS if name not in raw]
        if missing:
            raise ValueError(f"Raw geology stack is missing required bands: {missing}")
        zone = raw["extraction_layer_zone_code"]
        valid_mask = (
            np.isfinite(raw["clay_total_m"])
            & np.isfinite(raw["clay_confined_m"])
            & np.isfinite(raw["quaternary_thickness_m"])
            & np.isfinite(zone)
        )
        model_arrays = []
        metadata = {
            "status": "complete",
            "formal_input_mode": "pre_rasterized",
            "raw_stack": str(raw_stack),
            "raw_stack_sha256": sha256_file(raw_stack),
            "covariates": {},
            "model_bands": list(MODEL_BANDS),
        }
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
        write_geotiff(model_output, ref, model_arrays, list(MODEL_BANDS))
        empty_quality = np.full((ref.height, ref.width), np.nan, "float32")
        write_geotiff(quality_output, ref, [empty_quality], ["no_formal_quality_layers"])
    rows = []
    import rasterio

    with rasterio.open(model_output) as src:
        for idx, name in enumerate(src.descriptions, 1):
            rows.append(raster_stats(src.read(idx), name, "standardized" if name.endswith("_z") else "dummy"))
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    metadata["model_stack"] = str(model_output)
    metadata["model_stack_sha256"] = sha256_file(model_output)
    write_json(metadata_path, metadata)
    write_json(stats_output, metadata)
    return list(MODEL_BANDS), ["no_formal_quality_layers"], metadata


rasterize_geology = build_geological_products
