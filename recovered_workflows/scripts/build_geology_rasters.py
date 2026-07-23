#!/usr/bin/env python3
"""Build raw physical-unit geology GeoTIFF rasters from audited shapefiles."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.ops import polygonize, unary_union

from geology_preprocessing import (
    RAW_BANDS,
    CONTINUOUS_BANDS,
    geology_output_root,
    repaired_layer_values,
    raster_stats,
    resolve_layer_specs,
    stack_all_valid,
    write_geotiff,
    write_json,
)
from io_utils import ROOT, load_config, resolve_config_path


def _make_valid(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.is_valid:
        return geom
    try:
        from shapely.validation import make_valid

        return make_valid(geom)
    except Exception:
        return geom.buffer(0)


def _rasterize(frame, values, ref):
    shapes = [(geom, float(value)) for geom, value in zip(frame.geometry, values) if geom is not None and not geom.is_empty]
    return rasterize(
        shapes,
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        fill=np.nan,
        dtype="float32",
        all_touched=False,
    )


def _build_atomic_interval_polygons(frame, values, spec, out_dir, conflict_threshold=0.001):
    try:
        projected_crs = frame.estimate_utm_crs() or "EPSG:3857"
    except Exception:
        projected_crs = "EPSG:3857"
    source = frame.to_crs(projected_crs).copy()
    source["geometry"] = source.geometry.map(_make_valid)
    source["midpoint"] = values.astype(float)
    source["source_area_m2"] = source.geometry.area
    dissolved = source.dissolve(by="midpoint", as_index=False)
    boundaries = [geom.boundary for geom in dissolved.geometry if geom is not None and not geom.is_empty]
    merged = unary_union(boundaries)
    atoms = list(polygonize(merged))
    assigned = []
    unresolved = []
    uncovered = []
    resolution_rows = []
    for atom_index, atom in enumerate(atoms):
        if atom.is_empty or atom.area <= 0:
            continue
        point = atom.representative_point()
        hits = source[source.geometry.covers(point)].copy()
        if hits.empty:
            uncovered.append({"layer": spec.name, "atom_index": atom_index, "area_m2": float(atom.area), "status": "uncovered_gap", "geometry": atom})
            continue
        mids = sorted(set(float(v) for v in hits["midpoint"]))
        if len(mids) == 1:
            assigned.append({"midpoint": mids[0], "status": "single_or_same_interval", "geometry": atom})
            continue
        contains_relation = False
        geoms = list(hits.geometry)
        for i, geom in enumerate(geoms):
            contains_relation = contains_relation or any(i != j and geom.contains(other) for j, other in enumerate(geoms))
        if contains_relation:
            selected = hits.sort_values("source_area_m2").iloc[0]
            assigned.append({"midpoint": float(selected.midpoint), "status": "most_specific_polygon", "geometry": atom})
            resolution_rows.append(
                {
                    "layer": spec.name,
                    "atom_index": atom_index,
                    "resolution_rule": "most_specific_polygon",
                    "candidate_midpoints": ",".join(str(x) for x in mids),
                    "selected_midpoint": float(selected.midpoint),
                    "atom_area_m2": float(atom.area),
                }
            )
        else:
            unresolved.append(
                {
                    "layer": spec.name,
                    "atom_index": atom_index,
                    "candidate_midpoints": ",".join(str(x) for x in mids),
                    "area_m2": float(atom.area),
                    "status": "unresolved_conflict",
                    "geometry": atom,
                }
            )
    union_area = float(source.geometry.unary_union.area) or 1.0
    conflict_area = float(sum(row["area_m2"] for row in unresolved))
    conflict_fraction = conflict_area / union_area
    out_dir.mkdir(parents=True, exist_ok=True)
    if unresolved:
        gpd.GeoDataFrame(unresolved, geometry="geometry", crs=projected_crs).to_file(out_dir / "geology_unresolved_conflicts.gpkg", layer=spec.name, driver="GPKG")
    if uncovered:
        gpd.GeoDataFrame(uncovered, geometry="geometry", crs=projected_crs).to_file(out_dir / "missing_area.gpkg", layer=spec.name, driver="GPKG")
    if conflict_fraction > conflict_threshold:
        raise RuntimeError(f"{spec.name}: unresolved_conflict_fraction={conflict_fraction:.6f} exceeds {conflict_threshold}")
    if not assigned:
        raise RuntimeError(f"{spec.name}: no atomic polygons were assigned")
    assigned_gdf = gpd.GeoDataFrame(assigned, geometry="geometry", crs=projected_crs)
    diagnostics = {
        "layer": spec.name,
        "source_area_km2": union_area / 1e6,
        "atom_count": len(atoms),
        "assigned_atom_count": len(assigned),
        "uncovered_gap_area_km2": float(sum(row["area_m2"] for row in uncovered) / 1e6),
        "unresolved_conflict_area_km2": conflict_area / 1e6,
        "unresolved_conflict_fraction": conflict_fraction,
        "most_specific_resolution_count": len(resolution_rows),
        "resolution_rows": resolution_rows,
    }
    return assigned_gdf.to_crs(frame.crs), diagnostics


def _preview(path, output):
    output.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path) as src:
        for idx, name in enumerate(src.descriptions, 1):
            scale = max(1, int(np.ceil(max(src.width, src.height) / 900)))
            arr = src.read(
                idx,
                out_shape=(max(1, src.height // scale), max(1, src.width // scale)),
                masked=True,
            )
            fig, ax = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
            if name and "zone" in name:
                from matplotlib.colors import BoundaryNorm, ListedColormap

                cmap = ListedColormap(["#f0f0f0", "#756bb1"])
                norm = BoundaryNorm([0.5, 1.5, 2.5], cmap.N)
                im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
            else:
                im = ax.imshow(arr, cmap="viridis", interpolation="nearest")
            ax.set_title(name or f"band_{idx}", loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            fig.savefig(output / f"{idx:02d}_{name or 'band'}.png", dpi=160)
            plt.close(fig)


def _pixel_area_km2(ref):
    if ref.crs and ref.crs.is_projected:
        return abs(ref.transform.a * ref.transform.e) / 1e6
    if ref.crs and ref.crs.to_epsg() == 4326:
        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        area, _ = geod.polygon_area_perimeter(
            [ref.bounds.left, ref.bounds.right, ref.bounds.right, ref.bounds.left],
            [ref.bounds.bottom, ref.bounds.bottom, ref.bounds.top, ref.bounds.top],
        )
        return abs(area) / (ref.width * ref.height) / 1e6
    return abs(ref.transform.a * ref.transform.e) / 1e6


def _coverage_audit(arrays, ref, topology_rows):
    study = np.isfinite(arrays["extraction_layer_zone_code"])
    study_pixels = int(study.sum())
    if study_pixels == 0:
        raise RuntimeError("Extraction zone has no valid study-area pixels")
    pixel_area = _pixel_area_km2(ref)
    topo = {row["layer"]: row for row in topology_rows}
    layer_map = {
        "L1": "clay_group_1_m",
        "L2": "clay_group_2_m",
        "L3": "clay_group_3_m",
        "L4": "clay_group_4_m",
        "Q4": "quaternary_thickness_m",
        "zone": "extraction_layer_zone_code",
    }
    rows = []
    for label, name in layer_map.items():
        valid = study & np.isfinite(arrays[name])
        missing = study & ~np.isfinite(arrays[name])
        spec_name = {"L1": "clay_group_1", "L2": "clay_group_2", "L3": "clay_group_3", "L4": "clay_group_4", "Q4": "quaternary_thickness", "zone": "extraction_layer_zone"}[label]
        trow = topo.get(spec_name, {})
        coverage = float(valid.sum() / study_pixels)
        rows.append(
            {
                "layer": label,
                "study_area_pixels": study_pixels,
                "valid_pixels": int(valid.sum()),
                "missing_pixels": int(missing.sum()),
                "coverage_fraction": coverage,
                "missing_area_km2": float(missing.sum() * pixel_area),
                "overlap_before_fix_km2": np.nan,
                "conflict_after_fix_km2": float(trow.get("unresolved_conflict_area_km2", 0.0)),
                "topology_gap_area_km2": float(trow.get("uncovered_gap_area_km2", 0.0)),
                "status": "pass" if coverage >= 0.99 else "fail_coverage_lt_0.99",
            }
        )
    return pd.DataFrame(rows)


def run(config_path="config.yaml"):
    config = load_config(config_path)
    geo = config["geology"]
    output = ROOT / geology_output_root(config)
    raw_dir = ROOT / geo["raw_raster_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    reference = resolve_config_path(config, geo["preprocessing"]["reference_raster"])
    previous_audit_path = output / "geology_raw_raster_audit.csv"
    previous_audit = pd.read_csv(previous_audit_path) if previous_audit_path.exists() else pd.DataFrame()
    arrays = {}
    repair_rows = []
    topology_rows = []
    atomic_dir = output / "atomic_polygon_diagnostics"
    with rasterio.open(reference) as ref:
        for spec in resolve_layer_specs(config, ROOT):
            print(f"build_geology_layer {spec.name}", flush=True)
            frame = gpd.read_file(spec.file)
            values, repairs = repaired_layer_values(frame, spec)
            repair_rows.extend(repairs)
            atomic, diag = _build_atomic_interval_polygons(frame, values, spec, atomic_dir)
            topology_rows.append(diag)
            arr = _rasterize(atomic.to_crs(ref.crs), atomic["midpoint"].to_numpy("float32"), ref)
            if spec.name.startswith("clay_group_"):
                band_name = f"{spec.name}_m"
            elif spec.name == "quaternary_thickness":
                band_name = "quaternary_thickness_m"
            else:
                band_name = "extraction_layer_zone_code"
            arrays[band_name] = arr.astype("float32")
            write_geotiff(raw_dir / f"{band_name}.tif", ref, [arrays[band_name]], [band_name])
            if spec.name.startswith("clay_group_"):
                layer_number = spec.name.rsplit("_", 1)[-1]
                write_geotiff(raw_dir / f"clay_thickness_L{layer_number}_midpoint_m.tif", ref, [arrays[band_name]], [band_name])
        coverage = _coverage_audit(arrays, ref, topology_rows)
        coverage.to_csv(output / "geology_layer_coverage_audit.csv", index=False)
        if (coverage["status"] != "pass").any():
            write_json(
                output / "geology_build_blocked.json",
                {"status": "blocked_layer_coverage", "failed_layers": coverage.loc[coverage["status"] != "pass", "layer"].tolist()},
            )
            raise RuntimeError("Geology layer coverage gate failed; formal model covariates were not generated")
        arrays["clay_total_m"] = stack_all_valid([arrays[f"clay_group_{i}_m"] for i in range(1, 5)])
        confined_groups = geo.get("confined_groups")
        if not confined_groups:
            raise RuntimeError("geology.confined_groups must be explicitly configured")
        arrays["clay_confined_m"] = stack_all_valid([arrays[f"clay_group_{i}_m"] for i in confined_groups])
        for name in ("clay_total_m", "clay_confined_m"):
            write_geotiff(raw_dir / f"{name}.tif", ref, [arrays[name]], [name])
        stack_path = output / "geological_raw_covariates.tif"
        pd.DataFrame(topology_rows).to_csv(output / "geology_atomic_polygon_audit.csv", index=False)
        print("write_geological_raw_covariates_stack", flush=True)
        try:
            write_geotiff(stack_path, ref, [arrays[name] for name in RAW_BANDS], list(RAW_BANDS))
            stack_write_status = "canonical_written"
        except PermissionError as exc:
            fallback = output / "geological_raw_covariates_atomic.tif"
            write_geotiff(fallback, ref, [arrays[name] for name in RAW_BANDS], list(RAW_BANDS))
            stack_path = fallback
            stack_write_status = f"canonical_locked_fallback_written: {exc}"
    rows = [raster_stats(arrays[name], name, "m" if name in CONTINUOUS_BANDS else "category code") for name in RAW_BANDS]
    pd.DataFrame(rows).to_csv(output / "geology_raw_raster_audit.csv", index=False)
    pd.DataFrame(repair_rows).to_csv(output / "geology_input_repairs.csv", index=False)
    if not previous_audit.empty:
        previous_audit.to_csv(output / "geology_raw_raster_audit_before_atomic_fix.csv", index=False)
    print("write_window_validation_previews", flush=True)
    _preview(stack_path, output / "window_validation")
    write_json(
        output / "geology_raw_raster_manifest.json",
        {
            "status": "complete",
            "reference_raster": str(reference),
            "raw_raster_dir": str(raw_dir),
            "stack": str(stack_path),
            "stack_write_status": stack_write_status,
            "bands": list(RAW_BANDS),
            "nodata_policy": "continuous and outside-category pixels are stored as NaN; no zero fill is used",
            "all_touched": False,
            "clay_total_formula": "L1 + L2 + L3 + L4",
            "clay_confined_formula": " + ".join([f"L{i}" for i in geo.get("confined_groups", [])]),
            "confined_groups": geo.get("confined_groups", []),
            "repairs": repair_rows,
            "overlap_pixel_policy": "overlap pixels are resolved in vector space by atomic polygons; no count>1 raster pixels are forced to NaN",
            "atomic_polygon_diagnostics": topology_rows,
        },
    )
    print(pd.DataFrame(rows).to_string(index=False))
    return stack_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
