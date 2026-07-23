"""Blocking Phase-1 audit for real InSAR, groundwater, bulletin, and geology inputs."""
from __future__ import annotations

import glob
import platform
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from groundwater_processing import ALLOWED_AQUIFERS, load_groundwater_wide
from insar_processing import GeoTiffCube
from io_utils import ROOT, resolve_path
from bulletin_processing import validate_parser
from geological_prior import inspect_vector


def run_phase1_audit(config):
    rows = []
    insar_config = config["insar"]
    pattern = resolve_path(insar_config["geotiff_glob"])
    try:
        cube = GeoTiffCube.from_glob(pattern, unit=insar_config["displacement_unit"])
        rows.extend([
            gate("insar_epochs", len(cube.epochs) >= 2, True, f"n={len(cube.epochs)}"),
            gate("insar_crs", bool(cube.crs.strip()), True, cube.crs),
            gate("insar_grid", True, True, f"shape={cube.shape}"),
            gate("insar_sign", insar_config.get("los_sign_convention") == "positive_toward_satellite", True, insar_config.get("los_sign_convention")),
        ])
        incidence_path = resolve_path(insar_config["incidence_grid"])
        incidence = np.load(incidence_path, mmap_mode="r")
        rows.append(gate("incidence_grid", tuple(incidence.shape) == tuple(cube.shape), True, f"shape={incidence.shape};expected={cube.shape}"))
        estimated=int(np.prod(cube.cube_shape)*4);free=shutil.disk_usage(ROOT).free
        rows.append(gate("vertical_h5_disk_space",free>=estimated*1.1,True,f"estimated={estimated};free={free}"))
    except Exception as exc:
        cube = None
        rows.append(gate("insar_readable", False, True, f"{type(exc).__name__}: {exc}"))

    groundwater_config = config["groundwater"]
    groundwater_path = resolve_path(groundwater_config["file"])
    try:
        groundwater = load_groundwater_wide(
            groundwater_path,
            groundwater_config["field_map"],
            groundwater_config["aquifer_labels"],
        )
        metadata = groundwater.drop_duplicates("well_id")
        labels = set(metadata["aquifer_type"].dropna().unique())
        rows.extend([
            gate("groundwater_wells", metadata["well_id"].nunique() == groundwater_config.get("expected_well_count", 164), True,
                 f"n={metadata['well_id'].nunique()};expected={groundwater_config.get('expected_well_count',164)}"),
            gate("groundwater_coordinates", metadata[["lon", "lat"]].notna().all(axis=None), True, "lon/lat complete"),
            gate("groundwater_explicit_aquifer", labels.issubset(ALLOWED_AQUIFERS) and labels == ALLOWED_AQUIFERS, True, ",".join(sorted(labels))),
            gate("groundwater_elevation", metadata["ground_elevation_m"].notna().all(), True, f"complete={metadata['ground_elevation_m'].notna().mean():.3f}"),
        ])
    except Exception as exc:
        groundwater = None
        rows.append(gate("groundwater_readable", False, True, f"{type(exc).__name__}: {exc}"))

    if cube is not None and groundwater is not None:
        gw_start,gw_end=groundwater.date.min(),groundwater.date.max();insar_dates=pd.DatetimeIndex(cube.epochs.date)
        overlap=(insar_dates>=gw_start)&(insar_dates<=gw_end);pairs=int(overlap.sum())
        max_lag=config.get("lag",{}).get("maximum_days",365);lag_covered=int((insar_dates-pd.to_timedelta(max_lag,unit="D")>=gw_start).sum())
        rows.extend([gate("temporal_overlap",pairs>0,True,f"{max(gw_start,insar_dates.min())} to {min(gw_end,insar_dates.max())}"),
                     gate("sar_head_pairs",pairs>=24,True,f"n={pairs}"),gate("lagged_sar_head_pairs",lag_covered>=24,True,f"n={lag_covered};lag={max_lag}d")])
    forbidden=[key for key in config.get("geology",{}) if any(word in key.lower() for word in ("fresh","saline"))]
    rows.append(gate("no_fresh_saline_configuration",not forbidden,True,str(forbidden)))
    bulletin_path = resolve_path(config["bulletin"]["file"])
    try:
        verified_path=resolve_path(config["bulletin"]["verified_file"]);comparison,conflicts=validate_parser(bulletin_path,verified_path)
        rows.append(gate("bulletin_verified_match",conflicts.empty,True,f"rows={len(comparison)};conflicts={len(conflicts)}"))
    except Exception as exc: rows.append(gate("bulletin_parseable",False,True,str(exc)))
    for name, configured in config.get("geology", {}).items():
        if name in {"attribute_fields","confined_groups","model_covariates","quality_layers","spatial_basis"}: continue
        if name=="extraction_layer_zone":
            path=resolve_path(configured["file"]);field=configured["value_field"]
            try:
                inspect_vector(path,field);rows.append(gate("geology_schema_extraction_layer_zone",True,True,f"{field};mapping={configured['category_mapping']}"))
            except Exception as exc:rows.append(gate("geology_schema_extraction_layer_zone",False,True,str(exc)))
            continue
        if configured is None:
            rows.append(gate(f"geology_{name}", False, False, "not configured"))
        else:
            path = resolve_path(configured)
            exists=path.exists(); rows.append(gate(f"geology_{name}", exists, True, str(path)))
            field=config["geology"].get("attribute_fields",{}).get(name)
            if exists:
                try: inspect_vector(path,field);rows.append(gate(f"geology_schema_{name}",True,True,field))
                except Exception as exc: rows.append(gate(f"geology_schema_{name}",False,True,str(exc)))
    table = pd.DataFrame(rows)
    report = {
        "phase": 1,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "critical_passed": bool(table.loc[table["critical"], "passed"].all()) if table["critical"].any() else False,
        "n_gates": len(table),
        "n_passed": int(table["passed"].sum()),
        "gates": table.to_dict("records"),
    }
    return report, cube, groundwater


def gate(name, passed, critical, detail):
    return {"gate": name, "passed": bool(passed), "critical": bool(critical), "detail": str(detail)}


def require_audit(report):
    if not report.get("critical_passed", False):
        failures = [row["gate"] for row in report.get("gates", []) if row["critical"] and not row["passed"]]
        raise RuntimeError(f"Phase-1 audit blocked the pipeline: {failures}")
