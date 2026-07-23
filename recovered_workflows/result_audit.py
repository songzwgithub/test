from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from io_utils import ROOT, load_config, write_json, write_table


PARAMETER_RASTERS = {
    "Ske_MAP": "Ske_MAP.tif",
    "lag_c_MAP_days": "lag_c_MAP_days.tif",
    "Cu_MAP": "Cu_MAP.tif",
    "lag_u_MAP_days": "lag_u_MAP_days.tif",
    "geological_contribution": "geological_contribution.tif",
    "spatial_basis_contribution": "spatial_basis_contribution.tif",
    "residual_rmse_mm": "residual_rmse_mm.tif",
}

IDENTIFIABILITY_RASTERS = {
    "Ske": "Ske_identifiability.tif",
    "lag_c": "lag_c_identifiability.tif",
    "Cu": "Cu_identifiability.tif",
    "lag_u": "lag_u_identifiability.tif",
    "combined_deformation": "combined_deformation_identifiability.tif",
}


def _raster_values(path: Path) -> tuple[np.ndarray, int]:
    values = []
    total = 0
    with rasterio.open(path) as src:
        total = src.width * src.height
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window, masked=True).compressed()
            if arr.size:
                values.append(arr.astype(float))
    if not values:
        return np.array([], dtype=float), total
    return np.concatenate(values), total


def raster_stats(path: Path, name: str) -> dict:
    vals, total = _raster_values(path)
    row = {"name": name, "path": str(path), "valid_pixel_count": int(vals.size)}
    if vals.size == 0:
        row.update({k: np.nan for k in ["minimum","p01","p02","p05","p25","median","p75","p95","p98","p99","maximum","mean","std","IQR","CV"]})
        row.update({"unique_value_count": 0, "nodata_fraction": 1.0, "status": "empty"})
        return row
    qs = np.nanpercentile(vals, [1,2,5,25,50,75,95,98,99])
    mean = float(np.nanmean(vals)); std = float(np.nanstd(vals))
    row.update({
        "minimum": float(np.nanmin(vals)), "p01": float(qs[0]), "p02": float(qs[1]), "p05": float(qs[2]),
        "p25": float(qs[3]), "median": float(qs[4]), "p75": float(qs[5]), "p95": float(qs[6]),
        "p98": float(qs[7]), "p99": float(qs[8]), "maximum": float(np.nanmax(vals)),
        "mean": mean, "std": std, "IQR": float(qs[5]-qs[3]),
        "CV": float(std/abs(mean)) if mean else np.nan,
        "unique_value_count": int(np.unique(vals).size),
        "nodata_fraction": float(1 - vals.size / total) if total else np.nan,
    })
    status = "ok"
    if row["unique_value_count"] <= 1:
        status = "constant_result_not_plotted"
    elif name == "Cu_MAP" and row["CV"] < 0.01:
        status = "approximately_constant_or_prior_dominated"
    elif name == "lag_u_MAP_days" and row["IQR"] < 2:
        status = "approximately_spatially_uniform"
    row["status"] = status
    return row


def posterior_checks(output: Path) -> list[dict]:
    checks = []
    pairs = [
        ("Ske_screened", "Ske_posterior_median_screened.tif", "Ske_ci95_low_screened.tif", "Ske_ci95_high_screened.tif", "Ske_relative_ci95_width_screened.tif"),
        ("Cu_screened", "Cu_posterior_median_screened.tif", "Cu_ci95_low_screened.tif", "Cu_ci95_high_screened.tif", "Cu_relative_ci95_width_screened.tif"),
    ]
    for name, med, low, high, rel in pairs:
        paths = [output / x for x in (med, low, high, rel)]
        if not all(p.exists() for p in paths):
            checks.append({"name": name, "posterior_status": "missing", "missing": [str(p) for p in paths if not p.exists()]})
            continue
        m,_ = _raster_values(paths[0]); lo,_ = _raster_values(paths[1]); hi,_ = _raster_values(paths[2]); rw,_ = _raster_values(paths[3])
        n = min(len(m), len(lo), len(hi), len(rw))
        violations = int(np.sum((lo[:n] > m[:n]) | (m[:n] > hi[:n]))) if n else 0
        unstable = float(np.nanmean(rw[:n] > 10)) if n else np.nan
        checks.append({"name": name, "posterior_median": float(np.nanmedian(m)) if len(m) else np.nan,
                       "posterior_relative_ci_width_median": float(np.nanmedian(rw)) if len(rw) else np.nan,
                       "CI_ordering_violations": violations,
                       "nonfinite_fraction": float(np.mean(~np.isfinite(rw))) if len(rw) else np.nan,
                       "posterior_status": "unstable_transformed_laplace" if unstable > 0.05 else "ok"})
    return checks


def identifiability_area_summary(output: Path) -> pd.DataFrame:
    rows = []
    for name, filename in IDENTIFIABILITY_RASTERS.items():
        path = output / filename
        if not path.exists():
            rows.append({"parameter": name, "path": str(path), "status": "missing"})
            continue
        with rasterio.open(path) as src:
            transform = src.transform
            dx = abs(transform.a)
            dy = abs(transform.e)
            class_area = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
            valid_area = 0.0
            valid_pixels = 0
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window, masked=True)
                mask = ~np.ma.getmaskarray(arr)
                if not mask.any():
                    continue
                rr = np.arange(int(window.row_off), int(window.row_off + window.height))
                lat = transform.f + (rr + 0.5) * transform.e
                row_area = dx * 111.32 * np.cos(np.deg2rad(lat)) * dy * 110.574
                area_grid = row_area[:, None]
                valid_area += float(np.sum(area_grid * mask))
                valid_pixels += int(mask.sum())
                data = np.asarray(arr.filled(-9999), dtype=int)
                for cls in class_area:
                    class_area[cls] += float(np.sum(area_grid * (mask & (data == cls))))
            identified_area = class_area[2] + class_area[3]
            rows.append({
                "parameter": name,
                "path": str(path),
                "valid_pixels": valid_pixels,
                "valid_area_km2": valid_area,
                "not_identified_area_km2": class_area[0],
                "weak_area_km2": class_area[1],
                "identified_area_km2": identified_area,
                "strong_area_km2": class_area[3],
                "identified_area_fraction": identified_area / valid_area if valid_area else np.nan,
                "status": "complete",
            })
    return pd.DataFrame(rows)


def run(config_path="config.yaml"):
    config = load_config(config_path)
    output = ROOT / config["project"]["output_dir"]
    raster_rows = []
    for name, filename in PARAMETER_RASTERS.items():
        path = output / filename
        if path.exists():
            raster_rows.append(raster_stats(path, name))
        else:
            raster_rows.append({"name": name, "path": str(path), "status": "missing"})
    stats = {row["name"]: row for row in raster_rows}
    derived = {
        "Cu_map_status": stats.get("Cu_MAP", {}).get("status"),
        "lag_u_map_status": stats.get("lag_u_MAP_days", {}).get("status"),
        "Cu_spatial_CV": stats.get("Cu_MAP", {}).get("CV"),
        "Ske_spatial_CV": stats.get("Ske_MAP", {}).get("CV"),
        "lag_c_IQR_days": stats.get("lag_c_MAP_days", {}).get("IQR"),
        "lag_u_IQR_days": stats.get("lag_u_MAP_days", {}).get("IQR"),
    }
    mv = output / "model_variant.tif"
    if mv.exists():
        mv_stats = raster_stats(mv, "model_variant")
        derived["model_variant_map_status"] = "constant_not_suitable_as_map" if mv_stats.get("unique_value_count") == 1 else "ok"
        raster_rows.append(mv_stats)
    posterior = posterior_checks(output)
    area_summary = identifiability_area_summary(output)
    write_table(area_summary, output / "identifiability_area_summary.csv")
    audit = {"raster_products": raster_rows, "posterior_products": posterior,
             "identifiability_area_summary": area_summary.to_dict(orient="records"),
             "derived_metrics": derived}
    write_json(audit, output / "result_audit.json")
    write_table(pd.DataFrame(raster_rows + posterior + [{"name": k, "value": v, "status": "derived"} for k, v in derived.items()]), output / "result_audit.csv")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result["derived_metrics"], indent=2))
