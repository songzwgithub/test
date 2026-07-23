from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from io_utils import ROOT, load_config, write_json


FIELDS = [
    ("Fig06", "total_clay_thickness", "geological_model_covariates.tif", 1),
    ("Fig06", "confined_clay_thickness", "geological_model_covariates.tif", 2),
    ("Fig06", "quaternary_thickness", "geological_model_covariates.tif", 3),
    ("Fig06", "extraction_layer_zone", "geological_model_covariates.tif", 4),
    ("Fig06", "geological_contribution", "geological_contribution.tif", 1),
    ("Fig06", "spatial_basis_contribution", "spatial_basis_contribution.tif", 1),
    ("Fig07", "Ske_MAP", "Ske_MAP.tif", 1),
    ("Fig07", "lag_c_MAP_days", "lag_c_MAP_days.tif", 1),
    ("Fig07", "residual_rmse_mm", "residual_rmse_mm.tif", 1),
    ("Fig07", "geological_contribution", "geological_contribution.tif", 1),
    ("Fig07", "spatial_basis_contribution", "spatial_basis_contribution.tif", 1),
    ("Fig08", "Ske_relative_ci95_width_screened", "Ske_relative_ci95_width_screened.tif", 1),
    ("Fig08", "logSke_posterior_std", "logSke_posterior_std.tif", 1),
    ("Fig08", "lag_c_ci95_width_days", "lag_c_ci95_width_days.tif", 1),
    ("Fig08", "Ske_identifiability", "Ske_identifiability.tif", 1),
    ("Fig08", "lag_c_identifiability", "lag_c_identifiability.tif", 1),
]


def _values(path: Path, band: int) -> tuple[np.ndarray, dict]:
    vals = []
    with rasterio.open(path) as src:
        meta = {"shape": [src.height, src.width], "crs": str(src.crs), "nodata": src.nodata}
        for _, window in src.block_windows(band):
            arr = src.read(band, window=window, masked=True).compressed()
            if arr.size:
                vals.append(arr.astype(float))
    return (np.concatenate(vals) if vals else np.array([], dtype=float)), meta


def audit_field(output: Path, figure: str, name: str, filename: str, band: int) -> dict:
    path = output / filename
    row = {"figure": figure, "field": name, "path": str(path), "band": band, "exists": path.exists()}
    if not path.exists():
        return row | {"status": "missing"}
    vals, meta = _values(path, band)
    row.update(meta)
    total = int(np.prod(meta["shape"]))
    row["finite_count"] = int(vals.size)
    row["finite_fraction"] = float(vals.size / total) if total else np.nan
    if vals.size == 0:
        return row | {"unique_value_count": 0, "status": "empty"}
    qs = np.nanpercentile(vals, [1, 5, 25, 50, 75, 95, 99])
    unique = int(np.unique(vals).size)
    mean = float(np.nanmean(vals))
    std = float(np.nanstd(vals))
    iqr = float(qs[4] - qs[2])
    cv = float(std / abs(mean)) if mean else np.inf
    discrete = unique <= 5 and np.isclose(iqr, 0)
    nearly = bool(cv < 0.01 or (iqr == 0 and unique <= 10))
    suspicious = bool(name in {"Ske_relative_ci95_width_screened", "logSke_posterior_std", "lag_c_ci95_width_days"} and qs[6] > qs[3] * 1.5)
    row.update({"unique_value_count": unique, "min": float(np.nanmin(vals)), "p01": float(qs[0]),
                "p05": float(qs[1]), "median": float(qs[3]), "p95": float(qs[5]),
                "p99": float(qs[6]), "max": float(np.nanmax(vals)), "std": std, "IQR": iqr,
                "CV": cv, "discrete_field": bool(discrete),
                "nearly_constant_field": bool(nearly),
                "suspicious_ring_artifact_candidate": suspicious,
                "status": "ok"})
    return row


def run(config_path: str = "config.yaml") -> dict:
    config = load_config(config_path)
    output = ROOT / config["project"]["output_dir"]
    rows = [audit_field(output, *field) for field in FIELDS]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "figure_source_field_audit.csv", index=False)
    result = {"fields": rows}
    write_json(result, output / "figure_source_field_audit.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
