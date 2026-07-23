from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from io_utils import ROOT, load_config, write_json


PRODUCTS = {
    "mean_vertical_velocity": "insar_mean_vertical_velocity_mm_yr.tif",
    "annual_vertical_amplitude": "insar_annual_amplitude_mm.tif",
    "annual_vertical_phase": "insar_annual_phase_days.tif",
}


def _valid_window(path: Path) -> dict | None:
    rows = []
    cols = []
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window, masked=True)
            mask = np.isfinite(np.asarray(arr.filled(np.nan), dtype=float))
            if mask.any():
                rr, cc = np.where(mask)
                rows.extend((rr + int(window.row_off)).tolist())
                cols.extend((cc + int(window.col_off)).tolist())
    if not rows:
        return None
    return {"row_min": int(min(rows)), "row_max": int(max(rows)),
            "col_min": int(min(cols)), "col_max": int(max(cols))}


def raster_diagnostics(path: Path, source_h5: str | None, source_dates: dict, config: dict) -> dict:
    row = {"path": str(path), "exists": path.exists(), "source_h5": source_h5,
           "source_dates": source_dates,
           "harmonic_origin": config["temporal"].get("harmonic_origin"),
           "annual_period_days": config["temporal"].get("annual_period_days")}
    if not path.exists():
        return row | {"status": "missing"}
    vals = []
    mask_count = 0
    total = 0
    with rasterio.open(path) as src:
        row.update({"shape": [src.height, src.width], "crs": str(src.crs),
                    "transform": list(src.transform)[:6], "nodata": src.nodata,
                    "dtype": src.dtypes[0]})
        total = src.width * src.height
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window, masked=True)
            filled = np.asarray(arr.filled(np.nan), dtype=float)
            finite = filled[np.isfinite(filled)]
            mask_count += int((~np.ma.getmaskarray(arr)).sum())
            if finite.size:
                vals.append(finite)
    if vals:
        data = np.concatenate(vals)
        qs = np.nanpercentile(data, [1, 5, 50, 95, 99])
        row.update({"finite_pixel_count": int(data.size),
                    "finite_fraction": float(data.size / total) if total else np.nan,
                    "minimum": float(np.nanmin(data)), "p01": float(qs[0]),
                    "p05": float(qs[1]), "median": float(qs[2]),
                    "p95": float(qs[3]), "p99": float(qs[4]),
                    "maximum": float(np.nanmax(data)),
                    "mask_finite_count": int(mask_count),
                    "valid_window": _valid_window(path),
                    "blank_diagnosis": "finite_sparse_raster; use finite p02-p98 color limits and sparse-aware preview",
                    "status": "valid"})
    else:
        row.update({"finite_pixel_count": 0, "finite_fraction": 0.0,
                    "minimum": None, "p01": None, "p05": None, "median": None,
                    "p95": None, "p99": None, "maximum": None,
                    "mask_finite_count": int(mask_count), "valid_window": None,
                    "blank_diagnosis": "all_nan_or_all_masked", "status": "invalid_input"})
    return row


def run(config_path: str = "config.yaml") -> dict:
    config = load_config(config_path)
    output = ROOT / config["project"]["output_dir"]
    source_dates = {}
    epochs = output / "insar_epochs.csv"
    if epochs.exists():
        df = pd.read_csv(epochs)
        source_dates = {"first": str(df.iloc[0]["date"]), "last": str(df.iloc[-1]["date"]), "n_epochs": int(len(df))}
    source_h5 = None
    diag = output / "map_diagnostics.json"
    if diag.exists():
        payload = json.loads(diag.read_text(encoding="utf-8"))
        source_h5 = payload.get("cache_path")
    products = {name: raster_diagnostics(output / filename, source_h5, source_dates, config)
                for name, filename in PRODUCTS.items()}
    required = ["mean_vertical_velocity", "annual_vertical_amplitude", "annual_vertical_phase"]
    status = "complete" if all(products[x].get("finite_pixel_count", 0) > 0 for x in required) else "invalid_input"
    result = {"status": status, "products": products}
    write_json(result, output / "insar_overview_product_diagnostics.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False))
