from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio

from insar_processing import los_to_vertical
from io_utils import ROOT, load_config, resolve_config_path, write_json


def _profile_from_template(template: Path) -> dict:
    with rasterio.open(template) as src:
        profile = src.profile.copy()
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan, compress="lzw",
                   tiled=True, blockxsize=256, blockysize=256)
    return profile


def _cache_path(output: Path) -> Path:
    diagnostics = json.loads((output / "map_diagnostics.json").read_text(encoding="utf-8"))
    path = Path(diagnostics.get("cache_path", ""))
    if not path.exists():
        raise FileNotFoundError(f"Recorded Phase 4 cache is missing: {path}")
    return path


def write_harmonic_overview(output: Path, template: Path, period_days: float) -> list[dict]:
    cache = _cache_path(output)
    profile = _profile_from_template(template)
    targets = {
        "amplitude": output / "insar_annual_amplitude_mm.tif",
        "phase": output / "insar_annual_phase_days.tif",
    }
    stats = []
    width = int(profile["width"])
    with h5py.File(cache, "r") as h5, rasterio.open(targets["amplitude"], "w", **profile) as amp_dst, rasterio.open(targets["phase"], "w", **profile) as phase_dst:
        starts = h5["block_start"][:]
        counts = h5["block_count"][:]
        rows = h5["block_row"][:]
        cols = h5["block_col"][:]
        heights = h5["block_height"][:]
        widths = h5["block_width"][:]
        for bi, (start, count, row, col, height, block_width) in enumerate(zip(starts, counts, rows, cols, heights, widths), 1):
            obs = h5["obs"][start:start + count].astype("float32")
            flat = h5["flat_index"][start:start + count].astype("int64")
            row = int(row); col = int(col); height = int(height); block_width = int(block_width)
            amp = np.full((int(height), int(block_width)), np.nan, dtype="float32")
            phase = np.full_like(amp, np.nan)
            sine = obs[:, 0]
            cosine = obs[:, 1]
            amp.reshape(-1)[flat] = np.sqrt(sine * sine + cosine * cosine)
            phase_days = (np.arctan2(sine, cosine) % (2 * np.pi)) * period_days / (2 * np.pi)
            phase.reshape(-1)[flat] = phase_days.astype("float32")
            window = rasterio.windows.Window(int(col), int(row), int(block_width), int(height))
            amp_dst.write(amp, 1, window=window)
            phase_dst.write(phase, 1, window=window)
            if bi % 10 == 0 or bi == len(starts):
                print(f"insar_overview_harmonic_block {bi}/{len(starts)}", flush=True)
    for name, path in targets.items():
        stats.append({"product": path.name, "source": str(cache), "method": f"Phase 4 real InSAR annual harmonic {name}"})
    return stats


def write_endpoint_velocity(config: dict, output: Path, template: Path) -> dict:
    epochs = pd.read_csv(output / "insar_epochs.csv")
    first = Path(epochs.iloc[0]["source_file"])
    last = Path(epochs.iloc[-1]["source_file"])
    years = (pd.to_datetime(epochs.iloc[-1]["date"]) - pd.to_datetime(epochs.iloc[0]["date"])).days / 365.2425
    if years <= 0:
        raise ValueError("InSAR epoch span is non-positive")
    incidence_path = resolve_config_path(config, config["insar"]["incidence_grid"])
    incidence = np.load(incidence_path, mmap_mode="r")
    profile = _profile_from_template(template)
    out = output / "insar_mean_vertical_velocity_mm_yr.tif"
    with rasterio.open(first) as src0, rasterio.open(last) as src1, rasterio.open(out, "w", **profile) as dst:
        for wi, (_, window) in enumerate(src0.block_windows(1), 1):
            los0 = src0.read(1, window=window, masked=True).astype("float32")
            los1 = src1.read(1, window=window, masked=True).astype("float32")
            inc = np.asarray(incidence[int(window.row_off):int(window.row_off + window.height),
                                      int(window.col_off):int(window.col_off + window.width)], dtype="float32")
            diff_mm = (los1 - los0) * 1000.0
            vertical = los_to_vertical(diff_mm, inc) / years
            arr = np.asarray(vertical, dtype="float32")
            arr[np.ma.getmaskarray(los0) | np.ma.getmaskarray(los1) | ~np.isfinite(arr)] = np.nan
            dst.write(arr, 1, window=window)
            if wi % 100 == 0:
                print(f"insar_overview_velocity_window {wi}", flush=True)
    return {"product": out.name, "source_first": str(first), "source_last": str(last), "method": "endpoint mean vertical velocity from real SAR rasters"}


def run(config_path: str = "config.yaml") -> list[dict]:
    config = load_config(config_path)
    output = ROOT / config["project"]["output_dir"]
    template = output / "geological_model_covariates.tif"
    records = []
    records.extend(write_harmonic_overview(output, template, float(config["temporal"]["annual_period_days"])))
    records.append(write_endpoint_velocity(config, output, template))
    write_json({"products": records}, output / "insar_overview_products.json")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False))
