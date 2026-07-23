#!/usr/bin/env python3
"""Compare legacy outputs with the geology-fix revision output root."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from plotting.common import raster_summary
from io_utils import ROOT, load_config
from geology_preprocessing import geology_output_root


PRODUCTS = [
    "geological_model_covariates.tif",
    "Ske_MAP.tif",
    "lag_c_MAP_days.tif",
    "residual_rmse_mm.tif",
    "geological_contribution.tif",
    "spatial_basis_contribution.tif",
    "Ske_relative_ci95_width_screened.tif",
    "lag_c_ci95_width_days.tif",
]


def run(config_path="config.yaml"):
    config = load_config(config_path)
    old = ROOT / "outputs"
    new = ROOT / geology_output_root(config)
    rows = []
    for name in PRODUCTS:
        old_path = old / name
        new_path = new / name
        row = {"product": name, "legacy_exists": old_path.exists(), "new_exists": new_path.exists()}
        if old_path.exists():
            row.update({f"legacy_{k}": v for k, v in raster_summary(old_path).items() if k in {"valid_pixels", "median", "p05", "p95", "min", "max"}})
        if new_path.exists():
            row.update({f"new_{k}": v for k, v in raster_summary(new_path).items() if k in {"valid_pixels", "median", "p05", "p95", "min", "max"}})
        rows.append(row)
    table = pd.DataFrame(rows)
    out_csv = new / "geology_fix_before_after_comparison.csv"
    new.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    phase_status = json.loads((new / "phase_status.json").read_text(encoding="utf-8")) if (new / "phase_status.json").exists() else {}
    status = "complete" if phase_status.get("phase_5") == "complete" and all((new / p).exists() for p in PRODUCTS[:6]) else "partial_geology_fix_outputs"
    (new / "geology_fix_comparison_status.json").write_text(
        json.dumps({"status": status, "comparison_csv": str(out_csv), "phase_status": phase_status}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
