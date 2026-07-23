#!/usr/bin/env python3
"""Audit candidate geological covariate design matrices G0-G3."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import spearmanr

from io_utils import ROOT, load_config


def _pixel_area_km2(src):
    if src.crs and src.crs.is_projected:
        return abs(src.transform.a * src.transform.e) / 1e6
    if src.crs and src.crs.to_epsg() == 4326:
        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        area, _ = geod.polygon_area_perimeter(
            [src.bounds.left, src.bounds.right, src.bounds.right, src.bounds.left],
            [src.bounds.bottom, src.bounds.bottom, src.bounds.top, src.bounds.top],
        )
        return abs(area) / (src.width * src.height) / 1e6
    return abs(src.transform.a * src.transform.e) / 1e6


def _read_layer(raw_dir, name):
    path = raw_dir / f"{name}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Missing covariate raster: {path}")
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src


def _vif_matrix(X):
    if X.shape[1] == 0:
        return {}
    out = {}
    for i in range(X.shape[1]):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        if others.shape[1] == 0:
            out[i] = 1.0
            continue
        A = np.column_stack([np.ones(len(others)), others])
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        pred = A @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        out[i] = float(1.0 / max(1.0e-12, 1.0 - r2))
    return out


def _audit_model(model_id, candidates, rasters, pixel_area):
    covariates = list(candidates.get("continuous", []))
    forbidden = {"extraction_layer_zone_code", "cumulative_total_clay_thickness_m", "clay_total_m", "clay_confined_m"}
    if forbidden.intersection(covariates):
        raise ValueError(f"{model_id}: forbidden inversion covariates present: {sorted(forbidden.intersection(covariates))}")
    if {"cumulative_confined_clay_thickness_m", "quaternary_thickness_m", "confined_clay_fraction"}.issubset(set(covariates)):
        raise ValueError(f"{model_id}: fraction and numerator/denominator cannot all be used together")
    if not covariates:
        return {
            "summary": {
                "geology_model_id": model_id,
                "covariates": "",
                "valid_pixel_count": int(next(iter(rasters.values())).size),
                "valid_area_km2": float(next(iter(rasters.values())).size * pixel_area),
                "condition_number": 1.0,
                "effective_rank": 0,
                "singular_values": [],
                "maximum_vif": 1.0,
                "status": "ok",
            },
            "correlations": [],
            "vif": [],
        }
    arrays = [rasters[name] for name in covariates]
    valid = np.logical_and.reduce([np.isfinite(a) for a in arrays])
    X = np.column_stack([a[valid].astype(float) for a in arrays])
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    _, singular, _ = np.linalg.svd(Xs, full_matrices=False)
    condition = float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else float("inf")
    rank = int(np.linalg.matrix_rank(Xs))
    vif_raw = _vif_matrix(Xs)
    vif_rows = [{"geology_model_id": model_id, "covariate": covariates[i], "vif": value} for i, value in vif_raw.items()]
    corr_rows = []
    pearson = np.corrcoef(Xs, rowvar=False) if Xs.shape[1] > 1 else np.array([[1.0]])
    for i, a in enumerate(covariates):
        for j, b in enumerate(covariates):
            if j <= i:
                continue
            rho = spearmanr(Xs[:, i], Xs[:, j]).correlation
            corr_rows.append(
                {
                    "geology_model_id": model_id,
                    "covariate_i": a,
                    "covariate_j": b,
                    "pearson_r": float(pearson[i, j]),
                    "spearman_rho": float(rho),
                    "pearson_status": "high_correlation_warning" if abs(pearson[i, j]) > 0.85 else "ok",
                    "spearman_status": "high_rank_correlation_warning" if abs(rho) > 0.85 else "ok",
                }
            )
    max_vif = max([row["vif"] for row in vif_rows] or [1.0])
    status = "ok"
    if max_vif > 10 or condition > 1000:
        status = "blocked_design"
    elif max_vif > 5 or condition > 100:
        status = "warning"
    return {
        "summary": {
            "geology_model_id": model_id,
            "covariates": ",".join(covariates),
            "valid_pixel_count": int(valid.sum()),
            "valid_area_km2": float(valid.sum() * pixel_area),
            "condition_number": condition,
            "effective_rank": rank,
            "singular_values": [float(x) for x in singular],
            "maximum_vif": float(max_vif),
            "status": status,
        },
        "correlations": corr_rows,
        "vif": vif_rows,
    }


def run(config_path="config.yaml", output_root="outputs/aquifer_model_revision"):
    config = load_config(config_path)
    output = ROOT / output_root
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / config["geology"].get("raw_raster_dir", "data/geology_rasters")
    layer_names = sorted({name for item in config["geology"]["model_covariate_candidates"].values() for name in item.get("continuous", [])})
    if "extraction_layer_zone_code" in layer_names:
        raise ValueError("extraction_layer_zone_code is not allowed in candidate design matrices")
    rasters = {}
    template = None
    for name in layer_names or ["cumulative_confined_clay_thickness_m"]:
        arr, src = _read_layer(raw_dir, name)
        rasters[name] = arr
        template = src
    pixel_area = _pixel_area_km2(template)
    summaries, correlations, vifs = [], [], []
    payload = {"status": "complete", "models": {}}
    for model_id, spec in config["geology"]["model_covariate_candidates"].items():
        result = _audit_model(model_id, spec, rasters, pixel_area)
        summaries.append(result["summary"])
        correlations.extend(result["correlations"])
        vifs.extend(result["vif"])
        payload["models"][model_id] = result
    pd.DataFrame(summaries).to_csv(output / "geological_design_matrix_audit.csv", index=False)
    pd.DataFrame(correlations).to_csv(output / "geological_covariate_correlation.csv", index=False)
    pd.DataFrame(vifs).to_csv(output / "geological_covariate_vif.csv", index=False)
    (output / "geological_design_matrix_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    args = parser.parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    main()
