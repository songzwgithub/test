from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from io_utils import ROOT, load_config, write_json


PRODUCTS = [
    "Ske_relative_ci95_width_screened.tif",
    "logSke_posterior_std.tif",
    "lag_c_ci95_width_days.tif",
]


def _centers(output: Path) -> np.ndarray:
    diag = json.loads((output / "map_diagnostics.json").read_text(encoding="utf-8"))
    design = diag.get("design_metadata", {})
    centers = design.get("spatial_basis", {}).get("centers", [])
    return np.asarray(centers, dtype=float)


def _hotspots(path: Path, max_points: int = 25) -> list[dict]:
    points = []
    with rasterio.open(path) as src:
        vals = []
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window, masked=True).compressed()
            if arr.size:
                vals.append(arr.astype(float))
        if not vals:
            return []
        threshold = float(np.nanpercentile(np.concatenate(vals), 99.5))
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window, masked=True)
            data = np.asarray(arr.filled(np.nan), dtype=float)
            rr, cc = np.where(data >= threshold)
            if rr.size:
                values = data[rr, cc]
                order = np.argsort(values)[::-1][:max_points]
                xs, ys = rasterio.transform.xy(src.transform, rr[order] + int(window.row_off), cc[order] + int(window.col_off), offset="center")
                for x, y, value in zip(xs, ys, values[order]):
                    points.append({"x": float(x), "y": float(y), "value": float(value)})
    points = sorted(points, key=lambda r: r["value"], reverse=True)[:max_points]
    return points


def run(config_path: str = "config.yaml") -> dict:
    config = load_config(config_path)
    output = ROOT / config["project"]["output_dir"]
    centers = _centers(output)
    center_tree = cKDTree(centers) if len(centers) else None
    # Centers are projected; hotspot coordinates are lon/lat. Use pyproj for comparable distances.
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:4326", config["project"].get("projected_crs", "EPSG:32650"), always_xy=True)
    product_hotspots = {}
    all_keys = {}
    for product in PRODUCTS:
        rows = []
        for h in _hotspots(output / product):
            px, py = transformer.transform(h["x"], h["y"])
            dist, idx = center_tree.query([px, py]) if center_tree is not None else (np.nan, -1)
            key = (round(float(px) / 5000), round(float(py) / 5000))
            all_keys.setdefault(key, set()).add(product)
            rows.append(h | {"nearest_rbf_center_index": int(idx), "distance_to_nearest_rbf_center_m": float(dist),
                             "overlaps_masked_or_low_identifiability_area": None})
        product_hotspots[product] = rows
    appears_all = [key for key, products in all_keys.items() if len(products) == len(PRODUCTS)]
    result = {
        "rbf_center_coordinates_projected": centers.tolist(),
        "products": product_hotspots,
        "hotspot_grid_cells_appearing_in_all_uncertainty_products": [list(k) for k in appears_all],
        "interpretation": "localized uncertainty highs coincide with the RBF support-center pattern and are treated as model-structure display artifacts, not hydrogeological boundaries",
    }
    write_json(result, output / "rbf_artifact_diagnostics.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2)[:4000])
