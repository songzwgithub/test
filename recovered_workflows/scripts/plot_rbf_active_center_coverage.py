#!/usr/bin/env python
"""Plot and audit active RBF center coverage over the common mask."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import xy


def load_selection(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload


def load_centers(cache_path):
    with h5py.File(cache_path, "r") as h5:
        meta = json.loads(h5.attrs["design_metadata"])
    spatial = meta["spatial_basis"]
    return np.asarray(spatial["centers"], float), float(spatial["scale_m"]), spatial["projected_crs"]


def iter_projected_mask_points(mask_path, target_crs):
    with rasterio.open(mask_path) as src:
        transformer = Transformer.from_crs(src.crs, target_crs, always_xy=True) if str(src.crs) != str(target_crs) else None
        for _, window in src.block_windows(1):
            mask = src.read(1, window=window) == 1
            if not mask.any():
                continue
            rows, cols = np.nonzero(mask)
            rr = rows + int(window.row_off)
            cc = cols + int(window.col_off)
            xs, ys = xy(src.transform, rr, cc, offset="center")
            xs = np.asarray(xs, float)
            ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float)
                ys = np.asarray(ys, float)
            yield xs, ys


def nearest_distance_stats(mask_path, centers, active, scale_m, target_crs):
    active_centers = centers[np.asarray(active, dtype=int)]
    distances = []
    total = 0
    beyond1 = 0
    beyond2 = 0
    min_dist = np.inf
    max_dist = 0.0
    for xs, ys in iter_projected_mask_points(mask_path, target_crs):
        points = np.column_stack([xs, ys])
        diff = points[:, None, :] - active_centers[None, :, :]
        nearest = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1)) / 1000.0
        distances.append(nearest.astype("float32"))
        total += nearest.size
        beyond1 += int(np.sum(nearest > scale_m / 1000.0))
        beyond2 += int(np.sum(nearest > 2.0 * scale_m / 1000.0))
        min_dist = min(min_dist, float(nearest.min()))
        max_dist = max(max_dist, float(nearest.max()))
    d = np.concatenate(distances)
    return {
        "minimum_nearest_center_distance_km": float(min_dist),
        "median_nearest_center_distance_km": float(np.median(d)),
        "p90_nearest_center_distance_km": float(np.percentile(d, 90)),
        "p95_nearest_center_distance_km": float(np.percentile(d, 95)),
        "maximum_nearest_center_distance_km": float(max_dist),
        "area_fraction_beyond_1_rbf_scale": float(beyond1 / max(total, 1)),
        "area_fraction_beyond_2_rbf_scales": float(beyond2 / max(total, 1)),
        "common_mask_pixel_count": int(total),
    }


def plot_coverage(mask_path, centers, selection, output_png, target_crs):
    active = set(selection["active_column_indices"])
    support_drop = set(selection.get("support_dropped_column_indices", []))
    gram_drop = set(selection.get("full_gram_condition_dropped_column_indices", []))
    qr_drop = set(selection.get("qr_dropped_column_indices", []))
    with rasterio.open(mask_path) as src:
        scale = max(1, int(max(src.width, src.height) / 1200))
        mask = src.read(1, out_shape=(max(1, src.height // scale), max(1, src.width // scale))) == 1
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        to_mask_crs = Transformer.from_crs(target_crs, src.crs, always_xy=True) if str(src.crs) != str(target_crs) else None
    centers_plot = centers.copy()
    if to_mask_crs is not None:
        x, y = to_mask_crs.transform(centers[:, 0], centers[:, 1])
        centers_plot = np.column_stack([x, y])
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=220)
    ax.imshow(mask, extent=extent, origin="upper", cmap="Greys", alpha=0.35, interpolation="nearest")
    ax.contour(mask.astype(float), levels=[0.5], extent=extent, colors="black", linewidths=0.6)
    groups = [
        ("support-filter removed", support_drop, "#bdbdbd", "x", 24),
        ("Gram/condition removed", gram_drop | qr_drop, "#d95f02", "s", 18),
        ("active centers", active, "#1b9e77", "o", 26),
    ]
    all_ids = set(range(len(centers)))
    inactive = all_ids - active - support_drop - gram_drop - qr_drop
    groups.insert(0, ("original retained candidates", inactive, "#7570b3", ".", 12))
    for label, ids, color, marker, size in groups:
        ids = sorted(ids)
        if not ids:
            continue
        pts = centers_plot[ids]
        ax.scatter(pts[:, 0], pts[:, 1], s=size, c=color, marker=marker, label=label, edgecolors="none")
    ax.set_title("Active RBF center coverage")
    ax.set_xlabel("Longitude" if abs(extent[0]) < 180 else "x")
    ax.set_ylabel("Latitude" if abs(extent[2]) < 90 else "y")
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", default="outputs/aquifer_model_revision/comparison_common_mask.tif")
    parser.add_argument("--cache", default="outputs/cache/phase4_harmonic_blocks_d0283cfacbadc767.h5")
    parser.add_argument("--selection", default="outputs/aquifer_model_revision/rbf_global_basis_selection.json")
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    centers, scale_m, target_crs = load_centers(args.cache)
    selection = load_selection(args.selection)
    stats = nearest_distance_stats(args.mask, centers, selection["active_column_indices"], scale_m, target_crs)
    payload = {
        **stats,
        "active_center_count": int(len(selection["active_column_indices"])),
        "original_center_count": int(len(centers)),
        "support_filtered_removed_count": int(len(selection.get("support_dropped_column_indices", []))),
        "gram_condition_removed_count": int(len(selection.get("full_gram_condition_dropped_column_indices", []))),
        "rbf_scale_km": float(scale_m / 1000.0),
        "selection_mask_hash": selection.get("selection_mask_hash"),
        "note": "Coverage audit uses common mask and RBF centers only; no observations or validation targets are used.",
    }
    (output / "rbf_active_center_coverage.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_coverage(args.mask, centers, selection, output / "rbf_active_center_coverage.png", target_crs)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
