#!/usr/bin/env python
"""Generate and audit spatially covered adaptive RBF candidates.

The center placement uses only the comparison common mask geometry. It does
not read observations, residuals, or validation targets.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import xy


def _json_hash(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(text.encode("utf-8")).hexdigest()


def _safe_condition(matrix: np.ndarray) -> tuple[float, np.ndarray, int]:
    if matrix.size == 0:
        return float("inf"), np.array([], float), 0
    sv = np.linalg.svd(matrix, compute_uv=False)
    rank_tol = max(float(sv.max(initial=0.0)) * 1e-8, 1e-12)
    rank = int(np.sum(sv > rank_tol))
    cond = float("inf") if sv.size == 0 or sv[-1] <= 0 else float(sv[0] / sv[-1])
    return cond, sv, rank


def _mask_pixel_count(mask_path: Path) -> int:
    with rasterio.open(mask_path) as src:
        total = 0
        for _, window in src.block_windows(1):
            total += int(np.sum(src.read(1, window=window) == 1))
    return total


def iter_mask_xy(mask_path: Path, target_crs: str | None = None, sample_stride: int = 1):
    with rasterio.open(mask_path) as src:
        transformer = None
        if target_crs and src.crs and str(src.crs) != str(target_crs):
            transformer = Transformer.from_crs(src.crs, target_crs, always_xy=True)
        sequence_offset = 0
        for _, window in src.block_windows(1):
            mask = src.read(1, window=window) == 1
            if not mask.any():
                sequence_offset += int(mask.size)
                continue
            rows, cols = np.nonzero(mask)
            if sample_stride > 1:
                keep = ((np.arange(rows.size) + sequence_offset) % sample_stride) == 0
                rows = rows[keep]
                cols = cols[keep]
            sequence_offset += int(mask.size)
            if rows.size == 0:
                continue
            global_rows = rows + int(window.row_off)
            global_cols = cols + int(window.col_off)
            xs, ys = xy(src.transform, global_rows, global_cols, offset="center")
            xs = np.asarray(xs, float)
            ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float)
                ys = np.asarray(ys, float)
            yield xs, ys


def deterministic_mask_sample(mask_path: Path, target_crs: str | None, sample_limit: int) -> np.ndarray:
    total = _mask_pixel_count(mask_path)
    stride = max(1, int(np.ceil(total / max(sample_limit, 1))))
    xs_all = []
    ys_all = []
    for xs, ys in iter_mask_xy(mask_path, target_crs=target_crs, sample_stride=stride):
        xs_all.append(xs)
        ys_all.append(ys)
    if not xs_all:
        raise RuntimeError("comparison common mask contains no valid pixels")
    pts = np.column_stack([np.concatenate(xs_all), np.concatenate(ys_all)]).astype("float64")
    if pts.shape[0] > sample_limit:
        order = np.lexsort((pts[:, 0], pts[:, 1]))
        idx = order[np.linspace(0, len(order) - 1, sample_limit).round().astype(int)]
        pts = pts[idx]
    return pts


def farthest_point_centers(points: np.ndarray, center_count: int) -> np.ndarray:
    if points.shape[0] < center_count:
        raise ValueError(f"Only {points.shape[0]} sample points available for {center_count} centers")
    centroid = points.mean(axis=0)
    first = int(np.argmin(np.sum((points - centroid) ** 2, axis=1)))
    selected = [first]
    d2 = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, center_count):
        idx = int(np.argmax(d2))
        selected.append(idx)
        d2 = np.minimum(d2, np.sum((points - points[idx]) ** 2, axis=1))
    return points[np.asarray(selected, dtype=int)]


def pairwise_spacing(centers: np.ndarray) -> dict:
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1) / 1000.0
    return {
        "minimum_pairwise_center_distance_km": float(np.min(nearest)),
        "median_pairwise_nearest_center_distance_km": float(np.median(nearest)),
        "maximum_pairwise_nearest_center_distance_km": float(np.max(nearest)),
    }


def rbf_values(points: np.ndarray, centers: np.ndarray, sigma_m: float) -> np.ndarray:
    diff = points[:, None, :] - centers[None, :, :]
    return np.exp(-0.5 * np.sum(diff * diff, axis=2) / max(sigma_m * sigma_m, 1e-30))


def _chunk_points(xs: np.ndarray, ys: np.ndarray, chunk_size: int):
    for start in range(0, xs.size, chunk_size):
        stop = min(xs.size, start + chunk_size)
        yield np.column_stack([xs[start:stop], ys[start:stop]])


def _valid_area_km2(mask_path: Path, summary_path: Path | None = None) -> float:
    if summary_path and summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("valid_area_km2") is not None:
            return float(payload["valid_area_km2"])
    with rasterio.open(mask_path) as src:
        pixel_area_km2 = abs(src.transform.a * src.transform.e) / 1_000_000.0
        return float(_mask_pixel_count(mask_path) * pixel_area_km2)


def coverage_audit(mask_path: Path, centers: np.ndarray, sigma_km: float, target_crs: str | None, support_threshold_value: float, chunk_size: int = 100000, valid_area_km2: float | None = None) -> dict:
    nearest_values = []
    total = 0
    beyond = {1.0: 0, 1.5: 0, 2.0: 0}
    support = np.zeros(len(centers), dtype=np.int64)
    assignments = np.zeros(len(centers), dtype=np.int64)
    sigma_m = float(sigma_km * 1000.0)
    for xs, ys in iter_mask_xy(mask_path, target_crs=target_crs, sample_stride=1):
        for pts in _chunk_points(xs, ys, chunk_size):
            diff = pts[:, None, :] - centers[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            nearest_d = np.sqrt(np.min(d2, axis=1))
            nearest_idx = np.argmin(d2, axis=1)
            total += int(pts.shape[0])
            nearest_values.append(nearest_d)
            for mult in beyond:
                beyond[mult] += int(np.sum(nearest_d > mult * sigma_m))
            phi = np.exp(-0.5 * d2 / max(sigma_m * sigma_m, 1e-30))
            support += np.sum(phi > support_threshold_value, axis=0)
            assignments += np.bincount(nearest_idx, minlength=len(centers))
    nearest = np.concatenate(nearest_values) / 1000.0
    support_fraction = support / max(total, 1)
    assigned_fraction = assignments / max(total, 1)
    if valid_area_km2 is None:
        valid_area_km2 = _valid_area_km2(mask_path)
    return {
        "nearest_center_min_km": float(np.min(nearest)),
        "nearest_center_median_km": float(np.median(nearest)),
        "nearest_center_p90_km": float(np.percentile(nearest, 90)),
        "nearest_center_p95_km": float(np.percentile(nearest, 95)),
        "nearest_center_max_km": float(np.max(nearest)),
        "area_fraction_beyond_1_sigma": float(beyond[1.0] / max(total, 1)),
        "area_fraction_beyond_1_5_sigma": float(beyond[1.5] / max(total, 1)),
        "area_fraction_beyond_2_sigma": float(beyond[2.0] / max(total, 1)),
        "center_support_fraction": support_fraction.tolist(),
        "minimum_support_fraction": float(np.min(support_fraction)),
        "maximum_support_fraction": float(np.max(support_fraction)),
        "support_cv": float(np.std(support_fraction) / max(float(np.mean(support_fraction)), 1e-30)),
        "assigned_pixel_fraction": assigned_fraction.tolist(),
        "mean_voronoi_cell_area_km2": float(valid_area_km2 / max(len(centers), 1)),
        "effective_spacing_km": float(np.sqrt(valid_area_km2 / max(len(centers), 1))),
        "valid_pixel_count": int(total),
    }


def gram_conditioning(mask_path: Path, block_path: Path, centers: np.ndarray, sigma_km: float, target_crs: str | None, fold_id: int = 0, chunk_size: int = 100000) -> dict:
    sigma_m = float(sigma_km * 1000.0)
    n = len(centers)
    global_gram = np.zeros((n, n), float)
    train_sum = np.zeros(n, float)
    train2_sum = np.zeros(n, float)
    train_gram = np.zeros((n, n), float)
    total = 0
    train_total = 0
    with rasterio.open(mask_path) as mask_src, rasterio.open(block_path) as block_src:
        transformer = None
        if target_crs and mask_src.crs and str(mask_src.crs) != str(target_crs):
            transformer = Transformer.from_crs(mask_src.crs, target_crs, always_xy=True)
        for _, window in mask_src.block_windows(1):
            mask = mask_src.read(1, window=window) == 1
            if not mask.any():
                continue
            folds = block_src.read(1, window=window)
            rows, cols = np.nonzero(mask)
            global_rows = rows + int(window.row_off)
            global_cols = cols + int(window.col_off)
            xs, ys = xy(mask_src.transform, global_rows, global_cols, offset="center")
            xs = np.asarray(xs, float)
            ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float)
                ys = np.asarray(ys, float)
            train = folds[rows, cols] != fold_id
            for start in range(0, xs.size, chunk_size):
                stop = min(xs.size, start + chunk_size)
                pts = np.column_stack([xs[start:stop], ys[start:stop]])
                phi = rbf_values(pts, centers, sigma_m)
                global_gram += phi.T @ phi
                total += int(phi.shape[0])
                tr = train[start:stop]
                if np.any(tr):
                    ptr = phi[tr]
                    train_sum += ptr.sum(axis=0)
                    train2_sum += np.sum(ptr * ptr, axis=0)
                    train_gram += ptr.T @ ptr
                    train_total += int(ptr.shape[0])
    global_gram /= max(total, 1)
    cond_global, sv_global, rank_global = _safe_condition(global_gram)
    mean = train_sum / max(train_total, 1)
    centered_gram = train_gram / max(train_total, 1) - np.outer(mean, mean)
    scale = np.sqrt(np.maximum(np.diag(centered_gram), 1e-30))
    scaled_gram = centered_gram / np.outer(scale, scale)
    cond_fold, sv_fold, rank_fold = _safe_condition(scaled_gram)
    corr = scaled_gram.copy()
    np.fill_diagonal(corr, 0.0)
    return {
        "global_gram_condition_number": cond_global,
        "global_effective_rank": rank_global,
        "global_singular_values": sv_global.tolist(),
        "fold0_training_pixel_count": int(train_total),
        "fold0_condition_number": cond_fold,
        "fold0_effective_rank": rank_fold,
        "fold0_singular_values": sv_fold.tolist(),
        "fold0_maximum_correlation": float(np.nanmax(np.abs(corr))) if corr.size else 0.0,
        "fold0_centered_scaled": True,
        "fold0_residualized_against": ["intercept"],
    }


def audit_center_set(mask_path: Path, block_path: Path, centers: np.ndarray, sigma_values_km: list[float], target_crs: str | None, support_threshold_value: float, fold_id: int = 0, chunk_size: int = 100000, valid_area_km2: float | None = None) -> dict[float, dict]:
    """Audit one center set for several sigma values with shared raster scans."""
    sigma_values_km = [float(x) for x in sigma_values_km]
    sigma_values_m = [x * 1000.0 for x in sigma_values_km]
    n = len(centers)
    coverage = {
        sigma: {
            "nearest": [],
            "beyond": {1.0: 0, 1.5: 0, 2.0: 0},
            "support": np.zeros(n, dtype=np.int64),
            "assignments": np.zeros(n, dtype=np.int64),
            "total": 0,
        }
        for sigma in sigma_values_km
    }
    for xs, ys in iter_mask_xy(mask_path, target_crs=target_crs, sample_stride=1):
        for pts in _chunk_points(xs, ys, chunk_size):
            diff = pts[:, None, :] - centers[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            nearest_d = np.sqrt(np.min(d2, axis=1))
            nearest_idx = np.argmin(d2, axis=1)
            for sigma_km, sigma_m in zip(sigma_values_km, sigma_values_m):
                item = coverage[sigma_km]
                item["total"] += int(pts.shape[0])
                item["nearest"].append(nearest_d)
                for mult in item["beyond"]:
                    item["beyond"][mult] += int(np.sum(nearest_d > mult * sigma_m))
                phi = np.exp(-0.5 * d2 / max(sigma_m * sigma_m, 1e-30))
                item["support"] += np.sum(phi > support_threshold_value, axis=0)
                item["assignments"] += np.bincount(nearest_idx, minlength=n)

    if valid_area_km2 is None:
        valid_area_km2 = _valid_area_km2(mask_path)
    coverage_metrics = {}
    for sigma_km in sigma_values_km:
        item = coverage[sigma_km]
        total = int(item["total"])
        nearest = np.concatenate(item["nearest"]) / 1000.0
        support_fraction = item["support"] / max(total, 1)
        assigned_fraction = item["assignments"] / max(total, 1)
        coverage_metrics[sigma_km] = {
            "nearest_center_min_km": float(np.min(nearest)),
            "nearest_center_median_km": float(np.median(nearest)),
            "nearest_center_p90_km": float(np.percentile(nearest, 90)),
            "nearest_center_p95_km": float(np.percentile(nearest, 95)),
            "nearest_center_max_km": float(np.max(nearest)),
            "area_fraction_beyond_1_sigma": float(item["beyond"][1.0] / max(total, 1)),
            "area_fraction_beyond_1_5_sigma": float(item["beyond"][1.5] / max(total, 1)),
            "area_fraction_beyond_2_sigma": float(item["beyond"][2.0] / max(total, 1)),
            "center_support_fraction": support_fraction.tolist(),
            "minimum_support_fraction": float(np.min(support_fraction)),
            "maximum_support_fraction": float(np.max(support_fraction)),
            "support_cv": float(np.std(support_fraction) / max(float(np.mean(support_fraction)), 1e-30)),
            "assigned_pixel_fraction": assigned_fraction.tolist(),
            "mean_voronoi_cell_area_km2": float(valid_area_km2 / max(n, 1)),
            "effective_spacing_km": float(np.sqrt(valid_area_km2 / max(n, 1))),
            "valid_pixel_count": total,
        }

    grams = {
        sigma: {
            "global_gram": np.zeros((n, n), float),
            "train_sum": np.zeros(n, float),
            "train_gram": np.zeros((n, n), float),
            "total": 0,
            "train_total": 0,
        }
        for sigma in sigma_values_km
    }
    with rasterio.open(mask_path) as mask_src, rasterio.open(block_path) as block_src:
        transformer = None
        if target_crs and mask_src.crs and str(mask_src.crs) != str(target_crs):
            transformer = Transformer.from_crs(mask_src.crs, target_crs, always_xy=True)
        for _, window in mask_src.block_windows(1):
            mask = mask_src.read(1, window=window) == 1
            if not mask.any():
                continue
            folds = block_src.read(1, window=window)
            rows, cols = np.nonzero(mask)
            global_rows = rows + int(window.row_off)
            global_cols = cols + int(window.col_off)
            xs, ys = xy(mask_src.transform, global_rows, global_cols, offset="center")
            xs = np.asarray(xs, float)
            ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float)
                ys = np.asarray(ys, float)
            train = folds[rows, cols] != fold_id
            for start in range(0, xs.size, chunk_size):
                stop = min(xs.size, start + chunk_size)
                pts = np.column_stack([xs[start:stop], ys[start:stop]])
                diff = pts[:, None, :] - centers[None, :, :]
                d2 = np.sum(diff * diff, axis=2)
                tr = train[start:stop]
                for sigma_km, sigma_m in zip(sigma_values_km, sigma_values_m):
                    item = grams[sigma_km]
                    phi = np.exp(-0.5 * d2 / max(sigma_m * sigma_m, 1e-30))
                    item["global_gram"] += phi.T @ phi
                    item["total"] += int(phi.shape[0])
                    if np.any(tr):
                        ptr = phi[tr]
                        item["train_sum"] += ptr.sum(axis=0)
                        item["train_gram"] += ptr.T @ ptr
                        item["train_total"] += int(ptr.shape[0])

    out = {}
    for sigma_km in sigma_values_km:
        item = grams[sigma_km]
        global_gram = item["global_gram"] / max(item["total"], 1)
        cond_global, sv_global, rank_global = _safe_condition(global_gram)
        mean = item["train_sum"] / max(item["train_total"], 1)
        centered_gram = item["train_gram"] / max(item["train_total"], 1) - np.outer(mean, mean)
        scale = np.sqrt(np.maximum(np.diag(centered_gram), 1e-30))
        scaled_gram = centered_gram / np.outer(scale, scale)
        cond_fold, sv_fold, rank_fold = _safe_condition(scaled_gram)
        corr = scaled_gram.copy()
        np.fill_diagonal(corr, 0.0)
        out[sigma_km] = {
            **coverage_metrics[sigma_km],
            "global_gram_condition_number": cond_global,
            "global_effective_rank": rank_global,
            "global_singular_values": sv_global.tolist(),
            "fold0_training_pixel_count": int(item["train_total"]),
            "fold0_condition_number": cond_fold,
            "fold0_effective_rank": rank_fold,
            "fold0_singular_values": sv_fold.tolist(),
            "fold0_maximum_correlation": float(np.nanmax(np.abs(corr))) if corr.size else 0.0,
            "fold0_centered_scaled": True,
            "fold0_residualized_against": ["intercept"],
        }
    return out


def write_centers(output_dir: Path, label: str, centers: np.ndarray, crs: str | None):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"center_id": i, "x": float(x), "y": float(y)} for i, (x, y) in enumerate(centers)]
    pd.DataFrame(rows).to_csv(output_dir / f"{label}_centers.csv", index=False)
    features = [
        {
            "type": "Feature",
            "properties": {"center_id": int(i), "crs": crs},
            "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
        }
        for i, (x, y) in enumerate(centers)
    ]
    (output_dir / f"{label}_centers.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def select_design(rows: list[dict], coverage_support_threshold: float) -> dict | None:
    passed = []
    for row in rows:
        coverage_ok = (
            bool(row["centers_inside_mask_fraction"] == 1.0)
            and row["nearest_center_p95_km"] <= 1.5 * row["rbf_sigma_km"]
            and row["nearest_center_max_km"] <= 2.5 * row["rbf_sigma_km"]
            and row["area_fraction_beyond_2_sigma"] < 0.05
            and row["minimum_support_fraction"] > coverage_support_threshold
        )
        conditioning_ok = (
            row["fold0_effective_rank"] == row["center_count"]
            and np.isfinite(row["fold0_condition_number"])
            and row["fold0_condition_number"] < 1e4
            and row["fold0_maximum_correlation"] < 0.98
        )
        row["coverage_passed"] = bool(coverage_ok)
        row["conditioning_passed"] = bool(conditioning_ok)
        row["candidate_passed"] = bool(coverage_ok and conditioning_ok)
        if row["candidate_passed"]:
            passed.append(row)
    if not passed:
        return None
    return sorted(
        passed,
        key=lambda r: (
            r["center_count"],
            r["area_fraction_beyond_2_sigma"],
            r["nearest_center_p95_km"],
            r["fold0_condition_number"],
        ),
    )[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", default="outputs/aquifer_model_revision/comparison_common_mask.tif")
    parser.add_argument("--blocks", default="outputs/aquifer_model_revision/spatial_validation_blocks.tif")
    parser.add_argument("--mask-summary", default="outputs/aquifer_model_revision/comparison_common_mask_summary.json")
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    parser.add_argument("--projected-crs", default="EPSG:32650")
    parser.add_argument("--center-counts", default="32,48,64")
    parser.add_argument("--sigma-multipliers", default="1.0,1.5,2.0")
    parser.add_argument("--sample-limit", type=int, default=250000)
    parser.add_argument("--support-threshold-value", type=float, default=1e-12)
    parser.add_argument("--minimum-support-fraction", type=float, default=0.001)
    args = parser.parse_args()

    mask = Path(args.mask)
    blocks = Path(args.blocks)
    output_root = Path(args.output_root)
    center_dir = output_root / "rbf_candidate_centers"
    counts = [int(x) for x in args.center_counts.split(",") if x.strip()]
    multipliers = [float(x) for x in args.sigma_multipliers.split(",") if x.strip()]
    valid_area_km2 = _valid_area_km2(mask, Path(args.mask_summary))

    sample = deterministic_mask_sample(mask, args.projected_crs, args.sample_limit)
    all_rows: list[dict] = []
    centers_by_label: dict[str, np.ndarray] = {}
    for count in counts:
        label = f"R{count}"
        centers = farthest_point_centers(sample, count)
        centers_by_label[label] = centers
        write_centers(center_dir, label, centers, args.projected_crs)
        spacing = pairwise_spacing(centers)
        median_spacing = spacing["median_pairwise_nearest_center_distance_km"]
        sigma_values = [float(median_spacing * mult) for mult in multipliers]
        combined_audit = audit_center_set(
            mask,
            blocks,
            centers,
            sigma_values,
            args.projected_crs,
            args.support_threshold_value,
            valid_area_km2=valid_area_km2,
        )
        for mult in multipliers:
            sigma_km = float(median_spacing * mult)
            audit = combined_audit[sigma_km]
            row = {
                "candidate_id": f"{label}_sigma{mult:g}",
                "center_count": int(count),
                "sigma_multiplier": float(mult),
                "requested_spacing_km": None,
                "spacing_selection_status": "candidate_comparison",
                "rbf_sigma_km": sigma_km,
                "centers_inside_mask_fraction": 1.0,
                **spacing,
                **{k: v for k, v in audit.items() if not k.endswith("_singular_values")},
            }
            row["design_hash"] = _json_hash({
                "centers": np.round(centers, 6).tolist(),
                "sigma_km": round(sigma_km, 9),
                "projected_crs": args.projected_crs,
                "candidate_id": row["candidate_id"],
            })
            all_rows.append(row)

    selected = select_design(all_rows, args.minimum_support_fraction)
    df = pd.DataFrame(all_rows)
    df.to_csv(output_root / "rbf_candidate_coverage_comparison.csv", index=False)
    df.to_csv(output_root / "rbf_candidate_conditioning_comparison.csv", index=False)
    payload = {
        "candidate_generation": {
            "method": "deterministic_farthest_point_sampling_from_common_mask",
            "uses_observations_or_residuals": False,
            "sample_limit": int(args.sample_limit),
            "projected_crs": args.projected_crs,
        },
        "minimum_support_fraction_threshold": float(args.minimum_support_fraction),
        "candidates": all_rows,
    }
    (output_root / "rbf_candidate_coverage_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if selected is not None:
        label = f"R{selected['center_count']}"
        centers = centers_by_label[label]
        selected_payload = {
            "rbf_basis_status": "adaptive_candidate_selected_pending_model_refit",
            "center_count": int(selected["center_count"]),
            "center_coordinates": centers.tolist(),
            "projected_crs": args.projected_crs,
            "effective_spacing_km": float(selected["effective_spacing_km"]),
            "median_pairwise_nearest_center_distance_km": float(selected["median_pairwise_nearest_center_distance_km"]),
            "sigma_km": float(selected["rbf_sigma_km"]),
            "requested_spacing_km": None,
            "spacing_selection_status": "candidate_comparison",
            "selection_reason": "first candidate passing coverage and fold0 conditioning, preferring fewer centers",
            "coverage_metrics": {k: selected[k] for k in selected if k.startswith("nearest_") or k.startswith("area_fraction") or k in {"minimum_support_fraction", "maximum_support_fraction", "support_cv"}},
            "conditioning_metrics": {k: selected[k] for k in selected if k.startswith("fold0_") or k.startswith("global_")},
            "design_hash": selected["design_hash"],
            "source": "scripts/generate_adaptive_rbf_centers.py",
        }
        (output_root / "selected_rbf_design.json").write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
        print(json.dumps({"selected": selected["candidate_id"], "design_hash": selected["design_hash"]}, indent=2), flush=True)
    else:
        print(json.dumps({"selected": None, "reason": "no candidate passed coverage and conditioning gates"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
