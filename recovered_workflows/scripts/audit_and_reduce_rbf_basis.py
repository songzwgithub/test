#!/usr/bin/env python
"""Audit and select a global full-rank RBF basis without using observations."""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from scipy.linalg import qr
from pyproj import Transformer


def load_rbf_metadata(cache_path):
    with h5py.File(cache_path, "r") as h5:
        meta = json.loads(h5.attrs.get("design_metadata", "{}"))
    spatial = meta.get("spatial_basis", {})
    centers = np.asarray(spatial.get("centers", []), dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("RBF centers are missing or invalid in design_metadata")
    scale_m = float(spatial.get("scale_m"))
    names = spatial.get("names") or [f"spatial_rbf_{i:02d}" for i in range(len(centers))]
    return centers, scale_m, names


def rbf_values(x, y, centers, scale_m):
    points = np.column_stack([x, y]).astype(float)
    diff = points[:, None, :] - centers[None, :, :]
    return np.exp(-0.5 * np.sum(diff * diff, axis=2) / (scale_m**2))


def iter_mask_blocks(mask_path, target_crs=None):
    with rasterio.open(mask_path) as src:
        transformer = None
        if target_crs and src.crs and str(src.crs) != str(target_crs):
            transformer = Transformer.from_crs(src.crs, target_crs, always_xy=True)
        for _, window in src.block_windows(1):
            mask = src.read(1, window=window) == 1
            if not mask.any():
                continue
            rows, cols = np.nonzero(mask)
            global_rows = rows + int(window.row_off)
            global_cols = cols + int(window.col_off)
            xs, ys = xy(src.transform, global_rows, global_cols, offset="center")
            xs = np.asarray(xs, float)
            ys = np.asarray(ys, float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, float)
                ys = np.asarray(ys, float)
            yield xs, ys, int(mask.sum()), src


def support_audit(mask_path, centers, scale_m, target_crs=None, support_value_threshold=1e-12):
    n = len(centers)
    count = 0
    support = np.zeros(n, dtype=np.int64)
    sums = np.zeros(n, float)
    sums2 = np.zeros(n, float)
    maxv = np.zeros(n, float)
    minx = miny = np.inf
    maxx = maxy = -np.inf
    for xs, ys, block_count, _src in iter_mask_blocks(mask_path, target_crs=target_crs):
        phi = rbf_values(xs, ys, centers, scale_m)
        count += block_count
        support += np.sum(phi > support_value_threshold, axis=0)
        sums += phi.sum(axis=0)
        sums2 += np.sum(phi * phi, axis=0)
        maxv = np.maximum(maxv, phi.max(axis=0))
        minx = min(minx, float(xs.min())); maxx = max(maxx, float(xs.max()))
        miny = min(miny, float(ys.min())); maxy = max(maxy, float(ys.max()))
    if count == 0:
        raise RuntimeError("comparison_common_mask contains no valid pixels")
    mean = sums / count
    rms = np.sqrt(np.maximum(sums2 / count, 0.0))
    std = np.sqrt(np.maximum(sums2 / count - mean * mean, 0.0))
    dx = np.maximum(np.maximum(minx - centers[:, 0], centers[:, 0] - maxx), 0.0)
    dy = np.maximum(np.maximum(miny - centers[:, 1], centers[:, 1] - maxy), 0.0)
    dist_km = np.sqrt(dx * dx + dy * dy) / 1000.0
    return {
        "valid_pixel_count": int(count),
        "support_pixel_count": support,
        "support_fraction": support / count,
        "column_mean": mean,
        "column_std": std,
        "column_rms": rms,
        "column_max": maxv,
        "distance_to_study_area_km": dist_km,
    }


def deterministic_spatial_sample(mask_path, centers, scale_m, target_crs=None, sample_limit=250000):
    xs_all = []
    ys_all = []
    total = 0
    for xs, ys, block_count, _src in iter_mask_blocks(mask_path, target_crs=target_crs):
        xs_all.append(xs)
        ys_all.append(ys)
        total += block_count
    xs = np.concatenate(xs_all)
    ys = np.concatenate(ys_all)
    if len(xs) > sample_limit:
        order = np.lexsort((xs, ys))
        idx = order[np.linspace(0, len(order) - 1, sample_limit).round().astype(int)]
        xs = xs[idx]
        ys = ys[idx]
    return rbf_values(xs, ys, centers, scale_m), int(total)


def gram_for_columns(mask_path, centers, scale_m, columns, target_crs=None):
    columns = np.asarray(columns, dtype=int)
    gram = np.zeros((len(columns), len(columns)), float)
    count = 0
    for xs, ys, block_count, _src in iter_mask_blocks(mask_path, target_crs=target_crs):
        phi = rbf_values(xs, ys, centers[columns], scale_m)
        gram += phi.T @ phi
        count += block_count
    return gram / max(count, 1), int(count)


def gram_condition(gram):
    if gram.size == 0:
        return np.inf, np.array([])
    sv = np.linalg.svd(gram, compute_uv=False)
    cond = float(np.inf if sv[-1] <= 0 else sv[0] / sv[-1])
    return cond, sv


def choose_full_gram_stable_prefix(mask_path, centers, scale_m, pivot_order, rank_tolerance, max_condition=1e4, target_crs=None):
    best = []
    best_cond = np.inf
    best_sv = np.array([])
    for k in range(len(pivot_order), 0, -1):
        cols = pivot_order[:k]
        gram, _ = gram_for_columns(mask_path, centers, scale_m, cols, target_crs=target_crs)
        cond, sv = gram_condition(gram)
        min_rel = float(sv[-1] / max(sv[0], 1e-30)) if sv.size else 0.0
        if np.isfinite(cond) and cond < max_condition and min_rel > rank_tolerance:
            return cols, cond, sv
        if cond < best_cond:
            best = cols
            best_cond = cond
            best_sv = sv
    return best, best_cond, best_sv


def select_active_columns(phi_sample, candidate_indices, rank_tolerance=1e-8):
    if phi_sample.size == 0 or len(candidate_indices) == 0:
        return [], [], [], []
    _q, r, piv = qr(phi_sample, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    threshold = float(rank_tolerance * max(diag.max(initial=0.0), 1.0))
    rank = int(np.sum(diag > threshold))
    pivot_order = [int(candidate_indices[i]) for i in piv.tolist()]
    active = pivot_order[:rank]
    dropped = pivot_order[rank:]
    return active, dropped, pivot_order, diag.tolist()


def selection_hash(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(text.encode("utf-8")).hexdigest()


def audit_and_reduce(
    mask_path,
    cache_path,
    output_root,
    support_threshold=0.001,
    std_threshold=1e-8,
    rms_threshold=1e-8,
    rank_tolerance=1e-8,
    sample_limit=250000,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    centers, scale_m, names = load_rbf_metadata(cache_path)
    _centers2, _scale2, _names2 = centers, scale_m, names
    with h5py.File(cache_path, "r") as h5:
        target_crs = json.loads(h5.attrs.get("design_metadata", "{}")).get("spatial_basis", {}).get("projected_crs")
    audit = support_audit(mask_path, centers, scale_m, target_crs=target_crs)
    status = []
    keep = []
    for i in range(len(centers)):
        reasons = []
        if audit["support_fraction"][i] < support_threshold:
            reasons.append("low_support")
        if audit["column_std"][i] < std_threshold:
            reasons.append("near_zero_std")
        if audit["column_rms"][i] < rms_threshold:
            reasons.append("near_zero_rms")
        status.append("active_candidate" if not reasons else "drop_" + "+".join(reasons))
        if not reasons:
            keep.append(i)
    rows = []
    for i, (x, y) in enumerate(centers):
        rows.append({
            "center_id": int(i),
            "center_name": names[i],
            "center_x": float(x),
            "center_y": float(y),
            "support_pixel_count": int(audit["support_pixel_count"][i]),
            "support_fraction": float(audit["support_fraction"][i]),
            "column_mean": float(audit["column_mean"][i]),
            "column_std": float(audit["column_std"][i]),
            "column_rms": float(audit["column_rms"][i]),
            "column_max": float(audit["column_max"][i]),
            "distance_to_study_area_km": float(audit["distance_to_study_area_km"][i]),
            "status": status[i],
        })
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_root / "rbf_basis_audit.csv", index=False)
    phi_sample, total_pixels = deterministic_spatial_sample(mask_path, centers[keep], scale_m, target_crs=target_crs, sample_limit=sample_limit)
    active_rel, qr_dropped_rel, pivot_rel, rdiag = select_active_columns(phi_sample, np.arange(len(keep)), rank_tolerance)
    pivot_order = [int(keep[i]) for i in pivot_rel]
    qr_active = [int(keep[i]) for i in active_rel]
    qr_dropped = [int(keep[i]) for i in qr_dropped_rel]
    active, stable_cond, stable_sv = choose_full_gram_stable_prefix(
        mask_path,
        centers,
        scale_m,
        qr_active,
        rank_tolerance,
        max_condition=1e4,
        target_crs=target_crs,
    )
    full_gram_dropped = [int(i) for i in qr_active if i not in active]
    support_dropped = [int(i) for i in range(len(centers)) if i not in keep]
    dropped = sorted(set(support_dropped + qr_dropped + full_gram_dropped))
    gram_before, _ = gram_for_columns(mask_path, centers, scale_m, keep, target_crs=target_crs)
    gram_after, _ = gram_for_columns(mask_path, centers, scale_m, active, target_crs=target_crs)
    cond_before, sv_before = gram_condition(gram_before)
    cond_after, sv_after = gram_condition(gram_after)
    selection_payload = {
        "requested_centers": int(len(centers)),
        "initial_center_count": int(len(centers)),
        "support_filtered_count": int(len(keep)),
        "effective_rank_before": int(np.sum(sv_before > max(sv_before.max(initial=0.0) * rank_tolerance, 1e-12))) if sv_before.size else 0,
        "selected_active_count": int(len(active)),
        "active_column_indices": active,
        "dropped_column_indices": dropped,
        "support_dropped_column_indices": support_dropped,
        "qr_dropped_column_indices": qr_dropped,
        "full_gram_condition_dropped_column_indices": full_gram_dropped,
        "pivot_order": pivot_order,
        "rank_tolerance": float(rank_tolerance),
        "singular_values_or_R_diagonal": rdiag,
        "full_gram_singular_values_before": sv_before.tolist(),
        "full_gram_singular_values_after": sv_after.tolist(),
        "condition_number_before": cond_before,
        "condition_number_after": cond_after,
        "support_threshold": float(support_threshold),
        "std_threshold": float(std_threshold),
        "rms_threshold": float(rms_threshold),
        "sample_limit": int(sample_limit),
        "total_common_mask_pixels": int(total_pixels),
        "rbf_projected_crs": target_crs,
    }
    selection_payload["selection_mask_hash"] = selection_hash(selection_payload)
    (output_root / "rbf_basis_audit.json").write_text(json.dumps({"columns": rows, **selection_payload}, indent=2), encoding="utf-8")
    (output_root / "rbf_global_basis_selection.json").write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
    audit_df[audit_df["center_id"].isin(active)].to_csv(output_root / "rbf_active_centers.csv", index=False)
    audit_df[audit_df["center_id"].isin(dropped)].to_csv(output_root / "rbf_dropped_centers.csv", index=False)
    return selection_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", default="outputs/aquifer_model_revision/comparison_common_mask.tif")
    parser.add_argument("--cache", default="outputs/cache/phase4_harmonic_blocks_d0283cfacbadc767.h5")
    parser.add_argument("--output-root", default="outputs/aquifer_model_revision")
    parser.add_argument("--support-threshold", type=float, default=0.001)
    parser.add_argument("--rank-tolerance", type=float, default=1e-8)
    parser.add_argument("--sample-limit", type=int, default=250000)
    args = parser.parse_args()
    result = audit_and_reduce(
        args.mask,
        args.cache,
        args.output_root,
        support_threshold=args.support_threshold,
        rank_tolerance=args.rank_tolerance,
        sample_limit=args.sample_limit,
    )
    print(json.dumps({
        "initial_center_count": result["initial_center_count"],
        "support_filtered_count": result["support_filtered_count"],
        "selected_active_count": result["selected_active_count"],
        "condition_number_before": result["condition_number_before"],
        "condition_number_after": result["condition_number_after"],
        "selection_mask_hash": result["selection_mask_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()
