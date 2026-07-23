#!/usr/bin/env python
"""Build a fixed weighted-orthogonal RBF basis from an adaptive raw candidate."""
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


ORTHO_VERSION = "weighted_global_eigendecomposition_v1"


def sha_array(arr: np.ndarray) -> str:
    return sha256(np.asarray(arr, dtype="float64").tobytes()).hexdigest()


def sha_json(payload) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def rbf_values(points: np.ndarray, centers: np.ndarray, sigma_m: float) -> np.ndarray:
    diff = points[:, None, :] - centers[None, :, :]
    return np.exp(-0.5 * np.sum(diff * diff, axis=2) / max(sigma_m * sigma_m, 1e-30))


def iter_mask_points(mask_path: Path, target_crs: str | None, block_path: Path | None = None):
    with rasterio.open(mask_path) as mask_src:
        block_src = rasterio.open(block_path) if block_path else None
        try:
            transformer = None
            if target_crs and mask_src.crs and str(mask_src.crs) != str(target_crs):
                transformer = Transformer.from_crs(mask_src.crs, target_crs, always_xy=True)
            for _, window in mask_src.block_windows(1):
                mask = mask_src.read(1, window=window) == 1
                if not mask.any():
                    continue
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
                folds = None
                if block_src is not None:
                    folds = block_src.read(1, window=window)[rows, cols]
                yield np.column_stack([xs, ys]), folds
        finally:
            if block_src is not None:
                block_src.close()


def weighted_gram(mask_path: Path, centers: np.ndarray, sigma_m: float, target_crs: str | None, block_path: Path | None = None, fold_id: int | None = None):
    n = len(centers)
    gram = np.zeros((n, n), float)
    count = 0
    for pts, folds in iter_mask_points(mask_path, target_crs, block_path=block_path):
        if fold_id is not None:
            if folds is None:
                raise ValueError("fold_id requires block_path")
            keep = folds != fold_id
            if not np.any(keep):
                continue
            pts = pts[keep]
        phi = rbf_values(pts, centers, sigma_m)
        gram += phi.T @ phi
        count += int(phi.shape[0])
    if count == 0:
        raise RuntimeError("No pixels available for weighted Gram")
    return gram / count, count


def audit_basis(mask_path: Path, centers: np.ndarray, sigma_m: float, transform: np.ndarray, target_crs: str | None, block_path: Path | None = None, fold_id: int | None = None):
    k = transform.shape[1]
    gram = np.zeros((k, k), float)
    sums = np.zeros(k, float)
    sums2 = np.zeros(k, float)
    count = 0
    for pts, folds in iter_mask_points(mask_path, target_crs, block_path=block_path):
        if fold_id is not None:
            keep = folds != fold_id
            if not np.any(keep):
                continue
            pts = pts[keep]
        phi = rbf_values(pts, centers, sigma_m)
        b = phi @ transform
        gram += b.T @ b
        sums += b.sum(axis=0)
        sums2 += np.sum(b * b, axis=0)
        count += int(b.shape[0])
    gram /= max(count, 1)
    sv = np.linalg.svd(gram, compute_uv=False)
    max_sv = float(np.max(sv)) if sv.size else 0.0
    min_sv = float(np.min(sv)) if sv.size else 0.0
    rank = int(np.sum(sv > max(max_sv * 1e-8, 1e-12)))
    cond = float("inf") if min_sv <= 0 else float(max_sv / min_sv)
    mean = sums / max(count, 1)
    var = np.maximum(sums2 / max(count, 1) - mean * mean, 1e-30)
    corr = (gram - np.outer(mean, mean)) / np.outer(np.sqrt(var), np.sqrt(var))
    np.fill_diagonal(corr, 0.0)
    identity = np.eye(k)
    return {
        "pixel_count": int(count),
        "basis_count": int(k),
        "effective_rank": rank,
        "condition_number": cond,
        "minimum_singular_value": min_sv,
        "maximum_singular_value": max_sv,
        "orthogonality_error": float(np.linalg.norm(gram - identity, ord="fro") / max(np.sqrt(k), 1.0)),
        "maximum_column_correlation": float(np.nanmax(np.abs(corr))) if corr.size else 0.0,
        "singular_values": sv.tolist(),
    }


def _audit_from_moments(gram_sum: np.ndarray, sums: np.ndarray, sums2: np.ndarray, count: int, k: int):
    gram = gram_sum[:k, :k] / max(count, 1)
    mean = sums[:k] / max(count, 1)
    var = np.maximum(sums2[:k] / max(count, 1) - mean * mean, 1e-30)
    sv = np.linalg.svd(gram, compute_uv=False)
    max_sv = float(np.max(sv)) if sv.size else 0.0
    min_sv = float(np.min(sv)) if sv.size else 0.0
    rank = int(np.sum(sv > max(max_sv * 1e-8, 1e-12)))
    cond = float("inf") if min_sv <= 0 else float(max_sv / min_sv)
    corr = (gram - np.outer(mean, mean)) / np.outer(np.sqrt(var), np.sqrt(var))
    np.fill_diagonal(corr, 0.0)
    return {
        "pixel_count": int(count),
        "basis_count": int(k),
        "effective_rank": rank,
        "condition_number": cond,
        "minimum_singular_value": min_sv,
        "maximum_singular_value": max_sv,
        "orthogonality_error": float(np.linalg.norm(gram - np.eye(k), ord="fro") / max(np.sqrt(k), 1.0)),
        "maximum_column_correlation": float(np.nanmax(np.abs(corr))) if corr.size else 0.0,
        "singular_values": sv.tolist(),
    }


def audit_transform_prefixes(mask_path: Path, centers: np.ndarray, sigma_m: float, transform_full: np.ndarray, target_crs: str | None, counts: list[int], block_path: Path, fold_id: int = 0):
    max_k = int(max(counts))
    transform_full = transform_full[:, :max_k]
    global_gram = np.zeros((max_k, max_k), float)
    global_sum = np.zeros(max_k, float)
    global_sum2 = np.zeros(max_k, float)
    global_count = 0
    fold_gram = np.zeros((max_k, max_k), float)
    fold_sum = np.zeros(max_k, float)
    fold_sum2 = np.zeros(max_k, float)
    fold_count = 0
    for pts, folds in iter_mask_points(mask_path, target_crs, block_path=block_path):
        phi = rbf_values(pts, centers, sigma_m)
        b = phi @ transform_full
        global_gram += b.T @ b
        global_sum += b.sum(axis=0)
        global_sum2 += np.sum(b * b, axis=0)
        global_count += int(b.shape[0])
        keep = folds != fold_id
        if np.any(keep):
            bf = b[keep]
            fold_gram += bf.T @ bf
            fold_sum += bf.sum(axis=0)
            fold_sum2 += np.sum(bf * bf, axis=0)
            fold_count += int(bf.shape[0])
    return {
        int(k): {
            "global": _audit_from_moments(global_gram, global_sum, global_sum2, global_count, int(k)),
            "fold0": _audit_from_moments(fold_gram, fold_sum, fold_sum2, fold_count, int(k)),
        }
        for k in counts
    }


def build_basis(candidate_path: Path, mask_path: Path, block_path: Path, output_dir: Path):
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    centers = np.asarray(candidate["center_coordinates"], float)
    sigma_m = float(candidate["sigma_km"]) * 1000.0
    target_crs = candidate.get("projected_crs")
    output_dir.mkdir(parents=True, exist_ok=True)

    gram, count = weighted_gram(mask_path, centers, sigma_m, target_crs)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    lam_max = float(eigenvalues[0])
    keep_all = np.where(eigenvalues / max(lam_max, 1e-30) >= 1e-8)[0]
    max_count = int(len(keep_all))
    candidate_counts = [x for x in [32, 30, 28, 26, 24] if x <= max_count]
    eig_rows = []
    total_energy = float(np.sum(np.maximum(eigenvalues, 0.0)))
    cumulative = 0.0
    for i, lam in enumerate(eigenvalues):
        cumulative += float(max(lam, 0.0))
        eig_rows.append({
            "direction": i,
            "eigenvalue": float(lam),
            "relative_eigenvalue": float(lam / max(lam_max, 1e-30)),
            "cumulative_energy_ratio": float(cumulative / max(total_energy, 1e-30)),
            "retained_by_1e_minus_8": bool(i in keep_all),
        })
    pd.DataFrame(eig_rows).to_csv(output_dir / "rbf_eigenvalues.csv", index=False)
    np.save(output_dir / "rbf_weighted_gram.npy", gram)

    audits = []
    selected = None
    transform_max = eigenvectors[:, :max(candidate_counts)] @ np.diag(np.maximum(eigenvalues[:max(candidate_counts)], 1e-30) ** -0.5)
    prefix_audits = audit_transform_prefixes(mask_path, centers, sigma_m, transform_max, target_crs, candidate_counts, block_path=block_path, fold_id=0)
    for k in candidate_counts:
        vals = eigenvalues[:k]
        vecs = eigenvectors[:, :k]
        transform = vecs @ np.diag(np.maximum(vals, 1e-30) ** -0.5)
        global_audit = prefix_audits[int(k)]["global"]
        fold_audit = prefix_audits[int(k)]["fold0"]
        retained_energy = float(np.sum(np.maximum(vals, 0.0)) / max(total_energy, 1e-30))
        row = {
            "basis_count": int(k),
            "retained_energy_fraction": retained_energy,
            "global_condition_number": global_audit["condition_number"],
            "global_orthogonality_error": global_audit["orthogonality_error"],
            "fold0_condition_number": fold_audit["condition_number"],
            "fold0_effective_rank": fold_audit["effective_rank"],
            "fold0_maximum_column_correlation": fold_audit["maximum_column_correlation"],
            "global_passed": bool(global_audit["effective_rank"] == k and np.isfinite(global_audit["condition_number"]) and global_audit["orthogonality_error"] < 1e-6),
            "fold0_passed": bool(fold_audit["effective_rank"] == k and fold_audit["condition_number"] < 1e4 and fold_audit["maximum_column_correlation"] < 0.98),
        }
        row["candidate_passed"] = bool(row["global_passed"] and row["fold0_passed"])
        audits.append((row, transform, global_audit, fold_audit))
        if selected is None and row["candidate_passed"]:
            selected = (row, transform, global_audit, fold_audit)

    if selected is None:
        selected = audits[-1]
        selected[0]["selected_despite_failure"] = True
    row, transform, global_audit, fold_audit = selected
    np.save(output_dir / "rbf_transform.npy", transform)
    (output_dir / "rbf_orthogonal_basis_audit.json").write_text(json.dumps(global_audit, indent=2), encoding="utf-8")
    (output_dir / "rbf_orthogonal_basis_fold0_audit.json").write_text(json.dumps(fold_audit, indent=2), encoding="utf-8")
    pd.DataFrame([a[0] for a in audits]).to_csv(output_dir / "rbf_orthogonal_truncation_audit.csv", index=False)

    metadata = {
        "orthogonalization_version": ORTHO_VERSION,
        "raw_candidate_id": candidate["raw_candidate_id"],
        "raw_center_count": int(candidate["raw_center_count"]),
        "raw_center_coordinates_hash": sha_json(candidate["center_coordinates"]),
        "sigma_km": float(candidate["sigma_km"]),
        "weighting_rule": "uniform_pixel_area_normalized_weights_equal_to_1_over_common_mask_pixel_count",
        "weighted_gram_hash": sha_array(gram),
        "orthogonal_transform_hash": sha_array(transform),
        "selected_basis_count": int(row["basis_count"]),
        "retained_energy_fraction": float(row["retained_energy_fraction"]),
        "basis_design_hash": sha_json({
            "raw_center_coordinates_hash": sha_json(candidate["center_coordinates"]),
            "sigma_km": float(candidate["sigma_km"]),
            "weighted_gram_hash": sha_array(gram),
            "orthogonal_transform_hash": sha_array(transform),
            "selected_basis_count": int(row["basis_count"]),
            "retained_energy_fraction": float(row["retained_energy_fraction"]),
            "weighting_rule": "uniform_pixel_area_normalized_weights_equal_to_1_over_common_mask_pixel_count",
            "orthogonalization_version": ORTHO_VERSION,
        }),
        "eigenvalue_min": float(eigenvalues[-1]),
        "eigenvalue_max": float(eigenvalues[0]),
        "eigenvalue_count_above_relative_1e_minus_8": max_count,
        "prior_coordinate_system": "weighted_orthogonal_rbf_basis",
        "prior_description": "standard ridge penalty on orthogonal coordinates gamma",
        "global_audit": global_audit,
        "fold0_audit": fold_audit,
        "selection_row": row,
        "all_truncation_audits": [a[0] for a in audits],
        "raw_fold0_condition_number": candidate["raw_condition_number_fold0"],
        "raw_rank": candidate["raw_rank"],
        "source_candidate_path": str(candidate_path),
    }
    (output_dir / "rbf_orthogonal_basis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="outputs/aquifer_model_revision/selected_raw_rbf_candidate.json")
    parser.add_argument("--mask", default="outputs/aquifer_model_revision/comparison_common_mask.tif")
    parser.add_argument("--blocks", default="outputs/aquifer_model_revision/spatial_validation_blocks.tif")
    parser.add_argument("--output-dir", default="outputs/aquifer_model_revision/rbf_orthogonalization")
    args = parser.parse_args()
    metadata = build_basis(Path(args.candidate), Path(args.mask), Path(args.blocks), Path(args.output_dir))
    print(json.dumps({
        "selected_basis_count": metadata["selected_basis_count"],
        "retained_energy_fraction": metadata["retained_energy_fraction"],
        "basis_design_hash": metadata["basis_design_hash"],
        "fold0_condition_number": metadata["fold0_audit"]["condition_number"],
        "fold0_passed": metadata["selection_row"]["fold0_passed"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
