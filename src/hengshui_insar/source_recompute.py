"""Source-level recomputation for the L01028 bounded release.

These routines read the canonical HDF5/mask/RBF inputs and the saved release
parameters.  They intentionally avoid the historical ``outputs/reference_frames``
and ``outputs/aquifer_model_revision`` paths so the published tree remains
reproducible after cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
from pyproj import Geod, Transformer
import rasterio
from rasterio.transform import xy
from rasterio.windows import Window

from .bounded_model import ske_and_derivative
from .constants import (
    ANNUAL_PERIOD_DAYS,
    AUTHORITATIVE_CACHE,
    COMMON_MASK,
    FOLD_MAP,
    LAG_U_DAYS,
    LAG_C_DAYS,
    RELEASE_ROOT,
    RBF_DIMENSION,
    SKE_MAX,
    SKE_MIN,
    FOLD_MAP_SHA256,
    COMMON_MASK_SHA256,
)
from .harmonics import phase_days, rotate_sin_cos_coefficients
from .io import read_json
from .hashing import sha256_file
from .rbf import apply_orthogonal_transform, gaussian_rbf

OBSERVATION_SIGMA_MM = 5.0


@dataclass(frozen=True)
class StreamInputs:
    cache: Path = AUTHORITATIVE_CACHE
    common_mask: Path = COMMON_MASK
    fold_map: Path = FOLD_MAP
    release_root: Path = RELEASE_ROOT
    selected_rbf_design: Path | None = None
    rbf_transform: Path | None = None
    rbf_dim: int = RBF_DIMENSION
    observation_sigma_mm: float = OBSERVATION_SIGMA_MM


def stream_inputs_from_config(config) -> StreamInputs:
    return StreamInputs(
        cache=config.authoritative_cache,
        common_mask=config.common_mask,
        fold_map=config.fold_map,
        release_root=config.release_root,
        selected_rbf_design=config.selected_rbf_design,
        rbf_transform=config.rbf_transform,
        rbf_dim=config.rbf_dimension,
        observation_sigma_mm=config.observation_sigma_mm,
    )


def _load_rbf(inputs: StreamInputs) -> tuple[np.ndarray, float, str | None, np.ndarray]:
    design_path = inputs.selected_rbf_design or inputs.release_root.parents[1] / "canonical_inputs" / "L01028_bounded_memmaps_v1" / "rbf" / "selected_rbf_design.json"
    transform_path = inputs.rbf_transform or inputs.release_root.parents[1] / "canonical_inputs" / "L01028_bounded_memmaps_v1" / "rbf" / "rbf_transform.npy"
    design = read_json(design_path)
    transform = np.load(transform_path)
    centers = np.asarray(design["center_coordinates"], dtype=float)
    sigma_m = float(design["sigma_km"]) * 1000.0
    return centers, sigma_m, design.get("projected_crs"), np.asarray(transform[:, : inputs.rbf_dim], dtype=float)


def decode_parameters(theta: np.ndarray, rbf_dim: int = RBF_DIMENSION) -> tuple[float, np.ndarray, float, float]:
    theta = np.asarray(theta, dtype=float)
    if theta.size != rbf_dim + 3:
        raise ValueError(f"expected {rbf_dim + 3} parameters, got {theta.size}")
    return float(theta[0]), theta[1 : 1 + rbf_dim], float(np.exp(theta[1 + rbf_dim])), float(theta[2 + rbf_dim])


def load_release_parameters(release_root: Path = RELEASE_ROOT, fold_id: int | None = None) -> np.ndarray:
    if fold_id is None:
        path = release_root / "final_refit" / "parameters.npy"
    else:
        path = release_root / "formal_cv" / f"fold_{fold_id:02d}" / "parameters.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def iter_model_blocks(inputs: StreamInputs, fold_id: int | None, split: str) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield obs, confined head, unconfined head, and transformed RBF basis.

    ``split`` is ``"validation"``, ``"training"``, or ``"all"``.  For final
    refit use ``fold_id=None, split="all"``.
    """
    centers, sigma_m, target_crs, transform = _load_rbf(inputs)
    with h5py.File(inputs.cache, "r") as h5, rasterio.open(inputs.common_mask) as mask_src:
        fold_src = rasterio.open(inputs.fold_map) if fold_id is not None and split != "all" else None
        try:
            transformer = None
            if target_crs and mask_src.crs and str(mask_src.crs) != str(target_crs):
                transformer = Transformer.from_crs(mask_src.crs, target_crs, always_xy=True)
            for bi in range(len(h5["block_start"])):
                start = int(h5["block_start"][bi])
                count = int(h5["block_count"][bi])
                if count <= 0:
                    continue
                row = int(h5["block_row"][bi])
                col = int(h5["block_col"][bi])
                height = int(h5["block_height"][bi])
                width = int(h5["block_width"][bi])
                window = Window(col, row, width, height)
                flat = h5["flat_index"][start : start + count].astype(np.int64)
                local_rows = flat // width
                local_cols = flat % width
                common = mask_src.read(1, window=window).reshape(-1)[flat] == 1
                if fold_src is not None:
                    folds = fold_src.read(1, window=window).reshape(-1)[flat]
                    if split == "validation":
                        common &= folds == fold_id
                    elif split == "training":
                        common &= folds != fold_id
                    else:
                        raise ValueError(f"unsupported split: {split}")
                elif split != "all":
                    raise ValueError("fold_id is required for training/validation split")
                obs = h5["obs"][start : start + count]
                hc = h5["hc"][start : start + count]
                hu = h5["hu"][start : start + count]
                valid = common & np.isfinite(obs).all(axis=1) & np.isfinite(hc).all(axis=1) & np.isfinite(hu).all(axis=1)
                if not valid.any():
                    continue
                rr = row + local_rows[valid]
                cc = col + local_cols[valid]
                xs, ys = xy(mask_src.transform, rr, cc, offset="center")
                xs = np.asarray(xs, dtype=float)
                ys = np.asarray(ys, dtype=float)
                if transformer is not None:
                    xs, ys = transformer.transform(xs, ys)
                    xs = np.asarray(xs, dtype=float)
                    ys = np.asarray(ys, dtype=float)
                raw = gaussian_rbf(np.column_stack([xs, ys]), centers, sigma_m)
                basis = apply_orthogonal_transform(raw, transform)
                yield obs[valid].astype(float), hc[valid].astype(float), hu[valid].astype(float), basis.astype(float)
        finally:
            if fold_src is not None:
                fold_src.close()


def evaluate_parameters(theta: np.ndarray, inputs: StreamInputs, fold_id: int | None = None, split: str = "all") -> dict[str, Any]:
    eta0, gamma, cu, lag_c = decode_parameters(theta, inputs.rbf_dim)
    sse = 0.0
    ae = 0.0
    ncoef = 0
    pixel_count = 0
    nonfinite = 0
    ske_min = np.inf
    ske_max = -np.inf
    ske_values: list[np.ndarray] = []
    pred_abs_values: list[np.ndarray] = []
    upper_count = 0
    for obs, hc, hu, basis in iter_model_blocks(inputs, fold_id, split):
        eta = eta0 + basis @ gamma
        ske, _ = ske_and_derivative(eta, SKE_MIN, SKE_MAX)
        pred = 1000.0 * (
            ske[:, None] * rotate_sin_cos_coefficients(hc, lag_c, ANNUAL_PERIOD_DAYS)
            + cu * rotate_sin_cos_coefficients(hu, LAG_U_DAYS, ANNUAL_PERIOD_DAYS)
        )
        res = obs - pred
        finite = np.isfinite(pred).all(axis=1) & np.isfinite(res).all(axis=1)
        nonfinite += int(np.count_nonzero(~finite))
        sse += float(np.sum(res[finite] * res[finite]))
        ae += float(np.sum(np.abs(res[finite])))
        ncoef += int(res[finite].size)
        pixel_count += int(obs.shape[0])
        if finite.any():
            s = ske[finite]
            ske_min = min(ske_min, float(np.min(s)))
            ske_max = max(ske_max, float(np.max(s)))
            ske_values.append(s.astype("float32", copy=False))
            pred_abs_values.append(np.max(np.abs(pred[finite]), axis=1).astype("float32", copy=False))
            upper_count += int(np.count_nonzero((SKE_MAX - s) <= 1e-6))
    ske_sample = np.concatenate(ske_values) if ske_values else np.asarray([], dtype="float32")
    pred_abs = np.concatenate(pred_abs_values) if pred_abs_values else np.asarray([], dtype="float32")
    return {
        "pixel_count": pixel_count,
        "observation_count": ncoef,
        "rmse": float(np.sqrt(sse / max(ncoef, 1))),
        "mae": float(ae / max(ncoef, 1)),
        "Ske_min": None if not np.isfinite(ske_min) else float(ske_min),
        "Ske_p50": None if ske_sample.size == 0 else float(np.percentile(ske_sample, 50)),
        "Ske_max": None if not np.isfinite(ske_max) else float(ske_max),
        "upper_bound_fraction": float(upper_count / max(pixel_count, 1)),
        "prediction_abs_p99": None if pred_abs.size == 0 else float(np.percentile(pred_abs, 99)),
        "Cu_global": cu,
        "lag_c_days": lag_c,
        "lag_u_days": LAG_U_DAYS,
        "gamma_norm": float(np.linalg.norm(gamma)),
        "nonfinite_prediction_count": nonfinite,
        "source_level_recalculation": True,
    }


def fold_partition_audit(inputs: StreamInputs = StreamInputs()) -> dict[str, Any]:
    common = 0
    fold_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    development = 0
    undefined = 0
    with rasterio.open(inputs.common_mask) as ms, rasterio.open(inputs.fold_map) as fs:
        same_geometry = (
            ms.width == fs.width
            and ms.height == fs.height
            and ms.transform == fs.transform
            and str(ms.crs) == str(fs.crs)
        )
        for _, window in ms.block_windows(1):
            m = ms.read(1, window=window) == 1
            f = fs.read(1, window=window)
            common += int(m.sum())
            development += int((m & (f == 0)).sum())
            valid_fold = np.isin(f, [0, 1, 2, 3, 4])
            undefined += int((m & ~valid_fold).sum())
            for fold in fold_counts:
                fold_counts[fold] += int((m & (f == fold)).sum())
    rows = []
    ok = same_geometry and undefined == 0 and sum(fold_counts.values()) + development == common
    for fold in sorted(fold_counts):
        train = common - fold_counts[fold]
        val = fold_counts[fold]
        rows.append(
            {
                "fold_id": fold,
                "training_pixel_count": train,
                "validation_pixel_count": val,
                "training_validation_intersection_count": 0,
                "training_validation_union_count": common,
                "status": "passed" if train + val == common else "failed",
            }
        )
        ok = ok and train + val == common
    return {
        "fold_partition_status": "passed" if ok else "failed",
        "common_mask_pixel_count": common,
        "fold_counts": fold_counts,
        "development_fold0_pixel_count": development,
        "undefined_fold_value_count": undefined,
        "allowed_fold_values": [0, 1, 2, 3, 4],
        "formal_validation_fold_values": [1, 2, 3, 4],
        "common_mask_hash_match": sha256_file(inputs.common_mask) == COMMON_MASK_SHA256,
        "fold_map_hash_match": sha256_file(inputs.fold_map) == FOLD_MAP_SHA256,
        "same_geometry": same_geometry,
        "rows": rows,
    }


def recompute_cv_metrics(inputs: StreamInputs = StreamInputs()) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for fold_id in (1, 2, 3, 4):
        theta = load_release_parameters(inputs.release_root, fold_id)
        rows[fold_id] = evaluate_parameters(theta, inputs, fold_id=fold_id, split="validation")
    return rows


def recompute_final_metrics(inputs: StreamInputs = StreamInputs()) -> dict[str, Any]:
    theta = load_release_parameters(inputs.release_root, None)
    return evaluate_parameters(theta, inputs, fold_id=None, split="all")


def geodesic_pixel_area_rows(transform: rasterio.Affine, width: int, height: int) -> np.ndarray:
    geod = Geod(ellps="WGS84")
    areas = np.empty(height, dtype=float)
    x0 = transform.c
    x1 = transform.c + transform.a
    for row in range(height):
        y0 = transform.f + row * transform.e
        y1 = transform.f + (row + 1) * transform.e
        area, _ = geod.polygon_area_perimeter([x0, x1, x1, x0], [y0, y0, y1, y1])
        areas[row] = abs(area)
    return areas


def recompute_storage_metrics(inputs: StreamInputs = StreamInputs(), compare_ske_tif: bool = True) -> dict[str, Any]:
    ske_path = inputs.release_root / "products" / "Ske.tif"
    theta = load_release_parameters(inputs.release_root, None)
    eta0, gamma, _cu, lag_c = decode_parameters(theta, inputs.rbf_dim)
    regional = np.zeros(2, dtype=float)
    delayed = np.zeros(2, dtype=float)
    local_amplitude = 0.0
    valid_count = 0
    max_ske_tif_abs_diff = 0.0
    rms_ske_tif_diff_num = 0.0
    rms_ske_tif_diff_n = 0
    with rasterio.open(ske_path) as ske_src, rasterio.open(inputs.common_mask) as mask_src, h5py.File(inputs.cache, "r") as h5:
        centers, sigma_m, target_crs, transform = _load_rbf(inputs)
        transformer = None
        if target_crs and mask_src.crs and str(mask_src.crs) != str(target_crs):
            transformer = Transformer.from_crs(mask_src.crs, target_crs, always_xy=True)
        areas = geodesic_pixel_area_rows(ske_src.transform, ske_src.width, ske_src.height)
        for bi in range(len(h5["block_start"])):
            start = int(h5["block_start"][bi])
            count = int(h5["block_count"][bi])
            if count <= 0:
                continue
            row = int(h5["block_row"][bi])
            col = int(h5["block_col"][bi])
            height = int(h5["block_height"][bi])
            width = int(h5["block_width"][bi])
            window = Window(col, row, width, height)
            flat = h5["flat_index"][start : start + count].astype(np.int64)
            local_rows = flat // width
            local_cols = flat % width
            common = mask_src.read(1, window=window).reshape(-1)[flat] == 1
            ske_tif = ske_src.read(1, window=window).reshape(-1)[flat].astype(float)
            h = h5["hc"][start : start + count].astype(float)
            keep = common & np.isfinite(h).all(axis=1)
            if not keep.any():
                continue
            rr = row + local_rows[keep]
            cc = col + local_cols[keep]
            xs, ys = xy(mask_src.transform, rr, cc, offset="center")
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, dtype=float)
                ys = np.asarray(ys, dtype=float)
            basis = apply_orthogonal_transform(gaussian_rbf(np.column_stack([xs, ys]), centers, sigma_m), transform)
            ske, _ = ske_and_derivative(eta0 + basis @ gamma, SKE_MIN, SKE_MAX)
            keep2 = np.isfinite(ske)
            if not keep2.all():
                rr = rr[keep2]
                cc = cc[keep2]
                h_keep = h[keep][keep2]
                ske_tif_keep = ske_tif[keep][keep2]
                ske = ske[keep2]
            else:
                h_keep = h[keep]
                ske_tif_keep = ske_tif[keep]
            if compare_ske_tif:
                valid_tif = np.isfinite(ske_tif_keep)
                if valid_tif.any():
                    diff = ske[valid_tif].astype(float) - ske_tif_keep[valid_tif].astype(float)
                    max_ske_tif_abs_diff = max(max_ske_tif_abs_diff, float(np.max(np.abs(diff))))
                    rms_ske_tif_diff_num += float(np.sum(diff * diff))
                    rms_ske_tif_diff_n += int(diff.size)
            area = areas[rr]
            real = ske * h_keep[:, 0] * area
            imag = ske * h_keep[:, 1] * area
            delayed_coeff = rotate_sin_cos_coefficients(np.column_stack([real, imag]), lag_c, ANNUAL_PERIOD_DAYS)
            regional += np.array([float(np.sum(real)), float(np.sum(imag))])
            delayed += np.array([float(np.sum(delayed_coeff[:, 0])), float(np.sum(delayed_coeff[:, 1]))])
            local_amplitude += float(np.sum(np.hypot(real, imag)))
            valid_count += int(np.count_nonzero(keep))
    coherent = float(np.hypot(regional[0], regional[1]))
    phi = float(phase_days(regional[0], regional[1], ANNUAL_PERIOD_DAYS))
    delayed_phi = float(phase_days(delayed[0], delayed[1], ANNUAL_PERIOD_DAYS))
    shift = (delayed_phi - phi) % ANNUAL_PERIOD_DAYS
    if shift > ANNUAL_PERIOD_DAYS / 2.0:
        shift -= ANNUAL_PERIOD_DAYS
    return {
        "valid_pixel_count": valid_count,
        "regional_real_m3": float(regional[0]),
        "regional_imag_m3": float(regional[1]),
        "regional_coherent_amplitude_m3": coherent,
        "sum_local_amplitudes_m3": local_amplitude,
        "seasonal_max_minus_min_m3": 2.0 * coherent,
        "phase_days": phi,
        "delayed_peak_shift_days": float(shift),
        "delayed_peak_shift_sign": "positive_delay" if shift > 0 else "negative_or_zero_delay",
        "lag_c_days_from_final_parameters": lag_c,
        "Ske_tif_max_abs_diff": max_ske_tif_abs_diff,
        "Ske_tif_rms_diff": float(np.sqrt(rms_ske_tif_diff_num / max(rms_ske_tif_diff_n, 1))),
        "Ske_tif_comparison_count": rms_ske_tif_diff_n,
        "source_level_recalculation": True,
    }
