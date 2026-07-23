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
)
from .harmonics import phase_days, rotate_sin_cos_coefficients
from .io import read_json
from .rbf import apply_orthogonal_transform, gaussian_rbf

OBSERVATION_SIGMA_MM = 5.0


@dataclass(frozen=True)
class StreamInputs:
    cache: Path = AUTHORITATIVE_CACHE
    common_mask: Path = COMMON_MASK
    fold_map: Path = FOLD_MAP
    release_root: Path = RELEASE_ROOT
    rbf_dim: int = RBF_DIMENSION


def _load_rbf(inputs: StreamInputs) -> tuple[np.ndarray, float, str | None, np.ndarray]:
    design = read_json(inputs.release_root.parents[1] / "canonical_inputs" / "L01028_bounded_memmaps_v1" / "rbf" / "selected_rbf_design.json")
    transform = np.load(inputs.release_root.parents[1] / "canonical_inputs" / "L01028_bounded_memmaps_v1" / "rbf" / "rbf_transform.npy")
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
    ske_sample = np.concatenate(ske_values) if ske_values else np.asarray([], dtype="float32")
    return {
        "pixel_count": pixel_count,
        "observation_count": ncoef,
        "rmse": float(np.sqrt(sse / max(ncoef, 1))),
        "mae": float(ae / max(ncoef, 1)),
        "Ske_min": None if not np.isfinite(ske_min) else float(ske_min),
        "Ske_p50": None if ske_sample.size == 0 else float(np.percentile(ske_sample, 50)),
        "Ske_max": None if not np.isfinite(ske_max) else float(ske_max),
        "Cu_global": cu,
        "lag_c_days": lag_c,
        "lag_u_days": LAG_U_DAYS,
        "gamma_norm": float(np.linalg.norm(gamma)),
        "nonfinite_prediction_count": nonfinite,
        "source_level_recalculation": True,
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


def recompute_storage_metrics(inputs: StreamInputs = StreamInputs()) -> dict[str, Any]:
    ske_path = inputs.release_root / "products" / "Ske.tif"
    regional = np.zeros(2, dtype=float)
    delayed = np.zeros(2, dtype=float)
    local_amplitude = 0.0
    valid_count = 0
    with rasterio.open(ske_path) as ske_src, rasterio.open(inputs.common_mask) as mask_src, h5py.File(inputs.cache, "r") as h5:
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
            ske = ske_src.read(1, window=window).reshape(-1)[flat].astype(float)
            h = h5["hc"][start : start + count].astype(float)
            keep = common & np.isfinite(ske) & np.isfinite(h).all(axis=1)
            if not keep.any():
                continue
            rr = row + local_rows[keep]
            area = areas[rr]
            real = ske[keep] * h[keep, 0] * area
            imag = ske[keep] * h[keep, 1] * area
            delayed_coeff = rotate_sin_cos_coefficients(np.column_stack([real, imag]), LAG_C_DAYS, ANNUAL_PERIOD_DAYS)
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
        "source_level_recalculation": True,
    }
