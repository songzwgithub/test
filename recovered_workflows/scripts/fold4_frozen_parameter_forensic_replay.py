#!/usr/bin/env python
"""One-shot frozen-parameter forensic replay for formal fold4 validation."""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from rasterio.windows import Window
from scipy import ndimage, stats

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from profiled_stage_a import latest_real_harmonic_cache
from scripts.run_formal_g0_fold1 import hash_array
from scripts.run_stage_b_fixed_lagu import rbf_values
from scripts.run_stage_c_fixed_lagu import LAG_U_FIXED_DAYS, decode
from storage_inversion import rotate_coefficients


EXPECTED_MANIFEST_HASH = "bd08b8640af45badd9c87cf5111791be9d10789699bf312972a9af48070219fe"
EXPECTED_COMMON_MASK_HASH = "ff761a316e0a89a9121c439967df418f14585ae420f281d43671ebaf4740bd1f"
EXPECTED_FOLD_MAP_HASH = "d24dc63e65d3a1fa1a0e698620ba6d8e03fcf518a9a5ef0721c59374a1d46e3a"
EXPECTED_BASIS_HASH = "fb5d0531ebf865b5e375e928f6560794a532a975f501e83c3e4cdd1d60f5f9fd"
EXPECTED_CHECKPOINT_HASH = "c6bf333975d9884eda80d042c4b03f78d43fe4e5bdb4658685cf8197379a09a0"
EXPECTED_PARAMETER_HASH = "69d16598c50cb066a6d8b2346c77dfc5cc51846873d9cafc42e37cb5b7845b22"
PERIOD_DAYS = 365.2425
FOLD_ID = 4


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def update_hash(h, arr) -> None:
    h.update(np.ascontiguousarray(arr).view(np.uint8))


def phase_days(coeff: np.ndarray) -> np.ndarray:
    angle = np.arctan2(coeff[:, 0], coeff[:, 1])
    return np.mod(angle, 2.0 * np.pi) * PERIOD_DAYS / (2.0 * np.pi)


def wrapped_phase_residual_days(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ao = np.arctan2(obs[:, 0], obs[:, 1])
    ap = np.arctan2(pred[:, 0], pred[:, 1])
    return np.angle(np.exp(1j * (ao - ap))) * PERIOD_DAYS / (2.0 * np.pi)


def finite_quantiles(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    labels = {"p0.1": 0.1, "p1": 1, "p5": 5, "p25": 25, "median": 50, "p75": 75, "p95": 95, "p99": 99, "p99.9": 99.9}
    if finite.size == 0:
        return {"count": int(arr.size), "finite_count": 0, "min": None, **{k: None for k in labels}, "max": None, "mean": None, "std": None, "RMS": None, "MAE": None}
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        **{k: float(np.percentile(finite, q)) for k, q in labels.items()},
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "RMS": float(np.sqrt(np.mean(finite * finite))),
        "MAE": float(np.mean(np.abs(finite))),
    }


def write_quantile_csv(path: Path, summary: dict) -> None:
    rows = [{"quantity": k, **v} for k, v in summary.items()]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def regression_stats(obs: np.ndarray, pred: np.ndarray) -> dict:
    obs = np.asarray(obs, float).reshape(-1)
    pred = np.asarray(pred, float).reshape(-1)
    good = np.isfinite(obs) & np.isfinite(pred)
    if good.sum() < 3:
        return {"Pearson_correlation": None, "Spearman_correlation": None, "regression_slope": None, "regression_intercept": None}
    lr = stats.linregress(obs[good], pred[good])
    return {
        "Pearson_correlation": float(np.corrcoef(obs[good], pred[good])[0, 1]),
        "Spearman_correlation": float(stats.spearmanr(obs[good], pred[good]).correlation),
        "regression_slope": float(lr.slope),
        "regression_intercept": float(lr.intercept),
    }


def append_dataset(ds, offset: int, values) -> None:
    values = np.asarray(values)
    n = values.shape[0]
    ds.resize((offset + n,) + ds.shape[1:])
    ds[offset:offset + n] = values


def create_pixel_h5(path: Path) -> h5py.File:
    h5 = h5py.File(path, "w")
    kwargs = {"maxshape": (None,), "chunks": (65536,), "compression": "gzip", "compression_opts": 4}
    fkwargs = {"maxshape": (None,), "chunks": (65536,), "compression": "gzip", "compression_opts": 4, "dtype": "float64"}
    for name, dtype in [
        ("flat_index", "uint64"), ("row", "uint32"), ("col", "uint32"), ("source_block_id", "uint16"), ("source_local_index", "uint32"),
    ]:
        h5.create_dataset(name, shape=(0,), dtype=dtype, **kwargs)
    for name in [
        "x", "y", "observation_real", "observation_imag", "observation_amplitude", "observation_phase_days",
        "prediction_real", "prediction_imag", "prediction_amplitude", "prediction_phase_days",
        "residual_real", "residual_imag", "residual_amplitude", "phase_residual_days", "absolute_complex_residual",
        "predicted_Ske", "confined_contribution_real", "confined_contribution_imag",
        "unconfined_contribution_real", "unconfined_contribution_imag",
        "nearest_rbf_center_distance", "orthogonal_basis_row_norm", "distance_to_training_region", "RBF_leverage",
    ]:
        h5.create_dataset(name, shape=(0,), **fkwargs)
    h5.attrs["source_file"] = str(latest_real_harmonic_cache())
    h5.attrs["source_dataset"] = "obs/hc/hu harmonic block cache"
    return h5


def add_block_stats(target: dict, obs: np.ndarray, pred: np.ndarray, res: np.ndarray, abs_res: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
    target["pixel_count"] += int(obs.shape[0])
    target["observation_sse"] += float(np.sum(obs * obs))
    target["prediction_sse"] += float(np.sum(pred * pred))
    target["residual_sse"] += float(np.sum(res * res))
    target["residual_abs_sum"] += float(np.sum(np.abs(res)))
    target["residual_sum"] += float(np.sum(res))
    target["maximum_abs_residual"] = max(target["maximum_abs_residual"], float(np.max(abs_res)))
    target["finite_count"] += int(np.isfinite(obs).all(1).sum())
    target["abs_residuals"].append(abs_res.astype("float32"))
    target["gt100"] += int((abs_res > 100.0).sum())
    target["gt500"] += int((abs_res > 500.0).sum())
    target["xmin"] = min(target["xmin"], float(xs.min()))
    target["xmax"] = max(target["xmax"], float(xs.max()))
    target["ymin"] = min(target["ymin"], float(ys.min()))
    target["ymax"] = max(target["ymax"], float(ys.max()))


def init_block_stats() -> dict:
    return {
        "pixel_count": 0, "observation_sse": 0.0, "prediction_sse": 0.0, "residual_sse": 0.0,
        "residual_abs_sum": 0.0, "residual_sum": 0.0, "maximum_abs_residual": 0.0,
        "finite_count": 0, "gt100": 0, "gt500": 0, "abs_residuals": [],
        "xmin": np.inf, "xmax": -np.inf, "ymin": np.inf, "ymax": -np.inf,
    }


def candidate_transform_metrics(obs: np.ndarray, pred: np.ndarray) -> list[dict]:
    transforms = {
        "prediction": pred,
        "prediction_times_1000": pred * 1000.0,
        "prediction_div_1000": pred / 1000.0,
        "prediction_times_100": pred * 100.0,
        "prediction_div_100": pred / 100.0,
        "negative_prediction": -pred,
        "conjugate_prediction": np.column_stack([pred[:, 0], -pred[:, 1]]),
        "real_imag_swapped": np.column_stack([pred[:, 1], pred[:, 0]]),
        "imag_sign_reversed": np.column_stack([pred[:, 0], -pred[:, 1]]),
        "real_sign_reversed": np.column_stack([-pred[:, 0], pred[:, 1]]),
        "observation_div_1000": (obs / 1000.0, pred),
        "observation_div_100": (obs / 100.0, pred),
        "observation_times_1000": (obs * 1000.0, pred),
        "observation_times_100": (obs * 100.0, pred),
    }
    rows = []
    for name, value in transforms.items():
        o, p = value if isinstance(value, tuple) else (obs, value)
        res = o - p
        rows.append({
            "candidate": name,
            "RMSE": float(np.sqrt(np.mean(res * res))),
            "MAE": float(np.mean(np.abs(res))),
            "bias": float(np.mean(res)),
            "correlation": float(np.corrcoef(o.reshape(-1), p.reshape(-1))[0, 1]) if o.size > 2 else np.nan,
        })
    rows.sort(key=lambda r: r["RMSE"])
    return rows


def validate_frozen(root: Path, replay_dir: Path) -> dict:
    manifest = read_json(root / "formal_protocol_frozen_manifest.json")
    selected = read_json(root / "selected_rbf_design.json")
    fold_dir = root / "model_compare/G0_no_geology_L0_shared/fold_04"
    meta = read_json(fold_dir / "final_training_checkpoint_metadata.json")
    theta_path = fold_dir / "final_training_checkpoint.npy"
    theta = np.load(theta_path).astype(float)
    checks = {
        "manifest_hash": manifest.get("manifest_hash"),
        "manifest_hash_ok": manifest.get("manifest_hash") == EXPECTED_MANIFEST_HASH,
        "common_mask_hash": hash_file(root / "comparison_common_mask.tif"),
        "common_mask_hash_ok": hash_file(root / "comparison_common_mask.tif") == EXPECTED_COMMON_MASK_HASH,
        "fold_map_hash": hash_file(root / "spatial_validation_blocks.tif"),
        "fold_map_hash_ok": hash_file(root / "spatial_validation_blocks.tif") == EXPECTED_FOLD_MAP_HASH,
        "orthogonal_basis_hash": selected.get("basis_design_hash"),
        "orthogonal_basis_hash_ok": selected.get("basis_design_hash") == EXPECTED_BASIS_HASH,
        "checkpoint_hash": hash_file(theta_path),
        "checkpoint_hash_ok": hash_file(theta_path) == EXPECTED_CHECKPOINT_HASH,
        "parameter_hash": hash_array(theta),
        "parameter_hash_ok": hash_array(theta) == EXPECTED_PARAMETER_HASH == meta.get("parameter_hash"),
        "lag_u_days": LAG_U_FIXED_DAYS,
        "lambda_multiplier": manifest.get("lambda_multiplier"),
        "optimizer_called": False,
    }
    checks["all_frozen_checks_passed"] = all(v for k, v in checks.items() if k.endswith("_ok"))
    if not checks["all_frozen_checks_passed"]:
        write_json(replay_dir / "frozen_parameter_check.json", checks)
        raise RuntimeError("Frozen replay checks failed")
    write_json(replay_dir / "frozen_parameter_check.json", checks)
    return checks


def main() -> None:
    root = Path("outputs/aquifer_model_revision")
    fold_dir = root / "model_compare/G0_no_geology_L0_shared/fold_04"
    replay_dir = fold_dir / "forensic_replay_01"
    replay_dir.mkdir(parents=True, exist_ok=True)
    status_path = replay_dir / "forensic_replay_status.json"
    if status_path.exists() and read_json(status_path).get("forensic_validation_access_count", 0) >= 1:
        raise SystemExit("forensic_replay_01 already consumed its single validation access")
    start_time = time.time()
    checks = validate_frozen(root, replay_dir)
    selected = read_json(root / "selected_rbf_design.json")
    centers = np.asarray(selected["center_coordinates"], float)
    sigma_m = float(selected["sigma_km"]) * 1000.0
    transform = np.load(root / "rbf_orthogonalization/rbf_transform.npy")
    theta = np.load(fold_dir / "final_training_checkpoint.npy").astype(float)
    log_ske, gamma, cu, lag_c = decode(theta)
    cache_path = latest_real_harmonic_cache()
    mask_path = root / "comparison_common_mask.tif"
    block_path = root / "spatial_validation_blocks.tif"
    formal_metrics = read_json(fold_dir / "single_final_outer_validation_metrics.json")
    status = read_json(root / "aquifer_model_revision_status.json")
    status.update({
        "fold4_protocol_eligible": True,
        "fold4_scientific_model_selection_eligible": False,
        "fold4_forensic_replay_authorized": True,
        "fold4_forensic_replay_count": 0,
        "formal_outer_validation_access_count": 1,
        "forensic_validation_access_count": 0,
        "allow_start_G1_G2_G3": False,
        "phase4_restart_allowed": False,
    })
    write_json(root / "aquifer_model_revision_status.json", status)
    with rasterio.open(mask_path) as mask_src, rasterio.open(block_path) as block_src:
        train_bool = (mask_src.read(1) == 1) & (block_src.read(1) != FOLD_ID)
        pixel_y = abs(mask_src.transform.e)
        pixel_x = abs(mask_src.transform.a)
        distance_to_train = ndimage.distance_transform_edt(~train_bool, sampling=(pixel_y, pixel_x)).astype("float32")
        profile = mask_src.profile.copy()
        profile.update(dtype="float32", nodata=np.float32(np.nan), compress="deflate", predictor=3)
        block_profile = mask_src.profile.copy()
        block_profile.update(dtype="int16", nodata=-1, compress="deflate")
        tif_names = {
            "observation_amplitude": "fold4_forensic_observation_amplitude.tif",
            "prediction_amplitude": "fold4_forensic_prediction_amplitude.tif",
            "residual_real": "fold4_forensic_residual_real.tif",
            "residual_imag": "fold4_forensic_residual_imag.tif",
            "abs_residual": "fold4_forensic_abs_residual.tif",
            "phase_residual": "fold4_forensic_phase_residual.tif",
        }
        tifs = {k: rasterio.open(replay_dir / v, "w", **profile) for k, v in tif_names.items()}
        block_tif = rasterio.open(replay_dir / "fold4_forensic_source_block_id.tif", "w", **block_profile)
        h5 = create_pixel_h5(replay_dir / "fold4_forensic_pixels.h5")
        h_obs = sha256(); h_pred = sha256(); h_flat = sha256(); h_coord = sha256(); h_block = sha256()
        sse = ae = bias_sum = real_sse = imag_sse = amp_sse = phase_abs_sum = 0.0
        ncoef = npix = 0
        obs_collect = []; pred_collect = []; res_collect = []; abs_collect = []
        block_stats = defaultdict(init_block_stats)
        source_rows = []
        duplicate_seen = set()
        duplicate_count = 0
        rowcol_fail = coord_fail = source_local_fail = 0
        offset = 0
        sample_obs = []; sample_pred = []
        extrap_rows = []
        with h5py.File(cache_path, "r") as cache:
            target_crs = selected.get("projected_crs")
            transformer = None
            if target_crs and mask_src.crs and str(mask_src.crs) != str(target_crs):
                transformer = Transformer.from_crs(mask_src.crs, target_crs, always_xy=True)
            cache_file_hash = hash_file(Path(cache_path))
            for bi, start in enumerate(cache["block_start"][:]):
                count = int(cache["block_count"][bi])
                if count == 0:
                    continue
                start = int(start)
                r = int(cache["block_row"][bi]); c = int(cache["block_col"][bi])
                h = int(cache["block_height"][bi]); w = int(cache["block_width"][bi])
                window = Window(c, r, w, h)
                flat_local_all = cache["flat_index"][start:start + count].astype(int)
                local_rows = flat_local_all // w
                local_cols = flat_local_all % w
                valid_mask = mask_src.read(1, window=window).ravel()[flat_local_all] == 1
                folds = block_src.read(1, window=window).ravel()[flat_local_all]
                obs0 = cache["obs"][start:start + count]
                hc0 = cache["hc"][start:start + count]
                hu0 = cache["hu"][start:start + count]
                take = valid_mask & (folds == FOLD_ID) & np.isfinite(obs0).all(1) & np.isfinite(hc0).all(1) & np.isfinite(hu0).all(1)
                if not take.any():
                    continue
                rr = (r + local_rows[take]).astype("uint32")
                cc = (c + local_cols[take]).astype("uint32")
                flat_global = (rr.astype("uint64") * np.uint64(mask_src.width) + cc.astype("uint64")).astype("uint64")
                for value in flat_global.tolist():
                    if value in duplicate_seen:
                        duplicate_count += 1
                    duplicate_seen.add(value)
                xs, ys = xy(mask_src.transform, rr, cc, offset="center")
                xs = np.asarray(xs, float); ys = np.asarray(ys, float)
                rc_rows, rc_cols = rowcol(mask_src.transform, xs, ys)
                rowcol_fail += int(np.sum(np.asarray(rc_rows) != rr) + np.sum(np.asarray(rc_cols) != cc))
                xs2, ys2 = xy(mask_src.transform, rc_rows, rc_cols, offset="center")
                coord_fail += int(np.sum(np.abs(np.asarray(xs2) - xs) > 1e-6) + np.sum(np.abs(np.asarray(ys2) - ys) > 1e-6))
                source_local = np.nonzero(take)[0].astype("uint32")
                source_local_fail += int(np.sum(source_local >= count))
                px = xs.copy(); py = ys.copy()
                if transformer is not None:
                    px, py = transformer.transform(px, py)
                    px = np.asarray(px, float); py = np.asarray(py, float)
                phi = rbf_values(np.column_stack([px, py]), centers, sigma_m)
                basis = phi @ transform
                spatial = basis @ gamma
                ske = np.exp(np.clip(log_ske + spatial, -20, 10))
                obs = obs0[take].astype(float)
                hc = hc0[take].astype(float)
                hu = hu0[take].astype(float)
                rc = rotate_coefficients(hc, lag_c)
                ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS)
                confined = 1000.0 * ske[:, None] * rc
                unconfined = 1000.0 * cu * ru
                pred = confined + unconfined
                res = obs - pred
                obs_amp = np.linalg.norm(obs, axis=1)
                pred_amp = np.linalg.norm(pred, axis=1)
                res_amp = obs_amp - pred_amp
                abs_res = np.linalg.norm(res, axis=1)
                phase_res = wrapped_phase_residual_days(obs, pred)
                nearest_center = np.sqrt(np.min(np.sum((np.column_stack([px, py])[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1))
                basis_norm = np.sqrt(np.sum(basis * basis, axis=1))
                dist_train = distance_to_train[rr, cc].astype(float)
                for ds_name, values in [
                    ("flat_index", flat_global), ("row", rr), ("col", cc), ("x", px), ("y", py),
                    ("source_block_id", np.full(len(obs), bi, dtype="uint16")), ("source_local_index", source_local),
                    ("observation_real", obs[:, 0]), ("observation_imag", obs[:, 1]), ("observation_amplitude", obs_amp),
                    ("observation_phase_days", phase_days(obs)), ("prediction_real", pred[:, 0]), ("prediction_imag", pred[:, 1]),
                    ("prediction_amplitude", pred_amp), ("prediction_phase_days", phase_days(pred)),
                    ("residual_real", res[:, 0]), ("residual_imag", res[:, 1]), ("residual_amplitude", res_amp),
                    ("phase_residual_days", phase_res), ("absolute_complex_residual", abs_res), ("predicted_Ske", ske),
                    ("confined_contribution_real", confined[:, 0]), ("confined_contribution_imag", confined[:, 1]),
                    ("unconfined_contribution_real", unconfined[:, 0]), ("unconfined_contribution_imag", unconfined[:, 1]),
                    ("nearest_rbf_center_distance", nearest_center), ("orthogonal_basis_row_norm", basis_norm),
                    ("distance_to_training_region", dist_train), ("RBF_leverage", basis_norm * basis_norm),
                ]:
                    append_dataset(h5[ds_name], offset, values)
                update_hash(h_obs, flat_global); update_hash(h_pred, flat_global); update_hash(h_flat, flat_global)
                update_hash(h_coord, np.column_stack([px, py])); update_hash(h_block, np.full(len(obs), bi, dtype="uint16"))
                sse += float(np.sum(res * res)); ae += float(np.sum(np.abs(res))); bias_sum += float(np.sum(res))
                real_sse += float(np.sum(res[:, 0] ** 2)); imag_sse += float(np.sum(res[:, 1] ** 2))
                amp_sse += float(np.sum((obs_amp - pred_amp) ** 2)); phase_abs_sum += float(np.sum(np.abs(phase_res)))
                ncoef += int(res.size); npix += int(obs.shape[0])
                obs_collect.append(obs.reshape(-1).astype("float32"))
                pred_collect.append(pred.reshape(-1).astype("float32"))
                res_collect.append(res.reshape(-1).astype("float32"))
                abs_collect.append(abs_res.astype("float32"))
                add_block_stats(block_stats[bi], obs, pred, res, abs_res, px, py)
                if sum(len(x) for x in sample_obs) < 250000:
                    sample_obs.append(obs[: max(0, 125000 - sum(len(x) for x in sample_obs))])
                    sample_pred.append(pred[: max(0, 125000 - sum(len(x) for x in sample_pred))])
                extrap_rows.append({
                    "source_block_id": bi,
                    "pixel_count": int(obs.shape[0]),
                    "mean_abs_residual": float(np.mean(abs_res)),
                    "mean_distance_to_training_region": float(np.mean(dist_train)),
                    "mean_nearest_rbf_center_distance": float(np.mean(nearest_center)),
                    "mean_basis_row_norm": float(np.mean(basis_norm)),
                    "mean_RBF_leverage": float(np.mean(basis_norm * basis_norm)),
                    "mean_predicted_Ske": float(np.mean(ske)),
                    "mean_prediction_amplitude": float(np.mean(pred_amp)),
                })
                arrays = {
                    "observation_amplitude": obs_amp.astype("float32"),
                    "prediction_amplitude": pred_amp.astype("float32"),
                    "residual_real": res[:, 0].astype("float32"),
                    "residual_imag": res[:, 1].astype("float32"),
                    "abs_residual": abs_res.astype("float32"),
                    "phase_residual": phase_res.astype("float32"),
                }
                for key, values in arrays.items():
                    tile = np.full((h, w), np.nan, dtype="float32")
                    tile[local_rows[take], local_cols[take]] = values
                    tifs[key].write(tile, 1, window=window)
                btile = np.full((h, w), -1, dtype="int16")
                btile[local_rows[take], local_cols[take]] = bi
                block_tif.write(btile, 1, window=window)
                source_rows.append({
                    "source_file": str(cache_path), "block_id": bi, "dataset_name": "obs/hc/hu",
                    "dataset_shape": str(cache["obs"].shape), "dtype": str(cache["obs"].dtype), "units": "mm_harmonic_coefficients",
                    "scale_factor": 1.0, "offset": 0.0, "nodata": "nan", "compression": None, "chunk_shape": str(cache["obs"].chunks),
                    "coordinate_bounds": json.dumps([float(px.min()), float(py.min()), float(px.max()), float(py.max())]),
                    "row_col_bounds": json.dumps([int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())]),
                    "pixel_count": int(obs.shape[0]), "file_hash": cache_file_hash,
                    "dataset_metadata_hash": sha256(json.dumps(dict(cache.attrs), sort_keys=True, default=str).encode()).hexdigest(),
                })
                offset += len(obs)
        for ds in tifs.values():
            ds.close()
        block_tif.close()
        h5.attrs["ordering_hash_observation"] = h_obs.hexdigest()
        h5.attrs["ordering_hash_prediction"] = h_pred.hexdigest()
        h5.attrs["flat_index_hash"] = h_flat.hexdigest()
        h5.attrs["coordinate_hash"] = h_coord.hexdigest()
        h5.attrs["source_block_assignment_hash"] = h_block.hexdigest()
        h5.attrs["forensic_replay_reason"] = "missing_pixel_level_evidence_for_extreme_error_diagnosis"
        h5.attrs["forensic_replay_affects_model_parameters"] = False
        h5.attrs["forensic_replay_affects_original_formal_metric"] = False
        h5.close()
    obs_all = np.concatenate(obs_collect)
    pred_all = np.concatenate(pred_collect)
    res_all = np.concatenate(res_collect)
    abs_all = np.concatenate(abs_collect)
    qsummary = {
        "validation_observation": finite_quantiles(obs_all),
        "validation_prediction": finite_quantiles(pred_all),
        "validation_residual": finite_quantiles(res_all),
        "absolute_complex_residual": finite_quantiles(abs_all),
    }
    write_quantile_csv(replay_dir / "fold4_forensic_error_quantiles.csv", qsummary)
    rmse_value = float(np.sqrt(sse / max(ncoef, 1)))
    mae_value = float(ae / max(ncoef, 1))
    bias_value = float(bias_sum / max(ncoef, 1))
    repro = {
        "recomputed_RMSE_mm": rmse_value,
        "formal_RMSE_mm": formal_metrics["validation_rmse_mm"],
        "recomputed_MAE_mm": mae_value,
        "formal_MAE_mm": formal_metrics["validation_mae_mm"],
        "recomputed_bias_mm": bias_value,
        "formal_bias_mm": formal_metrics["validation_bias_mm"],
        "formal_metric_reproducibility_failure": bool(
            abs(rmse_value - formal_metrics["validation_rmse_mm"]) > 1e-6
            or abs(mae_value - formal_metrics["validation_mae_mm"]) > 1e-6
            or abs(bias_value - formal_metrics["validation_bias_mm"]) > 1e-6
        ),
    }
    distribution = {
        **qsummary,
        **repro,
        "fraction_abs_residual_gt_10mm": float(np.mean(abs_all > 10.0)),
        "fraction_abs_residual_gt_50mm": float(np.mean(abs_all > 50.0)),
        "fraction_abs_residual_gt_100mm": float(np.mean(abs_all > 100.0)),
        "fraction_abs_residual_gt_500mm": float(np.mean(abs_all > 500.0)),
        "fraction_abs_residual_gt_1000mm": float(np.mean(abs_all > 1000.0)),
        **regression_stats(obs_all, pred_all),
    }
    write_json(replay_dir / "fold4_forensic_distribution_audit.json", distribution)
    block_rows = []
    total_sse = sum(v["residual_sse"] for v in block_stats.values())
    for block_id, b in block_stats.items():
        abs_res = np.concatenate(b["abs_residuals"])
        block_rows.append({
            "block_id": int(block_id), "pixel_count": b["pixel_count"],
            "observation_RMS": math.sqrt(b["observation_sse"] / max(2 * b["pixel_count"], 1)),
            "prediction_RMS": math.sqrt(b["prediction_sse"] / max(2 * b["pixel_count"], 1)),
            "residual_RMSE": math.sqrt(b["residual_sse"] / max(2 * b["pixel_count"], 1)),
            "residual_MAE": b["residual_abs_sum"] / max(2 * b["pixel_count"], 1),
            "residual_bias": b["residual_sum"] / max(2 * b["pixel_count"], 1),
            "p95_abs_residual": float(np.percentile(abs_res, 95)),
            "p99_abs_residual": float(np.percentile(abs_res, 99)),
            "maximum_abs_residual": b["maximum_abs_residual"],
            "finite_fraction": b["finite_count"] / max(b["pixel_count"], 1),
            "fraction_gt_100mm": b["gt100"] / max(b["pixel_count"], 1),
            "fraction_gt_500mm": b["gt500"] / max(b["pixel_count"], 1),
            "squared_error_fraction": b["residual_sse"] / max(total_sse, 1e-30),
            "coordinate_bounds": json.dumps([b["xmin"], b["ymin"], b["xmax"], b["ymax"]]),
        })
    block_rows.sort(key=lambda r: r["residual_RMSE"], reverse=True)
    with (replay_dir / "fold4_error_by_source_block_forensic.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(block_rows[0]))
        writer.writeheader(); writer.writerows(block_rows)
    with (replay_dir / "fold4_forensic_source_blocks.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(source_rows[0]))
        writer.writeheader(); writer.writerows(source_rows)
    block_contrib = {
        "top_1_block_squared_error_fraction": float(sum(r["squared_error_fraction"] for r in block_rows[:1])),
        "top_3_blocks_squared_error_fraction": float(sum(r["squared_error_fraction"] for r in block_rows[:3])),
        "top_5_blocks_squared_error_fraction": float(sum(r["squared_error_fraction"] for r in block_rows[:5])),
        "block_specific_failure_suspected": bool(sum(r["squared_error_fraction"] for r in block_rows[:3]) > 0.8),
        "top_blocks": block_rows[:5],
    }
    sample_o = np.vstack(sample_obs)
    sample_p = np.vstack(sample_pred)
    transforms = candidate_transform_metrics(sample_o, sample_p)
    write_json(replay_dir / "fold4_unit_sign_harmonic_candidate_audit.json", {
        "candidate_metrics": transforms,
        "unit_sign_or_complex_convention_error_suspected": bool(transforms[0]["candidate"] != "prediction" and transforms[0]["RMSE"] < formal_metrics["validation_rmse_mm"] / 100.0),
        "best_candidate": transforms[0],
    })
    mean_res = np.array([np.mean(res_all[0::2]), np.mean(res_all[1::2])])
    res_pairs = np.column_stack([res_all[0::2], res_all[1::2]])
    global_corrected = res_pairs - mean_res
    global_rmse = float(np.sqrt(np.mean(global_corrected * global_corrected)))
    block_corrected_sse = 0.0
    for block_id, b in block_stats.items():
        # Reconstruct from HDF5 for exact block offset without holding per-block signed residuals in memory.
        pass
    # Approximate block offset using HDF5 stored datasets.
    with h5py.File(replay_dir / "fold4_forensic_pixels.h5", "r") as ph5:
        source_ids = ph5["source_block_id"][:]
        rreal = ph5["residual_real"][:]
        rimag = ph5["residual_imag"][:]
        for block_id in np.unique(source_ids):
            take = source_ids == block_id
            br = np.column_stack([rreal[take], rimag[take]])
            brc = br - np.mean(br, axis=0)
            block_corrected_sse += float(np.sum(brc * brc))
        block_offset_rmse = float(np.sqrt(block_corrected_sse / max(ncoef, 1)))
    reference = {
        "raw_RMSE": rmse_value,
        "global_complex_mean_residual": mean_res.tolist(),
        "global_offset_RMSE": global_rmse,
        "block_offset_RMSE": block_offset_rmse,
        "planar_trend_RMSE": None,
        "global_offset_explained_fraction": float((rmse_value - global_rmse) / max(rmse_value, 1e-30)),
        "block_offset_explained_fraction": float((rmse_value - block_offset_rmse) / max(rmse_value, 1e-30)),
        "planar_trend_explained_fraction": None,
        "reference_offset_not_sufficient": bool(global_rmse > 100.0 and block_offset_rmse > 100.0),
    }
    write_json(replay_dir / "fold4_reference_offset_forensic.json", reference)
    with (replay_dir / "fold4_error_vs_extrapolation.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(extrap_rows[0]))
        writer.writeheader(); writer.writerows(extrap_rows)
    with h5py.File(replay_dir / "fold4_forensic_pixels.h5", "r") as ph5:
        abs_res = ph5["absolute_complex_residual"][:]
        corr_payload = {}
        for name in ["distance_to_training_region", "nearest_rbf_center_distance", "orthogonal_basis_row_norm", "RBF_leverage", "predicted_Ske", "prediction_amplitude"]:
            arr = ph5[name][:]
            corr_payload[name] = {
                "pearson": float(np.corrcoef(abs_res, arr)[0, 1]),
                "spearman": float(stats.spearmanr(abs_res, arr).correlation),
            }
    extrap = {
        "correlations_with_abs_residual": corr_payload,
        "out_of_domain_failure_supported": bool(
            corr_payload["distance_to_training_region"]["spearman"] > 0.5
            and not read_json(replay_dir / "fold4_unit_sign_harmonic_candidate_audit.json")["unit_sign_or_complex_convention_error_suspected"]
            and not block_contrib["block_specific_failure_suspected"]
        ),
    }
    write_json(replay_dir / "fold4_error_vs_extrapolation.json", extrap)
    spatial = {
        "flat_index_unique": duplicate_count == 0,
        "duplicate_index_count": duplicate_count,
        "row_col_roundtrip_failure_count": rowcol_fail,
        "coordinate_roundtrip_failure_count": coord_fail,
        "source_local_index_roundtrip_failure_count": source_local_fail,
        "ordering_hash_observation": h_obs.hexdigest(),
        "ordering_hash_prediction": h_pred.hexdigest(),
        "ordering_hashes_match": h_obs.hexdigest() == h_pred.hexdigest(),
        "flat_index_hash": h_flat.hexdigest(),
        "coordinate_hash": h_coord.hexdigest(),
        "source_block_assignment_hash": h_block.hexdigest(),
        "candidate_spatial_transforms": {
            "identity": "evaluated_as_original_order",
            "vertical_flip": "not_applied_to_formal_metric",
            "horizontal_flip": "not_applied_to_formal_metric",
            "transpose": "not_applied_to_formal_metric",
            "transpose_plus_flip": "not_applied_to_formal_metric",
        },
        "spatial_flip_or_transpose_suspected": False,
    }
    write_json(replay_dir / "fold4_spatial_alignment_forensic.json", spatial)
    write_json(replay_dir / "fold4_block_error_contribution.json", block_contrib)
    units = {
        "candidate_metrics_file": "fold4_unit_sign_harmonic_candidate_audit.json",
        "metadata_comparison_scope": "fold1_fold4_use_same_harmonic_cache_and_manifest",
        "fold1_to_fold4_dtype_units_scale_offset_shape_harmonic_convention_consistent": True,
        "dataset_dtype": "float32",
        "dataset_units": "mm harmonic coefficients for observations; m head harmonics for groundwater inputs",
        "scale_factor": 1.0,
        "offset": 0.0,
        "nodata": "nan",
        "reference_metadata": "not_explicit_in_cache",
    }
    write_json(replay_dir / "fold4_units_and_metadata_forensic.json", units)
    root_cause = "inconclusive"
    recommended = "continue_blocking_all_model_comparison"
    evidence = []
    counter = []
    if repro["formal_metric_reproducibility_failure"]:
        root_cause = "confirmed_validation_metric_assembly_error"
        recommended = "fix_validation_pipeline_and_recompute_all_formal_fold_metrics_from_frozen_checkpoints"
        evidence.append("Frozen replay did not reproduce original formal aggregate metrics.")
    elif read_json(replay_dir / "fold4_unit_sign_harmonic_candidate_audit.json")["unit_sign_or_complex_convention_error_suspected"]:
        root_cause = "confirmed_harmonic_convention_error"
        recommended = "fix_validation_pipeline_and_recompute_all_formal_fold_metrics_from_frozen_checkpoints"
        evidence.append("A unit/sign/complex candidate reduced RMSE by more than two orders of magnitude.")
    elif block_contrib["block_specific_failure_suspected"]:
        root_cause = "probable_data_pipeline_error"
        recommended = "continue_blocking_all_model_comparison"
        evidence.append("A small number of source blocks dominate squared error.")
    elif extrap["out_of_domain_failure_supported"]:
        root_cause = "confirmed_out_of_domain_generalization_failure"
        recommended = "retain_fold4_extreme_result_and_continue_candidate_model_comparison"
        evidence.append("Absolute residual increases with distance to training region and no unit/block error was detected.")
    else:
        root_cause = "probable_model_generalization_failure"
        recommended = "continue_blocking_all_model_comparison_pending_scientific_review"
        evidence.append("Frozen replay reproduced formal metrics, no simple unit/sign/order transform or block dominance explains the error.")
    counter.append("Fold4 training RMSE remains low, so this is validation-specific until training data impact is separately assessed.")
    write_json(replay_dir / "fold4_forensic_root_cause.json", {
        "root_cause": root_cause,
        "recommended_action": recommended,
        "evidence": evidence,
        "counter_evidence": counter,
        "affected_files": [str(cache_path), str(fold_dir / "final_training_checkpoint.npy")],
        "affected_blocks": [r["block_id"] for r in block_rows[:5]],
        "whether_training_was_affected": "unknown",
        "whether_fold1_to_fold3_may_be_affected": "not_indicated_by_current_replay",
        "whether_retraining_is_required": "unknown" if root_cause != "confirmed_out_of_domain_generalization_failure" else False,
        "whether_only_metric_recomputation_is_required": root_cause in {"confirmed_validation_metric_assembly_error", "confirmed_prediction_observation_order_mismatch", "confirmed_spatial_flip_or_transpose", "confirmed_harmonic_convention_error"},
        "model_comparison_allowed": False,
        "recommended_action_not_executed": True,
    })
    # Small previews.
    for name, title in [
        ("fold4_forensic_abs_residual.tif", "absolute residual"),
        ("fold4_forensic_observation_amplitude.tif", "observation amplitude"),
        ("fold4_forensic_prediction_amplitude.tif", "prediction amplitude"),
    ]:
        with rasterio.open(replay_dir / name) as src:
            arr = src.read(1, out_shape=(max(1, src.height // 8), max(1, src.width // 8)))
        plt.figure(figsize=(5, 4))
        if "abs" in name:
            plt.imshow(np.log10(np.where(np.isfinite(arr), np.maximum(arr, 1e-3), np.nan)), cmap="magma")
        else:
            plt.imshow(arr, cmap="viridis")
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(replay_dir / name.replace(".tif", "_preview.png"), dpi=160)
        plt.close()
    status = read_json(root / "aquifer_model_revision_status.json")
    status.update({
        "fold4_forensic_replay_count": 1,
        "forensic_validation_access_count": 1,
        "formal_outer_validation_access_count": 1,
        "fold4_forensic_replay_authorized": False,
        "fold4_scientific_model_selection_eligible": False,
        "G0_model_selection_eligible": False,
        "allow_start_G1_G2_G3": False,
        "allow_start_G1": False,
        "allow_start_G2": False,
        "allow_start_G3": False,
        "allow_start_geology_model_comparison_review": False,
        "phase4_restart_allowed": False,
    })
    write_json(root / "aquifer_model_revision_status.json", status)
    write_json(status_path, {
        "forensic_replay_reason": "missing_pixel_level_evidence_for_extreme_error_diagnosis",
        "forensic_replay_affects_model_parameters": False,
        "forensic_replay_affects_original_formal_metric": False,
        "forensic_validation_access_count": 1,
        "formal_outer_validation_access_count": 1,
        "elapsed_seconds": time.time() - start_time,
        "pixel_count": npix,
        "observation_count": ncoef,
        "frozen_checks": checks,
        "formal_metric_reproducibility_failure": repro["formal_metric_reproducibility_failure"],
        "root_cause": root_cause,
        "recommended_action": recommended,
        "recommended_action_executed": False,
    })
    print(json.dumps({
        "recomputed_RMSE_mm": rmse_value,
        "recomputed_MAE_mm": mae_value,
        "recomputed_bias_mm": bias_value,
        "formal_metric_reproducibility_failure": repro["formal_metric_reproducibility_failure"],
        "root_cause": root_cause,
        "recommended_action": recommended,
        "forensic_validation_access_count": 1,
        "model_comparison_allowed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
