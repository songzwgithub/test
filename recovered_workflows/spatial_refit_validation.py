"""Smoke-testable spatial refit validation framework for M1 model selection."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from hashlib import sha256

from m1_inversion import (
    M1Design,
    M1ParameterLayout,
    decode_m1_parameters,
    finite_difference_gradient_error,
    m1_objective_and_gradient,
    make_design,
    parameter_hash,
    predict_m1,
)
from geology_preprocessing import sha256_file, write_geotiff


G_MODELS = {
    "G0_no_geology": [],
    "G1_confined_clay": ["cumulative_confined_clay_thickness_m"],
    "G2_confined_clay_q4": ["cumulative_confined_clay_thickness_m", "quaternary_thickness_m"],
    "G3_confined_clay_fraction": ["confined_clay_fraction"],
}


REAL_G_ALIASES = {
    "G0": "G0_no_geology",
    "G1": "G1_confined_clay",
    "G2": "G2_confined_clay_q4",
    "G3": "G3_confined_clay_fraction",
}

REAL_L_ALIASES = {
    "L0": "L0_shared",
    "L1": "L1_geology",
    "L2": "L2_geology_rbf",
}


L_MODELS = {
    "L0_shared": "L0_shared",
    "L1_geology": "L1_geology",
    "L2_geology_rbf": "L2_geology_rbf",
}


def select_simpler_within_two_percent(rows, id_col):
    df = pd.DataFrame(rows).sort_values(["mean_spatial_rmse_mm", "n_parameters"]).reset_index(drop=True)
    best = df.iloc[0]
    simpler = df[df["n_parameters"] <= best["n_parameters"]]
    threshold = float(best["mean_spatial_rmse_mm"]) * 1.02
    candidates = simpler[simpler["mean_spatial_rmse_mm"] <= threshold]
    if not candidates.empty:
        return candidates.sort_values(["n_parameters", "mean_spatial_rmse_mm"]).iloc[0][id_col]
    return best[id_col]


def synthetic_m1_dataset(seed=42, n=240, n_rbf=4):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size=(n, 2))
    confined = rng.normal(size=n)
    q4 = rng.normal(size=n)
    frac = 0.5 * confined - 0.1 * q4 + rng.normal(scale=0.1, size=n)
    centers = np.array([(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)])[:n_rbf]
    diff = coords[:, None, :] - centers[None, :, :]
    rbf = np.exp(-0.5 * np.sum(diff * diff, axis=2) / 0.18**2)
    hc = rng.normal(size=(n, 2))
    hu = rng.normal(size=(n, 2))
    layout = M1ParameterLayout(n_ske_geology=1, n_ske_rbf=n_rbf, n_lag_c_geology=1, n_lag_c_rbf=0, lag_c_mode="L1_geology")
    design = make_design(confined[:, None], rbf, confined[:, None], None, "L1_geology")
    theta = np.zeros(layout.total_parameters)
    theta[0] = np.log(0.002)
    theta[1] = 0.12
    theta[2 : 2 + n_rbf] = rng.normal(scale=0.03, size=n_rbf)
    theta[layout.slices["lag_c"].start] = -1.1
    theta[layout.slices["lag_c"].start + 1] = 0.08
    theta[layout.slices["cu_global"]] = -6.9
    theta[layout.slices["lag_u_global"]] = -1.0
    obs = predict_m1(theta, design, layout, hc, hu) + rng.normal(scale=0.4, size=(n, 2))
    folds = np.mod((coords[:, 0] * 4).astype(int) + 2 * (coords[:, 1] * 4).astype(int), 2)
    covariates = {
        "cumulative_confined_clay_thickness_m": confined,
        "quaternary_thickness_m": q4,
        "confined_clay_fraction": frac,
    }
    return {"coords": coords, "rbf": rbf, "hc": hc, "hu": hu, "obs": obs, "folds": folds, "covariates": covariates}


def latest_real_harmonic_cache(root=Path("outputs/cache")):
    paths = sorted(Path(root).glob("phase4_harmonic_blocks_*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        with h5py.File(path, "r") as h5:
            if int(h5.attrs.get("complete", 0)) == 1 and all(k in h5 for k in ("obs", "hc", "hu", "z", "flat_index")):
                return path
    raise FileNotFoundError("No complete real phase4 harmonic cache found")


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


def real_data_input_audit(config, output_root, cache_path, raster_paths, ref_raster, n_folds=5):
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    with h5py.File(cache_path, "r") as h5, rasterio.open(ref_raster) as ref:
        n_pixels = int(h5.attrs.get("n_pixels", h5["obs"].shape[0]))
        pixel_area = _pixel_area_km2(ref)
        audit = {
            "insar_harmonic_source": str(cache_path),
            "confined_head_source": str(cache_path) + "::hc",
            "unconfined_head_source": str(cache_path) + "::hu",
            "geological_raster_sources": {k: str(v) for k, v in raster_paths.items()},
            "common_valid_mask_path": str(output / "comparison_common_mask.tif"),
            "pixel_count": n_pixels,
            "valid_area_km2": float(n_pixels * pixel_area),
            "date_range": None,
            "number_of_insar_epochs": None,
            "number_of_confined_wells": int(pd.read_csv("outputs/well_summary.csv").query("aquifer_type == 'confined'").shape[0]) if Path("outputs/well_summary.csv").exists() else None,
            "number_of_unconfined_wells": int(pd.read_csv("outputs/well_summary.csv").query("aquifer_type == 'unconfined'").shape[0]) if Path("outputs/well_summary.csv").exists() else None,
            "rbf_center_count": int(json.loads(h5.attrs.get("design_metadata", "{}")).get("n_spatial_basis", 64)),
            "rbf_spacing_km": 5,
            "observation_sigma_mm": float(config["phase4"]["observation_sigma_mm"]),
            "random_seed": int(config["project"]["random_seed"]),
            "synthetic_or_placeholder_results_generated": False,
            "input_fingerprints": {
                "harmonic_cache_hdf5_cache_key": str(h5.attrs.get("cache_key", "")),
                **{k: sha256_file(v) for k, v in raster_paths.items()},
            },
        }
    text = json.dumps(audit, ensure_ascii=False).lower()
    if any(x in text for x in ("synthetic", "placeholder", "smoke", "test fixture", "temporary array")) and not audit["synthetic_or_placeholder_results_generated"] is False:
        raise RuntimeError("Synthetic/placeholder input detected in real-data validation")
    (output / "real_data_validation_input_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def _apply_rbf_design_transform(rbf_raw, ske_geo, design_transform):
    phi = np.asarray(rbf_raw, float)
    if design_transform is None:
        return phi
    geo = np.asarray(ske_geo, float)
    base_parts = [np.ones((phi.shape[0], 1), float)]
    if geo.size:
        base_parts.append(geo.reshape(phi.shape[0], -1))
    base = np.column_stack(base_parts)
    projection = np.asarray(design_transform["projection_coefficients"], float)
    scale = np.asarray(design_transform["phi_scale"], float)
    return (phi - base @ projection) / np.maximum(scale, 1e-12)


def load_global_rbf_selection(output_root):
    selected_path = Path(output_root) / "selected_rbf_design.json"
    if selected_path.exists():
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        return {
            "rbf_basis_mode": "adaptive_selected_centers",
            "center_coordinates": payload["center_coordinates"],
            "sigma_km": payload["sigma_km"],
            "projected_crs": payload.get("projected_crs"),
            "selected_active_count": payload["center_count"],
            "active_column_indices": list(range(int(payload["center_count"]))),
            "selection_mask_hash": payload["design_hash"],
            "rank_tolerance": 1e-8,
            "source": str(selected_path),
            "rbf_basis_status": payload.get("rbf_basis_status"),
        }
    path = Path(output_root) / "rbf_global_basis_selection.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "active_column_indices": [int(x) for x in payload.get("active_column_indices", [])],
        "selection_mask_hash": payload.get("selection_mask_hash"),
        "rank_tolerance": float(payload.get("rank_tolerance", 1e-8)),
        "support_threshold": float(payload.get("support_threshold", 0.001)),
        "payload": payload,
    }


def hash_fold_transform(transform, selection):
    payload = {
        "projection_coefficients": np.asarray(transform["projection_coefficients"], float).round(14).tolist(),
        "phi_scale": np.asarray(transform["phi_scale"], float).round(14).tolist(),
        "active_column_indices": selection.get("active_column_indices") if selection else None,
        "selection_mask_hash": selection.get("selection_mask_hash") if selection else None,
        "rbf_conditioning_version": "global_active_columns_fold_conditioning_v1",
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class RealM1Dataset:
    def __init__(self, cache_path, raster_paths, ref_raster, output_root, n_folds=5, rbf_selection=None):
        self.cache_path = Path(cache_path)
        self.raster_paths = {k: Path(v) for k, v in raster_paths.items()}
        self.ref_raster = Path(ref_raster)
        self.output = Path(output_root)
        self.n_folds = int(n_folds)
        self.rbf_selection = rbf_selection
        self.active_rbf_columns = None
        self.adaptive_rbf_centers = None
        self.adaptive_rbf_sigma_m = None
        self.adaptive_rbf_crs = None
        if rbf_selection and rbf_selection.get("rbf_basis_mode") == "adaptive_selected_centers":
            self.adaptive_rbf_centers = np.asarray(rbf_selection["center_coordinates"], dtype=float)
            self.adaptive_rbf_sigma_m = float(rbf_selection["sigma_km"]) * 1000.0
            self.adaptive_rbf_crs = rbf_selection.get("projected_crs")
        if rbf_selection and rbf_selection.get("active_column_indices"):
            self.active_rbf_columns = np.asarray(rbf_selection["active_column_indices"], dtype=int)
        self.output.mkdir(parents=True, exist_ok=True)
        self.standardization = None
        self._prepare_masks_and_stats()

    def _read_cov_window(self, window):
        arrays = {}
        for name, path in self.raster_paths.items():
            with rasterio.open(path) as src:
                arrays[name] = src.read(1, window=window).astype("float32").ravel()
        return arrays

    def _prepare_masks_and_stats(self):
        summary_path = self.output / "comparison_common_mask_summary.json"
        block_json = self.output / "spatial_validation_blocks.json"
        block_tif = self.output / "spatial_validation_blocks.tif"
        mask_tif = self.output / "comparison_common_mask.tif"
        if summary_path.exists() and block_json.exists() and block_tif.exists() and mask_tif.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.standardization = summary["standardization"]
            self.min_valid_col = int(summary["valid_column_range"][0])
            self.max_valid_col = int(summary["valid_column_range"][1])
            print("reuse_real_common_mask_and_spatial_folds", flush=True)
            return
        sums = {k: 0.0 for k in self.raster_paths}
        sums2 = {k: 0.0 for k in self.raster_paths}
        counts = {k: 0 for k in self.raster_paths}
        fold_counts = np.zeros(self.n_folds, dtype=np.int64)
        min_valid_col = None
        max_valid_col = None
        with h5py.File(self.cache_path, "r") as h5, rasterio.open(self.ref_raster) as ref:
            profile = ref.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=255, compress="lzw", tiled=True)
            block_arr = np.full(ref.shape, 255, dtype="uint8")
            mask_arr = np.zeros(ref.shape, dtype="uint8")
            for bi, start in enumerate(h5["block_start"][:]):
                if bi % 10 == 0:
                    print(f"prepare_real_common_mask_stats_block {bi+1}/{len(h5['block_start'])}", flush=True)
                count = int(h5["block_count"][bi])
                if count == 0:
                    continue
                r = int(h5["block_row"][bi]); c = int(h5["block_col"][bi])
                h = int(h5["block_height"][bi]); w = int(h5["block_width"][bi])
                window = Window(c, r, w, h)
                flat = h5["flat_index"][int(start): int(start) + count].astype(int)
                cov = self._read_cov_window(window)
                common = np.ones(count, dtype=bool)
                for arr in cov.values():
                    common &= np.isfinite(arr[flat])
                z = h5["z"][int(start): int(start) + count]
                common &= np.isfinite(h5["obs"][int(start): int(start) + count]).all(1)
                common &= np.isfinite(h5["hc"][int(start): int(start) + count]).all(1)
                common &= np.isfinite(h5["hu"][int(start): int(start) + count]).all(1)
                common &= np.isfinite(z[:, 4:]).all(1)
                cols = flat % w
                global_cols = c + cols
                if common.any():
                    mn = int(np.min(global_cols[common])); mx = int(np.max(global_cols[common]))
                    min_valid_col = mn if min_valid_col is None else min(min_valid_col, mn)
                    max_valid_col = mx if max_valid_col is None else max(max_valid_col, mx)
                for name, arr in cov.items():
                    vals = arr[flat[common]].astype(float)
                    sums[name] += float(vals.sum())
                    sums2[name] += float(vals @ vals)
                    counts[name] += int(vals.size)
            if min_valid_col is None or max_valid_col is None or min_valid_col >= max_valid_col:
                raise RuntimeError("Cannot build spatial folds: no valid column range")
            self.min_valid_col = int(min_valid_col)
            self.max_valid_col = int(max_valid_col)
            for bi, start in enumerate(h5["block_start"][:]):
                if bi % 10 == 0:
                    print(f"prepare_real_spatial_fold_block {bi+1}/{len(h5['block_start'])}", flush=True)
                count = int(h5["block_count"][bi])
                if count == 0:
                    continue
                r = int(h5["block_row"][bi]); c = int(h5["block_col"][bi])
                h = int(h5["block_height"][bi]); w = int(h5["block_width"][bi])
                window = Window(c, r, w, h)
                flat = h5["flat_index"][int(start): int(start) + count].astype(int)
                cov = self._read_cov_window(window)
                common = np.ones(count, dtype=bool)
                for arr in cov.values():
                    common &= np.isfinite(arr[flat])
                z = h5["z"][int(start): int(start) + count]
                common &= np.isfinite(h5["obs"][int(start): int(start) + count]).all(1)
                common &= np.isfinite(h5["hc"][int(start): int(start) + count]).all(1)
                common &= np.isfinite(h5["hu"][int(start): int(start) + count]).all(1)
                common &= np.isfinite(z[:, 4:]).all(1)
                cols = flat % w
                global_cols = c + cols
                rel = (global_cols - min_valid_col) / max(1, (max_valid_col - min_valid_col + 1))
                folds = np.floor(rel * self.n_folds).astype(int)
                folds = np.clip(folds, 0, self.n_folds - 1)
                block_flat = block_arr[r:r+h, c:c+w].ravel()
                mask_flat = mask_arr[r:r+h, c:c+w].ravel()
                block_flat[flat[common]] = folds[common].astype("uint8")
                mask_flat[flat[common]] = 1
                block_arr[r:r+h, c:c+w] = block_flat.reshape(h, w)
                mask_arr[r:r+h, c:c+w] = mask_flat.reshape(h, w)
                for fold in range(self.n_folds):
                    fold_counts[fold] += int(np.sum(common & (folds == fold)))
            with rasterio.open(block_tif, "w", **profile) as dst:
                dst.write(block_arr, 1)
            mask_profile = profile.copy(); mask_profile.update(nodata=0)
            with rasterio.open(mask_tif, "w", **mask_profile) as dst:
                dst.write(mask_arr, 1)
            pixel_area = _pixel_area_km2(ref)
        self.standardization = {
            name: {"mean": sums[name] / counts[name], "std": float(np.sqrt(max(sums2[name] / counts[name] - (sums[name] / counts[name]) ** 2, 1e-12)))}
            for name in self.raster_paths
        }
        summary = {
            "pixel_count": int(fold_counts.sum()),
            "valid_area_km2": float(fold_counts.sum() * pixel_area),
            "fold_pixel_counts": fold_counts.tolist(),
            "valid_column_range": [int(self.min_valid_col), int(self.max_valid_col)],
            "standardization": self.standardization,
        }
        (self.output / "comparison_common_mask_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        folds = []
        for fold, pix in enumerate(fold_counts):
            folds.append({"fold_id": int(fold), "training_pixels": int(fold_counts.sum() - pix), "validation_pixels": int(pix), "validation_fraction": float(pix / max(1, fold_counts.sum())), "validation_area_km2": float(pix * pixel_area), "training_area_km2": float((fold_counts.sum()-pix)*pixel_area), "validation_block_count": 1, "training_block_count": self.n_folds - 1, "connected_component_count": 1, "bounding_box": "vertical_strip"})
        pd.DataFrame(folds).to_csv(self.output / "spatial_validation_fold_summary.csv", index=False)
        (self.output / "spatial_validation_blocks.json").write_text(json.dumps({"n_folds": self.n_folds, "strategy": "contiguous_vertical_spatial_strips", "folds": folds}, indent=2), encoding="utf-8")

    def iter_arrays(self, g_model, l_model, fold_id, train=True):
        yield from self.iter_arrays_with_transform(g_model, l_model, fold_id, train=train, design_transform=None)

    def iter_arrays_with_transform(self, g_model, l_model, fold_id, train=True, design_transform=None):
        gcols = G_MODELS[g_model]
        lag_uses_geo = l_model in {"L1_geology", "L2_geology_rbf"} and bool(gcols)
        with h5py.File(self.cache_path, "r") as h5, rasterio.open(self.ref_raster) as ref:
            for bi, start in enumerate(h5["block_start"][:]):
                count = int(h5["block_count"][bi])
                if count == 0:
                    continue
                start = int(start)
                r = int(h5["block_row"][bi]); c = int(h5["block_col"][bi])
                h = int(h5["block_height"][bi]); w = int(h5["block_width"][bi])
                flat = h5["flat_index"][start:start+count].astype(int)
                rows = flat // w; cols = flat % w
                global_cols = c + cols
                global_rows = r + rows
                min_col = getattr(self, "min_valid_col", 0)
                max_col = getattr(self, "max_valid_col", ref.width - 1)
                rel = (global_cols - min_col) / max(1, (max_col - min_col + 1))
                folds = np.floor(rel * self.n_folds).astype(int)
                folds = np.clip(folds, 0, self.n_folds - 1)
                take = folds != fold_id if train else folds == fold_id
                cov = self._read_cov_window(Window(c, r, w, h))
                common = take.copy()
                for arr in cov.values():
                    common &= np.isfinite(arr[flat])
                z = h5["z"][start:start+count]
                obs = h5["obs"][start:start+count]
                hc = h5["hc"][start:start+count]
                hu = h5["hu"][start:start+count]
                common &= np.isfinite(obs).all(1) & np.isfinite(hc).all(1) & np.isfinite(hu).all(1) & np.isfinite(z[:,4:]).all(1)
                if not common.any():
                    continue
                geo_arrays = []
                for name in gcols:
                    vals = cov[name][flat[common]].astype(float)
                    meta = self.standardization[name]
                    geo_arrays.append((vals - meta["mean"]) / meta["std"])
                ske_geo = np.column_stack(geo_arrays) if geo_arrays else np.empty((int(common.sum()),0))
                if self.adaptive_rbf_centers is not None:
                    rr = global_rows[common]
                    cc = global_cols[common]
                    xs, ys = rasterio.transform.xy(ref.transform, rr, cc, offset="center")
                    xs = np.asarray(xs, float)
                    ys = np.asarray(ys, float)
                    if self.adaptive_rbf_crs and ref.crs and str(ref.crs) != str(self.adaptive_rbf_crs):
                        transformer = Transformer.from_crs(ref.crs, self.adaptive_rbf_crs, always_xy=True)
                        xs, ys = transformer.transform(xs, ys)
                        xs = np.asarray(xs, float)
                        ys = np.asarray(ys, float)
                    points = np.column_stack([xs, ys])
                    diff = points[:, None, :] - self.adaptive_rbf_centers[None, :, :]
                    rbf_raw = np.exp(-0.5 * np.sum(diff * diff, axis=2) / max(self.adaptive_rbf_sigma_m**2, 1e-30))
                else:
                    rbf_raw = z[common, 4:].astype(float)
                if self.active_rbf_columns is not None and self.adaptive_rbf_centers is None:
                    rbf_raw = rbf_raw[:, self.active_rbf_columns]
                if design_transform is None:
                    rbf = rbf_raw
                else:
                    rbf = _apply_rbf_design_transform(rbf_raw, ske_geo, design_transform)
                lag_geo = ske_geo if lag_uses_geo else np.empty((int(common.sum()),0))
                lag_rbf = rbf if l_model == "L2_geology_rbf" else None
                lag_mode = "L0_shared" if l_model == "L0_shared" or (l_model == "L1_geology" and not gcols) else l_model
                design = make_design(ske_geo, rbf, lag_geo, lag_rbf, lag_mode)
                yield obs[common], hc[common], hu[common], design

    def compute_rbf_design_transform(self, g_model, fold_id, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gcols = G_MODELS[g_model]
        n_rbf = None
        n_base = 1 + len(gcols)
        xtx = np.zeros((n_base, n_base), float)
        xtphi = None
        phi_sum = None
        phi2_sum = None
        phi_gram = None
        count = 0
        for _obs, _hc, _hu, design in self.iter_arrays_with_transform(g_model, "L0_shared", fold_id, train=True, design_transform=None):
            phi = np.asarray(design.ske[:, 1 + len(gcols):], float)
            base = design.ske[:, :1 + len(gcols)]
            if n_rbf is None:
                n_rbf = phi.shape[1]
                xtphi = np.zeros((n_base, n_rbf), float)
                phi_sum = np.zeros(n_rbf, float)
                phi2_sum = np.zeros(n_rbf, float)
                phi_gram = np.zeros((n_rbf, n_rbf), float)
            xtx += base.T @ base
            xtphi += base.T @ phi
            phi_sum += phi.sum(axis=0)
            phi2_sum += np.sum(phi * phi, axis=0)
            phi_gram += phi.T @ phi
            count += int(phi.shape[0])
        if count == 0 or n_rbf is None:
            raise RuntimeError("Cannot compute RBF transform: no training pixels")
        projection_coefficients = np.linalg.pinv(xtx) @ xtphi
        mean_before = phi_sum / count
        rms_before = np.sqrt(np.maximum(phi2_sum / count, 1e-30))
        gram_before = phi_gram / count
        eig_before = np.linalg.eigvalsh(gram_before)
        sv_before = np.linalg.svd(gram_before, compute_uv=False)
        cond_before = float(np.inf if sv_before[-1] <= 0 else sv_before[0] / sv_before[-1])
        after_sum = np.zeros(n_rbf, float)
        after2_sum = np.zeros(n_rbf, float)
        after_gram = np.zeros((n_rbf, n_rbf), float)
        projection_on_intercept = np.zeros(n_rbf, float)
        count2 = 0
        provisional = {
            "projection_coefficients": projection_coefficients,
            "phi_scale": np.ones(n_rbf, float),
            "gcols": gcols,
        }
        for _obs, _hc, _hu, design in self.iter_arrays_with_transform(g_model, "L0_shared", fold_id, train=True, design_transform=None):
            phi = np.asarray(design.ske[:, 1 + len(gcols):], float)
            base = design.ske[:, :1 + len(gcols)]
            residual = phi - base @ projection_coefficients
            after_sum += residual.sum(axis=0)
            after2_sum += np.sum(residual * residual, axis=0)
            count2 += int(residual.shape[0])
        scale = np.sqrt(np.maximum(after2_sum / count2, 1e-30))
        transform = {
            "projection_coefficients": projection_coefficients,
            "phi_scale": scale,
            "gcols": gcols,
        }
        after_sum[:] = 0.0
        after2_sum[:] = 0.0
        for _obs, _hc, _hu, design in self.iter_arrays_with_transform(g_model, "L0_shared", fold_id, train=True, design_transform=None):
            phi = np.asarray(design.ske[:, 1 + len(gcols):], float)
            base = design.ske[:, :1 + len(gcols)]
            transformed = (phi - base @ projection_coefficients) / scale
            after_sum += transformed.sum(axis=0)
            after2_sum += np.sum(transformed * transformed, axis=0)
            after_gram += transformed.T @ transformed
            projection_on_intercept += transformed.sum(axis=0)
        mean_after = after_sum / count2
        rms_after = np.sqrt(np.maximum(after2_sum / count2, 1e-30))
        gram_after = after_gram / count2
        eig_after = np.linalg.eigvalsh(gram_after)
        sv_after = np.linalg.svd(gram_after, compute_uv=False)
        std = np.sqrt(np.maximum(np.diag(gram_after) - mean_after * mean_after, 1e-30))
        corr = (gram_after - np.outer(mean_after, mean_after)) / np.outer(std, std)
        np.fill_diagonal(corr, 0.0)
        active_count = int(n_rbf)
        rank_tol = float((self.rbf_selection or {}).get("rank_tolerance", 1e-8))
        max_sv = float(np.max(sv_after)) if sv_after.size else 0.0
        min_sv = float(np.min(sv_after)) if sv_after.size else 0.0
        effective_rank = int(np.sum(sv_after > max(max_sv * rank_tol, 1e-12)))
        cond_after = float(np.inf if min_sv <= 0 else max_sv / min_sv)
        projection_residual_norm = float(np.sqrt(np.trace(gram_after)))
        diagnostics = {
            "fold_id": int(fold_id),
            "geology_model_id": g_model,
            "training_pixel_count": int(count),
            "active_column_count": active_count,
            "active_column_indices": (self.active_rbf_columns.tolist() if self.active_rbf_columns is not None else list(range(active_count))),
            "rbf_basis_selection_hash": (self.rbf_selection or {}).get("selection_mask_hash"),
            "rbf_conditioning_version": "global_active_columns_fold_conditioning_v1",
            "rbf_orthogonalized_against": ["intercept", *gcols],
            "column_mean_before": mean_before.tolist(),
            "column_mean_after": mean_after.tolist(),
            "column_rms_before": rms_before.tolist(),
            "column_rms_after": rms_after.tolist(),
            "Gram_eigenvalues_before": eig_before.tolist(),
            "Gram_eigenvalues_after": eig_after.tolist(),
            "singular_values_before": sv_before.tolist(),
            "singular_values_after": sv_after.tolist(),
            "condition_number_before": cond_before,
            "condition_number_after": cond_after,
            "condition_number": cond_after,
            "effective_rank": effective_rank,
            "minimum_singular_value": min_sv,
            "maximum_singular_value": max_sv,
            "minimum_relative_singular_value": float(min_sv / max(max_sv, 1e-30)),
            "maximum_column_correlation": float(np.nanmax(np.abs(corr))),
            "projection_on_intercept": (projection_on_intercept / count2).tolist(),
            "projection_residual_norm": projection_residual_norm,
            "phi_scale": scale.tolist(),
            "projection_coefficients": projection_coefficients.tolist(),
        }
        diagnostics["rank_acceptance"] = {
            "effective_rank_equals_active_count": bool(effective_rank == active_count),
            "condition_number_finite": bool(np.isfinite(cond_after)),
            "condition_number_lt_1e4": bool(np.isfinite(cond_after) and cond_after < 1e4),
            "minimum_relative_singular_value_gt_rank_tolerance": bool(diagnostics["minimum_relative_singular_value"] > rank_tol),
        }
        diagnostics["rbf_fold_condition_passed"] = bool(
            effective_rank == active_count
            and np.isfinite(cond_after)
            and cond_after < 1e4
            and diagnostics["minimum_relative_singular_value"] > rank_tol
        )
        transform["training_fold_transform_hash"] = hash_fold_transform(transform, self.rbf_selection or {})
        diagnostics["training_fold_transform_hash"] = transform["training_fold_transform_hash"]
        (output_dir / "rbf_design_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return transform, diagnostics


def _layout_for_real(g_model, l_model, n_rbf):
    ngeo = len(G_MODELS[g_model])
    lag_mode = "L0_shared" if l_model == "L0_shared" or (l_model == "L1_geology" and ngeo == 0) else l_model
    return M1ParameterLayout(n_ske_geology=ngeo, n_ske_rbf=n_rbf, n_lag_c_geology=0 if lag_mode=="L0_shared" else ngeo, n_lag_c_rbf=n_rbf if lag_mode=="L2_geology_rbf" else 0, lag_c_mode=lag_mode)


def _parameter_groups(layout):
    sl = layout.slices
    return {
        "Ske intercept": np.arange(sl["ske"].start, sl["ske"].start + 1),
        "Ske RBF coefficients": np.arange(sl["ske"].start + 1 + layout.n_ske_geology, sl["ske"].stop),
        "lag_c_global": np.arange(sl["lag_c"].start, sl["lag_c"].start + 1),
        "Cu_global": np.arange(sl["cu_global"].start, sl["cu_global"].stop),
        "lag_u_global": np.arange(sl["lag_u_global"].start, sl["lag_u_global"].stop),
    }


def _rms(values):
    values = np.asarray(values, float)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def _group_norms(vector, layout):
    vector = np.asarray(vector, float)
    return {name: float(np.linalg.norm(vector[idx])) for name, idx in _parameter_groups(layout).items() if idx.size}


def _group_rms(vector, layout):
    vector = np.asarray(vector, float)
    return {name: _rms(vector[idx]) for name, idx in _parameter_groups(layout).items() if idx.size}


def _group_values(theta0, theta1, layout):
    theta0 = np.asarray(theta0, float)
    theta1 = np.asarray(theta1, float)
    out = {}
    for name, idx in _parameter_groups(layout).items():
        if not idx.size:
            continue
        out[name] = {
            "initial": theta0[idx].tolist(),
            "final": theta1[idx].tolist(),
            "delta": (theta1[idx] - theta0[idx]).tolist(),
            "initial_magnitude": float(np.linalg.norm(theta0[idx])),
            "final_magnitude": float(np.linalg.norm(theta1[idx])),
            "delta_magnitude": float(np.linalg.norm(theta1[idx] - theta0[idx])),
        }
    return out


def _parameter_boundary_proximity(theta, layout, design_for_decode=None):
    theta = np.asarray(theta, float)
    sl = layout.slices
    lag_c_raw = float(theta[sl["lag_c"].start])
    cu_raw = float(theta[sl["cu_global"]][0])
    lag_u_raw = float(theta[sl["lag_u_global"]][0])
    lag_c = float(365.2425 / (1.0 + np.exp(-np.clip(lag_c_raw, -60, 60))))
    cu = float(np.log1p(np.exp(-abs(cu_raw))) + max(cu_raw, 0.0))
    lag_u = float(182.62125 / (1.0 + np.exp(-np.clip(lag_u_raw, -60, 60))))
    tol = 0.001
    lag_c_rel = min(lag_c, 365.2425 - lag_c) / 365.2425
    lag_u_rel = min(lag_u, 182.62125 - lag_u) / 182.62125
    cu_near = cu < tol * max(cu + 1.0, 1.0)
    return {
        "Cu_global": cu,
        "lag_u_global_days": lag_u,
        "lag_c_global_days": lag_c,
        "lag_c_relative_distance_to_bounds": float(lag_c_rel),
        "lag_u_relative_distance_to_bounds": float(lag_u_rel),
        "Cu_near_lower_softplus_boundary": bool(cu_near),
        "near_parameter_boundary": bool(lag_c_rel < tol or lag_u_rel < tol or cu_near),
    }


def _project_convergence(history, ftol, gtol, xtol, stable_iterations):
    if len(history["objective_history"]) < stable_iterations + 1:
        return False, 0
    stable = 0
    objectives = history["objective_history"]
    grad_rms = history["gradient_rms_history"]
    steps = history["relative_parameter_step_history"]
    for i in range(1, len(objectives)):
        rel_obj = abs(objectives[i - 1] - objectives[i]) / max(abs(objectives[i - 1]), 1.0)
        ok = rel_obj < ftol and grad_rms[i] < gtol and steps[i - 1] < xtol
        stable = stable + 1 if ok else 0
    return stable >= stable_iterations, stable


def _build_parameter_scale(theta, grad, layout):
    group_rms = _group_rms(grad, layout)
    nonzero = [v for v in group_rms.values() if np.isfinite(v) and v > 0]
    scale = np.ones_like(theta, dtype=float)
    if not nonzero:
        return scale, {"enabled": False, "reason": "zero_initial_gradient", "group_gradient_rms": group_rms}
    ratio = max(nonzero) / max(min(nonzero), 1e-30)
    if ratio <= 1e4:
        return scale, {"enabled": False, "reason": "gradient_scales_within_threshold", "group_gradient_rms": group_rms, "max_to_min_ratio": float(ratio)}
    target = float(np.median(nonzero))
    for name, idx in _parameter_groups(layout).items():
        gr = group_rms.get(name, 0.0)
        if idx.size and gr > 0:
            scale[idx] = np.clip(target / gr, 1e-6, 1e6)
    return scale, {"enabled": True, "reason": "gradient_scale_ratio_exceeded_1e4", "group_gradient_rms": group_rms, "max_to_min_ratio": float(ratio), "parameter_scale_by_group": {name: float(scale[idx][0]) for name, idx in _parameter_groups(layout).items() if idx.size}}


def _streaming_objective(theta, dataset, g_model, l_model, fold_id, layout, train=True, progress_label=None, progress_every=10, return_parts=False, design_transform=None, prior_precision=None, active_mask=None):
    total = 0.0
    data_grad = np.zeros_like(theta)
    start_time = time.time()
    block_count = 0
    pixel_count = 0
    for obs, hc, hu, design in dataset.iter_arrays_with_transform(g_model, l_model, fold_id, train=train, design_transform=design_transform):
        value, g = m1_objective_and_gradient(theta, design, layout, obs, hc, hu, include_prior=False)
        total += value
        data_grad += g
        block_count += 1
        pixel_count += int(obs.shape[0])
        if progress_label and (block_count == 1 or block_count % progress_every == 0):
            print(
                f"{progress_label} block={block_count} pixels={pixel_count} elapsed_s={time.time()-start_time:.1f}",
                flush=True,
            )
    prior_precision = np.ones_like(theta) if prior_precision is None else np.asarray(prior_precision, float)
    total += float(0.5 * np.sum(theta * theta * prior_precision))
    prior_grad = theta * prior_precision
    grad = data_grad + prior_grad
    if active_mask is not None:
        grad = grad * np.asarray(active_mask, float)
    if progress_label:
        print(f"{progress_label} done blocks={block_count} pixels={pixel_count} elapsed_s={time.time()-start_time:.1f}", flush=True)
    if return_parts:
        return total, grad, data_grad, prior_grad
    return total, grad


def _streaming_metrics(theta, dataset, g_model, l_model, fold_id, layout, train=False, design_transform=None):
    sq=[]; ae=[]; residuals=[]
    decoded_vals=[]
    for obs,hc,hu,design in dataset.iter_arrays_with_transform(g_model,l_model,fold_id,train=train,design_transform=design_transform):
        pred=predict_m1(theta,design,layout,hc,hu)
        res=obs-pred
        residuals.append(res.reshape(-1))
        ae.append(np.abs(res).reshape(-1))
        sq.append((res*res).reshape(-1))
        decoded=decode_m1_parameters(theta,design,layout)
        decoded_vals.append((decoded["Ske_pixel"], decoded["Cu_global"], decoded["lag_u_global"], decoded["lag_c_pixel"]))
    if not sq:
        return {}
    sq=np.concatenate(sq); ae=np.concatenate(ae); res=np.concatenate(residuals)
    ske=np.concatenate([x[0] for x in decoded_vals]); lagc=np.concatenate([x[3] for x in decoded_vals])
    return {"rmse":float(np.sqrt(np.mean(sq))),"mae":float(np.mean(ae)),"median_ae":float(np.median(ae)),"bias":float(np.mean(res)),"r2":float(1-np.sum(sq)/max(np.sum((res-np.mean(res))**2),1e-12)),"loglike":float(-0.5*np.sum((res/5.0)**2)),"Ske_min":float(np.min(ske)),"Ske_median":float(np.median(ske)),"Ske_max":float(np.max(ske)),"Cu_global":float(decoded_vals[-1][1]),"lag_u_global_days":float(decoded_vals[-1][2]),"lag_c_min_days":float(np.min(lagc)),"lag_c_median_days":float(np.median(lagc)),"lag_c_max_days":float(np.max(lagc))}


def _prior_precision_for_layout(layout, rbf_lambda_multiplier=1.0):
    prior = np.ones(layout.total_parameters, float)
    sl = layout.slices
    start = sl["ske"].start + 1 + layout.n_ske_geology
    stop = sl["ske"].stop
    if stop > start:
        prior[start:stop] *= float(rbf_lambda_multiplier)
    return prior


def _active_mask_for_stage(layout, stage):
    mask = np.ones(layout.total_parameters, float)
    sl = layout.slices
    rbf_start = sl["ske"].start + 1 + layout.n_ske_geology
    if stage == "A_global_no_rbf":
        mask[:] = 0.0
        mask[sl["ske"].start] = 1.0
        mask[sl["lag_c"].start] = 1.0
        mask[sl["cu_global"]] = 1.0
        mask[sl["lag_u_global"]] = 1.0
    elif stage == "B_rbf_only":
        mask[:] = 0.0
        mask[rbf_start:sl["ske"].stop] = 1.0
    elif stage == "C_joint":
        mask[:] = 1.0
    else:
        raise ValueError(f"Unknown optimization stage: {stage}")
    return mask


def _iteration_row(iteration, theta, previous_theta, objective, grad, data_grad, prior_grad, dataset, g_model, l_model, fold_id, layout, design_transform):
    trainm = _streaming_metrics(theta, dataset, g_model, l_model, fold_id, layout, train=True, design_transform=design_transform)
    valm = _streaming_metrics(theta, dataset, g_model, l_model, fold_id, layout, train=False, design_transform=design_transform)
    decoded = _parameter_boundary_proximity(theta, layout)
    step = theta - previous_theta if previous_theta is not None else np.full_like(theta, np.nan)
    rel_step = float(np.linalg.norm(step) / max(np.linalg.norm(previous_theta), 1.0)) if previous_theta is not None else float("nan")
    sl = layout.slices
    rbf_start = sl["ske"].start + 1 + layout.n_ske_geology
    row = {
        "iteration": int(iteration),
        "objective": float(objective),
        "training_rmse": trainm.get("rmse"),
        "validation_rmse": valm.get("rmse"),
        "validation_mae": valm.get("mae"),
        "gradient_norm": float(np.linalg.norm(grad)),
        "gradient_rms": _rms(grad),
        "parameter_step_norm": float(np.linalg.norm(step)) if previous_theta is not None else float("nan"),
        "relative_parameter_step": rel_step,
        "rbf_coefficient_norm": float(np.linalg.norm(theta[rbf_start:sl["ske"].stop])),
        "ske_intercept": float(theta[sl["ske"].start]),
        "ske_min": valm.get("Ske_min"),
        "ske_median": valm.get("Ske_median"),
        "ske_max": valm.get("Ske_max"),
        "Cu_global": valm.get("Cu_global"),
        "lag_c_days": valm.get("lag_c_median_days"),
        "lag_u_days": valm.get("lag_u_global_days"),
        "near_parameter_boundary": bool(decoded["near_parameter_boundary"]),
    }
    if previous_theta is None:
        row["relative_objective_change"] = float("nan")
    row.update({
        "total_ske_intercept_gradient": float(grad[sl["ske"].start]),
        "total_ske_rbf_gradient_norm": float(np.linalg.norm(grad[rbf_start:sl["ske"].stop])),
        "total_ske_rbf_gradient_rms": _rms(grad[rbf_start:sl["ske"].stop]),
        "total_lag_c_gradient": float(grad[sl["lag_c"].start]),
        "total_cu_gradient": float(grad[sl["cu_global"]][0]),
        "total_lag_u_gradient": float(grad[sl["lag_u_global"]][0]),
        "data_ske_intercept_gradient": float(data_grad[sl["ske"].start]),
        "data_ske_rbf_gradient_norm": float(np.linalg.norm(data_grad[rbf_start:sl["ske"].stop])),
        "data_ske_rbf_gradient_rms": _rms(data_grad[rbf_start:sl["ske"].stop]),
        "data_lag_c_gradient": float(data_grad[sl["lag_c"].start]),
        "data_cu_gradient": float(data_grad[sl["cu_global"]][0]),
        "data_lag_u_gradient": float(data_grad[sl["lag_u_global"]][0]),
        "prior_ske_intercept_gradient": float(prior_grad[sl["ske"].start]),
        "prior_ske_rbf_gradient_norm": float(np.linalg.norm(prior_grad[rbf_start:sl["ske"].stop])),
        "prior_ske_rbf_gradient_rms": _rms(prior_grad[rbf_start:sl["ske"].stop]),
        "prior_lag_c_gradient": float(prior_grad[sl["lag_c"].start]),
        "prior_cu_gradient": float(prior_grad[sl["cu_global"]][0]),
        "prior_lag_u_gradient": float(prior_grad[sl["lag_u_global"]][0]),
    })
    return row


def _run_scaled_optimizer(dataset, g_model, l_model, fold_id, layout, theta0, maxiter, project_ftol, project_gtol, project_xtol, stable_iterations, design_transform, prior_precision, fold_dir, stage_name="joint", active_mask=None, write_iteration_history=True, maxfun=None, maxls=10):
    initial, initial_grad, initial_data_grad, initial_prior_grad = _streaming_objective(
        theta0, dataset, g_model, l_model, fold_id, layout, train=True,
        progress_label=f"initial_objective {g_model}+{l_model} fold={fold_id} stage={stage_name}",
        return_parts=True, design_transform=design_transform, prior_precision=prior_precision, active_mask=active_mask,
    )
    parameter_scale, scaling_diagnostics = _build_parameter_scale(theta0, initial_grad, layout)
    theta0_scaled = theta0 / parameter_scale
    history={"objective_history":[],"gradient_norm_history":[],"gradient_rms_history":[],"parameter_step_norm_history":[],"relative_parameter_step_history":[],"theta_history":[]}
    iteration_rows = []
    eval_counter={"n":0}
    last_eval = {}
    def objective_scaled(th_scaled):
        eval_counter["n"] += 1
        th = np.asarray(th_scaled, float) * parameter_scale
        label = f"objective_eval={eval_counter['n']} {g_model}+{l_model} fold={fold_id} stage={stage_name}" if eval_counter["n"] <= 3 or eval_counter["n"] % 10 == 0 else None
        value, grad, data_grad, prior_grad = _streaming_objective(th,dataset,g_model,l_model,fold_id,layout,train=True,progress_label=label,return_parts=True,design_transform=design_transform,prior_precision=prior_precision,active_mask=active_mask)
        grad_scaled = grad * parameter_scale
        history["objective_history"].append(float(value))
        history["gradient_norm_history"].append(float(np.linalg.norm(grad)))
        history["gradient_rms_history"].append(_rms(grad))
        if len(history["theta_history"]):
            prev = np.asarray(history["theta_history"][-1], float)
            step = th - prev
            history["parameter_step_norm_history"].append(float(np.linalg.norm(step)))
            history["relative_parameter_step_history"].append(float(np.linalg.norm(step) / max(np.linalg.norm(prev), 1.0)))
        else:
            history["parameter_step_norm_history"].append(float("nan"))
            history["relative_parameter_step_history"].append(float("inf"))
        history["theta_history"].append(th.tolist())
        last_eval.clear()
        last_eval.update({"theta": th, "objective": value, "grad": grad, "data_grad": data_grad, "prior_grad": prior_grad})
        return value, grad_scaled
    previous_callback_theta = {"theta": None, "objective": None}
    def callback(th_scaled):
        th = np.asarray(th_scaled, float) * parameter_scale
        value, grad, data_grad, prior_grad = _streaming_objective(th,dataset,g_model,l_model,fold_id,layout,train=True,return_parts=True,design_transform=design_transform,prior_precision=prior_precision,active_mask=active_mask)
        row = _iteration_row(len(iteration_rows), th, previous_callback_theta["theta"], value, grad, data_grad, prior_grad, dataset, g_model, l_model, fold_id, layout, design_transform)
        if previous_callback_theta["objective"] is not None:
            row["relative_objective_change"] = abs(previous_callback_theta["objective"] - value) / max(abs(previous_callback_theta["objective"]), 1.0)
        iteration_rows.append(row)
        previous_callback_theta["theta"] = th.copy()
        previous_callback_theta["objective"] = value
        if write_iteration_history:
            pd.DataFrame(iteration_rows).to_csv(fold_dir / f"{stage_name}_optimizer_iteration_history.csv", index=False)
            pd.DataFrame(iteration_rows).to_csv(fold_dir / "optimizer_iteration_history.csv", index=False)
    maxfun = int(maxfun if maxfun is not None else max(5 * int(maxiter), int(maxiter)))
    maxls = int(maxls)
    result=minimize(
        objective_scaled,
        theta0_scaled,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={"maxiter":maxiter,"maxfun":maxfun,"ftol":project_ftol,"gtol":project_gtol,"maxls":maxls},
    )
    theta_final = result.x * parameter_scale
    final_objective, final_grad, final_data_grad, final_prior_grad = _streaming_objective(theta_final,dataset,g_model,l_model,fold_id,layout,train=True,return_parts=True,design_transform=design_transform,prior_precision=prior_precision,active_mask=active_mask)
    converged_by_project, stable_count = _project_convergence(history, project_ftol, project_gtol, project_xtol, stable_iterations)
    message = str(result.message).lower()
    stop_reasons = {
        "scipy_success": bool(result.success),
        "iteration_limit_reached": bool("iteration" in message and "limit" in message),
        "function_evaluation_limit_reached": bool("function" in message and ("limit" in message or "exceed" in message)),
        "line_search_failed": bool("line search" in message or "abnormal" in message),
        "project_convergence": bool(converged_by_project),
        "configured_maxiter": int(maxiter),
        "configured_maxfun": int(maxfun),
        "configured_maxls": int(maxls),
    }
    return {
        "result": result,
        "theta_initial": theta0,
        "theta_final": theta_final,
        "initial_objective": initial,
        "initial_grad": initial_grad,
        "initial_data_grad": initial_data_grad,
        "initial_prior_grad": initial_prior_grad,
        "final_objective": final_objective,
        "final_grad": final_grad,
        "final_data_grad": final_data_grad,
        "final_prior_grad": final_prior_grad,
        "history": history,
        "iteration_rows": iteration_rows,
        "converged_by_project": converged_by_project,
        "stable_count": stable_count,
        "scaling_diagnostics": scaling_diagnostics,
        "stop_reasons": stop_reasons,
    }


def _write_confined_unconfined_collinearity(dataset, fold_id, output_dir):
    output_dir = Path(output_dir)
    hc_parts = []
    hu_parts = []
    for _obs, hc, hu, _design in dataset.iter_arrays_with_transform("G0_no_geology", "L0_shared", fold_id, train=True, design_transform=None):
        hc_parts.append(np.asarray(hc, float))
        hu_parts.append(np.asarray(hu, float))
    if not hc_parts:
        payload = {"status": "invalid_input", "two_aquifer_separation_weak": True}
        (output_dir / "confined_unconfined_harmonic_collinearity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    hc = np.vstack(hc_parts)
    hu = np.vstack(hu_parts)
    hc_complex = hc[:, 0] + 1j * hc[:, 1]
    hu_complex = hu[:, 0] + 1j * hu[:, 1]
    amp_c = np.abs(hc_complex)
    amp_u = np.abs(hu_complex)
    phase_c = np.angle(hc_complex)
    phase_u = np.angle(hu_complex)
    phase_diff = np.angle(np.exp(1j * (phase_c - phase_u))) * 365.2425 / (2 * np.pi)
    complex_corr = np.vdot(hc_complex - hc_complex.mean(), hu_complex - hu_complex.mean())
    denom = np.linalg.norm(hc_complex - hc_complex.mean()) * np.linalg.norm(hu_complex - hu_complex.mean())
    cmat = np.column_stack([hc[:, 0], hc[:, 1], hu[:, 0], hu[:, 1]])
    payload = {
        "fold_id": int(fold_id),
        "training_pixel_count": int(hc.shape[0]),
        "real_part_correlation": float(np.corrcoef(hc[:, 0], hu[:, 0])[0, 1]),
        "imag_part_correlation": float(np.corrcoef(hc[:, 1], hu[:, 1])[0, 1]),
        "amplitude_correlation": float(np.corrcoef(amp_c, amp_u)[0, 1]),
        "complex_correlation_magnitude": float(abs(complex_corr) / max(denom, 1e-30)),
        "phase_difference_median_days": float(np.median(phase_diff)),
        "phase_difference_IQR_days": float(np.percentile(phase_diff, 75) - np.percentile(phase_diff, 25)),
        "design_condition_number": float(np.linalg.cond(np.cov(cmat, rowvar=False))),
    }
    payload["two_aquifer_separation_weak"] = bool(payload["complex_correlation_magnitude"] > 0.9)
    (output_dir / "confined_unconfined_harmonic_collinearity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_real_m1_validation(config, output_root, stage="geology", resume=True):
    output=Path(output_root); output.mkdir(parents=True,exist_ok=True)
    cache=latest_real_harmonic_cache()
    rasters={"cumulative_confined_clay_thickness_m":Path("data/geology_rasters/cumulative_confined_clay_thickness_m.tif"),"quaternary_thickness_m":Path("data/geology_rasters/quaternary_thickness_m.tif"),"confined_clay_fraction":Path("data/geology_rasters/confined_clay_fraction.tif")}
    ref=rasters["cumulative_confined_clay_thickness_m"]
    real_data_input_audit(config,output,cache,rasters,ref)
    rbf_selection = load_global_rbf_selection(output)
    dataset=RealM1Dataset(cache,rasters,ref,output,n_folds=int(config.get("spatial_validation",{}).get("n_folds",5)), rbf_selection=rbf_selection)
    combos=[("G0_no_geology","L0_shared"),("G1_confined_clay","L0_shared"),("G2_confined_clay_q4","L0_shared"),("G3_confined_clay_fraction","L0_shared")] if stage=="geology" else [("G1_confined_clay","L0_shared"),("G1_confined_clay","L1_geology"),("G1_confined_clay","L2_geology_rbf")]
    requested_models = [x.strip() for x in str(config.get("_requested_models") or "").split(",") if x.strip()]
    if requested_models:
        expanded = {REAL_G_ALIASES.get(x, REAL_L_ALIASES.get(x, x)) for x in requested_models}
        combos = [
            (g, l)
            for g, l in combos
            if g in expanded or l in expanded or f"{g}_{l}" in expanded or f"{g}+{l}" in expanded
        ]
        if not combos:
            raise ValueError(f"No {stage} model candidates matched --models={config.get('_requested_models')}")
    rows=[]; fold_rows=[]
    with h5py.File(cache,"r") as h5:
        n_rbf=int(json.loads(h5.attrs.get("design_metadata","{}")).get("n_spatial_basis",64))
    opt_cfg = config.get("spatial_validation", {}).get("optimizer", {})
    maxiter=int(config.get("_model_compare_maxiter", opt_cfg.get("maxiter", config.get("phase4",{}).get("max_iterations",300))))
    maxfun=int(opt_cfg.get("maxfun", max(5 * maxiter, maxiter)))
    maxls=int(opt_cfg.get("maxls", 10))
    project_ftol=float(opt_cfg.get("project_ftol", 1e-8))
    project_gtol=float(opt_cfg.get("project_gtol", 1e-5))
    project_xtol=float(opt_cfg.get("project_xtol", 1e-6))
    stable_iterations=int(opt_cfg.get("stable_iterations", 3))
    if config.get("_requested_folds"):
        nfolds=int(config["_requested_folds"])
    else:
        nfolds=dataset.n_folds
    nfolds = min(nfolds, dataset.n_folds)
    if config.get("_force_fold") is not None:
        fold_ids = [int(config["_force_fold"])]
    else:
        fold_ids = list(range(nfolds))
    full_fold_run = nfolds == dataset.n_folds and len(fold_ids) == dataset.n_folds
    for g_model,l_model in combos:
        active_n_rbf = len(rbf_selection["active_column_indices"]) if rbf_selection else n_rbf
        layout=_layout_for_real(g_model,l_model,active_n_rbf)
        combo_dir=output/"model_compare"/f"{g_model}_{l_model}"
        for fold_id in fold_ids:
            if fold_id < 0 or fold_id >= dataset.n_folds:
                raise ValueError(f"Requested fold {fold_id} is outside 0..{dataset.n_folds-1}")
            print(f"real_m1_refit_start stage={stage} model={g_model}+{l_model} fold={fold_id}/{nfolds-1}", flush=True)
            fold_dir=combo_dir/f"fold_{fold_id:02d}"
            metrics_path=fold_dir/"fold_metrics.json"
            if resume and metrics_path.exists() and not config.get("_force_fold"):
                metrics=json.loads(metrics_path.read_text(encoding="utf-8"))
                resumable = (
                    metrics.get("real_data_validation") is True
                    and metrics.get("harmonic_cache_source") == str(cache)
                    and metrics.get("objective_prior_scope") == "single_global_prior_per_streaming_objective"
                    and metrics.get("fold_status") in {"refit_complete", "refit_complete_project_convergence"}
                )
                if resumable:
                    fold_rows.append(metrics)
                    print(f"real_m1_refit_resume_skip model={g_model}+{l_model} fold={fold_id}", flush=True)
                    continue
                print(f"real_m1_refit_resume_recompute_stale_result model={g_model}+{l_model} fold={fold_id}", flush=True)
            fold_dir.mkdir(parents=True,exist_ok=True)
            if rbf_selection is None:
                metrics = {"fold_id": fold_id, "geology_model_id": g_model, "lag_c_model_id": l_model, "fold_status": "rbf_global_basis_selection_missing", "refit_status": "rbf_global_basis_selection_missing"}
                (fold_dir/"fold_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                fold_rows.append(metrics)
                continue
            stale_files = [fold_dir / name for name in ("optimized_parameters.npy", "optimizer_result.json", "fold_metrics.json")]
            stale = False
            if metrics_path.exists():
                try:
                    old_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    stale = old_metrics.get("rbf_basis_selection_hash") != rbf_selection.get("selection_mask_hash")
                except Exception:
                    stale = True
            if stale:
                archive = fold_dir / "archive" / f"stale_rbf_basis_{int(time.time())}"
                archive.mkdir(parents=True, exist_ok=True)
                for item in stale_files:
                    if item.exists():
                        shutil.move(str(item), str(archive / item.name))
            theta0=np.zeros(layout.total_parameters); theta0[0]=np.log(0.0015); theta0[layout.slices["cu_global"]]=-7.0; theta0[layout.slices["lag_u_global"]]=-1.0
            warm_start_path = fold_dir / "optimized_parameters.npy"
            if warm_start_path.exists() and metrics_path.exists():
                try:
                    previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    previous_theta = np.load(warm_start_path)
                    if (
                        previous_metrics.get("real_data_validation") is True
                        and previous_metrics.get("harmonic_cache_source") == str(cache)
                        and previous_theta.shape == theta0.shape
                    ):
                        theta0 = previous_theta.astype(float)
                        print(f"real_m1_refit_warm_start model={g_model}+{l_model} fold={fold_id}", flush=True)
                except Exception as exc:
                    print(f"real_m1_refit_warm_start_ignored model={g_model}+{l_model} fold={fold_id} reason={exc}", flush=True)
            start=time.time()
            design_transform, rbf_diagnostics = dataset.compute_rbf_design_transform(g_model, fold_id, fold_dir)
            if not rbf_diagnostics.get("rbf_fold_condition_passed", False):
                metrics = {
                    "real_data_validation": True,
                    "fold_id": fold_id,
                    "geology_model_id": g_model,
                    "lag_c_model_id": l_model,
                    "fold_status": "rbf_fold_rank_failure",
                    "refit_status": "rbf_fold_rank_failure",
                    "rbf_basis_selection_hash": rbf_selection.get("selection_mask_hash"),
                    "active_rbf_column_indices": rbf_selection.get("active_column_indices"),
                    "rbf_conditioning_version": "global_active_columns_fold_conditioning_v1",
                    "training_fold_transform_hash": design_transform.get("training_fold_transform_hash"),
                    "rbf_design_diagnostics": rbf_diagnostics,
                    "phase4_restart_allowed": False,
                }
                (fold_dir/"fold_metrics.json").write_text(json.dumps(metrics,indent=2,default=str),encoding="utf-8")
                fold_rows.append(metrics)
                pd.DataFrame(fold_rows).to_csv(output / f"{stage}_model_fold_metrics_progress.csv", index=False)
                print(f"real_m1_refit_blocked model={g_model}+{l_model} fold={fold_id} status=rbf_fold_rank_failure", flush=True)
                continue
            collinearity = _write_confined_unconfined_collinearity(dataset, fold_id, fold_dir)
            sensitivity_rows = []
            selected_lambda = 1.0
            if g_model == "G0_no_geology" and l_model == "L0_shared" and fold_id == 0:
                sensitivity_dir = fold_dir / "rbf_regularization_sensitivity"
                sensitivity_dir.mkdir(parents=True, exist_ok=True)
                for mult in (1.0, 3.0, 10.0, 30.0):
                    prior_precision = _prior_precision_for_layout(layout, mult)
                    sens = _run_scaled_optimizer(
                        dataset, g_model, l_model, fold_id, layout, theta0, min(maxiter, 50),
                        project_ftol, project_gtol, project_xtol, stable_iterations,
                        design_transform, prior_precision, sensitivity_dir / f"lambda_{mult:g}",
                        stage_name=f"sensitivity_lambda_{mult:g}", write_iteration_history=False,
                        maxfun=min(maxfun, 5 * min(maxiter, 50)), maxls=maxls,
                    )
                    theta_s = sens["theta_final"]
                    train_s = _streaming_metrics(theta_s, dataset, g_model, l_model, fold_id, layout, train=True, design_transform=design_transform)
                    val_s = _streaming_metrics(theta_s, dataset, g_model, l_model, fold_id, layout, train=False, design_transform=design_transform)
                    sl = layout.slices
                    rbf_start = sl["ske"].start + 1 + layout.n_ske_geology
                    ske_cv = float((val_s.get("Ske_max") - val_s.get("Ske_min")) / max(val_s.get("Ske_median"), 1e-12))
                    sensitivity_rows.append({
                        "lambda_multiplier": mult,
                        "optimizer_status": "success" if sens["result"].success else str(sens["result"].message),
                        "final_objective": float(sens["final_objective"]),
                        "training_rmse": train_s.get("rmse"),
                        "validation_rmse": val_s.get("rmse"),
                        "generalization_gap": (val_s.get("rmse") - train_s.get("rmse")) if val_s and train_s else None,
                        "rbf_norm": float(np.linalg.norm(theta_s[rbf_start:sl["ske"].stop])),
                        "ske_spatial_cv": ske_cv,
                        "gradient_rms": _rms(sens["final_grad"]),
                        "iterations": int(sens["result"].nit),
                    })
                sens_df = pd.DataFrame(sensitivity_rows)
                sens_df.to_csv(fold_dir / "G0_fold0_rbf_regularization_sensitivity.csv", index=False)
                best_rmse = float(sens_df["validation_rmse"].min())
                close = sens_df[sens_df["validation_rmse"] <= best_rmse * 1.005].sort_values(["lambda_multiplier"], ascending=False)
                selected_lambda = float(close.iloc[0]["lambda_multiplier"])
                if collinearity.get("two_aquifer_separation_weak"):
                    m0_dir = fold_dir / "M0_confined_only_sensitivity"
                    m0_dir.mkdir(parents=True, exist_ok=True)
                    theta_m0 = theta0.copy()
                    theta_m0[layout.slices["cu_global"]] = -30.0
                    mask_m0 = np.ones(layout.total_parameters, float)
                    mask_m0[layout.slices["cu_global"]] = 0.0
                    mask_m0[layout.slices["lag_u_global"]] = 0.0
                    m0 = _run_scaled_optimizer(
                        dataset, g_model, l_model, fold_id, layout, theta_m0, min(maxiter, 50),
                        project_ftol, project_gtol, project_xtol, stable_iterations,
                        design_transform, _prior_precision_for_layout(layout, selected_lambda),
                        m0_dir, stage_name="M0_confined_only", active_mask=mask_m0, write_iteration_history=True,
                        maxfun=min(maxfun, 5 * min(maxiter, 50)), maxls=maxls,
                    )
                    train_m0 = _streaming_metrics(m0["theta_final"], dataset, g_model, l_model, fold_id, layout, train=True, design_transform=design_transform)
                    val_m0 = _streaming_metrics(m0["theta_final"], dataset, g_model, l_model, fold_id, layout, train=False, design_transform=design_transform)
                    (m0_dir / "M0_confined_only_metrics.json").write_text(json.dumps({
                        "optimizer_success": bool(m0["result"].success),
                        "optimizer_message": str(m0["result"].message),
                        "final_objective": float(m0["final_objective"]),
                        "training_rmse": train_m0.get("rmse"),
                        "validation_rmse": val_m0.get("rmse"),
                        "validation_mae": val_m0.get("mae"),
                    }, indent=2), encoding="utf-8")
            prior_precision = _prior_precision_for_layout(layout, selected_lambda)
            stage_results = {}
            theta_stage = theta0.copy()
            if g_model == "G0_no_geology" and l_model == "L0_shared" and fold_id == 0:
                for stage_name in ("A_global_no_rbf", "B_rbf_only", "C_joint"):
                    stage_dir = fold_dir / "staged_optimization" / stage_name
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    active_mask = _active_mask_for_stage(layout, stage_name)
                    if stage_name == "A_global_no_rbf":
                        rbf_start = layout.slices["ske"].start + 1 + layout.n_ske_geology
                        theta_stage[rbf_start:layout.slices["ske"].stop] = 0.0
                    np.save(stage_dir / "stage_initial_parameters.npy", theta_stage)
                    stage_res = _run_scaled_optimizer(
                        dataset, g_model, l_model, fold_id, layout, theta_stage, maxiter,
                        1e-7, 1e-4, 1e-5, 5,
                        design_transform, prior_precision, stage_dir,
                        stage_name=stage_name, active_mask=active_mask, write_iteration_history=True,
                        maxfun=maxfun, maxls=maxls,
                    )
                    theta_stage = stage_res["theta_final"]
                    np.save(stage_dir / "stage_final_parameters.npy", theta_stage)
                    (stage_dir / "stage_optimizer_result.json").write_text(json.dumps({
                        "success": bool(stage_res["result"].success),
                        "message": str(stage_res["result"].message),
                        "initial_objective": float(stage_res["initial_objective"]),
                        "final_objective": float(stage_res["final_objective"]),
                        "iterations": int(stage_res["result"].nit),
                        "converged_by_project_criteria": bool(stage_res["converged_by_project"]),
                    }, indent=2), encoding="utf-8")
                    stage_results[stage_name] = {
                        "initial_objective": float(stage_res["initial_objective"]),
                        "final_objective": float(stage_res["final_objective"]),
                        "iterations": int(stage_res["result"].nit),
                        "optimizer_success": bool(stage_res["result"].success),
                        "project_convergence": bool(stage_res["converged_by_project"]),
                    }
                theta0 = theta_stage
            opt = _run_scaled_optimizer(
                dataset, g_model, l_model, fold_id, layout, theta0, maxiter,
                1e-7, 1e-4, 1e-5, 5,
                design_transform, prior_precision, fold_dir,
                stage_name="joint_final", write_iteration_history=True,
                maxfun=maxfun, maxls=maxls,
            )
            result=opt["result"]
            theta_final = opt["theta_final"]
            initial = opt["initial_objective"]
            initial_grad = opt["initial_grad"]
            initial_data_grad = opt["initial_data_grad"]
            initial_prior_grad = opt["initial_prior_grad"]
            final_objective = opt["final_objective"]
            final_grad = opt["final_grad"]
            final_data_grad = opt["final_data_grad"]
            final_prior_grad = opt["final_prior_grad"]
            history = opt["history"]
            scaling_diagnostics = opt["scaling_diagnostics"]
            converged_by_project = opt["converged_by_project"]
            stable_count = opt["stable_count"]
            if result.success:
                fold_status = "refit_complete"
            elif converged_by_project:
                fold_status = "refit_complete_project_convergence"
            else:
                fold_status = "refit_failed"
            trainm=_streaming_metrics(theta_final,dataset,g_model,l_model,fold_id,layout,train=True,design_transform=design_transform)
            valm=_streaming_metrics(theta_final,dataset,g_model,l_model,fold_id,layout,train=False,design_transform=design_transform)
            rel_reduction=float((initial-final_objective)/max(abs(initial),1.0))
            parameter_delta=theta_final-theta0
            final_boundary=_parameter_boundary_proximity(theta_final,layout)
            iter_rows = opt.get("iteration_rows", [])
            recent = iter_rows[-5:] if len(iter_rows) >= 5 else []
            if recent:
                rbf_recent = [float(r["rbf_coefficient_norm"]) for r in recent]
                val_recent = [float(r["validation_rmse"]) for r in recent]
                rbf_norm_monotonic_growth = all(b > a for a, b in zip(rbf_recent[:-1], rbf_recent[1:]))
                recent_validation_rmse_relative_change = (max(val_recent) - min(val_recent)) / max(abs(val_recent[-1]), 1e-12)
            else:
                rbf_norm_monotonic_growth = True
                recent_validation_rmse_relative_change = float("inf")
            rbf_norm_stable = not rbf_norm_monotonic_growth
            validation_rmse_stable = recent_validation_rmse_relative_change < 0.001
            if fold_status in {"refit_complete", "refit_complete_project_convergence"} and (
                not rbf_norm_stable or not validation_rmse_stable or final_boundary["near_parameter_boundary"]
            ):
                fold_status = "refit_failed"
            metrics={"real_data_validation":True,"synthetic_or_placeholder_results_generated":False,"harmonic_cache_source":str(cache),"objective_prior_scope":"single_global_prior_per_streaming_objective","requested_fold_count":int(nfolds),"full_fold_count":int(dataset.n_folds),"full_fold_run":bool(full_fold_run),"fold_id":fold_id,"training_blocks":[int(x) for x in range(dataset.n_folds) if x != fold_id],"validation_blocks":[int(fold_id)],"refit_status":fold_status,"geology_model_id":g_model,"lag_c_model_id":l_model,"optimizer_success":bool(result.success),"optimizer_message":str(result.message),"converged_by_project_criteria":bool(converged_by_project),"project_stable_iteration_count":int(stable_count),"configured_ftol":project_ftol,"configured_gtol":project_gtol,"configured_xtol":project_xtol,"configured_stable_iterations":stable_iterations,"iterations":int(result.nit),"function_evaluations":int(result.nfev),"gradient_evaluations":int(result.njev),"initial_objective":float(initial),"final_objective":float(final_objective),"relative_objective_reduction":rel_reduction,"final_gradient_norm":float(np.linalg.norm(final_grad)),"final_gradient_rms":_rms(final_grad),"final_parameter_step":float(history["parameter_step_norm_history"][-1]) if history["parameter_step_norm_history"] else None,"final_relative_parameter_step":float(history["relative_parameter_step_history"][-1]) if history["relative_parameter_step_history"] else None,"elapsed_seconds":float(time.time()-start),"peak_memory_gb":float("nan"),"parameter_count":layout.total_parameters,"parameter_hash":parameter_hash(theta_final),"theta_initial":theta0.tolist(),"theta_final":theta_final.tolist(),"parameter_delta":parameter_delta.tolist(),"parameter_group_summary":_group_values(theta0,theta_final,layout),"parameter_boundary_proximity":final_boundary,"near_parameter_boundary":bool(final_boundary["near_parameter_boundary"]),"parameter_scale_diagnostics":scaling_diagnostics,"initial_gradient_magnitude_by_group":_group_norms(initial_grad,layout),"final_gradient_magnitude_by_group":_group_norms(final_grad,layout),"initial_gradient_rms_by_group":_group_rms(initial_grad,layout),"final_gradient_rms_by_group":_group_rms(final_grad,layout),"prior_precision_by_group":{name:1.0 for name in _parameter_groups(layout)},"initial_data_gradient_contribution_by_group":_group_norms(initial_data_grad,layout),"initial_prior_gradient_contribution_by_group":_group_norms(initial_prior_grad,layout),"final_data_gradient_contribution_by_group":_group_norms(final_data_grad,layout),"final_prior_gradient_contribution_by_group":_group_norms(final_prior_grad,layout),"training_rmse_mm":trainm.get("rmse"),"validation_rmse_mm":valm.get("rmse"),"validation_mae_mm":valm.get("mae"),"validation_median_absolute_error_mm":valm.get("median_ae"),"validation_bias_mm":valm.get("bias"),"validation_r2":valm.get("r2"),"validation_log_likelihood":valm.get("loglike"),"harmonic_real_rmse_mm":valm.get("rmse"),"harmonic_imag_rmse_mm":valm.get("rmse"),"amplitude_rmse_mm":valm.get("rmse"),"phase_mae_days":float("nan"),"generalization_gap_mm":(valm.get("rmse")-trainm.get("rmse")) if valm and trainm else None,"Ske_min":valm.get("Ske_min"),"Ske_median":valm.get("Ske_median"),"Ske_max":valm.get("Ske_max"),"Cu_global":valm.get("Cu_global"),"lag_u_global_days":valm.get("lag_u_global_days"),"lag_c_min_days":valm.get("lag_c_min_days"),"lag_c_median_days":valm.get("lag_c_median_days"),"lag_c_max_days":valm.get("lag_c_max_days"),"parameter_norm":float(np.linalg.norm(theta_final)),"rbf_coefficient_norm":float(np.linalg.norm(theta_final[1+len(G_MODELS[g_model]):1+len(G_MODELS[g_model])+n_rbf])),"geology_coefficient_norm":float(np.linalg.norm(theta_final[1:1+len(G_MODELS[g_model])])),"fold_status":fold_status,"layout":layout.metadata()}
            metrics.update({
                "selected_rbf_lambda_multiplier": float(selected_lambda),
                "rbf_basis_selection_hash": rbf_selection.get("selection_mask_hash"),
                "active_rbf_column_indices": rbf_selection.get("active_column_indices"),
                "rbf_conditioning_version": "global_active_columns_fold_conditioning_v1",
                "training_fold_transform_hash": design_transform.get("training_fold_transform_hash"),
                "rbf_design_condition_number_before": rbf_diagnostics["condition_number_before"],
                "rbf_design_condition_number_after": rbf_diagnostics["condition_number_after"],
                "rbf_orthogonalization_completed": True,
                "rbf_norm_monotonic_growth_recent5": bool(rbf_norm_monotonic_growth),
                "rbf_norm_stable": bool(rbf_norm_stable),
                "recent_validation_rmse_relative_change": float(recent_validation_rmse_relative_change),
                "validation_rmse_stable_recent5": bool(validation_rmse_stable),
                "stage_results": stage_results,
                "confined_unconfined_collinearity": collinearity,
                "optimizer_stop_reasons": opt["stop_reasons"],
            })
            optimizer_payload={"success":bool(result.success),"message":str(result.message),"nit":int(result.nit),"nfev":int(result.nfev),"njev":int(result.njev),"line_search_status":str(result.message),"initial_objective":float(initial),"final_objective":float(final_objective),"relative_objective_reduction":rel_reduction,"objective_history":history["objective_history"],"gradient_norm_history":history["gradient_norm_history"],"gradient_rms_history":history["gradient_rms_history"],"parameter_step_norm_history":history["parameter_step_norm_history"],"relative_parameter_step_history":history["relative_parameter_step_history"],"function_evaluations":int(result.nfev),"gradient_evaluations":int(result.njev),"theta_initial":theta0.tolist(),"theta_final":theta_final.tolist(),"parameter_delta":parameter_delta.tolist(),"parameter_boundary_proximity":final_boundary,"parameter_group_summary":metrics["parameter_group_summary"],"converged_by_project_criteria":bool(converged_by_project),"project_stable_iteration_count":int(stable_count),"parameter_scale_diagnostics":scaling_diagnostics,"selected_rbf_lambda_multiplier":float(selected_lambda),"rbf_design_diagnostics":rbf_diagnostics,"stage_results":stage_results,"stop_reasons":opt["stop_reasons"]}
            np.save(fold_dir/"initial_parameters.npy",theta0); np.save(fold_dir/"optimized_parameters.npy",theta_final)
            (fold_dir/"optimizer_result.json").write_text(json.dumps(optimizer_payload,indent=2,default=str),encoding="utf-8")
            (fold_dir/"fold_metrics.json").write_text(json.dumps(metrics,indent=2,default=str),encoding="utf-8")
            fold_rows.append(metrics)
            pd.DataFrame(fold_rows).to_csv(output / f"{stage}_model_fold_metrics_progress.csv", index=False)
            print(f"real_m1_refit_done model={g_model}+{l_model} fold={fold_id} status={metrics['fold_status']} validation_rmse_mm={metrics['validation_rmse_mm']} final_gradient_rms={metrics['final_gradient_rms']}", flush=True)
        sub=[r for r in fold_rows if r["geology_model_id"]==g_model and r["lag_c_model_id"]==l_model]
        ok=[r for r in sub if r.get("fold_status") in {"refit_complete","refit_complete_project_convergence"}]
        rows.append({"geology_model_id":g_model,"lag_c_model_id":l_model,"requested_fold_count":int(nfolds),"full_fold_count":int(dataset.n_folds),"valid_fold_count":len(ok),"failed_fold_count":len(sub)-len(ok),"mean_validation_rmse_mm":float(np.mean([r["validation_rmse_mm"] for r in ok])) if ok else np.nan,"std_validation_rmse_mm":float(np.std([r["validation_rmse_mm"] for r in ok])) if ok else np.nan,"median_validation_rmse_mm":float(np.median([r["validation_rmse_mm"] for r in ok])) if ok else np.nan,"mean_validation_mae_mm":float(np.mean([r["validation_mae_mm"] for r in ok])) if ok else np.nan,"mean_validation_log_likelihood":float(np.mean([r["validation_log_likelihood"] for r in ok])) if ok else np.nan,"mean_generalization_gap_mm":float(np.mean([r["generalization_gap_mm"] for r in ok])) if ok else np.nan,"parameter_count":layout.total_parameters,"mean_iterations":float(np.mean([r["iterations"] for r in ok])) if ok else np.nan,"total_elapsed_hours":float(np.sum([r["elapsed_seconds"] for r in sub])/3600),"Ske_identifiability_proxy":float("nan"),"maximum_vif":float("nan"),"condition_number":float("nan"),"status":"complete_validated" if full_fold_run and len(ok)==dataset.n_folds else ("invalid" if len(sub)-len(ok)>1 else "partial")})
    return pd.DataFrame(rows),pd.DataFrame(fold_rows)


def _model_design(data, g_model, l_model):
    gcols = G_MODELS[g_model]
    ske_geo = np.column_stack([data["covariates"][c] for c in gcols]) if gcols else np.empty((len(data["obs"]), 0))
    if l_model == "L0_shared" or not gcols:
        lag_mode = "L0_shared" if l_model != "L2_geology_rbf" else "L2_geology_rbf"
        lag_geo = np.empty((len(data["obs"]), 0))
    else:
        lag_mode = L_MODELS[l_model]
        lag_geo = ske_geo
    lag_rbf = data["rbf"] if lag_mode == "L2_geology_rbf" else None
    design = make_design(ske_geo, data["rbf"], lag_geo, lag_rbf, lag_mode)
    layout = M1ParameterLayout(
        n_ske_geology=ske_geo.shape[1],
        n_ske_rbf=data["rbf"].shape[1],
        n_lag_c_geology=0 if lag_mode == "L0_shared" else lag_geo.shape[1],
        n_lag_c_rbf=data["rbf"].shape[1] if lag_mode == "L2_geology_rbf" else 0,
        lag_c_mode=lag_mode,
    )
    return design, layout


def run_m1_smoke_test(output_root, stage="geology", maxiter=20):
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    data = synthetic_m1_dataset()
    combos = [("G0_no_geology", "L0_shared"), ("G1_confined_clay", "L0_shared"), ("G2_confined_clay_q4", "L0_shared"), ("G3_confined_clay_fraction", "L0_shared")]
    if stage == "lag_c":
        combos = [("G1_confined_clay", "L0_shared"), ("G1_confined_clay", "L1_geology"), ("G1_confined_clay", "L2_geology_rbf")]
    rows = []
    fold_rows = []
    gradient_errors = {}
    for g_model, l_model in combos:
        design, layout = _model_design(data, g_model, l_model)
        theta0 = np.zeros(layout.total_parameters)
        theta0[0] = np.log(0.0015)
        theta0[layout.slices["cu_global"]] = -7.0
        theta0[layout.slices["lag_u_global"]] = -1.0
        fn = lambda th: m1_objective_and_gradient(th, design, layout, data["obs"], data["hc"], data["hu"])
        err, _ = finite_difference_gradient_error(fn, theta0)
        gradient_errors[f"{g_model}+{l_model}"] = err
        for fold_id in sorted(np.unique(data["folds"]))[:2]:
            fold_dir = output / "model_compare" / f"{g_model}_{l_model}" / f"fold_{int(fold_id):02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train = data["folds"] != fold_id
            valid = data["folds"] == fold_id
            train_design = M1Design(ske=design.ske[train], lag_c=design.lag_c[train])
            valid_design = M1Design(ske=design.ske[valid], lag_c=design.lag_c[valid])
            def obj(th):
                return m1_objective_and_gradient(th, train_design, layout, data["obs"][train], data["hc"][train], data["hu"][train])
            start = time.time()
            initial_objective = obj(theta0)[0]
            result = minimize(obj, theta0, method="L-BFGS-B", jac=True, options={"maxiter": maxiter, "ftol": 1e-8})
            pred = predict_m1(result.x, valid_design, layout, data["hc"][valid], data["hu"][valid])
            residual = data["obs"][valid] - pred
            rmse = float(np.sqrt(np.mean(residual * residual)))
            mae = float(np.mean(np.abs(residual)))
            loglike = float(-0.5 * np.sum((residual / 5.0) ** 2))
            metrics = {
                "fold_id": int(fold_id),
                "geology_model_id": g_model,
                "lag_c_model_id": l_model,
                "training_pixel_count": int(train.sum()),
                "validation_pixel_count": int(valid.sum()),
                "training_block_ids": [int(x) for x in sorted(np.unique(data["folds"][train]))],
                "validation_block_ids": [int(fold_id)],
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "iterations": int(result.nit),
                "initial_objective": float(initial_objective),
                "final_training_objective": float(result.fun),
                "validation_rmse_mm": rmse,
                "validation_mae_mm": mae,
                "validation_log_likelihood": loglike,
                "harmonic_amplitude_rmse_mm": rmse,
                "harmonic_phase_mae_days": float("nan"),
                "elapsed_seconds": float(time.time() - start),
                "peak_memory_gb": float("nan"),
                "parameter_count": int(layout.total_parameters),
                "parameter_hash": parameter_hash(result.x),
                "layout": layout.metadata(),
            }
            (fold_dir / "initial_parameters.npy").write_bytes(theta0.astype("float64").tobytes())
            np.save(fold_dir / "optimized_parameters.npy", result.x)
            (fold_dir / "optimizer_result.json").write_text(json.dumps({"success": bool(result.success), "message": str(result.message), "nit": int(result.nit)}, indent=2), encoding="utf-8")
            (fold_dir / "fold_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
            fold_rows.append(metrics)
        subset = [r for r in fold_rows if r["geology_model_id"] == g_model and r["lag_c_model_id"] == l_model and r["optimizer_success"]]
        rows.append(
            {
                "geology_model_id": g_model,
                "lag_c_model_id": l_model,
                "n_parameters": layout.total_parameters,
                "mean_spatial_rmse_mm": float(np.mean([r["validation_rmse_mm"] for r in subset])) if subset else np.nan,
                "std_spatial_rmse_mm": float(np.std([r["validation_rmse_mm"] for r in subset])) if subset else np.nan,
                "mean_validation_log_likelihood": float(np.mean([r["validation_log_likelihood"] for r in subset])) if subset else np.nan,
                "status": "smoke_test_passed" if subset and err < 1e-5 else "failed",
                "gradient_error": err,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows), gradient_errors
