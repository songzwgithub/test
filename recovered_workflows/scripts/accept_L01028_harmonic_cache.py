#!/usr/bin/env python3
"""Accept the authoritative L01028 Phase-4 harmonic cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
from rasterio.windows import Window


OLD_CACHE = Path("outputs/cache/phase4_harmonic_blocks_d0283cfacbadc767.h5")
NEW_CACHE = Path("outputs/cache/phase4_harmonic_blocks_L01028_authoritative.h5")
BUILDING_CACHE = Path("outputs/cache/phase4_harmonic_blocks_L01028_authoritative.building.h5")
REBUILD_AUDIT = Path("outputs/cache/phase4_harmonic_blocks_L01028_authoritative_audit.json")
REFERENCE_DIR = Path("outputs/reference_frames/L01028_500m_fixed_quality_median_v1")
REFERENCE_MANIFEST = REFERENCE_DIR / "reference_frame_manifest.json"
ANNUAL_REAL = REFERENCE_DIR / "harmonic" / "annual_vertical_real_sin_mm.tif"
ANNUAL_IMAG = REFERENCE_DIR / "harmonic" / "annual_vertical_imag_cos_mm.tif"
COMMON_MASK = Path("outputs/aquifer_model_revision/comparison_common_mask.tif")
PRE_FOLD0_GATE = REFERENCE_DIR / "L01028_pre_fold0_audit_gate.json"
CACHE_ACCEPTANCE = REFERENCE_DIR / "L01028_final_harmonic_cache_acceptance.json"
REFERENCE_FRAME_ID = "L01028_500m_fixed_quality_median_v1"
STATIC_DATASETS = [
    "hc",
    "hu",
    "z",
    "flat_index",
    "block_start",
    "block_count",
    "block_row",
    "block_col",
    "block_height",
    "block_width",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_sha256(ds: h5py.Dataset, chunk_rows: int = 500_000) -> str:
    h = hashlib.sha256()
    if ds.shape == ():
        h.update(np.asarray(ds[()]).tobytes())
        return h.hexdigest()
    n = int(ds.shape[0])
    for start in range(0, n, chunk_rows):
        arr = np.asarray(ds[start : min(start + chunk_rows, n)])
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def dataset_storage(ds: h5py.Dataset) -> dict[str, Any]:
    return {
        "shape": list(ds.shape),
        "dtype": str(ds.dtype),
        "chunks": list(ds.chunks) if ds.chunks else None,
        "compression": ds.compression,
        "compression_opts": ds.compression_opts,
        "shuffle": bool(ds.shuffle),
        "fletcher32": bool(ds.fletcher32),
    }


def common_mask_count(path: Path) -> int:
    with rasterio.open(path) as src:
        count = 0
        for _, window in src.block_windows(1):
            count += int(np.count_nonzero(src.read(1, window=window)))
    return count


def static_dataset_audit(old: h5py.File, new: h5py.File) -> tuple[dict[str, Any], str, bool]:
    results: dict[str, Any] = {}
    combined = []
    all_ok = True
    for name in STATIC_DATASETS:
        entry: dict[str, Any] = {"exists_old": name in old, "exists_new": name in new}
        if name not in old or name not in new:
            entry["identical"] = False
            results[name] = entry
            all_ok = False
            continue
        old_store = dataset_storage(old[name])
        new_store = dataset_storage(new[name])
        old_hash = dataset_sha256(old[name])
        new_hash = dataset_sha256(new[name])
        identical = old_store == new_store and old_hash == new_hash
        entry.update(old_store)
        entry.update(
            {
                "old_sha256": old_hash,
                "new_sha256": new_hash,
                "identical": bool(identical),
            }
        )
        results[name] = entry
        combined.append({"name": name, "shape": old_store["shape"], "dtype": old_store["dtype"], "sha256": new_hash})
        all_ok = all_ok and identical
    combined_json = json.dumps(sorted(combined, key=lambda x: x["name"]), sort_keys=True, separators=(",", ":"))
    return results, hashlib.sha256(combined_json.encode("utf-8")).hexdigest(), bool(all_ok)


def spatial_index_audit(h5: h5py.File, n_pixels: int) -> tuple[bool, list[str]]:
    failures: list[str] = []
    starts = h5["block_start"]
    counts = h5["block_count"]
    flat = h5["flat_index"]
    expected = 0
    for i in range(len(starts)):
        start = int(starts[i])
        count = int(counts[i])
        if start != expected:
            failures.append(f"block_{i}_start_not_contiguous")
            break
        expected = start + count
        width = int(h5["block_width"][i])
        height = int(h5["block_height"][i])
        for j in range(start, start + count, 500_000):
            vals = flat[j : min(j + 500_000, start + count)]
            if vals.size and (int(np.min(vals)) < 0 or int(np.max(vals)) >= width * height):
                failures.append(f"block_{i}_flat_index_out_of_bounds")
                break
    if expected != n_pixels:
        failures.append("last_block_end_mismatch")
    return not failures, failures


def obs_sample_audit(h5: h5py.File) -> dict[str, Any]:
    block_count = int(len(h5["block_start"]))
    block_ids = np.unique(np.linspace(0, block_count - 1, 40, dtype=int))
    sse = 0.0
    n_values = 0
    checked_pixels = 0
    max_abs = 0.0
    with rasterio.open(ANNUAL_REAL) as real_src, rasterio.open(ANNUAL_IMAG) as imag_src:
        for bid in block_ids:
            start = int(h5["block_start"][bid])
            count = int(h5["block_count"][bid])
            row = int(h5["block_row"][bid])
            col = int(h5["block_col"][bid])
            height = int(h5["block_height"][bid])
            width = int(h5["block_width"][bid])
            flat = h5["flat_index"][start : start + count].astype(np.int64)
            window = Window(col, row, width, height)
            real = real_src.read(1, window=window).reshape(-1)[flat]
            imag = imag_src.read(1, window=window).reshape(-1)[flat]
            obs = h5["obs"][start : start + count]
            ref = np.column_stack([real, imag])
            finite = np.isfinite(obs) & np.isfinite(ref)
            diff = obs[finite] - ref[finite]
            if diff.size:
                max_abs = max(max_abs, float(np.max(np.abs(diff))))
                sse += float(np.sum(diff.astype(np.float64) ** 2))
                n_values += int(diff.size)
            checked_pixels += count
    return {
        "checked_block_count": int(len(block_ids)),
        "checked_pixel_count": int(checked_pixels),
        "checked_value_count": int(n_values),
        "maximum_absolute_difference_mm": float(max_abs),
        "rms_difference_mm": float(math.sqrt(sse / max(n_values, 1))),
    }


def weights_audit(h5: h5py.File, n_pixels: int) -> dict[str, Any]:
    weights = h5["weights"]
    finite = 0
    positive = 0
    total = 0.0
    min_v = math.inf
    max_v = -math.inf
    below = 0
    samples = []
    for start in range(0, n_pixels, 500_000):
        vals = np.asarray(weights[start : min(start + 500_000, n_pixels)], dtype=np.float64)
        finite += int(np.count_nonzero(np.isfinite(vals)))
        positive += int(np.count_nonzero(vals > 0))
        total += float(np.sum(vals))
        min_v = min(min_v, float(np.min(vals)))
        max_v = max(max_v, float(np.max(vals)))
        below += int(np.count_nonzero(vals < 0.04))
        samples.append(vals[:: max(1, vals.size // 5000)])
    sample = np.concatenate(samples) if samples else np.array([], dtype=float)
    attr_total = float(h5.attrs["total_weight"])
    rel = abs(total - attr_total) / max(abs(attr_total), 1.0)
    return {
        "count": int(n_pixels),
        "finite_count": int(finite),
        "positive_count": int(positive),
        "minimum": float(min_v),
        "median_sample": float(np.median(sample)) if sample.size else None,
        "maximum": float(max_v),
        "count_below_0_04": int(below),
        "total_recomputed": float(total),
        "total_hdf5_attribute": attr_total,
        "total_relative_difference": float(rel),
    }


def file_stat(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"size": int(st.st_size), "mtime": float(st.st_mtime)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance-json", default=str(CACHE_ACCEPTANCE))
    parser.add_argument("--pre-fold0-gate", default=str(PRE_FOLD0_GATE))
    args = parser.parse_args()

    failures: list[str] = []
    if not NEW_CACHE.exists():
        failures.append("new_cache_missing")
    if BUILDING_CACHE.exists():
        failures.append("building_cache_present")
    if not REBUILD_AUDIT.exists():
        failures.append("rebuild_audit_missing")
    rebuild = read_json(REBUILD_AUDIT) if REBUILD_AUDIT.exists() else {}
    common_count = common_mask_count(COMMON_MASK) if COMMON_MASK.exists() else -1
    hashes = {
        "old_cache_sha256": sha256_file(OLD_CACHE) if OLD_CACHE.exists() else None,
        "new_cache_sha256": sha256_file(NEW_CACHE) if NEW_CACHE.exists() else None,
        "rebuild_audit_sha256": sha256_file(REBUILD_AUDIT) if REBUILD_AUDIT.exists() else None,
        "reference_manifest_hash": sha256_file(REFERENCE_MANIFEST) if REFERENCE_MANIFEST.exists() else None,
        "annual_real_hash": sha256_file(ANNUAL_REAL) if ANNUAL_REAL.exists() else None,
        "annual_imag_hash": sha256_file(ANNUAL_IMAG) if ANNUAL_IMAG.exists() else None,
        "common_mask_hash": sha256_file(COMMON_MASK) if COMMON_MASK.exists() else None,
    }
    n_blocks = n_pixels = 0
    obs_check = {}
    weight_check = {}
    static_results = {}
    combined_static_hash = None
    static_ok = False
    cache_key = None
    if NEW_CACHE.exists():
        with h5py.File(NEW_CACHE, "r") as new, h5py.File(OLD_CACHE, "r") as old:
            n_blocks = int(new.attrs.get("n_blocks", len(new["block_start"])))
            n_pixels = int(new.attrs.get("n_pixels", len(new["flat_index"])))
            cache_key = str(new.attrs.get("cache_key", ""))
            attr_expect = {
                "complete": int(new.attrs.get("complete", 0)) == 1,
                "reference_frame_id": str(new.attrs.get("reference_frame_id", "")) == REFERENCE_FRAME_ID,
                "n_blocks": n_blocks == 87,
                "n_pixels": n_pixels == 15_242_540,
                "obs_shape": tuple(new["obs"].shape) == (15_242_540, 2),
                "weights_shape": tuple(new["weights"].shape) == (15_242_540,),
                "reference_manifest_hash": str(new.attrs.get("reference_manifest_hash", "")) == hashes["reference_manifest_hash"],
                "annual_real_hash": str(new.attrs.get("annual_real_hash", "")) == hashes["annual_real_hash"],
                "annual_imag_hash": str(new.attrs.get("annual_imag_hash", "")) == hashes["annual_imag_hash"],
                "common_mask_hash": str(new.attrs.get("common_mask_hash", "")) == hashes["common_mask_hash"],
                "source_cache_hash": str(new.attrs.get("source_cache_hash", "")) == hashes["old_cache_sha256"],
                "observation_sigma_mm": abs(float(new.attrs.get("observation_sigma_mm", np.nan)) - 5.0) < 1e-12,
                "cache_key_nonempty": bool(cache_key),
                "source_timeseries_directory": "geo_timeseries_gacos_filtered_L01028" in str(new.attrs.get("source_timeseries_directory", "")),
            }
            provenance = str(new.attrs.get("cache_provenance", ""))
            attr_expect["reference_not_reapplied"] = (
                "reference_application_count" not in new.attrs
                or str(new.attrs.get("reference_application_count")) in {"1", "1.0"}
                or '"reference_application_count":1' in provenance
            )
            for key, ok in attr_expect.items():
                if not ok:
                    failures.append(f"hdf5_attribute_check_failed:{key}")
            static_results, combined_static_hash, static_ok = static_dataset_audit(old, new)
            if not static_ok:
                failures.append("static_datasets_not_identical")
            index_ok, index_failures = spatial_index_audit(new, n_pixels)
            if not index_ok:
                failures.extend(index_failures)
            obs_check = obs_sample_audit(new)
            if obs_check["maximum_absolute_difference_mm"] > 1e-4 or obs_check["rms_difference_mm"] > 1e-5:
                failures.append("obs_sample_mismatch")
            weight_check = weights_audit(new, n_pixels)
            if weight_check["finite_count"] != n_pixels:
                failures.append("weights_nonfinite")
            if weight_check["positive_count"] != n_pixels:
                failures.append("weights_nonpositive")
            if weight_check["maximum"] > 0.0400001:
                failures.append("weights_above_limit")
            if weight_check["total_relative_difference"] > 1e-6:
                failures.append("weights_total_mismatch")
    rebuild_expect = {
        "audit_status": rebuild.get("audit_status") == "passed",
        "date_count": rebuild.get("date_count") == 245,
        "block_count": rebuild.get("block_count") == 87,
        "record_count": rebuild.get("record_count") == 15_242_540,
        "common_mask_pixel_count": rebuild.get("common_mask_pixel_count") == 15_241_589,
        "extra_pixels_vs_common_mask": rebuild.get("extra_pixels_vs_common_mask") == 951,
        "obs_max_abs_diff_mm": float(rebuild.get("obs_max_abs_diff_mm", math.inf)) <= 1e-4,
        "static_datasets_identical": rebuild.get("static_datasets_identical") is True,
        "cache_matches_L01028_response": rebuild.get("cache_matches_L01028_response") is True,
        "new_cache_hash": rebuild.get("new_cache_hash") == hashes["new_cache_sha256"],
    }
    for key, ok in rebuild_expect.items():
        if not ok:
            failures.append(f"rebuild_audit_check_failed:{key}")
    if common_count != 15_241_589:
        failures.append("common_mask_count_mismatch")
    passed = not failures
    acceptance = {
        "audit_status": "passed" if passed else "failed",
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_frame_id": REFERENCE_FRAME_ID,
        "old_cache_path": str(OLD_CACHE),
        "old_cache_sha256": hashes["old_cache_sha256"],
        "new_cache_path": str(NEW_CACHE),
        "new_cache_file_size_bytes": file_stat(NEW_CACHE)["size"] if NEW_CACHE.exists() else None,
        "new_cache_mtime": file_stat(NEW_CACHE)["mtime"] if NEW_CACHE.exists() else None,
        "new_cache_sha256": hashes["new_cache_sha256"],
        "rebuild_audit_path": str(REBUILD_AUDIT),
        "rebuild_audit_sha256": hashes["rebuild_audit_sha256"],
        "rebuild_audit_status": rebuild.get("audit_status"),
        "date_count": rebuild.get("date_count"),
        "block_count": n_blocks,
        "record_count": n_pixels,
        "common_mask_pixel_count": common_count,
        "extra_pixels_vs_common_mask": n_pixels - common_count if n_pixels else None,
        "obs_checked_block_count": obs_check.get("checked_block_count"),
        "obs_checked_pixel_count": obs_check.get("checked_pixel_count"),
        "obs_checked_value_count": obs_check.get("checked_value_count"),
        "obs_max_abs_difference_mm": obs_check.get("maximum_absolute_difference_mm"),
        "obs_rms_difference_mm": obs_check.get("rms_difference_mm"),
        "weights_count": weight_check.get("count"),
        "weights_minimum": weight_check.get("minimum"),
        "weights_median_sample": weight_check.get("median_sample"),
        "weights_maximum": weight_check.get("maximum"),
        "weights_below_0_04_count": weight_check.get("count_below_0_04"),
        "weights_total_recomputed": weight_check.get("total_recomputed"),
        "weights_total_hdf5_attribute": weight_check.get("total_hdf5_attribute"),
        "weights_total_relative_difference": weight_check.get("total_relative_difference"),
        "static_dataset_results": static_results,
        "combined_static_datasets_sha256": combined_static_hash,
        "static_datasets_identical": static_ok,
        "reference_manifest_hash": hashes["reference_manifest_hash"],
        "annual_real_hash": hashes["annual_real_hash"],
        "annual_imag_hash": hashes["annual_imag_hash"],
        "common_mask_hash": hashes["common_mask_hash"],
        "source_cache_hash": hashes["old_cache_sha256"],
        "cache_key": cache_key,
        "cache_matches_L01028_response": obs_check.get("maximum_absolute_difference_mm", math.inf) <= 1e-4,
        "building_cache_absent": not BUILDING_CACHE.exists(),
        "all_acceptance_checks_passed": passed,
        "failure_reasons": failures,
    }
    write_json(Path(args.acceptance_json), acceptance)
    gate = {
        "audit_gate_status": "passed" if passed else "failed",
        "fold0_confirmation_allowed": passed,
        "formal_L01028_execution_allowed": False,
        "formal_execution_block_reason": "fold0_confirmation_and_frozen_manifest_pending" if passed else "harmonic_cache_acceptance_failed",
        "checks": {
            "harmonic_cache_matches_L01028": passed,
            "harmonic_cache_acceptance_passed": passed,
            "static_cache_datasets_identical": static_ok,
            "final_hdf5_sha256_verified": rebuild.get("new_cache_hash") == hashes["new_cache_sha256"],
        },
        "L01028_harmonic_cache_path": str(NEW_CACHE),
        "L01028_harmonic_cache_sha256": hashes["new_cache_sha256"],
        "L01028_harmonic_cache_key": cache_key,
        "L01028_cache_acceptance_path": str(CACHE_ACCEPTANCE),
        "L01028_cache_acceptance_sha256": sha256_file(Path(args.acceptance_json)),
        "combined_static_datasets_sha256": combined_static_hash,
    }
    write_json(Path(args.pre_fold0_gate), gate)
    print(json.dumps({"passed": passed, "failure_reasons": failures, "new_cache_sha256": hashes["new_cache_sha256"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
