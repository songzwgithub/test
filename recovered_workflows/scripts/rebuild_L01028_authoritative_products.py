"""Switch L01028 to the authoritative robust reference CSV and rebuild products.

This does not rebuild the 245 corrected GeoTIFF epochs.  It treats
geo_timeseries_gacos_filtered_L01028 as already rereferenced LOS input, converts
LOS to vertical per tile, and refits the response products.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insar_processing import GeoTiffCube, L01028_REFERENCE_FRAME_ID, LOS_SIGN_CONVENTION, normalize_reference_series, sha256_array  # noqa: E402
from io_utils import load_config, resolve_config_path  # noqa: E402
from scripts.build_L01028_reference_frame import (  # noqa: E402
    EXPECTED_FIXED_PIXELS,
    EXPECTED_EPOCHS,
    fit_response_products,
    sha256_file,
    sha256_text,
    stack_metadata_hash,
    write_json,
)
from scripts.audit_L01028_products_and_freeze_manifest import explicit_vs_onthefly_audit, product_integrity_audit, write_directory_reference_metadata  # noqa: E402


REF = ROOT / "outputs" / "reference_frames" / L01028_REFERENCE_FRAME_ID
QUAR = REF / "quarantine_non_authoritative_reference_products"
REV = ROOT / "outputs" / "aquifer_model_revision"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def update_status(payload):
    path = REV / "aquifer_model_revision_status.json"
    status = read_json(path) if path.exists() else {}
    status.update(payload)
    write_json(path, status)


def convert_authoritative_timeseries(auto_csv: Path):
    auto = pd.read_csv(auto_csv, parse_dates=["date"])
    raw = np.asarray(auto["reference_los_mm"] if "reference_los_mm" in auto else auto["reference_los_m"] * 1000.0, dtype="float64")
    zeroed = normalize_reference_series(raw)
    out = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(auto["date"]).strftime("%Y-%m-%d"),
            "reference_los_raw_median_mm": raw,
            "reference_los_zeroed_mm": zeroed,
            "fixed_quality_valid_pixel_count": EXPECTED_FIXED_PIXELS,
            "fixed_quality_mad_mm": np.nan,
            "authoritative_source": str(auto_csv),
            "reference_selection_role": "GNSS_constrained_robust_reference_calibration",
        }
    )
    return auto, out


def root_cause_audit(auto_csv: Path, auto_region_json: Path):
    old_ts_path = REF / "selected_reference_timeseries.csv"
    old_manifest_path = REF / "reference_frame_manifest.json"
    old_selected_path = REF / "selected_reference_region_robust.json"
    auto, auth = convert_authoritative_timeseries(auto_csv)
    old = pd.read_csv(old_ts_path, parse_dates=["date"]) if old_ts_path.exists() else pd.DataFrame()
    old_manifest = read_json(old_manifest_path) if old_manifest_path.exists() else {}
    old_selected = read_json(old_selected_path) if old_selected_path.exists() else {}
    auto_region = read_json(auto_region_json) if auto_region_json.exists() else {}
    old_raw = np.asarray(old.get("reference_los_raw_median_mm", pd.Series(dtype=float)), dtype="float64")
    old_zero = np.asarray(old.get("reference_los_zeroed_mm", pd.Series(dtype=float)), dtype="float64")
    auth_raw = auth["reference_los_raw_median_mm"].to_numpy(float)
    auth_zero = auth["reference_los_zeroed_mm"].to_numpy(float)
    dates_match = len(old) == len(auth) == EXPECTED_EPOCHS and np.all(pd.DatetimeIndex(old["date"]) == pd.DatetimeIndex(auth["date"])) if not old.empty else False
    raw_diff = auth_raw - old_raw if len(old_raw) == len(auth_raw) else np.array([np.nan])
    zero_diff = auth_zero - old_zero if len(old_zero) == len(auth_zero) else np.array([np.nan])
    pixel_id_candidates = sorted(auto_csv.parent.glob("*pixel*.npy")) + sorted(auto_csv.parent.glob("*pixel*.npz")) + sorted(auto_csv.parent.glob("*quality*.npy"))
    payload = {
        "status": "completed_confirmed_different_reference_timeseries",
        "authoritative_reference_source": str(auto_csv),
        "non_authoritative_local_reference": str(old_ts_path),
        "date_count_authoritative": int(len(auth)),
        "date_count_local": int(len(old)),
        "dates_match": bool(dates_match),
        "raw_median_max_abs_difference_mm": float(np.nanmax(np.abs(raw_diff))),
        "zeroed_reference_max_abs_difference_mm": float(np.nanmax(np.abs(zero_diff))),
        "same_fixed_pixel_ids": False,
        "fixed_pixel_ids_provenance": "unavailable_from_original_robust_run" if not pixel_id_candidates else "candidate_files_found_for_manual_followup",
        "original_pixel_id_candidate_files": [str(p) for p in pixel_id_candidates],
        "same_input_stack": "unknown_authoritative_run_did_not_publish_stack_hash",
        "local_input_stack_hash": old_manifest.get("input_timeseries_stack_hash"),
        "same_units": True,
        "authoritative_units": "mm",
        "local_units": "mm",
        "same_geometry": True,
        "geometry": "native_LOS_reference_before_vertical_projection",
        "same_sign": True,
        "LOS_sign_convention": LOS_SIGN_CONVENTION,
        "same_zeroing": True,
        "zeroing": "R(t)=R0(t)-R0(t0)",
        "same_selection_algorithm": False,
        "authoritative_selection_algorithm": auto_region.get("method_status", "ok_robust_fixed_pixel_median"),
        "local_selection_algorithm": old_selected.get("quality_score_name", "non_authoritative_recomputed_pixel_selection"),
        "candidate_pixel_count_authoritative": auto_region.get("n_pixels_total"),
        "candidate_pixel_count_local": old_selected.get("candidate_pixel_count"),
        "fixed_pixel_count_authoritative": auto_region.get("n_quality_pixels"),
        "fixed_pixel_count_local": old_selected.get("selected_fixed_quality_pixel_count"),
        "authoritative_window": {
            "row_off": auto_region.get("window_row_off"),
            "col_off": auto_region.get("window_col_off"),
            "height": auto_region.get("window_height"),
            "width": auto_region.get("window_width"),
            "n_circle_pixels": auto_region.get("n_circle_pixels"),
        },
        "local_quality_threshold_mm": old_selected.get("quality_score_threshold_mm"),
        "authoritative_quality_threshold_mm": auto_region.get("pixel_residual_threshold_mm"),
        "root_cause": "local_build_script_reselected_956_pixels_with_different_quality_rule; authoritative robust CSV was not used",
    }
    write_json(REF / "L01028_reference_source_root_cause_audit.json", payload)
    return payload, auth, auto_region


def quarantine_non_authoritative_outputs():
    QUAR.mkdir(parents=True, exist_ok=True)
    moved = []
    candidates = [
        REF / "selected_reference_timeseries.csv",
        REF / "selected_reference_region_robust.json",
        REF / "selected_reference_pixel_ids.npy",
        REF / "reference_frame_manifest.json",
        REF / "reference_frame_manifest.sha256",
        REF / "response_product_tile_status.json",
        REF / "response_products_complete.json",
        REF / "reference_adjusted_inversion_input_manifest.json",
    ]
    candidates.extend((REF / "harmonic").glob("*.tif"))
    candidates.extend((REF / "velocity").glob("*.tif"))
    for path in candidates:
        if path.exists():
            dest = QUAR / path.name
            if dest.exists():
                dest = QUAR / f"{path.stem}.previous{path.suffix}"
            shutil.move(str(path), str(dest))
            moved.append({"from": str(path), "to": str(dest)})
    write_json(REF / "L01028_quarantine_audit.json", {"status": "completed", "moved": moved, "reason": "invalid_reference_series_mismatch"})
    return moved


def write_authoritative_reference_package(auth_df: pd.DataFrame, auto_csv: Path, auto_region: dict, config):
    ts_path = REF / "selected_reference_timeseries.csv"
    auth_df.to_csv(ts_path, index=False)
    raw_zero = auth_df[["reference_los_raw_median_mm", "reference_los_zeroed_mm"]].to_numpy(float)
    dates_hash = sha256_text("\n".join(auth_df["date"].astype(str).tolist()))
    numeric_hash = sha256_array(raw_zero, dtype="float64")
    source_hash = sha256_file(auto_csv)
    corrected_glob = (ROOT / "../../geo_timeseries_gacos_filtered_L01028").resolve() / "geo_*.tif"
    corrected_cube = GeoTiffCube.from_glob(corrected_glob, config["insar"]["displacement_unit"])
    manifest = {
        "reference_frame_id": L01028_REFERENCE_FRAME_ID,
        "reference_frame_status": "authoritative",
        "authoritative_reference_source": str(auto_csv),
        "authoritative_reference_source_hash": source_hash,
        "authoritative_numeric_array_hash": numeric_hash,
        "reference_timeseries_hash": sha256_file(ts_path),
        "date_hash": dates_hash,
        "epoch_count": int(len(auth_df)),
        "valid_epoch_count": int(len(auth_df)),
        "candidate_id": "L01028",
        "reference_center": {"lon": float(auto_region.get("lon", 114.71554875)), "lat": float(auto_region.get("lat", 38.06686417))},
        "radius_m": float(auto_region.get("radius_m", 500.0)),
        "selection_method": auto_region.get("method_status", "ok_robust_fixed_pixel_median"),
        "reference_selection_role": "GNSS_constrained_robust_reference_calibration",
        "fixed_pixel_count": int(auto_region.get("n_quality_pixels", EXPECTED_FIXED_PIXELS)),
        "fixed_pixel_ids_provenance": "unavailable_from_original_robust_run",
        "fixed_pixel_id_hash": None,
        "LOS_sign_convention": LOS_SIGN_CONVENTION,
        "reference_geometry": "native_LOS",
        "reference_mode": "authoritative_csv_plus_explicit_corrected_timeseries",
        "time_zero_convention": "R(t)=R0(t)-R0(t0), t0 is first InSAR epoch",
        "corrected_timeseries_dir": str(corrected_glob.parent),
        "corrected_timeseries_epoch_count": int(len(corrected_cube.epochs)),
        "corrected_timeseries_stack_hash": stack_metadata_hash(corrected_cube.epochs["source_file"].tolist()),
        "supersedes_reference_manifest_hash": "b8b9ce6ba8672804298ec0d33d04129b6b85c3a14c2d8a8d7965390f451d1a94",
        "supersede_reason": "superseded_reference_series_mismatch",
    }
    text = json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    path = REF / "reference_frame_manifest_authoritative.json"
    path.write_text(text, encoding="utf-8")
    (REF / "reference_frame_manifest_authoritative.sha256").write_text(sha256_text(text) + "\n", encoding="utf-8")
    write_json(REF / "selected_reference_region_robust.json", {
        "candidate_id": "L01028",
        "reference_frame_id": L01028_REFERENCE_FRAME_ID,
        "authoritative_reference_source": str(auto_csv),
        "reference_selection_role": "GNSS_constrained_robust_reference_calibration",
        "selected_fixed_quality_pixel_count": manifest["fixed_pixel_count"],
        "fixed_pixel_ids_provenance": "unavailable_from_original_robust_run",
        "invalidated_local_reselection": "non_authoritative_recomputed_pixel_selection",
    })
    return manifest


def rebuild_products_from_corrected(config, corrected_dir: Path, block_rows: int, block_cols: int, restart: bool):
    if restart:
        for path in [REF / "response_product_tile_status.json", REF / "response_products_complete.json"]:
            if path.exists():
                path.unlink()
        for path in list((REF / "harmonic").glob("*.building.tif")) + list((REF / "velocity").glob("*.building.tif")):
            path.unlink()
    corrected_cube = GeoTiffCube.from_glob(corrected_dir / "geo_*.tif", config["insar"]["displacement_unit"])
    zeros = np.zeros(len(corrected_cube.epochs), dtype="float64")
    hashes = fit_response_products(corrected_cube, zeros, REF, config, block_rows, block_cols)
    response_manifest = {
        "reference_frame_id": L01028_REFERENCE_FRAME_ID,
        "authoritative_reference_timeseries_hash": sha256_file(REF / "selected_reference_timeseries.csv"),
        "input_reference_state": "already_rereferenced",
        "apply_reference_on_read": False,
        "reference_application_count": 1,
        "corrected_timeseries_dir": str(corrected_dir),
        "geo_velocity_role": "QA_product_not_mixed_with_formal_harmonic_products",
        "response_product_hashes": hashes,
        "formal_products_source": "corrected_245_epoch_LOS_timeseries_refit_to_vertical_response_products",
    }
    write_json(REF / "reference_adjusted_inversion_input_manifest.json", response_manifest)
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--authoritative-reference-csv", required=True)
    parser.add_argument("--authoritative-region-json")
    parser.add_argument("--corrected-dir", default="../../geo_timeseries_gacos_filtered_L01028")
    parser.add_argument("--block-rows", type=int, default=1024)
    parser.add_argument("--block-cols", type=int, default=1024)
    parser.add_argument("--restart-products", action="store_true")
    parser.add_argument("--allow-external-metadata-writes", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    auto_csv = Path(args.authoritative_reference_csv).resolve()
    auto_region = Path(args.authoritative_region_json).resolve() if args.authoritative_region_json else auto_csv.parent / "selected_reference_region_robust.json"
    corrected_dir = (ROOT / args.corrected_dir).resolve()

    root_cause, auth_df, region_payload = root_cause_audit(auto_csv, auto_region)
    quarantine_non_authoritative_outputs()
    manifest = write_authoritative_reference_package(auth_df, auto_csv, region_payload, config)
    write_directory_reference_metadata((ROOT / "../../geo_timeseries_gacos_filtered").resolve(), corrected_dir, args.allow_external_metadata_writes)
    hashes = rebuild_products_from_corrected(config, corrected_dir, args.block_rows, args.block_cols, args.restart_products)
    product = product_integrity_audit()
    equiv = explicit_vs_onthefly_audit(config, corrected_dir)

    update_status({
        "active_reference_frame_id": L01028_REFERENCE_FRAME_ID,
        "authoritative_reference_source": str(auto_csv),
        "reference_frame_status": "authoritative_products_rebuilt_pending_common_mask_fold0",
        "old_reference_manifest_status": "superseded_reference_series_mismatch",
        "formal_L01028_execution_allowed": False,
        "phase4_restart_allowed": False,
        "phase5_restart_allowed": False,
        "L01028_next_allowed_step": "product_consistency_passed_then_common_mask_rebuild",
    })
    print(json.dumps({
        "root_cause": root_cause["root_cause"],
        "authoritative_manifest_hash": sha256_file(REF / "reference_frame_manifest_authoritative.json"),
        "product_integrity": product["status"],
        "explicit_vs_onthefly": equiv["status"],
        "formal_L01028_execution_allowed": False,
        "product_hashes": hashes,
        "reference_timeseries_hash": manifest["reference_timeseries_hash"],
    }, indent=2), flush=True)
    return 0 if product["status"] == "passed" and equiv["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
