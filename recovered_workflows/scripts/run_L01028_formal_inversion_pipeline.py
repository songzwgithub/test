#!/usr/bin/env python3
"""Resumable L01028 formal inversion pipeline entry point."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ID = "L01028_500m_fixed_quality_median_v1"
DEFAULT_REF = ROOT / "outputs" / "reference_frames" / REFERENCE_ID
DEFAULT_OUTPUT = DEFAULT_REF / "formal_inversion"
STAGES = ["finalize_fold0", "freeze_manifest", "formal_cv", "aggregate_cv", "final_full_data_refit", "final_acceptance"]
STATUS_2 = ROOT / "outputs" / "aquifer_model_revision" / "aquifer_model_revision_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_status(reference_dir: Path, payload: dict[str, Any]) -> None:
    for path in [reference_dir / "L01028_reference_frame_status.json", STATUS_2]:
        data = read_json(path) if path.exists() else {}
        data.update(payload)
        atomic_json(path, data)


def stage_status_path(output: Path) -> Path:
    return output / "stage_status.json"


def write_stage(output: Path, stage: str, status: str, input_hash: str, output_hash: str | None = None, failure_reason: str | None = None) -> None:
    payload = read_json(stage_status_path(output)) if stage_status_path(output).exists() else {}
    payload[stage] = {
        "stage": stage,
        "status": status,
        "started_at_utc": payload.get(stage, {}).get("started_at_utc") or utc_now(),
        "completed_at_utc": utc_now() if status == "completed" else None,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "failure_reason": failure_reason,
    }
    atomic_json(stage_status_path(output), payload)


def file_count(path: Path) -> int:
    return sum(1 for _ in path.iterdir()) if path.exists() else 0


def check_only(cache: Path, expected: str, reference_dir: Path, output: Path) -> dict[str, Any]:
    acceptance = read_json(reference_dir / "L01028_final_harmonic_cache_acceptance.json")
    fold0_dir = reference_dir / "fold0_confirmation"
    rbf = ROOT / "outputs" / "aquifer_model_revision" / "selected_rbf_design.json"
    if not rbf.exists():
        rbf = ROOT / "outputs" / "aquifer_model_revision" / "rbf_global_basis_selection.json"
    lock_path = output / "L01028_formal_inversion.lock"
    checks = {
        "cache_acceptance_passed": acceptance.get("audit_status") == "passed" and acceptance.get("all_acceptance_checks_passed") is True,
        "cache_sha256_matches": cache.exists() and sha256_file(cache) == expected,
        "building_cache_absent": not (cache.parent / "phase4_harmonic_blocks_L01028_authoritative.building.h5").exists(),
        "fold0_memmap_manifest_exists": (fold0_dir / "memmap_manifest.json").exists(),
        "fold0_checkpoint_file_count": file_count(fold0_dir / "checkpoints"),
        "common_mask_exists": (ROOT / "outputs" / "aquifer_model_revision" / "comparison_common_mask.tif").exists(),
        "fold_map_exists": (ROOT / "outputs" / "aquifer_model_revision" / "spatial_validation_blocks.tif").exists(),
        "rbf_selection_exists": rbf.exists(),
        "freeze_manifest_script_exists": (ROOT / "scripts" / "audit_L01028_products_and_freeze_manifest.py").exists(),
        "formal_pipeline_output_writable": output.exists() or os.access(output.parent, os.W_OK),
        "formal_lock_path": str(lock_path),
    }
    ok = (
        checks["cache_acceptance_passed"]
        and checks["cache_sha256_matches"]
        and checks["building_cache_absent"]
        and checks["common_mask_exists"]
        and checks["fold_map_exists"]
        and checks["rbf_selection_exists"]
        and checks["freeze_manifest_script_exists"]
        and checks["formal_pipeline_output_writable"]
    )
    return {"check_only_status": "passed" if ok else "failed", "checks": checks}


def count_common_mask(path: Path) -> int:
    total = 0
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            total += int((src.read(1, window=window) == 1).sum())
    return total


def write_frozen_manifest(reference_dir: Path, cache: Path, expected: str, output: Path) -> tuple[Path, str]:
    fold0_summary = reference_dir / "fold0_confirmation" / "fold0_confirmation_summary.json"
    summary = read_json(fold0_summary)
    if summary.get("fold0_confirmation_status") != "passed" or summary.get("no_formal_fold_accessed") is not True:
        raise RuntimeError("fold0 summary is not passed")
    products = {
        "velocity": reference_dir / "velocity" / "insar_vertical_velocity_mm_yr.tif",
        "annual_real": reference_dir / "harmonic" / "annual_vertical_real_sin_mm.tif",
        "annual_imag": reference_dir / "harmonic" / "annual_vertical_imag_cos_mm.tif",
        "annual_amplitude": reference_dir / "harmonic" / "annual_vertical_amplitude_mm.tif",
        "annual_phase": reference_dir / "harmonic" / "annual_vertical_phase_rad.tif",
        "n_observations": reference_dir / "harmonic" / "n_observations.tif",
    }
    common = ROOT / "outputs" / "aquifer_model_revision" / "comparison_common_mask.tif"
    fold_map = ROOT / "outputs" / "aquifer_model_revision" / "spatial_validation_blocks.tif"
    rbf = ROOT / "outputs" / "aquifer_model_revision" / "selected_rbf_design.json"
    if not rbf.exists():
        rbf = ROOT / "outputs" / "aquifer_model_revision" / "rbf_global_basis_selection.json"
    rbf_payload = read_json(rbf)
    acceptance = reference_dir / "L01028_final_harmonic_cache_acceptance.json"
    ref_manifest = reference_dir / "reference_frame_manifest.json"
    ref_csv = reference_dir / "selected_reference_timeseries.csv"
    payload = {
        "formal_protocol_version": "v2_L01028_formal_pipeline",
        "freeze_timestamp_utc": utc_now(),
        "reference_frame_id": REFERENCE_ID,
        "reference_frame_manifest_path": str(ref_manifest),
        "reference_frame_manifest_hash": sha256_file(ref_manifest),
        "authoritative_reference_csv_path": str(ref_csv),
        "authoritative_reference_csv_hash": sha256_file(ref_csv),
        "date_count": 245,
        "date_hash": read_json(ref_manifest).get("date_hash") or read_json(ref_manifest).get("reference_dates_hash"),
        "response_product_hashes": {k: sha256_file(v) for k, v in products.items()},
        "authoritative_harmonic_cache_path": str(cache),
        "authoritative_harmonic_cache_sha256": expected,
        "cache_key": read_json(acceptance).get("cache_key"),
        "cache_acceptance_path": str(acceptance),
        "cache_acceptance_sha256": sha256_file(acceptance),
        "common_mask_path": str(common),
        "common_mask_hash": sha256_file(common),
        "common_mask_pixel_count": count_common_mask(common),
        "fold_map_path": str(fold_map),
        "fold_map_hash": sha256_file(fold_map),
        "RBF_selection_path": str(rbf),
        "RBF_selection_file_hash": sha256_file(rbf),
        "RBF_selection_internal_hash": rbf_payload.get("selection_mask_hash") or rbf_payload.get("basis_design_hash"),
        "geology_raster_hashes": {},
        "config_yaml_hash": sha256_file(ROOT / "config.yaml"),
        "formal_inversion_source_hashes": {
            "run_L01028_fold0_confirmation": sha256_file(ROOT / "scripts" / "run_L01028_fold0_confirmation.py"),
            "run_L01028_formal_inversion_pipeline": sha256_file(Path(__file__)),
        },
        "fold0_summary_hash": sha256_file(fold0_summary),
        "selected_lag_u_days": summary["selected_lag_u_days"],
        "selected_lambda": summary["selected_lambda"],
        "selected_stage_c_budget": summary["selected_stage_c_budget"],
        "no_formal_fold_accessed_during_confirmation": True,
        "old_reference_formal_results": "historical_only",
    }
    manifest = reference_dir / "formal_protocol_v2_L01028_frozen_manifest.json"
    atomic_json(manifest, payload)
    digest = sha256_file(manifest)
    sidecar = reference_dir / "formal_protocol_v2_L01028_frozen_manifest.sha256"
    tmp = sidecar.with_suffix(sidecar.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(digest + "\n", encoding="utf-8")
    os.replace(tmp, sidecar)
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(manifest):
        raise RuntimeError("manifest SHA256 sidecar mismatch")
    update_status(
        reference_dir,
        {
            "formal_L01028_execution_allowed": True,
            "phase4_restart_allowed": False,
            "phase5_restart_allowed": False,
            "final_full_data_refit_allowed": False,
            "frozen_manifest_path": str(manifest),
            "frozen_manifest_hash": digest,
        },
    )
    write_stage(output, "freeze_manifest", "completed", sha256_file(fold0_summary), digest)
    return manifest, digest


def run_finalize_fold0(cache: Path, reference_dir: Path, output: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_L01028_fold0_confirmation.py"),
        "--cache",
        str(cache),
        "--fold-id",
        "0",
        "--output-dir",
        str(reference_dir / "fold0_confirmation"),
        "--finalize-only",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    write_stage(output, "finalize_fold0", "completed", sha256_file(cache), sha256_file(reference_dir / "fold0_confirmation" / "fold0_confirmation_summary.json"))


def selected_stages(start: str, stop: str) -> list[str]:
    i = STAGES.index(start)
    j = STAGES.index(stop)
    if j < i:
        raise ValueError("stop-after-stage cannot precede start-stage")
    return STAGES[i : j + 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--reference-dir", default=str(DEFAULT_REF))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--start-stage", choices=STAGES, default="finalize_fold0")
    parser.add_argument("--stop-after-stage", choices=STAGES, default="final_acceptance")
    args = parser.parse_args()
    cache = Path(args.cache).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        print(json.dumps(check_only(cache, args.expected_cache_sha256, reference_dir, output), indent=2, sort_keys=True))
        return 0
    lock_path = output / "L01028_formal_inversion.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another L01028 formal pipeline is already running")
        if sha256_file(cache) != args.expected_cache_sha256:
            raise RuntimeError("cache SHA256 mismatch")
        for stage in selected_stages(args.start_stage, args.stop_after_stage):
            if stage == "finalize_fold0":
                run_finalize_fold0(cache, reference_dir, output)
            elif stage == "freeze_manifest":
                write_frozen_manifest(reference_dir, cache, args.expected_cache_sha256, output)
            else:
                write_stage(output, stage, "blocked_not_run_by_this_invocation", sha256_file(cache), failure_reason="long stage requires explicit later invocation")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
