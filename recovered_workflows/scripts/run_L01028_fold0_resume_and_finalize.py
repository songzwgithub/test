#!/usr/bin/env python3
"""Run resumable L01028 fold0 confirmation and finalize the frozen manifest."""
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


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ID = "L01028_500m_fixed_quality_median_v1"
DEFAULT_REF = ROOT / "outputs" / "reference_frames" / REFERENCE_ID
STATUS_1 = DEFAULT_REF / "L01028_reference_frame_status.json"
STATUS_2 = ROOT / "outputs" / "aquifer_model_revision" / "aquifer_model_revision_status.json"
PRE_GATE = DEFAULT_REF / "L01028_pre_fold0_audit_gate.json"
ACCEPTANCE = DEFAULT_REF / "L01028_final_harmonic_cache_acceptance.json"
FROZEN = DEFAULT_REF / "formal_protocol_v2_L01028_frozen_manifest.json"
FROZEN_SHA = DEFAULT_REF / "formal_protocol_v2_L01028_frozen_manifest.sha256"
FINAL_ACCEPTANCE = DEFAULT_REF / "L01028_final_protocol_acceptance.json"


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


def fail(reference_dir: Path, reason: str) -> int:
    update_status(
        reference_dir,
        {
            "formal_L01028_execution_allowed": False,
            "final_full_data_refit_allowed": False,
            "phase4_restart_allowed": False,
            "phase5_restart_allowed": False,
            "formal_execution_block_reason": reason,
        },
    )
    atomic_json(reference_dir / "L01028_final_protocol_acceptance.json", {"status": "failed", "failure_reason": reason, "timestamp_utc": utc_now()})
    print(json.dumps({"status": "failed", "failure_reason": reason}, indent=2))
    return 1


def validate_gates(reference_dir: Path, expected_cache_sha: str) -> dict[str, Any]:
    acceptance = read_json(reference_dir / "L01028_final_harmonic_cache_acceptance.json")
    if acceptance.get("audit_status") != "passed" or acceptance.get("all_acceptance_checks_passed") is not True:
        raise RuntimeError("cache acceptance is not passed")
    if acceptance.get("new_cache_sha256") != expected_cache_sha:
        raise RuntimeError("cache acceptance SHA256 mismatch")
    gate = read_json(reference_dir / "L01028_pre_fold0_audit_gate.json")
    if gate.get("fold0_confirmation_allowed") is not True:
        raise RuntimeError("pre-fold0 gate does not allow fold0")
    if gate.get("formal_L01028_execution_allowed") is not False:
        raise RuntimeError("formal execution must still be false before fold0")
    return {"acceptance": acceptance, "gate": gate}


def write_manifest(reference_dir: Path, cache: Path, expected_cache_sha: str, fold0_summary: Path) -> tuple[Path, str]:
    summary = read_json(fold0_summary)
    response_manifest = reference_dir / "reference_frame_manifest.json"
    reference_csv = reference_dir / "selected_reference_timeseries.csv"
    acceptance = reference_dir / "L01028_final_harmonic_cache_acceptance.json"
    common_mask = ROOT / "outputs" / "aquifer_model_revision" / "comparison_common_mask.tif"
    fold_map = ROOT / "outputs" / "aquifer_model_revision" / "spatial_validation_blocks.tif"
    rbf = ROOT / "outputs" / "aquifer_model_revision" / "selected_rbf_design.json"
    if not rbf.exists():
        rbf = ROOT / "outputs" / "aquifer_model_revision" / "rbf_global_basis_selection.json"
    products = {
        "velocity": reference_dir / "velocity" / "insar_vertical_velocity_mm_yr.tif",
        "annual_real": reference_dir / "harmonic" / "annual_vertical_real_sin_mm.tif",
        "annual_imag": reference_dir / "harmonic" / "annual_vertical_imag_cos_mm.tif",
        "annual_amplitude": reference_dir / "harmonic" / "annual_vertical_amplitude_mm.tif",
        "annual_phase": reference_dir / "harmonic" / "annual_vertical_phase_rad.tif",
        "n_observations": reference_dir / "harmonic" / "n_observations.tif",
    }
    payload = {
        "formal_protocol_version": "v2_L01028_fold0_resumable",
        "freeze_timestamp_utc": utc_now(),
        "reference_frame_id": REFERENCE_ID,
        "reference_frame_manifest_path": str(response_manifest),
        "reference_frame_manifest_hash": sha256_file(response_manifest),
        "authoritative_reference_csv_path": str(reference_csv),
        "authoritative_reference_csv_hash": sha256_file(reference_csv) if reference_csv.exists() else None,
        "response_product_hashes": {k: sha256_file(v) for k, v in products.items() if v.exists()},
        "L01028_harmonic_cache_path": str(cache),
        "L01028_harmonic_cache_sha256": expected_cache_sha,
        "cache_key": read_json(acceptance).get("cache_key"),
        "cache_acceptance_json_path": str(acceptance),
        "cache_acceptance_json_sha256": sha256_file(acceptance),
        "combined_static_datasets_sha256": read_json(acceptance).get("combined_static_datasets_sha256"),
        "common_mask_path": str(common_mask),
        "common_mask_sha256": sha256_file(common_mask),
        "fold_map_path": str(fold_map),
        "fold_map_sha256": sha256_file(fold_map),
        "RBF_selection_path": str(rbf),
        "RBF_selection_sha256": sha256_file(rbf),
        "config_yaml_sha256": sha256_file(ROOT / "config.yaml"),
        "source_code_hashes": {
            "scripts/run_L01028_fold0_confirmation.py": sha256_file(ROOT / "scripts" / "run_L01028_fold0_confirmation.py"),
            "scripts/run_L01028_fold0_resume_and_finalize.py": sha256_file(Path(__file__)),
        },
        "fold0_summary_path": str(fold0_summary),
        "fold0_summary_sha256": sha256_file(fold0_summary),
        "selected_lag_u_days": summary["selected_lag_u_days"],
        "selected_lambda": summary["selected_lambda"],
        "selected_stage_c_budget": summary["selected_stage_c_budget"],
        "two_percent_retention_rule": summary.get("two_percent_retention_rule") is True,
        "old_reference_formal_results": "historical_only",
        "no_formal_fold_accessed_during_confirmation": summary.get("no_formal_fold_accessed") is True,
    }
    tmp = FROZEN.with_suffix(FROZEN.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, FROZEN)
    digest = sha256_file(FROZEN)
    tmp_sha = FROZEN_SHA.with_suffix(FROZEN_SHA.suffix + f".tmp.{os.getpid()}")
    tmp_sha.write_text(digest + "\n", encoding="utf-8")
    os.replace(tmp_sha, FROZEN_SHA)
    if FROZEN_SHA.read_text(encoding="utf-8").strip() != sha256_file(FROZEN):
        raise RuntimeError("frozen manifest sidecar SHA256 mismatch")
    return FROZEN, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--reference-dir", default=str(DEFAULT_REF))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir).resolve()
    cache = Path(args.cache).resolve()
    try:
        gates = validate_gates(reference_dir, args.expected_cache_sha256)
        if sha256_file(cache) != args.expected_cache_sha256:
            raise RuntimeError("cache file SHA256 mismatch")
        fold0_dir = reference_dir / "fold0_confirmation"
        lock_path = fold0_dir / "fold0_resume.lock"
        if args.check_only:
            print(json.dumps({"check_only_status": "passed", "cache_acceptance_status": gates["acceptance"]["audit_status"], "formal_L01028_execution_allowed": False, "would_lock": str(lock_path)}, indent=2))
            return 0
        fold0_dir.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return fail(reference_dir, "another_fold0_resume_process_is_running")
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_L01028_fold0_confirmation.py"),
                "--cache",
                str(cache),
                "--fold-id",
                "0",
                "--output-dir",
                str(fold0_dir),
                "--resume",
            ]
            subprocess.run(cmd, cwd=ROOT, check=True)
            summary_path = fold0_dir / "fold0_confirmation_summary.json"
            summary = read_json(summary_path)
            if summary.get("fold0_confirmation_status") != "passed" or summary.get("no_formal_fold_accessed") is not True:
                return fail(reference_dir, "fold0_confirmation_not_passed")
            manifest_path, manifest_hash = write_manifest(reference_dir, cache, args.expected_cache_sha256, summary_path)
            update_status(
                reference_dir,
                {
                    "formal_L01028_execution_allowed": True,
                    "final_full_data_refit_allowed": False,
                    "phase4_restart_allowed": False,
                    "phase5_restart_allowed": False,
                    "formal_execution_block_reason": None,
                    "frozen_manifest_path": str(manifest_path),
                    "frozen_manifest_hash": manifest_hash,
                    "harmonic_cache_path": str(cache),
                    "harmonic_cache_hash": args.expected_cache_sha256,
                    "fold0_confirmation_path": str(summary_path),
                    "fold0_confirmation_hash": sha256_file(summary_path),
                },
            )
            atomic_json(
                reference_dir / "L01028_final_protocol_acceptance.json",
                {
                    "status": "passed",
                    "timestamp_utc": utc_now(),
                    "frozen_manifest_path": str(manifest_path),
                    "frozen_manifest_hash": manifest_hash,
                    "formal_L01028_execution_allowed": True,
                    "formal_cv_started": False,
                    "phase4_started": False,
                    "phase5_started": False,
                },
            )
            print(json.dumps({"status": "passed", "frozen_manifest_hash": manifest_hash}, indent=2))
            return 0
    except Exception as exc:
        if not args.check_only:
            return fail(reference_dir, str(exc))
        print(json.dumps({"check_only_status": "failed", "failure_reason": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
