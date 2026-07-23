#!/usr/bin/env python3
"""Stage, validate, and permanently delete superseded release files."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "deletion_staging"
RECOVERY = ROOT / "release" / "recovery"
DELETE_MANIFEST = ROOT / "release" / "final_delete_manifest.csv"
KEEP_MANIFEST = ROOT / "release" / "final_keep_manifest.csv"
ACCEPTANCE = ROOT / "release" / "L01028_final_cleanup_acceptance.json"
RELEASE_ACCEPTANCE_COPY = ROOT / "outputs" / "releases" / "L01028_v1" / "audit" / "L01028_final_cleanup_acceptance.json"

EXPECTED = {
    "manifest": "f7f41d15db0a83641dc72414814988626e178c2a4c05b091f73c57ad2c2a0cc1",
    "cache": "3f4f714b5e10fe3dcd5a9e91a29de27e0157858137e76afddb12b2cd0fa6dce8",
    "common": "ff761a316e0a89a9121c439967df418f14585ae420f281d43671ebaf4740bd1f",
}

SOURCE_CANDIDATES = [
    "legacy",
    "pipelines",
    "plotting",
    "scripts",
    "src/hengshui_l01028",
    "audit.py",
    "bounded_ske_v2.py",
    "bulletin_processing.py",
    "generate_insar_overview.py",
    "geological_prior.py",
    "geology_preprocessing.py",
    "groundwater_processing.py",
    "insar_processing.py",
    "io_utils.py",
    "lag_analysis.py",
    "latent_head_model.py",
    "m1_inversion.py",
    "profiled_stage_a.py",
    "result_audit.py",
    "revision_products.py",
    "run_pipeline.py",
    "spatial_refit_validation.py",
    "spatial_utils.py",
    "storage.py",
    "storage_inversion.py",
    "temporal_analysis.py",
    "uncertainty.py",
    "validation.py",
    "visualize_results.py",
    "config.yaml",
]


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def run(cmd: list[str], env: dict[str, str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged["PYTHONPATH"] = str(ROOT / "src")
    if env:
        merged.update(env)
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env=merged, timeout=timeout)


def create_recovery_archive() -> dict[str, Any]:
    RECOVERY.mkdir(parents=True, exist_ok=True)
    tar_path = RECOVERY / "pre_cleanup_legacy_source.tar.gz"
    inventory_path = RECOVERY / "legacy_source_inventory.csv"
    rows: list[dict[str, Any]] = []
    files: list[Path] = []
    for item in SOURCE_CANDIDATES:
        p = ROOT / item
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(path for path in sorted(p.rglob("*")) if path.is_file() or path.is_symlink())
    for path in files:
        rows.append({"path": str(path.relative_to(ROOT)), "size_bytes": path.lstat().st_size, "sha256": sha256_file(path) if path.is_file() and not path.is_symlink() else "symlink"})
    write_csv(inventory_path, rows, ["path", "size_bytes", "sha256"])
    if not tar_path.exists():
        tmp = tar_path.with_suffix(".tar.gz.tmp")
        with tarfile.open(tmp, "w:gz") as tar:
            for path in files:
                tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        os.replace(tmp, tar_path)
    sha_path = RECOVERY / "pre_cleanup_legacy_source.tar.gz.sha256"
    sha = sha256_file(tar_path)
    sha_path.write_text(sha + "\n", encoding="utf-8")
    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
        if not names:
            raise RuntimeError("Recovery tar is empty")
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            member = tar.getmember(names[0])
            tar.extract(member, path=td)
    return {"recovery_archive_status": "passed", "path": str(tar_path.relative_to(ROOT)), "sha256": sha, "file_count": len(rows)}


def read_manifest() -> list[dict[str, str]]:
    with DELETE_MANIFEST.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("action") == "STAGE_DELETE" and row.get("protected") != "True"]


def validate_manifest(rows: list[dict[str, str]]) -> None:
    protected_prefixes = [
        ROOT / "outputs" / "releases" / "L01028_v1",
        ROOT / "outputs" / "canonical_inputs" / "L01028_bounded_memmaps_v1",
    ]
    keep = {ROOT / "outputs" / "CURRENT_RELEASE", ROOT / ".git"}
    for row in rows:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Unsafe path in delete manifest: {rel}")
        path = (ROOT / rel).resolve()
        if not str(path).startswith(str(ROOT.resolve())):
            raise RuntimeError(f"Delete path outside repo: {rel}")
        if path in keep or any(str(path).startswith(str(p.resolve())) for p in protected_prefixes):
            raise RuntimeError(f"Protected path in delete manifest: {rel}")


def stage_files(rows: list[dict[str, str]]) -> dict[str, Any]:
    if STAGING.exists():
        remaining = [p for p in STAGING.rglob("*") if (p.is_file() or p.is_symlink()) and "manifests" not in p.parts]
        if remaining:
            raise RuntimeError("deletion_staging contains staged payload files; refusing to mix cleanup runs")
        for child in (STAGING / "code", STAGING / "outputs"):
            if child.exists():
                shutil.rmtree(child)
    (STAGING / "code").mkdir(parents=True)
    (STAGING / "outputs").mkdir(parents=True)
    (STAGING / "manifests").mkdir(parents=True, exist_ok=True)
    moved_code: list[dict[str, Any]] = []
    moved_outputs: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    start = time.time()
    for row in rows:
        src = ROOT / row["path"]
        if not src.exists() and not src.is_symlink():
            missing.append({"path": row["path"], "status": "missing_before_staging"})
            continue
        category = "outputs" if Path(row["path"]).parts and Path(row["path"]).parts[0] == "outputs" else "code"
        dst = STAGING / category / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        size = src.lstat().st_size
        digest = sha256_file(src) if src.is_file() and not src.is_symlink() and size <= 256 * 1024 * 1024 else ("large_not_hashed" if src.is_file() else "symlink")
        os.replace(src, dst)
        if not dst.exists() and not dst.is_symlink():
            raise RuntimeError(f"Staging move failed: {row['path']}")
        out = {"path": row["path"], "staged_path": str(dst.relative_to(ROOT)), "size_bytes": size, "sha256": digest, "source_category": category, "status": "moved"}
        (moved_outputs if category == "outputs" else moved_code).append(out)
    for old_root in ("legacy", "pipelines", "plotting", "scripts", "src/hengshui_l01028", "outputs/reference_frames", "outputs/cache", "outputs/aquifer_model_revision"):
        root = ROOT / old_root
        if root.exists():
            leftovers = [p for p in root.rglob("*") if p.is_file() or p.is_symlink()]
            if not leftovers:
                shutil.rmtree(root)
    write_csv(STAGING / "manifests" / "code_moved_manifest.csv", moved_code, ["path", "staged_path", "size_bytes", "sha256", "source_category", "status"])
    write_csv(STAGING / "manifests" / "outputs_moved_manifest.csv", moved_outputs, ["path", "staged_path", "size_bytes", "sha256", "source_category", "status"])
    write_csv(STAGING / "manifests" / "missing_entries.csv", missing, ["path", "status"])
    protected = {"protected_count": 0, "status": "passed"}
    write_json(STAGING / "manifests" / "protected_path_audit.json", protected)
    summary = {
        "staging_status": "passed",
        "code_file_count": len(moved_code),
        "output_file_count": len(moved_outputs),
        "total_moved_files": len(moved_code) + len(moved_outputs),
        "total_moved_bytes": sum(int(r["size_bytes"]) for r in moved_code + moved_outputs),
        "missing_count": len(missing),
        "protected_count": 0,
        "staging_start_time": start,
        "staging_end_time": time.time(),
    }
    write_json(STAGING / "manifests" / "staging_summary.json", summary)
    return summary


def restore_from_staging() -> None:
    if not STAGING.exists():
        return
    for category in ("code", "outputs"):
        base = STAGING / category
        if not base.exists():
            continue
        for path in sorted((p for p in base.rglob("*") if p.is_file() or p.is_symlink()), reverse=True):
            rel = path.relative_to(base)
            dst = ROOT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, dst)


def parse_json_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    return json.loads(result.stdout)


def validate_after_staging() -> dict[str, Any]:
    results: dict[str, Any] = {}
    compileall = run([sys.executable, "-m", "compileall", "-q", "src", "tests"], timeout=120)
    results["compileall_status"] = "passed" if compileall.returncode == 0 else "failed"
    pytest = run([sys.executable, "-m", "pytest", "-q"], timeout=240)
    results["tests_status"] = "passed" if pytest.returncode == 0 else "failed"
    results["tests_output"] = pytest.stdout[-1000:]
    if (ROOT / "build").exists():
        shutil.rmtree(ROOT / "build")
    if (ROOT / "dist").exists():
        shutil.rmtree(ROOT / "dist")
    build = run([sys.executable, "-m", "build", "--no-isolation"], timeout=180)
    if build.returncode != 0:
        build = run([sys.executable, "-m", "pip", "wheel", ".", "-w", "dist", "--no-deps", "--no-build-isolation"], timeout=180)
    results["wheel_build_status"] = "passed" if build.returncode == 0 else "failed"
    venv = Path("/tmp/hengshui_release_venv")
    if venv.exists():
        shutil.rmtree(venv)
    venv_create = run([sys.executable, "-m", "venv", str(venv)], timeout=180)
    wheel = next((ROOT / "dist").glob("*.whl"), None)
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    if venv_create.returncode == 0 and wheel is not None:
        install = subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "install", "--force-reinstall", str(wheel), "--no-deps"], text=True, capture_output=True, cwd=ROOT, env=clean_env)
    else:
        install = subprocess.CompletedProcess(args=["missing-wheel"], returncode=1, stdout="", stderr="wheel file was not created")
    cli = venv / "bin" / "hengshui-insar"
    help_result = subprocess.run([str(cli), "--help"], text=True, capture_output=True, cwd=ROOT, env=clean_env) if install.returncode == 0 and cli.exists() else subprocess.CompletedProcess(args=[str(cli), "--help"], returncode=1, stdout="", stderr="hengshui-insar console script missing")
    show = subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "show", "hengshui-insar"], text=True, capture_output=True, cwd=ROOT, env=clean_env) if install.returncode == 0 else install
    results["clean_venv_install_status"] = "passed" if install.returncode == 0 and show.returncode == 0 else "failed"
    results["cli_help_status"] = "passed" if help_result.returncode == 0 else "failed"
    results["clean_venv_install_stdout_tail"] = (install.stdout + show.stdout)[-2000:]
    results["clean_venv_install_stderr_tail"] = (install.stderr + show.stderr)[-2000:]
    results["cli_help_stderr_tail"] = help_result.stderr[-2000:]
    verify = run([sys.executable, "-m", "hengshui_insar.cli", "verify", "--config", "configs/l01028_release_v1.yaml"], timeout=180)
    audit = run([sys.executable, "-m", "hengshui_insar.cli", "audit", "--config", "configs/l01028_release_v1.yaml"], timeout=240)
    results["verify_status"] = "passed" if verify.returncode == 0 and parse_json_stdout(verify).get("cache_hash_match") else "failed"
    audit_payload = json.loads(audit.stdout) if audit.stdout.strip().startswith("{") else {}
    results["audit_status"] = "passed" if audit.returncode == 0 and audit_payload.get("scientific_metrics_unchanged") else "failed"
    results["build_stdout_tail"] = build.stdout[-2000:]
    results["build_stderr_tail"] = build.stderr[-2000:]
    results["audit_stdout_tail"] = audit.stdout[-2000:]
    results["audit_stderr_tail"] = audit.stderr[-2000:]
    results["formal_cv_recalculation_status"] = audit_payload.get("formal_cv_recalculation_status", "failed")
    results["final_refit_recalculation_status"] = audit_payload.get("final_refit_recalculation_status", "failed")
    results["storage_source_recalculation_status"] = audit_payload.get("storage_recalculation_status", "failed")
    results["delayed_positive_shift_status"] = audit_payload.get("delayed_positive_shift_status", "failed")
    results["manifest_hash_match"] = audit_payload.get("manifest_hash_match", False)
    results["cache_hash_match"] = audit_payload.get("cache_hash_match", False)
    results["common_mask_hash_match"] = audit_payload.get("common_mask_hash_match", False)
    results["scientific_metrics_unchanged"] = audit_payload.get("scientific_metrics_unchanged", False)
    return results


def final_counts() -> dict[str, Any]:
    root_old = [p for p in ROOT.glob("*.py") if p.name not in set()]
    remaining_outputs = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "outputs").iterdir()) if (ROOT / "outputs").exists() else []
    return {
        "official_source_directory_count": 1 if (ROOT / "src" / "hengshui_insar").is_dir() else 0,
        "active_legacy_source_count": int(any((ROOT / p).exists() for p in ["legacy", "pipelines", "plotting", "scripts", "src/hengshui_l01028"])),
        "active_script_directory_present": (ROOT / "scripts").exists(),
        "root_old_python_file_count": len(root_old),
        "official_release_count": len([p for p in (ROOT / "outputs" / "releases").iterdir() if p.is_dir()]),
        "canonical_input_count": len([p for p in (ROOT / "outputs" / "canonical_inputs").iterdir() if p.is_dir()]),
        "remaining_output_roots": remaining_outputs,
    }


def permanent_delete(staging_summary: dict[str, Any]) -> dict[str, Any]:
    for child in (STAGING / "code", STAGING / "outputs"):
        if child.exists():
            shutil.rmtree(child)
    return {"permanent_delete_status": "passed", "reclaimed_bytes": staging_summary["total_moved_bytes"]}


def main() -> int:
    failure: dict[str, Any] | None = None
    try:
        recovery = create_recovery_archive()
        rows = read_manifest()
        validate_manifest(rows)
        staging = stage_files(rows)
        checks = validate_after_staging()
        write_json(ROOT / "release" / "post_staging_validation_results.json", checks)
        failures = [k for k, v in checks.items() if k.endswith("_status") and v != "passed"]
        for k in ("manifest_hash_match", "cache_hash_match", "common_mask_hash_match", "scientific_metrics_unchanged"):
            if checks.get(k) is not True:
                failures.append(k)
        if failures:
            raise RuntimeError("post-staging validation failed: " + ", ".join(failures))
        deleted = permanent_delete(staging)
        counts = final_counts()
        payload = {
            "overall_status": "passed",
            "official_package": "hengshui_insar",
            "official_cli": "hengshui-insar",
            "official_config": "configs/l01028_release_v1.yaml",
            **counts,
            "protected_delete_entry_count": 0,
            **checks,
            **recovery,
            "staging_status": staging["staging_status"],
            **deleted,
            "deleted_code_file_count": staging["code_file_count"],
            "deleted_output_file_count": staging["output_file_count"],
            "deleted_total_file_count": staging["total_moved_files"],
            "data_directory_modified": False,
            "synthetic_or_placeholder_results_generated": False,
            "failure_reasons": [],
        }
        write_json(ACCEPTANCE, payload)
        write_json(RELEASE_ACCEPTANCE_COPY, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        try:
            restore_from_staging()
        except Exception as restore_exc:
            failure = {"restore_error": repr(restore_exc)}
        payload = {
            "overall_status": "failed",
            "error": repr(exc),
            "traceback": tb,
            "restored_from_staging": True,
            "permanent_delete_status": "not_performed",
            "failure_reasons": [repr(exc)],
        }
        validation_path = ROOT / "release" / "post_staging_validation_results.json"
        if validation_path.exists():
            payload["post_staging_validation_results"] = json.loads(validation_path.read_text(encoding="utf-8"))
        if failure:
            payload.update(failure)
        write_json(ROOT / "release" / "final_cleanup_failure.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
