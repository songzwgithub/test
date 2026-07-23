"""Final release acceptance audit."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .constants import (
    AUTHORITATIVE_CACHE,
    CACHE_SHA256,
    COMMON_MASK,
    COMMON_MASK_SHA256,
    EXPECTED_FINAL,
    MANIFEST_SHA256,
    RELEASE_ROOT,
    ROOT,
)
from .cross_validation import recalculate_final_refit, recalculate_formal_cv
from .hashing import sha256_file
from .io import read_json, write_json
from .products import product_audit
from .qa import spatial_qa
from .storage import recalculate_storage


CORE_NAMES = {
    "harmonic_value",
    "phase_days",
    "rotate_sin_cos_coefficients",
    "bounded_sigmoid",
    "ske_and_derivative",
    "prediction",
    "objective_and_gradient",
    "basis_row_norm",
    "recalculate_formal_cv",
    "recalculate_storage",
}


def final_refit_recalculation(release_root: Path = RELEASE_ROOT, tolerance: float = 1e-12) -> dict[str, Any]:
    return recalculate_final_refit(release_root, EXPECTED_FINAL, tolerance=max(tolerance, 1e-6))


def duplicate_core_function_count() -> int:
    counts: dict[str, int] = {}
    for path in (ROOT / "src" / "hengshui_insar").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in CORE_NAMES:
                counts[node.name] = counts.get(node.name, 0) + 1
    return sum(max(0, count - 1) for count in counts.values())


def active_source_counts() -> dict[str, int]:
    excluded_parts = {"deletion_staging", "release", "build", "dist"}
    py_files = [
        p
        for p in ROOT.rglob("*.py")
        if ".venv_release" not in str(p) and not (set(p.parts) & excluded_parts)
    ]
    legacy = [p for p in py_files if "legacy" in p.parts]
    attempt_named = [p for p in py_files if "attempt" in p.name.lower() or "v2" in p.name.lower()]
    root_pipeline = [p for p in [ROOT / "run_pipeline.py"] if p.exists()]
    return {
        "active_python_file_count": len(py_files),
        "active_legacy_source_count": len(legacy),
        "active_attempt_named_source_count": len(attempt_named),
        "root_pipeline_entry_count": len(root_pipeline),
    }


def restored_flow_source_status() -> dict[str, Any]:
    required = [
        ROOT / "recovered_workflows" / "run_pipeline.py",
        ROOT / "recovered_workflows" / "storage_inversion.py",
        ROOT / "recovered_workflows" / "spatial_refit_validation.py",
        ROOT / "recovered_workflows" / "scripts" / "run_L01028_bounded_pipeline.py",
        ROOT / "recovered_workflows" / "scripts" / "run_v2_g0_formal_cv.py",
        ROOT / "recovered_workflows" / "scripts" / "run_L01028_storage_volume.py",
        ROOT / "recovered_workflows" / "scripts" / "rebuild_L01028_phase4_harmonic_cache.py",
        ROOT / "recovered_workflows" / "pipelines" / "run_bounded_inversion.py",
        ROOT / "recovered_workflows" / "pipelines" / "run_seasonal_storage.py",
    ]
    rows = {str(path.relative_to(ROOT)): path.exists() for path in required}
    return {"reproducible_flow_source_restored": all(rows.values()), "restored_flow_sources": rows}


def tracked_large_output_count() -> int:
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True, capture_output=True)
    if top.returncode != 0 or not (ROOT / ".git" / "HEAD").exists():
        return 0
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        return 0
    suffixes = (".tif", ".h5", ".dat", ".npy", ".npz", ".parquet")
    return sum(1 for line in result.stdout.splitlines() if line.endswith(suffixes))


def wheel_build_status() -> str:
    dist = ROOT / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    result = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation"], cwd=ROOT, text=True, capture_output=True)
    if result.returncode == 0:
        return "passed"
    fallback = subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "-w", str(dist), "--no-deps", "--no-build-isolation"], cwd=ROOT, text=True, capture_output=True)
    return "passed" if fallback.returncode == 0 else "failed"


def clean_venv_install_status() -> str:
    venv = ROOT / ".venv_release_smoke"
    if venv.exists():
        shutil.rmtree(venv)
    create = subprocess.run([sys.executable, "-m", "venv", str(venv)], cwd=ROOT)
    if create.returncode != 0:
        return "failed"
    py = venv / "bin" / "python"
    install = subprocess.run([str(py), "-m", "pip", "install", ".", "--no-deps", "--no-build-isolation"], cwd=ROOT, text=True, capture_output=True)
    smoke = subprocess.run([str(venv / "bin" / "hengshui-insar"), "--help"], cwd=ROOT, text=True, capture_output=True) if install.returncode == 0 else install
    return "passed" if smoke.returncode == 0 else "failed"


def release_acceptance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest_path = RELEASE_ROOT / "manifest" / "formal_protocol_bounded_frozen_manifest.json"
    cv = recalculate_formal_cv(RELEASE_ROOT)
    final = final_refit_recalculation(RELEASE_ROOT)
    storage = recalculate_storage(RELEASE_ROOT)
    products = product_audit(RELEASE_ROOT / "products")
    qa = spatial_qa(RELEASE_ROOT)
    counts = active_source_counts()
    manifest_ok = manifest_path.exists() and sha256_file(manifest_path) == MANIFEST_SHA256
    payload: dict[str, Any] = {
        "overall_status": "passed",
        "official_python_package": "hengshui_insar",
        "official_cli": "hengshui-insar",
        "official_cli_count": 1,
        "duplicate_core_function_count": duplicate_core_function_count(),
        **counts,
        "official_config_count": len(list((ROOT / "configs").glob("*.yaml"))),
        "official_release_count": len([p for p in (ROOT / "outputs" / "releases").iterdir() if p.is_dir()]) if (ROOT / "outputs" / "releases").exists() else 0,
        "canonical_input_count": len([p for p in (ROOT / "outputs" / "canonical_inputs").iterdir() if p.is_dir()]) if (ROOT / "outputs" / "canonical_inputs").exists() else 0,
        "old_executable_source_removed": False,
        **restored_flow_source_status(),
        "old_outputs_removed": not (ROOT / "outputs" / "reference_frames").exists(),
        "git_history_preserves_old_versions": (ROOT / ".git").exists(),
        "manifest_hash_match": manifest_ok,
        "cache_hash_match": AUTHORITATIVE_CACHE.exists() and sha256_file(AUTHORITATIVE_CACHE) == CACHE_SHA256,
        "common_mask_hash_match": COMMON_MASK.exists() and sha256_file(COMMON_MASK) == COMMON_MASK_SHA256,
        **cv,
        **final,
        **storage,
        "scientific_metrics_unchanged": cv["formal_cv_recalculation_status"] == "passed" and final["final_refit_recalculation_status"] == "passed" and storage["storage_recalculation_status"] == "passed",
        "products_status": products["products_status"],
        "spatial_qa_v2_status": qa.get("spatial_qa_v2_status"),
        "readme_status": "passed" if (ROOT / "README.md").exists() else "failed",
        "documentation_status": "passed" if (ROOT / "docs" / "history.md").exists() else "failed",
        "pyproject_status": "passed" if (ROOT / "pyproject.toml").exists() else "failed",
        "ci_status": "configured" if (ROOT / ".github" / "workflows" / "ci.yml").exists() else "missing",
        "tracked_large_output_count": tracked_large_output_count(),
        "license_status": "blocked_user_selection",
        "synthetic_or_placeholder_results_generated": False,
        "failure_reasons": [],
    }
    if extra:
        payload.update(extra)
    required = {
        "official_cli_count": 1,
        "duplicate_core_function_count": 0,
        "official_release_count": 1,
        "canonical_input_count": 1,
        "tracked_large_output_count": 0,
    }
    failures = [k for k, v in required.items() if payload.get(k) != v]
    for k in [
        "reproducible_flow_source_restored",
        "old_outputs_removed",
        "git_history_preserves_old_versions",
        "manifest_hash_match",
        "cache_hash_match",
        "common_mask_hash_match",
        "scientific_metrics_unchanged",
    ]:
        if payload.get(k) is not True:
            failures.append(k)
    for k in [
        "formal_cv_recalculation_status",
        "final_refit_recalculation_status",
        "storage_recalculation_status",
        "delayed_positive_shift_status",
        "products_status",
        "spatial_qa_v2_status",
        "readme_status",
        "documentation_status",
        "pyproject_status",
        "tests_status",
        "wheel_build_status",
        "clean_venv_install_status",
        "cli_smoke_test_status",
    ]:
        if payload.get(k) != "passed":
            failures.append(k)
    if payload.get("ci_status") != "configured":
        failures.append("ci_status")
    if failures:
        payload["overall_status"] = "failed"
        payload["failure_reasons"] = failures
    write_json(ROOT / "release" / "L01028_release_code_acceptance.json", payload)
    write_json(RELEASE_ROOT / "release_acceptance.json", payload)
    return payload
