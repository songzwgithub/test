"""Command-line interface for the single L01028 release implementation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_config(args: argparse.Namespace):
    from .config import load_config

    return load_config(Path(args.config))


def cmd_verify(args: argparse.Namespace) -> dict:
    from .hashing import sha256_file

    cfg = _load_config(args)
    return {
        "cache_hash_match": cfg.authoritative_cache.exists() and sha256_file(cfg.authoritative_cache) == cfg.authoritative_cache_sha256,
        "common_mask_hash_match": cfg.common_mask.exists() and sha256_file(cfg.common_mask) == cfg.common_mask_sha256,
        "fold_map_hash_match": cfg.fold_map.exists() and sha256_file(cfg.fold_map) == cfg.fold_map_sha256,
    }


def cmd_invert(args: argparse.Namespace) -> dict:
    from .cross_validation import recalculate_final_refit

    cfg = _load_config(args)
    from .source_recompute import stream_inputs_from_config

    inputs = stream_inputs_from_config(cfg)
    if not getattr(args, "optimize", False):
        payload = recalculate_final_refit(cfg.release_root, inputs=inputs)
        payload["status"] = "final_refit_recomputed_from_saved_release_parameters"
        payload["release_root"] = str(cfg.release_root)
        return payload
    from .optimization import optimize_formal_inversion

    output_dir = cfg.project_root / getattr(args, "output_dir", "outputs/releases/L01028_v1/recomputed_inversion")
    budgets = dict(cfg.optimizer_budgets)
    if int(args.maxiter) == 0:
        budgets = {name: 0 for name in budgets}
    else:
        budgets["all"] = int(args.maxiter)
    return optimize_formal_inversion(output_dir=output_dir, inputs=inputs, maxiter=int(args.maxiter), budgets=budgets)


def cmd_cv(args: argparse.Namespace) -> dict:
    from .cross_validation import recalculate_formal_cv

    cfg = _load_config(args)
    from .source_recompute import stream_inputs_from_config

    return recalculate_formal_cv(cfg.release_root, inputs=stream_inputs_from_config(cfg))


def cmd_products(args: argparse.Namespace) -> dict:
    from .products import product_audit

    cfg = _load_config(args)
    return product_audit(cfg.release_root / "products")


def cmd_storage(args: argparse.Namespace) -> dict:
    from .storage import recalculate_storage

    cfg = _load_config(args)
    from .source_recompute import stream_inputs_from_config

    return recalculate_storage(cfg.release_root, inputs=stream_inputs_from_config(cfg))


def cmd_figures(args: argparse.Namespace) -> dict:
    cfg = _load_config(args)
    RELEASE_ROOT = cfg.release_root
    figures = {
        "bounded_Ske_map": RELEASE_ROOT / "figures" / "bounded_Ske_map.png",
        "bounded_formal_cv_rmse": RELEASE_ROOT / "figures" / "bounded_formal_cv_rmse.png",
        "figures_acceptance": RELEASE_ROOT / "figures" / "publication_figures_acceptance.json",
    }
    rows = {name: {"path": str(path), "exists": path.exists()} for name, path in figures.items()}
    ok = all(row["exists"] for row in rows.values())
    return {"figures_status": "passed" if ok else "failed_missing_figures", "figures": rows, "figures_path": str(RELEASE_ROOT / "figures")}


def _actual_check_statuses() -> dict:
    from .audit import clean_venv_install_status, wheel_build_status
    from .constants import ROOT

    compile_result = subprocess.run([sys.executable, "-m", "compileall", "-q", "src", "tests"], cwd=ROOT)
    test_result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    smoke_result = subprocess.run(
        [sys.executable, "-m", "hengshui_insar.cli", "verify", "--config", "configs/l01028_release_v1.yaml"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    return {
        "compileall_status": "passed" if compile_result.returncode == 0 else "failed",
        "tests_status": "passed" if test_result.returncode == 0 else "failed",
        "wheel_build_status": wheel_build_status(),
        "clean_venv_install_status": clean_venv_install_status(),
        "cli_smoke_test_status": "passed" if smoke_result.returncode == 0 else "failed",
    }


def cmd_audit(args: argparse.Namespace) -> dict:
    from .audit import release_acceptance

    cfg = _load_config(args)
    from .source_recompute import stream_inputs_from_config

    return release_acceptance(_actual_check_statuses(), release_root=cfg.release_root, inputs=stream_inputs_from_config(cfg))


def cmd_all(args: argparse.Namespace) -> dict:
    from .audit import release_acceptance

    cfg = _load_config(args)
    from .source_recompute import stream_inputs_from_config

    return release_acceptance({
        **_actual_check_statuses(),
    }, release_root=cfg.release_root, inputs=stream_inputs_from_config(cfg))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hengshui-insar", description="Hengshui L01028 release CLI")
    sub = parser.add_subparsers(dest="command")
    commands = {
        "verify": cmd_verify,
        "invert": cmd_invert,
        "cv": cmd_cv,
        "products": cmd_products,
        "storage": cmd_storage,
        "figures": cmd_figures,
        "audit": cmd_audit,
        "all": cmd_all,
    }
    for name, func in commands.items():
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/l01028_release_v1.yaml")
        if name == "invert":
            p.add_argument("--optimize", action="store_true", help="Run formal optimization. Without this flag, invert only recomputes saved final-fit metrics.")
            p.add_argument("--maxiter", type=int, default=300)
            p.add_argument("--output-dir", default="outputs/releases/L01028_v1/recomputed_inversion")
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        payload = args.func(args)
    except Exception as exc:
        payload = {"overall_status": "failed", "failure_reasons": [type(exc).__name__], "error": str(exc)}
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default))
    failed = False
    for key, value in payload.items():
        if key.endswith("_match") and value is not True:
            failed = True
        if key.endswith("_status") and (str(value).startswith("failed") or str(value).startswith("missing")):
            failed = True
    if payload.get("overall_status", "passed") != "passed":
        failed = True
    if str(payload.get("status", "")).startswith("failed"):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
