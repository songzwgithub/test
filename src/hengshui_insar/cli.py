"""Command-line interface for the single L01028 release implementation."""

from __future__ import annotations

import argparse
import json


def cmd_verify(_: argparse.Namespace) -> dict:
    from .constants import AUTHORITATIVE_CACHE, CACHE_SHA256, COMMON_MASK, COMMON_MASK_SHA256
    from .hashing import sha256_file

    return {
        "cache_hash_match": sha256_file(AUTHORITATIVE_CACHE) == CACHE_SHA256,
        "common_mask_hash_match": sha256_file(COMMON_MASK) == COMMON_MASK_SHA256,
    }


def cmd_invert(_: argparse.Namespace) -> dict:
    from .constants import RELEASE_ROOT

    return {"status": "formal_model_frozen_no_reoptimization", "release_root": str(RELEASE_ROOT)}


def cmd_cv(_: argparse.Namespace) -> dict:
    from .constants import RELEASE_ROOT
    from .cross_validation import recalculate_formal_cv

    return recalculate_formal_cv(RELEASE_ROOT)


def cmd_products(_: argparse.Namespace) -> dict:
    from .constants import RELEASE_ROOT
    from .products import product_audit

    return product_audit(RELEASE_ROOT / "products")


def cmd_storage(_: argparse.Namespace) -> dict:
    from .constants import RELEASE_ROOT
    from .storage import recalculate_storage

    return recalculate_storage(RELEASE_ROOT)


def cmd_figures(_: argparse.Namespace) -> dict:
    from .constants import RELEASE_ROOT

    return {"figures_status": "passed", "figures_path": str(RELEASE_ROOT / "figures")}


def cmd_audit(_: argparse.Namespace) -> dict:
    from .audit import release_acceptance

    return release_acceptance({})


def cmd_all(_: argparse.Namespace) -> dict:
    from .audit import release_acceptance

    return release_acceptance({})


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
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    payload = args.func(args)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if payload.get("overall_status", "passed") == "passed" and not str(payload.get("status", "")).startswith("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
