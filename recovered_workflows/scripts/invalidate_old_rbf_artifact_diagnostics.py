#!/usr/bin/env python
"""Mark old 64-center RBF artifact diagnostics as invalidated by basis reduction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def invalidate(path, original_center_count=64, new_active_center_count=22):
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload.update(
        {
            "status": "invalidated_by_rbf_basis_reduction",
            "original_center_count": int(original_center_count),
            "new_active_center_count": int(new_active_center_count),
            "requires_recomputation_after_final_phase4": True,
            "do_not_use_for_current_22_center_model": True,
        }
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="outputs/rbf_artifact_diagnostics.json")
    parser.add_argument("--original-center-count", type=int, default=64)
    parser.add_argument("--new-active-center-count", type=int, default=22)
    args = parser.parse_args()
    payload = invalidate(args.path, args.original_center_count, args.new_active_center_count)
    print(json.dumps({k: payload[k] for k in ("status", "original_center_count", "new_active_center_count", "requires_recomputation_after_final_phase4")}, indent=2))


if __name__ == "__main__":
    main()
