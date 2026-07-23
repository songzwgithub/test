#!/usr/bin/env python
"""Pre-run mask partition audit for formal fold4."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import rasterio


EXPECTED = {
    "common_mask_pixel_count": 15_241_589,
    "training_pixel_count": 14_011_445,
    "validation_pixel_count": 1_230_144,
    "intersection_pixel_count": 0,
    "union_pixel_count": 15_241_589,
    "training_validation_intersection_count": 0,
    "training_validation_union_count": 15_241_589,
    "common_mask_hash": "ff761a316e0a89a9121c439967df418f14585ae420f281d43671ebaf4740bd1f",
    "fold_map_hash": "d24dc63e65d3a1fa1a0e698620ba6d8e03fcf518a9a5ef0721c59374a1d46e3a",
    "manifest_hash": "bd08b8640af45badd9c87cf5111791be9d10789699bf312972a9af48070219fe",
}


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path("outputs/aquifer_model_revision")
    fold_dir = root / "model_compare/G0_no_geology_L0_shared/fold_04"
    fold_dir.mkdir(parents=True, exist_ok=True)
    mask_path = root / "comparison_common_mask.tif"
    blocks_path = root / "spatial_validation_blocks.tif"
    manifest = json.loads((root / "formal_protocol_frozen_manifest.json").read_text())
    common = training = validation = intersection = union = 0
    with rasterio.open(mask_path) as msrc, rasterio.open(blocks_path) as bsrc:
        for _, window in msrc.block_windows(1):
            mask = msrc.read(1, window=window) == 1
            folds = bsrc.read(1, window=window)
            train = mask & (folds != 4)
            val = mask & (folds == 4)
            common += int(mask.sum())
            training += int(train.sum())
            validation += int(val.sum())
            intersection += int((train & val).sum())
            union += int((train | val).sum())
    audit = {
        "common_mask_pixel_count": common,
        "training_pixel_count": training,
        "validation_pixel_count": validation,
        "intersection_pixel_count": intersection,
        "union_pixel_count": union,
        "training_validation_intersection_count": intersection,
        "training_validation_union_count": union,
        "common_mask_hash": hash_file(mask_path),
        "fold_map_hash": hash_file(blocks_path),
        "manifest_hash": manifest["manifest_hash"],
    }
    audit["status"] = "passed" if all(audit[k] == v for k, v in EXPECTED.items()) else "failed"
    (fold_dir / "fold4_mask_partition_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
