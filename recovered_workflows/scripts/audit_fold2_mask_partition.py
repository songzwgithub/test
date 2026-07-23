#!/usr/bin/env python
"""Audit fold2 mask partition counts and Cu practical identifiability."""
from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiled_stage_a import latest_real_harmonic_cache
from scripts.run_formal_g0_fold1 import iter_blocks_fold
from scripts.run_stage_c_fixed_lagu import LAG_U_FIXED_DAYS, decode
from storage_inversion import rotate_coefficients


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def mask_partition_counts(mask_path: Path, blocks_path: Path, fold_id: int = 2) -> dict:
    fold_counts = {}
    common = training = validation = intersection = union = 0
    with rasterio.open(mask_path) as msrc, rasterio.open(blocks_path) as bsrc:
        for _, window in msrc.block_windows(1):
            mask = msrc.read(1, window=window) == 1
            folds = bsrc.read(1, window=window)
            common += int(mask.sum())
            train = mask & (folds != fold_id)
            val = mask & (folds == fold_id)
            training += int(train.sum())
            validation += int(val.sum())
            intersection += int((train & val).sum())
            union += int((train | val).sum())
            vals, counts = np.unique(folds[mask], return_counts=True)
            for value, count in zip(vals, counts):
                fold_counts[int(value)] = fold_counts.get(int(value), 0) + int(count)
    return {
        "common_mask_pixel_count": common,
        "training_pixel_count": training,
        "validation_pixel_count": validation,
        "training_validation_intersection_count": intersection,
        "training_validation_union_count": union,
        "fold_map_common_mask_counts": fold_counts,
    }


def cu_identifiability(root: Path, selected: dict, transform: np.ndarray) -> dict:
    fold = root / "model_compare/G0_no_geology_L0_shared/fold_02"
    stage_a = json.loads((fold / "stage_A_training_only_result.json").read_text())
    theta = np.load(fold / "final_training_checkpoint.npy").astype(float)
    _log_ske, _gamma, cu, _lag_c = decode(theta)
    cache = latest_real_harmonic_cache()
    mask = root / "comparison_common_mask.tif"
    blocks = root / "spatial_validation_blocks.tif"
    unconf_sq = total_sq = 0.0
    n = 0
    for obs, hc, hu, basis in iter_blocks_fold(cache, mask, blocks, selected, transform, 2, True):
        log_ske, gamma, cu, lag_c = decode(theta)
        spatial = basis @ gamma
        ske = np.exp(np.clip(log_ske + spatial, -20, 10))
        rc = rotate_coefficients(hc, lag_c)
        ru = rotate_coefficients(hu, LAG_U_FIXED_DAYS)
        unconf = 1000.0 * cu * ru
        total = 1000.0 * (ske[:, None] * rc + cu * ru)
        unconf_sq += float(np.sum(unconf * unconf))
        total_sq += float(np.sum(total * total))
        n += int(unconf.size)
    ratio = float(cu / stage_a["Cu_global"]) if stage_a["Cu_global"] else np.nan
    return {
        "Cu_stageA": float(stage_a["Cu_global"]),
        "Cu_stageC": float(cu),
        "Cu_stageC_to_stageA_ratio": ratio,
        "unconfined_contribution_rms_mm": float(np.sqrt(unconf_sq / max(n, 1))),
        "unconfined_variance_fraction": float(unconf_sq / max(total_sq, 1e-30)),
        "Cu_practically_zero": bool(ratio < 0.001 or cu < 1e-6),
        "model_modified_due_to_Cu_audit": False,
    }


def main() -> None:
    root = Path("outputs/aquifer_model_revision")
    fold = root / "model_compare/G0_no_geology_L0_shared/fold_02"
    manifest = json.loads((root / "formal_protocol_frozen_manifest.json").read_text())
    metrics_path = fold / "single_final_outer_validation_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    counts = mask_partition_counts(root / "comparison_common_mask.tif", root / "spatial_validation_blocks.tif", 2)
    manifest_hash_ok = manifest["manifest_hash"] == "bd08b8640af45badd9c87cf5111791be9d10789699bf312972a9af48070219fe"
    expected_ok = (
        counts["common_mask_pixel_count"] == 15241589
        and counts["training_pixel_count"] == 10333028
        and counts["validation_pixel_count"] == 4908561
        and counts["training_validation_intersection_count"] == 0
        and counts["training_validation_union_count"] == 15241589
        and manifest_hash_ok
    )
    audit = {
        **counts,
        "common_mask_hash": hash_file(root / "comparison_common_mask.tif"),
        "fold_map_hash": hash_file(root / "spatial_validation_blocks.tif"),
        "manifest_hash": manifest["manifest_hash"],
        "manifest_hash_unchanged": manifest_hash_ok,
        "reported_training_pixel_count_in_metrics": metrics.get("training_pixel_count"),
        "reported_validation_pixel_count_in_metrics": metrics.get("validation_pixel_count"),
        "metrics_pixel_counts_consistent": bool(
            metrics.get("training_pixel_count") == counts["training_pixel_count"]
            and metrics.get("validation_pixel_count") == counts["validation_pixel_count"]
        ),
        "status": "passed" if expected_ok else "failed_mask_partition_mismatch",
    }
    selected = json.loads((root / "selected_rbf_design.json").read_text())
    transform = np.load(root / "rbf_orthogonalization" / "rbf_transform.npy")
    cu_audit = cu_identifiability(root, selected, transform)
    audit["cu_practical_identifiability"] = cu_audit
    write_json(fold / "fold2_mask_partition_audit.json", audit)
    write_json(fold / "fold2_Cu_practical_identifiability_audit.json", cu_audit)
    status_path = root / "aquifer_model_revision_status.json"
    status = json.loads(status_path.read_text())
    status["fold2_mask_partition_audit"] = audit["status"]
    status["fold2_Cu_practically_zero"] = cu_audit["Cu_practically_zero"]
    status["phase4_restart_allowed"] = False
    status["selected_model_config"] = "not_generated"
    if expected_ok:
        status["fold2_formal_cv_eligible"] = True
        status["allow_continue_g0_fold3"] = True
    else:
        status["fold2_formal_cv_eligible"] = False
        status["allow_continue_g0_fold3"] = False
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
