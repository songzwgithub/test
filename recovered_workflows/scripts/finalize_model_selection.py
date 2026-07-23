from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from io_utils import ROOT, load_config, write_json
from run_pipeline import _spatial_basis_builder


def _sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _basis_hashes(config, spacing_km: float) -> dict:
    output = ROOT / config["project"]["output_dir"]
    spatial = dict(config["geology"].get("spatial_basis", {}))
    spatial["candidate_spacing_km"] = [float(spacing_km)]
    with rasterio.open(output / "geological_model_covariates.tif") as src:
        metadata, _, basis = _spatial_basis_builder(src, spatial, config["project"].get("projected_crs", "EPSG:32650"))
        rows = np.linspace(0, src.height - 1, 32).astype(int)
        cols = np.linspace(0, src.width - 1, 32).astype(int)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        xs, ys = rasterio.transform.xy(src.transform, rr.ravel(), cc.ravel(), offset="center")
        coords = np.column_stack([xs, ys])
        design = basis(coords)
    return {
        "rbf_spacing_km": float(spacing_km),
        "n_centers": len(metadata["centers"]) if metadata else 0,
        "rbf_centers_sha256": _sha_json(metadata["centers"] if metadata else []),
        "design_matrix_sha256": hashlib.sha256(np.asarray(design, dtype="float32").tobytes()).hexdigest(),
        "basis_hash": _sha_json(metadata),
    }


def _cache_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        attrs = {k: h5.attrs.get(k) for k in h5.attrs.keys()}
        return _sha_json(attrs | {"path": str(path), "size": path.stat().st_size})


def run(config_path: str = "config.yaml") -> dict:
    config = load_config(config_path)
    output = ROOT / config["project"]["output_dir"]
    mc_path = output / "model_comparison.csv"
    mc = pd.read_csv(mc_path)
    diagnostics = json.loads((output / "map_diagnostics.json").read_text(encoding="utf-8"))
    cache_path = Path(diagnostics.get("cache_path", ""))
    cache_hash = _cache_hash(cache_path)
    spacings = config["geology"]["spatial_basis"].get("candidate_spacing_km", [5, 10, 15])
    hashes = pd.DataFrame([_basis_hashes(config, s) for s in spacings])
    hashes.to_csv(output / "rbf_basis_hashes.csv", index=False)
    current_spacing = float(spacings[0])
    mc["display_spacing_label_km"] = mc["rbf_spacing_km"]
    mc["design_basis_spacing_km"] = current_spacing
    mc["retrained_phase4"] = False
    mc.loc[mc["rbf_spacing_km"].eq(current_spacing), "retrained_phase4"] = True
    mc["validation_type"] = "same_design_holdout_evaluation_not_refit"
    mc["cache_hash"] = cache_hash
    current_hash = hashes.loc[hashes.rbf_spacing_km.eq(current_spacing)].iloc[0].to_dict()
    for key in ["basis_hash", "rbf_centers_sha256", "design_matrix_sha256"]:
        mc[key] = current_hash[key]
    mc["status"] = "partial_same_design_comparison"
    mc.to_csv(mc_path, index=False)
    # Explicitly record that strict leave-block refitting has not yet been run.
    folds = []
    for model_id in ["M0", "M1", "M2"]:
        for fold_id in range(5):
            folds.append({"model_id": model_id, "fold_id": fold_id,
                          "training_blocks": None, "validation_blocks": None,
                          "refit_status": "not_complete_full_training_block_refit_not_run",
                          "validation_rmse": np.nan, "validation_log_likelihood": np.nan})
    pd.DataFrame(folds).to_csv(output / "spatial_block_refit_validation.csv", index=False)
    selected = {
        "selected_model": "M1",
        "selection_status": "provisional_on_current_5km_design",
        "selected_rbf_spacing_km": current_spacing,
        "rbf_spacing_selection_status": "not_evaluated",
        "M0": "sensitivity_baseline",
        "M1": "provisional_selected_same_design",
        "M2": "overparameterized_not_selected",
        "reason": [
            "M1 and M2 predictions are effectively equivalent on the current 5 km design",
            "M1 has fewer parameters and lower AIC/BIC-like metrics",
            "Cu and lag_u spatial fields are not identifiable",
            "Strict RBF spacing comparison and leave-block refit validation are not complete",
        ],
    }
    write_json(selected, output / "model_selection.json")
    phase_status_path = output / "phase_status.json"
    status = json.loads(phase_status_path.read_text(encoding="utf-8")) if phase_status_path.exists() else {}
    status.update({
        "model_compare": "partial_same_design_comparison",
        "model_structure_comparison_on_current_5km_design": "complete",
        "rbf_spacing_comparison": "not_complete",
        "rbf_10km_retrained": False,
        "rbf_15km_retrained": False,
        "current_provisional_model": "M1",
    })
    write_json(status, phase_status_path)
    manifest_path = output / "phase_model_compare_run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "partial_same_design_comparison"
        manifest["rbf_spacing_comparison"] = "not_complete"
        write_json(manifest, manifest_path)
    return {"model_selection": selected, "basis_hashes": hashes.to_dict(orient="records")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
