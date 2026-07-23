"""Strict release configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import constants


@dataclass(frozen=True)
class ReleaseConfig:
    config_path: Path
    project_root: Path
    reference_frame: str
    release_id: str
    authoritative_cache: Path
    authoritative_cache_sha256: str
    common_mask: Path
    common_mask_sha256: str
    fold_map: Path
    fold_map_sha256: str
    selected_rbf_design: Path
    rbf_transform: Path
    rbf_dimension: int
    ske_min: float
    ske_max: float
    lag_u_days: float
    lag_c_days: float
    lambda_value: float
    optimizer_budgets: dict[str, int]
    convergence: dict[str, float]
    observation_sigma_mm: float
    uncertainty_name: str
    output_release_path: str
    release_root: Path


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_config(path: Path) -> ReleaseConfig:
    path = path.resolve()
    project_root = path.parent.parent
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    inputs = data["inputs"]
    model = data["model"]
    optimizer = data["optimizer"]
    storage = data["storage"]
    release = data["release"]
    cfg = ReleaseConfig(
        config_path=path,
        project_root=project_root,
        reference_frame=data["reference_frame"]["id"],
        release_id=release["id"],
        authoritative_cache=_resolve(project_root, inputs["authoritative_cache"]),
        authoritative_cache_sha256=str(inputs["authoritative_cache_sha256"]),
        common_mask=_resolve(project_root, inputs["common_mask"]),
        common_mask_sha256=str(inputs["common_mask_sha256"]),
        fold_map=_resolve(project_root, inputs["fold_map"]),
        fold_map_sha256=str(inputs["fold_map_sha256"]),
        selected_rbf_design=_resolve(project_root, inputs["selected_rbf_design"]),
        rbf_transform=_resolve(project_root, inputs["rbf_transform"]),
        rbf_dimension=int(model["rbf_dimension"]),
        ske_min=float(model["ske_min"]),
        ske_max=float(model["ske_max"]),
        lag_u_days=float(model["lag_u_days"]),
        lag_c_days=float(model["lag_c_days"]),
        lambda_value=float(model["lambda"]),
        optimizer_budgets={k: int(v) for k, v in optimizer["budgets"].items()},
        convergence={k: float(v) for k, v in optimizer["convergence"].items()},
        observation_sigma_mm=float(model.get("observation_sigma_mm", data.get("phase4", {}).get("observation_sigma_mm", 5.0))),
        uncertainty_name=str(storage["uncertainty_name"]),
        output_release_path=release["output_path"],
        release_root=_resolve(project_root, release["output_path"]),
    )
    if storage["delayed_response_positive_lag_definition"] != "y(t-lag)":
        raise ValueError("positive lag definition must be y(t-lag)")
    expected = {
        "reference_frame": constants.REFERENCE_FRAME_ID,
        "release_id": constants.RELEASE_ID,
        "rbf_dimension": constants.RBF_DIMENSION,
        "ske_min": constants.SKE_MIN,
        "ske_max": constants.SKE_MAX,
        "lag_u_days": constants.LAG_U_DAYS,
        "lag_c_days": constants.LAG_C_DAYS,
        "lambda_value": constants.LAMBDA,
        "output_release_path": str(constants.RELEASE_ROOT.relative_to(constants.ROOT)),
        "authoritative_cache": str(constants.AUTHORITATIVE_CACHE.relative_to(constants.ROOT)),
        "authoritative_cache_sha256": constants.CACHE_SHA256,
        "common_mask": str(constants.COMMON_MASK.relative_to(constants.ROOT)),
        "common_mask_sha256": constants.COMMON_MASK_SHA256,
        "fold_map": str(constants.FOLD_MAP.relative_to(constants.ROOT)),
        "fold_map_sha256": constants.FOLD_MAP_SHA256,
        "observation_sigma_mm": 5.0,
        "uncertainty_name": "95% structural amplitude envelope",
    }
    actual = {
        "reference_frame": cfg.reference_frame,
        "release_id": cfg.release_id,
        "rbf_dimension": cfg.rbf_dimension,
        "ske_min": cfg.ske_min,
        "ske_max": cfg.ske_max,
        "lag_u_days": cfg.lag_u_days,
        "lag_c_days": cfg.lag_c_days,
        "lambda_value": cfg.lambda_value,
        "output_release_path": cfg.output_release_path,
        "authoritative_cache": str(cfg.authoritative_cache.relative_to(project_root)),
        "authoritative_cache_sha256": cfg.authoritative_cache_sha256,
        "common_mask": str(cfg.common_mask.relative_to(project_root)),
        "common_mask_sha256": cfg.common_mask_sha256,
        "fold_map": str(cfg.fold_map.relative_to(project_root)),
        "fold_map_sha256": cfg.fold_map_sha256,
        "observation_sigma_mm": cfg.observation_sigma_mm,
        "uncertainty_name": cfg.uncertainty_name,
    }
    mismatches = {
        key: {"actual": actual[key], "expected": expected[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"config does not match frozen L01028 release constants: {mismatches}")
    return cfg
