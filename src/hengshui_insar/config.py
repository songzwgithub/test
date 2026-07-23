"""Strict release configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import constants


@dataclass(frozen=True)
class ReleaseConfig:
    reference_frame: str
    release_id: str
    rbf_dimension: int
    ske_min: float
    ske_max: float
    lag_u_days: float
    lag_c_days: float
    lambda_value: float
    output_release_path: str


def load_config(path: Path) -> ReleaseConfig:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = data["model"]
    storage = data["storage"]
    release = data["release"]
    cfg = ReleaseConfig(
        reference_frame=data["reference_frame"]["id"],
        release_id=release["id"],
        rbf_dimension=int(model["rbf_dimension"]),
        ske_min=float(model["ske_min"]),
        ske_max=float(model["ske_max"]),
        lag_u_days=float(model["lag_u_days"]),
        lag_c_days=float(model["lag_c_days"]),
        lambda_value=float(model["lambda"]),
        output_release_path=release["output_path"],
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
    }
    mismatches = {
        key: {"actual": actual[key], "expected": expected[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"config does not match frozen L01028 release constants: {mismatches}")
    return cfg
