"""Frozen L01028 release constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE_ID = "L01028_v1"
REFERENCE_FRAME_ID = "L01028_500m_fixed_quality_median_v1"

CANONICAL_INPUTS = ROOT / "outputs" / "canonical_inputs" / "L01028_bounded_memmaps_v1"
RELEASE_ROOT = ROOT / "outputs" / "releases" / RELEASE_ID
CURRENT_RELEASE = ROOT / "outputs" / "CURRENT_RELEASE"

AUTHORITATIVE_CACHE = CANONICAL_INPUTS / "static" / "phase4_harmonic_blocks_L01028_authoritative.h5"
COMMON_MASK = CANONICAL_INPUTS / "static" / "comparison_common_mask.tif"
FOLD_MAP = CANONICAL_INPUTS / "static" / "spatial_validation_blocks.tif"

MANIFEST_SHA256 = "f7f41d15db0a83641dc72414814988626e178c2a4c05b091f73c57ad2c2a0cc1"
CACHE_SHA256 = "3f4f714b5e10fe3dcd5a9e91a29de27e0157858137e76afddb12b2cd0fa6dce8"
COMMON_MASK_SHA256 = "ff761a316e0a89a9121c439967df418f14585ae420f281d43671ebaf4740bd1f"
FOLD_MAP_SHA256 = "d24dc63e65d3a1fa1a0e698620ba6d8e03fcf518a9a5ef0721c59374a1d46e3a"

RBF_DIMENSION = 24
SKE_MIN = 1e-8
SKE_MAX = 0.05
LAG_U_DAYS = 10.0
LAG_C_DAYS = 55.77321162652655
LAMBDA = 30.0
ANNUAL_PERIOD_DAYS = 365.2425

EXPECTED_FOLD_RMSE = {
    1: 4.536152410535863,
    2: 4.144879395846749,
    3: 4.569405044130565,
    4: 4.696569337025768,
}
EXPECTED_FINAL = {
    "rmse": 4.0772853898810295,
    "mae": 3.148299950604808,
    "Ske_min": 2.272730886598373e-05,
    "Ske_p50": 0.0013117336279792533,
    "Ske_max": 0.004240259298484303,
    "Cu_global": 1.1733283951532961e-08,
}
EXPECTED_STORAGE = {
    "coherent_amplitude_m3": 90387409.95126072,
    "local_amplitude_sum_m3": 91937693.49036154,
    "peak_to_trough_m3": 180774819.90252143,
    "delayed_peak_shift_days": 55.77321162652652,
    "delayed_direction": "positive_delay",
}
