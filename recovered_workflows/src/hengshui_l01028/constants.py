"""Stable paths and hashes for the accepted L01028 products."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REF_ID = "L01028_500m_fixed_quality_median_v1"
REFDIR = ROOT / "outputs" / "reference_frames" / REF_ID
BOUNDED = REFDIR / "bounded_model_redevelopment"
ATTEMPT_V3 = BOUNDED / "attempt_v3_001"
STORAGE_ROOT = BOUNDED / "groundwater_storage_volume"
CACHE = ROOT / "outputs" / "cache" / "phase4_harmonic_blocks_L01028_authoritative.h5"
COMMON_MASK = ROOT / "outputs" / "aquifer_model_revision" / "comparison_common_mask.tif"
FOLD_MAP = ROOT / "outputs" / "aquifer_model_revision" / "spatial_validation_blocks.tif"
SKE = ATTEMPT_V3 / "parameter_products" / "Ske.tif"
MANIFEST = ATTEMPT_V3 / "formal_protocol_bounded_frozen_manifest.json"

CACHE_SHA256 = "3f4f714b5e10fe3dcd5a9e91a29de27e0157858137e76afddb12b2cd0fa6dce8"
COMMON_MASK_SHA256 = "ff761a316e0a89a9121c439967df418f14585ae420f281d43671ebaf4740bd1f"
MANIFEST_SHA256 = "f7f41d15db0a83641dc72414814988626e178c2a4c05b091f73c57ad2c2a0cc1"

ANNUAL_PERIOD_DAYS = 365.2425
HARMONIC_ORIGIN = "2018-01-01"
LAG_C_DAYS = 55.77321162652655
