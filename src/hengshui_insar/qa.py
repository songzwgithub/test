"""Release QA helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


def spatial_qa(release_root: Path) -> dict[str, Any]:
    qa = release_root / "audit" / "spatial_qa_v2_acceptance.json"
    if qa.exists():
        return read_json(qa)
    return {
        "spatial_qa_v2_status": "failed_missing_qa_evidence",
        "basis_row_norm_status": "missing",
        "false_support_distance_product_removed": None,
        "real_distance_product_status": "missing",
    }
