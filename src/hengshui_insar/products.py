"""Release product audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from .hashing import sha256_file


FORMAL_PRODUCTS = {
    "Ske": "Ske.tif",
    "prediction_real": "predicted_annual_real_mm.tif",
    "prediction_imag": "predicted_annual_imag_mm.tif",
    "residual_real": "residual_annual_real_mm.tif",
    "residual_imag": "residual_annual_imag_mm.tif",
    "residual_amplitude": "residual_amplitude_mm.tif",
    "basis_row_norm": "rbf_basis_row_norm.tif",
    "saturation_mask": "upper_bound_saturation_mask.tif",
}


def raster_summary(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        finite = 0
        vmin = np.inf
        vmax = -np.inf
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)
            mask = np.isfinite(arr)
            finite += int(mask.sum())
            if mask.any():
                vmin = min(vmin, float(np.nanmin(arr[mask])))
                vmax = max(vmax, float(np.nanmax(arr[mask])))
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "shape": [src.height, src.width],
            "nodata": src.nodata,
            "finite_count": finite,
            "min": None if not np.isfinite(vmin) else vmin,
            "max": None if not np.isfinite(vmax) else vmax,
        }


def product_audit(products_dir: Path) -> dict[str, Any]:
    rows = {}
    ok = True
    for name, filename in FORMAL_PRODUCTS.items():
        path = products_dir / filename
        if not path.exists():
            rows[name] = {"path": str(path), "status": "missing"}
            ok = False
        else:
            rows[name] = {"status": "passed", **raster_summary(path)}
    return {"products_status": "passed" if ok else "failed", "products": rows}
