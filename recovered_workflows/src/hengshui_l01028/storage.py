"""Storage-specific numerical helpers."""

from __future__ import annotations

import numpy as np
from pyproj import Geod
import rasterio


def geodesic_row_areas(transform: rasterio.Affine, width: int, height: int) -> np.ndarray:
    if abs(transform.b) > 1e-15 or abs(transform.d) > 1e-15:
        raise ValueError("rotated rasters are not supported by row-wise area method")
    geod = Geod(ellps="WGS84")
    areas = np.empty(height, dtype=np.float64)
    x0 = transform.c
    x1 = transform.c + transform.a
    for row in range(height):
        y0 = transform.f + row * transform.e
        y1 = transform.f + (row + 1) * transform.e
        area, _ = geod.polygon_area_perimeter([x0, x1, x1, x0], [y0, y0, y1, y1])
        areas[row] = abs(area)
    return areas
