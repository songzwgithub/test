"""Coordinate and raster-window helpers without interpolation."""
from __future__ import annotations

import numpy as np


def radius_window(src, lon, lat, radius_m):
    """Return a clipped raster window around a geographic point."""
    from rasterio.windows import Window
    from rasterio.warp import transform as transform_coordinates

    x, y = lon, lat
    if src.crs is None:
        raise ValueError("Raster CRS is required")
    if not src.crs.is_geographic:
        x_values, y_values = transform_coordinates("EPSG:4326", src.crs, [lon], [lat])
        x, y = x_values[0], y_values[0]
    row, col = src.index(x, y)
    if row < 0 or row >= src.height or col < 0 or col >= src.width:
        raise ValueError("Point lies outside raster bounds")
    if src.crs.is_geographic:
        half_cols = max(1, int(np.ceil(radius_m / (111320 * max(np.cos(np.deg2rad(lat)), 0.2) * abs(src.transform.a)))))
        half_rows = max(1, int(np.ceil(radius_m / (110540 * abs(src.transform.e)))))
    else:
        half_cols = max(1, int(np.ceil(radius_m / abs(src.transform.a))))
        half_rows = max(1, int(np.ceil(radius_m / abs(src.transform.e))))
    col0, row0 = max(0, col - half_cols), max(0, row - half_rows)
    col1, row1 = min(src.width, col + half_cols + 1), min(src.height, row + half_rows + 1)
    return Window(col0, row0, col1 - col0, row1 - row0)


def circular_mask(src, window, lon, lat, radius_m):
    """True pixel-center mask for a geodesic/projected circular buffer."""
    from rasterio.transform import xy
    from rasterio.warp import transform as transform_coordinates
    rows = np.arange(int(window.row_off), int(window.row_off+window.height))
    cols = np.arange(int(window.col_off), int(window.col_off+window.width))
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xs, ys = xy(src.transform, rr, cc, offset="center")
    xs, ys = np.asarray(xs).reshape(rr.shape), np.asarray(ys).reshape(rr.shape)
    if src.crs.is_geographic:
        # Local equirectangular metric is accurate at sub-km reference radii.
        dx = (xs-lon)*111320*np.cos(np.deg2rad(lat)); dy = (ys-lat)*110540
    else:
        px, py = transform_coordinates("EPSG:4326", src.crs, [lon], [lat])
        dx, dy = xs-px[0], ys-py[0]
    return dx*dx+dy*dy <= float(radius_m)**2


def iter_windows(height, width, block_rows=128, block_cols=128):
    """Yield deterministic non-overlapping windows covering a raster."""
    from rasterio.windows import Window

    for row in range(0, height, block_rows):
        for col in range(0, width, block_cols):
            yield Window(col, row, min(block_cols, width - col), min(block_rows, height - row))
