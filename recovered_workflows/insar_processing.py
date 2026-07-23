"""Lazy full-pixel GeoTIFF cube, reference-frame, and LOS geometry operations."""
from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
import os
import hashlib

import numpy as np
import pandas as pd

from spatial_utils import circular_mask, iter_windows, radius_window


LOS_SIGN_CONVENTION = "positive_toward_satellite"
L01028_REFERENCE_FRAME_ID = "L01028_500m_fixed_quality_median_v1"


def _date_from_name(path):
    name = Path(path).name
    match = re.search(r"geo_\d{8}_(\d{8})", name, re.I) or re.search(r"(\d{8})(?=\.tif|\.tiff)", name, re.I)
    return pd.to_datetime(match.group(1), errors="coerce") if match else pd.NaT


def displacement_scale_to_mm(unit):
    scale = {"m": 1000.0, "cm": 10.0, "mm": 1.0}.get(str(unit).lower())
    if scale is None:
        raise ValueError(f"Unsupported displacement unit: {unit}")
    return scale


def sha256_array(values, dtype=None) -> str:
    """Stable SHA256 for numeric arrays used in reference manifests."""
    array = np.asarray(values if dtype is None else np.asarray(values, dtype=dtype))
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view("uint8")).hexdigest()


def normalize_reference_series(reference_los_mm):
    """Return R(t)=R0(t)-R0(t0) for a fixed-pixel LOS reference series."""
    series = np.asarray(reference_los_mm, dtype=float)
    if series.ndim != 1 or series.size == 0:
        raise ValueError("reference_los_mm must be a non-empty 1-D series")
    if not np.isfinite(series).all():
        raise ValueError("reference_los_mm contains NaN or inf")
    return series - float(series[0])


def assert_reference_dates_match(cube_dates, reference_dates):
    cube_index = pd.DatetimeIndex(pd.to_datetime(cube_dates))
    reference_index = pd.DatetimeIndex(pd.to_datetime(reference_dates))
    if len(cube_index) != len(reference_index) or not np.all(cube_index == reference_index):
        raise ValueError("Reference dates do not exactly match InSAR epoch dates")
    return True


def reference_application_metadata(reference_manifest, reference_timeseries_hash):
    """Metadata carried by all on-the-fly L01028 response products."""
    return {
        "reference_applied": True,
        "reference_application_count": 1,
        "reference_frame_id": reference_manifest["reference_frame_id"],
        "reference_center": reference_manifest["reference_center"],
        "reference_radius_m": reference_manifest["radius_m"],
        "reference_method": reference_manifest["selection_method"],
        "reference_mode": reference_manifest["reference_mode"],
        "reference_geometry": reference_manifest["reference_geometry"],
        "reference_timeseries_hash": reference_timeseries_hash,
        "reference_applied_before_vertical_projection": True,
    }


def ensure_reference_not_already_applied(metadata, reference_frame_id):
    """Guard against subtracting the same on-the-fly reference twice."""
    if not metadata:
        return True
    applied = bool(metadata.get("reference_applied", False))
    count = int(metadata.get("reference_application_count", 0) or 0)
    same_frame = metadata.get("reference_frame_id") == reference_frame_id
    if applied and same_frame and count >= 1:
        raise RuntimeError(f"Reference frame {reference_frame_id} has already been applied")
    return True


def apply_reference_to_los(values_mm, epoch_index, reference_series_mm, metadata=None, reference_frame_id=None):
    """Apply D_new(x,t)=D_old(x,t)-R(t) in native LOS geometry."""
    if reference_series_mm is None:
        return np.asarray(values_mm, dtype=float)
    frame_id = reference_frame_id or (metadata or {}).get("reference_frame_id")
    if frame_id:
        ensure_reference_not_already_applied(metadata, frame_id)
    series = np.asarray(reference_series_mm, dtype=float)
    if epoch_index < 0 or epoch_index >= series.size:
        raise IndexError("epoch_index is outside reference_series_mm")
    return np.asarray(values_mm, dtype=float) - float(series[epoch_index])


@dataclass(frozen=True)
class GeoTiffCube:
    """A lazy `(row, column, time)` cube backed by one GeoTIFF per epoch."""

    epochs: pd.DataFrame
    shape: tuple
    crs: str
    transform: tuple
    unit: str

    @classmethod
    def from_glob(cls, pattern, unit="m"):
        import rasterio

        files = [path for path in sorted(glob.glob(str(pattern))) if "velocity" not in Path(path).name.lower()]
        if not files:
            raise FileNotFoundError(f"No InSAR GeoTIFF epochs match {pattern}")
        rows = [{"date": _date_from_name(path), "source_file": str(Path(path).resolve())} for path in files]
        raw=pd.DataFrame(rows)
        if raw["date"].isna().any():raise ValueError(f"Unparseable InSAR dates: {raw.loc[raw.date.isna(),'source_file'].tolist()}")
        epochs = raw.sort_values("date").reset_index(drop=True)
        if epochs.empty or epochs["date"].duplicated().any():
            raise ValueError("InSAR dates must be valid and unique")
        with rasterio.open(epochs["source_file"].iloc[0]) as src:
            if src.crs is None:
                raise ValueError("InSAR raster CRS is missing")
            reference = (src.height, src.width, str(src.crs), tuple(src.transform))
        for source in epochs["source_file"].iloc[1:]:
            with rasterio.open(source) as src:
                current = (src.height, src.width, str(src.crs), tuple(src.transform))
                if current != reference:
                    raise ValueError(f"InSAR grid mismatch: {source}")
        displacement_scale_to_mm(unit)
        return cls(epochs, reference[:2], reference[2], reference[3], unit)

    @property
    def cube_shape(self):
        return (len(self.epochs), self.shape[0], self.shape[1])

    def read_window(self, window, referenced=True, reference_series_mm=None, reference_metadata=None):
        """Read one spatial window for all dates as `(time,row,column)` in mm."""
        import rasterio

        scale = displacement_scale_to_mm(self.unit)
        arrays = []
        for index, source in enumerate(self.epochs["source_file"]):
            with rasterio.open(source) as src:
                values = src.read(1, window=window, masked=True).filled(np.nan).astype("float32") * scale
            if referenced and reference_series_mm is not None:
                values = apply_reference_to_los(
                    values,
                    index,
                    reference_series_mm,
                    metadata=reference_metadata,
                    reference_frame_id=(reference_metadata or {}).get("reference_frame_id"),
                )
            arrays.append(values)
        return np.stack(arrays, axis=0)

    def manifest(self, reference_metadata=None):
        return {
            "array_order": "time,row,column",
            "cube_shape": list(self.cube_shape),
            "n_epochs": len(self.epochs),
            "time_start": self.epochs["date"].min(),
            "time_end": self.epochs["date"].max(),
            "crs": self.crs,
            "transform": self.transform,
            "source_unit": self.unit,
            "working_unit": "mm",
            "los_sign_convention": LOS_SIGN_CONVENTION,
            "storage": "lazy_geotiff_backed_cube",
            **(reference_metadata or {}),
        }


def compute_reference_series(cube, lon, lat, radius_m=500, method="median",
                             min_valid_pixels=10, minimum_valid_epoch_fraction=.9):
    """Compute the epoch-wise reference displacement without materializing the cube."""
    import rasterio

    if method != "median":
        raise ValueError("Only robust median reference is permitted")
    scale = displacement_scale_to_mm(cube.unit)
    values, diagnostics = [], []
    for source in cube.epochs["source_file"]:
        with rasterio.open(source) as src:
            window = radius_window(src, lon, lat, radius_m)
            data = src.read(1, window=window, masked=True).filled(np.nan).astype(float) * scale
            mask = circular_mask(src, window, lon, lat, radius_m)
        finite = data[mask & np.isfinite(data)]
        median = float(np.median(finite)) if len(finite) else np.nan
        mad = float(np.median(np.abs(finite-median))) if len(finite) else np.nan
        values.append(median)
        diagnostics.append({"n_valid_reference_pixels": int(len(finite)),
                            "valid_fraction": float(len(finite)/max(1, mask.sum())),
                            "reference_median_mm": median, "reference_mad_mm": mad})
    series = pd.Series(values, index=pd.DatetimeIndex(cube.epochs["date"]), name="reference_los_mm")
    valid_epochs = np.asarray([d["n_valid_reference_pixels"] >= min_valid_pixels for d in diagnostics])
    if valid_epochs.mean() < minimum_valid_epoch_fraction:
        raise ValueError("Reference area fails minimum valid epoch fraction")
    metadata = {
        "reference_frame_id": f"median_{lon:.6f}_{lat:.6f}_{int(radius_m)}m",
        "reference_lon": float(lon),
        "reference_lat": float(lat),
        "reference_radius_m": float(radius_m),
        "reference_method": "median",
        "reference_min_valid_pixels": int(min_valid_pixels),
        "reference_minimum_valid_epoch_fraction": float(minimum_valid_epoch_fraction),
        "reference_valid_epoch_fraction": float(valid_epochs.mean()),
        "epoch_diagnostics": diagnostics,
    }
    return series, metadata


def los_to_vertical(los_mm, incidence_deg):
    """Convert LOS to vertical under the explicitly recorded vertical-dominant assumption."""
    incidence = np.asarray(incidence_deg, dtype=float)
    cosine = np.cos(np.deg2rad(incidence))
    cosine[np.abs(cosine) < 1e-6] = np.nan
    return np.asarray(los_mm, dtype=float) / cosine


def sample_points(cube, coordinates, incidence_grid=None, reference_series_mm=None,
                  method="buffer_median", radius_m=400, reference_metadata=None):
    """Sample every epoch at lon/lat points without materializing the full cube."""
    import rasterio
    coordinates = [tuple(map(float, point)) for point in coordinates]
    rows = []
    incidence = np.load(incidence_grid, mmap_mode="r") if incidence_grid is not None else None
    for epoch_index, epoch in cube.epochs.iterrows():
        with rasterio.open(epoch["source_file"]) as src:
            print(f"sample_points_epoch {epoch_index+1}/{len(cube.epochs)} {pd.Timestamp(epoch['date']).date()}", flush=True)
            source_points = coordinates
            if str(src.crs) != "EPSG:4326":
                from rasterio.warp import transform
                xs, ys = transform("EPSG:4326", src.crs, *zip(*coordinates))
                source_points = list(zip(xs, ys))
            rc = [src.index(x, y) for x, y in source_points]
            buffer_stats=[]
            if method == "nearest":
                los = np.asarray([v[0] for v in src.sample(source_points)], float)*displacement_scale_to_mm(cube.unit)
                buffer_stats=[(1,np.nan,1.,float(incidence[row,col]) if incidence is not None else np.nan) for row,col in rc]
            elif method == "buffer_median":
                windows = [radius_window(src, lon, lat, radius_m) for lon, lat in coordinates]
                row0 = min(int(window.row_off) for window in windows)
                col0 = min(int(window.col_off) for window in windows)
                row1 = max(int(window.row_off + window.height) for window in windows)
                col1 = max(int(window.col_off + window.width) for window in windows)
                from rasterio.windows import Window
                union_window = Window(col0, row0, col1-col0, row1-row0)
                full_data = src.read(1, window=union_window, masked=True).filled(np.nan).astype(float)
                los = []
                for (lon, lat), window in zip(coordinates, windows):
                    r0,c0=int(window.row_off),int(window.col_off);h,w=int(window.height),int(window.width)
                    data = full_data[r0-row0:r0-row0+h,c0-col0:c0-col0+w]
                    mask = circular_mask(src, window, lon, lat, radius_m)
                    values=data[mask];finite=values[np.isfinite(values)];median=np.nanmedian(finite);los.append(median)
                    inc_values=np.asarray(incidence[r0:r0+h,c0:c0+w])[mask] if incidence is not None else np.array([np.nan])
                    buffer_stats.append((len(finite),float(np.nanmedian(abs(finite-median))) if len(finite) else np.nan,
                                         float(len(finite)/max(1,mask.sum())),float(np.nanmedian(inc_values))))
                los = np.asarray(los)*displacement_scale_to_mm(cube.unit)
            else:
                raise ValueError("Point sampling method must be nearest or buffer_median")
        if reference_series_mm is not None:
            los = apply_reference_to_los(
                los,
                epoch_index,
                reference_series_mm,
                metadata=reference_metadata,
                reference_frame_id=(reference_metadata or {}).get("reference_frame_id"),
            )
        for point_index, (((row, col), value),stats) in enumerate(zip(zip(rc, los),buffer_stats)):
            inc = stats[3]
            vertical = float(los_to_vertical(np.array([value]), np.array([inc]))[0]) if np.isfinite(inc) else np.nan
            rows.append({"point_index": point_index, "date": epoch["date"], "los_mm": value,
                         "incidence_deg": inc, "vertical_mm": vertical,"n_valid_buffer_pixels":stats[0],
                         "buffer_mad_source_unit":stats[1],"buffer_valid_fraction":stats[2]})
    return pd.DataFrame(rows)


def write_vertical_h5(cube, incidence_grid, output_path, reference_series_mm=None, reference_metadata=None, block_rows=128, block_cols=128):
    """Stream the complete vertical-dominant cube to chunked HDF5.

    This is opt-in because the full product is tens of gigabytes. No pixels are
    sampled or discarded when enabled.
    """
    import h5py

    incidence = np.load(incidence_grid, mmap_mode="r")
    if tuple(incidence.shape) != tuple(cube.shape):
        raise ValueError("Incidence grid does not match the InSAR grid")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary=output.with_suffix(output.suffix+f".{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w") as h5:
        dataset = h5.create_dataset(
            "vertical_displacement_mm",
            shape=(len(cube.epochs), *cube.shape),
            dtype="float32",
            chunks=(1, min(block_rows, cube.shape[0]), min(block_cols, cube.shape[1])),
            compression="lzf",
            fillvalue=np.nan,
        )
        scale = displacement_scale_to_mm(cube.unit)
        import rasterio
        for epoch_index, source in enumerate(cube.epochs["source_file"]):
            print(f"write_vertical_h5_epoch {epoch_index+1}/{len(cube.epochs)}", flush=True)
            with rasterio.open(source) as src:
                for window in iter_windows(*cube.shape, block_rows, block_cols):
                    row0, col0 = int(window.row_off), int(window.col_off)
                    height, width = int(window.height), int(window.width)
                    los = src.read(1, window=window, masked=True).filled(np.nan).astype("float32") * scale
                    if reference_series_mm is not None:
                        los = apply_reference_to_los(
                            los,
                            epoch_index,
                            reference_series_mm,
                            metadata=reference_metadata,
                            reference_frame_id=(reference_metadata or {}).get("reference_frame_id"),
                        )
                    inc = np.asarray(incidence[row0:row0 + height, col0:col0 + width])
                    dataset[epoch_index, row0:row0 + height, col0:col0 + width] = los_to_vertical(los, inc).astype("float32")
        h5.create_dataset("date", data=np.asarray([d.isoformat().encode() for d in cube.epochs["date"]]))
        h5.attrs["UNIT"] = "mm"
        h5.attrs["array_order"] = "time,row,column"
        h5.attrs["crs"] = cube.crs
        h5.attrs["transform"] = tuple(cube.transform)
        h5.attrs["vertical_dominant_assumption"] = True
        h5.attrs["los_sign_convention"] = LOS_SIGN_CONVENTION
        for key,value in (reference_metadata or {}).items():
            if value is not None and not isinstance(value,(dict,list)):h5.attrs[key]=value
    os.replace(temporary,output)
    return output
