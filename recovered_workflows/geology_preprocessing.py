"""Audited geology vector-to-raster preprocessing for formal inversion inputs."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CONTINUOUS_BANDS = (
    "clay_group_1_m",
    "clay_group_2_m",
    "clay_group_3_m",
    "clay_group_4_m",
    "clay_total_m",
    "clay_confined_m",
    "quaternary_thickness_m",
)
RAW_BANDS = (*CONTINUOUS_BANDS, "extraction_layer_zone_code")
MODEL_BANDS = (
    "clay_total_z",
    "clay_confined_z",
    "quaternary_thickness_z",
    "extraction_layer_zone_2",
)


def parse_interval_from_bounds(lower, upper) -> float:
    lower = float(lower)
    upper = float(upper)
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError(f"Non-finite interval bounds: {lower}, {upper}")
    if upper <= lower:
        raise ValueError(f"Invalid interval bounds, upper <= lower: {lower}, {upper}")
    return (lower + upper) / 2.0


def parse_interval_string(value) -> float:
    text = str(value).strip()
    match = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*[-–]\s*([-+]?\d+(?:\.\d+)?)\s*", text)
    if not match:
        raise ValueError(f"Expected exactly two numeric bounds, got: {value!r}")
    return parse_interval_from_bounds(float(match.group(1)), float(match.group(2)))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(arr).view("uint8"))
    return digest.hexdigest()


def sidecar_files(path: str | Path) -> list[Path]:
    path = Path(path)
    suffixes = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj"]
    return [path.with_suffix(s) for s in suffixes if path.with_suffix(s).exists()]


def raster_profile_like(ref, count=1, dtype="float32", nodata=np.nan):
    profile = ref.profile.copy()
    profile.update(
        driver="GTiff",
        count=count,
        dtype=dtype,
        nodata=nodata,
        compress="lzw",
        tiled=True,
        blockxsize=min(256, ref.width),
        blockysize=min(256, ref.height),
    )
    return profile


def check_raster_alignment(src, ref) -> None:
    checks = {
        "crs": src.crs == ref.crs,
        "transform": src.transform == ref.transform,
        "width": src.width == ref.width,
        "height": src.height == ref.height,
        "bounds": tuple(src.bounds) == tuple(ref.bounds),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Raster grid is not aligned to reference: {failed}")


def stack_all_valid(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([np.asarray(a, "float32") for a in arrays])
    valid = np.isfinite(stack).all(axis=0)
    out = np.full(stack.shape[1:], np.nan, "float32")
    out[valid] = stack[:, valid].sum(axis=0)
    return out


@dataclass(frozen=True)
class LayerSpec:
    name: str
    file: Path
    kind: str
    lower_field: str | None = None
    upper_field: str | None = None
    interval_field: str | None = None
    code_field: str | None = None
    label_field: str | None = None
    allowed_codes: tuple[int, ...] = ()
    reference_category: int | None = None


def geology_output_root(config: dict, default="outputs/geology_fix_revision") -> Path:
    return Path(config.get("geology", {}).get("revision_output_dir", default))


def resolve_layer_specs(config: dict, root: Path) -> list[LayerSpec]:
    geo = config["geology"]
    preprocessing = geo["preprocessing"]
    specs = []
    for i in range(1, 5):
        name = f"clay_group_{i}"
        item = preprocessing[name]
        specs.append(
            LayerSpec(
                name=name,
                file=(root / item["file"]).resolve() if not Path(item["file"]).is_absolute() else Path(item["file"]),
                kind="bounds_interval_polygon",
                lower_field=item["lower_field"],
                upper_field=item["upper_field"],
                label_field=item.get("label_field"),
            )
        )
    q4 = preprocessing["quaternary_thickness"]
    specs.append(
        LayerSpec(
            name="quaternary_thickness",
            file=(root / q4["file"]).resolve() if not Path(q4["file"]).is_absolute() else Path(q4["file"]),
            kind="string_interval_polygon",
            interval_field=q4["interval_field"],
            label_field=q4.get("label_field"),
        )
    )
    zone = preprocessing["extraction_layer_zone"]
    specs.append(
        LayerSpec(
            name="extraction_layer_zone",
            file=(root / zone["file"]).resolve() if not Path(zone["file"]).is_absolute() else Path(zone["file"]),
            kind="categorical_polygon",
            code_field=zone["code_field"],
            label_field=zone.get("label_field"),
            allowed_codes=tuple(int(x) for x in zone["allowed_codes"]),
            reference_category=int(zone["reference_category"]),
        )
    )
    return specs


def layer_values(frame, spec: LayerSpec) -> np.ndarray:
    if spec.kind == "bounds_interval_polygon":
        return np.asarray([parse_interval_from_bounds(a, b) for a, b in zip(frame[spec.lower_field], frame[spec.upper_field])], "float32")
    if spec.kind == "string_interval_polygon":
        return np.asarray([parse_interval_string(v) for v in frame[spec.interval_field]], "float32")
    if spec.kind == "categorical_polygon":
        values = np.asarray(frame[spec.code_field], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{spec.name}: non-finite categorical code")
        int_values = values.astype(int)
        if not np.all(values == int_values):
            raise ValueError(f"{spec.name}: categorical code must be integer")
        invalid = sorted(set(int_values.tolist()) - set(spec.allowed_codes))
        if invalid:
            raise ValueError(f"{spec.name}: unexpected category codes {invalid}")
        return int_values.astype("float32")
    raise ValueError(f"Unsupported layer kind: {spec.kind}")


def repaired_layer_values(frame, spec: LayerSpec) -> tuple[np.ndarray, list[dict]]:
    repairs: list[dict] = []
    if spec.kind != "bounds_interval_polygon":
        return layer_values(frame, spec), repairs
    values = []
    for index, row in frame.iterrows():
        lower = row[spec.lower_field]
        upper = row[spec.upper_field]
        try:
            value = parse_interval_from_bounds(lower, upper)
            values.append(value)
        except ValueError as exc:
            if not spec.label_field:
                raise
            repaired = parse_interval_string(row[spec.label_field])
            numbers = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*[-–]\s*([-+]?\d+(?:\.\d+)?)\s*", str(row[spec.label_field]).strip())
            repairs.append(
                {
                    "layer": spec.name,
                    "feature_index": int(index),
                    "original_lower": float(lower),
                    "original_upper": float(upper),
                    "label": row[spec.label_field],
                    "repaired_lower": float(numbers.group(1)) if numbers else np.nan,
                    "repaired_upper": float(numbers.group(2)) if numbers else np.nan,
                    "repair_reason": f"{exc}; used explicit label interval",
                    "midpoint": float(repaired),
                }
            )
            values.append(repaired)
    return np.asarray(values, "float32"), repairs


def polygon_overlap_fraction(frame) -> float:
    if len(frame) < 2:
        return 0.0
    try:
        projected = frame.to_crs(frame.estimate_utm_crs() or "EPSG:3857")
    except Exception:
        projected = frame.to_crs("EPSG:3857")
    total_area = float(projected.geometry.area.sum()) or 1.0
    overlap_area = 0.0
    geoms = list(projected.geometry)
    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        for other in geoms[i + 1 :]:
            if other is None or other.is_empty or not geom.intersects(other):
                continue
            overlap_area += float(geom.intersection(other).area)
    return overlap_area / total_area


def raster_stats(array: np.ndarray, name: str, unit: str) -> dict:
    arr = np.asarray(array, "float32")
    valid = np.isfinite(arr)
    row = {"name": name, "unit": unit, "valid_pixels": int(valid.sum()), "nan_pixels": int((~valid).sum())}
    if valid.any():
        data = arr[valid].astype(float)
        row.update(
            {
                "minimum": float(np.min(data)),
                "median": float(np.median(data)),
                "maximum": float(np.max(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "unique_values": int(np.unique(data).size),
                "sha256": sha256_array(np.where(valid, arr, np.nan).astype("float32")),
            }
        )
    else:
        row.update({"minimum": np.nan, "median": np.nan, "maximum": np.nan, "mean": np.nan, "std": np.nan, "unique_values": 0, "sha256": sha256_array(arr)})
    return row


def standardize_continuous(name: str, array: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    arr = np.asarray(array, "float32")
    valid = valid_mask & np.isfinite(arr)
    if int(valid.sum()) == 0:
        raise ValueError(f"{name}: no valid pixels for standardization")
    values = arr[valid].astype(float)
    unique = np.unique(values)
    std = float(np.std(values))
    if unique.size < 3:
        raise ValueError(f"{name}: fewer than 3 unique finite values")
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"{name}: zero or invalid standard deviation")
    mean = float(np.mean(values))
    z = np.full(arr.shape, np.nan, "float32")
    z[valid] = ((values - mean) / std).astype("float32")
    zvals = z[valid].astype(float)
    checks = {
        "mean_z": float(np.mean(zvals)),
        "std_z": float(np.std(zvals)),
        "p99_abs_z": float(np.percentile(np.abs(zvals), 99)),
    }
    if abs(checks["mean_z"]) >= 1e-3 or not (0.95 < checks["std_z"] < 1.05) or checks["p99_abs_z"] >= 10:
        raise ValueError(f"{name}: standardized covariate failed checks {checks}")
    meta = {"source_band": name, "mean": mean, "std": std, "valid_pixels": int(valid.sum()), "unique_values": int(unique.size), **checks}
    return z, meta


def categorical_dummy(name: str, array: np.ndarray, code: int, valid_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    arr = np.asarray(array, "float32")
    valid = valid_mask & np.isfinite(arr)
    values = np.unique(arr[valid].astype(int)) if valid.any() else np.array([], dtype=int)
    dummy = np.full(arr.shape, np.nan, "float32")
    dummy[valid] = (arr[valid].astype(int) == int(code)).astype("float32")
    return dummy, {"source_band": name, "dummy_code": int(code), "valid_pixels": int(valid.sum()), "source_unique_values": values.tolist()}


def write_geotiff(path: str | Path, ref, arrays: list[np.ndarray], names: list[str], dtype="float32", nodata=np.nan) -> None:
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = raster_profile_like(ref, count=len(arrays), dtype=dtype, nodata=nodata)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with rasterio.open(tmp, "w", **profile) as dst:
        for idx, (array, name) in enumerate(zip(arrays, names), 1):
            dst.write(np.asarray(array, dtype=dtype), idx)
            dst.set_band_description(idx, name)
    tmp.replace(path)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
