"""Explicit-aquifer groundwater normalization and non-destructive quality control."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ALLOWED_AQUIFERS = {"unconfined", "confined"}


def _date_columns(columns):
    result = []
    for column in columns:
        parsed = pd.to_datetime(str(column), errors="coerce")
        if pd.notna(parsed) and re.search(r"\d{4}[/.-]\d{1,2}[/.-]\d{1,2}", str(column)):
            result.append(column)
    return result


def load_groundwater_wide(path, field_map, aquifer_labels):
    """Convert the 164-well wide CSV to a strict long-table contract."""
    frame = pd.read_csv(path, encoding="utf-8")
    required_source = list(field_map.values())
    missing = [column for column in required_source if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing groundwater fields: {missing}")
    dates = _date_columns(frame.columns)
    if not dates:
        raise ValueError("No daily date columns found in groundwater CSV")
    metadata = frame[required_source].rename(columns={value: key for key, value in field_map.items()})
    metadata["aquifer_type"] = metadata["aquifer_type"].map(aquifer_labels)
    if metadata["aquifer_type"].isna().any():
        unknown = frame.loc[metadata["aquifer_type"].isna(), field_map["aquifer_type"]].dropna().unique().tolist()
        raise ValueError(f"Unmapped explicit aquifer labels: {unknown}; depth-based inference is forbidden")
    if not set(metadata["aquifer_type"].unique()).issubset(ALLOWED_AQUIFERS):
        raise ValueError("aquifer_type must be unconfined or confined")
    wide = pd.concat([metadata, frame[dates]], axis=1)
    long = wide.melt(
        id_vars=list(metadata.columns),
        value_vars=dates,
        var_name="date",
        value_name="water_depth_m",
    )
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    for column in ["lon", "lat", "well_depth_m", "ground_elevation_m", "water_depth_m"]:
        long[column] = pd.to_numeric(long[column], errors="coerce")
    long["hydraulic_head_m"] = long["ground_elevation_m"] - long["water_depth_m"]
    if long.duplicated(["well_id","date"]).any(): raise ValueError("Duplicate well_id-date observations")
    if not long["lon"].between(-180,180).all() or not long["lat"].between(-90,90).all(): raise ValueError("Groundwater coordinates outside valid range")
    consistency=long.groupby("well_id")[["lon","lat","well_depth_m","ground_elevation_m"]].nunique(dropna=False)
    if (consistency>1).any(axis=None): raise ValueError("Well metadata is inconsistent across records")
    return long.sort_values(["well_id", "date"]).reset_index(drop=True)


def apply_quality_flags(frame, spike_threshold_m_per_day=5.0, plateau_min_days=30, plateau_tolerance_m=0.001,
                        minimum_allowed_water_depth_m=-.5):
    """Flag invalid, spike, plateau, and missing observations without deleting rows."""
    result = frame.copy().sort_values(["well_id", "date"]).reset_index(drop=True)
    result["flag_missing"] = result["water_depth_m"].isna()
    result["flag_negative_depth"] = result["water_depth_m"].lt(minimum_allowed_water_depth_m).fillna(False)
    result["flag_below_well_bottom"] = result["water_depth_m"].gt(result["well_depth_m"]).fillna(False)
    result["flag_spike"] = False
    result["flag_plateau"] = False
    result["sensor_reset_flag"] = False
    result["hampel_outlier_flag"] = False
    result["neighbor_inconsistency_flag"] = False
    for _, indices in result.groupby("well_id").groups.items():
        group = result.loc[indices]
        day_step = group["date"].diff().dt.total_seconds().div(86400).replace(0, np.nan)
        rate = group["water_depth_m"].diff().abs().div(day_step)
        result.loc[indices, "flag_spike"] = rate.gt(float(spike_threshold_m_per_day)).fillna(False).to_numpy()
        signed_rate=group["water_depth_m"].diff().div(day_step)
        result.loc[indices,"sensor_reset_flag"] = signed_rate.abs().gt(2*float(spike_threshold_m_per_day)).fillna(False).to_numpy()
        values = group["water_depth_m"]
        median=values.rolling(15,center=True,min_periods=5).median();mad=(values-median).abs().rolling(15,center=True,min_periods=5).median()
        result.loc[indices,"hampel_outlier_flag"] = ((values-median).abs()>5*1.4826*mad.replace(0,np.nan)).fillna(False).to_numpy()
        change = values.diff().abs().gt(float(plateau_tolerance_m)) | values.isna()
        runs = change.cumsum()
        run_size = runs.groupby(runs).transform("size")
        result.loc[indices, "flag_plateau"] = run_size.ge(int(plateau_min_days)).to_numpy() & values.notna().to_numpy()
    flag_columns = ["flag_missing", "flag_negative_depth", "flag_below_well_bottom", "flag_spike", "flag_plateau",
                    "sensor_reset_flag", "hampel_outlier_flag", "neighbor_inconsistency_flag"]
    result["quality_flag"] = result[flag_columns].apply(
        lambda row: ";".join(column.removeprefix("flag_").removesuffix("_flag") for column, value in row.items() if bool(value)) or "ok",
        axis=1,
    )
    result["is_valid_for_model"] = ~result[["flag_missing", "flag_negative_depth", "flag_below_well_bottom"]].any(axis=1)
    return result


def build_weighted_daily_series(frame,max_gap_days=7,observed_weight=1.,interpolated_weight=.35,
                                suspicious_weight=.10,invalid_weight=0.):
    """Preserve observations and fill only complete internal gaps no longer than max_gap_days."""
    rows=[]
    for well_id,group in frame.groupby("well_id",sort=False):
        group=group.sort_values("date").set_index("date");index=pd.date_range(group.index.min(),group.index.max(),freq="D")
        daily=group.reindex(index);daily["well_id"]=well_id
        for column in ["lon","lat","well_depth_m","ground_elevation_m","aquifer_type"]: daily[column]=daily[column].ffill().bfill()
        daily["is_observed"]=daily["water_depth_m"].notna();missing=daily["water_depth_m"].isna();groups=missing.ne(missing.shift()).cumsum()
        daily["is_interpolated"]=False
        for _,idx in missing[missing].groupby(groups).groups.items():
            if len(idx)<=max_gap_days and idx.min()>index.min() and idx.max()<index.max():
                daily.loc[idx,"water_depth_m"]=daily["water_depth_m"].interpolate("time").loc[idx];daily.loc[idx,"is_interpolated"]=True
        daily["hydraulic_head_m"]=daily["ground_elevation_m"]-daily["water_depth_m"]
        suspicious=daily[[c for c in ("flag_spike","flag_plateau","sensor_reset_flag","hampel_outlier_flag","neighbor_inconsistency_flag") if c in daily]].fillna(False).any(axis=1)
        invalid=daily["water_depth_m"].isna()|daily.get("flag_negative_depth",False)|daily.get("flag_below_well_bottom",False)
        daily["observation_weight"]=np.where(invalid,invalid_weight,np.where(daily.is_interpolated,interpolated_weight,np.where(suspicious,suspicious_weight,observed_weight)))
        daily["is_valid_for_model"]=daily["observation_weight"]>0;daily.index.name="date";rows.append(daily.reset_index())
    return pd.concat(rows,ignore_index=True)


def add_fixed_head_anomaly(frame, baseline_start, baseline_end):
    """Add per-well fixed-baseline head anomaly; never use the full period implicitly."""
    result = frame.copy()
    mask = result["date"].between(pd.Timestamp(baseline_start), pd.Timestamp(baseline_end)) & result["is_valid_for_model"]
    baseline = result.loc[mask].groupby("well_id")["hydraulic_head_m"].agg(head_baseline_m="median", baseline_n="count")
    result = result.merge(baseline, on="well_id", how="left")
    result["head_anomaly_m"] = result["hydraulic_head_m"] - result["head_baseline_m"]
    result["baseline_sufficient"] = result["baseline_n"].fillna(0).ge(30)
    return result


def well_summary(frame):
    summary=frame.groupby(["well_id", "aquifer_type"], as_index=False).agg(
        lon=("lon", "first"),
        lat=("lat", "first"),
        well_depth_m=("well_depth_m", "first"),
        ground_elevation_m=("ground_elevation_m", "first"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_records=("date", "size"),
        valid_fraction=("is_valid_for_model", "mean"),
        daily_coverage_fraction=("is_observed", "mean"),
        baseline_sufficient=("baseline_sufficient", "max"),
        baseline_n=("baseline_n", "max"),
        interpolated_fraction=("is_interpolated","mean"),
        invalid_fraction=("observation_weight",lambda x:float(np.mean(x<=0))),
        suspicious_fraction=("observation_weight",lambda x:float(np.mean((x>0)&(x<1)))),
    )
    gaps=[]
    for well_id,group in frame.groupby("well_id"):
        missing=group["water_depth_m"].isna();run=missing.groupby(missing.ne(missing.shift()).cumsum()).sum().max()
        gaps.append((well_id,int(run or 0)))
    return summary.merge(pd.DataFrame(gaps,columns=["well_id","longest_gap_days"]),on="well_id",how="left")
