"""Reference-consistent predictive validation metrics and observation geometry."""
from __future__ import annotations

import numpy as np


def predictive_metrics(observed,predicted,n_bootstrap=500,random_seed=42):
    observed,predicted=np.asarray(observed,float),np.asarray(predicted,float); valid=np.isfinite(observed)&np.isfinite(predicted)
    y,p=observed[valid],predicted[valid]; e=p-y
    fields=("bias","MAE","RMSE","predictive_R2","slope","rmse_ci95_low","rmse_ci95_high","slope_ci95_low","slope_ci95_high","bias_ci95_low","bias_ci95_high")
    if len(y)<2:return {k:np.nan for k in fields}|{"n":len(y)}
    slope=np.polyfit(y,p,1)[0]; denom=np.sum((y-y.mean())**2)
    rng=np.random.default_rng(random_seed);rmses=[];slopes=[];biases=[]
    for _ in range(n_bootstrap):
        idx=rng.integers(0,len(y),len(y));rmses.append(np.sqrt(np.mean(e[idx]**2)));biases.append(np.mean(e[idx]));slopes.append(np.polyfit(y[idx],p[idx],1)[0] if np.std(y[idx])>0 else np.nan)
    return {"bias":float(e.mean()),"MAE":float(np.mean(abs(e))),"RMSE":float(np.sqrt(np.mean(e**2))),
            "predictive_R2":float(1-np.sum(e**2)/denom) if denom>0 else np.nan,"slope":float(slope),
            "rmse_ci95_low":float(np.quantile(rmses,.025)),"rmse_ci95_high":float(np.quantile(rmses,.975)),
            "slope_ci95_low":float(np.nanquantile(slopes,.025)),"slope_ci95_high":float(np.nanquantile(slopes,.975)),
            "bias_ci95_low":float(np.quantile(biases,.025)),"bias_ci95_high":float(np.quantile(biases,.975)),"n":len(y)}


def require_same_reference_frame(*metadata):
    ids={item.get("reference_frame_id") for item in metadata}
    if None in ids or len(ids)!=1: raise ValueError(f"Reference frame mismatch: {ids}")
    return ids.pop()


def enu_to_los(east,north,up,incidence_deg,heading_deg,positive_toward_satellite=True):
    """Project ENU to LOS; heading is clockwise from geographic north along satellite flight."""
    inc=np.deg2rad(incidence_deg);heading=np.deg2rad(heading_deg)
    # Right-looking unit vector from ground toward satellite.
    los=(-np.sin(inc)*np.cos(heading)*np.asarray(east)
         +np.sin(inc)*np.sin(heading)*np.asarray(north)+np.cos(inc)*np.asarray(up))
    return los if positive_toward_satellite else -los


def validate_leveling_interval(insar_dates,insar_displacement,level_start,level_end,level_change,max_offset_days=45):
    dates=np.asarray(insar_dates,dtype="datetime64[D]");start=np.datetime64(level_start,"D");end=np.datetime64(level_end,"D")
    i=int(np.argmin(abs(dates-start)));j=int(np.argmin(abs(dates-end)))
    if abs((dates[i]-start).astype(int))>max_offset_days or abs((dates[j]-end).astype(int))>max_offset_days:
        raise ValueError("No InSAR epoch within leveling endpoint tolerance")
    return predictive_metrics(level_change,np.asarray(insar_displacement)[...,j]-np.asarray(insar_displacement)[...,i])
