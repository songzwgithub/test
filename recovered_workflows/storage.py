"""Lag-aware storage components with strict sign and compatibility gates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from storage_inversion import lagged_series

def _sum_or_nan(values):
    values=np.asarray(values,float);return float(np.nansum(values)) if np.isfinite(values).any() else np.nan


def storage_components(dates, unconfined_head_m, confined_head_m, long_deformation_m,
                       ske, lag_c_days, pixel_area_m2, specific_yield_scenarios,
                       identifiable_mask=None, compatibility=None, posterior_draws=None):
    dates = pd.DatetimeIndex(pd.to_datetime(dates)); hu, hc, observed = map(np.asarray,
        (unconfined_head_m, confined_head_m, long_deformation_m))
    area = np.asarray(pixel_area_m2, float); ske = np.asarray(ske, float); lag = np.asarray(lag_c_days, float)
    ok = np.ones(len(area), bool) if identifiable_mask is None else np.asarray(identifiable_mask, bool)
    hc_lag = np.vstack([lagged_series(hc[i], lag[i], dates) for i in range(len(area))])
    elastic_deformation = ske[:, None]*hc_lag
    compaction_change = observed-elastic_deformation  # subsidence-negative convention
    compatible = all((compatibility or {}).get(k, True) for k in
                     ("time", "space", "sign", "baseline", "reference_frame", "identifiability"))
    rows = []
    for j, date in enumerate(dates):
        confined = _sum_or_nan(np.where(ok, ske*hc[:, j]*area, np.nan))
        change = _sum_or_nan(np.where(ok, compaction_change[:, j]*area, np.nan))
        row = {"date": date, "confined_elastic_storage_change_m3": confined,
               "compaction_equivalent_storage_change_m3": change,
               "compaction_equivalent_irreversible_storage_loss_m3": max(-change, 0) if np.isfinite(change) else np.nan}
        for name, sy in specific_yield_scenarios.items():
            unconfined = _sum_or_nan(np.where(ok, float(sy)*hu[:, j]*area, np.nan))
            row[f"unconfined_storage_change_{name}_m3"] = unconfined
            row[f"total_storage_change_{name}_m3"] = unconfined+confined+change if compatible and np.all(np.isfinite([unconfined,confined,change])) else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    # Draw-level propagation supplies empirical intervals when available.
    if posterior_draws is not None:
        result.attrs["uncertainty_propagated"] = True
    else:
        result.attrs["uncertainty_propagated"] = False
    return result
