"""Continuous daily-head versus true-SAR-date annual-component lag diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from temporal_analysis import evaluate_components,extract_annual_component,fit_linear_annual_harmonic


def _harmonic_design(dates,origin,period_days):
    t=(pd.DatetimeIndex(pd.to_datetime(dates))-pd.Timestamp(origin)).days.to_numpy(float)
    return np.column_stack([np.ones(len(t)),t/365.2425,np.sin(2*np.pi*t/period_days),np.cos(2*np.pi*t/period_days)])


def _fit_dict_from_beta(beta,origin,period_days,n_obs):
    return {"intercept":float(beta[0]),"trend":float(beta[1]),"sin_coefficient":float(beta[2]),
            "cos_coefficient":float(beta[3]),"annual_amplitude":float(np.hypot(beta[2],beta[3])),
            "annual_phase":float(np.arctan2(beta[3],beta[2])),"coefficient_covariance":None,
            "rmse":np.nan,"snr":np.nan,"n_obs":int(n_obs),"origin":pd.Timestamp(origin),
            "period_days":period_days,"include_semiannual":False}


def _spectrum(gw_fit,insar_fit,insar_dates,lags,minimum_pairs=24):
    lag_values,corr_values,pair_values=_lag_arrays(gw_fit,insar_fit,insar_dates,lags,minimum_pairs)
    return pd.DataFrame({"lag_days":lag_values,"correlation":corr_values,"n_pairs":pair_values})


def _lag_arrays(gw_fit,insar_fit,insar_dates,lags,minimum_pairs=24):
    insar_dates=pd.DatetimeIndex(pd.to_datetime(insar_dates));deformation=extract_annual_component(insar_dates,insar_fit);corrs=[];pairs=[]
    for lag in lags:
        head=extract_annual_component(insar_dates-pd.to_timedelta(float(lag),unit="D"),gw_fit)
        valid=np.isfinite(head)&np.isfinite(deformation);n=int(valid.sum())
        corrs.append(np.corrcoef(head[valid],deformation[valid])[0,1] if n>=minimum_pairs else np.nan);pairs.append(n)
    return np.asarray(lags,float),np.asarray(corrs,float),np.asarray(pairs,int)


def _best_lag(gw_fit,insar_fit,insar_dates,lags,minimum_pairs=24):
    lag_values,corr_values,pair_values=_lag_arrays(gw_fit,insar_fit,insar_dates,lags,minimum_pairs)
    valid=np.isfinite(corr_values)&(corr_values>0)
    if not valid.any():return np.nan,np.nan,0
    local=np.where(valid)[0][np.nanargmax(corr_values[valid])]
    return float(lag_values[local]),float(corr_values[local]),int(pair_values[local])


def _annual_matrix(fit,insar_dates,lags,period_days):
    t=(pd.DatetimeIndex(pd.to_datetime(insar_dates))-fit["origin"]).days.to_numpy(float)
    shifted=t[None,:]-np.asarray(lags,float)[:,None]
    angle=2*np.pi*shifted/period_days
    return fit["sin_coefficient"]*np.sin(angle)+fit["cos_coefficient"]*np.cos(angle)


def _annual_series(fit,dates,period_days):
    t=(pd.DatetimeIndex(pd.to_datetime(dates))-fit["origin"]).days.to_numpy(float)
    angle=2*np.pi*t/period_days
    return fit["sin_coefficient"]*np.sin(angle)+fit["cos_coefficient"]*np.cos(angle)


def _best_from_matrix(head_matrix,deformation,lags,minimum_pairs=24):
    deformation=np.asarray(deformation,float);valid=np.isfinite(deformation)
    n=int(valid.sum())
    if n<minimum_pairs:return np.nan,np.nan,n
    H=np.asarray(head_matrix,float)[:,valid];d=deformation[valid]
    Hc=H-H.mean(axis=1,keepdims=True);dc=d-d.mean()
    denom=np.sqrt(np.sum(Hc*Hc,axis=1)*np.sum(dc*dc))
    corr=np.divide(Hc@dc,denom,out=np.full(H.shape[0],np.nan),where=denom>0)
    ok=np.isfinite(corr)&(corr>0)
    if not ok.any():return np.nan,np.nan,n
    idx=np.where(ok)[0][np.nanargmax(corr[ok])]
    return float(np.asarray(lags,float)[idx]),float(corr[idx]),n


def _spectrum_from_matrix(head_matrix,deformation,lags,minimum_pairs=24):
    rows=[]
    for lag,row in zip(lags,head_matrix):
        _,corr,n=_best_from_matrix(row[None,:],deformation,[lag],minimum_pairs)
        rows.append({"lag_days":float(lag),"correlation":corr,"n_pairs":n})
    return pd.DataFrame(rows)


def select_lag(spectrum):
    valid=spectrum[(spectrum.correlation>0)&spectrum.correlation.notna()]
    if valid.empty:return np.nan,np.nan
    row=valid.loc[valid.correlation.idxmax()];return float(row.lag_days),float(row.correlation)


def _phase_lag_days(gw_fit,insar_fit,period_days,minimum_days,maximum_days):
    if not (np.isfinite(gw_fit["annual_phase"]) and np.isfinite(insar_fit["annual_phase"])):
        return np.nan
    delta=(gw_fit["annual_phase"]-insar_fit["annual_phase"]+np.pi)%(2*np.pi)-np.pi
    lag=(delta*period_days/(2*np.pi))%period_days
    if lag<minimum_days:
        lag+=period_days
    if lag>maximum_days:
        lag-=period_days
    return float(lag) if minimum_days<=lag<=maximum_days else np.nan


def _signed_lag_days(lag,period_days):
    if not np.isfinite(lag):return np.nan
    return float((lag+period_days/2)%period_days-period_days/2)


def _phase_lag_signed_from_coefficients(gw_sin,gw_cos,insar_sin,insar_cos,period_days):
    if not np.all(np.isfinite([gw_sin,gw_cos,insar_sin,insar_cos])):
        return np.nan
    gw_phase=np.arctan2(gw_cos,gw_sin);insar_phase=np.arctan2(insar_cos,insar_sin)
    delta=(gw_phase-insar_phase+np.pi)%(2*np.pi)-np.pi
    return float(delta*period_days/(2*np.pi))


def _sample_annual_coefficients(fit,rng,n):
    cov=fit.get("coefficient_covariance")
    mean=np.asarray([fit.get("sin_coefficient",np.nan),fit.get("cos_coefficient",np.nan)],float)
    if cov is None or not np.all(np.isfinite(mean)):
        return np.full((n,2),np.nan)
    cov=np.asarray(cov,float)
    if cov.shape[0]<4:
        return np.full((n,2),np.nan)
    cov2=cov[2:4,2:4]
    try:
        return rng.multivariate_normal(mean,cov2,size=n)
    except np.linalg.LinAlgError:
        return rng.multivariate_normal(mean,cov2+np.eye(2)*1e-12,size=n)


def _circular_shortest_ci(samples,period_days,alpha=.05):
    values=np.asarray(samples,float);values=values[np.isfinite(values)]
    if len(values)==0:return np.nan,np.nan,np.nan
    wrapped=np.sort((values+period_days/2)%period_days)
    n=len(wrapped);k=max(1,int(np.ceil((1-alpha)*n)))
    extended=np.concatenate([wrapped,wrapped+period_days])
    widths=extended[np.arange(n)+k-1]-extended[np.arange(n)]
    start=int(np.nanargmin(widths));low=extended[start];high=extended[start+k-1]
    center=(low+high)/2
    low_signed=_signed_lag_days(low-period_days/2,period_days)
    high_signed=_signed_lag_days(high-period_days/2,period_days)
    center_signed=_signed_lag_days(center-period_days/2,period_days)
    return float(low_signed),float(high_signed),float(center_signed)


def _phase_std_days(fit):
    cov=fit.get("coefficient_covariance")
    sin=float(fit.get("sin_coefficient",np.nan));cos=float(fit.get("cos_coefficient",np.nan))
    if cov is None or not (np.isfinite(sin) and np.isfinite(cos)):return np.nan
    cov=np.asarray(cov,float)
    if cov.shape[0]<4:return np.nan
    r2=sin*sin+cos*cos
    if r2<=0:return np.nan
    grad=np.array([-cos/r2,sin/r2])
    phase_var=float(grad@cov[2:4,2:4]@grad)
    return float(np.sqrt(max(phase_var,0))*fit["period_days"]/(2*np.pi))


def phase_random_surrogate(values,rng):
    x=np.asarray(values,float);valid=np.isfinite(x);filled=np.interp(np.arange(len(x)),np.flatnonzero(valid),x[valid])
    f=np.fft.rfft(filled-filled.mean());phase=rng.uniform(0,2*np.pi,len(f));phase[0]=0
    if len(x)%2==0:phase[-1]=0
    return np.fft.irfft(np.abs(f)*np.exp(1j*(np.angle(f)+phase)),n=len(x))+filled.mean()


def infer_lag(groundwater_dates,groundwater_values,insar_dates,insar_values,groundwater_weights=None,
              harmonic_origin="2018-01-01",annual_period_days=365.2425,minimum_days=0,maximum_days=365,
              coarse_step_days=7,fine_step_days=1,fine_half_width_days=14,minimum_insar_pairs=24,
              bootstrap_replicates=500,surrogate_replicates=500,bootstrap_block_days=90,
              maximum_ci_width_days=90,expected_correlation_sign="positive",
              minimum_annual_snr=1.0,maximum_phase_std_days=45,
              random_seed=42,reference_metadata=None,aquifer_type=None):
    gw_fit=fit_linear_annual_harmonic(groundwater_dates,groundwater_values,groundwater_weights,harmonic_origin,annual_period_days,False,minimum_insar_pairs)
    insar_fit=fit_linear_annual_harmonic(insar_dates,insar_values,None,harmonic_origin,annual_period_days,False,minimum_insar_pairs)
    coarse=np.arange(minimum_days,maximum_days+1e-9,coarse_step_days)
    insar_deformation=_annual_series(insar_fit,insar_dates,annual_period_days)
    coarse_head=_annual_matrix(gw_fit,insar_dates,coarse,annual_period_days)
    coarse_spectrum=_spectrum_from_matrix(coarse_head,insar_deformation,coarse,minimum_insar_pairs)
    coarse_peak,_=select_lag(coarse_spectrum)
    if np.isfinite(coarse_peak):
        fine=np.arange(max(minimum_days,coarse_peak-fine_half_width_days),min(maximum_days,coarse_peak+fine_half_width_days)+1e-9,fine_step_days)
        fine_head=_annual_matrix(gw_fit,insar_dates,fine,annual_period_days)
        fine_spectrum=_spectrum_from_matrix(fine_head,insar_deformation,fine,minimum_insar_pairs);peak,corr=select_lag(fine_spectrum)
    else:fine_spectrum=pd.DataFrame();peak,corr=np.nan,np.nan
    spectrum=pd.concat([coarse_spectrum.assign(search="coarse"),fine_spectrum.assign(search="fine")],ignore_index=True)
    phase_lag=_phase_lag_days(gw_fit,insar_fit,annual_period_days,minimum_days,maximum_days)
    phase_lag_signed=_signed_lag_days(phase_lag,annual_period_days)
    rng=np.random.default_rng(random_seed);gw_values=np.asarray(groundwater_values,float);gw_dates=pd.DatetimeIndex(pd.to_datetime(groundwater_dates))
    n_boot=int(bootstrap_replicates)
    gw_coeff=_sample_annual_coefficients(gw_fit,rng,n_boot)
    insar_coeff=_sample_annual_coefficients(insar_fit,rng,n_boot)
    finite=np.asarray([_phase_lag_signed_from_coefficients(g[0],g[1],u[0],u[1],annual_period_days) for g,u in zip(gw_coeff,insar_coeff)],float)
    low,high,center=_circular_shortest_ci(finite,annual_period_days)
    null=[]
    insar_values_array=np.asarray(insar_values,float);insar_valid=np.isfinite(insar_values_array)
    Xi=_harmonic_design(pd.DatetimeIndex(pd.to_datetime(insar_dates))[insar_valid],harmonic_origin,annual_period_days)
    insar_solver=np.linalg.pinv(Xi.T@Xi)@Xi.T
    for _ in range(int(surrogate_replicates)):
        surrogate=phase_random_surrogate(insar_values_array,rng)
        surrogate_fit=_fit_dict_from_beta(insar_solver@surrogate[insar_valid],harmonic_origin,annual_period_days,insar_valid.sum())
        lag_grid=fine if np.isfinite(coarse_peak) else coarse
        head_matrix=fine_head if np.isfinite(coarse_peak) else coarse_head
        _,c,_=_best_from_matrix(head_matrix,_annual_series(surrogate_fit,insar_dates,annual_period_days),lag_grid,minimum_insar_pairs);null.append(c if np.isfinite(c) else 0)
    p=float((1+np.sum(np.asarray(null)>=corr))/(len(null)+1)) if np.isfinite(corr) else np.nan
    row=spectrum[spectrum.lag_days==peak];pairs=int(row.n_pairs.max()) if len(row) else 0
    ci_width=float(((high-low)+annual_period_days)%annual_period_days) if np.isfinite(low) and np.isfinite(high) else np.nan
    gw_phase_std_days=_phase_std_days(gw_fit);insar_phase_std_days=_phase_std_days(insar_fit)
    boundary=bool(np.isfinite(phase_lag_signed) and abs(phase_lag_signed)>=annual_period_days/2-1);alias=bool(np.isfinite(phase_lag_signed) and abs(phase_lag_signed)>=annual_period_days/2-14)
    sign_ok=(corr>0) if expected_correlation_sign=="positive" else (corr<0 if expected_correlation_sign=="negative" else np.isfinite(corr))
    snr_ok=gw_fit.get("snr",np.nan)>=minimum_annual_snr and insar_fit.get("snr",np.nan)>=minimum_annual_snr
    phase_ok=np.nanmax([gw_phase_std_days,insar_phase_std_days])<=maximum_phase_std_days
    summary={"aquifer_type":aquifer_type,**(reference_metadata or {}),"peak_lag_days":phase_lag_signed,"phase_lag_unsigned_days":phase_lag,
             "tlcc_peak_lag_days":peak,"peak_correlation":corr,"n_pairs":pairs,
             "lag_ci_low":float(low),"lag_ci_high":float(high),"surrogate_p_value":p,"boundary_peak":boundary,"annual_alias":alias,
             "lag_method":"dual_endpoint_annual_phase_difference_circular","lag_ci_width_days":ci_width,"lag_circular_mean_days":center,
             "groundwater_annual_snr":gw_fit.get("snr"),"insar_annual_snr":insar_fit.get("snr"),
             "groundwater_phase_std_days":gw_phase_std_days,"insar_phase_std_days":insar_phase_std_days,
             "groundwater_phase":gw_fit["annual_phase"],"insar_phase":insar_fit["annual_phase"],
             "lag_reliable":bool(pairs>=minimum_insar_pairs and sign_ok and p<=.05 and snr_ok and phase_ok and np.isfinite(ci_width) and ci_width<=maximum_ci_width_days and not boundary and not alias)}
    return spectrum,summary
