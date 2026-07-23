"""Weighted harmonic analysis on irregular dates with one fixed project origin."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _design(dates,origin,period_days,semiannual=False):
    t=(pd.DatetimeIndex(pd.to_datetime(dates))-pd.Timestamp(origin)).days.to_numpy(float)
    columns=[np.ones(len(t)),t/365.2425,np.sin(2*np.pi*t/period_days),np.cos(2*np.pi*t/period_days)]
    if semiannual:columns.extend([np.sin(4*np.pi*t/period_days),np.cos(4*np.pi*t/period_days)])
    return np.column_stack(columns)


def fit_linear_annual_harmonic(dates,values,weights=None,harmonic_origin="2018-01-01",
                               annual_period_days=365.2425,include_semiannual=False,min_observations=24):
    dates=pd.DatetimeIndex(pd.to_datetime(dates));y=np.asarray(values,float);w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    valid=np.isfinite(y)&np.isfinite(w)&(w>0)&~dates.isna();n=int(valid.sum())
    empty={"intercept":np.nan,"trend":np.nan,"sin_coefficient":np.nan,"cos_coefficient":np.nan,
           "annual_amplitude":np.nan,"annual_phase":np.nan,"coefficient_covariance":None,"rmse":np.nan,
           "snr":np.nan,"n_obs":n,"origin":pd.Timestamp(harmonic_origin),"period_days":annual_period_days,"include_semiannual":False}
    if n<min_observations:return empty
    def fit(semi):
        X=_design(dates[valid],harmonic_origin,annual_period_days,semi);sw=np.sqrt(w[valid]);Xw=X*sw[:,None];yw=y[valid]*sw
        beta=np.linalg.lstsq(Xw,yw,rcond=None)[0];res=y[valid]-X@beta;rss=np.sum(w[valid]*res**2);dof=max(1,n-len(beta))
        covariance=np.linalg.pinv(Xw.T@Xw)*(rss/dof);bic=n*np.log(max(rss/n,1e-30))+len(beta)*np.log(n)
        return beta,res,covariance,bic
    beta,res,cov,bic=fit(False);used=False
    if include_semiannual:
        b2,r2,c2,bic2=fit(True)
        if bic2+2<bic:beta,res,cov,bic,used=b2,r2,c2,bic2,True
    amplitude=float(np.hypot(beta[2],beta[3]));rmse=float(np.sqrt(np.average(res**2,weights=w[valid])))
    return {"intercept":float(beta[0]),"trend":float(beta[1]),"sin_coefficient":float(beta[2]),
            "cos_coefficient":float(beta[3]),"annual_amplitude":amplitude,"annual_phase":float(np.arctan2(beta[3],beta[2])),
            "coefficient_covariance":cov,"rmse":rmse,"snr":amplitude/rmse if rmse>0 else np.inf,"n_obs":n,
            "origin":pd.Timestamp(harmonic_origin),"period_days":annual_period_days,"include_semiannual":used,
            "semiannual_sin":float(beta[4]) if used else 0.,"semiannual_cos":float(beta[5]) if used else 0.}


def evaluate_components(dates,fit):
    X=_design(dates,fit["origin"],fit["period_days"],fit.get("include_semiannual",False));linear=X[:,:2]@np.array([fit["intercept"],fit["trend"]])
    annual=X[:,2:4]@np.array([fit["sin_coefficient"],fit["cos_coefficient"]])
    if fit.get("include_semiannual",False):annual+=X[:,4:6]@np.array([fit["semiannual_sin"],fit["semiannual_cos"]])
    return linear,annual


def fit_continuous_harmonic(dates,values,weights=None,**kwargs):
    result=fit_linear_annual_harmonic(dates,values,weights,**kwargs)
    if "trend_per_year" not in result:result["trend_per_year"]=result.get("trend")
    if "annual_phase_rad" not in result:result["annual_phase_rad"]=result.get("annual_phase")
    if "annual_snr" not in result:result["annual_snr"]=result.get("snr")
    if "n_observations" not in result:result["n_observations"]=result.get("n_obs")
    phase=result.get("annual_phase")
    result["annual_peak_day"]=float((phase*result["period_days"]/(2*np.pi))%result["period_days"]) if np.isfinite(phase) else np.nan
    return result


def extract_linear_component(dates,fit):
    return evaluate_components(dates,fit)[0]


def extract_annual_component(dates,fit):
    return evaluate_components(dates,fit)[1]


def extract_residual_component(dates,values,fit):
    linear,annual=evaluate_components(dates,fit)
    return np.asarray(values,float)-linear-annual


evaluate_linear_component=extract_linear_component
evaluate_annual_component=extract_annual_component
evaluate_residual=extract_residual_component


def evaluate_semiannual_component(dates,fit):
    if not fit.get("include_semiannual",False):return np.zeros(len(pd.DatetimeIndex(pd.to_datetime(dates))))
    X=_design(dates,fit["origin"],fit["period_days"],True)
    return X[:,4:6]@np.array([fit["semiannual_sin"],fit["semiannual_cos"]])


def evaluate_at_dates(dates,fit):
    linear,annual=evaluate_components(dates,fit)
    return linear+annual


def remove_linear_trend(dates,values,fit=None,**kwargs):
    fit=fit or fit_linear_annual_harmonic(dates,values,**kwargs);return np.asarray(values,float)-extract_linear_component(dates,fit)
harmonic_decomposition=fit_linear_annual_harmonic


def decompose_groups(frame,group_col,date_col,value_col,weight_col="observation_weight",**kwargs):
    rows=[]
    for key,group in frame.groupby(group_col,sort=True):
        fit=fit_linear_annual_harmonic(group[date_col],group[value_col],group[weight_col] if weight_col in group else None,**kwargs)
        fit["coefficient_covariance"]=np.asarray(fit["coefficient_covariance"]).tolist() if fit["coefficient_covariance"] is not None else None
        rows.append({group_col:key,**fit})
    return pd.DataFrame(rows)


def rotate_harmonic_coefficients(sin_coefficient,cos_coefficient,lag_days,period_days=365.2425):
    """Coefficients of h(t-lag) in the same sin/cos basis."""
    angle=2*np.pi*np.asarray(lag_days,float)/period_days;c=np.cos(angle);s=np.sin(angle)
    return sin_coefficient*c+cos_coefficient*s,cos_coefficient*c-sin_coefficient*s


def annual_component_matrix(dates,values,weights=None,**kwargs):
    values=np.asarray(values,float);output=np.full_like(values,np.nan)
    for i,row in enumerate(values):
        fit=fit_linear_annual_harmonic(dates,row,None if weights is None else weights[i],**kwargs)
        if np.isfinite(fit["annual_amplitude"]):output[i]=extract_annual_component(dates,fit)
    return output
