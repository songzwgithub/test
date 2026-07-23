"""Robust seasonal two-aquifer MAP inversion over low-dimensional coefficients."""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import minimize


def lagged_series(values, lag_days, dates):
    dates = np.asarray(dates, dtype="datetime64[ns]"); source = dates.astype("int64").astype(float)
    target = source-float(lag_days)*86400e9
    return np.interp(target, source, np.asarray(values, float), left=np.nan, right=np.nan)


def _softplus(x):
    return np.logaddexp(0, x)


def _sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -40, 40)))


def decode_fields(theta, Z, model_variant):
    p = Z.shape[1]; cursor = 0
    ske = np.exp(np.clip(Z@theta[cursor:cursor+p], -20, 10)); cursor += p
    lag_c = 365*_sigmoid(Z@theta[cursor:cursor+p]); cursor += p
    if model_variant == "two_aquifer":
        cu = _softplus(Z@theta[cursor:cursor+p]); cursor += p
        lag_u = 365*_sigmoid(Z@theta[cursor:cursor+p])
    else:
        cu, lag_u = np.full(len(Z), np.nan), np.full(len(Z), np.nan)
    return ske, lag_c, cu, lag_u


@dataclass
class MAPResult:
    coefficients: np.ndarray
    success: bool
    message: str
    objective: float
    parameter_names: list
    ske: np.ndarray
    lag_c_days: np.ndarray
    cu: np.ndarray
    lag_u_days: np.ndarray
    model_variant: str
    residual_rmse_mm: float
    n_observations: int
    optimization_diagnostics: dict
    prior_mean: np.ndarray
    prior_precision: np.ndarray

    @property
    def tau_days(self):  # compatibility
        return self.lag_c_days


def _huber(residual, delta):
    a = np.abs(residual)
    return np.where(a <= delta, .5*residual**2, delta*(a-.5*delta))


def map_inversion(deformation_mm, confined_head_m, unconfined_head_m, dates,
                  geological_design, observation_sigma_mm=5.0, prior_mean=None,
                  prior_std=None, pixel_weights=None, huber_delta=1.5,
                  collinearity_threshold=.95, maxiter=300):
    u, hc, hu, Z0 = map(lambda x: np.asarray(x, float),
                         (deformation_mm, confined_head_m, unconfined_head_m, geological_design))
    if u.shape != hc.shape or u.shape != hu.shape or u.shape[0] != Z0.shape[0]:
        raise ValueError("Pixel/time dimensions do not agree")
    common = np.isfinite(hc) & np.isfinite(hu)
    corr = np.corrcoef(hc[common], hu[common])[0, 1] if common.sum() >= 3 else np.nan
    variant = "confined_only" if np.isfinite(corr) and abs(corr) >= collinearity_threshold else "two_aquifer"
    Z = np.column_stack([np.ones(len(Z0)), Z0]); p = Z.shape[1]
    npar = (2 if variant == "confined_only" else 4)*p
    names = [f"log_ske_beta_{i}" for i in range(p)]+[f"lag_c_eta_{i}" for i in range(p)]
    if variant == "two_aquifer":
        names += [f"cu_eta_{i}" for i in range(p)]+[f"lag_u_eta_{i}" for i in range(p)]
    mean = np.asarray(prior_mean if prior_mean is not None else np.zeros(npar), float)
    std = np.asarray(prior_std if prior_std is not None else np.ones(npar), float)
    if mean.shape != (npar,) or std.shape != (npar,) or np.any(std <= 0):
        raise ValueError(f"prior_mean/prior_std must have length {npar} and positive std")
    precision = np.diag(1/std**2)
    weights = np.ones(len(u)) if pixel_weights is None else np.asarray(pixel_weights, float)
    if weights.shape != (len(u),): raise ValueError("pixel_weights must have one value per pixel")

    def predict(theta):
        ske, lagc, cu, lagu = decode_fields(theta, Z, variant); pred = np.full_like(u, np.nan)
        for i in range(len(u)):
            pc = ske[i]*lagged_series(hc[i], lagc[i], dates)
            pu = 0 if variant == "confined_only" else cu[i]*lagged_series(hu[i], lagu[i], dates)
            pred[i] = 1000*(pc+pu)
        return pred

    def objective(theta):
        pred = predict(theta); valid = np.isfinite(u) & np.isfinite(pred)
        residual = (u-pred)/observation_sigma_mm
        data_loss = np.sum(_huber(residual[valid], huber_delta)*np.broadcast_to(weights[:, None], u.shape)[valid])
        d = theta-mean
        return float(data_loss+.5*d@precision@d)
    result = minimize(objective, mean, method="L-BFGS-B", options={"maxiter": maxiter, "maxls": 30})
    pred = predict(result.x); valid = np.isfinite(u) & np.isfinite(pred)
    rmse = float(np.sqrt(np.mean((u[valid]-pred[valid])**2))) if valid.any() else np.nan
    ske, lagc, cu, lagu = decode_fields(result.x, Z, variant)
    diagnostics = {"nit": int(result.nit), "nfev": int(result.nfev), "gradient_norm":
                   float(np.linalg.norm(result.jac)), "head_correlation": float(corr),
                   "collinearity_threshold": float(collinearity_threshold)}
    return MAPResult(result.x, bool(result.success), str(result.message), float(result.fun), names,
                     ske, lagc, cu, lagu, variant, rmse, int(valid.sum()), diagnostics, mean, precision)


def map_inversion_streaming(block_factory,n_covariates,dates,observation_sigma_mm,
                            prior_mean,prior_std,huber_delta=1.5,maxiter=300,
                            model_variant="two_aquifer"):
    """All blocks are revisited for every objective call; no pixels are sampled."""
    p=n_covariates+1;npar=(2 if model_variant=="confined_only" else 4)*p
    mean=np.asarray(prior_mean,float);std=np.asarray(prior_std,float)
    if mean.shape!=(npar,) or std.shape!=(npar,): raise ValueError(f"Streaming priors require {npar} coefficients")
    precision=np.diag(1/std**2);names=[f"coefficient_{i}" for i in range(npar)]
    def block_prediction(theta,hc,hu,Z0):
        Z=np.column_stack([np.ones(len(Z0)),Z0]);ske,lagc,cu,lagu=decode_fields(theta,Z,model_variant);pred=np.full_like(hc,np.nan)
        for i in range(len(Z)):
            value=ske[i]*lagged_series(hc[i],lagc[i],dates)
            if model_variant=="two_aquifer":value+=cu[i]*lagged_series(hu[i],lagu[i],dates)
            pred[i]=1000*value
        return pred
    def objective(theta):
        loss=0.
        for u,hc,hu,Z,weights,_ in block_factory():
            pred=block_prediction(theta,hc,hu,Z);valid=np.isfinite(u)&np.isfinite(pred)
            residual=(u-pred)/observation_sigma_mm
            loss+=np.sum(_huber(residual[valid],huber_delta)*np.broadcast_to(weights[:,None],u.shape)[valid])
        d=theta-mean;return float(loss+.5*d@precision@d)
    result=minimize(objective,mean,method="L-BFGS-B",options={"maxiter":maxiter,"maxls":30})
    sq=0.;n=0;field_blocks=[]
    for u,hc,hu,Z,weights,block_id in block_factory():
        pred=block_prediction(result.x,hc,hu,Z);valid=np.isfinite(u)&np.isfinite(pred);sq+=np.sum((u[valid]-pred[valid])**2);n+=valid.sum()
        design=np.column_stack([np.ones(len(Z)),Z]);field_blocks.append((block_id,decode_fields(result.x,design,model_variant)))
    diagnostics={"nit":int(result.nit),"nfev":int(result.nfev),"gradient_norm":float(np.linalg.norm(result.jac)),"streaming_all_valid_pixels":True}
    # Pixel fields are returned blockwise to avoid materializing the study grid.
    return {"coefficients":result.x,"success":bool(result.success),"message":str(result.message),"objective":float(result.fun),
            "parameter_names":names,"model_variant":model_variant,"residual_rmse_mm":float(np.sqrt(sq/n)) if n else np.nan,
            "n_observations":int(n),"optimization_diagnostics":diagnostics,"prior_mean":mean,
            "prior_precision":precision,"field_blocks":field_blocks}


def rotate_coefficients(coefficients,lag_days,period_days=365.2425):
    coefficients=np.asarray(coefficients,float);angle=2*np.pi*np.asarray(lag_days,float)/period_days
    c,s=np.cos(angle),np.sin(angle);sin0,cos0=coefficients[:,0],coefficients[:,1]
    return np.column_stack([sin0*c+cos0*s,cos0*c-sin0*s])


def harmonic_map_inversion(insar_coefficients,confined_coefficients,unconfined_coefficients,
                           design,covariance=None,pixel_weights=None,prior_mean=None,prior_std=None,
                           period_days=365.2425,collinearity_threshold=.95,huber_delta=1.5,maxiter=300):
    """MAP using two harmonic coefficients per pixel, not reconstructed epochs."""
    obs,hc,hu,Z0=map(lambda x:np.asarray(x,float),(insar_coefficients,confined_coefficients,unconfined_coefficients,design))
    if obs.shape[1:]!=(2,) or hc.shape!=obs.shape or hu.shape!=obs.shape:raise ValueError("Harmonic inputs must be (pixel,2)")
    correlation=np.corrcoef(hc.ravel(),hu.ravel())[0,1];variant="confined_only" if abs(correlation)>=collinearity_threshold else "two_aquifer"
    Z=np.column_stack([np.ones(len(Z0)),Z0]);p=Z.shape[1];npar=(2 if variant=="confined_only" else 4)*p
    mean=np.zeros(npar) if prior_mean is None else np.asarray(prior_mean,float);std=np.ones(npar) if prior_std is None else np.asarray(prior_std,float)
    if len(mean)!=npar or len(std)!=npar:raise ValueError(f"Prior length must be {npar}")
    precision=np.diag(1/std**2);weights=np.ones(len(obs)) if pixel_weights is None else np.asarray(pixel_weights,float)
    invcov=np.asarray([np.eye(2) for _ in obs]) if covariance is None else np.asarray([np.linalg.pinv(x) for x in covariance])
    def predict(theta):
        ske,lagc,cu,lagu=decode_fields(theta,Z,variant);pred=ske[:,None]*rotate_coefficients(hc,lagc,period_days)
        if variant=="two_aquifer":pred+=cu[:,None]*rotate_coefficients(hu,lagu,period_days)
        return 1000*pred
    def objective(theta):
        residual=obs-predict(theta);valid=np.isfinite(residual).all(axis=1)
        mahal=np.sqrt(np.maximum(0,np.einsum("ni,nij,nj->n",residual[valid],invcov[valid],residual[valid])))
        d=theta-mean;return float(np.sum(weights[valid]*_huber(mahal,huber_delta))+.5*d@precision@d)
    result=minimize(objective,mean,method="L-BFGS-B",options={"maxiter":maxiter,"maxls":30});pred=predict(result.x);valid=np.isfinite(obs).all(1)&np.isfinite(pred).all(1)
    fields=decode_fields(result.x,Z,variant)
    return {"coefficients":result.x,"fields":fields,"model_variant":variant,"collinearity":float(correlation),"success":bool(result.success),
            "message":str(result.message),"objective":float(result.fun),"residual_rmse":float(np.sqrt(np.mean((obs[valid]-pred[valid])**2))),
            "prior_mean":mean,"prior_precision":precision,"n_pixels":int(valid.sum()),"parameter_names":[f"coefficient_{i}" for i in range(npar)]}


def harmonic_map_inversion_streaming(block_factory,n_covariates,prior_mean,prior_std,
                                     period_days=365.2425,collinearity_threshold=.95,
                                     huber_delta=1.5,maxiter=300,model_variant="auto",
                                     return_field_blocks=True,observation_sigma_mm=5.0):
    """Streaming MAP over per-pixel sin/cos coefficients for all valid blocks."""
    p=n_covariates+1
    cached_stats=[]
    sx=sy=sxx=syy=sxy=0.;count=0
    for obs,hc,hu,Z0,weights,block_id in block_factory():
        valid=np.isfinite(obs).all(1)&np.isfinite(hc).all(1)&np.isfinite(hu).all(1)&np.isfinite(Z0).all(1)
        if valid.any():
            x=hc[valid].ravel();y=hu[valid].ravel()
            sx+=float(x.sum());sy+=float(y.sum());sxx+=float(x@x);syy+=float(y@y);sxy+=float(x@y);count+=len(x)
        cached_stats.append((int(valid.sum()),block_id))
    if count>=3:
        denom=np.sqrt(max(sxx-sx*sx/count,0)*max(syy-sy*sy/count,0))
        correlation=(sxy-sx*sy/count)/denom if denom>0 else np.nan
    else:
        correlation=np.nan
    variant="confined_only" if model_variant=="confined_only" or (model_variant=="auto" and np.isfinite(correlation) and abs(correlation)>=collinearity_threshold) else "two_aquifer"
    npar=(2 if variant=="confined_only" else 4)*p
    mean=np.asarray(prior_mean,float)[:npar];std=np.asarray(prior_std,float)[:npar]
    if mean.shape!=(npar,) or std.shape!=(npar,) or np.any(std<=0):raise ValueError(f"Harmonic priors require {npar} positive entries")
    precision=np.diag(1/std**2)
    names=[f"log_ske_beta_{i}" for i in range(p)]+[f"lag_c_eta_{i}" for i in range(p)]
    if variant=="two_aquifer":names += [f"cu_eta_{i}" for i in range(p)]+[f"lag_u_eta_{i}" for i in range(p)]
    def predict(theta,hc,hu,Z0):
        Z=np.column_stack([np.ones(len(Z0)),Z0]);ske,lagc,cu,lagu=decode_fields(theta,Z,variant)
        pred=ske[:,None]*rotate_coefficients(hc,lagc,period_days)
        if variant=="two_aquifer":pred+=cu[:,None]*rotate_coefficients(hu,lagu,period_days)
        return 1000*pred
    eval_counter={"n":0}
    def objective_and_grad(theta):
        started=time.time();eval_counter["n"]+=1
        loss=0.;grad=np.zeros_like(theta)
        cursor=0
        ske_beta=slice(cursor,cursor+p);cursor+=p
        lagc_beta=slice(cursor,cursor+p);cursor+=p
        if variant=="two_aquifer":
            cu_beta=slice(cursor,cursor+p);cursor+=p
            lagu_beta=slice(cursor,cursor+p)
        else:
            cu_beta=lagu_beta=None
        for obs,hc,hu,Z0,weights,_ in block_factory():
            Z=np.column_stack([np.ones(len(Z0)),Z0])
            weights=np.asarray(weights,float)
            a=Z@theta[ske_beta];ske=np.exp(np.clip(a,-20,10))
            eta_c=Z@theta[lagc_beta];sig_c=_sigmoid(eta_c);lagc=365*sig_c
            rc=rotate_coefficients(hc,lagc,period_days)
            pred=ske[:,None]*rc
            if variant=="two_aquifer":
                b=Z@theta[cu_beta];cu=_softplus(b)
                eta_u=Z@theta[lagu_beta];sig_u=_sigmoid(eta_u);lagu=365*sig_u
                ru=rotate_coefficients(hu,lagu,period_days)
                pred+=cu[:,None]*ru
            pred*=1000
            residual=(obs-pred)/float(observation_sigma_mm);valid=np.isfinite(residual).all(1)
            if not valid.any():
                continue
            res=residual[valid];mahal=np.sqrt(np.sum(res**2,axis=1))
            loss+=float(np.sum(weights[valid]*_huber(mahal,huber_delta)))
            scale=np.ones_like(mahal)
            large=mahal>huber_delta
            scale[large]=huber_delta/np.maximum(mahal[large],1e-12)
            gpred=-(weights[valid]*scale)[:,None]*res/float(observation_sigma_mm)
            Zv=Z[valid]
            k=2*np.pi/period_days
            hcv=hc[valid];rcv=rc[valid];skev=ske[valid]
            drc=np.column_stack([-hcv[:,0]*np.sin(k*lagc[valid])*k+hcv[:,1]*np.cos(k*lagc[valid])*k,
                                  -hcv[:,1]*np.sin(k*lagc[valid])*k-hcv[:,0]*np.cos(k*lagc[valid])*k])
            ske_clip=(a[valid]>-20)&(a[valid]<10)
            grad[ske_beta]+=Zv.T@(1000*skev*ske_clip*np.sum(gpred*rcv,axis=1))
            grad[lagc_beta]+=Zv.T@(1000*skev*(365*sig_c[valid]*(1-sig_c[valid]))*np.sum(gpred*drc,axis=1))
            if variant=="two_aquifer":
                huv=hu[valid];ruv=ru[valid];cuv=cu[valid]
                dru=np.column_stack([-huv[:,0]*np.sin(k*lagu[valid])*k+huv[:,1]*np.cos(k*lagu[valid])*k,
                                      -huv[:,1]*np.sin(k*lagu[valid])*k-huv[:,0]*np.cos(k*lagu[valid])*k])
                grad[cu_beta]+=Zv.T@(1000*_sigmoid(b[valid])*np.sum(gpred*ruv,axis=1))
                grad[lagu_beta]+=Zv.T@(1000*cuv*(365*sig_u[valid]*(1-sig_u[valid]))*np.sum(gpred*dru,axis=1))
        d=theta-mean
        value=float(loss+.5*d@precision@d)
        print(f"harmonic_map_eval {eval_counter['n']} loss {value:.6g} seconds {time.time()-started:.1f}",flush=True)
        return value,grad+precision@d
    def callback(theta):
        print(f"harmonic_map_iter {eval_counter['n']}",flush=True)
    result=minimize(objective_and_grad,mean,method="L-BFGS-B",jac=True,callback=callback,options={"maxiter":maxiter,"maxls":30,"ftol":1e-6})
    sq=0.;n=0;field_blocks=[]
    for obs,hc,hu,Z0,weights,block_id in block_factory():
        pred=predict(result.x,hc,hu,Z0);valid=np.isfinite(obs).all(1)&np.isfinite(pred).all(1)
        residual=np.full(len(obs),np.nan)
        if valid.any():
            residual[valid]=np.sqrt(np.mean((obs[valid]-pred[valid])**2,axis=1));sq+=float(np.sum((obs[valid]-pred[valid])**2));n+=int(2*valid.sum())
        if return_field_blocks:
            design=np.column_stack([np.ones(len(Z0)),Z0])
            field_blocks.append((block_id,decode_fields(result.x,design,variant),residual))
    diagnostics={"nit":int(result.nit),"nfev":int(result.nfev),"gradient_norm":float(np.linalg.norm(result.jac)),
                 "head_coefficient_correlation":float(correlation) if np.isfinite(correlation) else None,
                 "collinearity_threshold":float(collinearity_threshold),"valid_block_pixels":sum(x[0] for x in cached_stats)}
    return {"coefficients":result.x,"success":bool(result.success),"message":str(result.message),
            "objective":float(result.fun),"parameter_names":names,"model_variant":variant,
            "residual_rmse_mm":float(np.sqrt(sq/n)) if n else np.nan,"n_observations":int(n),
            "optimization_diagnostics":diagnostics,"prior_mean":mean,"prior_precision":precision,
            "field_blocks":field_blocks}
