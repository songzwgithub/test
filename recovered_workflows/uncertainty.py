"""Explicit finite-difference Gauss-Newton/Laplace coefficient uncertainty."""
from __future__ import annotations

import numpy as np

from storage_inversion import decode_fields, lagged_series, rotate_coefficients


def prediction_vector(theta, Z, variant, hc, hu, dates):
    ske, lagc, cu, lagu = decode_fields(theta, Z, variant); rows = []
    for i in range(len(Z)):
        value = ske[i]*lagged_series(hc[i], lagc[i], dates)
        if variant == "two_aquifer": value += cu[i]*lagged_series(hu[i], lagu[i], dates)
        rows.append(1000*value)
    return np.asarray(rows).ravel()


def explicit_gauss_newton(map_result, geological_design, confined_head_m,
                          unconfined_head_m, dates, observation_sigma_mm=5.0,
                          finite_difference_step=1e-4):
    Z = np.column_stack([np.ones(len(geological_design)), geological_design])
    theta = map_result.coefficients; base = prediction_vector(theta, Z, map_result.model_variant,
                                                               confined_head_m, unconfined_head_m, dates)
    valid = np.isfinite(base); J = np.empty((valid.sum(), len(theta)))
    for j in range(len(theta)):
        step = finite_difference_step*max(1, abs(theta[j])); shifted = theta.copy(); shifted[j] += step
        J[:, j] = (prediction_vector(shifted, Z, map_result.model_variant,
                                     confined_head_m, unconfined_head_m, dates)[valid]-base[valid])/step
    hessian = J.T@J/(observation_sigma_mm**2)+map_result.prior_precision
    covariance = np.linalg.pinv(hessian, rcond=1e-10)
    return covariance, hessian


def laplace_posterior(map_result, geological_design, confined_head_m,
                      unconfined_head_m, dates, n_draws=1000, random_seed=42,
                      observation_sigma_mm=5.0):
    covariance, hessian = explicit_gauss_newton(map_result, geological_design, confined_head_m,
        unconfined_head_m, dates, observation_sigma_mm)
    rng = np.random.default_rng(random_seed)
    draws = rng.multivariate_normal(map_result.coefficients, covariance, size=n_draws)
    Z = np.column_stack([np.ones(len(geological_design)), geological_design])
    fields = [decode_fields(draw, Z, map_result.model_variant) for draw in draws]
    ske = np.asarray([x[0] for x in fields]); lag = np.asarray([x[1] for x in fields])
    prior_variance = np.diag(np.linalg.pinv(map_result.prior_precision)); post_variance = np.diag(covariance)
    reduction = 1-post_variance/prior_variance
    coefficient_summary = {"posterior_mean": map_result.coefficients,
                           "posterior_std": np.sqrt(post_variance),
                           "ci95_low": map_result.coefficients-1.96*np.sqrt(post_variance),
                           "ci95_high": map_result.coefficients+1.96*np.sqrt(post_variance),
                           "prior_to_posterior_variance_reduction": reduction,
                           "identifiability_class": np.where(reduction >= .5, "identified",
                               np.where(reduction >= .1, "weak", "not_identified"))}
    identifiable = np.all(reduction >= .1)
    return {"coefficient_draws": draws, "coefficient_summary": coefficient_summary,
            "covariance": covariance, "hessian": hessian,
            "Ske_mean": np.mean(ske, axis=0) if identifiable else np.full(len(Z), np.nan),
            "Ske_std": np.std(ske, axis=0) if identifiable else np.full(len(Z), np.nan),
            "Ske_ci95": np.quantile(ske, [.025, .975], axis=0) if identifiable else np.full((2,len(Z)), np.nan),
            "lag_mean": np.mean(lag, axis=0) if identifiable else np.full(len(Z), np.nan),
            "lag_uncertainty": np.std(lag, axis=0) if identifiable else np.full(len(Z), np.nan)}


def streaming_gauss_newton(coefficients,model_variant,block_factory,dates,prior_precision,
                           observation_sigma_mm=5.0,finite_difference_step=1e-4):
    """Accumulate J'J blockwise over every valid pixel without L-BFGS hess_inv."""
    theta=np.asarray(coefficients,float);hessian=np.asarray(prior_precision,float).copy()
    for observed,hc,hu,Z0,weights,_ in block_factory():
        Z=np.column_stack([np.ones(len(Z0)),Z0])
        def predict(v):
            ske,lagc,cu,lagu=decode_fields(v,Z,model_variant);rows=[]
            for i in range(len(Z)):
                value=ske[i]*lagged_series(hc[i],lagc[i],dates)
                if model_variant=="two_aquifer":value+=cu[i]*lagged_series(hu[i],lagu[i],dates)
                rows.append(1000*value)
            return np.asarray(rows).ravel()
        base=predict(theta);observed=np.asarray(observed).ravel();valid=np.isfinite(base)&np.isfinite(observed);J=np.empty((valid.sum(),len(theta)))
        for j in range(len(theta)):
            step=finite_difference_step*max(1,abs(theta[j]));shift=theta.copy();shift[j]+=step
            J[:,j]=(predict(shift)[valid]-base[valid])/step
        standardized=np.abs((observed[valid]-base[valid])/observation_sigma_mm);robust=np.where(standardized<=1.5,1.,1.5/np.maximum(standardized,1e-12))
        w=np.repeat(weights,hc.shape[1])[valid]*robust/observation_sigma_mm**2
        hessian+=J.T@(J*w[:,None])
    covariance=np.linalg.pinv(hessian,rcond=1e-10)
    return covariance,hessian


def harmonic_streaming_gauss_newton(coefficients,model_variant,block_factory,prior_precision,
                                    period_days=365.2425,finite_difference_step=1e-4,max_pixels_per_block=4096,
                                    observation_sigma_mm=5.0,huber_delta=1.5):
    """Blockwise Gauss-Newton Hessian for two-coefficient harmonic observations."""
    theta=np.asarray(coefficients,float);hessian=np.asarray(prior_precision,float).copy()
    def predict(v,hc,hu,Z0):
        Z=np.column_stack([np.ones(len(Z0)),Z0]);ske,lagc,cu,lagu=decode_fields(v,Z,model_variant)
        pred=ske[:,None]*rotate_coefficients(hc,lagc,period_days)
        if model_variant=="two_aquifer":pred+=cu[:,None]*rotate_coefficients(hu,lagu,period_days)
        return 1000*pred
    for observed,hc,hu,Z0,weights,_ in block_factory():
        full_weight_sum=float(np.asarray(weights,float).sum())
        if len(Z0)>max_pixels_per_block:
            idx=np.linspace(0,len(Z0)-1,max_pixels_per_block).round().astype(int)
            observed,hc,hu,Z0,weights=observed[idx],hc[idx],hu[idx],Z0[idx],weights[idx]
            sampled_sum=float(np.asarray(weights,float).sum())
            if sampled_sum>0:
                weights=weights*full_weight_sum/sampled_sum
        base=predict(theta,hc,hu,Z0).reshape(-1);obs=np.asarray(observed,float).reshape(-1)
        valid=np.isfinite(base)&np.isfinite(obs);J=np.empty((valid.sum(),len(theta)))
        for j in range(len(theta)):
            step=finite_difference_step*max(1,abs(theta[j]));shift=theta.copy();shift[j]+=step
            J[:,j]=(predict(shift,hc,hu,Z0).reshape(-1)[valid]-base[valid])/step
        standardized=np.abs((obs[valid]-base[valid])/float(observation_sigma_mm))
        robust=np.where(standardized<=huber_delta,1.,huber_delta/np.maximum(standardized,1e-12))
        w=np.repeat(np.asarray(weights,float),2)[valid]*robust/(float(observation_sigma_mm)**2)
        hessian+=J.T@(J*w[:,None])
    covariance=np.linalg.pinv(hessian,rcond=1e-10)
    return covariance,hessian
