"""M1 two-aquifer inversion with shared unconfined response parameters."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from storage_inversion import rotate_coefficients


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def softplus(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def softplus_derivative(x):
    return sigmoid(x)


@dataclass(frozen=True)
class M1ParameterLayout:
    n_ske_geology: int
    n_ske_rbf: int
    n_lag_c_geology: int
    n_lag_c_rbf: int
    lag_c_mode: str
    has_global_cu: bool = True
    has_global_lag_u: bool = True

    def __post_init__(self):
        if self.lag_c_mode not in {"L0_shared", "L1_geology", "L2_geology_rbf"}:
            raise ValueError(f"Unsupported lag_c_mode: {self.lag_c_mode}")
        if self.lag_c_mode == "L0_shared" and (self.n_lag_c_geology or self.n_lag_c_rbf):
            raise ValueError("L0_shared cannot include lag_c geology or RBF columns")
        if self.lag_c_mode == "L1_geology" and self.n_lag_c_rbf:
            raise ValueError("L1_geology cannot include lag_c RBF columns")

    @property
    def n_ske_columns(self):
        return 1 + self.n_ske_geology + self.n_ske_rbf

    @property
    def n_lag_c_columns(self):
        if self.lag_c_mode == "L0_shared":
            return 1
        return 1 + self.n_lag_c_geology + self.n_lag_c_rbf

    @property
    def total_parameters(self):
        return self.n_ske_columns + self.n_lag_c_columns + 2

    @property
    def slices(self):
        start = 0
        ske = slice(start, start + self.n_ske_columns)
        start = ske.stop
        lag_c = slice(start, start + self.n_lag_c_columns)
        start = lag_c.stop
        cu = slice(start, start + 1)
        start = cu.stop
        lag_u = slice(start, start + 1)
        return {"ske": ske, "lag_c": lag_c, "cu_global": cu, "lag_u_global": lag_u}

    @property
    def column_names(self):
        names = ["ske_intercept"]
        names += [f"ske_geology_beta_{i}" for i in range(self.n_ske_geology)]
        names += [f"ske_rbf_alpha_{i}" for i in range(self.n_ske_rbf)]
        names += ["lag_c_intercept_or_global"]
        names += [f"lag_c_geology_beta_{i}" for i in range(self.n_lag_c_geology)]
        names += [f"lag_c_rbf_alpha_{i}" for i in range(self.n_lag_c_rbf)]
        names += ["log_Cu_global", "raw_lag_u_global"]
        return names

    def metadata(self):
        return {
            "model_variant": "M1_two_aquifer_shared_unconfined",
            "n_ske_geology": self.n_ske_geology,
            "n_ske_rbf": self.n_ske_rbf,
            "n_lag_c_geology": self.n_lag_c_geology,
            "n_lag_c_rbf": self.n_lag_c_rbf,
            "lag_c_mode": self.lag_c_mode,
            "has_global_cu": self.has_global_cu,
            "has_global_lag_u": self.has_global_lag_u,
            "total_parameters": self.total_parameters,
            "slices": {k: [v.start, v.stop] for k, v in self.slices.items()},
            "column_names": self.column_names,
        }


@dataclass
class M1Design:
    ske: np.ndarray
    lag_c: np.ndarray


def make_design(ske_geology, rbf, lag_c_geology=None, lag_c_rbf=None, lag_c_mode="L0_shared"):
    n = len(rbf) if rbf is not None and np.asarray(rbf).ndim == 2 else len(ske_geology)
    ske_parts = [np.ones((n, 1), float)]
    if ske_geology is not None and np.asarray(ske_geology).size:
        ske_parts.append(np.asarray(ske_geology, float).reshape(n, -1))
    if rbf is not None and np.asarray(rbf).size:
        ske_parts.append(np.asarray(rbf, float).reshape(n, -1))
    if lag_c_mode == "L0_shared":
        lag_parts = [np.ones((n, 1), float)]
    else:
        lag_parts = [np.ones((n, 1), float)]
        if lag_c_geology is not None and np.asarray(lag_c_geology).size:
            lag_parts.append(np.asarray(lag_c_geology, float).reshape(n, -1))
        if lag_c_mode == "L2_geology_rbf" and lag_c_rbf is not None and np.asarray(lag_c_rbf).size:
            lag_parts.append(np.asarray(lag_c_rbf, float).reshape(n, -1))
    return M1Design(ske=np.column_stack(ske_parts), lag_c=np.column_stack(lag_parts))


def decode_m1_parameters(theta, design: M1Design, layout: M1ParameterLayout, period_days=365.2425, lag_u_upper_days=182.62125):
    theta = np.asarray(theta, float)
    sl = layout.slices
    log_ske = design.ske @ theta[sl["ske"]]
    ske = np.exp(np.clip(log_ske, -20, 10))
    eta_lag_c = design.lag_c @ theta[sl["lag_c"]]
    lag_c = period_days * sigmoid(eta_lag_c)
    cu_raw = float(theta[sl["cu_global"]][0])
    cu = float(softplus(cu_raw))
    lag_u_raw = float(theta[sl["lag_u_global"]][0])
    lag_u = float(lag_u_upper_days * sigmoid(lag_u_raw))
    return {"Ske_pixel": ske, "lag_c_pixel": lag_c, "Cu_global": cu, "lag_u_global": lag_u}


def predict_m1(theta, design, layout, hc, hu, period_days=365.2425, lag_u_upper_days=182.62125):
    decoded = decode_m1_parameters(theta, design, layout, period_days, lag_u_upper_days)
    rc = rotate_coefficients(hc, decoded["lag_c_pixel"], period_days)
    ru = rotate_coefficients(hu, decoded["lag_u_global"], period_days)
    pred_m = decoded["Ske_pixel"][:, None] * rc + decoded["Cu_global"] * ru
    return 1000.0 * pred_m


def _huber_loss_and_scale(mahal, delta):
    loss = np.where(mahal <= delta, 0.5 * mahal**2, delta * (mahal - 0.5 * delta))
    scale = np.ones_like(mahal)
    large = mahal > delta
    scale[large] = delta / np.maximum(mahal[large], 1e-12)
    return loss, scale


def m1_objective_and_gradient(
    theta,
    design,
    layout,
    obs,
    hc,
    hu,
    weights=None,
    prior_mean=None,
    prior_std=None,
    period_days=365.2425,
    lag_u_upper_days=182.62125,
    observation_sigma_mm=5.0,
    huber_delta=1.5,
    include_prior=True,
):
    theta = np.asarray(theta, float)
    n = len(obs)
    weights = np.ones(n, float) / max(n, 1) if weights is None else np.asarray(weights, float)
    prior_mean = np.zeros_like(theta) if prior_mean is None else np.asarray(prior_mean, float)
    prior_std = np.ones_like(theta) if prior_std is None else np.asarray(prior_std, float)
    if theta.shape[0] != layout.total_parameters:
        raise ValueError("theta size does not match M1 layout")
    sl = layout.slices
    decoded = decode_m1_parameters(theta, design, layout, period_days, lag_u_upper_days)
    ske = decoded["Ske_pixel"]
    lag_c = decoded["lag_c_pixel"]
    cu = decoded["Cu_global"]
    lag_u = decoded["lag_u_global"]
    k = 2 * np.pi / period_days
    rc = rotate_coefficients(hc, lag_c, period_days)
    ru = rotate_coefficients(hu, lag_u, period_days)
    pred = 1000.0 * (ske[:, None] * rc + cu * ru)
    residual = (np.asarray(obs, float) - pred) / observation_sigma_mm
    valid = np.isfinite(residual).all(axis=1) & np.isfinite(design.ske).all(axis=1) & np.isfinite(design.lag_c).all(axis=1)
    if not valid.any():
        raise ValueError("No finite M1 observations")
    res = residual[valid]
    mahal = np.sqrt(np.sum(res * res, axis=1))
    loss_vec, scale = _huber_loss_and_scale(mahal, huber_delta)
    loss = float(np.sum(weights[valid] * loss_vec))
    gpred = -(weights[valid] * scale)[:, None] * res / observation_sigma_mm
    grad = np.zeros_like(theta)
    Xs = design.ske[valid]
    Xl = design.lag_c[valid]
    hcv = hc[valid]
    huv = hu[valid]
    rcv = rc[valid]
    ruv = ru[valid]
    skev = ske[valid]
    grad[sl["ske"]] += Xs.T @ (1000.0 * skev * np.sum(gpred * rcv, axis=1))
    dlagc_deta = period_days * sigmoid(Xl @ theta[sl["lag_c"]]) * (1 - sigmoid(Xl @ theta[sl["lag_c"]]))
    drc = np.column_stack(
        [
            -hcv[:, 0] * np.sin(k * lag_c[valid]) * k + hcv[:, 1] * np.cos(k * lag_c[valid]) * k,
            -hcv[:, 1] * np.sin(k * lag_c[valid]) * k - hcv[:, 0] * np.cos(k * lag_c[valid]) * k,
        ]
    )
    grad[sl["lag_c"]] += Xl.T @ (1000.0 * skev * dlagc_deta * np.sum(gpred * drc, axis=1))
    grad[sl["cu_global"]] += np.array([1000.0 * softplus_derivative(theta[sl["cu_global"]][0]) * np.sum(gpred * ruv)])
    dlagu_draw = lag_u_upper_days * sigmoid(theta[sl["lag_u_global"]][0]) * (1 - sigmoid(theta[sl["lag_u_global"]][0]))
    dru = np.column_stack(
        [
            -huv[:, 0] * np.sin(k * lag_u) * k + huv[:, 1] * np.cos(k * lag_u) * k,
            -huv[:, 1] * np.sin(k * lag_u) * k - huv[:, 0] * np.cos(k * lag_u) * k,
        ]
    )
    grad[sl["lag_u_global"]] += np.array([1000.0 * cu * dlagu_draw * np.sum(gpred * dru)])
    if include_prior:
        prior_precision = 1.0 / np.maximum(prior_std, 1e-12) ** 2
        diff = theta - prior_mean
        loss += float(0.5 * np.sum(diff * diff * prior_precision))
        grad += diff * prior_precision
    return loss, grad


def finite_difference_gradient_error(fn, theta, eps=1e-6):
    theta = np.asarray(theta, float)
    value, grad = fn(theta)
    fd = np.zeros_like(theta)
    for i in range(len(theta)):
        step = np.zeros_like(theta)
        step[i] = eps
        fp = fn(theta + step)[0]
        fm = fn(theta - step)[0]
        fd[i] = (fp - fm) / (2 * eps)
    denom = max(1.0, float(np.linalg.norm(fd) + np.linalg.norm(grad)))
    return float(np.linalg.norm(fd - grad) / denom), value


def parameter_hash(theta):
    arr = np.asarray(theta, dtype="float64")
    return sha256(arr.tobytes()).hexdigest()
