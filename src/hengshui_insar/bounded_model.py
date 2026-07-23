"""Bounded Ske model math."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundedProtocol:
    rbf_dim: int
    ske_min: float
    ske_max: float
    lambda_value: float
    lag_u_days: float


def bounded_sigmoid(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def ske_and_derivative(eta: np.ndarray, ske_min: float, ske_max: float) -> tuple[np.ndarray, np.ndarray]:
    sig = bounded_sigmoid(eta)
    span = float(ske_max) - float(ske_min)
    return float(ske_min) + span * sig, span * sig * (1.0 - sig)


def prediction(ske: np.ndarray, confined: np.ndarray, unconfined: np.ndarray, cu_global: float) -> np.ndarray:
    return 1000.0 * (ske[:, None] * confined + float(cu_global) * unconfined)


def objective_and_gradient(eta: np.ndarray, basis: np.ndarray, gamma: np.ndarray, target_ske: np.ndarray, ske_min: float, ske_max: float, lambda_value: float) -> tuple[float, np.ndarray]:
    field_eta = np.asarray(eta, dtype=float) + basis @ gamma
    ske, dske = ske_and_derivative(field_eta, ske_min, ske_max)
    residual = ske - target_ske
    objective = 0.5 * float(np.sum(residual * residual)) + 0.5 * float(lambda_value) * float(gamma @ gamma)
    grad_gamma = basis.T @ (residual * dske) + float(lambda_value) * gamma
    return objective, grad_gamma
