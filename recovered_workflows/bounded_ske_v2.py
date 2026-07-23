"""Bounded-Ske utilities for M1_v2_bounded_Ske development."""
from __future__ import annotations

import numpy as np


SKE_LOWER_BOUND = 1e-6
SKE_UPPER_BOUND = 0.05
PARAMETERIZATION_VERSION = "bounded_logistic_ske_v1"
OBJECTIVE_VERSION = "stage_c_bounded_ske_fixed_lagu_squared_loss_v1"


def stable_sigmoid(eta):
    """Overflow-safe logistic sigmoid."""
    eta = np.asarray(eta, dtype=float)
    out = np.empty_like(eta, dtype=float)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    exp_eta = np.exp(eta[~pos])
    out[~pos] = exp_eta / (1.0 + exp_eta)
    return out


def bounded_ske(eta, lower=SKE_LOWER_BOUND, upper=SKE_UPPER_BOUND):
    s = stable_sigmoid(eta)
    return lower + (upper - lower) * s


def bounded_ske_derivative(eta, lower=SKE_LOWER_BOUND, upper=SKE_UPPER_BOUND):
    s = stable_sigmoid(eta)
    return (upper - lower) * s * (1.0 - s)


def inverse_bounded_ske(ske, lower=SKE_LOWER_BOUND, upper=SKE_UPPER_BOUND):
    ske = np.asarray(ske, dtype=float)
    eps = np.finfo(float).eps
    p = (ske - lower) / (upper - lower)
    p = np.minimum(np.maximum(p, eps), 1.0 - eps)
    return np.log(p / (1.0 - p))


def saturation_fractions(ske, lower=SKE_LOWER_BOUND, upper=SKE_UPPER_BOUND):
    ske = np.asarray(ske, dtype=float)
    span = upper - lower
    return {
        "fraction_Ske_within_1pct_of_lower_bound": float(np.mean(ske <= lower + 0.01 * span)),
        "fraction_Ske_within_1pct_of_upper_bound": float(np.mean(ske >= upper - 0.01 * span)),
        "fraction_Ske_within_5pct_of_upper_bound": float(np.mean(ske >= upper - 0.05 * span)),
    }
