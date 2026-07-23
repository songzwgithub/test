"""Annual sine/cosine harmonic helpers."""

from __future__ import annotations

import numpy as np


def harmonic_value(real: np.ndarray | float, imag: np.ndarray | float, day: np.ndarray | float, period_days: float) -> np.ndarray:
    angle = 2.0 * np.pi * np.asarray(day, dtype=float) / float(period_days)
    return np.asarray(real, dtype=float) * np.sin(angle) + np.asarray(imag, dtype=float) * np.cos(angle)


def phase_days(real: np.ndarray | float, imag: np.ndarray | float, period_days: float) -> np.ndarray:
    return (np.arctan2(np.asarray(real, dtype=float), np.asarray(imag, dtype=float)) * float(period_days) / (2.0 * np.pi)) % float(period_days)


def rotate_sin_cos_coefficients(coefficients: np.ndarray, lag_days: float, period_days: float) -> np.ndarray:
    """Rotate coefficients to represent y(t-lag_days).

    Positive lag is a delayed response, so the peak day advances on the
    circular calendar by +lag_days.
    """
    coeff = np.asarray(coefficients, dtype=float)
    if coeff.shape[-1] != 2:
        raise ValueError("coefficients last dimension must be [sin, cos]")
    angle = 2.0 * np.pi * float(lag_days) / float(period_days)
    c = np.cos(angle)
    s = np.sin(angle)
    out = np.empty_like(coeff, dtype=float)
    real = coeff[..., 0]
    imag = coeff[..., 1]
    out[..., 0] = real * c + imag * s
    out[..., 1] = imag * c - real * s
    return out
