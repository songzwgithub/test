from __future__ import annotations

import numpy as np

from hengshui_insar.constants import ANNUAL_PERIOD_DAYS, LAG_C_DAYS
from hengshui_insar.harmonics import harmonic_value, phase_days, rotate_sin_cos_coefficients


def _circular_diff(a: float, b: float) -> float:
    d = (a - b) % ANNUAL_PERIOD_DAYS
    if d > ANNUAL_PERIOD_DAYS / 2:
        d -= ANNUAL_PERIOD_DAYS
    return float(d)


def test_positive_delayed_rotation() -> None:
    coeff = np.array([[0.0, 1.0]])
    delayed = rotate_sin_cos_coefficients(coeff, LAG_C_DAYS, ANNUAL_PERIOD_DAYS)[0]
    shift = _circular_diff(float(phase_days(delayed[0], delayed[1], ANNUAL_PERIOD_DAYS)), 0.0)
    assert abs(shift - LAG_C_DAYS) < 1e-12


def test_phase_day_and_harmonic_value() -> None:
    assert abs(float(phase_days(0.0, 1.0, ANNUAL_PERIOD_DAYS))) < 1e-12
    assert abs(float(harmonic_value(0.0, 1.0, 0.0, ANNUAL_PERIOD_DAYS)) - 1.0) < 1e-12
