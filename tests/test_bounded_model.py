from __future__ import annotations

import numpy as np

from hengshui_insar.bounded_model import bounded_sigmoid, ske_and_derivative


def test_bounded_sigmoid_and_derivative() -> None:
    eta = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    sig = bounded_sigmoid(eta)
    assert np.all(np.isfinite(sig))
    ske, deriv = ske_and_derivative(eta, 1e-8, 0.05)
    assert np.all((ske >= 1e-8) & (ske <= 0.05))
    assert np.all(deriv >= 0.0)
