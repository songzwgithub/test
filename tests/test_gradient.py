from __future__ import annotations

import numpy as np

from hengshui_insar.bounded_model import objective_and_gradient


def test_objective_finite_difference_gradient() -> None:
    basis = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, -0.25]])
    gamma = np.array([0.1, -0.2])
    eta = np.array([0.0, 0.0, 0.0])
    target = np.array([0.02, 0.021, 0.019])
    _, grad = objective_and_gradient(eta, basis, gamma, target, 1e-8, 0.05, 0.3)
    eps = 1e-6
    numeric = []
    for i in range(gamma.size):
        step = np.zeros_like(gamma)
        step[i] = eps
        fp, _ = objective_and_gradient(eta, basis, gamma + step, target, 1e-8, 0.05, 0.3)
        fm, _ = objective_and_gradient(eta, basis, gamma - step, target, 1e-8, 0.05, 0.3)
        numeric.append((fp - fm) / (2 * eps))
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-8)
