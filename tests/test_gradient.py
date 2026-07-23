from __future__ import annotations

import numpy as np

from hengshui_insar.bounded_model import objective_and_gradient
from hengshui_insar.source_recompute import StreamInputs
from hengshui_insar import optimization


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


def test_formal_objective_analytical_gradient(monkeypatch) -> None:
    obs = np.array([[3.0, -1.0], [1.5, 2.0], [-2.0, 0.25], [0.5, -1.5]], dtype=float)
    hc = np.array([[1.2, 0.5], [0.2, -0.6], [-0.4, 1.1], [0.8, -0.2]], dtype=float)
    hu = np.array([[0.4, 0.7], [0.9, -0.1], [-0.3, 0.4], [0.1, -0.8]], dtype=float)
    basis = np.array([[1.0, 0.0], [0.4, -0.5], [-0.2, 0.7], [0.3, 0.2]], dtype=float)

    def fake_iter_model_blocks(inputs, fold_id, split):
        yield obs, hc, hu, basis

    monkeypatch.setattr(optimization, "iter_model_blocks", fake_iter_model_blocks)
    inputs = StreamInputs(rbf_dim=2, observation_sigma_mm=5.0)
    theta = np.array([-3.8, 0.25, -0.12, np.log(0.004), 42.0], dtype=float)
    value, grad, _ = optimization.formal_objective_and_gradient(theta, inputs, fold_id=1, split="training")
    assert np.isfinite(value)
    assert np.isfinite(grad).all()

    eps = 1e-6
    numeric = np.empty_like(theta)
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = eps
        fp, _, _ = optimization.formal_objective_and_gradient(theta + step, inputs, fold_id=1, split="training")
        fm, _, _ = optimization.formal_objective_and_gradient(theta - step, inputs, fold_id=1, split="training")
        numeric[i] = (fp - fm) / (2.0 * eps)
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-7)
