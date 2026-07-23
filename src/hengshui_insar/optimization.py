"""Formal bounded inversion optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bounded_model import ske_and_derivative
from .constants import ANNUAL_PERIOD_DAYS, LAG_U_DAYS, LAMBDA, RELEASE_ROOT, RBF_DIMENSION, SKE_MAX, SKE_MIN
from .harmonics import rotate_sin_cos_coefficients
from .io import write_json
from .source_recompute import OBSERVATION_SIGMA_MM, StreamInputs, evaluate_parameters, iter_model_blocks, load_release_parameters


@dataclass(frozen=True)
class ConvergenceGate:
    rel_objective_last10: float
    rel_step_last10: float
    scaled_gradient_rms: float


def convergence_gate(history: list[dict], gate: ConvergenceGate) -> bool:
    if len(history) < 10:
        return False
    tail = history[-10:]
    first = abs(float(tail[0]["objective_total"]))
    rel_drop = abs(float(tail[-1]["objective_total"]) - float(tail[0]["objective_total"])) / max(first, 1e-30)
    max_step = max(float(row.get("relative_parameter_step", float("inf"))) for row in tail[1:])
    grad = float(tail[-1].get("scaled_gradient_rms", float("inf")))
    return rel_drop <= gate.rel_objective_last10 and max_step <= gate.rel_step_last10 and grad <= gate.scaled_gradient_rms


def formal_objective_and_gradient(theta: np.ndarray, inputs: StreamInputs, fold_id: int | None = None, split: str = "all") -> tuple[float, np.ndarray, dict[str, Any]]:
    """Return scaled objective and analytical gradient for the bounded model."""
    theta = np.asarray(theta, dtype=float)
    rbf_dim = inputs.rbf_dim
    eta0 = float(theta[0])
    gamma = theta[1 : 1 + rbf_dim]
    cu = float(np.exp(np.clip(theta[1 + rbf_dim], -40.0, 20.0)))
    lag_c = float(theta[2 + rbf_dim])
    grad = np.zeros_like(theta)
    data_loss = 0.0
    ncoef = 0
    k = 2.0 * np.pi / ANNUAL_PERIOD_DAYS
    sigma2 = OBSERVATION_SIGMA_MM**2
    for obs, hc, hu, basis in iter_model_blocks(inputs, fold_id=fold_id, split=split):
        eta = eta0 + basis @ gamma
        ske, dske = ske_and_derivative(eta, SKE_MIN, SKE_MAX)
        rc = rotate_sin_cos_coefficients(hc, lag_c, ANNUAL_PERIOD_DAYS)
        ru = rotate_sin_cos_coefficients(hu, LAG_U_DAYS, ANNUAL_PERIOD_DAYS)
        pred = 1000.0 * (ske[:, None] * rc + cu * ru)
        res = obs - pred
        data_loss += 0.5 * float(np.sum(res * res) / sigma2)
        ncoef += int(res.size)
        common = -1000.0 * dske * np.sum(res * rc, axis=1) / sigma2
        grad[0] += float(np.sum(common))
        grad[1 : 1 + rbf_dim] += basis.T @ common
        grad[1 + rbf_dim] += -float(np.sum(res * (1000.0 * cu * ru)) / sigma2)
        s0, c0 = hc[:, 0], hc[:, 1]
        angle = 2.0 * np.pi * lag_c / ANNUAL_PERIOD_DAYS
        ca, sa = np.cos(angle), np.sin(angle)
        drc = np.column_stack([(-s0 * sa + c0 * ca) * k, (-c0 * sa - s0 * ca) * k])
        grad[2 + rbf_dim] += -float(np.sum(res * (1000.0 * ske[:, None] * drc)) / sigma2)
    prior = 0.5 * LAMBDA * float(gamma @ gamma)
    grad[1 : 1 + rbf_dim] += LAMBDA * gamma
    scale = float(max(ncoef, 1))
    total = (data_loss + prior) / scale
    grad /= scale
    return total, grad, {
        "objective_total": total,
        "objective_data": data_loss / scale,
        "objective_regularization": prior / scale,
        "n_coefficients": ncoef,
        "gradient_norm": float(np.linalg.norm(grad)),
        "gradient_rms": float(np.sqrt(np.mean(grad * grad))),
    }


def optimize_formal_inversion(
    output_dir: Path,
    inputs: StreamInputs = StreamInputs(),
    fold_id: int | None = None,
    split: str = "all",
    maxiter: int = 300,
    theta0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run L-BFGS-B optimization and save parameters plus fit summary."""
    from scipy.optimize import minimize

    output_dir.mkdir(parents=True, exist_ok=True)
    if theta0 is None:
        theta0 = load_release_parameters(inputs.release_root, fold_id)
    theta0 = np.asarray(theta0, dtype=float)
    history: list[dict[str, Any]] = []
    previous_theta: np.ndarray | None = None
    previous_obj: float | None = None

    def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad, _parts = formal_objective_and_gradient(theta, inputs, fold_id=fold_id, split=split)
        return value, grad

    def callback(theta: np.ndarray) -> None:
        nonlocal previous_theta, previous_obj
        value, grad, parts = formal_objective_and_gradient(theta, inputs, fold_id=fold_id, split=split)
        step = float(np.linalg.norm(theta - previous_theta)) if previous_theta is not None else float("inf")
        rel_step = step / max(float(np.linalg.norm(previous_theta)), 1.0) if previous_theta is not None else float("inf")
        rel_obj = abs(value - previous_obj) / max(abs(previous_obj), 1e-30) if previous_obj is not None else float("inf")
        metrics = evaluate_parameters(theta, inputs, fold_id=fold_id, split=split)
        row = {
            "iteration": len(history) + 1,
            "objective_total": value,
            "objective_data": parts["objective_data"],
            "objective_regularization": parts["objective_regularization"],
            "relative_objective_change": rel_obj,
            "gradient_norm": float(np.linalg.norm(grad)),
            "gradient_rms": float(np.sqrt(np.mean(grad * grad))),
            "parameter_step_norm": step,
            "relative_parameter_step": rel_step,
            "training_rmse": metrics["rmse"],
            "Ske_min": metrics["Ske_min"],
            "Ske_p50": metrics["Ske_p50"],
            "Ske_max": metrics["Ske_max"],
            "Cu_global": metrics["Cu_global"],
            "lag_c_days": metrics["lag_c_days"],
            "lag_u_days": metrics["lag_u_days"],
            "gamma_norm": metrics["gamma_norm"],
        }
        history.append(row)
        previous_theta = theta.copy()
        previous_obj = float(value)

    initial_obj, _initial_grad, initial_parts = formal_objective_and_gradient(theta0, inputs, fold_id=fold_id, split=split)
    result = minimize(
        fun,
        theta0,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={"maxiter": int(maxiter), "maxls": 20, "ftol": 1e-12, "gtol": 1e-8},
    )
    theta_final = np.asarray(result.x, dtype=float)
    final_obj, final_grad, final_parts = formal_objective_and_gradient(theta_final, inputs, fold_id=fold_id, split=split)
    metrics = evaluate_parameters(theta_final, inputs, fold_id=fold_id, split=split)
    np.save(output_dir / "parameters.npy", theta_final)
    summary = {
        "optimization_status": "passed" if bool(result.success) else "failed_optimizer_not_converged",
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "initial_objective": float(initial_obj),
        "final_objective": float(final_obj),
        "relative_objective_reduction": float((initial_obj - final_obj) / max(abs(initial_obj), 1e-30)),
        "initial_gradient_norm": initial_parts["gradient_norm"],
        "final_gradient_norm": final_parts["gradient_norm"],
        "final_gradient_rms": final_parts["gradient_rms"],
        "parameter_count": int(theta_final.size),
        "theta_initial": theta0.tolist(),
        "theta_final": theta_final.tolist(),
        "parameter_delta": (theta_final - theta0).tolist(),
        "metrics": metrics,
        "history": history,
        "source_level_optimization": True,
        "synthetic_or_placeholder_results_generated": False,
    }
    write_json(output_dir / "fit_summary.json", summary)
    return summary
