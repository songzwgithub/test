"""Optimization bookkeeping helpers."""

from __future__ import annotations

from dataclasses import dataclass


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
