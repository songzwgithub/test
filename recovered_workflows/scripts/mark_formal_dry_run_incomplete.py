#!/usr/bin/env python
"""Mark an interrupted formal-protocol dry run as incomplete and non-passing."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path("outputs/aquifer_model_revision")
    dry = root / "model_compare/G0_no_geology_L0_shared/fold_00/formal_protocol_dry_run"
    dry.mkdir(parents=True, exist_ok=True)
    hist = dry / "training_only_optimizer_history.csv"
    accepted = 0
    train_rmse = None
    param_hash = None
    checkpoint_hash = None
    if hist.exists():
        df = pd.read_csv(hist)
        accepted = int(len(df))
        if accepted:
            last = df.iloc[-1]
            train_rmse = float(last["training_rmse_mm"])
            param_hash = str(last["parameter_hash"])
            checkpoint_hash = str(last["checkpoint_hash"])
    selection = json.loads(
        (root / "model_compare/G0_no_geology_L0_shared/fold_00/stage_C/development_early_stopping_selection.json").read_text()
    )
    audit = {
        "protocol_dry_run_on_development_fold": True,
        "dry_run_execution_status": "interrupted_resource_control_after_progress_check",
        "checkpoint_alignment_passed": accepted > 0,
        "selected_iteration_budget": int(selection["selected_iteration_budget"]),
        "accepted_iterations_completed": accepted,
        "stage_A_training_only": (dry / "stage_A_training_only_result.json").exists(),
        "stage_B_training_only": (dry / "stage_B_training_only_result.json").exists(),
        "stage_C_training_only": True,
        "outer_validation_access_count_during_training": 0,
        "outer_validation_access_count_final": 0,
        "final_parameter_hash": param_hash,
        "last_checkpoint_hash": checkpoint_hash,
        "training_rmse": train_rmse,
        "single_final_validation_rmse": None,
        "physical_audit": {"physical_status": "not_evaluated_full_budget_not_reached"},
        "artifact_audit": {"artifact_status": "not_evaluated_full_budget_not_reached"},
        "formal_fit_status": "formal_fit_incomplete_fixed_budget_not_reached",
        "formal_protocol_passed": False,
        "fold1_pilot_allowed": False,
        "reason": "Full fixed-budget dry run was stopped after progress check to control runtime; no final outer validation was accessed.",
    }
    (dry / "formal_fit_status.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    (root / "formal_protocol_dry_run_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    (dry / "outer_validation_access_audit.json").write_text(
        json.dumps(
            {
                "outer_validation_access_count_during_training": 0,
                "outer_validation_access_count_final": 0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    status_path = root / "aquifer_model_revision_status.json"
    status = json.loads(status_path.read_text())
    status.update(
        {
            "allow_continue_g0_fold1_pilot": False,
            "allow_continue_g0_other_folds": False,
            "allow_continue_g0_fold2_fold4": False,
            "allow_continue_g1_g2_g3": False,
            "phase4_restart_allowed": False,
            "selected_model_config": "not_generated",
            "formal_protocol_dry_run": "incomplete_fixed_budget_not_reached",
        }
    )
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
