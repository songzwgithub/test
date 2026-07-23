# Scripts

This directory contains recovered historical workflow scripts. They are kept for provenance and full-flow reconstruction. The maintained entrypoint for routine verification is the `hengshui-insar` CLI.

## Reference And Cache

- `build_L01028_reference_frame.py`
- `rebuild_L01028_phase4_harmonic_cache.py`
- `rebuild_L01028_authoritative_products.py`
- `accept_L01028_harmonic_cache.py`
- `audit_L01028_products_and_freeze_manifest.py`

## Formal Inversion And CV

- `run_L01028_bounded_pipeline.py`
- `run_v2_g0_formal_cv.py`
- `run_v2_m0_formal_cv.py`
- `run_v2_geology_formal_cv.py`
- `run_L01028_formal_inversion_pipeline.py`
- `run_L01028_fold0_confirmation.py`
- `run_L01028_fold0_resume_and_finalize.py`
- `run_formal_g0_fold1.py`

## Stage Diagnostics

- `run_g0_fold0_staged.py`
- `run_stage_b_fixed_lagu.py`
- `run_stage_c_fixed_lagu.py`
- `run_bounded_ske_v2_development.py`
- `audit_stage_b_lambda_effect.py`
- `audit_and_reduce_rbf_basis.py`
- `build_orthogonal_rbf_basis.py`
- `generate_adaptive_rbf_centers.py`

## Geology And Aquifer Model Audits

- `build_geology_rasters.py`
- `validate_geology_rasters.py`
- `audit_geological_design_matrix.py`
- `audit_geological_shapefiles.py`
- `check_clay_thickness_semantics.py`
- `compare_geology_fix_results.py`
- `diagnose_polygon_overlaps.py`

## Forensics And Protocol Repair

- `fold4_forensic_readonly_audit.py`
- `fold4_frozen_parameter_forensic_replay.py`
- `fold4_offline_prediction_decomposition.py`
- `audit_fold2_mask_partition.py`
- `audit_fold4_mask_partition.py`
- `fix_formal_validation_protocol.py`
- `formal_protocol_checkpoint_and_dry_run.py`
- `resume_formal_protocol_dry_run.py`
- `mark_formal_dry_run_incomplete.py`
- `invalidate_old_rbf_artifact_diagnostics.py`

## Storage, Figures, And Summaries

- `run_L01028_storage_volume.py`
- `audit_L01028_storage_volume.py`
- `plot_L01028_storage_volume.py`
- `plot_rbf_active_center_coverage.py`
- `check_figure_source_fields.py`
- `check_insar_overview_products.py`
- `summarize_formal_g0_fourfold.py`
- `finalize_model_selection.py`

## Caution

Some scripts still point at historical output paths that were intentionally not restored. Prefer `hengshui-insar` commands for current release verification.
