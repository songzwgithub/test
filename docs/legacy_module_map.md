# Legacy Module Map

This file records why recovered root modules are still present.

## Maintained Equivalents

- `src/hengshui_insar/source_recompute.py` now covers source-level CV, final refit, and storage recomputation from canonical inputs.
- `src/hengshui_insar/harmonics.py` covers annual harmonic value, phase, and positive-lag rotation.
- `src/hengshui_insar/bounded_model.py` covers bounded Ske math helpers.
- `src/hengshui_insar/rbf.py` covers Gaussian RBF and orthogonal transform helpers.

## Historical Root Modules

These root modules were recovered because old workflow scripts import them directly:

- `recovered_workflows/bounded_ske_v2.py`
- `recovered_workflows/profiled_stage_a.py`
- `recovered_workflows/storage_inversion.py`
- `recovered_workflows/spatial_refit_validation.py`
- `recovered_workflows/spatial_utils.py`
- `recovered_workflows/insar_processing.py`
- `recovered_workflows/groundwater_processing.py`
- `recovered_workflows/geology_preprocessing.py`
- `recovered_workflows/geological_prior.py`
- `recovered_workflows/m1_inversion.py`
- `recovered_workflows/latent_head_model.py`
- `recovered_workflows/lag_analysis.py`
- `recovered_workflows/temporal_analysis.py`
- `recovered_workflows/uncertainty.py`
- `recovered_workflows/validation.py`
- `recovered_workflows/revision_products.py`
- `recovered_workflows/result_audit.py`
- `recovered_workflows/visualize_results.py`
- `recovered_workflows/generate_insar_overview.py`
- `recovered_workflows/io_utils.py`
- `recovered_workflows/bulletin_processing.py`

## Migration Rule

Do not delete a historical root module just because there is a maintained equivalent. First:

1. Update historical scripts or replace the workflow in `src/hengshui_insar/`.
2. Run compile checks over `src`, `tests`, `recovered_workflows`, and root release files.
3. Run `pytest -q`.
4. Run `hengshui-insar audit --config configs/l01028_release_v1.yaml`.

Only then mark a module deprecated or remove it.
