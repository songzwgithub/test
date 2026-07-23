# Reproducibility Entrypoints

Use the maintained CLI first. It reads the formal config, canonical inputs, and accepted release parameters.

## Fast Integrity Check

```bash
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli verify --config configs/l01028_release_v1.yaml
```

Checks the authoritative harmonic cache and common mask hashes.

## Source-Level Formal CV Recalculation

```bash
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli cv --config configs/l01028_release_v1.yaml
```

Streams the canonical HDF5, applies the fold map, rebuilds RBF basis values, loads each fold's saved parameters, and recomputes validation RMSE.

## Source-Level Final Refit Recalculation

```bash
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli invert --config configs/l01028_release_v1.yaml
```

This command does not re-optimize by default. It recomputes final full-data metrics from the saved final parameters and canonical inputs.

## Source-Level Storage Recalculation

```bash
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli storage --config configs/l01028_release_v1.yaml
```

Reintegrates confined elastic seasonal storage from `Ske.tif`, canonical `hc`, common mask, and geodesic pixel area.

## Full Release Audit

```bash
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli audit --config configs/l01028_release_v1.yaml
```

Runs tests/build/install checks plus source-level CV, final refit, storage, product, and QA gates. This can take several minutes because it reads real data.

## Historical Full-Workflow Scripts

Use these only when reproducing the original development pipeline or investigating provenance:

- `recovered_workflows/scripts/run_L01028_bounded_pipeline.py`
- `recovered_workflows/scripts/run_v2_g0_formal_cv.py`
- `recovered_workflows/scripts/run_L01028_storage_volume.py`
- `recovered_workflows/scripts/rebuild_L01028_phase4_harmonic_cache.py`
- `recovered_workflows/pipelines/run_bounded_inversion.py`
- `recovered_workflows/pipelines/run_seasonal_storage.py`
- `recovered_workflows/run_pipeline.py`

Many historical scripts still reference pre-cleanup output paths. Prefer the maintained CLI unless you are intentionally reconstructing historical intermediate folders.
