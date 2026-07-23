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

## Formal Optimization

```bash
./commands/run_inversion_optimization.sh --maxiter 300
```

This executes the maintained bounded objective and analytical gradient in
`src/hengshui_insar/optimization.py`. The source-final-refit command above uses
`--recompute-only` and does not optimize.

## Historical Workflows

Historical workflow scripts are outside the active release tree. A local external
copy may exist at `/tmp/hengshui_recovery_external/` for emergency inspection.
