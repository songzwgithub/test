# Code Layout

This repository currently keeps two kinds of code on purpose:

1. The maintained release package in `src/hengshui_insar/`.
2. Recovered historical workflow scripts used to reproduce the original L01028 processing chain.

The recovered scripts are not clutter to delete blindly. They preserve the procedural path that produced the accepted results. The maintained package is the preferred interface for verification, source-level recomputation, and future cleanup.

## Maintained Package

`src/hengshui_insar/` is the formal package installed by `pyproject.toml` and exposed through the `hengshui-insar` CLI.

Key modules:

- `cli.py`: command-line entrypoint.
- `config.py`: strict loading of `configs/l01028_release_v1.yaml`.
- `constants.py`: frozen L01028 release constants and expected scientific metrics.
- `source_recompute.py`: source-level recomputation from canonical HDF5, masks, RBF design, and saved parameters.
- `cross_validation.py`: formal fold RMSE and final refit source-level checks.
- `storage.py`: confined seasonal storage source-level recomputation.
- `audit.py`: release acceptance gate.
- `products.py` and `qa.py`: product and spatial QA checks.
- `bounded_model.py`, `harmonics.py`, and `rbf.py`: core math helpers.

## Canonical Inputs

Canonical inputs are under:

`outputs/canonical_inputs/L01028_bounded_memmaps_v1/`

This directory contains the authoritative harmonic HDF5 cache, common mask, fold map, selected RBF design, and RBF transform. Current source-level recomputation reads these paths, not the old `outputs/reference_frames/` or `outputs/aquifer_model_revision/` paths.

## Accepted Release

Accepted products and summaries are under:

`outputs/releases/L01028_v1/`

The package recomputes metrics from this release's saved parameters and canonical inputs. It should not require restored historical output folders.

## Recovered Workflows

The following directories and root modules were restored for reproducibility:

- `recovered_workflows/scripts/`: historical one-off and formal workflow scripts.
- `recovered_workflows/pipelines/`: historical high-level batch entrypoints.
- `recovered_workflows/plotting/`: historical plotting helpers.
- `recovered_workflows/legacy/`: older V2/unbounded reference material.
- root modules such as `recovered_workflows/run_pipeline.py`, `recovered_workflows/storage_inversion.py`, `recovered_workflows/bounded_ske_v2.py`, `recovered_workflows/profiled_stage_a.py`, and processing modules.

These files are intentionally retained because several scripts import root modules directly. Do not move them without first replacing imports and proving the full source-level checks still pass.

## Cleanup Rule

Future cleanup should migrate one workflow at a time into `src/hengshui_insar/`, then run:

```bash
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m pytest -q
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli audit --config configs/l01028_release_v1.yaml
```

Only after those checks pass should the migrated historical script be deprecated or removed.
