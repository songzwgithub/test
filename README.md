# Hengshui InSAR-Groundwater L01028 Bounded Inversion

This repository contains the formal L01028 bounded two-aquifer inversion release for the Hengshui InSAR-groundwater study, plus recovered historical workflow code needed for provenance and full-flow reconstruction.

## Start Here

Use the maintained package and CLI for current release verification:

```bash
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli verify --config configs/l01028_release_v1.yaml
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli cv --config configs/l01028_release_v1.yaml
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli invert --config configs/l01028_release_v1.yaml
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli storage --config configs/l01028_release_v1.yaml
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli audit --config configs/l01028_release_v1.yaml
```

The `audit` command can take several minutes because it performs source-level recomputation from real canonical inputs.

Equivalent shell wrappers live in `commands/` for day-to-day use.

## Current Formal Result

- Reference frame: `L01028_500m_fixed_quality_median_v1`
- Formal manifest SHA256: `f7f41d15db0a83641dc72414814988626e178c2a4c05b091f73c57ad2c2a0cc1`
- Authoritative Phase-4 harmonic cache SHA256: `3f4f714b5e10fe3dcd5a9e91a29de27e0157858137e76afddb12b2cd0fa6dce8`
- Common mask SHA256: `ff761a316e0a89a9121c439967df418f14585ae420f281d43671ebaf4740bd1f`
- Accepted bounded model: bounded Ske, G0 no geology, shared confined lag, fixed weakly identifiable unconfined lag.
- Seasonal storage product: confined elastic seasonal storage anomaly only.

## Directory Layout

- `src/hengshui_insar/`: maintained release package and CLI.
- `commands/`: human-facing shell wrappers for the maintained CLI.
- `configs/l01028_release_v1.yaml`: single formal release config.
- `outputs/canonical_inputs/L01028_bounded_memmaps_v1/`: canonical real-data inputs used by source-level recomputation.
- `outputs/releases/L01028_v1/`: accepted release products, parameters, tables, figures, and audit artifacts.
- `recovered_workflows/`: restored historical scripts, pipelines, plotting helpers, legacy code, and old root modules.

More detail:

- `docs/code_layout.md`
- `docs/reproducibility_entrypoints.md`
- `docs/legacy_module_map.md`

## Scientific Limits

This release does not claim total groundwater storage. It does not provide unconfined storage or an independently validated daily storage field. The storage uncertainty is a 95% structural amplitude envelope, not a full probabilistic 95% confidence or credible interval.

## Cleanup Policy

Do not bulk-delete recovered scripts or root modules. Migrate one workflow at a time into `src/hengshui_insar/`, then rerun tests and `hengshui-insar audit`. Only after those checks pass should a historical file be deprecated or removed.
