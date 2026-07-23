# Hengshui InSAR-Groundwater L01028 Bounded Inversion

This repository contains the formal single-line L01028 bounded two-aquifer inversion release for the Hengshui InSAR-groundwater study.

## Start Here

Use `commands/` or the maintained `hengshui-insar` CLI. Historical workflow scripts are not active release code.

```bash
./commands/verify_release.sh
./commands/source_cv.sh
./commands/source_final_refit.sh
./commands/source_storage.sh
./commands/full_audit.sh
```

The full audit can take several minutes because it performs source-level recomputation from real canonical inputs.

## Current Formal Result

- Reference frame: `L01028_500m_fixed_quality_median_v1`
- Formal manifest SHA256: `f7f41d15db0a83641dc72414814988626e178c2a4c05b091f73c57ad2c2a0cc1`
- Authoritative Phase-4 harmonic cache SHA256: `3f4f714b5e10fe3dcd5a9e91a29de27e0157858137e76afddb12b2cd0fa6dce8`
- Common mask SHA256: `ff761a316e0a89a9121c439967df418f14585ae420f281d43671ebaf4740bd1f`
- Accepted bounded model: bounded Ske, G0 no geology, shared confined lag, fixed weakly identifiable unconfined lag.
- Seasonal storage product: confined elastic seasonal storage anomaly only.

## Directory Layout

- `src/hengshui_insar/`: maintained release package and CLI.
- `commands/`: human-facing shell wrappers for maintained CLI commands.
- `configs/l01028_release_v1.yaml`: single frozen release config.
- `outputs/canonical_inputs/L01028_bounded_memmaps_v1/`: canonical real-data inputs.
- `outputs/releases/L01028_v1/`: accepted release products, parameters, tables, figures, and audit artifacts.
- `tests/`: release tests.

Historical recovered workflows were moved out of the active release tree to `/tmp/hengshui_recovery_external/` on this workstation. They are not part of the formal package.

## Scientific Limits

This release does not claim total groundwater storage. It does not provide unconfined storage or an independently validated daily storage field. The storage uncertainty is a 95% structural amplitude envelope, not a full probabilistic 95% confidence or credible interval.
