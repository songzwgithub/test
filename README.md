# Hengshui InSAR-Groundwater L01028 Bounded Inversion

This repository contains the current formal L01028 bounded two-aquifer inversion products for the Hengshui InSAR-groundwater study.

## Current Formal Result

- Reference frame: `L01028_500m_fixed_quality_median_v1`
- Formal manifest SHA256: `f7f41d15db0a83641dc72414814988626e178c2a4c05b091f73c57ad2c2a0cc1`
- Authoritative Phase-4 harmonic cache SHA256: `3f4f714b5e10fe3dcd5a9e91a29de27e0157858137e76afddb12b2cd0fa6dce8`
- Common mask SHA256: `ff761a316e0a89a9121c439967df418f14585ae420f281d43671ebaf4740bd1f`
- Accepted bounded model: bounded Ske, G0 no geology, shared confined lag, fixed weakly identifiable unconfined lag.
- Seasonal storage product: confined elastic seasonal storage anomaly only.

## Formal Entrypoints

- Check bounded inversion: `python pipelines/run_bounded_inversion.py --stage check-only`
- Seasonal storage: `python pipelines/run_seasonal_storage.py`
- Publication figures: `python pipelines/build_publication_figures.py`
- Final audit: `python pipelines/run_final_audit.py`
- Tests: `/home/s/miniconda3/envs/insar/bin/python -m pytest tests -q`

The legacy root `run_pipeline.py` is disabled. Historical V2 code is preserved under `legacy/v2_unbounded/` for provenance only.

## Data Note

Large input rasters, HDF5 caches, and memmaps are expected in `outputs/` and are not portable source code assets. The canonical memmap view is under `outputs/canonical_inputs/L01028_bounded_memmaps_v1/`.

## Scientific Limits

This is not total groundwater storage. It does not provide daily storage, unconfined storage, or independent external validation. The storage uncertainty is a 95% structural amplitude envelope, not a full probabilistic 95% confidence or credible interval.
