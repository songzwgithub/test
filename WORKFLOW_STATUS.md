# Workflow Status

The repository is intentionally in a hybrid restoration state:

- The maintained release implementation is `src/hengshui_insar/`.
- Historical workflow code has been restored to preserve full-flow provenance.
- Source-level reproduction of accepted L01028 results is available through the `hengshui-insar` CLI.

## Current Recommended Commands

```bash
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli verify --config configs/l01028_release_v1.yaml
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli cv --config configs/l01028_release_v1.yaml
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli invert --config configs/l01028_release_v1.yaml
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli storage --config configs/l01028_release_v1.yaml
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli audit --config configs/l01028_release_v1.yaml
```

## Last Verified Restoration

`release/L01028_reproducible_flow_restoration_acceptance.json` records the restoration acceptance. It confirms:

- restored historical workflow sources are present;
- formal CV is source-level recomputed;
- final refit is source-level recomputed;
- confined seasonal storage is source-level recomputed;
- tests, wheel build, and clean venv install pass.
