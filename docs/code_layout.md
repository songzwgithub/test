# Code Layout

The active repository is a single-line release tree.

## Active Release Code

`src/hengshui_insar/` is the maintained package installed by `pyproject.toml` and exposed through `hengshui-insar`.

Key modules:

- `cli.py`: command-line entrypoint.
- `config.py`: strict frozen config validation.
- `constants.py`: frozen release constants and expected scientific metrics.
- `source_recompute.py`: source-level recomputation from canonical HDF5, masks, RBF design, and saved parameters.
- `optimization.py`: formal bounded objective, analytical gradient, and optimizer.
- `cross_validation.py`: formal CV and final refit source-level audits.
- `storage.py`: confined storage source-level recomputation.
- `audit.py`: release acceptance gate.

## Human Entrypoints

`commands/` contains thin shell wrappers around the maintained CLI.

## Data And Results

- `outputs/canonical_inputs/L01028_bounded_memmaps_v1/`: canonical inputs.
- `outputs/releases/L01028_v1/`: accepted release.

## Historical Code

Historical workflow code is not active release code. A local external copy may exist at `/tmp/hengshui_recovery_external/` for emergency inspection, but it is outside the release tree.
