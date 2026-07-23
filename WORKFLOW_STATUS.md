# Workflow Status

The repository is back to a single-line release layout.

- Maintained implementation: `src/hengshui_insar/`
- Human command wrappers: `commands/`
- Frozen config: `configs/l01028_release_v1.yaml`
- Canonical inputs: `outputs/canonical_inputs/L01028_bounded_memmaps_v1/`
- Accepted release: `outputs/releases/L01028_v1/`

Recovered historical workflow code is not active release code. A local non-release copy was moved to `/tmp/hengshui_recovery_external/`.

## Current Recommended Commands

```bash
./commands/verify_release.sh
./commands/source_cv.sh
./commands/source_final_refit.sh
./commands/source_storage.sh
./commands/full_audit.sh
```
