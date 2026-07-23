# Recovered Workflows

This directory contains restored historical workflow code. It is intentionally
separate from the maintained release package so the repository root stays
readable.

Use `commands/` or `hengshui-insar` for current release verification. Use files
here only when reconstructing the original workflow or investigating provenance.

## Contents

- `scripts/`: recovered one-off and formal workflow scripts.
- `pipelines/`: recovered high-level batch wrappers.
- `plotting/`: recovered historical plotting helpers.
- `legacy/`: older provenance code.
- root-level modules in this directory: dependencies imported by recovered
  scripts.
- `src/hengshui_l01028/`: recovered older support package.

`outputs` and `data` are linked back to the project root when possible so
historical scripts can still resolve their old relative paths.
