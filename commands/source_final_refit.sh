#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src "${PYTHON:-python}" -m hengshui_insar.cli invert --config configs/l01028_release_v1.yaml
