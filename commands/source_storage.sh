#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src /home/s/miniconda3/envs/insar/bin/python -m hengshui_insar.cli storage --config configs/l01028_release_v1.yaml
