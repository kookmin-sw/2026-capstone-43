#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_stage3_experiment.sh" \
  overfit_16_logmel_only \
  --foa_stem_type logmel_only \
  --debug_overfit_subset_size 16 \
  --batch_size 4 \
  --epochs 80
