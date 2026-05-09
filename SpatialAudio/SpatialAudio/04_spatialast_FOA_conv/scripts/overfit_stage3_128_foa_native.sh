#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_stage3_experiment.sh" \
  overfit_128_foa_native \
  --foa_stem_type foa_native \
  --debug_overfit_subset_size 128 \
  --batch_size 8 \
  --epochs 120
