#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_stage4_experiment.sh" \
  foa_last4_longer_seed3407 \
  --foa_stem_type foa_native \
  --unfreeze_last_n_blocks 4 \
  --recipe_name longer_run \
  --seed 3407 \
  --epochs 60 \
  --batch_size 16
