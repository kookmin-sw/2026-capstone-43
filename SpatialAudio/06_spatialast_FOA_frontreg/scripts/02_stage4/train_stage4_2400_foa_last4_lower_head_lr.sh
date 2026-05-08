#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_stage4_experiment.sh" \
  foa_last4_lower_head_lr \
  --foa_stem_type foa_native \
  --unfreeze_last_n_blocks 4 \
  --recipe_name lower_head_lr \
  --lr_heads 5e-4 \
  --epochs 30 \
  --batch_size 16
