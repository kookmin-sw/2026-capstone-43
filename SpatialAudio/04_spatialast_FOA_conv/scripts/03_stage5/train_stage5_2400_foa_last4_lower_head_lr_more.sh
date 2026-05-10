#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_stage5_experiment.sh" \
  foa_last4_lower_head_lr_more \
  --foa_stem_type foa_native \
  --unfreeze_last_n_blocks 4 \
  --recipe_name lower_head_lr_more \
  --lr_heads 1e-4 \
  --lr_stem 1e-3 \
  --lr_adapter 1e-3 \
  --lr_transformer 1e-5 \
  --scheduler none \
  --warmup_epochs 0 \
  --epochs 30 \
  --batch_size 16
