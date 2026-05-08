#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_stage6_experiment.sh" \
  foa_full_tuning_front \
  --foa_stem_type foa_native \
  --full_tuning \
  --recipe_name full_tuning_test \
  --lr_stem 3e-4 \
  --lr_adapter 3e-4 \
  --lr_heads 5e-4 \
  --lr_transformer 1e-5 \
  --scheduler cosine \
  --warmup_epochs 3 \
  --epochs 20 \
  --batch_size 16
