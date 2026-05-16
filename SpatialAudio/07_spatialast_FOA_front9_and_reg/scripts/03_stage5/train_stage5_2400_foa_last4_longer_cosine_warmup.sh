#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_stage5_experiment.sh" \
  foa_last4_longer_cosine_warmup \
  --foa_stem_type foa_native \
  --unfreeze_last_n_blocks 4 \
  --recipe_name longer_cosine_warmup \
  --scheduler cosine \
  --warmup_epochs 3 \
  --epochs 60 \
  --batch_size 16
