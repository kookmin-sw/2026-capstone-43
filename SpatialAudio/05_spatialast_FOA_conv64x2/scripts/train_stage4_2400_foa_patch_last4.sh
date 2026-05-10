#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_stage4_experiment.sh" \
  foa_patch_last4 \
  --foa_stem_type foa_native \
  --unfreeze_last_n_blocks 4 \
  --unfreeze_patch_embed \
  --recipe_name default_recipe \
  --epochs 30 \
  --batch_size 16
