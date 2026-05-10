#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_stage5_experiment.sh" \
  foa_patch_last6 \
  --foa_stem_type foa_native \
  --unfreeze_last_n_blocks 6 \
  --unfreeze_patch_embed \
  --recipe_name default_recipe \
  --epochs 30 \
  --batch_size 16
