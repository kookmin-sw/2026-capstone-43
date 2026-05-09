#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEM_VARIANT="${1:-conv64_64_out8}"
"${SCRIPT_DIR}/run_stage14_subset_experiment.sh" \
  "subset_foa_${STEM_VARIANT}_cls_slow" \
  --foa_stem_type foa_native \
  --foa_stem_variant "${STEM_VARIANT}" \
  --azimuth_head_mode full360_classification \
  --unfreeze_last_n_blocks 2 \
  --freeze_patch_embed \
  --recipe_name slow_recipe \
  --lr_stem 1e-4 \
  --lr_adapter 1e-4 \
  --lr_heads 5e-4 \
  --lr_transformer 1e-5 \
  --scheduler cosine \
  --warmup_epochs 3 \
  --epochs 30 \
  --batch_size 16
