#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_stage14_full_experiment.sh" \
  full_foa_baseline_cls_slow \
  --foa_stem_type foa_native \
  --foa_stem_variant baseline \
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
