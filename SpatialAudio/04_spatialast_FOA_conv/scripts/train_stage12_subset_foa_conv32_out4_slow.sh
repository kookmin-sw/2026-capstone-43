#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_stage12_subset_experiment.sh" \
  subset_foa_conv32_out4_slow \
  --foa_stem_type foa_native \
  --foa_stem_variant conv32_out4 \
  --unfreeze_last_n_blocks 2 \
  --freeze_patch_embed \
  --recipe_name stage12_subset_slow \
  --lr_stem 1e-4 \
  --lr_adapter 1e-4 \
  --lr_heads 5e-4 \
  --lr_transformer 1e-5 \
  --scheduler cosine \
  --warmup_epochs 3 \
  --epochs 30 \
  --batch_size 16
