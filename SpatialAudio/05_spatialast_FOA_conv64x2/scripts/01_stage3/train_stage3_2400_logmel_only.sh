#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_stage3_experiment.sh" \
  train2400_logmel_only \
  --foa_stem_type logmel_only \
  --limit_train_samples 2400 \
  --batch_size 8 \
  --epochs 30
