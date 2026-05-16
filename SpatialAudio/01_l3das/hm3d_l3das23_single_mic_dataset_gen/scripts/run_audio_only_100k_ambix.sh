#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/hm3d_audio_only_100k_ambix.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/media/yu/Extreme SSD/hm3d_audio_only_100k_ambix}"
CONDA_SH="${CONDA_SH:-/home/yu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-habitat}"
SANITY_TRIALS="${SANITY_TRIALS:-1}"

mkdir -p "$DATASET_ROOT"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

if [[ "${SKIP_FOA_SANITY:-0}" != "1" ]]; then
  python scripts/check_foa_remap_sanity.py \
    --config "$CONFIG_PATH" \
    --trials "$SANITY_TRIALS"
fi

hm3d-l3das23-generate dump-config \
  --config "$CONFIG_PATH" \
  --output "$DATASET_ROOT/resolved_config.yaml"

hm3d-l3das23-generate build-splits \
  --config "$CONFIG_PATH" \
  --mode full

hm3d-l3das23-generate generate \
  --config "$CONFIG_PATH" \
  --mode full

hm3d-l3das23-generate qc \
  --dataset-root "$DATASET_ROOT" | tee "$DATASET_ROOT/qc_final.txt"
