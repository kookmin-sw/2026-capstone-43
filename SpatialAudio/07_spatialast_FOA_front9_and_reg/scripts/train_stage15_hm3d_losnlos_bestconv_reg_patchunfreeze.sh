#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yu/miniconda3/envs/spatial-ast/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/home/yu/Project_git/01_dataset/hm3d_losnlos_100k_balanced}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${PROJECT_ROOT}/manifests_stage15/hm3d_losnlos_100k_balanced_front9}"
STEM_VARIANT="${1:-conv64_64_out8}"
UNFREEZE_LAST_N_BLOCKS="${UNFREEZE_LAST_N_BLOCKS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RUN_NAME="${RUN_NAME:-hm3d_losnlos_front9_${STEM_VARIANT}_reg_last${UNFREEZE_LAST_N_BLOCKS}_patchunfreeze}"

if [ "$#" -gt 0 ]; then
  shift
fi

TRAIN_JSON="${MANIFEST_ROOT}/train.json"
VAL_JSON="${MANIFEST_ROOT}/val.json"

if [ ! -f "${TRAIN_JSON}" ] || [ ! -f "${VAL_JSON}" ]; then
  echo "[stage15-hm3d] manifests missing, building them first"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/build_hm3d_losnlos_front9_manifests.py" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${MANIFEST_ROOT}"
  fi

DATASET_ROOT="${DATASET_ROOT}" \
TRAIN_JSON="${TRAIN_JSON}" \
VAL_JSON="${VAL_JSON}" \
LIMIT_TRAIN_SAMPLES="${LIMIT_TRAIN_SAMPLES:-0}" \
LIMIT_VAL_SAMPLES="${LIMIT_VAL_SAMPLES:-0}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
"${SCRIPT_DIR}/run_stage15_full_experiment.sh" \
  "${RUN_NAME}" \
  --foa_stem_type foa_native \
  --foa_stem_variant "${STEM_VARIANT}" \
  --azimuth_head_mode front_regression \
  --azimuth_regression_range 45.0 \
  --azimuth_regression_loss smoothl1 \
  --unfreeze_last_n_blocks "${UNFREEZE_LAST_N_BLOCKS}" \
  --unfreeze_patch_embed \
  --recipe_name patchunfreeze_slow_recipe \
  --lr_stem 1e-4 \
  --lr_adapter 1e-4 \
  --lr_heads 5e-4 \
  --lr_patch_embed 1e-5 \
  --lr_transformer 1e-5 \
  --scheduler cosine \
  --warmup_epochs 3 \
  --epochs 30 \
  --batch_size "${BATCH_SIZE}" \
  "$@"
