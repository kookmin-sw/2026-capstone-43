#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <run_name> [train.py args ...]"
  exit 1
fi

RUN_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yu/miniconda3/envs/spatial-ast/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_ROOT="${DATASET_ROOT:-/home/yu/Project_git/01_FOV_LOS}"
TRAIN_JSON="${TRAIN_JSON:-${DATASET_ROOT}/manifests/train.json}"
VAL_JSON="${VAL_JSON:-${DATASET_ROOT}/manifests/val.json}"
LIMIT_TRAIN_SAMPLES="${LIMIT_TRAIN_SAMPLES:-2400}"
LIMIT_VAL_SAMPLES="${LIMIT_VAL_SAMPLES:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs_stage15"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"

if [ "${OVERWRITE_OUTPUT_DIR}" = "1" ] && [ -d "${OUTPUT_DIR}" ]; then
  rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/train.py" \
  --device "${DEVICE}" \
  --audio_path_root "${DATASET_ROOT}" \
  --train_json "${TRAIN_JSON}" \
  --val_json "${VAL_JSON}" \
  --limit_train_samples "${LIMIT_TRAIN_SAMPLES}" \
  --limit_val_samples "${LIMIT_VAL_SAMPLES}" \
  --num_classes 0 \
  --num_workers "${NUM_WORKERS}" \
  --class_loss_weight 0.0 \
  --distance_loss_weight 0.0 \
  --azimuth_loss_weight 2.0 \
  --elevation_loss_weight 2.0 \
  --vector_loss_weight 0.5 \
  --select_best_by angular_error \
  --output_dir "${OUTPUT_DIR}" \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/train.log"
