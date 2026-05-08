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
PYTHON_BIN="/home/yu/miniconda3/envs/spatial-ast/bin/python"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs_stage13"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
MANIFEST_ROOT="${PROJECT_ROOT}/manifests_stage13"
TRAIN_MANIFEST="${MANIFEST_ROOT}/train_stage13_subset_600.json"
VAL_MANIFEST="${MANIFEST_ROOT}/val_stage13_subset_150.json"

mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${TRAIN_MANIFEST}" ] || [ ! -f "${VAL_MANIFEST}" ]; then
  echo "[stage13] subset manifests missing, building them first"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/build_stage13_subset.py"
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/train.py" \
  --device cuda:0 \
  --audio_path_root /home/yu/Project_git/01_FOV_LOS \
  --train_json "${TRAIN_MANIFEST}" \
  --val_json "${VAL_MANIFEST}" \
  --limit_train_samples 0 \
  --limit_val_samples 0 \
  --num_classes 0 \
  --num_workers 0 \
  --class_loss_weight 0.0 \
  --distance_loss_weight 0.0 \
  --azimuth_loss_weight 2.0 \
  --elevation_loss_weight 2.0 \
  --vector_loss_weight 0.5 \
  --select_best_by angular_error \
  --output_dir "${OUTPUT_DIR}" \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/train.log"
