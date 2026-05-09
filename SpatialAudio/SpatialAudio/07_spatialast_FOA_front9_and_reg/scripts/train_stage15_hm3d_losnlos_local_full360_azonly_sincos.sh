#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/yu/miniconda3/envs/spatial-ast/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_ROOT="${DATASET_ROOT:-/home/yu/Project_git/01_dataset/hm3d_losnlos_100k_balanced}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${PROJECT_ROOT}/manifests_stage15/hm3d_losnlos_100k_balanced_local_full360_glos}"
TRAIN_JSON="${TRAIN_JSON:-${MANIFEST_ROOT}/train.json}"
VAL_JSON="${VAL_JSON:-${MANIFEST_ROOT}/val.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs_stage15}"

STEM_VARIANT="${STEM_VARIANT:-conv64_64_out8}"
UNFREEZE_LAST_N_BLOCKS="${UNFREEZE_LAST_N_BLOCKS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-30}"
SEED="${SEED:-1337}"
RUN_NAME="${RUN_NAME:-local_full360_glos_sincos_azonly_${STEM_VARIANT}_last${UNFREEZE_LAST_N_BLOCKS}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

LIMIT_TRAIN_SAMPLES="${LIMIT_TRAIN_SAMPLES:-0}"
LIMIT_VAL_SAMPLES="${LIMIT_VAL_SAMPLES:-0}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-0}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"
AUDIO_NORMALIZE="${AUDIO_NORMALIZE:-1}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

LR_STEM="${LR_STEM:-1e-4}"
LR_ADAPTER="${LR_ADAPTER:-1e-4}"
LR_HEADS="${LR_HEADS:-5e-4}"
LR_TRANSFORMER="${LR_TRANSFORMER:-1e-5}"
LR_PATCH_EMBED="${LR_PATCH_EMBED:-1e-5}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
AZIMUTH_LOSS_WEIGHT="${AZIMUTH_LOSS_WEIGHT:-2.0}"

if [ ! -f "${TRAIN_JSON}" ] || [ ! -f "${VAL_JSON}" ]; then
  echo "[stage15-local-azonly] local full360 manifests missing, building them first"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/build_hm3d_losnlos_local_full360_manifests.py" \
    --dataset_root "${DATASET_ROOT}"
fi

if [ -d "${OUTPUT_DIR}" ] && [ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]; then
  if [ "${OVERWRITE_OUTPUT_DIR}" = "1" ]; then
    rm -rf "${OUTPUT_DIR}"
  else
    echo "[stage15-local-azonly] output dir already exists and is not empty: ${OUTPUT_DIR}" >&2
    echo "[stage15-local-azonly] set OVERWRITE_OUTPUT_DIR=1 or choose a new RUN_NAME" >&2
    exit 1
  fi
fi
mkdir -p "${OUTPUT_DIR}"

args=(
  --device "${DEVICE}"
  --audio_path_root "${DATASET_ROOT}"
  --train_json "${TRAIN_JSON}"
  --val_json "${VAL_JSON}"
  --limit_train_samples "${LIMIT_TRAIN_SAMPLES}"
  --limit_val_samples "${LIMIT_VAL_SAMPLES}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --max_val_steps "${MAX_VAL_STEPS}"
  --num_classes 0
  --num_workers "${NUM_WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --sample_rate 32000
  --clip_seconds 10
  --foa_stem_type foa_native
  --foa_stem_variant "${STEM_VARIANT}"
  --patch_in_from_stem
  --azimuth_head_mode full360_sincos_regression
  --azimuth_regression_loss smoothl1
  --unfreeze_last_n_blocks "${UNFREEZE_LAST_N_BLOCKS}"
  --unfreeze_patch_embed
  --lr_stem "${LR_STEM}"
  --lr_adapter "${LR_ADAPTER}"
  --lr_heads "${LR_HEADS}"
  --lr_transformer "${LR_TRANSFORMER}"
  --lr_patch_embed "${LR_PATCH_EMBED}"
  --scheduler "${SCHEDULER}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --recipe_name local_full360_azimuth_only
  --select_best_by azimuth_mae
  --class_loss_weight 0.0
  --distance_loss_weight 0.0
  --azimuth_loss_weight "${AZIMUTH_LOSS_WEIGHT}"
  --elevation_loss_weight 0.0
  --vector_loss_weight 0.0
  --no_use_class_head
  --no_use_distance_head
  --no_use_elevation_head
  --no_use_vector_head
  --output_dir "${OUTPUT_DIR}"
)

if [ -n "${INIT_CHECKPOINT}" ]; then
  args+=(--init_checkpoint "${INIT_CHECKPOINT}")
fi

if [ "${AUDIO_NORMALIZE}" = "1" ]; then
  args+=(--audio_normalize)
fi

echo "[stage15-local-azonly] train_json=${TRAIN_JSON}"
echo "[stage15-local-azonly] val_json=${VAL_JSON}"
echo "[stage15-local-azonly] output_dir=${OUTPUT_DIR}"
if [ -n "${INIT_CHECKPOINT}" ]; then
  echo "[stage15-local-azonly] init_checkpoint=${INIT_CHECKPOINT}"
fi
"${PYTHON_BIN}" "${PROJECT_ROOT}/train.py" "${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
