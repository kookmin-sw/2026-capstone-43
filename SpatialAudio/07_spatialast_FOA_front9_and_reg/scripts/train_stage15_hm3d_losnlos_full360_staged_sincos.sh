#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/yu/miniconda3/envs/spatial-ast/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_ROOT="${DATASET_ROOT:-/home/yu/Project_git/01_dataset/hm3d_losnlos_100k_balanced}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${PROJECT_ROOT}/manifests_stage15/hm3d_losnlos_100k_balanced_full360}"
TRAIN_JSON="${TRAIN_JSON:-${MANIFEST_ROOT}/train.json}"
VAL_JSON="${VAL_JSON:-${MANIFEST_ROOT}/val.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs_stage15}"

STEM_VARIANT="${STEM_VARIANT:-conv64_64_out8}"
UNFREEZE_LAST_N_BLOCKS="${UNFREEZE_LAST_N_BLOCKS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1337}"
AUDIO_NORMALIZE="${AUDIO_NORMALIZE:-1}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"
LIMIT_TRAIN_SAMPLES="${LIMIT_TRAIN_SAMPLES:-0}"
LIMIT_VAL_SAMPLES="${LIMIT_VAL_SAMPLES:-0}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-0}"

RUN_PREFIX="${RUN_PREFIX:-full360_sincos_staged_$(date +%Y%m%d_%H%M%S)}"
STAGE_A_RUN_NAME="${STAGE_A_RUN_NAME:-${RUN_PREFIX}_stageA_azimuth_only}"
STAGE_B_RUN_NAME="${STAGE_B_RUN_NAME:-${RUN_PREFIX}_stageB_azimuth_protected}"
STAGE_C_RUN_NAME="${STAGE_C_RUN_NAME:-${RUN_PREFIX}_stageC_balanced_multitask}"

STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-30}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-15}"
STAGE_C_EPOCHS="${STAGE_C_EPOCHS:-15}"

STAGE_A_AZIMUTH_WEIGHT="${STAGE_A_AZIMUTH_WEIGHT:-2.0}"
STAGE_B_AZIMUTH_WEIGHT="${STAGE_B_AZIMUTH_WEIGHT:-4.0}"
STAGE_B_ELEVATION_WEIGHT="${STAGE_B_ELEVATION_WEIGHT:-0.5}"
STAGE_B_VECTOR_WEIGHT="${STAGE_B_VECTOR_WEIGHT:-0.0}"
STAGE_C_AZIMUTH_WEIGHT="${STAGE_C_AZIMUTH_WEIGHT:-3.0}"
STAGE_C_ELEVATION_WEIGHT="${STAGE_C_ELEVATION_WEIGHT:-2.0}"
STAGE_C_VECTOR_WEIGHT="${STAGE_C_VECTOR_WEIGHT:-0.5}"

LR_STEM="${LR_STEM:-1e-4}"
LR_ADAPTER="${LR_ADAPTER:-1e-4}"
LR_HEADS="${LR_HEADS:-5e-4}"
LR_TRANSFORMER="${LR_TRANSFORMER:-1e-5}"
LR_PATCH_EMBED="${LR_PATCH_EMBED:-1e-5}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"

if [ ! -f "${TRAIN_JSON}" ] || [ ! -f "${VAL_JSON}" ]; then
  echo "[stage15-staged] full360 manifests missing, building them first"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/build_hm3d_losnlos_full360_manifests.py" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${MANIFEST_ROOT}"
fi

prepare_output_dir() {
  local output_dir="$1"
  if [ -d "${output_dir}" ] && [ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]; then
    if [ "${OVERWRITE_OUTPUT_DIR}" = "1" ]; then
      rm -rf "${output_dir}"
    else
      echo "[stage15-staged] output dir already exists and is not empty: ${output_dir}" >&2
      echo "[stage15-staged] set OVERWRITE_OUTPUT_DIR=1 or choose a new RUN_PREFIX" >&2
      exit 1
    fi
  fi
  mkdir -p "${output_dir}"
}

run_stage() {
  local run_name="$1"
  shift
  local output_dir="${OUTPUT_ROOT}/${run_name}"
  prepare_output_dir "${output_dir}"

  local common_args=(
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
    --seed "${SEED}"
    --output_dir "${output_dir}"
  )

  if [ "${AUDIO_NORMALIZE}" = "1" ]; then
    common_args+=(--audio_normalize)
  fi

  echo "[stage15-staged] running ${run_name}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/train.py" \
    "${common_args[@]}" \
    "$@" 2>&1 | tee "${output_dir}/train.log"
}

if [ -n "${STAGE_A_CKPT:-}" ]; then
  echo "[stage15-staged] using provided STAGE_A_CKPT=${STAGE_A_CKPT}"
  stage_a_ckpt="${STAGE_A_CKPT}"
else
  run_stage "${STAGE_A_RUN_NAME}" \
    --epochs "${STAGE_A_EPOCHS}" \
    --recipe_name stageA_azimuth_only \
    --select_best_by azimuth_mae \
    --class_loss_weight 0.0 \
    --distance_loss_weight 0.0 \
    --azimuth_loss_weight "${STAGE_A_AZIMUTH_WEIGHT}" \
    --elevation_loss_weight 0.0 \
    --vector_loss_weight 0.0 \
    --no_use_class_head \
    --no_use_distance_head \
    --no_use_elevation_head \
    --no_use_vector_head
  stage_a_ckpt="${OUTPUT_ROOT}/${STAGE_A_RUN_NAME}/best_checkpoint.pt"
fi

if [ -n "${STAGE_B_CKPT:-}" ]; then
  echo "[stage15-staged] using provided STAGE_B_CKPT=${STAGE_B_CKPT}"
  stage_b_ckpt="${STAGE_B_CKPT}"
else
  run_stage "${STAGE_B_RUN_NAME}" \
    --epochs "${STAGE_B_EPOCHS}" \
    --recipe_name stageB_azimuth_protected \
    --init_checkpoint "${stage_a_ckpt}" \
    --select_best_by azimuth_mae \
    --class_loss_weight 0.0 \
    --distance_loss_weight 0.0 \
    --azimuth_loss_weight "${STAGE_B_AZIMUTH_WEIGHT}" \
    --elevation_loss_weight "${STAGE_B_ELEVATION_WEIGHT}" \
    --vector_loss_weight "${STAGE_B_VECTOR_WEIGHT}" \
    --no_use_class_head \
    --no_use_distance_head
  stage_b_ckpt="${OUTPUT_ROOT}/${STAGE_B_RUN_NAME}/best_checkpoint.pt"
fi

run_stage "${STAGE_C_RUN_NAME}" \
  --epochs "${STAGE_C_EPOCHS}" \
  --recipe_name stageC_balanced_multitask \
  --init_checkpoint "${stage_b_ckpt}" \
  --select_best_by angular_error \
  --class_loss_weight 0.0 \
  --distance_loss_weight 0.0 \
  --azimuth_loss_weight "${STAGE_C_AZIMUTH_WEIGHT}" \
  --elevation_loss_weight "${STAGE_C_ELEVATION_WEIGHT}" \
  --vector_loss_weight "${STAGE_C_VECTOR_WEIGHT}" \
  --no_use_class_head \
  --no_use_distance_head

echo "[stage15-staged] done"
echo "[stage15-staged] stage A: ${OUTPUT_ROOT}/${STAGE_A_RUN_NAME}"
echo "[stage15-staged] stage B: ${OUTPUT_ROOT}/${STAGE_B_RUN_NAME}"
echo "[stage15-staged] stage C: ${OUTPUT_ROOT}/${STAGE_C_RUN_NAME}"
