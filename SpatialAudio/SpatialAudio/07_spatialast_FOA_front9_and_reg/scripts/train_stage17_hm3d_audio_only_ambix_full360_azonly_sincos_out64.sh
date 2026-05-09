#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/yu/miniconda3/envs/spatial-ast/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_ROOT="${DATASET_ROOT:-/media/yu/Extreme SSD/hm3d_audio_only_100k_ambix}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${PROJECT_ROOT}/manifests_stage17/hm3d_audio_only_ambix_full360_70k}"
GLOS_MANIFEST_ROOT="${GLOS_MANIFEST_ROOT:-${PROJECT_ROOT}/manifests_stage17/hm3d_audio_only_ambix_full360_70k_glos}"
TRAIN_JSON="${TRAIN_JSON:-${MANIFEST_ROOT}/train.json}"
VAL_JSON="${VAL_JSON:-${MANIFEST_ROOT}/val.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs_stage17}"

STEM_VARIANT="${STEM_VARIANT:-conv64_64_out64}"
UNFREEZE_LAST_N_BLOCKS="${UNFREEZE_LAST_N_BLOCKS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-30}"
SEED="${SEED:-1337}"
RUN_NAME="${RUN_NAME:-stage17_audio_only_ambix70k_sincos_azonly_${STEM_VARIANT}_last${UNFREEZE_LAST_N_BLOCKS}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

REBUILD_MANIFESTS="${REBUILD_MANIFESTS:-1}"
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

if [ "${REBUILD_MANIFESTS}" = "1" ] || [ ! -f "${TRAIN_JSON}" ] || [ ! -f "${VAL_JSON}" ]; then
  echo "[stage17-ambix-azonly] building strict mic-local AmbiX ACN/SN3D WYZX manifests"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/build_hm3d_losnlos_local_full360_manifests.py" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${MANIFEST_ROOT}" \
    --glos_output_dir "${GLOS_MANIFEST_ROOT}" \
    --strict_audio_conventions \
    --skip_audio_exists_check \
    --compact_json \
    --train_ratio 0.9 \
    --val_ratio 0.1 \
    --test_ratio 0.0 \
    --seed "${SEED}"
fi

if [ -d "${OUTPUT_DIR}" ] && [ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]; then
  if [ "${OVERWRITE_OUTPUT_DIR}" = "1" ]; then
    rm -rf "${OUTPUT_DIR}"
  else
    echo "[stage17-ambix-azonly] output dir already exists and is not empty: ${OUTPUT_DIR}" >&2
    echo "[stage17-ambix-azonly] set OVERWRITE_OUTPUT_DIR=1 or choose a new RUN_NAME" >&2
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
  --recipe_name stage17_audio_only_ambix70k_full360_azimuth_only_out64
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

echo "[stage17-ambix-azonly] dataset_root=${DATASET_ROOT}"
echo "[stage17-ambix-azonly] train_json=${TRAIN_JSON}"
echo "[stage17-ambix-azonly] val_json=${VAL_JSON}"
echo "[stage17-ambix-azonly] output_dir=${OUTPUT_DIR}"
echo "[stage17-ambix-azonly] stored FOA: AmbiX ACN/SN3D WYZX; loader converts to WXYZ internally"
echo "[stage17-ambix-azonly] stem_variant=${STEM_VARIANT}"
echo "[stage17-ambix-azonly] batch_size=${BATCH_SIZE}"
if [ -n "${INIT_CHECKPOINT}" ]; then
  echo "[stage17-ambix-azonly] init_checkpoint=${INIT_CHECKPOINT}"
fi
"${PYTHON_BIN}" "${PROJECT_ROOT}/train.py" "${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
