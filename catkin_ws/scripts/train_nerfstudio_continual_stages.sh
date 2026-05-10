#!/usr/bin/env bash
set -eo pipefail

SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-$HOME/rgb_pose_dataset_01}"
STAGE_DATA_ROOT="${STAGE_DATA_ROOT:-$PWD/outputs/nerfstudio_staged_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/nerfstudio_continual}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-piper_continual}"

SAMPLE_STRIDE="${SAMPLE_STRIDE:-50}"
INITIAL_SAMPLED_FRAMES="${INITIAL_SAMPLED_FRAMES:-20}"
ADD_SAMPLED_FRAMES="${ADD_SAMPLED_FRAMES:-20}"
NUM_STAGES="${NUM_STAGES:-5}"
ITERATIONS_PER_STAGE="${ITERATIONS_PER_STAGE:-300}"

VIEWER_PORT="${VIEWER_PORT:-7007}"
VIS="${VIS:-viewer}"
VIEWER_MAX_NUM_DISPLAY_IMAGES="${VIEWER_MAX_NUM_DISPLAY_IMAGES:-16}"
VIEWER_NUM_RAYS_PER_CHUNK="${VIEWER_NUM_RAYS_PER_CHUNK:-4096}"
MAKE_SHARE_URL="${MAKE_SHARE_URL:-True}"
CAMERA_RES_SCALE_FACTOR="${CAMERA_RES_SCALE_FACTOR:-0.35}"
CACHE_IMAGES="${CACHE_IMAGES:-cpu}"
IMAGES_ON_GPU="${IMAGES_ON_GPU:-False}"
MODEL_NUM_DOWNSCALES="${MODEL_NUM_DOWNSCALES:-2}"
REFINE_EVERY="${REFINE_EVERY:-1000000}"
RESET_ALPHA_EVERY="${RESET_ALPHA_EVERY:-1000000}"
STOP_SPLIT_AT="${STOP_SPLIT_AT:-0}"
STOP_SCREEN_SIZE_AT="${STOP_SCREEN_SIZE_AT:-0}"
CULL_ALPHA_THRESH="${CULL_ALPHA_THRESH:-0.0}"
CULL_SCALE_THRESH="${CULL_SCALE_THRESH:-1000000.0}"
DENSIFY_GRAD_THRESH="${DENSIFY_GRAD_THRESH:-1000000.0}"

CONDA_ENV_DIR="${CONDA_ENV_DIR:-$HOME/miniconda3/envs/ns310}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"
NS_TRAIN="$CONDA_ENV_DIR/bin/ns-train"

if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$NS_TRAIN" ]; then
  echo "[continual-ns] Missing ns310 environment at $CONDA_ENV_DIR"
  exit 127
fi

export PATH="$CONDA_ENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

mkdir -p "$STAGE_DATA_ROOT" "$OUTPUT_ROOT"

LOAD_DIR=""
for ((stage=0; stage<NUM_STAGES; stage++)); do
  sampled_count=$((INITIAL_SAMPLED_FRAMES + stage * ADD_SAMPLED_FRAMES))
  stage_name="$(printf 'stage_%03d' "$stage")"
  stage_data_dir="$STAGE_DATA_ROOT/$stage_name"
  timestamp="$stage_name"
  run_dir="$OUTPUT_ROOT/$EXPERIMENT_NAME/splatfacto/$timestamp"
  model_dir="$run_dir/nerfstudio_models"

  echo "[continual-ns] stage=$stage_name sampled_count=$sampled_count stride=$SAMPLE_STRIDE"
  "$PYTHON_BIN" scripts/make_nerfstudio_incremental_subset.py \
    --source-data-dir "$SOURCE_DATA_DIR" \
    --output-data-dir "$stage_data_dir" \
    --sample-stride "$SAMPLE_STRIDE" \
    --num-sampled-frames "$sampled_count"

  load_args=()
  if [ -n "$LOAD_DIR" ]; then
    load_args=(--load-dir "$LOAD_DIR")
    echo "[continual-ns] resuming from $LOAD_DIR"
  fi

  "$NS_TRAIN" splatfacto \
    --output-dir "$OUTPUT_ROOT" \
    --experiment-name "$EXPERIMENT_NAME" \
    --timestamp "$timestamp" \
    --vis "$VIS" \
    --max-num-iterations "$ITERATIONS_PER_STAGE" \
    --save-only-latest-checkpoint True \
    --load-scheduler False \
    --viewer.websocket-port "$VIEWER_PORT" \
    --viewer.max-num-display-images "$VIEWER_MAX_NUM_DISPLAY_IMAGES" \
    --viewer.num-rays-per-chunk "$VIEWER_NUM_RAYS_PER_CHUNK" \
    --viewer.make-share-url "$MAKE_SHARE_URL" \
    --viewer.quit-on-train-completion True \
    --pipeline.datamanager.camera-res-scale-factor "$CAMERA_RES_SCALE_FACTOR" \
    --pipeline.datamanager.cache-images "$CACHE_IMAGES" \
    --pipeline.datamanager.images-on-gpu "$IMAGES_ON_GPU" \
    --pipeline.model.num-downscales "$MODEL_NUM_DOWNSCALES" \
    --pipeline.model.camera-optimizer.mode off \
    --pipeline.model.refine-every "$REFINE_EVERY" \
    --pipeline.model.reset-alpha-every "$RESET_ALPHA_EVERY" \
    --pipeline.model.stop-split-at "$STOP_SPLIT_AT" \
    --pipeline.model.stop-screen-size-at "$STOP_SCREEN_SIZE_AT" \
    --pipeline.model.cull-alpha-thresh "$CULL_ALPHA_THRESH" \
    --pipeline.model.cull-scale-thresh "$CULL_SCALE_THRESH" \
    --pipeline.model.densify-grad-thresh "$DENSIFY_GRAD_THRESH" \
    "${load_args[@]}" \
    --data "$stage_data_dir"

  LOAD_DIR="$model_dir"
  echo "[continual-ns] completed $stage_name; next load_dir=$LOAD_DIR"
done
