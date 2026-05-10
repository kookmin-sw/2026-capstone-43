#!/usr/bin/env bash
set -eo pipefail

DATA_DIR="${DATA_DIR:-$HOME/rgb_pose_dataset_01}"
KEEP_EVERY="${KEEP_EVERY:-20}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-3000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-piper_splatfacto}"
VIEWER_PORT="${VIEWER_PORT:-7007}"
VIEWER_MAX_NUM_DISPLAY_IMAGES="${VIEWER_MAX_NUM_DISPLAY_IMAGES:-32}"
VIEWER_NUM_RAYS_PER_CHUNK="${VIEWER_NUM_RAYS_PER_CHUNK:-8192}"
MAKE_SHARE_URL="${MAKE_SHARE_URL:-True}"
CAMERA_RES_SCALE_FACTOR="${CAMERA_RES_SCALE_FACTOR:-0.5}"
CACHE_IMAGES="${CACHE_IMAGES:-cpu}"
IMAGES_ON_GPU="${IMAGES_ON_GPU:-False}"
MODEL_NUM_DOWNSCALES="${MODEL_NUM_DOWNSCALES:-2}"

if [ -n "${NERFSTUDIO_BIN_DIR:-}" ]; then
  export PATH="$NERFSTUDIO_BIN_DIR:$PATH"
else
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda-12.1 ]; then
  export CUDA_HOME=/usr/local/cuda-12.1
fi
if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# CUDA extension builds can spawn many cudafe++/cicc compiler processes.
# On 16 GB RAM laptops this can OOM the whole desktop, so keep compilation serial.
export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

if ! command -v ns-train >/dev/null 2>&1; then
  echo "[nerfstudio] ns-train not found."
  echo "[nerfstudio] Install Nerfstudio first, then rerun this script."
  echo "[nerfstudio] See: https://docs.nerf.studio/"
  exit 127
fi

"$(dirname "$0")/export_nerfstudio_dataset.sh"

echo "[nerfstudio] training splatfacto"
echo "[nerfstudio] ns-train=$(command -v ns-train)"
echo "[nerfstudio] data_dir=$DATA_DIR"
echo "[nerfstudio] max_num_iterations=$MAX_NUM_ITERATIONS"
echo "[nerfstudio] viewer_port=$VIEWER_PORT"
echo "[nerfstudio] viewer_max_num_display_images=$VIEWER_MAX_NUM_DISPLAY_IMAGES"
echo "[nerfstudio] viewer_num_rays_per_chunk=$VIEWER_NUM_RAYS_PER_CHUNK"
echo "[nerfstudio] make_share_url=$MAKE_SHARE_URL"
echo "[nerfstudio] camera_res_scale_factor=$CAMERA_RES_SCALE_FACTOR"
echo "[nerfstudio] cache_images=$CACHE_IMAGES"
echo "[nerfstudio] images_on_gpu=$IMAGES_ON_GPU"
echo "[nerfstudio] model_num_downscales=$MODEL_NUM_DOWNSCALES"
echo "[nerfstudio] CUDA_HOME=${CUDA_HOME:-}"
echo "[nerfstudio] MAX_JOBS=$MAX_JOBS"
echo "[nerfstudio] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "[nerfstudio] TORCHINDUCTOR_COMPILE_THREADS=$TORCHINDUCTOR_COMPILE_THREADS"

exec ns-train splatfacto \
  --experiment-name "$EXPERIMENT_NAME" \
  --max-num-iterations "$MAX_NUM_ITERATIONS" \
  --viewer.websocket-port "$VIEWER_PORT" \
  --viewer.max-num-display-images "$VIEWER_MAX_NUM_DISPLAY_IMAGES" \
  --viewer.num-rays-per-chunk "$VIEWER_NUM_RAYS_PER_CHUNK" \
  --viewer.make-share-url "$MAKE_SHARE_URL" \
  --pipeline.datamanager.camera-res-scale-factor "$CAMERA_RES_SCALE_FACTOR" \
  --pipeline.datamanager.cache-images "$CACHE_IMAGES" \
  --pipeline.datamanager.images-on-gpu "$IMAGES_ON_GPU" \
  --pipeline.model.num-downscales "$MODEL_NUM_DOWNSCALES" \
  --data "$DATA_DIR"
