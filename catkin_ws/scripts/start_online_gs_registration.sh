#!/usr/bin/env bash
set -eo pipefail

DATA_DIR="${DATA_DIR:-$HOME/rgb_pose_dataset_01}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/scene01}"
CONFIG="${CONFIG:-configs/online_gs_slam.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PATH="$HOME/.local/bin:${CUDA_HOME:-/usr/local/cuda-12.1}/bin:$PATH"
if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda-12.1 ]; then
  export CUDA_HOME=/usr/local/cuda-12.1
fi

echo "[online-gs] data_dir=$DATA_DIR"
echo "[online-gs] output_dir=$OUTPUT_DIR"
echo "[online-gs] config=$CONFIG"

exec "$PYTHON_BIN" run_online_gs_slam.py \
  --data_dir "$DATA_DIR" \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR"
