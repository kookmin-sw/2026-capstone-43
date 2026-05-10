#!/usr/bin/env bash
set -eo pipefail

DATA_DIR="${DATA_DIR:-$HOME/rgb_pose_dataset_01}"
KEEP_EVERY="${KEEP_EVERY:-1}"
START_INDEX="${START_INDEX:-0}"
MAX_FRAMES="${MAX_FRAMES:-}"
OUTPUT="${OUTPUT:-$DATA_DIR/transforms.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[nerfstudio-export] data_dir=$DATA_DIR"
echo "[nerfstudio-export] keep_every=$KEEP_EVERY"
echo "[nerfstudio-export] start_index=$START_INDEX"
echo "[nerfstudio-export] max_frames=${MAX_FRAMES:-}"
echo "[nerfstudio-export] output=$OUTPUT"

args=(
  src/uni_navigation/scripts/rgb_pose_to_nerfstudio.py
  "$DATA_DIR"
  --keep-every "$KEEP_EVERY"
  --start-index "$START_INDEX"
  --output "$OUTPUT"
)

if [ -n "$MAX_FRAMES" ]; then
  args+=(--max-frames "$MAX_FRAMES")
fi

exec "$PYTHON_BIN" "${args[@]}"
