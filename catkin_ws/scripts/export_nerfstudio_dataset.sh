#!/usr/bin/env bash
set -eo pipefail

DATA_DIR="${DATA_DIR:-$HOME/rgb_pose_dataset_01}"
KEEP_EVERY="${KEEP_EVERY:-1}"
START_INDEX="${START_INDEX:-0}"
MAX_FRAMES="${MAX_FRAMES:-}"
OUTPUT="${OUTPUT:-$DATA_DIR/transforms.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INCLUDE_DEPTH="${INCLUDE_DEPTH:-False}"
GENERATE_POINT_CLOUD="${GENERATE_POINT_CLOUD:-False}"
DEPTH_DIR="${DEPTH_DIR:-depth}"
POINTCLOUD_OUTPUT="${POINTCLOUD_OUTPUT:-$DATA_DIR/sparse_pc.ply}"
DEPTH_SCALE="${DEPTH_SCALE:-0.001}"
DEPTH_MIN="${DEPTH_MIN:-0.15}"
DEPTH_MAX="${DEPTH_MAX:-5.0}"
POINT_STRIDE="${POINT_STRIDE:-6}"
MAX_POINTS_PER_FRAME="${MAX_POINTS_PER_FRAME:-12000}"
MAX_TOTAL_POINTS="${MAX_TOTAL_POINTS:-1500000}"
VOXEL_SIZE="${VOXEL_SIZE:-0.01}"

echo "[nerfstudio-export] data_dir=$DATA_DIR"
echo "[nerfstudio-export] keep_every=$KEEP_EVERY"
echo "[nerfstudio-export] start_index=$START_INDEX"
echo "[nerfstudio-export] max_frames=${MAX_FRAMES:-}"
echo "[nerfstudio-export] output=$OUTPUT"
echo "[nerfstudio-export] include_depth=$INCLUDE_DEPTH"
echo "[nerfstudio-export] generate_point_cloud=$GENERATE_POINT_CLOUD"
echo "[nerfstudio-export] pointcloud_output=$POINTCLOUD_OUTPUT"

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

if [ "$INCLUDE_DEPTH" = "True" ] || [ "$INCLUDE_DEPTH" = "true" ] || [ "$INCLUDE_DEPTH" = "1" ]; then
  args+=(--include-depth --depth-dir "$DEPTH_DIR")
fi

if [ "$GENERATE_POINT_CLOUD" = "True" ] || [ "$GENERATE_POINT_CLOUD" = "true" ] || [ "$GENERATE_POINT_CLOUD" = "1" ]; then
  args+=(
    --generate-point-cloud
    --depth-dir "$DEPTH_DIR"
    --pointcloud-output "$POINTCLOUD_OUTPUT"
    --depth-scale "$DEPTH_SCALE"
    --depth-min "$DEPTH_MIN"
    --depth-max "$DEPTH_MAX"
    --point-stride "$POINT_STRIDE"
    --max-points-per-frame "$MAX_POINTS_PER_FRAME"
    --max-total-points "$MAX_TOTAL_POINTS"
    --voxel-size "$VOXEL_SIZE"
  )
fi

exec "$PYTHON_BIN" "${args[@]}"
