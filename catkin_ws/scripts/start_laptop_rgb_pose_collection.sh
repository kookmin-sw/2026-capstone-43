#!/usr/bin/env bash
set -eo pipefail

ROS_MASTER_URI_VALUE="${ROS_MASTER_URI_VALUE:-http://192.168.0.4:11311}"
ROS_IP_VALUE="${ROS_IP_VALUE:-192.168.0.100}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/camera/color/image_raw_3hz}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/aligned_depth_to_color/image_raw_3hz}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/color/camera_info}"
DEPTH_CAMERA_INFO_TOPIC="${DEPTH_CAMERA_INFO_TOPIC:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/rgb_pose_dataset_01}"
MIN_INTERVAL="${MIN_INTERVAL:-0.333333}"
SAVE_DEPTH="${SAVE_DEPTH:-False}"
REQUIRE_DEPTH="${REQUIRE_DEPTH:-True}"
SYNC_SLOP="${SYNC_SLOP:-0.08}"
SYNC_ODOM="${SYNC_ODOM:-False}"
MAX_POSE_DT="${MAX_POSE_DT:-0.2}"
USE_LATEST_TF="${USE_LATEST_TF:-False}"
POSE_TIME_OFFSET="${POSE_TIME_OFFSET:-1359.2253}"
AUTO_POSE_TIME_OFFSET="${AUTO_POSE_TIME_OFFSET:-False}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source /opt/ros/noetic/setup.bash
source "$WORKSPACE_DIR/devel/setup.bash"

export ROS_MASTER_URI="$ROS_MASTER_URI_VALUE"
export ROS_IP="$ROS_IP_VALUE"
unset ROS_HOSTNAME

echo "[laptop-collector] ROS_MASTER_URI=$ROS_MASTER_URI"
echo "[laptop-collector] ROS_IP=$ROS_IP"
echo "[laptop-collector] image_topic=$IMAGE_TOPIC"
echo "[laptop-collector] depth_topic=$DEPTH_TOPIC"
echo "[laptop-collector] save_depth=$SAVE_DEPTH"
echo "[laptop-collector] output_dir=$OUTPUT_DIR"

exec roslaunch uni_navigation collect_rgb_pose.launch \
  image_topic:="$IMAGE_TOPIC" \
  depth_topic:="$DEPTH_TOPIC" \
  camera_info_topic:="$CAMERA_INFO_TOPIC" \
  depth_camera_info_topic:="$DEPTH_CAMERA_INFO_TOPIC" \
  output_dir:="$OUTPUT_DIR" \
  min_interval:="$MIN_INTERVAL" \
  min_translation:=0.0 \
  min_rotation:=0.0 \
  use_latest_tf:="$USE_LATEST_TF" \
  save_depth:="$SAVE_DEPTH" \
  require_depth:="$REQUIRE_DEPTH" \
  sync_slop:="$SYNC_SLOP" \
  sync_odom:="$SYNC_ODOM" \
  max_pose_dt:="$MAX_POSE_DT" \
  pose_time_offset:="$POSE_TIME_OFFSET" \
  auto_pose_time_offset:="$AUTO_POSE_TIME_OFFSET"
