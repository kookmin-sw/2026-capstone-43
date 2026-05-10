#!/usr/bin/env bash
set -eo pipefail

ROS_MASTER_URI_VALUE="${ROS_MASTER_URI_VALUE:-http://192.168.0.4:11311}"
ROS_IP_VALUE="${ROS_IP_VALUE:-192.168.0.100}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/camera/color/image_raw_3hz}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/rgb_pose_dataset_01}"
MIN_INTERVAL="${MIN_INTERVAL:-0.333333}"

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
echo "[laptop-collector] output_dir=$OUTPUT_DIR"

exec roslaunch uni_navigation collect_rgb_pose.launch \
  image_topic:="$IMAGE_TOPIC" \
  output_dir:="$OUTPUT_DIR" \
  min_interval:="$MIN_INTERVAL" \
  min_translation:=0.0 \
  min_rotation:=0.0
