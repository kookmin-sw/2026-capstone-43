#!/usr/bin/env bash
set -eo pipefail

ROS_MASTER_URI_VALUE="${ROS_MASTER_URI_VALUE:-http://192.168.0.4:11311}"
ROS_HOSTNAME_VALUE="${ROS_HOSTNAME_VALUE:-192.168.0.2}"
IN_TOPIC="${IN_TOPIC:-/camera/color/image_raw}"
OUT_TOPIC="${OUT_TOPIC:-/camera/color/image_raw_3hz}"
HZ="${HZ:-3.0}"

source /opt/ros/noetic/setup.bash
if [ -f "$HOME/catkin_ws/devel/setup.bash" ]; then
  source "$HOME/catkin_ws/devel/setup.bash"
fi

export ROS_MASTER_URI="$ROS_MASTER_URI_VALUE"
export ROS_HOSTNAME="$ROS_HOSTNAME_VALUE"

echo "[robot-throttle] $IN_TOPIC -> $OUT_TOPIC at ${HZ}Hz"
exec rosrun topic_tools throttle messages "$IN_TOPIC" "$HZ" "$OUT_TOPIC"
