#!/usr/bin/env python3

import argparse

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser(description="Estimate odom_stamp - image_stamp for RGB-D collection.")
    parser.add_argument("--image-topic", default="/camera/color/image_raw_3hz")
    parser.add_argument("--odom-topic", default="/robot/odom")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    rospy.init_node("estimate_ros_stamp_offset", anonymous=True)
    offsets = []
    latest_odom = {"msg": None}

    def odom_cb(msg):
        latest_odom["msg"] = msg

    def image_cb(msg):
        odom = latest_odom["msg"]
        if odom is None:
            return
        offsets.append(odom.header.stamp.to_sec() - msg.header.stamp.to_sec())
        if len(offsets) >= args.samples:
            rospy.signal_shutdown("done")

    rospy.Subscriber(args.odom_topic, Odometry, odom_cb, queue_size=100)
    rospy.Subscriber(args.image_topic, Image, image_cb, queue_size=20)
    rospy.spin()

    if not offsets:
        raise RuntimeError("No offset samples collected.")
    mean = sum(offsets) / len(offsets)
    print(f"samples={len(offsets)}")
    print(f"min={min(offsets):.9f}")
    print(f"max={max(offsets):.9f}")
    print(f"mean={mean:.9f}")
    print(f"Use: POSE_TIME_OFFSET={mean:.9f}")


if __name__ == "__main__":
    main()
