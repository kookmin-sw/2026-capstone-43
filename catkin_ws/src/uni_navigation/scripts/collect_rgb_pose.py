#!/usr/bin/env python3

import csv
import json
import math
import os
from pathlib import Path

import cv2
import rospy
import tf2_ros
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_rotate(q, v):
    rotated = quat_multiply(quat_multiply(q, (v[0], v[1], v[2], 0.0)), quat_conjugate(q))
    return (rotated[0], rotated[1], rotated[2])


def compose_pose(parent_t, parent_q, child_t, child_q):
    rotated_child_t = quat_rotate(parent_q, child_t)
    return (
        (
            parent_t[0] + rotated_child_t[0],
            parent_t[1] + rotated_child_t[1],
            parent_t[2] + rotated_child_t[2],
        ),
        quat_multiply(parent_q, child_q),
    )


def quat_angle(q1, q2):
    dot = abs(q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3])
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


class RgbPoseCollector:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.odom_topic = rospy.get_param("~odom_topic", "/robot/odom")
        self.pose_source = rospy.get_param("~pose_source", "odom").lower()
        self.target_frame = rospy.get_param("~target_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.camera_frame = rospy.get_param("~camera_frame", "")
        output_dir_param = rospy.get_param("~output_dir", str(Path.home() / "rgb_pose_dataset"))
        self.output_dir = Path(os.path.expandvars(os.path.expanduser(output_dir_param)))
        self.min_interval = float(rospy.get_param("~min_interval", 0.2))
        self.min_translation = float(rospy.get_param("~min_translation", 0.03))
        self.min_rotation = float(rospy.get_param("~min_rotation", 0.03))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 95))
        self.image_format = rospy.get_param("~image_format", "png").lower()
        self.use_latest_tf = bool(rospy.get_param("~use_latest_tf", True))

        if self.image_format not in ("png", "jpg", "jpeg"):
            raise ValueError("~image_format must be png, jpg, or jpeg")

        self.rgb_dir = self.output_dir / "rgb"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.count = 0
        self.last_stamp = None
        self.last_translation = None
        self.last_quaternion = None
        self.latest_odom = None
        self.camera_info = None

        self.pose_csv_file = open(self.output_dir / "poses.csv", "w", newline="")
        self.pose_csv = csv.writer(self.pose_csv_file)
        self.pose_csv.writerow(["filename", "image_stamp", "tf_stamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

        self.pose_tum_file = open(self.output_dir / "poses_tum.txt", "w")
        self.pose_tum_file.write("# timestamp tx ty tz qx qy qz qw\n")

        metadata = {
            "image_topic": self.image_topic,
            "camera_info_topic": self.camera_info_topic,
            "odom_topic": self.odom_topic,
            "pose_source": self.pose_source,
            "target_frame": self.target_frame,
            "base_frame": self.base_frame,
            "camera_frame_param": self.camera_frame,
            "image_format": self.image_format,
            "min_interval": self.min_interval,
            "min_translation": self.min_translation,
            "min_rotation": self.min_rotation,
            "use_latest_tf": self.use_latest_tf,
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=3, buff_size=2**24)
        self.info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)
        if self.pose_source == "odom":
            self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=20)
        rospy.on_shutdown(self.close)

        rospy.loginfo("Saving RGB+pose dataset to %s", self.output_dir)
        rospy.loginfo("Looking up TF %s -> image camera frame", self.target_frame)

    def odom_callback(self, msg):
        self.latest_odom = msg

    def camera_info_callback(self, msg):
        if self.camera_info is not None:
            return
        self.camera_info = msg
        intrinsics = {
            "width": msg.width,
            "height": msg.height,
            "fx": msg.K[0],
            "fy": msg.K[4],
            "cx": msg.K[2],
            "cy": msg.K[5],
            "distortion_model": msg.distortion_model,
            "distortion_coefficients": list(msg.D),
        }
        with open(self.output_dir / "camera_info.json", "w") as f:
            json.dump(intrinsics, f, indent=2)

    def lookup_pose(self, source_frame, stamp):
        if self.pose_source == "tf":
            lookup_time = rospy.Time(0) if self.use_latest_tf else stamp
            tf = self.tf_buffer.lookup_transform(self.target_frame, source_frame, lookup_time, rospy.Duration(0.2))
            t = tf.transform.translation
            q = tf.transform.rotation
            return (t.x, t.y, t.z), (q.x, q.y, q.z, q.w), tf.header.stamp

        if self.pose_source != "odom":
            raise ValueError("~pose_source must be odom or tf")

        if self.latest_odom is None:
            raise RuntimeError("No odom received yet on " + self.odom_topic)

        tf = self.tf_buffer.lookup_transform(self.base_frame, source_frame, rospy.Time(0), rospy.Duration(0.2))
        bt = tf.transform.translation
        bq = tf.transform.rotation
        base_to_camera_t = (bt.x, bt.y, bt.z)
        base_to_camera_q = (bq.x, bq.y, bq.z, bq.w)

        odom_pose = self.latest_odom.pose.pose
        ot = odom_pose.position
        oq = odom_pose.orientation
        odom_to_base_t = (ot.x, ot.y, ot.z)
        odom_to_base_q = (oq.x, oq.y, oq.z, oq.w)

        translation, quaternion = compose_pose(
            odom_to_base_t,
            odom_to_base_q,
            base_to_camera_t,
            base_to_camera_q,
        )
        return translation, quaternion, self.latest_odom.header.stamp

    def should_save(self, stamp, translation, quaternion):
        if self.last_stamp is None:
            return True

        dt = (stamp - self.last_stamp).to_sec()
        if dt < self.min_interval:
            return False

        dx = translation[0] - self.last_translation[0]
        dy = translation[1] - self.last_translation[1]
        dz = translation[2] - self.last_translation[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        angle = quat_angle(quaternion, self.last_quaternion)

        return dist >= self.min_translation or angle >= self.min_rotation

    def image_callback(self, msg):
        source_frame = self.camera_frame or msg.header.frame_id
        if not source_frame:
            rospy.logwarn_throttle(5.0, "Image header has no frame_id and ~camera_frame is empty")
            return

        try:
            translation, quaternion, pose_stamp = self.lookup_pose(source_frame, msg.header.stamp)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Pose lookup failed: %s", exc)
            return

        if not self.should_save(msg.header.stamp, translation, quaternion):
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logerr("Image conversion failed: %s", exc)
            return

        stamp_float = msg.header.stamp.to_sec()
        pose_stamp_float = pose_stamp.to_sec()
        filename = f"{self.count:06d}.{self.image_format}"
        image_path = self.rgb_dir / filename

        if self.image_format == "png":
            cv2.imwrite(str(image_path), cv_image)
        else:
            cv2.imwrite(str(image_path), cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])

        t = translation
        q = quaternion
        self.pose_csv.writerow([filename, f"{stamp_float:.9f}", f"{pose_stamp_float:.9f}", t[0], t[1], t[2], q[0], q[1], q[2], q[3]])
        self.pose_tum_file.write(f"{pose_stamp_float:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
        self.pose_csv_file.flush()
        self.pose_tum_file.flush()

        self.last_stamp = msg.header.stamp
        self.last_translation = translation
        self.last_quaternion = quaternion
        self.count += 1

        rospy.loginfo_throttle(2.0, "Saved %d frames to %s", self.count, self.output_dir)

    def close(self):
        self.pose_csv_file.close()
        self.pose_tum_file.close()


if __name__ == "__main__":
    rospy.init_node("collect_rgb_pose")
    RgbPoseCollector()
    rospy.spin()
