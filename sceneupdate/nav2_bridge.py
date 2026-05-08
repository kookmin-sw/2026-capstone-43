#!/usr/bin/env python3
"""
External ROS2 bridge between Isaac Sim and Nav2.
Runs as a separate process (Python 3.12 / ROS2 Jazzy).

Communication via /tmp/ files:
  Isaac Sim → /tmp/nav2_goal.json     → this bridge → Nav2 (NavigateToPose)
  Nav2      → /cmd_vel_smoothed       → this bridge → /tmp/nav2_cmd_vel.json → Isaac Sim
  Isaac Sim → /tmp/nav2_cancel.flag   → this bridge → Nav2 (cancel goal)

Usage:
  source /opt/ros/jazzy/setup.bash
  python3 nav2_bridge.py
"""
import os
import sys
import json
import time
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped, Point, TransformStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster

GOAL_FILE = "/tmp/nav2_goal.json"
CMD_VEL_FILE = "/tmp/nav2_cmd_vel.json"
CANCEL_FLAG = "/tmp/nav2_cancel.flag"
STATUS_FILE = "/tmp/nav2_status.json"
SG_VIZ_FILE = "/tmp/nav2_sg_viz.json"


class Nav2Bridge(Node):
    def __init__(self):
        super().__init__('nav2_isaac_bridge')

        # QoS matching Nav2 controller/collision_monitor output (RELIABLE, VOLATILE)
        cmd_vel_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribe to final /cmd_vel (collision_monitor output)
        # Also subscribe to /cmd_vel_smoothed as fallback in case collision_monitor not used
        self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_cb, cmd_vel_qos)
        self.create_subscription(
            Twist, '/cmd_vel_smoothed', self._cmd_vel_smoothed_cb, cmd_vel_qos)

        # Nav2 action client
        self._nav2_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')
        self._goal_handle = None
        self._nav2_status = "idle"

        # Current cmd_vel
        self._v = 0.0
        self._w = 0.0
        self._last_cmd_time = 0.0
        self._has_new_cmd = False

        # File polling timer (20Hz)
        self.create_timer(0.05, self._poll_files)

        # Cmd_vel write timer (50Hz)
        self.create_timer(0.02, self._write_cmd_vel)

        # Track last goal to avoid resending
        self._last_goal_mtime = 0.0

        # Scene graph visualization publisher
        self._sg_pub = self.create_publisher(MarkerArray, '/scene_graph', 10)
        self.create_timer(1.0, self._publish_sg_markers)  # 1Hz
        self._last_sg_mtime = 0.0

        # Clean up old files
        for f in [CMD_VEL_FILE, CANCEL_FLAG, STATUS_FILE, SG_VIZ_FILE]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

        # Static TF: base_link → laser_link (Fetch laser position)
        self._static_tf_pub = StaticTransformBroadcaster(self)
        self._publish_robot_static_tfs()

        self.get_logger().info("Nav2 Bridge ready. Waiting for goals...")

    def _publish_robot_static_tfs(self):
        """Publish static TF: base_link → laser_link (Fetch robot geometry)."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "laser_link"
        # Fetch laser_link offset from base_link (from fetch.usda)
        t.transform.translation.x = 0.235
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.2878
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self._static_tf_pub.sendTransform([t])

    def _cmd_vel_cb(self, msg):
        """Final /cmd_vel from collision_monitor (highest priority)."""
        self._v = msg.linear.x
        self._w = msg.angular.z
        self._last_cmd_time = time.time()
        self._has_new_cmd = True

    def _cmd_vel_smoothed_cb(self, msg):
        """Fallback: /cmd_vel_smoothed (used when collision_monitor not active)."""
        # Only use if /cmd_vel hasn't been updated recently
        if time.time() - self._last_cmd_time > 0.2:
            self._v = msg.linear.x
            self._w = msg.angular.z
            self._last_cmd_time = time.time()
            self._has_new_cmd = True

    def _write_cmd_vel(self):
        """Write current velocity to file for Isaac Sim to read."""
        try:
            now = time.time()
            # Decay to zero if Nav2 stopped sending commands
            if self._last_cmd_time > 0 and (now - self._last_cmd_time) > 0.5:
                self._v = 0.0
                self._w = 0.0
            # Debug: log non-zero velocities
            if self._has_new_cmd and (abs(self._v) > 0.001 or abs(self._w) > 0.001):
                self.get_logger().info(f"cmd_vel → v={self._v:.3f} w={self._w:.3f}", throttle_duration_sec=1.0)
                self._has_new_cmd = False

            data = {
                "v": round(self._v, 4),
                "w": round(self._w, 4),
                "t": round(self._last_cmd_time, 3),  # actual receive time
            }
            tmp = CMD_VEL_FILE + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, CMD_VEL_FILE)
        except Exception:
            pass

    def _poll_files(self):
        """Check for new goals or cancel requests from Isaac Sim."""
        # Check cancel
        if os.path.exists(CANCEL_FLAG):
            try:
                os.remove(CANCEL_FLAG)
            except Exception:
                pass
            self._cancel_goal()
            return

        # Check new goal
        if os.path.exists(GOAL_FILE):
            try:
                mtime = os.path.getmtime(GOAL_FILE)
            except OSError:
                return
            if mtime <= self._last_goal_mtime:
                return
            self._last_goal_mtime = mtime

            try:
                with open(GOAL_FILE, 'r') as f:
                    goal_data = json.load(f)
                x = float(goal_data["x"])
                y = float(goal_data["y"])
                yaw = goal_data.get("yaw")
                self.get_logger().info(f"New goal: ({x:.2f}, {y:.2f})")
                self._send_goal(x, y, yaw)
            except Exception as e:
                self.get_logger().warn(f"Goal parse error: {e}")

    def _send_goal(self, x, y, yaw=None):
        """Send goal to Nav2 via action client."""
        if not self._nav2_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Nav2 action server not available! Retrying in 3s...")
            self._update_status("nav2_unavailable")
            # Queue retry
            self._pending_goal = (x, y, yaw)
            self.create_timer(3.0, self._retry_goal)
            return

        # Cancel previous goal
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "odom"
        goal_msg.pose.header.stamp.sec = 0
        goal_msg.pose.header.stamp.nanosec = 0
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        if yaw is not None:
            goal_msg.pose.pose.orientation.z = math.sin(float(yaw) / 2)
            goal_msg.pose.pose.orientation.w = math.cos(float(yaw) / 2)
        else:
            goal_msg.pose.pose.orientation.w = 1.0

        future = self._nav2_client.send_goal_async(
            goal_msg, feedback_callback=self._feedback_cb)
        future.add_done_callback(self._goal_response_cb)
        self._nav2_status = "sending"
        self._update_status("sending")

    def _retry_goal(self):
        """Retry sending a pending goal."""
        if not hasattr(self, '_pending_goal') or self._pending_goal is None:
            return
        x, y, yaw = self._pending_goal
        self._pending_goal = None
        self.get_logger().info(f"Retrying goal: ({x:.2f}, {y:.2f})")
        self._send_goal(x, y, yaw)

    def _goal_response_cb(self, future):
        self._goal_handle = future.result()
        if self._goal_handle.accepted:
            self.get_logger().info("Nav2 goal accepted")
            self._nav2_status = "navigating"
            self._update_status("navigating")
            result_future = self._goal_handle.get_result_async()
            result_future.add_done_callback(self._result_cb)
        else:
            self.get_logger().warn("Nav2 goal rejected")
            self._nav2_status = "rejected"
            self._update_status("rejected")

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        remaining = fb.distance_remaining
        self._update_status("navigating", distance=round(remaining, 2))

    def _result_cb(self, future):
        result = future.result()
        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Nav2 goal reached!")
            self._nav2_status = "succeeded"
            self._update_status("succeeded")
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info("Nav2 goal cancelled")
            self._nav2_status = "cancelled"
            self._update_status("cancelled")
        else:
            self.get_logger().warn(f"Nav2 goal failed (status={status})")
            self._nav2_status = "failed"
            self._update_status("failed")
        self._goal_handle = None

    def _cancel_goal(self):
        if self._goal_handle is not None:
            self.get_logger().info("Cancelling Nav2 goal")
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass
            self._goal_handle = None
        self._nav2_status = "cancelled"
        self._v = 0.0
        self._w = 0.0
        self._update_status("cancelled")

    def _update_status(self, status, distance=None):
        try:
            data = {"status": status, "t": round(time.time(), 3)}
            if distance is not None:
                data["distance"] = distance
            tmp = STATUS_FILE + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, STATUS_FILE)
        except Exception:
            pass

    # ── Scene Graph Visualization ──

    # Colors by category
    _TAG_COLORS = {
        "chair": (0.2, 0.6, 1.0),
        "table": (0.8, 0.5, 0.2),
        "couch": (0.6, 0.3, 0.8),
        "sofa": (0.6, 0.3, 0.8),
        "bed": (0.9, 0.4, 0.6),
        "refrigerator": (0.3, 0.8, 0.4),
        "tv": (1.0, 0.8, 0.2),
        "sink": (0.4, 0.7, 0.9),
        "toilet": (0.9, 0.9, 0.9),
        "door": (0.5, 0.5, 0.5),
        "window": (0.7, 0.9, 1.0),
        "cabinet": (0.7, 0.5, 0.3),
        "shelf": (0.6, 0.6, 0.4),
        "lamp": (1.0, 1.0, 0.5),
        "plant": (0.2, 0.8, 0.3),
        "book": (0.8, 0.2, 0.2),
        "bottle": (0.3, 0.9, 0.7),
        "cup": (1.0, 0.6, 0.3),
        "bowl": (0.9, 0.7, 0.5),
    }

    def _get_tag_color(self, tag):
        tag_l = tag.lower()
        for key, rgb in self._TAG_COLORS.items():
            if key in tag_l:
                return rgb
        # Hash-based color for unknown tags
        h = hash(tag_l) % 360
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.7, 0.9)
        return (r, g, b)

    def _publish_sg_markers(self):
        """Read scene graph from file and publish as MarkerArray to RViz."""
        try:
            mtime = os.path.getmtime(SG_VIZ_FILE)
        except OSError:
            return
        if mtime <= self._last_sg_mtime:
            return
        self._last_sg_mtime = mtime

        try:
            with open(SG_VIZ_FILE, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        objects = data.get("objects", [])
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, obj in enumerate(objects):
            tag = obj["tag"]
            r, g, b = self._get_tag_color(tag)
            is_new = obj.get("is_new", False)

            # Wireframe bounding box (LINE_LIST)
            bbox_marker = Marker()
            bbox_marker.header.frame_id = "odom"
            bbox_marker.header.stamp = now
            bbox_marker.ns = "sg_bbox"
            bbox_marker.id = i
            bbox_marker.type = Marker.LINE_LIST
            bbox_marker.action = Marker.ADD
            bbox_marker.scale.x = 0.02  # line width
            bbox_marker.color = ColorRGBA(r=r, g=g, b=b, a=0.8)
            bbox_marker.lifetime.sec = 3

            cx, cy, cz = obj["x"], obj["y"], obj["z"]
            ex, ey, ez = obj["ex"] / 2, obj["ey"] / 2, obj["ez"] / 2

            # 8 corners of the box
            corners = [
                (cx - ex, cy - ey, cz - ez),
                (cx + ex, cy - ey, cz - ez),
                (cx + ex, cy + ey, cz - ez),
                (cx - ex, cy + ey, cz - ez),
                (cx - ex, cy - ey, cz + ez),
                (cx + ex, cy - ey, cz + ez),
                (cx + ex, cy + ey, cz + ez),
                (cx - ex, cy + ey, cz + ez),
            ]

            # 12 edges of the box
            edges = [
                (0,1),(1,2),(2,3),(3,0),  # bottom
                (4,5),(5,6),(6,7),(7,4),  # top
                (0,4),(1,5),(2,6),(3,7),  # vertical
            ]
            for a, b_idx in edges:
                pa = Point(x=corners[a][0], y=corners[a][1], z=corners[a][2])
                pb = Point(x=corners[b_idx][0], y=corners[b_idx][1], z=corners[b_idx][2])
                bbox_marker.points.append(pa)
                bbox_marker.points.append(pb)

            marker_array.markers.append(bbox_marker)

            # Text label (TEXT_VIEW_FACING)
            text_marker = Marker()
            text_marker.header.frame_id = "odom"
            text_marker.header.stamp = now
            text_marker.ns = "sg_label"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position = Point(
                x=cx, y=cy, z=cz + ez + 0.15)
            text_marker.scale.z = 0.15  # text height
            new_prefix = "[NEW] " if is_new else ""
            text_marker.text = f"{new_prefix}{tag} ({obj['id']})"
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            text_marker.lifetime.sec = 3
            marker_array.markers.append(text_marker)

            # Small sphere at center
            sphere_marker = Marker()
            sphere_marker.header.frame_id = "odom"
            sphere_marker.header.stamp = now
            sphere_marker.ns = "sg_center"
            sphere_marker.id = i
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position = Point(x=cx, y=cy, z=cz)
            sphere_marker.scale.x = 0.08
            sphere_marker.scale.y = 0.08
            sphere_marker.scale.z = 0.08
            sphere_marker.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            sphere_marker.lifetime.sec = 3
            marker_array.markers.append(sphere_marker)

        # Free-space tiles (green transparent cubes on surfaces)
        free_spaces = data.get("free_spaces", [])
        for j, fs in enumerate(free_spaces):
            fs_marker = Marker()
            fs_marker.header.frame_id = "odom"
            fs_marker.header.stamp = now
            fs_marker.ns = "sg_freespace"
            fs_marker.id = j
            fs_marker.type = Marker.CUBE
            fs_marker.action = Marker.ADD
            fs_marker.pose.position = Point(
                x=float(fs["x"]),
                y=float(fs["y"]),
                z=float(fs["z"]) + 0.005,
            )
            fs_marker.scale.x = float(fs["sx"])
            fs_marker.scale.y = float(fs["sy"])
            fs_marker.scale.z = 0.01
            fs_marker.color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.35)
            fs_marker.lifetime.sec = 3
            marker_array.markers.append(fs_marker)

            # Wireframe outline
            fs_wire = Marker()
            fs_wire.header.frame_id = "odom"
            fs_wire.header.stamp = now
            fs_wire.ns = "sg_freespace_edge"
            fs_wire.id = j
            fs_wire.type = Marker.LINE_STRIP
            fs_wire.action = Marker.ADD
            fs_wire.scale.x = 0.015
            fs_wire.color = ColorRGBA(r=0.1, g=0.8, b=0.1, a=0.7)
            fs_wire.lifetime.sec = 3

            fx, fy, fz = float(fs["x"]), float(fs["y"]), float(fs["z"]) + 0.01
            hx, hy = float(fs["sx"]) / 2, float(fs["sy"]) / 2
            corners_fs = [
                (fx - hx, fy - hy, fz),
                (fx + hx, fy - hy, fz),
                (fx + hx, fy + hy, fz),
                (fx - hx, fy + hy, fz),
                (fx - hx, fy - hy, fz),
            ]
            for c in corners_fs:
                fs_wire.points.append(Point(x=c[0], y=c[1], z=c[2]))
            marker_array.markers.append(fs_wire)

        # Depth-based free space tiles (cyan — from depth_freespace.py test mode)
        depth_free = data.get("depth_free_spaces", [])
        for j, fs in enumerate(depth_free):
            # Cyan transparent cube
            df_marker = Marker()
            df_marker.header.frame_id = "odom"
            df_marker.header.stamp = now
            df_marker.ns = "depth_freespace"
            df_marker.id = j
            df_marker.type = Marker.CUBE
            df_marker.action = Marker.ADD
            df_marker.pose.position = Point(
                x=float(fs["x"]),
                y=float(fs["y"]),
                z=float(fs["z"]) + 0.015,
            )
            df_marker.scale.x = float(fs["sx"])
            df_marker.scale.y = float(fs["sy"])
            df_marker.scale.z = 0.008
            df_marker.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.4)
            df_marker.lifetime.sec = 3
            marker_array.markers.append(df_marker)

            # Cyan wireframe outline
            df_wire = Marker()
            df_wire.header.frame_id = "odom"
            df_wire.header.stamp = now
            df_wire.ns = "depth_freespace_edge"
            df_wire.id = j
            df_wire.type = Marker.LINE_STRIP
            df_wire.action = Marker.ADD
            df_wire.scale.x = 0.012
            df_wire.color = ColorRGBA(r=0.0, g=0.7, b=0.9, a=0.8)
            df_wire.lifetime.sec = 3

            fx, fy, fz = float(fs["x"]), float(fs["y"]), float(fs["z"]) + 0.02
            hx, hy = float(fs["sx"]) / 2, float(fs["sy"]) / 2
            for c in [
                (fx - hx, fy - hy, fz),
                (fx + hx, fy - hy, fz),
                (fx + hx, fy + hy, fz),
                (fx - hx, fy + hy, fz),
                (fx - hx, fy - hy, fz),
            ]:
                df_wire.points.append(Point(x=c[0], y=c[1], z=c[2]))
            marker_array.markers.append(df_wire)

        # Delete old markers
        delete_marker = Marker()
        delete_marker.header.frame_id = "odom"
        delete_marker.header.stamp = now
        delete_marker.ns = "sg_bbox"
        delete_marker.action = Marker.DELETEALL
        delete_marker.id = 0

        self._sg_pub.publish(marker_array)


def main():
    rclpy.init()
    node = Nav2Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        for f in [CMD_VEL_FILE, CANCEL_FLAG, STATUS_FILE, SG_VIZ_FILE]:
            try:
                os.remove(f)
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
