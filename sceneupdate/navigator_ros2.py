"""
ROS2 Nav2 Navigator for Isaac Sim.

Uses Nav2 stack via file-based IPC bridge (nav2_bridge.py):
- OmniGraph: LiDAR → /scan, Odom → /odom, TF
- File IPC: /tmp/nav2_goal.json, /tmp/nav2_cmd_vel.json, /tmp/nav2_status.json
- nav2_bridge.py (separate Python 3.12 process) handles ROS2 action client

No rclpy import needed — avoids Python 3.11/3.12 version mismatch.

Usage in main.py:
    from navigator_ros2 import ROS2Navigator
    navigator = ROS2Navigator(robot_path, base_link_path, dc, art,
                              left_idxs, right_idxs, lidar_path)
"""
import os
import json
import time
import numpy as np

from robot_utils import get_prim_transform, quat_to_euler_z
from config import ARRIVAL_TOLERANCE, APPROACH_DISTANCE, WHEEL_RADIUS, WHEEL_BASE, SKID_STEER_MULTIPLIER

import omni.graph.core as og


# File paths (must match nav2_bridge.py)
GOAL_FILE = "/tmp/nav2_goal.json"
CMD_VEL_FILE = "/tmp/nav2_cmd_vel.json"
CANCEL_FLAG = "/tmp/nav2_cancel.flag"
STATUS_FILE = "/tmp/nav2_status.json"


class _DummyDraw:
    """Stub so sayplan_ui can call planner.draw.clear_lines() without crashing."""
    def clear_lines(self):
        pass
    def draw_lines(self, *a, **kw):
        pass


class _DummyPlanner:
    def __init__(self):
        self.draw = _DummyDraw()


class ROS2Navigator:
    """
    Navigator that delegates path planning to ROS2 Nav2 via file-based bridge.

    Interface is identical to LocalNavigator so sayplan_ui.py works with either.
    """

    WHEEL_BASE = WHEEL_BASE
    WHEEL_RADIUS = WHEEL_RADIUS

    def __init__(self, robot_path, base_link_path, dc, art,
                 left_idxs, right_idxs, lidar_path,
                 robot_start_pos=None, robot_start_yaw=0.0,
                 lidar_frame="base_link"):
        self.robot_path = robot_path
        self.base_link_path = base_link_path
        self.dc = dc
        self.art = art
        self.left_idxs = left_idxs
        self.right_idxs = right_idxs
        self.lidar_path = lidar_path
        self.lidar_frame = lidar_frame

        # Coordinate transform: Isaac Sim world → odom frame
        if robot_start_pos is not None:
            self._start_pos = np.array(robot_start_pos[:2])
        else:
            pos, _ = get_prim_transform(base_link_path)
            self._start_pos = np.array(pos[:2])
        self._start_yaw = robot_start_yaw
        print(f"[ROS2Nav] Start pos: {self._start_pos}, yaw: {np.degrees(self._start_yaw):.1f}°")

        # Dummy planner for interface compat (sayplan_ui accesses planner.draw)
        self.planner = _DummyPlanner()

        # Navigation state
        self.goal = None
        self.object_pos = None
        self.goal_yaw = None
        self.arrival_dist = APPROACH_DISTANCE
        self.is_navigating = False
        self.is_arrived = False
        self.status_text = "idle"
        self._state = "idle"  # idle / moving / aligning

        # Velocity from Nav2 (read from OmniGraph SubscribeTwist)
        self._nav2_v = 0.0
        self._nav2_w = 0.0
        self._last_cmd_time = 0.0
        self._last_file_read = 0.0

        # Nav2 status from bridge
        self._nav2_status = "idle"
        self._last_status_read = 0.0

        # OmniGraph twist node path (set after graph creation)
        self._twist_node = None

        # Cooldown: ignore external cmd_vel for N seconds after full_stop
        self._stop_until = 0.0

        # Setup OmniGraph for sensor publishing
        self._setup_omnigraph()

        # Clean up old IPC files
        for f in [GOAL_FILE, CMD_VEL_FILE, CANCEL_FLAG, STATUS_FILE]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

        print("[ROS2Nav] File-based bridge mode initialized")
        print("[ROS2Nav] Make sure nav2_bridge.py is running in a ROS2 terminal!")

    # ── OmniGraph Setup ──

    def _setup_omnigraph(self):
        """OmniGraph: Isaac Sim sensors → ROS2 topics (/scan, /odom, TF, /clock)."""
        graph_path = "/ROS2NavGraph"

        # Remove existing graph if present
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if stage.GetPrimAtPath(graph_path).IsValid():
                stage.RemovePrim(graph_path)
        except Exception:
            pass

        # Try isaacsim.* namespace first (newer Isaac Sim), then omni.isaac.* fallback
        node_sets = [
            {  # Newer Isaac Sim (isaacsim.* namespace)
                "label": "isaacsim.*",
                "ReadSimTime": "isaacsim.core.nodes.IsaacReadSimulationTime",
                "Ctx": "isaacsim.ros2.bridge.ROS2Context",
                "SubscribeTwist": "isaacsim.ros2.bridge.ROS2SubscribeTwist",
                "ReadLidar": "isaacsim.sensors.physx.IsaacReadLidarBeams",
                "PubScan": "isaacsim.ros2.bridge.ROS2PublishLaserScan",
                "PubClock": "isaacsim.ros2.bridge.ROS2PublishClock",
                "CompOdom": "isaacsim.core.nodes.IsaacComputeOdometry",
                "PubOdom": "isaacsim.ros2.bridge.ROS2PublishOdometry",
                "PubTF": "isaacsim.ros2.bridge.ROS2PublishRawTransformTree",
            },
            {  # Older Isaac Sim (omni.isaac.* namespace)
                "label": "omni.isaac.*",
                "ReadSimTime": "omni.isaac.core_nodes.OgnIsaacReadSimulationTime",
                "Ctx": "omni.isaac.ros2_bridge.ROS2Context",
                "SubscribeTwist": "omni.isaac.ros2_bridge.ROS2SubscribeTwist",
                "ReadLidar": "omni.isaac.range_sensor.IsaacReadLidarBeams",
                "PubScan": "omni.isaac.ros2_bridge.ROS2PublishLaserScan",
                "PubClock": "omni.isaac.ros2_bridge.ROS2PublishClock",
                "CompOdom": "omni.isaac.core_nodes.OgnIsaacComputeOdometry",
                "PubOdom": "omni.isaac.ros2_bridge.ROS2PublishOdometry",
                "PubTF": "omni.isaac.ros2_bridge.ROS2PublishRawTransformTree",
            },
        ]

        for ns in node_sets:
            try:
                self._create_omnigraph(graph_path, ns)
                self._twist_node = f"{graph_path}/SubscribeTwist"
                print(f"[ROS2Nav] OmniGraph created ({ns['label']}): /scan, /odom, /clock, TF, SubscribeTwist(/cmd_vel)")
                return
            except Exception as e:
                print(f"[ROS2Nav] OmniGraph failed with {ns['label']}: {e}")

        print("[ROS2Nav] All OmniGraph attempts failed!")
        print("[ROS2Nav] Set up OmniGraph manually in Isaac Sim GUI (Action Graph).")

    def _create_omnigraph(self, graph_path, ns):
        """Create OmniGraph with given node type namespace."""
        keys = og.Controller.Keys

        nodes = [
            ("OnTick",     "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", ns["ReadSimTime"]),
            ("Ctx",        ns["Ctx"]),
            ("SubscribeTwist", ns["SubscribeTwist"]),
            ("ReadLidar",  ns["ReadLidar"]),
            ("PubScan",    ns["PubScan"]),
            ("PubClock",   ns["PubClock"]),
            ("CompOdom",   ns["CompOdom"]),
            ("PubOdom",    ns["PubOdom"]),
            ("PubTF",      ns["PubTF"]),
        ]

        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {keys.CREATE_NODES: nodes},
        )

        # Helper to connect nodes
        def connect(src, dst):
            og.Controller.connect(
                f"{graph_path}/{src}", f"{graph_path}/{dst}")

        # Tick → ReadSimTime (needs execution trigger)
        try:
            connect("OnTick.outputs:tick", "ReadSimTime.inputs:execIn")
        except Exception:
            pass  # Some versions don't have execIn on ReadSimTime

        # Tick → all compute/publish nodes
        for target in ["SubscribeTwist", "ReadLidar", "PubScan", "PubClock",
                        "CompOdom", "PubOdom", "PubTF"]:
            connect("OnTick.outputs:tick", f"{target}.inputs:execIn")

        # ROS2 context
        for target in ["SubscribeTwist", "PubScan", "PubClock", "PubOdom", "PubTF"]:
            connect("Ctx.outputs:context", f"{target}.inputs:context")

        # Simulation time → timestamps
        for target in ["PubScan", "PubClock", "PubOdom", "PubTF"]:
            connect("ReadSimTime.outputs:simulationTime",
                    f"{target}.inputs:timeStamp")

        # LiDAR data → scan publisher
        for field in ["linearDepthData", "intensitiesData",
                       "numCols", "numRows", "azimuthRange",
                       "depthRange", "horizontalFov",
                       "horizontalResolution", "rotationRate"]:
            connect(f"ReadLidar.outputs:{field}", f"PubScan.inputs:{field}")

        # Odometry data → odom publisher + TF
        for field in ["position", "orientation",
                       "linearVelocity", "angularVelocity"]:
            connect(f"CompOdom.outputs:{field}", f"PubOdom.inputs:{field}")
        connect("CompOdom.outputs:position", "PubTF.inputs:translation")
        connect("CompOdom.outputs:orientation", "PubTF.inputs:rotation")

        # Set values
        def set_val(node_attr, val):
            og.Controller.set(
                og.Controller.attribute(f"{graph_path}/{node_attr}"), val)

        set_val("SubscribeTwist.inputs:topicName", "/cmd_vel")
        set_val("ReadLidar.inputs:lidarPrim", [self.lidar_path])
        set_val("PubScan.inputs:topicName", "/scan")
        set_val("PubScan.inputs:frameId", self.lidar_frame)
        set_val("CompOdom.inputs:chassisPrim", self.base_link_path)
        set_val("PubOdom.inputs:topicName", "/odom")
        set_val("PubOdom.inputs:odomFrameId", "odom")
        set_val("PubOdom.inputs:chassisFrameId", "base_link")
        set_val("PubTF.inputs:parentFrameId", "odom")
        set_val("PubTF.inputs:childFrameId", "base_link")

    # ── Coordinate Transform ──

    def _isaac_to_odom(self, isaac_pos):
        """Isaac Sim world coordinates → odom frame (translation + rotation)."""
        translated = np.array([isaac_pos[0] - self._start_pos[0],
                               isaac_pos[1] - self._start_pos[1]])
        cos_t = np.cos(-self._start_yaw)
        sin_t = np.sin(-self._start_yaw)
        x_odom = translated[0] * cos_t - translated[1] * sin_t
        y_odom = translated[0] * sin_t + translated[1] * cos_t
        return np.array([x_odom, y_odom])

    # ── File-based IPC ──

    def _read_cmd_vel(self):
        """Read velocity from OmniGraph SubscribeTwist node (direct ROS2, no file IPC)."""
        now = time.time()

        # During cooldown, force zero
        if now < self._stop_until:
            self._nav2_v = 0.0
            self._nav2_w = 0.0
            return

        # Primary: read from OmniGraph SubscribeTwist (like old working code)
        if self._twist_node is not None:
            try:
                lin = og.Controller.get(
                    og.Controller.attribute(f"{self._twist_node}.outputs:linearVelocity"))
                ang = og.Controller.get(
                    og.Controller.attribute(f"{self._twist_node}.outputs:angularVelocity"))
                if lin is not None and ang is not None:
                    v = float(lin[0])
                    w = float(ang[2])
                    if abs(v) > 0.001 or abs(w) > 0.001:
                        self._nav2_v = v
                        self._nav2_w = w
                        self._last_cmd_time = now
                        return
            except Exception:
                pass  # Fall through to file-based fallback

        # Fallback: file-based IPC (in case OmniGraph SubscribeTwist failed or returned 0)
        if now - self._last_file_read < 0.03:
            return
        self._last_file_read = now
        try:
            with open(CMD_VEL_FILE, 'r') as f:
                data = json.load(f)
            cmd_t = float(data.get("t", 0.0))
            if cmd_t > 0 and (now - cmd_t) < 1.0:
                self._nav2_v = float(data.get("v", 0.0))
                self._nav2_w = float(data.get("w", 0.0))
                self._last_cmd_time = cmd_t
            else:
                self._nav2_v = 0.0
                self._nav2_w = 0.0
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass

    def _read_nav2_status(self):
        """Read Nav2 status from bridge."""
        now = time.time()
        if now - self._last_status_read < 0.2:  # 5Hz
            return
        self._last_status_read = now

        try:
            with open(STATUS_FILE, 'r') as f:
                data = json.load(f)
            self._nav2_status = data.get("status", "unknown")
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass

    def _write_goal(self, goal_pos, yaw=None):
        """Write goal to file for nav2_bridge.py to read (transforms to odom frame)."""
        try:
            odom_pos = self._isaac_to_odom(goal_pos)
            odom_yaw = yaw - self._start_yaw if yaw is not None else None
            data = {
                "x": round(float(odom_pos[0]), 4),
                "y": round(float(odom_pos[1]), 4),
                "t": round(time.time(), 3),
            }
            if odom_yaw is not None:
                data["yaw"] = round(float(odom_yaw), 4)
            tmp = GOAL_FILE + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, GOAL_FILE)
            pass  # goal written
        except Exception as e:
            print(f"[ROS2Nav] Goal write error: {e}")

    def _write_cancel(self):
        """Signal cancellation to nav2_bridge.py."""
        try:
            with open(CANCEL_FLAG, 'w') as f:
                f.write("cancel")
        except Exception:
            pass

    # ── Interface (same as LocalNavigator) ──

    @property
    def navigating(self):
        return self.is_navigating

    @property
    def arrived(self):
        return self.is_arrived

    def _full_stop(self):
        """Immediately stop robot and cancel Nav2. Block external cmd for 1.5s."""
        self._cmd_vel(0, 0)
        self._nav2_v = 0.0
        self._nav2_w = 0.0
        self._last_cmd_time = 0.0  # invalidate freshness check
        self._write_cancel()
        self._stop_until = time.time() + 0.3  # ignore external cmd briefly
        # Overwrite cmd_vel file with zeros
        try:
            tmp = CMD_VEL_FILE + ".tmp"
            with open(tmp, 'w') as f:
                json.dump({"v": 0, "w": 0, "t": time.time()}, f)
            os.replace(tmp, CMD_VEL_FILE)
        except Exception:
            pass

    def set_goal(self, goal_pos, yaw=None, object_pos=None, arrival_dist=None):
        # Stop first, then send new goal
        self._full_stop()

        self.goal = np.array([goal_pos[0], goal_pos[1]])
        self.object_pos = (np.array([object_pos[0], object_pos[1]])
                           if object_pos is not None else self.goal.copy())
        self.goal_yaw = yaw
        self.arrival_dist = (arrival_dist if arrival_dist is not None
                             else APPROACH_DISTANCE)
        self.is_navigating = True
        self.is_arrived = False
        self._state = "moving"

        self._write_goal(goal_pos, yaw)

    def update_goal(self, goal_pos, yaw=None, object_pos=None, arrival_dist=None):
        """Update internal goal coords. Only resend to Nav2 if goal moved significantly.
        Does NOT stop the robot — just updates target smoothly."""
        new_goal = np.array([goal_pos[0], goal_pos[1]])

        # Check if goal moved significantly (> 0.8m) — only then resend to Nav2
        resend = False
        if self.goal is not None:
            if np.linalg.norm(new_goal - self.goal) > 0.8:
                resend = True
        else:
            resend = True

        self.goal = new_goal
        if object_pos is not None:
            self.object_pos = np.array([object_pos[0], object_pos[1]])
        if yaw is not None:
            self.goal_yaw = yaw
        if arrival_dist is not None:
            self.arrival_dist = arrival_dist

        if resend:
            # Don't full_stop — just send new goal without interrupting movement
            self._write_goal(goal_pos, yaw)

    def cancel(self):
        self._full_stop()
        self.is_navigating = False
        self.goal = None
        self.object_pos = None
        self._state = "idle"
        self.status_text = "cancelled"

    def update(self):
        now = time.time()

        # Always read cmd_vel from bridge
        self._read_cmd_vel()
        self._read_nav2_status()

        # During cooldown after full_stop, force zero velocity
        in_cooldown = now < self._stop_until
        if in_cooldown:
            self._cmd_vel(0, 0)

        # If not navigating (Brain goal), handle external RViz goals
        if not self.is_navigating or self.goal is None:
            if not in_cooldown and self._is_fresh_cmd():
                self._cmd_vel(self._nav2_v, self._nav2_w)
                return "ros2_external"
            if not in_cooldown:
                self._cmd_vel(0, 0)
            return "idle"

        # ── Aligning state: face object after Nav2 arrival ──
        if self._state == "aligning":
            # Wait for cooldown to finish before starting rotation
            if in_cooldown:
                return "stopping"
            robot_pos, robot_quat = get_prim_transform(self.base_link_path)
            robot_xy = robot_pos[:2]
            robot_yaw = quat_to_euler_z(robot_quat)

            face_dir = self.object_pos - robot_xy
            if np.linalg.norm(face_dir) > 0.05:
                face_yaw = np.arctan2(face_dir[1], face_dir[0])
            elif self.goal_yaw is not None:
                face_yaw = self.goal_yaw
            else:
                face_yaw = None

            if face_yaw is not None:
                yaw_err = self._angle_diff(face_yaw, robot_yaw)
                if abs(yaw_err) > 0.15:
                    w = np.clip(yaw_err * 2.0, -1.0, 1.0)
                    self._cmd_vel(0, w)
                    self.status_text = "aligning"
                    return self.status_text

            # Alignment done
            self._cmd_vel(0, 0)
            self.is_navigating = False
            self.is_arrived = True
            self._state = "idle"
            self.status_text = "arrived"
            pass  # aligned
            return self.status_text

        # ── Moving state: Nav2 driving ──
        robot_pos, _ = get_prim_transform(self.base_link_path)
        robot_xy = robot_pos[:2]

        dist_to_goal = float(np.linalg.norm(np.array(self.goal).ravel()[:2] - robot_xy[:2]))
        obj_xy = np.array(self.object_pos).ravel()[:2] if self.object_pos is not None else self.goal.ravel()[:2]
        dist_to_obj = float(np.linalg.norm(obj_xy - robot_xy[:2]))


        # Arrival check — Isaac Sim is authoritative
        if dist_to_goal <= ARRIVAL_TOLERANCE or dist_to_obj <= ARRIVAL_TOLERANCE:
            self._full_stop()  # Cancel Nav2 and stop
            self._state = "aligning"
            self.status_text = "aligning"
            return self.status_text

        # Apply Nav2 velocity with smoothing (skip during cooldown)
        if in_cooldown:
            return "stopping"
        self._cmd_vel(self._nav2_v, self._nav2_w)

        self.status_text = (f"ros2 d={dist_to_goal:.1f}m "
                            f"v={self._nav2_v:.2f} w={self._nav2_w:.2f}")
        return self.status_text

    # ── Helpers ──

    def _is_fresh_cmd(self):
        """Check if cmd_vel file has recent non-zero data."""
        now = time.time()
        return (
            self._last_cmd_time > 0
            and (now - self._last_cmd_time) < 2.0
            and (abs(self._nav2_v) > 0.01 or abs(self._nav2_w) > 0.01)
        )

    # ── Wheel Control ──

    def _cmd_vel(self, v, w):
        """Apply velocity to wheels via DC API.
        With UsdPhysics.DriveAPI damping=100000, physics engine handles torque limits.
        """
        effective_w = w * SKID_STEER_MULTIPLIER
        vl = (v - effective_w * self.WHEEL_BASE / 2.0) / self.WHEEL_RADIUS
        vr = (v + effective_w * self.WHEEL_BASE / 2.0) / self.WHEEL_RADIUS

        try:
            for i in self.left_idxs:
                self.dc.set_dof_velocity_target(
                    self.dc.get_articulation_dof(self.art, i), float(vl))
            for i in self.right_idxs:
                self.dc.set_dof_velocity_target(
                    self.dc.get_articulation_dof(self.art, i), float(vr))
        except Exception as e:
            print(f"[WHEEL_ERR] Failed to set velocity: {e}")

    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return d
