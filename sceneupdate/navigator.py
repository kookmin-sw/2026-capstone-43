import time
import math
import numpy as np

from robot_utils import get_prim_transform, quat_to_euler_z
from config import (MOVE_LINEAR_SPEED, MOVE_ANGULAR_SPEED,
                     ARRIVAL_TOLERANCE, APPROACH_DISTANCE,
                     WHEEL_RADIUS, WHEEL_BASE, SKID_STEER_MULTIPLIER)
from isaacsim.util.debug_draw import _debug_draw


# ====================================================================
# DWA Local Planner (Nav2-quality)
# - Smooth velocity profiles with proper acceleration limits
# - Proximity-based speed regulation (slow near obstacles)
# - Backward motion allowed for tight spaces
# - Smooth obstacle cost gradient (not binary)
# ====================================================================
class DWA_Planner:
    def __init__(self):
        # Robot velocity limits (indoor-appropriate)
        self.max_v = 0.5              # m/s forward (smooth indoor speed)
        self.min_v = -0.15            # m/s backward (for tight spaces)
        self.max_w = 1.5              # rad/s (smooth turning)
        self.max_accel_v = 1.0        # m/s^2 (gentle acceleration)
        self.max_accel_w = 3.0        # rad/s^2 (responsive but not jerky)

        # Trajectory simulation
        self.sim_time = 2.0           # prediction horizon (seconds)
        self.dt = 0.1                 # simulation step
        self.v_samples = 11           # velocity samples
        self.w_samples = 25           # angular velocity samples

        # Cost function weights
        self.goal_cost_gain = 4.0     # heading toward goal
        self.speed_cost_gain = 0.8    # prefer faster movement
        self.obstacle_cost_gain = 6.0 # avoid obstacles (smooth gradient)
        self.straight_cost_gain = 1.0 # prefer straight paths
        self.dist_cost_gain = 1.5     # prefer trajectories that reduce distance to goal

        # Robot safety
        self.robot_radius = 0.40      # collision check radius
        self.inflation_radius = 1.0   # soft obstacle cost starts here

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def plan(self, robot_xy, robot_yaw, curr_v, curr_w, goal_xy, ranges, angles):
        """Plan optimal (v, w) using DWA with smooth cost evaluation."""
        # 1. Build obstacle points (robot-local frame)
        obs_list = np.empty((0, 2), dtype=np.float32)
        if len(ranges) > 0:
            valid_mask = (ranges > 0.35) & (ranges < 5.0)
            valid_ranges = ranges[valid_mask]
            valid_angles = angles[valid_mask]
            if len(valid_ranges) > 0:
                obs_x = valid_ranges * np.cos(valid_angles)
                obs_y = valid_ranges * np.sin(valid_angles)
                obs_list = np.column_stack((obs_x, obs_y))
                if len(obs_list) > 100:
                    step = len(obs_list) // 100
                    obs_list = obs_list[::step]

        # 2. Goal in robot-local frame
        dx = goal_xy[0] - robot_xy[0]
        dy = goal_xy[1] - robot_xy[1]
        local_goal_x = dx * np.cos(-robot_yaw) - dy * np.sin(-robot_yaw)
        local_goal_y = dx * np.sin(-robot_yaw) + dy * np.cos(-robot_yaw)
        dist_to_goal = math.sqrt(local_goal_x**2 + local_goal_y**2)

        # 3. Proximity-based speed regulation (Nav2 style)
        # Slow down near obstacles for safety
        min_clearance = self._get_min_clearance(obs_list)
        speed_limit = self.max_v
        if min_clearance < self.inflation_radius:
            # Linear ramp: full speed at inflation_radius, 30% at robot_radius
            ratio = max(0.0, (min_clearance - self.robot_radius) /
                        (self.inflation_radius - self.robot_radius))
            speed_limit = max(0.1, self.max_v * (0.3 + 0.7 * ratio))

        # Also slow down near the goal for smooth arrival
        if dist_to_goal < 1.0:
            speed_limit = min(speed_limit, max(0.1, self.max_v * dist_to_goal))

        # 4. Dynamic window
        ctrl_dt = 0.1  # control period
        dw_v_min = max(self.min_v, curr_v - self.max_accel_v * ctrl_dt)
        dw_v_max = min(speed_limit, curr_v + self.max_accel_v * ctrl_dt)
        dw_w_min = max(-self.max_w, curr_w - self.max_accel_w * ctrl_dt)
        dw_w_max = min(self.max_w, curr_w + self.max_accel_w * ctrl_dt)

        # 5. Sample and evaluate trajectories
        best_u = (0.0, 0.0)
        best_traj = []
        min_cost = float('inf')

        vs = np.linspace(dw_v_min, dw_v_max, self.v_samples)
        ws = np.linspace(dw_w_min, dw_w_max, self.w_samples)

        # Ensure v=0 is always sampled (for in-place rotation)
        if dw_v_min <= 0 <= dw_v_max and 0.0 not in vs:
            vs = np.append(vs, 0.0)

        for v in vs:
            for w in ws:
                traj = self._simulate_trajectory(v, w)
                cost = self._evaluate_trajectory(
                    traj, v, w, local_goal_x, local_goal_y, obs_list, dist_to_goal)
                if math.isnan(cost):
                    continue
                if cost < min_cost:
                    min_cost = cost
                    best_u = (v, w)
                    best_traj = traj

        # 6. Visualization
        if best_traj:
            world_traj = []
            for (lx, ly, _) in best_traj:
                wx = robot_xy[0] + lx * np.cos(robot_yaw) - ly * np.sin(robot_yaw)
                wy = robot_xy[1] + lx * np.sin(robot_yaw) + ly * np.cos(robot_yaw)
                world_traj.append([wx, wy, 0.2])
            self._draw_trajectory(world_traj)
        else:
            self.draw.clear_lines()

        # 7. If all trajectories collide, try in-place rotation
        if min_cost == float('inf'):
            best_spin_w = 0.0
            max_clearance = 0.0
            for w in np.linspace(-self.max_w, self.max_w, 15):
                traj = self._simulate_trajectory(0.0, w)
                clearance = self._min_traj_clearance(traj, obs_list)
                if clearance > max_clearance:
                    max_clearance = clearance
                    best_spin_w = w
            # If still stuck, try backing up
            if max_clearance < self.robot_radius:
                return (-0.1, best_spin_w * 0.5), []
            return (0.0, best_spin_w), []

        return best_u, best_traj

    def _simulate_trajectory(self, v, w):
        """Simulate trajectory for sim_time seconds."""
        traj = []
        x, y, theta = 0.0, 0.0, 0.0
        steps = int(self.sim_time / self.dt)
        for _ in range(steps):
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            theta += w * self.dt
            traj.append((x, y, theta))
        return traj

    def _evaluate_trajectory(self, traj, v, w, goal_x, goal_y, obs_list, dist_to_goal):
        """Evaluate trajectory cost with smooth gradients."""
        if not traj:
            return float('inf')

        # 1. Obstacle cost — smooth gradient, hard reject on collision
        min_obs_dist = float('inf')
        obstacle_cost = 0.0
        has_obs = obs_list.shape[0] > 0

        if has_obs:
            for (x, y, _) in traj:
                dists_sq = (obs_list[:, 0] - x)**2 + (obs_list[:, 1] - y)**2
                d = math.sqrt(np.min(dists_sq))
                if d < self.robot_radius:
                    return float('inf')  # collision → reject
                if d < min_obs_dist:
                    min_obs_dist = d

            # Smooth gradient: cost increases as distance decreases
            if min_obs_dist < self.inflation_radius:
                # Exponential cost near obstacles
                proximity = 1.0 - (min_obs_dist - self.robot_radius) / \
                    (self.inflation_radius - self.robot_radius)
                proximity = max(0.0, min(1.0, proximity))
                obstacle_cost = proximity ** 2 * 10.0

        # 2. Goal heading cost
        final_x, final_y, final_theta = traj[-1]
        angle_to_goal = math.atan2(goal_y - final_y, goal_x - final_x)
        heading_err = abs(math.atan2(
            math.sin(angle_to_goal - final_theta),
            math.cos(angle_to_goal - final_theta)))
        goal_cost = heading_err

        # 3. Distance reduction cost
        final_dist = math.sqrt((goal_x - final_x)**2 + (goal_y - final_y)**2)
        dist_cost = final_dist

        # 4. Speed cost (prefer forward motion, penalize reverse slightly)
        if v >= 0:
            speed_cost = self.max_v - v
        else:
            speed_cost = self.max_v + abs(v) * 2.0  # penalize backward more

        # 5. Smoothness cost (prefer straight paths)
        straight_cost = abs(w)

        # 6. In-place rotation bonus: if goal is behind, encourage turning
        if v == 0.0:
            goal_angle = math.atan2(goal_y, goal_x)
            heading_after = abs(math.atan2(
                math.sin(goal_angle - w * self.sim_time),
                math.cos(goal_angle - w * self.sim_time)))
            goal_cost = heading_after * 1.5

        total_cost = (self.goal_cost_gain * goal_cost +
                      self.obstacle_cost_gain * obstacle_cost +
                      self.speed_cost_gain * speed_cost +
                      self.straight_cost_gain * straight_cost +
                      self.dist_cost_gain * dist_cost)
        return total_cost

    def _get_min_clearance(self, obs_list):
        """Get minimum distance from robot center to nearest obstacle."""
        if obs_list.shape[0] == 0:
            return float('inf')
        dists = np.sqrt(obs_list[:, 0]**2 + obs_list[:, 1]**2)
        return float(np.min(dists))

    def _min_traj_clearance(self, traj, obs_list):
        """Get minimum clearance along a trajectory."""
        if obs_list.shape[0] == 0:
            return float('inf')
        min_d = float('inf')
        for (x, y, _) in traj:
            dists_sq = (obs_list[:, 0] - x)**2 + (obs_list[:, 1] - y)**2
            d = math.sqrt(np.min(dists_sq))
            if d < min_d:
                min_d = d
        return min_d

    def _draw_trajectory(self, pts):
        self.draw.clear_lines()
        if len(pts) < 2:
            return
        starts = pts[:-1]
        ends = pts[1:]
        self.draw.draw_lines(starts, ends,
                             [(0, 1, 1, 1)] * len(starts),
                             [5.0] * len(starts))


# ====================================================================
# Navigator — smooth robot controller with velocity ramping
# ====================================================================
class LocalNavigator:

    WHEEL_BASE = WHEEL_BASE
    WHEEL_RADIUS = WHEEL_RADIUS
    MAX_LINEAR_VEL = MOVE_LINEAR_SPEED
    MAX_ANGULAR_VEL = MOVE_ANGULAR_SPEED
    LIDAR_RANGE = 5.0

    # Velocity smoothing (exponential filter)
    SMOOTH_ALPHA_V = 0.3   # lower = smoother (0.3 = gradual accel/decel)
    SMOOTH_ALPHA_W = 0.4   # angular is slightly more responsive

    def __init__(self, robot_path, base_link_path, dc, art,
                 left_idxs, right_idxs, lidar_path):
        self.robot_path = robot_path
        self.base_link_path = base_link_path
        self.dc = dc
        self.art = art
        self.left_idxs = left_idxs
        self.right_idxs = right_idxs
        self.lidar_path = lidar_path

        from omni.isaac.range_sensor import _range_sensor
        self.lidar_iface = _range_sensor.acquire_lidar_sensor_interface()

        # DWA Local Planner
        self.planner = DWA_Planner()

        # Navigation state
        self.goal = None
        self.object_pos = None
        self.goal_yaw = None
        self.arrival_dist = APPROACH_DISTANCE
        self.is_navigating = False
        self.is_arrived = False
        self.status_text = "idle"
        self.curr_v = 0.0
        self.curr_w = 0.0
        self._smooth_v = 0.0   # smoothed velocity output
        self._smooth_w = 0.0

        # Stuck recovery
        self._recovery_mode = False
        self._recovery_start = 0.0
        self.pos_history = []
        self.STUCK_WINDOW = 300
        self.STUCK_THRESHOLD = 0.02
        self.YAW_THRESHOLD = 0.05

    @property
    def navigating(self):
        return self.is_navigating

    @property
    def arrived(self):
        return self.is_arrived

    def set_goal(self, goal_pos, yaw=None, object_pos=None, arrival_dist=None):
        self.goal = np.array([goal_pos[0], goal_pos[1]])
        self.object_pos = (np.array([object_pos[0], object_pos[1]])
                           if object_pos is not None else self.goal.copy())
        self.goal_yaw = yaw
        self.arrival_dist = (arrival_dist if arrival_dist is not None
                             else APPROACH_DISTANCE)
        self.is_navigating = True
        self.is_arrived = False
        self._recovery_mode = False
        self.pos_history = []
        print(f"[Nav] Goal set: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")

    def update_goal(self, goal_pos, yaw=None, object_pos=None, arrival_dist=None):
        """Update goal coordinates without resetting navigation state."""
        self.goal = np.array([goal_pos[0], goal_pos[1]])
        if object_pos is not None:
            self.object_pos = np.array([object_pos[0], object_pos[1]])
        if yaw is not None:
            self.goal_yaw = yaw
        if arrival_dist is not None:
            self.arrival_dist = arrival_dist

    def cancel(self):
        self.is_navigating = False
        self.goal = None
        self.object_pos = None
        self.planner.draw.clear_lines()
        self._cmd_vel(0, 0)
        self._smooth_v = 0.0
        self._smooth_w = 0.0
        self.status_text = "cancelled"

    def update(self):
        if not self.is_navigating or self.goal is None:
            return "idle"

        # ── Robot state ──
        robot_pos, robot_quat = get_prim_transform(self.base_link_path)
        robot_xy = robot_pos[:2]
        robot_yaw = quat_to_euler_z(robot_quat)

        # ── LiDAR ──
        ranges, angles = self._read_lidar()

        # ── Arrival check ──
        dist_to_goal = np.linalg.norm(self.goal - robot_xy)
        dist_to_obj = np.linalg.norm(self.object_pos - robot_xy)

        if dist_to_goal < ARRIVAL_TOLERANCE or dist_to_obj < self.arrival_dist:
            # Final alignment toward object
            face_dir = self.object_pos - robot_xy
            if np.linalg.norm(face_dir) > 0.05:
                face_yaw = np.arctan2(face_dir[1], face_dir[0])
            elif self.goal_yaw is not None:
                face_yaw = self.goal_yaw
            else:
                face_yaw = None

            if face_yaw is not None:
                yaw_err = self._angle_diff(face_yaw, robot_yaw)
                if abs(yaw_err) > 0.1:
                    w = np.clip(yaw_err * 2.0, -1.0, 1.0)
                    self._cmd_vel_smooth(0, w)
                    self.status_text = "aligning"
                    return self.status_text

            self._cmd_vel_smooth(0, 0)
            self._cmd_vel(0, 0)  # hard stop
            self.is_navigating = False
            self.is_arrived = True
            self.planner.draw.clear_lines()
            self.status_text = "arrived"
            print("[Nav] Arrived at goal!")
            return self.status_text

        # ── Recovery Mode ──
        now = time.time()
        if self._recovery_mode:
            elapsed = now - self._recovery_start
            if elapsed < 0.5:
                self._cmd_vel(-0.15, 0.0)
                return "RECOVERY: backing"
            elif elapsed < 1.2:
                self._cmd_vel(0.0, 1.2)
                return "RECOVERY: turning"
            else:
                self._recovery_mode = False
                self.pos_history = []
                self._smooth_v = 0.0
                self._smooth_w = 0.0

        # ── Stuck Detection ──
        self.pos_history.append((robot_xy.copy(), robot_yaw))
        if len(self.pos_history) > self.STUCK_WINDOW:
            self.pos_history.pop(0)

        if (len(self.pos_history) == self.STUCK_WINDOW
                and dist_to_goal > 0.8):
            xy_start = self.pos_history[0][0]
            yaw_start = self.pos_history[0][1]
            xy_end = self.pos_history[-1][0]
            yaw_end = self.pos_history[-1][1]
            dist_moved = float(np.linalg.norm(xy_end - xy_start))
            yaw_diff = abs(self._angle_diff(yaw_end, yaw_start))

            if (dist_moved < self.STUCK_THRESHOLD
                    and yaw_diff < self.YAW_THRESHOLD
                    and abs(self._smooth_v) > 0.05):
                self._recovery_mode = True
                self._recovery_start = now
                self.pos_history = []
                print(f"[Nav] Stuck! (moved {dist_moved:.3f}m). Recovery.")
                return "RECOVERY start"

        # ── DWA Planning ──
        (dwa_v, dwa_w), _ = self.planner.plan(
            robot_xy, robot_yaw, self.curr_v, self.curr_w,
            self.goal, ranges, angles)

        # ── Apply velocity with smoothing ──
        self._cmd_vel_smooth(dwa_v, dwa_w)

        self.status_text = (f"nav d={dist_to_goal:.1f}m "
                            f"v={self._smooth_v:.2f} w={self._smooth_w:.2f}")
        return self.status_text

    def _read_lidar(self):
        try:
            depth = self.lidar_iface.get_linear_depth_data(self.lidar_path)
            if depth is None or len(depth) == 0:
                return np.array([]), np.array([])
            depth = np.array(depth).flatten()
            n = len(depth)
            angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
            valid = (depth > 0.05) & (depth < self.LIDAR_RANGE)
            return depth[valid], angles[valid]
        except Exception:
            return np.array([]), np.array([])

    def _cmd_vel_smooth(self, target_v, target_w):
        """Apply velocity with exponential smoothing for jerk-free motion."""
        self._smooth_v += self.SMOOTH_ALPHA_V * (target_v - self._smooth_v)
        self._smooth_w += self.SMOOTH_ALPHA_W * (target_w - self._smooth_w)

        # Dead-zone: stop completely if near zero
        if abs(self._smooth_v) < 0.01:
            self._smooth_v = 0.0
        if abs(self._smooth_w) < 0.02:
            self._smooth_w = 0.0

        self.curr_v = self._smooth_v
        self.curr_w = self._smooth_w
        self._cmd_vel(self._smooth_v, self._smooth_w)

    def _cmd_vel(self, v, w):
        effective_w = w * SKID_STEER_MULTIPLIER

        vl = (v - effective_w * self.WHEEL_BASE / 2.0) / self.WHEEL_RADIUS
        vr = (v + effective_w * self.WHEEL_BASE / 2.0) / self.WHEEL_RADIUS

        # Safety clamp
        vl = np.clip(vl, -30.0, 30.0)
        vr = np.clip(vr, -30.0, 30.0)

        try:
            for i in self.left_idxs:
                self.dc.set_dof_velocity_target(
                    self.dc.get_articulation_dof(self.art, i), float(vl))
            for i in self.right_idxs:
                self.dc.set_dof_velocity_target(
                    self.dc.get_articulation_dof(self.art, i), float(vr))
        except Exception:
            pass

    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return d
