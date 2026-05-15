"""
FetchArmController (drop-in for FrankaArmController)
- Fetch robot arm: 7-DOF arm + 2-finger gripper
- Arm is part of the main robot articulation → use DC API directly
- Same public interface as old FrankaArmController
  (prepare_articulation, initialize, update, go_home, pick, place,
   open_object, close_object, holding_object)
"""

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.prims import XFormPrim
from robot_utils import get_prim_transform, quat_to_euler_z, find_articulation_root


# ── Fetch joint names ────────────────────────────────────────
FETCH_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]
FETCH_GRIPPER_JOINTS = [
    "l_gripper_finger_joint",
    "r_gripper_finger_joint",
]
FETCH_TORSO_JOINT = "torso_lift_joint"

# Preset joint angles (radians)
# [shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex, wrist_roll]
POSE_REST  = [0.0,  1.32,  1.40,  1.72, 0.0,  1.66,  0.0]   # tucked home
POSE_READY = [0.0,  0.60,  0.00,  1.20, 0.0,  1.20,  0.0]   # pre-reach
POSE_REACH = [0.0,  0.20,  0.00,  0.80, 0.0,  0.80,  0.0]   # low reach

GRIPPER_OPEN  = 0.05   # metres per finger
GRIPPER_CLOSE = 0.001


class FrankaArmController:
    """Fetch arm controller with the same interface as the old Franka controller."""

    def __init__(self, world_or_robot_path, robot_path_or_base_link=None,
                 dc=None):
        if dc is not None:
            self.robot_path = str(world_or_robot_path)
            self.base_link_path = str(robot_path_or_base_link)
            self.dc = dc
        else:
            self.robot_path = str(robot_path_or_base_link) if robot_path_or_base_link else ""
            stage = omni.usd.get_context().get_stage()
            base = self.robot_path + "/base_link"
            self.base_link_path = base if stage and stage.GetPrimAtPath(base).IsValid() else self.robot_path
            self.dc = _dynamic_control.acquire_dynamic_control_interface()

        self.stage = omni.usd.get_context().get_stage()

        # Articulation handle (set after world.play)
        self.art = None
        self.arm_dof_indices  = []   # indices in articulation for arm joints
        self.grip_dof_indices = []   # indices for gripper finger joints
        self.torso_dof_index  = None

        # Gripper tip prim path (for FK / object attach)
        self.gripper_tip_path = self._find_gripper_tip()

        # Held object
        self.held_object_path = None
        self.held_object_xform = None

        # Action queue
        self.action_queue  = []
        self.current_phase = None
        self.phase_timer   = 0
        self.is_busy       = False

        # Current targets
        self._arm_targets  = list(POSE_REST)
        self._grip_target  = GRIPPER_OPEN

    # ----------------------------------------------------------
    # Prim discovery
    # ----------------------------------------------------------
    def _find_gripper_tip(self):
        if not self.stage or not self.robot_path:
            return self.base_link_path
        root = self.stage.GetPrimAtPath(self.robot_path)
        if not root.IsValid():
            return self.base_link_path
        for keyword in ["gripper_link", "wrist_roll_link", "eef_link"]:
            for prim in Usd.PrimRange(root):
                if prim.GetName().lower() == keyword:
                    path = str(prim.GetPath())
                    print(f"[FetchArm] Gripper tip: {path}")
                    return path
        return self.base_link_path

    # ----------------------------------------------------------
    # prepare_articulation — called BEFORE world.play()
    # For Fetch: arm is already part of main robot articulation,
    # no need to break physics. Just a no-op.
    # ----------------------------------------------------------
    def prepare_articulation(self):
        print(f"\n{'='*60}")
        print("[FetchArm] prepare_articulation: Fetch arm uses main articulation")
        print(f"{'='*60}\n")

    # ----------------------------------------------------------
    # initialize — called AFTER world.play()
    # ----------------------------------------------------------
    def initialize(self):
        print(f"\n{'='*60}")
        print("[FetchArm] === INITIALIZE ===")

        self.art, _ = find_articulation_root(self.robot_path, self.dc)
        if not self.art or self.art == 0:
            print("[FetchArm] WARNING: Could not get articulation handle")
            print(f"{'='*60}\n")
            return False

        n_dofs = self.dc.get_articulation_dof_count(self.art)
        print(f"[FetchArm] Articulation DOFs: {n_dofs}")

        self.arm_dof_indices  = []
        self.grip_dof_indices = []
        self.torso_dof_index  = None

        for i in range(n_dofs):
            dof_ptr  = self.dc.get_articulation_dof(self.art, i)
            dof_name = self.dc.get_dof_name(dof_ptr).lower()
            if dof_name in FETCH_ARM_JOINTS:
                self.arm_dof_indices.append(i)
                print(f"[FetchArm] ARM  DOF[{i}]: {dof_name}")
            elif dof_name in FETCH_GRIPPER_JOINTS:
                self.grip_dof_indices.append(i)
                print(f"[FetchArm] GRIP DOF[{i}]: {dof_name}")
            elif dof_name == FETCH_TORSO_JOINT:
                self.torso_dof_index = i
                print(f"[FetchArm] TORSO DOF[{i}]: {dof_name}")

        if not self.arm_dof_indices:
            print("[FetchArm] WARNING: No arm DOFs found!")
            print(f"{'='*60}\n")
            return False

        print(f"[FetchArm] Ready: {len(self.arm_dof_indices)} arm + "
              f"{len(self.grip_dof_indices)} gripper DOFs")
        print(f"{'='*60}\n")

        # Apply initial drive setup
        self._configure_drives()
        return True

    def _configure_drives(self):
        """Set position-control drives for arm and gripper joints."""
        if not self.art or self.art == 0:
            return
        for i in self.arm_dof_indices + self.grip_dof_indices:
            dof = self.dc.get_articulation_dof(self.art, i)
            props = self.dc.get_dof_properties(dof)
            props.stiffness = 500.0
            props.damping   = 50.0
            self.dc.set_dof_properties(dof, props)

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------
    @property
    def holding_object(self):
        return self.held_object_path is not None

    # ----------------------------------------------------------
    # Public actions
    # ----------------------------------------------------------
    def go_home(self):
        if self.held_object_path:
            gripper_pos, _ = get_prim_transform(self.gripper_tip_path)
            self._detach_object(gripper_pos)
        self.action_queue = [
            ("arm",     POSE_REST,    60),
            ("gripper", GRIPPER_OPEN, 20),
        ]
        self.is_busy = True

    def pick(self, prim_path, target_world_pos=None):
        pan = self._compute_shoulder_pan(target_world_pos)
        ready = list(POSE_READY); ready[0] = pan
        reach = list(POSE_REACH); reach[0] = pan
        rest  = list(POSE_REST);  rest[0]  = pan

        steps = [
            ("arm",     ready,         40),
            ("gripper", GRIPPER_OPEN,  20),
            ("arm",     reach,         50),
            ("gripper", GRIPPER_CLOSE, 25),
        ]
        if prim_path:
            steps.append(("attach", prim_path))
        steps += [
            ("arm", rest, 50),
        ]
        self.action_queue = steps
        self.is_busy = True
        print(f"[FetchArm] Pick: {prim_path} (pan={np.degrees(pan):.1f}°)")

    def place(self, target_pos):
        pan = self._compute_shoulder_pan(target_pos)
        ready = list(POSE_READY); ready[0] = pan
        reach = list(POSE_REACH); reach[0] = pan
        rest  = list(POSE_REST);  rest[0]  = pan

        steps = [
            ("arm",     ready,        40),
            ("arm",     reach,        50),
            ("gripper", GRIPPER_OPEN, 25),
        ]
        if self.held_object_path:
            steps.append(("detach", list(target_pos)))
        steps += [
            ("arm", rest, 40),
        ]
        self.action_queue = steps
        self.is_busy = True
        print(f"[FetchArm] Place at {target_pos}")

    def open_object(self, prim_path, target_world_pos=None):
        pan   = self._compute_shoulder_pan(target_world_pos)
        ready = list(POSE_READY); ready[0] = pan
        reach = list(POSE_REACH); reach[0] = pan
        rest  = list(POSE_REST);  rest[0]  = pan
        self.action_queue = [
            ("arm", ready, 40),
            ("arm", reach, 50),
            ("arm", ready, 40),
            ("arm", rest,  40),
        ]
        self.is_busy = True
        print(f"[FetchArm] Open: {prim_path}")

    def close_object(self, prim_path, target_world_pos=None):
        pan   = self._compute_shoulder_pan(target_world_pos)
        ready = list(POSE_READY); ready[0] = pan
        reach = list(POSE_REACH); reach[0] = pan
        rest  = list(POSE_REST);  rest[0]  = pan
        self.action_queue = [
            ("arm", ready, 40),
            ("arm", reach, 50),
            ("arm", ready, 40),
            ("arm", rest,  40),
        ]
        self.is_busy = True
        print(f"[FetchArm] Close: {prim_path}")

    # ----------------------------------------------------------
    # Frame update
    # ----------------------------------------------------------
    def update(self, dt=0.0):
        self._update_held_object()

        if not self.action_queue and self.current_phase is None:
            self.is_busy = False
            return True

        if self.current_phase is None:
            self.current_phase = self.action_queue.pop(0)
            self.phase_timer = 0
            print(f"[FetchArm] Phase: {self.current_phase[0]} "
                  f"(remaining: {len(self.action_queue)})")

        phase_type = self.current_phase[0]
        self.phase_timer += 1

        if phase_type == "arm":
            targets, duration = self.current_phase[1], self.current_phase[2]
            self._set_arm_targets(targets)
            if self.phase_timer >= duration:
                self.current_phase = None

        elif phase_type == "gripper":
            width, duration = self.current_phase[1], self.current_phase[2]
            self._set_gripper(width)
            if self.phase_timer >= duration:
                self.current_phase = None

        elif phase_type == "attach":
            self._attach_object(self.current_phase[1])
            self.current_phase = None

        elif phase_type == "detach":
            self._detach_object(self.current_phase[1])
            self.current_phase = None

        return False

    # ----------------------------------------------------------
    # Joint control
    # ----------------------------------------------------------
    def _set_arm_targets(self, angles):
        if not self.art or self.art == 0:
            return
        self._arm_targets = list(angles)
        for i, dof_idx in enumerate(self.arm_dof_indices):
            if i >= len(angles):
                break
            dof = self.dc.get_articulation_dof(self.art, dof_idx)
            self.dc.set_dof_position_target(dof, float(angles[i]))

    def _set_gripper(self, width):
        if not self.art or self.art == 0:
            return
        self._grip_target = width
        for dof_idx in self.grip_dof_indices:
            dof = self.dc.get_articulation_dof(self.art, dof_idx)
            self.dc.set_dof_position_target(dof, float(width))

    # ----------------------------------------------------------
    # Object attachment (teleport-based)
    # ----------------------------------------------------------
    def _attach_object(self, prim_path):
        if not prim_path:
            return
        try:
            self.held_object_path  = prim_path
            self.held_object_xform = XFormPrim(prim_path)
            print(f"[FetchArm] Attached: {prim_path}")
        except Exception as e:
            print(f"[FetchArm] Attach failed: {e}")
            self.held_object_path = None

    def _detach_object(self, target_position):
        if not self.held_object_path:
            return
        try:
            if self.held_object_xform and target_position is not None:
                pos = np.array(target_position, dtype=float)
                self.held_object_xform.set_world_pose(pos)
        except Exception as e:
            print(f"[FetchArm] Detach move failed: {e}")
        self.held_object_path  = None
        self.held_object_xform = None
        print("[FetchArm] Object released")

    def _update_held_object(self):
        if not self.held_object_path or not self.held_object_xform:
            return
        try:
            tip_pos, _ = get_prim_transform(self.gripper_tip_path)
            self.held_object_xform.set_world_pose(tip_pos)
        except Exception:
            pass

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _compute_shoulder_pan(self, target_world_pos):
        """Compute shoulder_pan_joint angle to face a world position."""
        if target_world_pos is None:
            return 0.0
        try:
            robot_pos, robot_quat = get_prim_transform(self.base_link_path)
            dx = target_world_pos[0] - robot_pos[0]
            dy = target_world_pos[1] - robot_pos[1]
            robot_yaw = quat_to_euler_z(robot_quat)
            target_yaw = np.arctan2(dy, dx)
            pan = target_yaw - robot_yaw
            pan = np.clip(pan, -np.pi / 2, np.pi / 2)
            return float(pan)
        except Exception:
            return 0.0
