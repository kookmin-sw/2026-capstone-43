#!/usr/bin/env python3

"""
Manual Fetch teleoperation for the current KARMA environment.

This script reuses the same robot/scene setup as execute_LLM_plan.py, but
replaces task-planner execution with direct keyboard teleoperation and
ConceptGraph-friendly posed RGB-D capture from the Fetch head camera.
"""

import argparse
import json
import math
import os
import shutil
import sys
from typing import Optional

os.environ.setdefault("KARMA_INTERACTIVE_VIEW", "0")
os.environ.setdefault("KARMA_START_TASK_EXECUTOR", "0")
os.environ.setdefault("KARMA_ENABLE_HEAD_PANOPTIC", "1")
os.environ.setdefault(
    "KARMA_ENABLE_HOV_MULTICAM",
    "1" if "--hov" in sys.argv[1:] else "0",
)

import cv2
import numpy as np
from habitat_sim.utils.common import quat_to_magnum
from scipy.spatial.transform import Rotation as R

import execute_LLM_plan as karma_env


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_OUTPUT_ROOT = None
DEFAULT_HOV_OUTPUT_ROOT = "/home/yuchaehee/long_term_memory_project/my_local_data/hssd-HOV"
DEFAULT_HOV_CAMERA_NAME = "front"
ROBOT_SELF_SEMANTIC_ID = 100000
T_HABITAT_TO_OPENCV = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _scene_name(scene_id: str) -> str:
    return os.path.splitext(os.path.basename(scene_id))[0]


def _resolve_recording_paths(
    scene_name: str,
    output_root: Optional[str],
    use_hov: bool = False,
    hov_camera_name: str = DEFAULT_HOV_CAMERA_NAME,
):
    if output_root:
        base_save_dir = os.path.abspath(output_root)
    elif use_hov:
        base_save_dir = DEFAULT_HOV_OUTPUT_ROOT
    else:
        base_save_dir = None

    if use_hov:
        if base_save_dir is None:
            base_save_dir = DEFAULT_HOV_OUTPUT_ROOT
        scene_dir = os.path.join(base_save_dir, scene_name)
        camera_dir = os.path.join(scene_dir, "cameras", hov_camera_name)
        render_camera_file = os.path.join(scene_dir, f"render_camera_{hov_camera_name}.json")
        return {
            "base_save_dir": base_save_dir,
            "scene_dir": scene_dir,
            "camera_dir": camera_dir,
            "results_dir": os.path.join(camera_dir, "results"),
            "traj_file": os.path.join(camera_dir, "traj.txt"),
            "params_file": os.path.join(camera_dir, "cam_params.json"),
            "render_camera_file": render_camera_file,
        }

    if output_root:
        scene_dir = os.path.join(base_save_dir, scene_name)
        render_camera_file = os.path.join(scene_dir, "render_camera.json")
        return {
            "base_save_dir": base_save_dir,
            "scene_dir": scene_dir,
            "camera_dir": scene_dir,
            "results_dir": os.path.join(scene_dir, "results"),
            "traj_file": os.path.join(scene_dir, "traj.txt"),
            "params_file": os.path.join(base_save_dir, "cam_params.json"),
            "render_camera_file": render_camera_file,
        }

    if scene_name.startswith("ArchitecTHOR") or scene_name.startswith("FloorPlan"):
        base_save_dir = "/home/yuchaehee/long_term_memory_project/my_local_data/ai2thor-hab"
        base_render_cam_dir = (
            "/home/yuchaehee/long_term_memory_project/concept-graphs/"
            "conceptgraph/dataset/dataconfigs/ai2thor-hab"
        )
        render_camera_file = os.path.join(
            base_render_cam_dir,
            f"{scene_name}.scene_instance.json",
        )
    else:
        base_save_dir = "/home/yuchaehee/long_term_memory_project/my_local_data/hssd"
        base_render_cam_dir = (
            "/home/yuchaehee/long_term_memory_project/concept-graphs/"
            "conceptgraph/dataset/dataconfigs/hssd"
        )
        render_camera_file = os.path.join(base_render_cam_dir, f"{scene_name}.json")

    scene_dir = os.path.join(base_save_dir, scene_name)
    return {
        "base_save_dir": base_save_dir,
        "scene_dir": scene_dir,
        "camera_dir": scene_dir,
        "results_dir": os.path.join(scene_dir, "results"),
        "traj_file": os.path.join(scene_dir, "traj.txt"),
        "params_file": os.path.join(base_save_dir, "cam_params.json"),
        "render_camera_file": render_camera_file,
    }


def _sensor_cfg_by_suffix(suffix: str):
    agents = karma_env.cfg.habitat.simulator.agents
    for agent_name in agents.keys():
        sensors = getattr(getattr(agents, agent_name), "sim_sensors", {})
        for key in sensors.keys():
            if key.endswith(suffix):
                return getattr(sensors, key)
    dynamic_cfgs = getattr(karma_env, "HOV_DYNAMIC_SENSOR_CONFIGS", {})
    for key, value in dynamic_cfgs.items():
        if str(key).endswith(suffix):
            return value
    raise KeyError(f"Sensor config with suffix '{suffix}' not found.")


def _sensor_cfg_by_suffix_optional(suffix: str):
    try:
        return _sensor_cfg_by_suffix(suffix)
    except KeyError:
        return None


def _get_sensor_state_by_suffix(sensor_name_suffix: str, agent_id: int = 0):
    state = karma_env.sim.get_agent(agent_id).get_state()
    if sensor_name_suffix in state.sensor_states:
        return state.sensor_states[sensor_name_suffix]
    for key, sensor_state in state.sensor_states.items():
        if str(key).endswith(sensor_name_suffix):
            return sensor_state
    raise KeyError(
        f"{sensor_name_suffix} sensor state not found. available={list(state.sensor_states.keys())}"
    )


def _quat_to_rotmat(quat) -> np.ndarray:
    try:
        return np.array(quat_to_magnum(quat).to_matrix(), dtype=np.float32)
    except Exception:
        pass

    if hasattr(quat, "to_matrix"):
        return np.array(quat.to_matrix(), dtype=np.float32)

    if hasattr(quat, "vector") and hasattr(quat, "scalar"):
        vec = np.array(quat.vector, dtype=np.float32).reshape(3)
        coeffs_xyzw = np.array(
            [float(vec[0]), float(vec[1]), float(vec[2]), float(quat.scalar)],
            dtype=np.float32,
        )
        return R.from_quat(coeffs_xyzw).as_matrix().astype(np.float32)

    if all(hasattr(quat, x) for x in ("x", "y", "z", "w")):
        coeffs_xyzw = np.array(
            [float(quat.x), float(quat.y), float(quat.z), float(quat.w)],
            dtype=np.float32,
        )
        return R.from_quat(coeffs_xyzw).as_matrix().astype(np.float32)

    coeffs = np.asarray(quat, dtype=np.float32).reshape(-1)
    if coeffs.size != 4:
        raise TypeError(f"Unsupported quaternion value: {quat!r}")
    return R.from_quat(coeffs).as_matrix().astype(np.float32)


def _resize_to_height(img: Optional[np.ndarray], target_h: int):
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = float(target_h) / float(max(h, 1))
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


def _node_semantic_attrs():
    return ("semantic_id", "object_semantic_id", "drawable_semantic_id")


def _set_node_semantic_id(node, semantic_id: int) -> int:
    updates = 0
    for attr in _node_semantic_attrs():
        if hasattr(node, attr):
            try:
                setattr(node, attr, int(semantic_id))
                updates += 1
            except Exception:
                pass
    return updates


def _apply_robot_semantic_ids(sim, semantic_id: int = ROBOT_SELF_SEMANTIC_ID) -> int:
    art = sim.get_agent_data(0).articulated_agent
    sim_obj = art.sim_obj
    tagged_nodes = 0
    successful_updates = 0

    def _tag_node(node):
        nonlocal tagged_nodes, successful_updates
        if node is None:
            return
        tagged_nodes += 1
        successful_updates += _set_node_semantic_id(node, semantic_id)

    _tag_node(getattr(sim_obj, "root_scene_node", None))
    for node in list(getattr(sim_obj, "visual_scene_nodes", [])):
        _tag_node(node)

    link_ids = [-1]
    try:
        link_ids.extend(list(sim_obj.get_link_ids()))
    except Exception:
        link_ids.extend(range(int(getattr(sim_obj, "num_links", 0))))

    for link_id in link_ids:
        try:
            _tag_node(sim_obj.get_link_scene_node(link_id))
        except Exception:
            pass
        try:
            for node in sim_obj.get_link_visual_nodes(link_id):
                _tag_node(node)
        except Exception:
            pass

    if successful_updates <= 0:
        raise RuntimeError("Failed to assign robot semantic ids for self-mask.")

    print(
        f"[SelfMask] robot semantic id={semantic_id} "
        f"nodes={tagged_nodes} updates={successful_updates}"
    )
    return semantic_id


def _refresh_observation():
    karma_env.obs = karma_env._env_step({"action": karma_env.EMPTY_ACTION, "action_args": {}})
    return karma_env.obs


def _ground_contact_y(sim, base_pos: np.ndarray) -> float:
    pf = getattr(sim, "pathfinder", None)
    if pf is None or not pf.is_loaded:
        return float(base_pos[1])

    candidate = np.asarray(base_pos, dtype=np.float32)
    snapped = None
    try:
        island_index = int(pf.get_island(candidate))
        snapped = np.array(pf.snap_point(candidate, island_index), dtype=np.float32)
    except Exception:
        try:
            snapped = np.array(pf.snap_point(candidate), dtype=np.float32)
        except Exception:
            snapped = None

    if snapped is not None and snapped.shape == (3,) and np.all(np.isfinite(snapped)):
        return float(snapped[1])
    return float(base_pos[1])


def _print_robot_world_position():
    base_pos, yaw = karma_env.get_fetch_base_pose(karma_env.sim, agent_idx=0)
    floor_pos = np.array(
        [float(base_pos[0]), _ground_contact_y(karma_env.sim, base_pos), float(base_pos[2])],
        dtype=np.float32,
    )

    print(
        "[Pose] raw_base_pos="
        f"[{base_pos[0]:.6f}, {base_pos[1]:.6f}, {base_pos[2]:.6f}] "
        f"yaw_deg={math.degrees(yaw):.2f}"
    )
    print(
        "[Pose] floor_contact_pos="
        f"[{floor_pos[0]:.6f}, {floor_pos[1]:.6f}, {floor_pos[2]:.6f}]"
    )
    print(
        "[Pose] rigid_objs translation="
        f"[{floor_pos[0]:.6f}, {floor_pos[1]:.6f}, {floor_pos[2]:.6f}]"
    )
    return floor_pos


def _panoptic_ids(panoptic_obs) -> np.ndarray:
    panoptic = np.asarray(panoptic_obs)
    if panoptic.ndim == 3:
        panoptic = panoptic[..., 0]
    if panoptic.ndim != 2:
        raise ValueError(f"Unexpected head_panoptic shape: {panoptic.shape}")
    return np.asarray(panoptic, dtype=np.int64)


class PosedRGBDRecorder:
    def __init__(
        self,
        output_root: Optional[str],
        scene_name: str,
        use_hov: bool = False,
        hov_camera_name: str = DEFAULT_HOV_CAMERA_NAME,
        embedding_rgb_only: bool = False,
    ):
        self.scene_name = scene_name
        self.output_root = output_root
        self.use_hov = bool(use_hov)
        self.hov_camera_name = hov_camera_name
        self.embedding_rgb_only = bool(embedding_rgb_only)

        self.robot_self_semantic_id = ROBOT_SELF_SEMANTIC_ID
        self.frame_count = 0
        self.is_recording = False
        self.session_initialized = False
        self.last_masked_pixels = 0
        self.last_masked_pixels_by_camera = {}

        self.camera_entries = {}
        self.camera_order = []
        self._register_camera(
            camera_name="front",
            paths=_resolve_recording_paths(
                scene_name,
                output_root,
                use_hov=self.use_hov,
                hov_camera_name="front",
            ),
            rgb_cfg_suffix="head_rgb_sensor",
            depth_cfg_suffix="head_depth_sensor",
            panoptic_cfg_suffix="head_panoptic_sensor",
            sensor_state_suffix="head_rgb",
            obs_rgb_suffix="head_rgb",
            obs_depth_suffix="head_depth",
            obs_panoptic_suffix="head_panoptic",
            agent_id=0,
        )
        if self.use_hov:
            self._register_camera(
                camera_name="up",
                paths=_resolve_recording_paths(
                    scene_name,
                    output_root,
                    use_hov=True,
                    hov_camera_name="up",
                ),
                rgb_cfg_suffix="observer_up_rgb_sensor",
                depth_cfg_suffix="observer_up_depth_sensor",
                panoptic_cfg_suffix="observer_up_panoptic_sensor",
                sensor_state_suffix="observer_up_rgb",
                obs_rgb_suffix="observer_up_rgb",
                obs_depth_suffix="observer_up_depth",
                obs_panoptic_suffix="observer_up_panoptic",
                agent_id=0,
            )

        self.primary_camera_name = "front"
        self.primary_camera = self.camera_entries[self.primary_camera_name]
        self.base_save_dir = self.primary_camera["base_save_dir"]
        self.scene_dir = self.primary_camera["scene_dir"]
        self.camera_dir = self.primary_camera["camera_dir"]
        self.results_dir = self.primary_camera["results_dir"]
        self.traj_file = self.primary_camera["traj_file"]
        self.params_file = self.primary_camera["params_file"]
        self.render_camera_file = self.primary_camera["render_camera_file"]
        self.width = self.primary_camera["width"]
        self.height = self.primary_camera["height"]
        self.hfov_deg = self.primary_camera["hfov_deg"]
        self.depth_min = self.primary_camera["depth_min"]
        self.depth_max = self.primary_camera["depth_max"]
        self.normalize_depth = self.primary_camera["normalize_depth"]
        self.depth_scale = self.primary_camera["depth_scale"]
        self.embedding_rgb_dir = os.path.join(self.scene_dir, "embedding_rgb")

    def _register_camera(
        self,
        camera_name: str,
        paths,
        rgb_cfg_suffix: str,
        depth_cfg_suffix: str,
        panoptic_cfg_suffix: Optional[str],
        sensor_state_suffix: str,
        obs_rgb_suffix: str,
        obs_depth_suffix: str,
        obs_panoptic_suffix: Optional[str],
        agent_id: int,
    ):
        rgb_cfg = _sensor_cfg_by_suffix(rgb_cfg_suffix)
        depth_cfg = _sensor_cfg_by_suffix(depth_cfg_suffix)
        panoptic_cfg = (
            _sensor_cfg_by_suffix_optional(panoptic_cfg_suffix)
            if panoptic_cfg_suffix
            else None
        )
        width = int(rgb_cfg.width)
        height = int(rgb_cfg.height)
        hfov_deg = float(getattr(rgb_cfg, "hfov", 90.0))
        depth_min = float(getattr(depth_cfg, "min_depth", 0.0))
        depth_max = float(getattr(depth_cfg, "max_depth", 10.0))
        normalize_depth = bool(getattr(depth_cfg, "normalize_depth", True))
        depth_scale = 65535.0 / max(depth_max, 1e-6)

        entry = {
            **paths,
            "camera_name": camera_name,
            "rgb_cfg": rgb_cfg,
            "depth_cfg": depth_cfg,
            "panoptic_cfg": panoptic_cfg,
            "width": width,
            "height": height,
            "hfov_deg": hfov_deg,
            "depth_min": depth_min,
            "depth_max": depth_max,
            "normalize_depth": normalize_depth,
            "depth_scale": depth_scale,
            "agent_id": agent_id,
            "sensor_state_suffix": sensor_state_suffix,
            "obs_rgb_suffix": obs_rgb_suffix,
            "obs_depth_suffix": obs_depth_suffix,
            "obs_panoptic_suffix": obs_panoptic_suffix,
        }
        self.camera_entries[camera_name] = entry
        self.camera_order.append(camera_name)
        self.last_masked_pixels_by_camera[camera_name] = 0

    def _ensure_clean_session(self):
        os.makedirs(self.base_save_dir, exist_ok=True)
        os.makedirs(self.scene_dir, exist_ok=True)
        if self.embedding_rgb_only:
            if os.path.isdir(self.embedding_rgb_dir):
                shutil.rmtree(self.embedding_rgb_dir)
            os.makedirs(self.embedding_rgb_dir, exist_ok=True)
            self.frame_count = 0
            self.session_initialized = True
            return

        for camera_name in self.camera_order:
            camera = self.camera_entries[camera_name]
            if os.path.isdir(camera["results_dir"]):
                shutil.rmtree(camera["results_dir"])
            os.makedirs(camera["results_dir"], exist_ok=True)
            os.makedirs(camera["camera_dir"], exist_ok=True)
            with open(camera["traj_file"], "w", encoding="utf-8") as f:
                f.write("")
            self._save_params(camera)

        self.frame_count = 0
        self.session_initialized = True

    def _save_params(self, camera):
        hfov_rad = math.radians(camera["hfov_deg"])
        fx = (camera["width"] / 2.0) / math.tan(hfov_rad / 2.0)
        fy = fx
        cx = camera["width"] / 2.0 - 0.5
        cy = camera["height"] / 2.0 - 0.5

        params = {
            "camera": {
                "w": camera["width"],
                "h": camera["height"],
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "scale": camera["depth_scale"],
            }
        }
        with open(camera["params_file"], "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4)

    def _camera_transform(self, sensor_state) -> np.ndarray:
        pos = np.array(sensor_state.position, dtype=np.float32)
        rot = _quat_to_rotmat(sensor_state.rotation)

        T_world_cam = np.eye(4, dtype=np.float32)
        T_world_cam[:3, :3] = rot
        T_world_cam[:3, 3] = pos
        return T_world_cam @ T_HABITAT_TO_OPENCV

    def _depth_meters(self, depth_obs, camera) -> np.ndarray:
        depth = np.asarray(depth_obs, dtype=np.float32)
        depth = np.nan_to_num(
            depth,
            nan=camera["depth_max"],
            posinf=camera["depth_max"],
            neginf=camera["depth_min"],
        )
        if camera["normalize_depth"]:
            depth = camera["depth_min"] + np.clip(depth, 0.0, 1.0) * (
                camera["depth_max"] - camera["depth_min"]
            )
        return np.clip(depth, camera["depth_min"], camera["depth_max"])

    def _mask_robot_self(self, rgb_bgr, depth_m, panoptic_obs):
        if panoptic_obs is None:
            return rgb_bgr, depth_m, 0
        panoptic_ids = _panoptic_ids(panoptic_obs)
        target_shape = None
        if depth_m is not None:
            target_shape = depth_m.shape[:2]
        elif rgb_bgr is not None:
            target_shape = rgb_bgr.shape[:2]

        if target_shape is None:
            raise ValueError("Either rgb_bgr or depth_m must be provided for masking.")

        if panoptic_ids.shape != target_shape:
            raise ValueError(
                "head_panoptic resolution does not match target image: "
                f"{panoptic_ids.shape} vs {target_shape}"
            )

        robot_mask = panoptic_ids == self.robot_self_semantic_id
        masked_pixels = int(np.count_nonzero(robot_mask))
        if masked_pixels <= 0:
            return rgb_bgr, depth_m, 0

        rgb_masked = None if rgb_bgr is None else np.array(rgb_bgr, copy=True)
        depth_masked = None if depth_m is None else np.array(depth_m, copy=True)
        if rgb_masked is not None:
            rgb_masked[robot_mask] = 0
        if depth_masked is not None:
            depth_masked[robot_mask] = 0.0
        return rgb_masked, depth_masked, masked_pixels

    def save_render_camera_json(self):
        for camera_name in self.camera_order:
            camera = self.camera_entries[camera_name]
            sensor_state = _get_sensor_state_by_suffix(
                camera["sensor_state_suffix"],
                agent_id=camera["agent_id"],
            )
            os.makedirs(os.path.dirname(camera["render_camera_file"]), exist_ok=True)

            hfov_rad = math.radians(camera["hfov_deg"])
            fx = (camera["width"] / 2.0) / math.tan(hfov_rad / 2.0)
            fy = fx
            cx = camera["width"] / 2.0 - 0.5
            cy = camera["height"] / 2.0 - 0.5

            T_final = self._camera_transform(sensor_state)
            data = {
                "class_name": "PinholeCameraParameters",
                "extrinsic": T_final.flatten().tolist(),
                "intrinsic": {
                    "width": camera["width"],
                    "height": camera["height"],
                    "intrinsic_matrix": [
                        fx,
                        0.0,
                        cx,
                        0.0,
                        fy,
                        cy,
                        0.0,
                        0.0,
                        1.0,
                    ],
                },
            }

            with open(camera["render_camera_file"], "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print(f"[Recorder] render camera saved ({camera_name}): {camera['render_camera_file']}")

    def _capture_camera(self, camera_name: str, obs):
        camera = self.camera_entries[camera_name]
        rgb_obs = karma_env._obs_by_suffix(obs, camera["obs_rgb_suffix"])
        panoptic_obs = (
            karma_env._obs_by_suffix_optional(obs, camera["obs_panoptic_suffix"])
            if camera["obs_panoptic_suffix"] is not None
            else None
        )

        rgb_bgr = karma_env.rgb_observation_to_bgr(rgb_obs)
        rgb_bgr, _, masked_pixels = self._mask_robot_self(
            rgb_bgr,
            None,
            panoptic_obs,
        )
        self.last_masked_pixels_by_camera[camera_name] = masked_pixels
        if camera_name == self.primary_camera_name:
            self.last_masked_pixels = masked_pixels

        if self.embedding_rgb_only:
            rgb_name = f"{self.frame_count:06d}.png"
            cv2.imwrite(os.path.join(self.embedding_rgb_dir, rgb_name), rgb_bgr)
            print(
                f"[Recorder] saved embedding rgb frame={self.frame_count:06d} "
                f"camera={camera_name} rgb={rgb_name} masked_self_px={masked_pixels}"
            )
            return

        depth_obs = karma_env._obs_by_suffix(obs, camera["obs_depth_suffix"])
        sensor_state = _get_sensor_state_by_suffix(
            camera["sensor_state_suffix"],
            agent_id=camera["agent_id"],
        )
        depth_m = self._depth_meters(depth_obs, camera)
        _, depth_m, _ = self._mask_robot_self(
            None,
            depth_m,
            panoptic_obs,
        )
        depth_u16 = np.clip(depth_m * camera["depth_scale"], 0, 65535).astype(np.uint16)

        rgb_name = f"frame{self.frame_count:06d}.jpg"
        depth_name = f"depth{self.frame_count:06d}.png"
        cv2.imwrite(os.path.join(camera["results_dir"], rgb_name), rgb_bgr)
        cv2.imwrite(os.path.join(camera["results_dir"], depth_name), depth_u16)

        T_final = self._camera_transform(sensor_state)
        with open(camera["traj_file"], "a", encoding="utf-8") as f:
            line = " ".join(format(x, ".18e") for x in T_final.flatten()) + "\n"
            f.write(line)

        print(
            f"[Recorder] saved camera={camera_name} frame={self.frame_count:06d} "
            f"rgb={rgb_name} depth={depth_name} masked_self_px={masked_pixels}"
        )

    def capture(self, obs, force: bool = False):
        if not self.is_recording and not force:
            return False

        if not self.session_initialized:
            self._ensure_clean_session()
        if self.embedding_rgb_only:
            self._capture_camera(self.primary_camera_name, obs)
        else:
            for camera_name in self.camera_order:
                self._capture_camera(camera_name, obs)
        self.frame_count += 1
        return True

    def toggle_recording(self, obs):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self._ensure_clean_session()
            if self.embedding_rgb_only:
                out_dir = self.embedding_rgb_dir
            else:
                out_dir = self.scene_dir if self.use_hov else self.results_dir
            print(f"[Recorder] recording started -> {out_dir}")
            self.capture(obs, force=True)
        else:
            print(f"[Recorder] recording stopped. total_frames={self.frame_count}")


def _depth_preview(
    depth_obs,
    recorder: PosedRGBDRecorder,
    camera_name: str,
    panoptic_obs=None,
) -> np.ndarray:
    camera = recorder.camera_entries[camera_name]
    depth_m = recorder._depth_meters(depth_obs, camera)
    if panoptic_obs is not None:
        _, depth_m, _ = recorder._mask_robot_self(None, depth_m, panoptic_obs)
    depth_norm = np.clip(depth_m / max(camera["depth_max"], 1e-6), 0.0, 1.0)
    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def _label_panel(img: Optional[np.ndarray], label: str) -> Optional[np.ndarray]:
    if img is None:
        return None
    annotated = np.array(img, copy=True)
    cv2.putText(
        annotated,
        label,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        label,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return annotated


def _build_panel(obs, recorder: PosedRGBDRecorder, status_text: str) -> np.ndarray:
    head_rgb = karma_env._obs_by_suffix_optional(obs, "head_rgb")
    head_depth = karma_env._obs_by_suffix_optional(obs, "head_depth")
    head_panoptic = karma_env._obs_by_suffix_optional(obs, "head_panoptic")
    third_rgb = karma_env._obs_by_suffix_optional(obs, "third_rgb")
    navmesh = karma_env._build_navmesh_topdown_panel(karma_env.sim)

    head_bgr = (
        karma_env.rgb_observation_to_bgr(head_rgb) if head_rgb is not None else None
    )
    if head_bgr is not None and head_panoptic is not None:
        head_bgr, _, _ = recorder._mask_robot_self(head_bgr, None, head_panoptic)
    depth_bgr = (
        _depth_preview(head_depth, recorder, "front", head_panoptic)
        if head_depth is not None
        else None
    )
    third_bgr = (
        karma_env.rgb_observation_to_bgr(third_rgb) if third_rgb is not None else None
    )

    panels = [
        _label_panel(head_bgr, "front rgb"),
        _label_panel(depth_bgr, "front depth"),
    ]

    if recorder.use_hov:
        up_rgb = karma_env._obs_by_suffix_optional(obs, "observer_up_rgb")
        up_depth = karma_env._obs_by_suffix_optional(obs, "observer_up_depth")
        up_panoptic = karma_env._obs_by_suffix_optional(obs, "observer_up_panoptic")
        up_bgr = (
            karma_env.rgb_observation_to_bgr(up_rgb) if up_rgb is not None else None
        )
        if up_bgr is not None and up_panoptic is not None:
            up_bgr, _, _ = recorder._mask_robot_self(up_bgr, None, up_panoptic)
        up_depth_bgr = (
            _depth_preview(up_depth, recorder, "up", up_panoptic)
            if up_depth is not None
            else None
        )
        panels.extend(
            [
                _label_panel(up_bgr, "up rgb"),
                _label_panel(up_depth_bgr, "up depth"),
            ]
        )

    panels.extend(
        [
            _label_panel(third_bgr, "third rgb"),
            _label_panel(navmesh, "navmesh"),
        ]
    )
    panels = [img for img in panels if img is not None]
    if not panels:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    target_h = min(img.shape[0] for img in panels)
    panels = [_resize_to_height(img, target_h) for img in panels]
    panel = cv2.hconcat(panels)

    pos, yaw = karma_env.get_fetch_base_pose(karma_env.sim, agent_idx=0)
    lines = [
        f"{status_text}",
        f"scene={_scene_name(karma_env.SCENE_ID)} | rec={'ON' if recorder.is_recording else 'OFF'} | frames={recorder.frame_count}",
        f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | yaw_deg={math.degrees(yaw):.1f}",
        f"robot-self-mask=ON | semantic_id={recorder.robot_self_semantic_id} | last_masked_px={recorder.last_masked_pixels}",
        "W/S: move | A/D: turn | P: print pose | R: rec on/off | C: capture | K: render json",
        "N: navmesh | H: help | Q or ESC: quit",
    ]

    y = 28
    for line in lines:
        cv2.putText(
            panel,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return panel


def _print_help(output_root: str):
    print("")
    print("[Manual Teleop]")
    print("  W / S : forward / backward")
    print("  A / D : rotate left / rotate right")
    print("  P     : print current robot world position")
    print("  R     : start or stop RGB-D recording")
    print("  C     : capture current head RGB-D frame once")
    print("  K     : save current render_camera.json")
    print("  N     : toggle navmesh visualization")
    print("  Q     : quit")
    print("  note  : saved RGB-D applies robot self-mask via head_panoptic")
    print(f"  output: {output_root}")
    print("")


def _apply_motion(lin: float, ang: float, repeat: int, status_text: str):
    karma_env._tuck_arm_for_nav()
    karma_env._step_base(lin, ang, repeat=repeat)
    return status_text


def run_manual_teleop(args) -> int:
    output_root = os.path.abspath(args.output_root) if args.output_root else None
    recorder = PosedRGBDRecorder(
        output_root=output_root,
        scene_name=_scene_name(karma_env.SCENE_ID),
        use_hov=bool(args.hov),
        embedding_rgb_only=bool(args.embedding_rgb_only),
    )
    if recorder.embedding_rgb_only:
        help_output_path = recorder.embedding_rgb_dir
    else:
        help_output_path = recorder.scene_dir if recorder.use_hov else recorder.results_dir
    _apply_robot_semantic_ids(karma_env.sim, recorder.robot_self_semantic_id)
    _refresh_observation()
    _print_help(help_output_path)

    window_name = "KARMA Manual RGB-D Teleop"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1800, 1000)
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV window could not be created. Run this script in a desktop session."
        ) from exc

    status_text = "IDLE"

    while True:
        panel = _build_panel(karma_env.obs, recorder, status_text)
        cv2.imshow(window_name, panel)

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            pass

        key = cv2.waitKeyEx(30)
        if key < 0:
            continue

        key_code = key & 0xFF
        if key_code in (ord("q"), 27):
            break
        if key_code in (ord("h"), ord("H")):
            _print_help(help_output_path)
            status_text = "HELP"
            continue
        if key_code in (ord("n"), ord("N")):
            if karma_env.sim.pathfinder.is_loaded:
                karma_env.sim.navmesh_visualization = not bool(
                    karma_env.sim.navmesh_visualization
                )
                status_text = (
                    f"NAVMESH {'ON' if karma_env.sim.navmesh_visualization else 'OFF'}"
                )
            else:
                status_text = "NAVMESH unavailable"
            continue
        if key_code in (ord("r"), ord("R")):
            recorder.toggle_recording(karma_env.obs)
            status_text = "REC ON" if recorder.is_recording else "REC OFF"
            continue
        if key_code in (ord("c"), ord("C")):
            recorder.capture(karma_env.obs, force=True)
            status_text = "CAPTURED"
            continue
        if key_code in (ord("k"), ord("K")):
            recorder.save_render_camera_json()
            status_text = "RENDER JSON SAVED"
            continue
        if key_code in (ord("p"), ord("P")):
            floor_pos = _print_robot_world_position()
            status_text = (
                f"POSE [{floor_pos[0]:.2f}, {floor_pos[1]:.2f}, {floor_pos[2]:.2f}]"
            )
            continue
        if key_code in (ord("w"), ord("W")):
            status_text = _apply_motion(+1.0, 0.0, args.move_repeat, "MOVE FORWARD")
        elif key_code in (ord("s"), ord("S")):
            status_text = _apply_motion(-1.0, 0.0, args.move_repeat, "MOVE BACKWARD")
        elif key_code in (ord("a"), ord("A")):
            status_text = _apply_motion(0.0, +1.0, args.turn_repeat, "TURN LEFT")
        elif key_code in (ord("d"), ord("D")):
            status_text = _apply_motion(0.0, -1.0, args.turn_repeat, "TURN RIGHT")
        else:
            continue

        if recorder.is_recording:
            recorder.capture(karma_env.obs, force=False)

    return 0


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Manual teleoperation + posed RGB-D capture for the KARMA Fetch environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Optional override for the dataset root. If omitted, use the same save layout as generate_posed_RGB-D_dataset.py.",
    )
    parser.add_argument(
        "--hov",
        action="store_true",
        help=(
            "Save into the HOV-style dataset layout under "
            f"{DEFAULT_HOV_OUTPUT_ROOT}/<scene>/cameras/{DEFAULT_HOV_CAMERA_NAME}/..."
        ),
    )
    parser.add_argument(
        "--move-repeat",
        type=int,
        default=4,
        help="How many low-level base steps to apply for one W/S key press.",
    )
    parser.add_argument(
        "--turn-repeat",
        type=int,
        default=1,
        help="How many low-level rotation steps to apply for one A/D key press.",
    )
    parser.add_argument(
        "--embedding-rgb-only",
        action="store_true",
        help=(
            "Record only masked head RGB frames for embedding-data collection. "
            "Saved to <scene>/embedding_rgb/000000.png ..."
        ),
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    try:
        return run_manual_teleop(args)
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        try:
            karma_env.shutdown_task_executor(timeout=1.0)
        except Exception:
            pass

        try:
            karma_env.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
