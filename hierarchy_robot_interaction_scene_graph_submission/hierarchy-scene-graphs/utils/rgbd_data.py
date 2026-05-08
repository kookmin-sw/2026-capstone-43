import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


@dataclass
class RGBDView:
    camera_name: str
    rgb_image: Image.Image
    depth_image: Image.Image
    pose: np.ndarray
    intrinsics: np.ndarray
    scale: float


def _load_camera_params(path: Path):
    with open(path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    camera_params = data["camera"]
    intrinsics = np.array(
        [
            [camera_params["fx"], 0, camera_params["cx"]],
            [0, camera_params["fy"], camera_params["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return intrinsics, float(camera_params["scale"])


def _load_pose_lines(path: Path):
    with open(path, "r", encoding="utf-8") as infile:
        lines = [line.strip() for line in infile.readlines() if line.strip()]
    poses = []
    for line in lines:
        values = [float(val) for val in line.split()]
        poses.append(np.array(values, dtype=np.float32).reshape((4, 4)))
    return poses


def _create_point_cloud(
    rgb,
    depth,
    intrinsics: np.ndarray,
    scale: float,
    camera_pose=None,
    filter_distance=np.inf,
):
    rgb = np.array(rgb).astype(np.uint8)
    depth = np.array(depth)
    if rgb.shape[:2] != depth.shape[:2]:
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

    y, x = np.meshgrid(np.arange(rgb.shape[0]), np.arange(rgb.shape[1]), indexing="ij")
    depth = depth.astype(np.float32) / float(scale)
    mask = depth > 0
    if np.isfinite(filter_distance):
        mask &= depth <= float(filter_distance)
    x = x[mask]
    y = y[mask]
    depth = depth[mask]
    if depth.size == 0:
        return o3d.geometry.PointCloud()

    x_3d = (x - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y_3d = (y - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    z_3d = depth
    if z_3d.size == 0:
        return o3d.geometry.PointCloud()

    points = np.hstack((x_3d.reshape(-1, 1), y_3d.reshape(-1, 1), z_3d.reshape(-1, 1)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = rgb[mask]
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    if camera_pose is not None:
        pcd.transform(camera_pose)
    return pcd


def resolve_replica_scene_path(dataset_path: str, scene_id: str) -> Path:
    dataset_path = Path(dataset_path)
    direct_scene_path = dataset_path
    candidate_scene_path = dataset_path / str(scene_id)
    if (direct_scene_path / "results").is_dir() and (direct_scene_path / "traj.txt").is_file():
        return direct_scene_path
    if (candidate_scene_path / "results").is_dir() and (candidate_scene_path / "traj.txt").is_file():
        return candidate_scene_path
    scene_id_str = str(scene_id)
    if scene_id_str.isdigit() and dataset_path.is_dir():
        for child in dataset_path.iterdir():
            if not child.is_dir():
                continue
            if child.name.replace("_", "") == scene_id_str:
                if (child / "results").is_dir() and (child / "traj.txt").is_file():
                    return child
    raise FileNotFoundError(
        f"Could not resolve replica-style scene path from dataset_path={dataset_path} scene_id={scene_id}"
    )


def resolve_hov_scene_path(dataset_path: str, scene_id: str) -> Path:
    dataset_path = Path(dataset_path)
    direct_scene_path = dataset_path
    candidate_scene_path = dataset_path / str(scene_id)
    if (direct_scene_path / "cameras").is_dir():
        return direct_scene_path
    if (candidate_scene_path / "cameras").is_dir():
        return candidate_scene_path
    scene_id_str = str(scene_id)
    if scene_id_str.isdigit() and dataset_path.is_dir():
        for child in dataset_path.iterdir():
            if not child.is_dir():
                continue
            if child.name.replace("_", "") == scene_id_str:
                if (child / "cameras").is_dir():
                    return child
    raise FileNotFoundError(
        f"Could not resolve HOV multi-camera scene path from dataset_path={dataset_path} scene_id={scene_id}"
    )


class ReplicaStyleRGBDDataset:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.depth_intrinsics, self.scale = _load_camera_params(self.root_dir.parent / "cam_params.json")
        self.data_list = self._get_data_list()
        self.poses = _load_pose_lines(self.root_dir / "traj.txt")
        if len(self.data_list) != len(self.poses):
            raise ValueError(
                f"RGB-D frame count ({len(self.data_list)}) and pose count ({len(self.poses)}) do not match "
                f"for scene {self.root_dir}"
            )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.data_list[idx]
        rgb_image = Image.open(rgb_path)
        depth_image = Image.open(depth_path)
        pose = self.poses[idx]
        return rgb_image, depth_image, pose

    def _get_data_list(self):
        rgb_data_list = []
        depth_data_list = []
        for root, _, files in os.walk(self.root_dir / "results"):
            for file in files:
                if file.startswith("frame"):
                    rgb_data_list.append(os.path.join(root, file))
                elif file.startswith("depth"):
                    depth_data_list.append(os.path.join(root, file))
        rgb_data_list.sort()
        depth_data_list.sort()
        if len(rgb_data_list) != len(depth_data_list):
            raise ValueError(
                f"RGB frame count ({len(rgb_data_list)}) and depth frame count ({len(depth_data_list)}) "
                f"do not match for scene {self.root_dir}"
            )
        return list(zip(rgb_data_list, depth_data_list))

    def get_views(self, idx) -> List[RGBDView]:
        rgb_image, depth_image, pose = self[idx]
        return [
            RGBDView(
                camera_name="default",
                rgb_image=rgb_image,
                depth_image=depth_image,
                pose=pose,
                intrinsics=self.depth_intrinsics,
                scale=self.scale,
            )
        ]

    def create_pcd(
        self,
        rgb,
        depth,
        camera_pose=None,
        filter_distance=np.inf,
        intrinsics=None,
        scale=None,
    ):
        intrinsics = self.depth_intrinsics if intrinsics is None else np.asarray(intrinsics, dtype=np.float32)
        scale = self.scale if scale is None else float(scale)
        return _create_point_cloud(
            rgb,
            depth,
            intrinsics=intrinsics,
            scale=scale,
            camera_pose=camera_pose,
            filter_distance=filter_distance,
        )


class HOVMultiCameraRGBDDataset:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.cameras_dir = self.root_dir / "cameras"
        if not self.cameras_dir.is_dir():
            raise FileNotFoundError(f"Missing cameras directory in {self.root_dir}")

        self.camera_names = sorted(
            path.name for path in self.cameras_dir.iterdir() if path.is_dir()
        )
        if not self.camera_names:
            raise ValueError(f"No camera directories found in {self.cameras_dir}")

        self.camera_data: Dict[str, Dict[str, object]] = {}
        counts = {}
        for camera_name in self.camera_names:
            camera_dir = self.cameras_dir / camera_name
            intrinsics, scale = _load_camera_params(camera_dir / "cam_params.json")
            records = self._load_camera_records(camera_dir)
            self.camera_data[camera_name] = {
                "intrinsics": intrinsics,
                "scale": scale,
                "records": records,
            }
            counts[camera_name] = len(records)

        unique_counts = set(counts.values())
        if len(unique_counts) != 1:
            raise ValueError(
                f"All cameras must have the same number of timesteps. counts={counts}"
            )
        self.length = next(iter(unique_counts))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.get_views(idx)

    @staticmethod
    def _load_camera_records(camera_dir: Path):
        results_dir = camera_dir / "results"
        rgb_paths = sorted(results_dir.glob("frame*"))
        depth_paths = sorted(results_dir.glob("depth*"))
        poses = _load_pose_lines(camera_dir / "traj.txt")

        if len(rgb_paths) != len(depth_paths):
            raise ValueError(
                f"RGB frame count ({len(rgb_paths)}) and depth frame count ({len(depth_paths)}) "
                f"do not match for camera {camera_dir.name}"
            )
        if len(rgb_paths) != len(poses):
            raise ValueError(
                f"Frame count ({len(rgb_paths)}) and pose count ({len(poses)}) do not match "
                f"for camera {camera_dir.name}"
            )
        return list(zip(rgb_paths, depth_paths, poses))

    def get_views(self, idx) -> List[RGBDView]:
        views = []
        for camera_name in self.camera_names:
            camera_info = self.camera_data[camera_name]
            rgb_path, depth_path, pose = camera_info["records"][idx]
            views.append(
                RGBDView(
                    camera_name=camera_name,
                    rgb_image=Image.open(rgb_path),
                    depth_image=Image.open(depth_path),
                    pose=pose,
                    intrinsics=np.asarray(camera_info["intrinsics"], dtype=np.float32),
                    scale=float(camera_info["scale"]),
                )
            )
        return views

    def create_pcd(
        self,
        rgb,
        depth,
        camera_pose=None,
        filter_distance=np.inf,
        intrinsics=None,
        scale=None,
    ):
        if intrinsics is None or scale is None:
            raise ValueError("Multi-camera RGB-D creation requires per-view intrinsics and scale.")
        return _create_point_cloud(
            rgb,
            depth,
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
            scale=float(scale),
            camera_pose=camera_pose,
            filter_distance=filter_distance,
        )
