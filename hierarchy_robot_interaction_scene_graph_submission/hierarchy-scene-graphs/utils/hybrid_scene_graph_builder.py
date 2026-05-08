import csv
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from utils.clip_utils import get_img_feats, load_clip_model
from utils.conceptgraph_utils import load_conceptgraph_objects
from utils.geometry_utils import (
    compute_room_embeddings,
    distance_transform,
    find_intersection_share,
    merge_room_objects,
    pcd_denoise_dbscan,
)
from utils.rgbd_data import HOVMultiCameraRGBDDataset, ReplicaStyleRGBDDataset
from utils.room_naming import name_rooms
from utils.topology import Floor, Object, Room


@dataclass
class FullPCDProcessingState:
    frame_indices: list[int]
    filter_distance: float
    enable_icp: bool
    registration_voxel_size: float
    max_correspondence_distance: float
    min_fitness: float
    max_translation_correction: float
    max_rotation_correction_deg: float
    min_registration_points: int
    recent_world_pcds: deque
    processed_frames: int = 0


class HybridSceneGraphBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.full_pcd = o3d.geometry.PointCloud()
        self.floors = []
        self.rooms = []
        self.objects = []
        self.room_masks = {}
        self.graph = nx.Graph()
        self.graph.add_node(0, name="building", type="building")

        self.device = cfg.main.device
        if self.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                self.device = "cpu"

        self.compute_room_embeddings = bool(cfg.pipeline.compute_room_embeddings)
        if (
            not self.compute_room_embeddings
            and (
                cfg.room_naming.method == "view_embedding"
                or cfg.room_naming.fallback_method == "view_embedding"
            )
        ):
            print(
                "Room view embeddings are disabled by config; "
                "view_embedding naming/fallback will be skipped."
            )
        self.clip_model = None
        self.preprocess = None
        self.clip_feat_dim = None
        if self.compute_room_embeddings:
            self.clip_model, self.preprocess, self.clip_feat_dim = load_clip_model(
                cfg.models.clip.type,
                cfg.models.clip.checkpoint,
                device=self.device,
            )

        if cfg.main.dataset == "replica":
            self.dataset = ReplicaStyleRGBDDataset(cfg.main.dataset_path)
        elif cfg.main.dataset == "replica_hov":
            self.dataset = HOVMultiCameraRGBDDataset(cfg.main.dataset_path)
        else:
            raise ValueError(
                "Only 'replica' and 'replica_hov' RGB-D inputs are currently supported."
            )

        self.output_dir = Path(cfg.main.save_path) / cfg.main.dataset / str(cfg.main.scene_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = self.output_dir / cfg.pipeline.debug_dir_name
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.icp_debug_rows = []

    def load_existing_layout(self):
        graph_dir = self.output_dir / "graph"
        floors_dir = graph_dir / "floors"
        rooms_dir = graph_dir / "rooms"
        if not floors_dir.exists() or not rooms_dir.exists():
            raise FileNotFoundError(
                f"Existing layout not found under {graph_dir}. "
                "Run full graph creation once before using --reuse-existing-layout."
            )

        self.floors = []
        self.rooms = []
        self.objects = []
        self.room_masks = {}

        rooms_by_id = {}
        for room_json in sorted(rooms_dir.glob("*.json")):
            room_id = room_json.stem
            room = Room(room_id=room_id, floor_id="")
            room.load(rooms_dir)
            room.objects = []
            room.object_counter = 0
            rooms_by_id[room.room_id] = room
            self.rooms.append(room)

        for floor_json in sorted(floors_dir.glob("*.json")):
            floor_id = floor_json.stem
            floor = Floor(floor_id=floor_id)
            floor.load(floors_dir)
            room_ids = list(floor.rooms)
            floor.rooms = [rooms_by_id[room_id] for room_id in room_ids if room_id in rooms_by_id]
            self.floors.append(floor)

    def _get_frame_indices(self):
        start_frame = int(self.cfg.pipeline.start_frame)
        max_frames = int(self.cfg.pipeline.max_frames)
        frame_indices = list(range(start_frame, len(self.dataset), int(self.cfg.pipeline.skip_frames)))
        if max_frames > 0:
            frame_indices = frame_indices[:max_frames]
        return frame_indices

    def _get_registration_cfg(self):
        registration_cfg = self.cfg.pipeline.get("registration")
        return registration_cfg if registration_cfg is not None else {}

    @staticmethod
    def _combine_point_clouds(point_clouds):
        combined = o3d.geometry.PointCloud()
        for point_cloud in point_clouds:
            combined += point_cloud
        return combined

    @staticmethod
    def _rotation_angle_deg(transform):
        rotation = transform[:3, :3]
        cosine = float((np.trace(rotation) - 1.0) / 2.0)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def _save_icp_debug_log(self):
        if not self.icp_debug_rows:
            return
        debug_path = self.debug_dir / "icp_refinement_metrics.csv"
        fieldnames = [
            "frame_idx",
            "pose_tx",
            "pose_ty",
            "pose_tz",
            "applied",
            "fitness",
            "inlier_rmse",
            "delta_translation_m",
            "delta_rotation_deg",
            "source_points",
            "target_points",
        ]
        with open(debug_path, "w", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.icp_debug_rows)

    @staticmethod
    def _write_debug_point_cloud(path: Path, point_cloud):
        if point_cloud is None or len(point_cloud.points) == 0:
            return
        o3d.io.write_point_cloud(str(path), point_cloud, write_ascii=True)

    @staticmethod # self를 안 써도 되는 클래스 내부 함수라는 뜻 (클래스 안에 들어있긴 하지만 사실상 일반 헬퍼 함수)
    def _xyz_to_point_cloud(xyz, color=None):
        # xyz 라는 numpy 배열을 Open3D point cloud 객체로 바꾸는 함수
        
        pcd = o3d.geometry.PointCloud() # 빈 point cloud 생성
        if xyz is None or xyz.size == 0: # 입력 없으면 걍 빈 point cloud 반환
            return pcd 
        # Open3D는 point 좌표를 그냥 numpy 배열로 안 받고, Vector3dVector로 받음
        pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64)) # 점이 있으면 point cloud 좌표로 넣기
        if color is not None:
            # 점 개수만큼 색을 복붙해서 만들기
            # color = [1.0, 0.3, 0.3]
            # numpy: array([1.0, 0.3, 0.3])
            # reshape(1, 3): [[1.0, 0.3, 0.3]] -> shape이 (3, )에서 (1, 3)로 바뀜
            # np.tile로 [[1.0, 0.3, 0.3]]를 xyz 점 개수만큼 복붙 ex: (100, 1) 행방향으로 100개 복붙, 열방향으로 1번 복붙(즉 1번이라함은 걍그대로 두겠다는 것)
            color_arr = np.tile(np.asarray(color, dtype=np.float64).reshape(1, 3), (xyz.shape[0], 1)) 
            pcd.colors = o3d.utility.Vector3dVector(color_arr) # 지정된 색으로 점 칠하기
        return pcd # point cloud 반환 

    @staticmethod
    def _save_debug_array(path: Path, array, cmap="gray", colorbar=False):
        plt.figure(figsize=(8, 6))
        plt.imshow(array, cmap=cmap, origin="lower")
        if colorbar:
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    @staticmethod
    def _save_debug_scatter(path: Path, points_2d, title):
        plt.figure(figsize=(8, 6))
        if points_2d is not None and len(points_2d) > 0:
            plt.scatter(points_2d[:, 0], points_2d[:, 1], s=0.2)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    @staticmethod
    def _project_points_2d_to_grid(points_2d, x_edges, y_edges, border_offset=10):
        # --- [수정] ---
        # 실제 world 좌표의 2D 점들(x-z 평면)을 room_mask가 만들어진 "정확히 동일한 histogram grid"
        # 좌표계로 되돌리는 helper.
        #
        # points_2d:
        #   - floor_pcd에서 가져온 실제 2D 점들. shape은 (N, 2)
        # x_edges / y_edges:
        #   - np.histogram2d가 실제로 사용한 bin 경계값
        #   - 이번 수정의 핵심은 "min / resolution / 10.5" 같은 추정식이 아니라
        #     histogram이 실제로 사용한 edge를 그대로 재사용하는 것
        # border_offset:
        #   - room_mask/full_map를 만들 때 copyMakeBorder()로 추가한 10 pixel border
        #   - histogram bin index에 이 border offset을 더하면 room_mask index와 바로 맞춰짐
        if points_2d is None or len(points_2d) == 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=bool),
            )

        # --- [수정] ---
        # np.searchsorted를 사용해서 각 실제 점이 어느 histogram bin 안에 들어가는지 계산.
        # 이렇게 하면 room_mask를 만든 histogram grid와 floor_pcd를 다시 올리는 grid가 정확히 같은 체계를 사용하게 됨.
        raw_x = np.searchsorted(x_edges, points_2d[:, 0], side="right") - 1
        raw_y = np.searchsorted(y_edges, points_2d[:, 1], side="right") - 1

        # histogram 오른쪽 마지막 edge와 정확히 같은 점은 마지막 bin에 포함되도록 보정
        raw_x = np.where(points_2d[:, 0] == x_edges[-1], len(x_edges) - 2, raw_x)
        raw_y = np.where(points_2d[:, 1] == y_edges[-1], len(y_edges) - 2, raw_y)

        in_bounds = (
            (points_2d[:, 0] >= x_edges[0])
            & (points_2d[:, 0] <= x_edges[-1])
            & (points_2d[:, 1] >= y_edges[0])
            & (points_2d[:, 1] <= y_edges[-1])
            & (raw_x >= 0)
            & (raw_x < len(x_edges) - 1)
            & (raw_y >= 0)
            & (raw_y < len(y_edges) - 1)
        )

        # room_mask는 histogram 결과에 border 10칸이 추가된 이미지이므로,
        # 실제 histogram bin index에 border_offset을 더해서 mask index로 맞춤
        grid_x = np.clip(raw_x, 0, len(x_edges) - 2).astype(np.int32) + int(border_offset)
        grid_y = np.clip(raw_y, 0, len(y_edges) - 2).astype(np.int32) + int(border_offset)
        return grid_y, grid_x, in_bounds

    @staticmethod
    def _map_room_mask_to_points_2d(room_mask, x_edges, y_edges, border_offset=10):
        # --- [수정] ---
        # room_mask의 흰 픽셀들을 다시 실제 world 좌표의 2D 점들로 되돌리는 helper.
        # 기존 map_grid_to_point_cloud()는 min/resolution/10.5 추정식 기반이었는데,
        # 여기서는 histogram이 실제로 사용한 bin edge를 가지고 cell center를 계산함.
        #
        # 즉 room.vertices 역시 room_mask가 만들어진 것과 같은 histogram grid 정의를 따르게 됨.
        if room_mask is None or room_mask.size == 0:
            return np.empty((0, 2), dtype=np.float64)

        room_mask_binary = (room_mask > 0).astype(np.uint8)
        if border_offset > 0:
            room_mask_binary = room_mask_binary[
                border_offset:-border_offset,
                border_offset:-border_offset,
            ]

        y_cells, x_cells = np.where(room_mask_binary == 1)
        if len(x_cells) == 0:
            return np.empty((0, 2), dtype=np.float64)

        mapped_x_coords = 0.5 * (x_edges[x_cells] + x_edges[x_cells + 1])
        mapped_y_coords = 0.5 * (y_edges[y_cells] + y_edges[y_cells + 1])
        return np.column_stack((mapped_x_coords, mapped_y_coords))

    def _initialize_full_pcd_processing_state(self):
        self.full_pcd = o3d.geometry.PointCloud()
        self.icp_debug_rows = []
        frame_indices = self._get_frame_indices()
        filter_distance = float(self.cfg.pipeline.get("rgbd_filter_distance", np.inf))
        registration_cfg = self._get_registration_cfg()
        enable_icp = bool(registration_cfg.get("enable_icp_refinement", False))
        registration_voxel_size = float(
            registration_cfg.get(
                "voxel_size",
                max(float(self.cfg.pipeline.voxel_size) * 2.0, 0.05),
            )
        )
        max_correspondence_distance = float(
            registration_cfg.get("max_correspondence_distance", 0.08)
        )
        min_fitness = float(registration_cfg.get("min_fitness", 0.25))
        max_translation_correction = float(
            registration_cfg.get("max_translation_correction", 0.15)
        )
        max_rotation_correction_deg = float(
            registration_cfg.get("max_rotation_correction_deg", 8.0)
        )
        reference_window = max(1, int(registration_cfg.get("reference_window", 3)))
        min_registration_points = max(
            25,
            int(registration_cfg.get("min_registration_points", 200)),
        )
        return FullPCDProcessingState(
            frame_indices=frame_indices,
            filter_distance=filter_distance,
            enable_icp=enable_icp,
            registration_voxel_size=registration_voxel_size,
            max_correspondence_distance=max_correspondence_distance,
            min_fitness=min_fitness,
            max_translation_correction=max_translation_correction,
            max_rotation_correction_deg=max_rotation_correction_deg,
            min_registration_points=min_registration_points,
            recent_world_pcds=deque(maxlen=reference_window),
        )

    def _process_full_pcd_frame(self, frame_idx, state):
        views = self.dataset.get_views(frame_idx)
        reference_pose = views[0].pose.copy()
        timestep_pcd = o3d.geometry.PointCloud()
        for view in views:
            local_pcd = self.dataset.create_pcd(
                view.rgb_image,
                view.depth_image,
                camera_pose=None,
                filter_distance=state.filter_distance,
                intrinsics=view.intrinsics,
                scale=view.scale,
            )
            if len(local_pcd.points) == 0:
                continue
            local_pcd.transform(view.pose)
            timestep_pcd += local_pcd

        if len(timestep_pcd.points) == 0:
            state.processed_frames += 1
            return {
                "frame_idx": frame_idx,
                "skipped": True,
                "icp_applied": False,
                "view_pose": reference_pose,
                "processed_frames": state.processed_frames,
                "total_frames": len(state.frame_indices),
                "accumulated_points": len(self.full_pcd.points),
            }

        debug_row = {
            "frame_idx": frame_idx,
            "pose_tx": float(reference_pose[0, 3]),
            "pose_ty": float(reference_pose[1, 3]),
            "pose_tz": float(reference_pose[2, 3]),
            "applied": False,
            "fitness": "",
            "inlier_rmse": "",
            "delta_translation_m": "",
            "delta_rotation_deg": "",
            "source_points": 0,
            "target_points": 0,
        }

        if state.enable_icp and state.recent_world_pcds:
            source_pcd = timestep_pcd.voxel_down_sample(voxel_size=state.registration_voxel_size)
            target_pcd = self._combine_point_clouds(state.recent_world_pcds)
            debug_row["source_points"] = len(source_pcd.points)
            debug_row["target_points"] = len(target_pcd.points)
            if (
                len(source_pcd.points) >= state.min_registration_points
                and len(target_pcd.points) >= state.min_registration_points
            ):
                registration = o3d.pipelines.registration.registration_icp(
                    source_pcd,
                    target_pcd,
                    max_correspondence_distance=state.max_correspondence_distance,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )
                delta_transform = registration.transformation
                delta_translation = float(np.linalg.norm(delta_transform[:3, 3]))
                delta_rotation_deg = self._rotation_angle_deg(delta_transform)
                debug_row.update(
                    {
                        "fitness": float(registration.fitness),
                        "inlier_rmse": float(registration.inlier_rmse),
                        "delta_translation_m": delta_translation,
                        "delta_rotation_deg": delta_rotation_deg,
                    }
                )
                if (
                    registration.fitness >= state.min_fitness
                    and delta_translation <= state.max_translation_correction
                    and delta_rotation_deg <= state.max_rotation_correction_deg
                ):
                    timestep_pcd.transform(delta_transform)
                    debug_row["applied"] = True

        self.full_pcd += timestep_pcd
        state.recent_world_pcds.append(
            timestep_pcd.voxel_down_sample(voxel_size=state.registration_voxel_size)
        )
        if state.enable_icp:
            self.icp_debug_rows.append(debug_row)
        state.processed_frames += 1
        return {
            "frame_idx": frame_idx,
            "skipped": False,
            "icp_applied": bool(debug_row["applied"]),
            "view_pose": reference_pose,
            "processed_frames": state.processed_frames,
            "total_frames": len(state.frame_indices),
            "accumulated_points": len(self.full_pcd.points),
        }

    def _finalize_full_pcd(self):
        if len(self.full_pcd.points) > 0:
            self.full_pcd = self.full_pcd.voxel_down_sample(
                voxel_size=float(self.cfg.pipeline.voxel_size)
            )
            if len(self.full_pcd.points) > 0:
                self.full_pcd = pcd_denoise_dbscan(self.full_pcd, eps=0.01, min_points=100)
        self._save_icp_debug_log()
        return self.full_pcd

    def create_full_pcd(self):
        state = self._initialize_full_pcd_processing_state()
        for frame_idx in tqdm(state.frame_indices, desc="Creating RGB-D point cloud"):
            self._process_full_pcd_frame(frame_idx, state)
        return self._finalize_full_pcd()

    def create_full_pcd_stepwise(self):
        state = self._initialize_full_pcd_processing_state()
        if not state.frame_indices:
            print("No RGB-D frames selected for full point cloud creation.")
            return self._finalize_full_pcd()

        glfw_key_escape = 256
        glfw_key_right = 262
        display_pcd = o3d.geometry.PointCloud()
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        controls = {
            "step_requested": False,
            "finished": False,
            "view_initialized": False,
            "follow_pose_view": True,
        }

        def sync_display_geometry(vis, reset_view=False):
            display_pcd.points = o3d.utility.Vector3dVector(
                np.asarray(self.full_pcd.points).copy()
            )
            display_colors = np.asarray(self.full_pcd.colors)
            if display_colors.size == 0:
                display_colors = np.empty((0, 3), dtype=np.float64)
            else:
                display_colors = display_colors.copy()
            display_pcd.colors = o3d.utility.Vector3dVector(display_colors)
            vis.update_geometry(display_pcd)
            if reset_view and len(self.full_pcd.points) > 0:
                vis.reset_view_point(True)
                controls["view_initialized"] = True

        def request_next_frame(_):
            if not controls["finished"]:
                controls["step_requested"] = True
            return False

        def reset_current_view(vis):
            sync_display_geometry(vis, reset_view=True)
            print("Refit view to the current accumulated point cloud.")
            return False

        def update_view_from_pose(vis, camera_pose):
            if camera_pose is None:
                return
            view_ctrl = vis.get_view_control()
            camera_param = view_ctrl.convert_to_pinhole_camera_parameters()
            camera_param.extrinsic = np.linalg.inv(camera_pose)
            view_ctrl.convert_from_pinhole_camera_parameters(camera_param, allow_arbitrary=True)
            view_ctrl.camera_local_translate(forward=-0.4, right=0.0, up=0.0)

        def toggle_follow_pose(_):
            controls["follow_pose_view"] = not controls["follow_pose_view"]
            state_text = "enabled" if controls["follow_pose_view"] else "disabled"
            print(f"Follow-pose camera is now {state_text}.")
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, lambda visualizer: visualizer.destroy_window() or False)
        vis.register_key_callback(glfw_key_right, request_next_frame)
        vis.register_key_callback(ord("."), request_next_frame)
        vis.register_key_callback(ord(">"), request_next_frame)
        vis.register_key_callback(ord("R"), reset_current_view)
        vis.register_key_callback(ord("F"), toggle_follow_pose)

        if not vis.create_window("Full PCD Step-Through", width=1280, height=720):
            raise RuntimeError("Failed to create Open3D window for step-through full PCD mode.")
        vis.add_geometry(display_pcd, reset_bounding_box=False)
        vis.add_geometry(coordinate_frame, reset_bounding_box=False)
        render_option = vis.get_render_option()
        if render_option is not None:
            render_option.point_size = 2.0

        print(
            "Step-through full PCD mode: press [Right Arrow] or [.] / [>] to process one frame. "
            "Use mouse drag / wheel to explore, press [F] to toggle follow-pose view, press [R] to refit the view, "
            "and close the window or press [ESC] to stop."
        )
        sync_display_geometry(vis)

        while vis.poll_events():
            if controls["step_requested"] and not controls["finished"]:
                controls["step_requested"] = False
                frame_idx = state.frame_indices[state.processed_frames]
                frame_result = self._process_full_pcd_frame(frame_idx, state)
                sync_display_geometry(
                    vis,
                    reset_view=not controls["view_initialized"] and not frame_result["skipped"],
                )
                if controls["follow_pose_view"]:
                    update_view_from_pose(vis, frame_result["view_pose"])
                status = "skipped (empty local point cloud)" if frame_result["skipped"] else "processed"
                icp_suffix = " [ICP applied]" if frame_result["icp_applied"] else ""
                print(
                    f"[{frame_result['processed_frames']}/{frame_result['total_frames']}] "
                    f"frame {frame_idx}: {status}{icp_suffix} | "
                    f"accumulated points={frame_result['accumulated_points']}"
                )
                if state.processed_frames == len(state.frame_indices):
                    self._finalize_full_pcd()
                    sync_display_geometry(vis)
                    controls["finished"] = True
                    print("All frames processed. Close the window to continue the pipeline.")
            vis.update_renderer()

        vis.destroy_window()
        if state.processed_frames < len(state.frame_indices):
            raise RuntimeError(
                "Step-through full PCD creation was stopped before all selected frames were processed."
            )
        return self.full_pcd

    def save_full_pcd(self):
        o3d.io.write_point_cloud(
            str(self.output_dir / "full_pcd.ply"),
            self.full_pcd,
            write_ascii=True,
        )

    def _create_single_floor(self, min_height, max_height):
        floors = [[min_height, max_height]]
        print("floors", floors)
        print("number of floors: ", len(floors))
        for i, floor in enumerate(floors):
            floor_obj = Floor(str(i), name=f"floor_{i}")
            floor_pcd = self.full_pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(-np.inf, floor[0], -np.inf),
                    max_bound=(np.inf, floor[1], np.inf),
                )
            )
            bbox = floor_pcd.get_axis_aligned_bounding_box()
            floor_obj.vertices = np.asarray(bbox.get_box_points())
            floor_obj.pcd = floor_pcd
            if np.asarray(floor_pcd.points).shape[0] == 0:
                floor_obj.floor_zero_level = float(min_height)
                floor_obj.floor_height = float(max_height - min_height)
            else:
                floor_obj.floor_zero_level = float(np.min(np.asarray(floor_pcd.points)[:, 1]))
                floor_obj.floor_height = float(floor[1] - floor_obj.floor_zero_level)
            self.floors.append(floor_obj)
        return floors

    def segment_floors(self):
        downpcd = self.full_pcd.voxel_down_sample(voxel_size=0.05)
        downpcd = np.asarray(downpcd.points)
        print("downpcd", downpcd.shape)

        resolution = 0.01
        bins = np.abs(np.max(downpcd[:, 1]) - np.min(downpcd[:, 1])) / resolution
        print("bins", bins)
        z_hist = np.histogram(downpcd[:, 1], bins=int(bins))
        z_hist_smooth = gaussian_filter1d(z_hist[0], sigma=2)
        distance_bins = 0.2 / resolution
        print("distance", distance_bins)
        print(np.mean(z_hist_smooth))
        min_peak_height = np.percentile(z_hist_smooth, 90)
        print("min_peak_height", min_peak_height)
        peaks, _ = find_peaks(z_hist_smooth, distance=distance_bins, height=min_peak_height)

        min_height = float(np.min(downpcd[:, 1]))
        max_height = float(np.max(downpcd[:, 1]))
        if len(peaks) < 2:
            print("Not enough floor peaks found, falling back to a single-floor segmentation.")
            return self._create_single_floor(min_height, max_height)

        peaks_locations = z_hist[1][peaks]
        clustering = DBSCAN(eps=1, min_samples=1).fit(peaks_locations.reshape(-1, 1))
        labels = clustering.labels_

        # peaks를 높이 순으로 정렬 (ex: [바닥1, 천장1, 바닥2, 천장2, ...])
        clustered_peaks = []
        for i in range(len(np.unique(labels))):
            p = peaks[labels == i]
            if i == 0 or i == len(np.unique(labels)) - 1:
                top_p = p[np.argsort(z_hist_smooth[p])[-1:]].tolist()
            else:
                top_p = p[np.argsort(z_hist_smooth[p])[-2:]].tolist()
            top_p = [z_hist[1][peak] for peak in top_p]
            clustered_peaks.append(top_p)
        clustered_peaks = np.sort([item for sublist in clustered_peaks for item in sublist])
        print("clustred_peaks", clustered_peaks)

        floors = [] # peaks 리스트에서 0번-1번을 한 층, 2번-3번을 다음 층으로 간주
        # 2칸씩 띄어가며(step=2) 쌍을 만듦
        for i in range(0, len(clustered_peaks) - 1, 2):
            floors.append([clustered_peaks[i], clustered_peaks[i + 1]])
        if len(floors) == 0:
            print("No floor ranges inferred from peaks, falling back to a single-floor segmentation.")
            return self._create_single_floor(min_height, max_height)

        print("floors", floors)
        floors[0][0] = (floors[0][0] + min_height) / 2
        floors[-1][1] = (floors[-1][1] + max_height) / 2
        print("number of floors: ", len(floors))

        for i, floor in enumerate(floors):
            floor_obj = Floor(str(i), name=f"floor_{i}")
            floor_pcd = self.full_pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(-np.inf, floor[0], -np.inf),
                    max_bound=(np.inf, floor[1], np.inf),
                )
            )
            bbox = floor_pcd.get_axis_aligned_bounding_box()
            floor_obj.vertices = np.asarray(bbox.get_box_points())
            floor_obj.pcd = floor_pcd
            floor_obj.floor_zero_level = float(np.min(np.asarray(floor_pcd.points)[:, 1]))
            floor_obj.floor_height = float(floor[1] - floor_obj.floor_zero_level)
            self.floors.append(floor_obj)
        return floors

    def _compute_room_feature_context(self, floor_pcd):
        if not self.compute_room_embeddings:
            return None, None, None

        rgb_list = []
        pose_list = []
        global_feats = []
        all_global_clip_feats = {}
        frame_indices = self._get_frame_indices()
        for _, img_id in tqdm(
            enumerate(frame_indices),
            desc="Computing room features",
            total=len(frame_indices),
        ):
            rgb_image, _, pose = self.dataset[img_id]
            global_feat = get_img_feats(np.array(rgb_image), self.preprocess, self.clip_model)
            all_global_clip_feats[str(img_id)] = global_feat
            rgb_list.append(rgb_image)
            pose_list.append(pose)
            global_feats.append(global_feat)

        np.savez(self.debug_dir / "room_views.npz", **all_global_clip_feats)
        pcd_min = np.min(np.asarray(floor_pcd.points), axis=0)
        pcd_max = np.max(np.asarray(floor_pcd.points), axis=0)
        return pose_list, global_feats, frame_indices, pcd_min, pcd_max

    def _add_single_room(self, floor, floor_pcd):
        room_index = len(floor.rooms)
        room = Room(f"{floor.floor_id}_{room_index}", floor.floor_id, name=f"room_{room_index}")
        room.pcd = floor_pcd
        room.vertices = np.asarray(floor_pcd.points)[:, [0, 2]]
        room.room_height = floor.floor_height
        room.room_zero_level = floor.floor_zero_level
        floor.add_room(room)
        self.rooms.append(room)

    def segment_rooms(self, floor):
        # 저장 경로 설정
        tmp_floor_path = self.debug_dir / str(floor.floor_id)
        tmp_floor_path.mkdir(parents=True, exist_ok=True)
        debug_stats = {
            "floor_id": floor.floor_id,
            "floor_zero_level": None,
            "floor_height": None,
            "original_points": 0,
            "wall_slice_points": 0,
            "boundary_slice_points": 0,
            "grid_resolution": float(self.cfg.pipeline.grid_resolution),
            "num_bins": None,
            "wall_hist_threshold": None,
            "num_room_masks": 0,
        }

        floor_pcd = floor.pcd
        """
        xyz 형태
        - 한 행: point 1개
        - 열 0: x
        - 열 1: y
        - 열 2: z
        [
            [x1, y1, z1],
            [x2, y2, z2],
            [x3, y3, z3],
            ...
        ]
        """
        xyz = np.asarray(floor_pcd.points)
        debug_stats["original_points"] = int(xyz.shape[0])
        self._write_debug_point_cloud(tmp_floor_path / "00_floor_full.ply", floor_pcd)

        # 데이터가 없으면 단일 방으로 처리하고 종료
        if xyz.shape[0] == 0:
            self._add_single_room(floor, floor_pcd)
            self.room_masks[floor.floor_id] = [np.ones((1, 1), dtype=np.uint8)]
            with open(tmp_floor_path / "room_segmentation_debug_stats.json", "w", encoding="utf-8") as outfile:
                json.dump(debug_stats, outfile, indent=2)
            return

        xyz_full = xyz.copy() # xyz를 복사해서 xyz_full이라는 새 배열 생성
        floor_zero_level = floor.floor_zero_level # 바닥 높이
        floor_height = floor.floor_height # 천장 높이
        debug_stats["floor_zero_level"] = float(floor_zero_level)
        debug_stats["floor_height"] = float(floor_height)

        # -------------------------------------------------- 1. 벽을 잘 찾기 위해 눈높이 구간만 추출 --------------------------------------------------
        # 가구 등에 의한 노이즈를 줄이고 벽면 데이터를 얻기 위함

        # xyz[:, 1]: 모든 행의 1번 열(y 값)만 뽑기
        # xyz[:, 1] < (floor_zero_level + floor_height - 0.1)로 boolean 배열 뽑기
        # xyz[boolean 배열] -> True인 값들로 이루어진 배열 뽑힘
        # 즉, 이건 천장 바로 위쪽 점은 제거한 상태
        xyz = xyz[xyz[:, 1] < floor_zero_level + floor_height - 0.1] 

        # 여기는 위와 같은 방식으로 y가 2.0m 이상인 부분만 남김
        # 즉 최종적으로 높은 벽 부분만 남기게 된 상태 (바닥 위 2m이상 ~ 천장 아래 0.1m )
        xyz = xyz[xyz[:, 1] >= floor_zero_level + 2.0]


        # 전체 경계 계산을 위한 데이터 (천장 근처까지 포함)
        # 얘는 천장 윗부분만 자르고 밑 부분은 그대로 둠
        # 점이 만 개 있는 예시: xyz.shape = (10000, 3)
        xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + floor_height - 0.1]
        # 남아 있는 점 개수 세기
        debug_stats["wall_slice_points"] = int(xyz.shape[0]) # 행 개수 = 점개수 
        debug_stats["boundary_slice_points"] = int(xyz_full.shape[0]) # 행 개수 = 점개수
        self._write_debug_point_cloud(
            tmp_floor_path / "01_wall_height_slice_y_ge_floor_plus_2m.ply",
            self._xyz_to_point_cloud(xyz, color=[1.0, 0.3, 0.3]), # _xyz_to_point_cloud 함수는 numpy 배열 xyz를 Open3D point cloud 객체로 바꿈
        )
        self._write_debug_point_cloud(
            tmp_floor_path / "02_boundary_height_slice_below_ceiling.ply",
            self._xyz_to_point_cloud(xyz_full, color=[0.3, 0.6, 1.0]),
        )

        # xyz에 점이 하나도 없거나 xyz_full에 점이 하나도 없으면 종료
        # - 이 상태론 room segmentation 불가능 하니깡
        # - floor 전체를 room 1개로 등록하고 끝냄
        if xyz.shape[0] == 0 or xyz_full.shape[0] == 0:
            self._add_single_room(floor, floor_pcd)
            self.room_masks[floor.floor_id] = [np.ones((1, 1), dtype=np.uint8)] # room_mask도 최소한 하나 넣어두기
            with open(tmp_floor_path / "room_segmentation_debug_stats.json", "w", encoding="utf-8") as outfile:
                json.dump(debug_stats, outfile, indent=2)
            return

        # -------------------------------------------------- 2. 3D points를 바닥(x-z축)으로 투영해서 2D 맵 생성 --------------------------------------------------
        pcd_2d = xyz[:, [0, 2]] # 모든 행(포인트)에서 x, z 값만 가져옴 (즉, [x, y, z] -> [x, z])
        xyz_full = xyz_full[:, [0, 2]] # 모든 행(포인트)에서 x, z 값만 가져옴 (즉, [x, y, z] -> [x, z])
        # pcd_2d를 scatter plot으로 저장
        #   - wall slice가 위에서 봤을 때 어떻게 분포하는지 보여줌
        self._save_debug_scatter(
            tmp_floor_path / "03_wall_slice_xz_projection.png",
            pcd_2d,
            "Wall Slice XZ Projection",
        )
        self._save_debug_scatter(
            tmp_floor_path / "04_boundary_slice_xz_projection.png",
            xyz_full,
            "Boundary Slice XZ Projection",
        )

        # -------------------------------------------------- 3. 2D Wall map 생성 로직 --------------------------------------------------
        
        # 03_wall_slice_xz_projection.png에 찍힌 점들을 이미지처럼 생긴 2D wall map으로 바꾸는 단계
        # grid_size는 이 floor가 x, z축으로 대충 몇 m 정도 퍼져 있는지 나타내는 값
        grid_size = (
            int(np.max(pcd_2d[:, 0]) - np.min(pcd_2d[:, 0])) + 1, # x의 길이 범위
            int(np.max(pcd_2d[:, 1]) - np.min(pcd_2d[:, 1])) + 1, # z의 길이 범위
        )
        # resolution은 한 칸 크기
        #   - 0.05면 한 칸이 5cm
        resolution = float(self.cfg.pipeline.grid_resolution)
        print("grid_size: ", resolution)

        # //는 나눗셈 후 소숫점 버림
        # 전체 길이를 5cm 칸으로 몇 칸 나눌지 계산
        # 결과적으로 num_bins은 2D 격자 크기
        #   - 첫 번째 값은 세로칸 수
        #   - 두 번째 값은 가로칸 수
        num_bins = (
            int(grid_size[1] // resolution) + 1,
            int(grid_size[0] // resolution) + 1,
        )
        debug_stats["num_bins"] = [int(num_bins[0]), int(num_bins[1])]

        # 2D 히스토그램으로 점들의 밀도 counting
        # 즉 hist[r, c] 값이 크면 그 칸에 벽 점이 많이 모여 있다는 의미
        # 첫 번째 인자로 z, 두 번째 인자로 x를 넣어서, 결과 배열이 이미지처럼 세로=z, 가로=x가 되게 만든 것
        #
        # --- [수정] ---
        # 여기서 반환되는 y_edges / x_edges를 꼭 저장해둔다.
        # 이유는 나중에 room_mask를 다시 실제 point cloud에 매핑할 때,
        # "같은 histogram grid 좌표계"를 그대로 재사용해야 좌표계 mismatch가 안 생기기 때문.
        hist, y_edges, x_edges = np.histogram2d(pcd_2d[:, 1], pcd_2d[:, 0], bins=num_bins)
        hist_raw = hist.copy() # 05_wall_histogram_raw.png

        # 이거 평탄화 작업(Histogram stretching) hist는 점 개수 배열인데 값 범위가 제멋대로 일 수 있으므로 가장 큰 값은 255, 가장 작은 값은 0근처로
        # GaussianBlur 적용해서 노이즈 제거
        #   - 주변 칸까지 부드럽게 퍼뜨리는 작업
        #   - 목적은 끊긴 벽을 좀 연결하고 노이즈를 줄이는 것
        #   - 대신 약한 벽은 더 흐려질 가능성
        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) 
        hist = cv2.GaussianBlur(hist, (5, 5), 1)
        hist_blur = hist.copy() # 06_wall_histogram_blur.png

        # 임계값 처리로 벽면 후보군(Binary image) 생성
        #   - hist_threshold 보다 큰 칸만 흰색 255, 나머지는 검정 0
        #   - 이 값을 작게 잡으면 노이즈가 좀 낄꺼고, 크게 잡으면 벽이 없어지공..
        #   - 이걸 내 데이터셋 구조에 맞게 설정해야 할 것 같다.
        hist_threshold = 0.25 * np.max(hist)
        debug_stats["wall_hist_threshold"] = float(hist_threshold)
        _, walls_threshold = cv2.threshold(hist, hist_threshold, 255, cv2.THRESH_BINARY) # 07_wall_threshold.png
        
        # 벽 맵 바깥에 검정 테두리 10픽셀 붙이기
        walls_skeleton = cv2.copyMakeBorder(
            walls_threshold, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )
        # MORPH_CROSS는 작은 틈 메우는 역할
        #   - 벽 선이 조금 끊겨 있으면 이어붙여 줌
        #   - 그런데 욕실 문 근처나 세면대 주변처럼 좁은 틈도 같이 메워버릴 가능성도..
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        walls_skeleton = cv2.morphologyEx(walls_skeleton, cv2.MORPH_CLOSE, kernel, iterations=1) # 08_wall_map.png
        self._save_debug_array(
            tmp_floor_path / "05_wall_histogram_raw.png",
            hist_raw,
            cmap="inferno",
            colorbar=True,
        )
        self._save_debug_array(
            tmp_floor_path / "06_wall_histogram_blur.png",
            hist_blur,
            cmap="inferno",
            colorbar=True,
        )
        self._save_debug_array(
            tmp_floor_path / "07_wall_threshold.png",
            walls_threshold,
            cmap="gray",
        )
        self._save_debug_array(
            tmp_floor_path / "08_wall_map.png",
            walls_skeleton,
            cmap="gray",
        )

        # -------------------------------------------------- 4. 건물 전체 외곽/실내 footprint 생성 --------------------------------------------------
        
        # xyz_full의 2D 점들을 grid에 넣어서 전체 점유 밀도맵 생성
        #   - 여기엔 벽뿐 아니라 바닥/가구/오브젝트도 많이 포함됨
        # --- [수정] ---
        # boundary 쪽 histogram도 wall histogram과 완전히 같은 bin edge를 사용.
        # 그래야 walls_skeleton / outside_boundary / full_map이 모두 같은 이미지 좌표계에 놓이게 됨.
        hist_full, _, _ = np.histogram2d(xyz_full[:, 1], xyz_full[:, 0], bins=[y_edges, x_edges])
        hist_full = cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # 값 범위를 0~255로 맞춤, stertching 해서 명암 대비 주는 역할
        hist_full_raw = hist_full.copy() # 09_boundary_histogram_raw.png (normalized raw map)

        # 꽤 강하게 blur 적용
        #   - 커널이 커서 작은 빈틈은 쉽게 메워질 수 있음
        hist_full = cv2.GaussianBlur(hist_full, (21, 21), 2)
        hist_full_blur = hist_full.copy() # 10_boundary_histogram_blur.png (부드럽게 퍼진 외곽 후보)

        # 0보다 크기만 하면 전부 흰색으로 만들기
        #   - blur 후 조금이라도 값이 남은 곳은 전부 실내 후보
        _, outside_boundary = cv2.threshold(hist_full, 0, 255, cv2.THRESH_BINARY)
        outside_boundary_threshold = outside_boundary.copy() # 11_outside_boundary_threshold.png

        # 가장자리 border 추가
        outside_boundary = cv2.copyMakeBorder(
            outside_boundary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0  
        ) 
        # CLOSE로 바깥 경계를 더 매끈하게 연결
        #   - 좁은 틈이나 작은 구멍은 더 쉽게 메워짐
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        outside_boundary = cv2.morphologyEx(
            outside_boundary, cv2.MORPH_CLOSE, kernel, iterations=3
        )
        outside_boundary_closed = outside_boundary.copy() # 12_outside_boudary_closed.png

        self._save_debug_array(
            tmp_floor_path / "09_boundary_histogram_raw.png",
            hist_full_raw,
            cmap="inferno",
            colorbar=True,
        )
        self._save_debug_array(
            tmp_floor_path / "10_boundary_histogram_blur.png",
            hist_full_blur,
            cmap="inferno",
            colorbar=True,
        )
        self._save_debug_array(
            tmp_floor_path / "11_outside_boundary_threshold.png",
            outside_boundary_threshold,
            cmap="gray",
        )
        self._save_debug_array(
            tmp_floor_path / "12_outside_boundary_closed.png",
            outside_boundary_closed,
            cmap="gray",
        )

        # -------------------------------------------------- 5. full map 생성 (walls_skeleton + outside_boundary)  --------------------------------------------------

        # outside_boundary 이미지에서 바깥 경계선들 찾기
        #   - outside_boundary는 이전 단계에서 만든 실내 footprint 후보 (12_outside_boundary_closed.png)
        #   - cv2.RETR_EXTERNAL은 가장 바깥 contour만 찾으라는 의미
        #   - 즉, 내부 구멍이나 내부 작은 경계는 무시하고, 바깥쪽 큰 외각선들만 뽑음
        contours, _ = cv2.findContours(
            outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # 빈 이미지 새로 생성
        #   - 아까까지 threshold/close 결과가 들어 있던 outside_boundary를 버리고
        #   - contour를 다시 깔끔하게 채워 그릴 빈 캔버스 생성한단 뜻
        outside_boundary = np.zeros_like(outside_boundary)
        # 찾은 contour들을 새 이미지에 다시 그리기
        #   - 처음 -1: 모든 contour 다 그리기
        #   - (255, 255, 255): 흰색
        #   - 마지막 -1: 선만 그리지 말고 내부까지 채우기
        #   - 결과적으로 실내 footprint의 바깥선을 따라 내부를 꽉 채운 흰색 마스크 생성
        cv2.drawContours(outside_boundary, contours, -1, (255, 255, 255), -1) 
        outside_boundary = outside_boundary.astype(np.uint8) # 자료형을 이미지용 8비트 정수로 맞추는 작업: OpenCV랑 bitwise 연산하려면 보통 이 형식이 안정적
        self._save_debug_array(
            tmp_floor_path / "13_outside_boundary_filled.png",
            outside_boundary,
            cmap="gray",
        )

        # full_map 생성
        #   - outside_boundary를 cv2.bitwise_not으로 흰색/검정 반전
        #       -- 실내 영역(흰색) -> 검정
        #       -- 실외 영역(검정) -> 흰색
        #   - cv2.bitwise_or로 walls_skeleton과 실외 영역을 OR로ㅗ 합침
        #       -- 최종적으로 흰색이 되는 부분
        #           --- 벽인 곳
        #           --- 실외인 곳
        full_map = cv2.bitwise_or(walls_skeleton, cv2.bitwise_not(outside_boundary))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        full_map = cv2.morphologyEx(full_map, cv2.MORPH_CLOSE, kernel, iterations=2)
        self._save_debug_array(
            tmp_floor_path / "14_full_map_before_distance_transform.png",
            full_map,
            cmap="gray",
        )

        # -------------------------------------------------- 6. distance_transform  --------------------------------------------------

        # full_map을 입력으로 받아서 방 중심 seed들을 찾고 seed를 기준으로 각 방 영역을 나눌 준비
        #   - vertices는 watershed로 분할된 방 영역들의 픽셀 좌표 집합        
        room_vertices = distance_transform(full_map, resolution, str(tmp_floor_path))

        # seed를 하나도 못 찾았거나 분할 결과가 하나도 안 나왔으면 fallback으로 floor 전체를 방 1개로 처리하고 끝냄
        if len(room_vertices) == 0:
            self._add_single_room(floor, floor_pcd)
            self.room_masks[floor.floor_id] = [np.ones_like(full_map, dtype=np.uint8)]
            with open(tmp_floor_path / "room_segmentation_debug_stats.json", "w", encoding="utf-8") as outfile:
                json.dump(debug_stats, outfile, indent=2)
            return
        
        # -------------------------------------------------- 7. 2D room segment -> 3D room segment --------------------------------------------------

        # 2D 방 영역을 다시 3D points로 매핑
        room_pcds = [] # 최종 3D point cloud
        room_masks = [] # 2D 이미지 마스크
        room_2d_points = [] # 실제 단위의 2D 좌표

        # --- [수정] ---
        # 기존 방식은:
        #   1) 2D room mask를 가상의 3D 방 부피로 extrude하고
        #   2) 그 가짜 부피 점마다 가장 가까운 실제 floor_pcd 점을 KDTree로 끌어왔음.
        #
        # 그런데 이 방식은 "벽을 사이에 둔 가까운 점"도 최근접점으로 뽑아버릴 수 있어서,
        # 얇은 벽 너머 세면대/욕실 점이 현재 방 room_pcd로 딸려오는 leakage 문제가 생길 수 있음.
        #
        # 그래서 이제는:
        #   - 실제 floor_pcd의 각 점을 x-z 평면으로 직접 grid에 투영하고
        #   - 그 점이 room_mask의 흰 칸 안에 들어가는지로 방 소속을 결정함.
        #
        # 즉 "가짜 점 -> 실제 점 최근접 검색"이 아니라
        # "실제 점 -> room_mask membership 검사"로 바꾼 것.
        floor_points = np.asarray(floor_pcd.points)
        floor_points_2d = floor_points[:, [0, 2]]
        floor_grid_y, floor_grid_x, floor_points_in_grid = self._project_points_2d_to_grid(
            floor_points_2d,
            x_edges,
            y_edges,
        )

        for i in tqdm(range(len(room_vertices)), desc="Assign floor points to rooms"):
            room_mask = np.zeros_like(full_map)

            # 2D room 영역 mask 생성 및 실제 좌표 변환
            #   - 각 방의 꼭짓점(vertices) 정보를 바탕으로 2D 이미지 형태의 mask 생성
            #   - 그런 다음 픽셀 단위의 2D mask를 map_grid_to_point_cloud를 통해 실제 물리적 스케일의 2D 점 좌표(room_points_2d로 변환)
            room_mask[room_vertices[i][0], room_vertices[i][1]] = 255 # 현재 방에 해당하는 픽셀만 흰색 255로 칠하는 것
            room_masks.append(room_mask)
            self._save_debug_array(
                tmp_floor_path / f"20_room_mask_{i:02d}.png",
                room_mask,
                cmap="gray",
            )

            # --- [수정] ---
            # 방 마스크의 흰 픽셀들을 실제 월드 좌표의 2D 점들로 바꾸기.
            # 기존에는 min/resolution/10.5 추정식 기반 역변환을 썼지만,
            # 이제는 histogram이 실제로 사용한 x_edges / y_edges로 각 cell center를 계산함.
            room_points_2d = self._map_room_mask_to_points_2d(room_mask, x_edges, y_edges)
            room_2d_points.append(room_points_2d)

            # --- [수정] ---
            # 실제 floor_pcd 점들 중, 현재 room_mask의 흰 칸에 투영되는 점만 선택.
            # 이렇게 하면 room_mask 밖의 세면대/욕실 점을 최근접점으로 잘못 끌고 오는 문제가 크게 줄어듦.
            room_membership = np.zeros(floor_points.shape[0], dtype=bool)
            valid_floor_indices = np.where(floor_points_in_grid)[0]
            room_membership[valid_floor_indices] = (
                room_mask[
                    floor_grid_y[valid_floor_indices],
                    floor_grid_x[valid_floor_indices],
                ]
                > 0
            )
            room_point_indices = np.flatnonzero(room_membership)

            # 현재 방 mask 안에 실제로 떨어진 floor point들만 room_pcd로 사용
            room_pcd = floor_pcd.select_by_index(room_point_indices.tolist())
            room_pcds.append(room_pcd)
            self._write_debug_point_cloud(
                tmp_floor_path / f"21_room_pcd_{i:02d}.ply",
                room_pcd,
            )

        self.room_masks[floor.floor_id] = room_masks
        debug_stats["num_room_masks"] = int(len(room_masks))
        with open(tmp_floor_path / "room_segmentation_debug_stats.json", "w", encoding="utf-8") as outfile:
            json.dump(debug_stats, outfile, indent=2)

        if self.compute_room_embeddings:
            pose_list, global_feats, frame_indices, pcd_min, pcd_max = self._compute_room_feature_context(
                floor_pcd
            )
            repr_embs_list, repr_img_ids_list = compute_room_embeddings(
                room_pcds,
                pose_list,
                global_feats,
                pcd_min,
                pcd_max,
                10,
                str(tmp_floor_path),
            )
        else:
            frame_indices = self._get_frame_indices()
            repr_embs_list = [[] for _ in room_pcds]
            repr_img_ids_list = [[] for _ in room_pcds]

        room_index = 0
        for i in range(len(room_2d_points)):
            room = Room(f"{floor.floor_id}_{room_index}", floor.floor_id, name=f"room_{room_index}")
            room.pcd = room_pcds[i]
            room.vertices = room_2d_points[i]
            room.room_height = floor_height
            room.room_zero_level = floor.floor_zero_level
            room.embeddings = repr_embs_list[i]
            room.represent_images = [frame_indices[k] for k in repr_img_ids_list[i]]
            floor.add_room(room)
            self.rooms.append(room)
            room_index += 1

        print(f"number of rooms in floor {floor.floor_id} is {len(floor.rooms)}")

    def assign_existing_objects_to_rooms(self, objects_data):
        self.objects = []
        for room in self.rooms:
            room.objects = [] # 기존에 방에 할당된 객체 리스트 초기화
            room.object_counter = 0 # 방별 객체 ID 생성을 위한 카운터 초기화

        # 설정값 로드: 층/방 판정 시 여유 오차 범위
        floor_margin = float(self.cfg.assignment.floor_margin)
        room_overlap_radius = float(self.cfg.assignment.room_overlap_radius)

        # 층별 오브젝트 분류
        pbar = tqdm(enumerate(self.floors), total=len(self.floors), desc="Floor: ")
        for f_idx, floor in pbar:
            pbar.set_description(f"Floor: {f_idx}")
            objects_inside_floor = []
            for obj_idx, obj_data in enumerate(objects_data):
                pcd = obj_data.get("pcd")
                if pcd is None or pcd.is_empty():
                    continue
                points = np.asarray(pcd.points)
                if points.shape[0] == 0:
                    continue

                # 객체의 높이 범위(Y축) 계산
                min_y = np.min(points[:, 1]) # 코드상 [:, 1]이 높이(Height)임
                max_y = np.max(points[:, 1])
                # 객체가 해당 층의 바닥~천장 높이 범위 내에 있는지 확인 (margin 포함)
                if min_y > floor.floor_zero_level - floor_margin and max_y < (
                    floor.floor_zero_level + floor.floor_height + floor_margin
                ):
                    objects_inside_floor.append(obj_idx)

            obj_pbar = tqdm(
                enumerate(objects_inside_floor),
                total=len(objects_inside_floor),
                desc="Object: ",
                leave=False,
            )
            for _, obj_idx in obj_pbar:
                obj_data = objects_data[obj_idx]
                points = np.asarray(obj_data["pcd"].points)

                # 객체의 2D 평면 좌표(X, Z)만 추출 (위에서 내려다본 위치)
                points_xz = points[:, [0, 2]] 
                room_assoc = [] # 각 방과의 연관성(겹침 정도) 저장 리스트
                # 현재 층에 있는 모든 방을 돌며 겹침 정도 계산
                for room in floor.rooms:
                    # find_intersection_share: 객체의 평면 좌표가 방의 영역(vertices)과 얼마나 겹치는지 계산
                    room_assoc.append(
                        find_intersection_share(room.vertices, points_xz, room_overlap_radius)
                    )
                # 만약 벽 속이나 복도 등 정의되지 않은 곳에 있어 겹침이 0이라면
                if np.sum(room_assoc) == 0:
                    # 가장 가까운 방의 중심점까지의 거리를 계산 (음수값으로 넣어 argmax 사용)
                    for r_idx, room in enumerate(floor.rooms):
                        room_assoc[r_idx] = -1 * np.linalg.norm(
                            np.mean(room.vertices, axis=0) - np.mean(points_xz, axis=0)
                        )
                # 연관성이 가장 높은(가장 많이 겹치거나 가장 가까운) 방 선택
                closest_room_idx = int(np.argmax(room_assoc))
                parent_room = floor.rooms[closest_room_idx]
                objectt = Object(
                    f"{parent_room.room_id}_{parent_room.object_counter}", # 고유 ID 생성
                    parent_room.room_id,
                )
                # 데이터 채우기
                parent_room.object_counter += 1
                objectt.name = obj_data.get("name", "object")
                objectt.pcd = obj_data["pcd"]
                objectt.vertices = points_xz
                objectt.embedding = obj_data.get("embedding")
                objectt.caption = obj_data.get("caption")
                objectt.source_id = obj_data.get("source_id")
                obj_pbar.set_description(f"object name: {objectt.name}, {objectt.object_id}")
                # 방 객체와 전체 리스트에 각각 추가
                parent_room.add_object(objectt)
                self.objects.append(objectt)
        print("number of objects: ", len(self.objects))

    def _get_room_center_2d(self, room):
        if room.pcd is not None and not room.pcd.is_empty():
            points = np.asarray(room.pcd.points)
            if points.shape[0] > 0:
                return np.mean(points[:, [0, 2]], axis=0)
        vertices = np.asarray(room.vertices)
        if vertices.size == 0:
            return np.zeros(2)
        if vertices.ndim == 2 and vertices.shape[1] == 3:
            vertices = vertices[:, [0, 2]]
        return np.mean(vertices, axis=0)

    def update_room_adjacency_metadata(self):
        room_id_to_room = {room.room_id: room for room in self.rooms}
        for room in self.rooms:
            room.adjacent_room_names = []
            room.adjacent_room_instance_names = []
            for adjacent_room_id in room.adjacent_room_ids:
                adjacent_room = room_id_to_room.get(adjacent_room_id)
                if adjacent_room is None:
                    continue
                room.adjacent_room_names.append(adjacent_room.name)
                room.adjacent_room_instance_names.append(
                    adjacent_room.instance_name or adjacent_room.name
                )

    def build_room_adjacency(self):
        for room in self.rooms:
            room.adjacent_room_ids = []
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                int(self.cfg.room_graph.adjacency_dilation_kernel_size),
                int(self.cfg.room_graph.adjacency_dilation_kernel_size),
            ),
        )
        dilation_iters = int(self.cfg.room_graph.adjacency_dilation_iters)
        min_touch_pixels = int(self.cfg.room_graph.adjacency_min_touch_pixels)
        for floor in self.floors:
            room_masks = self.room_masks.get(floor.floor_id, [])
            if len(room_masks) != len(floor.rooms):
                continue
            for i in range(len(floor.rooms)):
                for j in range(i + 1, len(floor.rooms)):
                    mask_i = cv2.dilate((room_masks[i] > 0).astype(np.uint8), kernel, iterations=dilation_iters)
                    mask_j = cv2.dilate((room_masks[j] > 0).astype(np.uint8), kernel, iterations=dilation_iters)
                    shared_pixels = int(np.logical_and(mask_i > 0, mask_j > 0).sum())
                    if shared_pixels < min_touch_pixels:
                        continue
                    room_i = floor.rooms[i]
                    room_j = floor.rooms[j]
                    if room_j.room_id not in room_i.adjacent_room_ids:
                        room_i.adjacent_room_ids.append(room_j.room_id)
                    if room_i.room_id not in room_j.adjacent_room_ids:
                        room_j.adjacent_room_ids.append(room_i.room_id)
        self.update_room_adjacency_metadata()

    def assign_room_instance_names(self):
        rooms_by_name = {}
        for room in self.rooms:
            room_name = room.name if room.name is not None else "room"
            rooms_by_name.setdefault(room_name, []).append(room)

        for room_name, rooms in rooms_by_name.items():
            rooms.sort(
                key=lambda room: (
                    int(room.floor_id),
                    float(self._get_room_center_2d(room)[0]),
                    float(self._get_room_center_2d(room)[1]),
                    room.room_id,
                )
            )
            for idx, room in enumerate(rooms, start=1):
                room.room_type_index = idx
                room.instance_name = room_name if len(rooms) == 1 else f"{room_name}_{idx}"
        self.update_room_adjacency_metadata()

    def create_graph(self):
        self.graph = nx.Graph()
        self.graph.add_node(0, name="building", type="building")
        for floor in self.floors:
            self.graph.add_node(floor, name="floor", type="floor")
            self.graph.add_edge(0, floor)
            for room in floor.rooms:
                self.graph.add_node(room, name="room", type="room")
                self.graph.add_edge(floor, room)
                for objectt in room.objects:
                    self.graph.add_node(objectt, name=objectt.name, type="object")
                    self.graph.add_edge(room, objectt)
        room_id_to_room = {room.room_id: room for room in self.rooms}
        for room in self.rooms:
            for adjacent_room_id in room.adjacent_room_ids:
                adjacent_room = room_id_to_room.get(adjacent_room_id)
                if adjacent_room is None:
                    continue
                if not self.graph.has_edge(room, adjacent_room):
                    self.graph.add_edge(room, adjacent_room, type="adjacent")

    def save_graph(self):
        graph_dir = self.output_dir / "graph"
        floors_dir = graph_dir / "floors"
        rooms_dir = graph_dir / "rooms"
        objects_dir = graph_dir / "objects"
        floors_dir.mkdir(parents=True, exist_ok=True)
        rooms_dir.mkdir(parents=True, exist_ok=True)
        objects_dir.mkdir(parents=True, exist_ok=True)
        for out_dir in (floors_dir, rooms_dir, objects_dir):
            for pattern in ("*.json", "*.ply"):
                for old_file in out_dir.glob(pattern):
                    old_file.unlink()
        for floor in self.floors:
            floor.save(floors_dir)
        for room in self.rooms:
            room.save(rooms_dir)
        for objectt in self.objects:
            objectt.save(objects_dir)

    def run(self):
        if bool(self.cfg.pipeline.get("reuse_existing_layout", False)):
            print("reusing existing floor/room layout from saved graph...")
            self.load_existing_layout()
        else:
            if bool(self.cfg.pipeline.get("step_through_full_pcd", False)):
                print("creating full scene point cloud from RGB-D in step-through mode...")
                self.create_full_pcd_stepwise()
            else:
                print("creating full scene point cloud from RGB-D...")
                self.create_full_pcd()
            self.save_full_pcd()

            print("segmenting floors...")
            self.segment_floors()

            print("segmenting rooms...")
            for floor in self.floors:
                self.segment_rooms(floor)

        print("loading ConceptGraph objects...")
        conceptgraph_objects = load_conceptgraph_objects(
            self.cfg.conceptgraph.mapping_path,
            min_points=self.cfg.conceptgraph.min_points,
            min_detections=self.cfg.conceptgraph.min_detections,
            include_background=self.cfg.conceptgraph.include_background,
        )
        print(f"loaded {len(conceptgraph_objects)} ConceptGraph objects")

        print("assigning ConceptGraph objects to rooms...")
        self.assign_existing_objects_to_rooms(conceptgraph_objects)

        if self.cfg.pipeline.merge_objects_graph:
            for room in tqdm(self.rooms, desc="Merging room objects"):
                print("room: ", room.room_id)
                print(" number of objects before merging: ", len(room.objects))
                merge_room_objects(room)
                print(" number of objects after merging: ", len(room.objects))
            self.objects = [obj for room in self.rooms for obj in room.objects]

        if not bool(self.cfg.pipeline.get("reuse_existing_layout", False)):
            print("naming rooms...")
            name_rooms(
                self.rooms,
                method=self.cfg.room_naming.method,
                default_room_types=list(self.cfg.room_naming.default_room_types),
                fallback_method=self.cfg.room_naming.fallback_method,
                clip_model=self.clip_model,
                clip_feat_dim=self.clip_feat_dim,
            )

            if self.cfg.room_graph.connect_adjacent_rooms:
                print("building room adjacency...")
                self.build_room_adjacency()

            if self.cfg.room_graph.assign_room_instance_names:
                print("assigning room instance names...")
                self.assign_room_instance_names()

        print("creating graph...")
        self.create_graph()
        self.save_graph()

        print("# floors: ", len(self.floors))
        print("# rooms: ", len(self.rooms))
        print("# objects: ", len(self.objects))
        print("--> hybrid hierarchy scene graph successfully built")
        return self.output_dir
