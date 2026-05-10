from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from .clearance import dense_probe_directions, probe_point_clearance
from .config import DatasetGenerationConfig
from .geometry import farthest_point_sampling, yaw_rad_to_deg, yaw_rad_to_quaternion_wxyz
from .schemas import MicPose


def _sample_yaws(num_poses: int, mode: str, fixed_yaw_deg: float, rng: np.random.Generator) -> np.ndarray:
    if num_poses <= 0:
        return np.empty((0,), dtype=np.float64)
    if mode == "fixed":
        return np.full((num_poses,), math.radians(float(fixed_yaw_deg)), dtype=np.float64)
    if mode == "random":
        return rng.uniform(0.0, 2.0 * math.pi, size=num_poses)
    start = float(rng.uniform(0.0, 2.0 * math.pi))
    return (start + (2.0 * math.pi * np.arange(num_poses) / float(num_poses))) % (2.0 * math.pi)


def sample_microphone_poses(
    session: Any,
    config: DatasetGenerationConfig,
    *,
    seed: int,
    max_poses: Optional[int] = None,
) -> tuple[list[MicPose], dict[str, int]]:
    rng = np.random.default_rng(int(seed))
    pathfinder = session.sim.pathfinder
    candidate_floor_points: list[np.ndarray] = []
    rejected_navmesh = 0
    rejected_clearance = 0
    rejected_island = 0
    rejected_point_clearance = 0
    rejected_point_clearance_unknown = 0
    rejected_dense_point_clearance = 0
    rejected_dense_enclosure = 0

    target_poses = int(config.mic_sampling.poses_per_scene)
    if max_poses is not None:
        target_poses = min(target_poses, int(max_poses))
    total_trials = max(int(config.mic_sampling.candidate_pool_size), target_poses * 20)

    for _ in range(total_trials):
        point = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float64)
        if bool(config.mic_sampling.require_navigable_snap_confirmation):
            # Mic floor points come from navmesh sampling, but we still re-check
            # snap consistency explicitly to keep the filtering auditable.
            snapped = np.asarray(pathfinder.snap_point(point.astype(np.float32)), dtype=np.float64)
            if not np.isfinite(snapped).all():
                rejected_navmesh += 1
                continue
            if not bool(pathfinder.is_navigable(snapped.astype(np.float32))):
                rejected_navmesh += 1
                continue
            delta = point - snapped
            snap_xy_offset = float(np.linalg.norm(delta[[0, 2]]))
            snap_vertical_offset = float(abs(delta[1]))
            if snap_xy_offset > float(config.mic_sampling.navmesh_snap_xy_tolerance_m):
                rejected_navmesh += 1
                continue
            if snap_vertical_offset > float(config.mic_sampling.navmesh_snap_vertical_tolerance_m):
                rejected_navmesh += 1
                continue

        clearance = float(
            pathfinder.distance_to_closest_obstacle(
                point.astype(np.float32),
                2.0,
            )
        )
        island_radius = float(pathfinder.island_radius(point.astype(np.float32)))
        if np.isfinite(clearance) and clearance < float(config.mic_sampling.min_clearance_m):
            rejected_clearance += 1
            continue
        if np.isfinite(island_radius) and island_radius < float(config.mic_sampling.min_island_radius_m):
            rejected_island += 1
            continue

        mic_position = point + np.array(
            [0.0, float(config.sensor_rig.mic_height_m), 0.0],
            dtype=np.float64,
        )
        point_clearance = probe_point_clearance(
            session.sim,
            mic_position,
            probe_radius_m=float(config.mic_sampling.probe_radius_m),
            ignore_hits_within_m=float(config.mic_sampling.probe_ignore_hits_within_m),
        )
        if point_clearance.min_hit_distance < float(config.mic_sampling.min_point_clearance_m):
            rejected_point_clearance += 1
            continue

        dense_clearance = probe_point_clearance(
            session.sim,
            mic_position,
            probe_radius_m=float(config.mic_sampling.dense_probe_radius_m),
            ignore_hits_within_m=float(config.mic_sampling.probe_ignore_hits_within_m),
            directions=dense_probe_directions(),
        )
        if dense_clearance.min_hit_distance < float(config.mic_sampling.dense_min_point_clearance_m):
            rejected_dense_point_clearance += 1
            continue
        if dense_clearance.hit_fraction >= float(config.mic_sampling.dense_enclosure_hit_fraction_threshold):
            rejected_dense_enclosure += 1
            continue

        candidate_floor_points.append(point)

    if not candidate_floor_points:
        return [], {
            "pool": 0,
            "rejected_navmesh": rejected_navmesh,
            "rejected_clearance": rejected_clearance,
            "rejected_island": rejected_island,
            "rejected_point_clearance": rejected_point_clearance,
            "rejected_point_clearance_unknown": rejected_point_clearance_unknown,
            "rejected_dense_point_clearance": rejected_dense_point_clearance,
            "rejected_dense_enclosure": rejected_dense_enclosure,
        }

    selected_floor_points = farthest_point_sampling(
        np.stack(candidate_floor_points, axis=0),
        n_target=target_poses,
        rng=rng,
    )
    yaws = _sample_yaws(
        len(selected_floor_points),
        config.mic_sampling.yaw_mode,
        config.mic_sampling.fixed_yaw_deg,
        rng,
    )

    mic_poses: list[MicPose] = []
    for mic_index, (floor_point, yaw_rad) in enumerate(zip(selected_floor_points, yaws)):
        mic_position = floor_point + np.array([0.0, float(config.sensor_rig.mic_height_m), 0.0], dtype=np.float64)
        mic_poses.append(
            MicPose(
                mic_index=mic_index,
                floor_point_world=floor_point.astype(float).tolist(),
                position_world=mic_position.astype(float).tolist(),
                quaternion_wxyz=yaw_rad_to_quaternion_wxyz(float(yaw_rad)),
                yaw_rad=float(yaw_rad),
                yaw_deg=float(yaw_rad_to_deg(float(yaw_rad))),
            )
        )

    return mic_poses, {
        "pool": len(candidate_floor_points),
        "rejected_navmesh": rejected_navmesh,
        "rejected_clearance": rejected_clearance,
        "rejected_island": rejected_island,
        "rejected_point_clearance": rejected_point_clearance,
        "rejected_point_clearance_unknown": rejected_point_clearance_unknown,
        "rejected_dense_point_clearance": rejected_dense_point_clearance,
        "rejected_dense_enclosure": rejected_dense_enclosure,
    }
