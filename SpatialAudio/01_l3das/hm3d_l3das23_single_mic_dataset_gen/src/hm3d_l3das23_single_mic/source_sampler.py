from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np

from .clearance import dense_probe_directions, probe_point_clearance
from .config import SourceSamplingConfig
from .geometry import (
    linspace_with_step,
    local_to_world,
    local_xyz_from_cylindrical,
    point_in_aabb,
    spherical_from_local_xyz,
)
from .schemas import CylindricalCoordinate, MicPose, SourceCandidate, SphericalCoordinate


def theta_values_deg_for_radius(rho: float, source_config: SourceSamplingConfig) -> list[float]:
    mode = str(source_config.theta_sampling_mode).strip().lower()
    offset = float(source_config.theta_offset_deg)
    if mode in {"fixed_angle", "angle", "angular", "degrees"}:
        step_deg = float(source_config.theta_step_deg)
        if step_deg <= 0.0:
            raise ValueError("theta_step_deg must be positive")
        return list(np.arange(offset, 360.0 + offset - 1.0e-9, step_deg))

    if mode in {"arc_length", "distance", "l3das"}:
        arc_step_m = float(source_config.theta_arc_step_m)
        if arc_step_m <= 0.0:
            raise ValueError("theta_arc_step_m must be positive")
        circumference_m = 2.0 * math.pi * max(float(rho), 1.0e-12)
        point_count = max(1, int(round(circumference_m / arc_step_m)))
        step_deg = 360.0 / float(point_count)
        return [offset + idx * step_deg for idx in range(point_count)]

    raise ValueError(f"Unsupported theta_sampling_mode: {source_config.theta_sampling_mode!r}")


def generate_source_candidates(
    mic_pose: MicPose,
    source_config: SourceSamplingConfig,
    *,
    seed: int,
    max_sources: Optional[int] = None,
) -> list[SourceCandidate]:
    rng = np.random.default_rng(int(seed))
    rho_values = linspace_with_step(
        source_config.rho_min_m,
        source_config.rho_max_m,
        source_config.rho_step_m,
    )
    z_values = linspace_with_step(
        source_config.z_min_m,
        source_config.z_max_m,
        source_config.z_step_m,
    )

    candidates: list[SourceCandidate] = []
    for rho in rho_values:
        for theta_deg in theta_values_deg_for_radius(rho, source_config):
            theta_rad = math.radians(float(theta_deg))
            for z in z_values:
                local_xyz = local_xyz_from_cylindrical(rho, theta_rad, z)
                world_xyz = local_to_world(
                    mic_pose.position_world,
                    mic_pose.yaw_rad,
                    local_xyz,
                )
                distance, azimuth_deg, elevation_deg = spherical_from_local_xyz(local_xyz)
                candidates.append(
                    SourceCandidate(
                        source_index=len(candidates),
                        local_xyz=local_xyz.astype(float).tolist(),
                        world_xyz=world_xyz.astype(float).tolist(),
                        cylindrical=CylindricalCoordinate(
                            rho=float(rho),
                            theta_rad=float(theta_rad),
                            theta_deg=float(theta_deg),
                            z=float(z),
                        ),
                        spherical=SphericalCoordinate(
                            distance=float(distance),
                            azimuth_deg=float(azimuth_deg),
                            elevation_deg=float(elevation_deg),
                        ),
                    )
                )

    if source_config.shuffle_candidates:
        rng.shuffle(candidates)
        for new_index, candidate in enumerate(candidates):
            candidate.source_index = new_index

    cap = int(source_config.max_sources_per_mic)
    if max_sources is not None:
        cap = min(cap, int(max_sources))
    return candidates[:cap]


def validate_source_candidate(
    session: Any,
    mic_pose: MicPose,
    candidate: SourceCandidate,
    source_config: SourceSamplingConfig,
) -> Tuple[bool, Optional[str], dict[str, float]]:
    point_world = np.asarray(candidate.world_xyz, dtype=np.float64)
    aabb_min, aabb_max = session.scene_bounds

    if not np.isfinite(point_world).all():
        return False, "nonfinite_source_position", {}

    if not point_in_aabb(
        point_world,
        aabb_min,
        aabb_max,
        margin=float(source_config.scene_bounds_margin_m),
    ):
        return False, "outside_scene_bounds", {}

    source_distance = float(candidate.spherical.distance)
    if source_distance < float(source_config.min_source_distance_m):
        return False, "source_too_close_to_mic", {"source_distance": source_distance}

    if source_distance > float(source_config.max_source_distance_m):
        return False, "source_too_far_from_mic", {"source_distance": source_distance}

    navmesh_xy_offset = 0.0
    navmesh_vertical_offset = 0.0
    source_anchor_floor_delta = 0.0
    anchor_clearance = 0.0
    anchor_island_radius = 0.0
    if bool(source_config.require_navigable_projection):
        pathfinder = session.sim.pathfinder
        # The source may be elevated, but its floor anchor must still lie on a
        # navigable floor region rather than on furniture or inside clutter.
        snapped = np.asarray(
            pathfinder.snap_point(point_world.astype(np.float32)),
            dtype=np.float64,
        )
        if not np.isfinite(snapped).all():
            return False, "source_navmesh_snap_invalid", {}
        if not bool(pathfinder.is_navigable(snapped.astype(np.float32))):
            return False, "source_navmesh_snap_not_navigable", {}

        delta = point_world - snapped
        navmesh_xy_offset = float(np.linalg.norm(delta[[0, 2]]))
        navmesh_vertical_offset = float(abs(delta[1]))
        if navmesh_xy_offset > float(source_config.navmesh_projection_xy_tolerance_m):
            return (
                False,
                "source_not_on_navigable_projection",
                {"navmesh_xy_offset": navmesh_xy_offset},
            )
        if navmesh_vertical_offset > float(source_config.navmesh_projection_vertical_tolerance_m):
            return (
                False,
                "source_too_far_from_navmesh_in_height",
                {"navmesh_vertical_offset": navmesh_vertical_offset},
            )
        if bool(source_config.require_same_floor_as_mic):
            mic_floor = np.asarray(mic_pose.floor_point_world, dtype=np.float64)
            if mic_floor.shape != (3,) or not np.isfinite(mic_floor).all():
                return False, "mic_floor_point_invalid", {}
            source_anchor_floor_delta = float(abs(snapped[1] - mic_floor[1]))
            if source_anchor_floor_delta > float(source_config.same_floor_vertical_tolerance_m):
                return (
                    False,
                    "source_not_on_same_floor_as_mic",
                    {"source_anchor_floor_delta": source_anchor_floor_delta},
                )

        anchor_clearance = float(
            pathfinder.distance_to_closest_obstacle(
                snapped.astype(np.float32),
                2.0,
            )
        )
        anchor_island_radius = float(pathfinder.island_radius(snapped.astype(np.float32)))
        if (
            np.isfinite(anchor_clearance)
            and anchor_clearance < float(source_config.min_anchor_clearance_m)
        ):
            return (
                False,
                "source_anchor_clearance_too_small",
                {"anchor_clearance": anchor_clearance},
            )
        if (
            np.isfinite(anchor_island_radius)
            and anchor_island_radius < float(source_config.min_anchor_island_radius_m)
        ):
            return (
                False,
                "source_anchor_island_too_small",
                {"anchor_island_radius": anchor_island_radius},
            )

    point_clearance = probe_point_clearance(
        session.sim,
        point_world,
        probe_radius_m=float(source_config.probe_radius_m),
        ignore_hits_within_m=float(source_config.probe_ignore_hits_within_m),
    )
    if point_clearance.min_hit_distance < float(source_config.min_clearance_m):
        return (
            False,
            "source_clearance_too_small",
            {"point_clearance": point_clearance.min_hit_distance},
        )

    dense_clearance = probe_point_clearance(
        session.sim,
        point_world,
        probe_radius_m=float(source_config.dense_probe_radius_m),
        ignore_hits_within_m=float(source_config.probe_ignore_hits_within_m),
        directions=dense_probe_directions(),
    )
    if dense_clearance.min_hit_distance < float(source_config.dense_min_clearance_m):
        return (
            False,
            "source_dense_clearance_too_small",
            {"dense_point_clearance": dense_clearance.min_hit_distance},
        )
    if dense_clearance.hit_fraction >= float(source_config.dense_enclosure_hit_fraction_threshold):
        return (
            False,
            "source_likely_inside_geometry",
            {"dense_hit_fraction": dense_clearance.hit_fraction},
        )

    return True, None, {
        "source_distance": source_distance,
        "point_clearance": float(point_clearance.min_hit_distance),
        "probe_hit_rays": float(point_clearance.num_rays_with_hits),
        "dense_point_clearance": float(dense_clearance.min_hit_distance),
        "dense_probe_hit_rays": float(dense_clearance.num_rays_with_hits),
        "dense_probe_hit_fraction": float(dense_clearance.hit_fraction),
        "mic_height_m": float(mic_pose.position_world[1]),
        "navmesh_xy_offset": navmesh_xy_offset,
        "navmesh_vertical_offset": navmesh_vertical_offset,
        "source_anchor_floor_delta": source_anchor_floor_delta,
        "anchor_clearance": anchor_clearance,
        "anchor_island_radius": anchor_island_radius,
    }
