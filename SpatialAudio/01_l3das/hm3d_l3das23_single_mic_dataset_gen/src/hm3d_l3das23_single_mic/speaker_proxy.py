from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .clearance import dense_probe_directions, probe_point_clearance
from .config import DatasetGenerationConfig
from .geometry import yaw_rad_to_deg
from .schemas import MicPose, SpeakerProxyPose


def resolve_humanoid_urdf_path(config: DatasetGenerationConfig) -> Optional[Path]:
    proxy_cfg = config.speaker_proxy
    if not proxy_cfg.enabled or proxy_cfg.humanoids_root is None:
        return None
    avatar_name = str(proxy_cfg.avatar_name)
    urdf_filename = proxy_cfg.urdf_filename or f"{avatar_name}.urdf"
    return proxy_cfg.humanoids_root / avatar_name / urdf_filename


def resolve_humanoid_render_asset_path(config: DatasetGenerationConfig) -> Optional[Path]:
    proxy_cfg = config.speaker_proxy
    if not proxy_cfg.enabled or proxy_cfg.humanoids_root is None:
        return None

    avatar_dir = proxy_cfg.humanoids_root / str(proxy_cfg.avatar_name)
    if proxy_cfg.render_asset_filename:
        return avatar_dir / str(proxy_cfg.render_asset_filename)

    ao_config_path = avatar_dir / f"{proxy_cfg.avatar_name}.ao_config.json"
    if ao_config_path.exists():
        try:
            payload = json.loads(ao_config_path.read_text(encoding="utf-8"))
            render_asset = payload.get("render_asset")
            if isinstance(render_asset, str) and render_asset.strip():
                return (avatar_dir / render_asset).resolve()
        except Exception:
            pass

    return avatar_dir / f"{proxy_cfg.avatar_name}.glb"


def _yaw_from_world_forward(forward: np.ndarray) -> float:
    horizontal = np.array([float(forward[0]), float(forward[2])], dtype=np.float64)
    norm = float(np.linalg.norm(horizontal))
    if norm <= 1.0e-8:
        return 0.0
    fx = horizontal[0] / norm
    fz = horizontal[1] / norm
    return math.atan2(-fx, -fz)


def build_speaker_proxy_pose(
    session: Any,
    mic_pose: MicPose,
    source_position_world: Iterable[float],
    config: DatasetGenerationConfig,
) -> Optional[SpeakerProxyPose]:
    proxy_cfg = config.speaker_proxy
    if not proxy_cfg.enabled:
        return None

    reference_world = np.asarray(list(source_position_world), dtype=np.float64)
    if reference_world.shape != (3,) or not np.isfinite(reference_world).all():
        raise ValueError("speaker proxy reference point must be a finite 3D point")

    root_world = reference_world.copy()
    floor_anchor_world = reference_world.copy()
    if proxy_cfg.render_root_from_floor_anchor:
        pathfinder = getattr(session.sim, "pathfinder", None)
        if pathfinder is None:
            raise RuntimeError("pathfinder is required to build speaker proxy pose")
        snapped = np.asarray(
            pathfinder.snap_point(reference_world.astype(np.float32)),
            dtype=np.float64,
        )
        if snapped.shape != (3,) or not np.isfinite(snapped).all():
            raise RuntimeError("speaker proxy navmesh anchor is invalid")
        floor_anchor_world = snapped
        root_world = snapped
    root_world = root_world + np.array(
        [0.0, float(proxy_cfg.render_root_height_offset_m), 0.0],
        dtype=np.float64,
    )

    yaw_rad = float(mic_pose.yaw_rad)
    if proxy_cfg.face_microphone:
        mic_position = np.asarray(mic_pose.position_world, dtype=np.float64)
        yaw_rad = _yaw_from_world_forward(mic_position - root_world)
    yaw_rad += math.radians(float(proxy_cfg.yaw_offset_deg))

    return SpeakerProxyPose(
        avatar_name=str(proxy_cfg.avatar_name),
        reference_world=reference_world.astype(float).tolist(),
        root_world=root_world.astype(float).tolist(),
        floor_anchor_world=floor_anchor_world.astype(float).tolist(),
        yaw_rad=float(yaw_rad),
        yaw_deg=float(yaw_rad_to_deg(float(yaw_rad))),
    )


def validate_speaker_proxy_pose(
    session: Any,
    speaker_proxy_pose: Optional[SpeakerProxyPose],
    config: DatasetGenerationConfig,
) -> tuple[bool, Optional[str], dict[str, float]]:
    if speaker_proxy_pose is None or not config.speaker_proxy.enabled:
        return True, None, {}

    body_base_world = np.asarray(speaker_proxy_pose.floor_anchor_world, dtype=np.float64)
    min_clearance = float("inf")
    lowest_probe_height = 0.0
    for probe_height in config.speaker_proxy.body_probe_heights_m:
        probe_point = body_base_world + np.array([0.0, float(probe_height), 0.0], dtype=np.float64)
        clearance = probe_point_clearance(
            session.sim,
            probe_point,
            probe_radius_m=float(config.speaker_proxy.body_probe_radius_m),
            ignore_hits_within_m=float(config.speaker_proxy.body_probe_ignore_hits_within_m),
            directions=dense_probe_directions(),
        )
        if clearance.min_hit_distance < min_clearance:
            min_clearance = float(clearance.min_hit_distance)
            lowest_probe_height = float(probe_height)
        if clearance.min_hit_distance < float(config.speaker_proxy.body_min_clearance_m):
            return False, "speaker_proxy_body_clearance_too_small", {
                "body_probe_height_m": float(probe_height),
                "speaker_proxy_body_clearance": float(clearance.min_hit_distance),
            }

    return True, None, {
        "speaker_proxy_body_min_clearance": float(min_clearance),
        "speaker_proxy_body_min_clearance_height_m": float(lowest_probe_height),
    }
