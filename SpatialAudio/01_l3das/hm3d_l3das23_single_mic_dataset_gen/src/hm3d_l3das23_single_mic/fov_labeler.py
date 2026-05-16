from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geometry import camera_xyz_from_local, world_to_local
from .schemas import ProjectionResult


@dataclass
class CameraModel:
    width: int
    height: int
    hfov_deg: float
    fx: float
    fy: float
    cx: float
    cy: float
    tan_half_hfov: float
    tan_half_vfov: float

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "CameraModel":
        half_h = math.tan(math.radians(float(hfov_deg)) * 0.5)
        fx = float(width) / (2.0 * half_h)
        fy = fx
        half_v = half_h * (float(height) / float(width))
        return cls(
            width=int(width),
            height=int(height),
            hfov_deg=float(hfov_deg),
            fx=fx,
            fy=fy,
            cx=(float(width) - 1.0) * 0.5,
            cy=(float(height) - 1.0) * 0.5,
            tan_half_hfov=half_h,
            tan_half_vfov=half_v,
        )


def compute_in_fov(
    camera_model: CameraModel,
    mic_position_world: Iterable[float],
    mic_yaw_rad: float,
    source_position_world: Iterable[float],
    *,
    depth_eps: float = 1.0e-6,
) -> ProjectionResult:
    local_xyz = world_to_local(mic_position_world, mic_yaw_rad, source_position_world)
    camera_xyz = camera_xyz_from_local(local_xyz)

    x_cam = float(camera_xyz[0])
    y_cam = float(camera_xyz[1])
    z_cam = float(camera_xyz[2])

    if z_cam <= float(depth_eps):
        return ProjectionResult(
            in_fov=False,
            pixel_xy=None,
            depth_cam=z_cam,
            normalized_xy=None,
            reason="behind_camera",
        )

    x_over_z = x_cam / z_cam
    y_over_z = y_cam / z_cam

    pixel_x = camera_model.fx * x_over_z + camera_model.cx
    pixel_y = camera_model.cy - (camera_model.fy * y_over_z)

    in_horizontal = abs(x_over_z) <= camera_model.tan_half_hfov
    in_vertical = abs(y_over_z) <= camera_model.tan_half_vfov

    if in_horizontal and in_vertical:
        return ProjectionResult(
            in_fov=True,
            pixel_xy=[float(pixel_x), float(pixel_y)],
            depth_cam=z_cam,
            normalized_xy=[float(x_over_z), float(y_over_z)],
            reason=None,
        )

    if not in_horizontal and x_over_z < -camera_model.tan_half_hfov:
        reason = "left_of_frame"
    elif not in_horizontal and x_over_z > camera_model.tan_half_hfov:
        reason = "right_of_frame"
    elif not in_vertical and y_over_z > camera_model.tan_half_vfov:
        reason = "above_frame"
    else:
        reason = "below_frame"

    return ProjectionResult(
        in_fov=False,
        pixel_xy=None,
        depth_cam=z_cam,
        normalized_xy=[float(x_over_z), float(y_over_z)],
        reason=reason,
    )
