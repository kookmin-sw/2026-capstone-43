from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

AZIMUTH_RANGE_RAD = (-math.pi, math.pi)
ELEVATION_RANGE_RAD = (-math.pi / 2.0, math.pi / 2.0)


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    hfov_deg: float
    vfov_deg: float
    approx_from_hfov: bool

    def matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matrix"] = self.matrix().tolist()
        payload["coordinate_convention"] = coordinate_convention_dict()
        payload["angular_convention"] = angular_convention_dict()
        return payload


def coordinate_convention_dict() -> dict[str, str]:
    return {
        "x": "camera right",
        "y": "camera up",
        "z": "camera forward",
    }


def angular_convention_dict() -> dict[str, Any]:
    return {
        "azimuth_zero": "forward",
        "azimuth_positive": "right",
        "azimuth_range_rad": list(AZIMUTH_RANGE_RAD),
        "azimuth_range_deg": [-180.0, 180.0],
        "elevation_zero": "horizontal",
        "elevation_positive": "up",
        "elevation_range_rad": list(ELEVATION_RANGE_RAD),
        "elevation_range_deg": [-90.0, 90.0],
    }


def compute_intrinsics(
    width: int,
    height: int,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    hfov_deg: float = 69.0,
) -> CameraIntrinsics:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got width={width}, height={height}.")

    approx_from_hfov = fx is None
    if fx is None:
        if hfov_deg <= 0.0 or hfov_deg >= 179.0:
            raise ValueError(f"hfov_deg must be in (0, 179), got {hfov_deg}.")
        hfov_rad = math.radians(hfov_deg)
        fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    if fy is None:
        fy = fx
    if cx is None:
        cx = (width - 1) / 2.0
    if cy is None:
        cy = (height - 1) / 2.0

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"fx and fy must be positive, got fx={fx}, fy={fy}.")

    hfov_effective = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    vfov_effective = math.degrees(2.0 * math.atan(height / (2.0 * fy)))

    return CameraIntrinsics(
        width=width,
        height=height,
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        hfov_deg=float(hfov_effective),
        vfov_deg=float(vfov_effective),
        approx_from_hfov=bool(approx_from_hfov),
    )


def describe_intrinsics(intrinsics: CameraIntrinsics) -> str:
    source = "hfov approximation" if intrinsics.approx_from_hfov else "direct fx/fy/cx/cy"
    return (
        f"fx={intrinsics.fx:.3f}, fy={intrinsics.fy:.3f}, "
        f"cx={intrinsics.cx:.3f}, cy={intrinsics.cy:.3f}, "
        f"hfov={intrinsics.hfov_deg:.2f} deg, vfov={intrinsics.vfov_deg:.2f} deg "
        f"({source})"
    )


def compute_fov_angle_ranges(intrinsics: CameraIntrinsics) -> dict[str, float]:
    left_x = (0.0 - intrinsics.cx) / intrinsics.fx
    right_x = ((intrinsics.width - 1.0) - intrinsics.cx) / intrinsics.fx
    top_y = -(0.0 - intrinsics.cy) / intrinsics.fy
    bottom_y = -(((intrinsics.height - 1.0) - intrinsics.cy) / intrinsics.fy)

    az_left = math.atan2(left_x, 1.0)
    az_right = math.atan2(right_x, 1.0)
    el_top = math.atan2(top_y, 1.0)
    el_bottom = math.atan2(bottom_y, 1.0)

    az_min, az_max = sorted((az_left, az_right))
    el_min, el_max = sorted((el_bottom, el_top))
    return {
        "azimuth_min_rad": float(az_min),
        "azimuth_max_rad": float(az_max),
        "elevation_min_rad": float(el_min),
        "elevation_max_rad": float(el_max),
    }


def pixel_grid_to_camera_rays(
    width: int,
    height: int,
    intrinsics: CameraIntrinsics,
    point_stride: int = 1,
) -> np.ndarray:
    if point_stride <= 0:
        raise ValueError(f"point_stride must be >= 1, got {point_stride}.")

    v_coords, u_coords = np.meshgrid(
        np.arange(0, height, point_stride, dtype=np.float32),
        np.arange(0, width, point_stride, dtype=np.float32),
        indexing="ij",
    )
    x = (u_coords - intrinsics.cx) / intrinsics.fx
    y = -(v_coords - intrinsics.cy) / intrinsics.fy
    z = np.ones_like(x, dtype=np.float32)
    rays = np.stack([x, y, z], axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = rays / np.maximum(norms, 1.0e-8)
    return rays.astype(np.float32)
