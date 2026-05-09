from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .camera_utils import CameraIntrinsics

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PointCloudStats:
    total_pixels: int
    sampled_pixels: int
    valid_points: int
    filtered_points: int
    kept_points: int
    point_stride: int
    max_points: int | None
    depth_clip_min: float | None
    depth_clip_max: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class PointCloudData:
    points: np.ndarray
    colors: np.ndarray
    uv: np.ndarray
    stats: PointCloudStats


def depth_to_point_cloud(
    depth_map: np.ndarray,
    rgb_image: np.ndarray,
    intrinsics: CameraIntrinsics,
    point_stride: int = 1,
    depth_clip_min: float | None = None,
    depth_clip_max: float | None = None,
) -> PointCloudData:
    if depth_map.ndim != 2:
        raise ValueError(f"Depth map must be HxW, got shape={depth_map.shape}.")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"RGB image must be HxWx3, got shape={rgb_image.shape}.")
    if depth_map.shape != rgb_image.shape[:2]:
        raise ValueError(
            f"Depth/RGB shape mismatch: depth={depth_map.shape}, rgb={rgb_image.shape[:2]}."
        )
    if point_stride <= 0:
        raise ValueError(f"point_stride must be >= 1, got {point_stride}.")

    sampled_depth = depth_map[::point_stride, ::point_stride]
    sampled_rgb = rgb_image[::point_stride, ::point_stride, :]

    v_coords, u_coords = np.meshgrid(
        np.arange(0, depth_map.shape[0], point_stride, dtype=np.float32),
        np.arange(0, depth_map.shape[1], point_stride, dtype=np.float32),
        indexing="ij",
    )

    valid_mask = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    if depth_clip_min is not None:
        valid_mask &= sampled_depth >= float(depth_clip_min)
    if depth_clip_max is not None:
        valid_mask &= sampled_depth <= float(depth_clip_max)

    z = sampled_depth[valid_mask].astype(np.float32)
    u = u_coords[valid_mask]
    v = v_coords[valid_mask]

    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = -(v - intrinsics.cy) * z / intrinsics.fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = sampled_rgb[valid_mask].reshape(-1, 3).astype(np.uint8)
    uv = np.stack([u, v], axis=1).astype(np.int32)

    sampled_pixels = int(sampled_depth.size)
    valid_points = int(points.shape[0])
    stats = PointCloudStats(
        total_pixels=int(depth_map.size),
        sampled_pixels=sampled_pixels,
        valid_points=valid_points,
        filtered_points=sampled_pixels - valid_points,
        kept_points=valid_points,
        point_stride=point_stride,
        max_points=None,
        depth_clip_min=None if depth_clip_min is None else float(depth_clip_min),
        depth_clip_max=None if depth_clip_max is None else float(depth_clip_max),
    )
    return PointCloudData(points=points, colors=colors, uv=uv, stats=stats)


def subsample_point_cloud(
    point_cloud: PointCloudData,
    max_points: int | None,
    seed: int = 0,
) -> PointCloudData:
    if max_points is None or max_points <= 0 or point_cloud.points.shape[0] <= max_points:
        return point_cloud

    rng = np.random.default_rng(seed)
    indices = rng.choice(point_cloud.points.shape[0], size=max_points, replace=False)
    indices = np.sort(indices)
    stats = replace(point_cloud.stats, kept_points=int(max_points), max_points=int(max_points))
    return PointCloudData(
        points=point_cloud.points[indices],
        colors=point_cloud.colors[indices],
        uv=point_cloud.uv[indices],
        stats=stats,
    )


def write_ply(output_path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    output_path = output_path.resolve()
    try:
        import open3d as o3d  # type: ignore

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(output_path), point_cloud)
        return output_path
    except Exception as exc:  # pragma: no cover - optional dependency path
        LOGGER.debug("Open3D save path unavailable, falling back to ASCII PLY: %s", exc)

    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {points.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        handle.write("\n")
        for xyz, rgb in zip(points, colors, strict=True):
            handle.write(
                f"{float(xyz[0]):.6f} {float(xyz[1]):.6f} {float(xyz[2]):.6f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )
    return output_path
