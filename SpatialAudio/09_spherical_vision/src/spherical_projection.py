from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .camera_utils import (
    AZIMUTH_RANGE_RAD,
    ELEVATION_RANGE_RAD,
    CameraIntrinsics,
    angular_convention_dict,
    compute_fov_angle_ranges,
    coordinate_convention_dict,
    pixel_grid_to_camera_rays,
)

REQUIRED_CHANNEL_NAMES = [
    "observed_mask",
    "fov_mask",
    "has_points",
    "occupancy",
    "density",
    "min_depth",
    "p10_depth",
    "mean_depth",
    "median_depth",
    "depth_std",
    "valid_ratio",
]
OPTIONAL_CHANNEL_NAMES = [
    "inverse_mean_depth",
    "inverse_p10_depth",
    "log_mean_depth",
]
DEPTH_LIKE_CHANNELS = {
    "min_depth",
    "p10_depth",
    "mean_depth",
    "median_depth",
    "depth_std",
    "inverse_mean_depth",
    "inverse_p10_depth",
    "log_mean_depth",
}
CHANNEL_DESCRIPTIONS: dict[str, str] = {
    "observed_mask": "1 if at least one sampled camera ray falls in the bin.",
    "fov_mask": "1 if the bin center lies inside the analytic camera FOV.",
    "has_points": "1 if at least one valid 3D point falls in the bin.",
    "occupancy": "Valid point count normalized by the maximum valid point count across bins.",
    "density": "Valid point count divided by the total valid point count for the sample.",
    "min_depth": "Minimum camera-centered range depth among valid points in the bin.",
    "p10_depth": "10th percentile camera-centered range depth among valid points in the bin.",
    "mean_depth": "Mean camera-centered range depth among valid points in the bin.",
    "median_depth": "Median camera-centered range depth among valid points in the bin.",
    "depth_std": "Standard deviation of camera-centered range depth among valid points in the bin.",
    "valid_ratio": "Valid point count divided by the total projected raw sample count in the bin.",
    "inverse_mean_depth": "Inverse of mean_depth for valid bins, zero elsewhere.",
    "inverse_p10_depth": "Inverse of p10_depth for valid bins, zero elsewhere.",
    "log_mean_depth": "Natural log of mean_depth for valid bins, zero elsewhere.",
}
DEPTH_FILL_VALUE = 0.0
EPSILON = 1.0e-6


@dataclass(frozen=True)
class AngularGrid:
    num_az_bins: int
    num_el_bins: int
    azimuth_edges: np.ndarray
    azimuth_centers: np.ndarray
    elevation_edges: np.ndarray
    elevation_centers: np.ndarray

    def get_bin_centers(self) -> dict[str, np.ndarray]:
        azimuth_grid, elevation_grid = np.meshgrid(
            self.azimuth_centers,
            self.elevation_centers,
            indexing="xy",
        )
        return {
            "azimuth_rad": azimuth_grid.astype(np.float32),
            "azimuth_deg": np.degrees(azimuth_grid).astype(np.float32),
            "elevation_rad": elevation_grid.astype(np.float32),
            "elevation_deg": np.degrees(elevation_grid).astype(np.float32),
        }

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "num_az_bins": int(self.num_az_bins),
            "num_el_bins": int(self.num_el_bins),
            "azimuth_edges_rad": self.azimuth_edges.tolist(),
            "azimuth_edges_deg": np.degrees(self.azimuth_edges).tolist(),
            "azimuth_centers_rad": self.azimuth_centers.tolist(),
            "azimuth_centers_deg": np.degrees(self.azimuth_centers).tolist(),
            "elevation_edges_rad": self.elevation_edges.tolist(),
            "elevation_edges_deg": np.degrees(self.elevation_edges).tolist(),
            "elevation_centers_rad": self.elevation_centers.tolist(),
            "elevation_centers_deg": np.degrees(self.elevation_centers).tolist(),
        }


@dataclass(frozen=True)
class FeatureBundle:
    mode: str
    channel_names: list[str]
    channels: dict[str, np.ndarray]
    raw_sample_count: np.ndarray
    valid_count: np.ndarray
    total_projected_samples: int
    total_valid_points: int
    azimuth_edges: np.ndarray
    azimuth_centers: np.ndarray
    elevation_edges: np.ndarray | None
    elevation_centers: np.ndarray | None
    depth_processing: dict[str, Any]

    def tensor(self) -> np.ndarray:
        return np.stack([self.channels[name] for name in self.channel_names], axis=-1).astype(np.float32)

    def summary_dict(self) -> dict[str, Any]:
        observed_mask = self.channels["observed_mask"] > 0.5
        has_points = self.channels["has_points"] > 0.5
        observed_bin_count = int(np.count_nonzero(observed_mask))
        occupied_bin_count = int(np.count_nonzero(has_points))
        empty_observed_bin_count = int(np.count_nonzero(observed_mask & ~has_points))

        summary: dict[str, Any] = {
            "mode": self.mode,
            "total_projected_samples": int(self.total_projected_samples),
            "total_valid_points": int(self.total_valid_points),
            "observed_bin_count": observed_bin_count,
            "occupied_bin_count": occupied_bin_count,
            "empty_but_observed_bin_count": empty_observed_bin_count,
            "num_az_bins": int(self.azimuth_centers.shape[0]),
            "num_el_bins": None if self.elevation_centers is None else int(self.elevation_centers.shape[0]),
        }

        if occupied_bin_count == 0:
            summary["most_occupied_bin"] = None
            summary["nearest_bin_by_p10_depth"] = None
            return summary

        occupancy = self.channels["occupancy"]
        p10_depth = self.channels["p10_depth"]
        most_occupied_index = np.unravel_index(np.argmax(occupancy), occupancy.shape)
        robust_nearest = np.where(has_points, p10_depth, np.inf)
        nearest_index = np.unravel_index(np.argmin(robust_nearest), robust_nearest.shape)

        summary["most_occupied_bin"] = describe_bin(self, most_occupied_index)
        summary["nearest_bin_by_p10_depth"] = describe_bin(self, nearest_index)
        return summary


def build_angular_grid(num_az_bins: int, num_el_bins: int) -> AngularGrid:
    if num_az_bins <= 0:
        raise ValueError(f"num_az_bins must be positive, got {num_az_bins}.")
    if num_el_bins <= 0:
        raise ValueError(f"num_el_bins must be positive, got {num_el_bins}.")

    azimuth_edges = np.linspace(AZIMUTH_RANGE_RAD[0], AZIMUTH_RANGE_RAD[1], num_az_bins + 1, dtype=np.float32)
    elevation_edges = np.linspace(ELEVATION_RANGE_RAD[0], ELEVATION_RANGE_RAD[1], num_el_bins + 1, dtype=np.float32)
    azimuth_centers = ((azimuth_edges[:-1] + azimuth_edges[1:]) * 0.5).astype(np.float32)
    elevation_centers = ((elevation_edges[:-1] + elevation_edges[1:]) * 0.5).astype(np.float32)
    return AngularGrid(
        num_az_bins=int(num_az_bins),
        num_el_bins=int(num_el_bins),
        azimuth_edges=azimuth_edges,
        azimuth_centers=azimuth_centers,
        elevation_edges=elevation_edges,
        elevation_centers=elevation_centers,
    )


def get_bin_centers(num_az_bins: int, num_el_bins: int) -> dict[str, np.ndarray]:
    return build_angular_grid(num_az_bins, num_el_bins).get_bin_centers()


def angle_to_bin(
    azimuth_rad: np.ndarray | float,
    elevation_rad: np.ndarray | float,
    grid: AngularGrid,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth = np.asarray(azimuth_rad, dtype=np.float32)
    elevation = np.asarray(elevation_rad, dtype=np.float32)
    azimuth_idx = np.clip(np.digitize(azimuth, grid.azimuth_edges) - 1, 0, grid.num_az_bins - 1).astype(np.int32)
    elevation_idx = np.clip(np.digitize(elevation, grid.elevation_edges) - 1, 0, grid.num_el_bins - 1).astype(np.int32)
    return azimuth_idx, elevation_idx


def bin_to_angle(
    azimuth_idx: int | np.ndarray,
    elevation_idx: int | np.ndarray,
    grid: AngularGrid,
    output_in_degrees: bool = False,
) -> dict[str, np.ndarray]:
    azimuth_indices = np.asarray(azimuth_idx, dtype=np.int32)
    elevation_indices = np.asarray(elevation_idx, dtype=np.int32)
    azimuth_indices = np.clip(azimuth_indices, 0, grid.num_az_bins - 1)
    elevation_indices = np.clip(elevation_indices, 0, grid.num_el_bins - 1)
    azimuth_values = grid.azimuth_centers[azimuth_indices]
    elevation_values = grid.elevation_centers[elevation_indices]
    if output_in_degrees:
        azimuth_values = np.degrees(azimuth_values)
        elevation_values = np.degrees(elevation_values)
    return {
        "azimuth": np.asarray(azimuth_values),
        "elevation": np.asarray(elevation_values),
    }


def xyz_to_spherical(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be Nx3, got shape={points.shape}.")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ranges = np.linalg.norm(points, axis=1).astype(np.float32)
    azimuth = np.arctan2(x, z).astype(np.float32)
    horizontal_radius = np.sqrt(np.maximum(x * x + z * z, EPSILON)).astype(np.float32)
    elevation = np.arctan2(y, horizontal_radius).astype(np.float32)
    return azimuth, elevation, ranges


def _compute_fov_mask(grid: AngularGrid, intrinsics: CameraIntrinsics) -> np.ndarray:
    fov_ranges = compute_fov_angle_ranges(intrinsics)
    azimuth_visible = (grid.azimuth_centers >= fov_ranges["azimuth_min_rad"]) & (
        grid.azimuth_centers <= fov_ranges["azimuth_max_rad"]
    )
    elevation_visible = (grid.elevation_centers >= fov_ranges["elevation_min_rad"]) & (
        grid.elevation_centers <= fov_ranges["elevation_max_rad"]
    )
    return (elevation_visible[:, None] & azimuth_visible[None, :]).astype(np.float32)


def _compute_azimuth_fov_mask(grid: AngularGrid, intrinsics: CameraIntrinsics) -> np.ndarray:
    fov_ranges = compute_fov_angle_ranges(intrinsics)
    mask = (grid.azimuth_centers >= fov_ranges["azimuth_min_rad"]) & (
        grid.azimuth_centers <= fov_ranges["azimuth_max_rad"]
    )
    return mask.astype(np.float32)


def _groupwise_depth_statistics(
    valid_flat_idx: np.ndarray,
    ranges: np.ndarray,
    flat_size: int,
) -> dict[str, np.ndarray]:
    min_depth = np.full(flat_size, DEPTH_FILL_VALUE, dtype=np.float32)
    p10_depth = np.full(flat_size, DEPTH_FILL_VALUE, dtype=np.float32)
    mean_depth = np.full(flat_size, DEPTH_FILL_VALUE, dtype=np.float32)
    median_depth = np.full(flat_size, DEPTH_FILL_VALUE, dtype=np.float32)
    depth_std = np.full(flat_size, DEPTH_FILL_VALUE, dtype=np.float32)

    if valid_flat_idx.size == 0:
        return {
            "min_depth": min_depth,
            "p10_depth": p10_depth,
            "mean_depth": mean_depth,
            "median_depth": median_depth,
            "depth_std": depth_std,
        }

    order = np.argsort(valid_flat_idx, kind="mergesort")
    sorted_bins = valid_flat_idx[order]
    sorted_ranges = ranges[order]
    unique_bins, start_indices, counts = np.unique(sorted_bins, return_index=True, return_counts=True)

    for flat_index, start_index, count in zip(unique_bins, start_indices, counts, strict=True):
        values = sorted_ranges[start_index : start_index + count]
        min_depth[flat_index] = float(np.min(values))
        p10_depth[flat_index] = float(np.percentile(values, 10.0))
        mean_depth[flat_index] = float(np.mean(values))
        median_depth[flat_index] = float(np.median(values))
        depth_std[flat_index] = float(np.std(values))

    return {
        "min_depth": min_depth,
        "p10_depth": p10_depth,
        "mean_depth": mean_depth,
        "median_depth": median_depth,
        "depth_std": depth_std,
    }


def _build_channels_from_statistics(
    raw_flat_idx: np.ndarray,
    valid_flat_idx: np.ndarray,
    ranges: np.ndarray,
    output_shape: tuple[int, ...],
    fov_mask: np.ndarray,
    include_extra_depth_channels: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    flat_size = int(np.prod(output_shape))
    raw_count = np.bincount(raw_flat_idx, minlength=flat_size).astype(np.int32).reshape(output_shape)
    valid_count = np.bincount(valid_flat_idx, minlength=flat_size).astype(np.int32).reshape(output_shape)

    stats_flat = _groupwise_depth_statistics(valid_flat_idx, ranges, flat_size)
    min_depth = stats_flat["min_depth"].reshape(output_shape)
    p10_depth = stats_flat["p10_depth"].reshape(output_shape)
    mean_depth = stats_flat["mean_depth"].reshape(output_shape)
    median_depth = stats_flat["median_depth"].reshape(output_shape)
    depth_std = stats_flat["depth_std"].reshape(output_shape)

    observed_mask = (raw_count > 0).astype(np.float32)
    has_points = (valid_count > 0).astype(np.float32)

    occupancy = np.zeros(output_shape, dtype=np.float32)
    max_valid_count = int(np.max(valid_count)) if valid_count.size > 0 else 0
    if max_valid_count > 0:
        occupancy = valid_count.astype(np.float32) / float(max_valid_count)

    density = np.zeros(output_shape, dtype=np.float32)
    total_valid_points = int(np.sum(valid_count))
    if total_valid_points > 0:
        density = valid_count.astype(np.float32) / float(total_valid_points)

    valid_ratio = np.zeros(output_shape, dtype=np.float32)
    valid_ratio = np.divide(
        valid_count.astype(np.float32),
        np.maximum(raw_count.astype(np.float32), 1.0),
        out=valid_ratio,
        where=raw_count > 0,
    )

    channels: dict[str, np.ndarray] = {
        "observed_mask": observed_mask.astype(np.float32),
        "fov_mask": fov_mask.astype(np.float32),
        "has_points": has_points.astype(np.float32),
        "occupancy": occupancy.astype(np.float32),
        "density": density.astype(np.float32),
        "min_depth": min_depth.astype(np.float32),
        "p10_depth": p10_depth.astype(np.float32),
        "mean_depth": mean_depth.astype(np.float32),
        "median_depth": median_depth.astype(np.float32),
        "depth_std": depth_std.astype(np.float32),
        "valid_ratio": valid_ratio.astype(np.float32),
    }

    if include_extra_depth_channels:
        inverse_mean_depth = np.zeros(output_shape, dtype=np.float32)
        inverse_p10_depth = np.zeros(output_shape, dtype=np.float32)
        log_mean_depth = np.zeros(output_shape, dtype=np.float32)
        has_points_mask = has_points > 0.5
        inverse_mean_depth[has_points_mask] = 1.0 / np.maximum(mean_depth[has_points_mask], EPSILON)
        inverse_p10_depth[has_points_mask] = 1.0 / np.maximum(p10_depth[has_points_mask], EPSILON)
        log_mean_depth[has_points_mask] = np.log(np.maximum(mean_depth[has_points_mask], EPSILON))
        channels["inverse_mean_depth"] = inverse_mean_depth
        channels["inverse_p10_depth"] = inverse_p10_depth
        channels["log_mean_depth"] = log_mean_depth

    return channels, raw_count, valid_count


def _bundle_channel_names(include_extra_depth_channels: bool) -> list[str]:
    channel_names = list(REQUIRED_CHANNEL_NAMES)
    if include_extra_depth_channels:
        channel_names.extend(OPTIONAL_CHANNEL_NAMES)
    return channel_names


def build_vision_feature_bundles(
    points: np.ndarray,
    image_width: int,
    image_height: int,
    intrinsics: CameraIntrinsics,
    num_az_bins: int,
    num_el_bins: int,
    point_stride: int,
    include_extra_depth_channels: bool,
    depth_processing: dict[str, Any],
) -> tuple[FeatureBundle, FeatureBundle]:
    grid = build_angular_grid(num_az_bins=num_az_bins, num_el_bins=num_el_bins)
    channel_names = _bundle_channel_names(include_extra_depth_channels)

    sampled_rays = pixel_grid_to_camera_rays(
        width=image_width,
        height=image_height,
        intrinsics=intrinsics,
        point_stride=point_stride,
    ).reshape(-1, 3)
    raw_azimuth, raw_elevation, _ = xyz_to_spherical(sampled_rays)
    raw_azimuth_idx, raw_elevation_idx = angle_to_bin(raw_azimuth, raw_elevation, grid)
    raw_flat_idx = (raw_elevation_idx * grid.num_az_bins + raw_azimuth_idx).astype(np.int32)

    point_azimuth, point_elevation, point_ranges = xyz_to_spherical(points)
    valid_azimuth_idx, valid_elevation_idx = angle_to_bin(point_azimuth, point_elevation, grid)
    valid_flat_idx = (valid_elevation_idx * grid.num_az_bins + valid_azimuth_idx).astype(np.int32)

    fov_mask = _compute_fov_mask(grid, intrinsics)
    channels_2d, raw_count_2d, valid_count_2d = _build_channels_from_statistics(
        raw_flat_idx=raw_flat_idx,
        valid_flat_idx=valid_flat_idx,
        ranges=point_ranges.astype(np.float32),
        output_shape=(grid.num_el_bins, grid.num_az_bins),
        fov_mask=fov_mask,
        include_extra_depth_channels=include_extra_depth_channels,
    )
    full_bundle = FeatureBundle(
        mode="full_spherical_grid",
        channel_names=channel_names,
        channels=channels_2d,
        raw_sample_count=raw_count_2d,
        valid_count=valid_count_2d,
        total_projected_samples=int(raw_flat_idx.size),
        total_valid_points=int(points.shape[0]),
        azimuth_edges=grid.azimuth_edges,
        azimuth_centers=grid.azimuth_centers,
        elevation_edges=grid.elevation_edges,
        elevation_centers=grid.elevation_centers,
        depth_processing=depth_processing,
    )

    raw_azimuth_only = np.clip(np.digitize(raw_azimuth, grid.azimuth_edges) - 1, 0, grid.num_az_bins - 1).astype(np.int32)
    valid_azimuth_only = np.clip(np.digitize(point_azimuth, grid.azimuth_edges) - 1, 0, grid.num_az_bins - 1).astype(np.int32)
    fov_mask_azimuth = _compute_azimuth_fov_mask(grid, intrinsics)
    channels_azimuth, raw_count_azimuth, valid_count_azimuth = _build_channels_from_statistics(
        raw_flat_idx=raw_azimuth_only,
        valid_flat_idx=valid_azimuth_only,
        ranges=point_ranges.astype(np.float32),
        output_shape=(grid.num_az_bins,),
        fov_mask=fov_mask_azimuth,
        include_extra_depth_channels=include_extra_depth_channels,
    )
    azimuth_bundle = FeatureBundle(
        mode="azimuth_aggregated",
        channel_names=channel_names,
        channels=channels_azimuth,
        raw_sample_count=raw_count_azimuth,
        valid_count=valid_count_azimuth,
        total_projected_samples=int(raw_azimuth_only.size),
        total_valid_points=int(points.shape[0]),
        azimuth_edges=grid.azimuth_edges,
        azimuth_centers=grid.azimuth_centers,
        elevation_edges=None,
        elevation_centers=None,
        depth_processing=depth_processing,
    )
    return full_bundle, azimuth_bundle


def describe_bin(bundle: FeatureBundle, index: tuple[int, ...]) -> dict[str, Any]:
    if bundle.valid_count.ndim == 1:
        az_idx = int(index[0])
        return {
            "index": [az_idx],
            "azimuth_center_deg": float(np.degrees(bundle.azimuth_centers[az_idx])),
            "azimuth_range_deg": [
                float(np.degrees(bundle.azimuth_edges[az_idx])),
                float(np.degrees(bundle.azimuth_edges[az_idx + 1])),
            ],
            "observed_mask": float(bundle.channels["observed_mask"][az_idx]),
            "has_points": float(bundle.channels["has_points"][az_idx]),
            "occupancy": float(bundle.channels["occupancy"][az_idx]),
            "p10_depth": float(bundle.channels["p10_depth"][az_idx]),
            "mean_depth": float(bundle.channels["mean_depth"][az_idx]),
            "valid_ratio": float(bundle.channels["valid_ratio"][az_idx]),
            "valid_count": int(bundle.valid_count[az_idx]),
        }

    el_idx = int(index[0])
    az_idx = int(index[1])
    assert bundle.elevation_centers is not None
    assert bundle.elevation_edges is not None
    return {
        "index": [el_idx, az_idx],
        "azimuth_center_deg": float(np.degrees(bundle.azimuth_centers[az_idx])),
        "azimuth_range_deg": [
            float(np.degrees(bundle.azimuth_edges[az_idx])),
            float(np.degrees(bundle.azimuth_edges[az_idx + 1])),
        ],
        "elevation_center_deg": float(np.degrees(bundle.elevation_centers[el_idx])),
        "elevation_range_deg": [
            float(np.degrees(bundle.elevation_edges[el_idx])),
            float(np.degrees(bundle.elevation_edges[el_idx + 1])),
        ],
        "observed_mask": float(bundle.channels["observed_mask"][el_idx, az_idx]),
        "has_points": float(bundle.channels["has_points"][el_idx, az_idx]),
        "occupancy": float(bundle.channels["occupancy"][el_idx, az_idx]),
        "p10_depth": float(bundle.channels["p10_depth"][el_idx, az_idx]),
        "mean_depth": float(bundle.channels["mean_depth"][el_idx, az_idx]),
        "valid_ratio": float(bundle.channels["valid_ratio"][el_idx, az_idx]),
        "valid_count": int(bundle.valid_count[el_idx, az_idx]),
    }


def build_vision_sphere_meta(
    full_bundle: FeatureBundle,
    azimuth_bundle: FeatureBundle,
    intrinsics: CameraIntrinsics,
) -> dict[str, Any]:
    grid = build_angular_grid(
        num_az_bins=full_bundle.azimuth_centers.shape[0],
        num_el_bins=1 if full_bundle.elevation_centers is None else full_bundle.elevation_centers.shape[0],
    )
    _ = grid
    return {
        "tensor_shape": list(full_bundle.tensor().shape),
        "azimuth_tensor_shape": list(azimuth_bundle.tensor().shape),
        "channel_names": list(full_bundle.channel_names),
        "channel_descriptions": {name: CHANNEL_DESCRIPTIONS[name] for name in full_bundle.channel_names},
        "coordinate_convention": coordinate_convention_dict(),
        "angular_convention": angular_convention_dict(),
        "binning": {
            "num_az_bins": int(full_bundle.azimuth_centers.shape[0]),
            "num_el_bins": 1 if full_bundle.elevation_centers is None else int(full_bundle.elevation_centers.shape[0]),
            "azimuth_edges_rad": full_bundle.azimuth_edges.tolist(),
            "azimuth_edges_deg": np.degrees(full_bundle.azimuth_edges).tolist(),
            "azimuth_centers_rad": full_bundle.azimuth_centers.tolist(),
            "azimuth_centers_deg": np.degrees(full_bundle.azimuth_centers).tolist(),
            "elevation_edges_rad": None if full_bundle.elevation_edges is None else full_bundle.elevation_edges.tolist(),
            "elevation_edges_deg": None
            if full_bundle.elevation_edges is None
            else np.degrees(full_bundle.elevation_edges).tolist(),
            "elevation_centers_rad": None
            if full_bundle.elevation_centers is None
            else full_bundle.elevation_centers.tolist(),
            "elevation_centers_deg": None
            if full_bundle.elevation_centers is None
            else np.degrees(full_bundle.elevation_centers).tolist(),
        },
        "definitions": {
            "unknown": "observed_mask == 0",
            "observed_but_empty": "observed_mask == 1 and has_points == 0",
            "occupied": "has_points == 1",
            "depth_stat_reference": "camera-centered Euclidean range depth",
            "depth_fill_value": float(DEPTH_FILL_VALUE),
        },
        "depth_processing": full_bundle.depth_processing,
        "intrinsics": intrinsics.to_dict(),
    }
