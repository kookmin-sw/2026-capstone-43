from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .camera_utils import CameraIntrinsics
from .pointcloud_utils import PointCloudData
from .spherical_projection import DEPTH_LIKE_CHANNELS, FeatureBundle


def compute_depth_map_stats(depth_map: np.ndarray) -> dict[str, float]:
    valid_mask = np.isfinite(depth_map) & (depth_map > 0.0)
    invalid_ratio = 1.0 - float(valid_mask.mean())
    if not np.any(valid_mask):
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "valid_ratio": 0.0,
            "invalid_ratio": invalid_ratio,
        }

    valid_values = depth_map[valid_mask]
    return {
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
        "mean": float(np.mean(valid_values)),
        "p10": float(np.percentile(valid_values, 10.0)),
        "p50": float(np.percentile(valid_values, 50.0)),
        "p90": float(np.percentile(valid_values, 90.0)),
        "valid_ratio": float(valid_mask.mean()),
        "invalid_ratio": invalid_ratio,
    }


def resolve_depth_clip_bounds(
    depth_map: np.ndarray,
    depth_clip_min: float | None,
    depth_clip_max: float | None,
    depth_clip_percentile_low: float | None,
    depth_clip_percentile_high: float | None,
) -> tuple[float | None, float | None, dict[str, Any]]:
    valid_values = depth_map[np.isfinite(depth_map) & (depth_map > 0.0)]
    resolved_min = None if depth_clip_min is None else float(depth_clip_min)
    resolved_max = None if depth_clip_max is None else float(depth_clip_max)

    if valid_values.size > 0 and depth_clip_percentile_low is not None:
        percentile_min = float(np.percentile(valid_values, depth_clip_percentile_low))
        resolved_min = percentile_min if resolved_min is None else max(resolved_min, percentile_min)
    if valid_values.size > 0 and depth_clip_percentile_high is not None:
        percentile_max = float(np.percentile(valid_values, depth_clip_percentile_high))
        resolved_max = percentile_max if resolved_max is None else min(resolved_max, percentile_max)

    if resolved_min is not None and resolved_max is not None and resolved_min > resolved_max:
        resolved_min, resolved_max = resolved_max, resolved_min

    meta = {
        "depth_clip_min_requested": depth_clip_min,
        "depth_clip_max_requested": depth_clip_max,
        "depth_clip_percentile_low": depth_clip_percentile_low,
        "depth_clip_percentile_high": depth_clip_percentile_high,
        "depth_clip_min_resolved": resolved_min,
        "depth_clip_max_resolved": resolved_max,
    }
    return resolved_min, resolved_max, meta


def summarize_pooled_tensor(
    pooled_tensor: np.ndarray,
    channel_names: list[str],
    labels: list[str],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for label_index, label in enumerate(labels):
        summary[label] = {
            channel_name: float(pooled_tensor[label_index, channel_index])
            for channel_index, channel_name in enumerate(channel_names)
        }
    return summary


def compute_channel_aggregate_payload(
    bundle: FeatureBundle,
) -> dict[str, dict[str, float | int | str]]:
    payload: dict[str, dict[str, float | int | str]] = {}
    has_points_mask = bundle.channels["has_points"] > 0.5

    for channel_name in bundle.channel_names:
        values = bundle.channels[channel_name].astype(np.float32)
        if channel_name in DEPTH_LIKE_CHANNELS:
            reduction_mask = has_points_mask
            reduction_name = "has_points"
        else:
            reduction_mask = np.ones_like(values, dtype=bool)
            reduction_name = "all_bins"

        selected = values[reduction_mask]
        if selected.size == 0:
            payload[channel_name] = {
                "reduction_mask": reduction_name,
                "count": 0,
                "sum": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            continue

        payload[channel_name] = {
            "reduction_mask": reduction_name,
            "count": int(selected.size),
            "sum": float(np.sum(selected)),
            "min": float(np.min(selected)),
            "max": float(np.max(selected)),
        }
    return payload


def build_sample_stats(
    image_path: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    intrinsics: CameraIntrinsics,
    depth_stats: dict[str, float],
    depth_processing_meta: dict[str, Any],
    point_cloud: PointCloudData,
    full_bundle: FeatureBundle,
    azimuth_bundle: FeatureBundle,
    pooled_tensor: np.ndarray,
    pooled_meta: dict[str, Any],
    channel_names: list[str],
) -> dict[str, Any]:
    observed_mask = full_bundle.channels["observed_mask"] > 0.5
    has_points = full_bundle.channels["has_points"] > 0.5

    sample_stats = {
        "input_image_path": str(image_path),
        "output_dir": str(output_dir),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "intrinsics": intrinsics.to_dict(),
        "hfov_deg": float(intrinsics.hfov_deg),
        "vfov_deg": float(intrinsics.vfov_deg),
        "num_az_bins": int(full_bundle.azimuth_centers.shape[0]),
        "num_el_bins": 1 if full_bundle.elevation_centers is None else int(full_bundle.elevation_centers.shape[0]),
        "channel_names": list(channel_names),
        "total_projected_points": int(full_bundle.total_projected_samples),
        "valid_points": int(full_bundle.total_valid_points),
        "point_cloud": point_cloud.stats.to_dict(),
        "depth_stats": depth_stats,
        "depth_processing": depth_processing_meta,
        "observed_bin_count": int(np.count_nonzero(observed_mask)),
        "occupied_bin_count": int(np.count_nonzero(has_points)),
        "empty_but_observed_bin_count": int(np.count_nonzero(observed_mask & ~has_points)),
        "fov_bin_count": int(np.count_nonzero(full_bundle.channels["fov_mask"] > 0.5)),
        "full_spherical_summary": full_bundle.summary_dict(),
        "azimuth_summary": azimuth_bundle.summary_dict(),
        "eight_way_pooled_feature_summary": summarize_pooled_tensor(
            pooled_tensor=pooled_tensor,
            channel_names=channel_names,
            labels=pooled_meta["labels"],
        ),
    }
    return sample_stats


def aggregate_run_summary(
    config_dict: dict[str, Any],
    processed_samples: list[dict[str, Any]],
    failed_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    observed_counts = np.asarray([sample["observed_bin_count"] for sample in processed_samples], dtype=np.float32)
    occupied_counts = np.asarray([sample["occupied_bin_count"] for sample in processed_samples], dtype=np.float32)
    empty_counts = np.asarray([sample["empty_but_observed_bin_count"] for sample in processed_samples], dtype=np.float32)
    invalid_ratios = np.asarray([sample["depth_stats"]["invalid_ratio"] for sample in processed_samples], dtype=np.float32)

    def summarize(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0}
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }

    run_summary = {
        "config": config_dict,
        "num_images": len(processed_samples) + len(failed_samples),
        "processed_count": len(processed_samples),
        "failure_count": len(failed_samples),
        "aggregate": {
            "observed_bin_count": summarize(observed_counts),
            "occupied_bin_count": summarize(occupied_counts),
            "empty_but_observed_bin_count": summarize(empty_counts),
            "depth_invalid_ratio": summarize(invalid_ratios),
        },
        "processed_samples": [
            {
                "input_image_path": sample["input_image_path"],
                "output_dir": sample["output_dir"],
                "observed_bin_count": sample["observed_bin_count"],
                "occupied_bin_count": sample["occupied_bin_count"],
                "depth_invalid_ratio": sample["depth_stats"]["invalid_ratio"],
            }
            for sample in processed_samples
        ],
        "failed_samples": failed_samples,
    }
    return run_summary


def aggregate_run_channel_stats(
    channel_names: list[str],
    payloads: list[dict[str, dict[str, float | int | str]]],
) -> dict[str, Any]:
    channel_stats: dict[str, dict[str, float | int | str]] = {}
    for channel_name in channel_names:
        total_count = 0
        total_sum = 0.0
        min_value = None
        max_value = None
        reduction_mask = ""

        for payload in payloads:
            channel_payload = payload[channel_name]
            reduction_mask = str(channel_payload["reduction_mask"])
            count = int(channel_payload["count"])
            total_count += count
            total_sum += float(channel_payload["sum"])
            if count == 0:
                continue
            current_min = float(channel_payload["min"])
            current_max = float(channel_payload["max"])
            min_value = current_min if min_value is None else min(min_value, current_min)
            max_value = current_max if max_value is None else max(max_value, current_max)

        mean_value = 0.0 if total_count == 0 else total_sum / float(total_count)
        channel_stats[channel_name] = {
            "reduction_mask": reduction_mask,
            "count": int(total_count),
            "mean": float(mean_value),
            "min": 0.0 if min_value is None else float(min_value),
            "max": 0.0 if max_value is None else float(max_value),
        }

    return {
        "channel_names": list(channel_names),
        "channel_stats": channel_stats,
    }
