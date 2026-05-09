from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heard_direction_overlay.filtering.point_scoring import score_points_with_directional_map
from heard_direction_overlay.utils.config import PipelineConfig
from heard_direction_overlay.visualization.overview import build_overview
from heard_direction_overlay.visualization.render_overlay import render_filtered_overlay


def score_and_render_overlay(
    rgb: np.ndarray,
    pointcloud: dict[str, np.ndarray],
    direction_grid: dict[str, np.ndarray],
    direction_map: np.ndarray,
    output_path: Path,
    title: str,
    subtitle: str,
    config: PipelineConfig,
    gt_annotation: dict[str, Any],
) -> dict[str, Any]:
    scoring = score_points_with_directional_map(
        points=pointcloud["points"],
        point_directions_camera=pointcloud["directions"],
        direction_map=direction_map,
        azimuth_axis_deg=direction_grid["azimuth_deg"],
        elevation_axis_deg=direction_grid["elevation_deg"],
        score_percentile=config.point_filter.overlay_score_percentile,
        gamma=config.point_filter.overlay_gamma,
        min_alpha=config.point_filter.overlay_min_alpha,
        max_alpha=config.point_filter.overlay_max_alpha,
    )
    render_filtered_overlay(
        rgb=rgb,
        pixels=pointcloud["pixels"],
        colors=pointcloud["colors"],
        normalized_scores=scoring["normalized_scores"],
        hard_mask=scoring["hard_mask"],
        soft_alpha=scoring["soft_alpha"],
        output_path=output_path,
        title=title,
        subtitle=subtitle,
        point_size=config.point_filter.filtered_point_size,
        gt_pixel_xy=gt_annotation["pixel_xy"],
        gt_in_view=gt_annotation["in_view"],
        out_of_view_reason=gt_annotation["out_of_view_reason"],
    )
    return {
        "threshold": float(scoring["threshold"]),
        "kept_count": int(scoring["kept_count"]),
        "score_stats": {
            "max": float(np.max(scoring["normalized_scores"])) if len(scoring["normalized_scores"]) else 0.0,
            "mean": float(np.mean(scoring["normalized_scores"])) if len(scoring["normalized_scores"]) else 0.0,
        },
    }


def score_and_render_overlay_with_presence_floor(
    rgb: np.ndarray,
    pointcloud: dict[str, np.ndarray],
    direction_grid: dict[str, np.ndarray],
    direction_map: np.ndarray,
    output_path: Path,
    title: str,
    subtitle: str,
    config: PipelineConfig,
    gt_annotation: dict[str, Any],
    min_presence_score: float,
) -> dict[str, Any]:
    scoring = score_points_with_directional_map(
        points=pointcloud["points"],
        point_directions_camera=pointcloud["directions"],
        direction_map=direction_map,
        azimuth_axis_deg=direction_grid["azimuth_deg"],
        elevation_axis_deg=direction_grid["elevation_deg"],
        score_percentile=config.point_filter.overlay_score_percentile,
        gamma=config.point_filter.overlay_gamma,
        min_alpha=config.point_filter.overlay_min_alpha,
        max_alpha=config.point_filter.overlay_max_alpha,
    )
    normalized = np.asarray(scoring["normalized_scores"], dtype=np.float32)
    percentile_threshold = float(scoring["threshold"])
    effective_threshold = float(max(percentile_threshold, float(min_presence_score)))
    hard_mask = normalized >= effective_threshold
    render_filtered_overlay(
        rgb=rgb,
        pixels=pointcloud["pixels"],
        colors=pointcloud["colors"],
        normalized_scores=normalized,
        hard_mask=hard_mask,
        soft_alpha=scoring["soft_alpha"],
        output_path=output_path,
        title=title,
        subtitle=subtitle,
        point_size=config.point_filter.filtered_point_size,
        gt_pixel_xy=gt_annotation["pixel_xy"],
        gt_in_view=gt_annotation["in_view"],
        out_of_view_reason=gt_annotation["out_of_view_reason"],
    )
    return {
        "threshold": effective_threshold,
        "percentile_threshold": percentile_threshold,
        "min_presence_score": float(min_presence_score),
        "kept_count": int(np.sum(hard_mask)),
        "score_stats": {
            "max": float(np.max(normalized)) if len(normalized) else 0.0,
            "mean": float(np.mean(normalized)) if len(normalized) else 0.0,
        },
    }


def save_gif_from_frames(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
) -> str | None:
    if not frame_paths:
        return None
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    if not images:
        return None
    duration_ms = int(max(1000.0 / max(float(fps), 0.1), 80.0))
    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        return str(output_path)
    finally:
        for image in images:
            image.close()


def build_subset_overview(
    output_dir: Path,
    columns: int = 4,
    extra_tail_paths: list[Path] | None = None,
) -> list[Path]:
    ordered = [
        output_dir / "01_rgb.png",
        output_dir / "02_depth.png",
        output_dir / "03_raw_pointcloud.png",
        output_dir / "04_raw_pointcloud_overlay.png",
        output_dir / "05_intensity_vector_direction_map.png",
        output_dir / "06_beam_power_direction_map.png",
        output_dir / "12_beam_filtered_overlay.png",
    ]
    if extra_tail_paths:
        ordered.extend(extra_tail_paths)
    existing = [path for path in ordered if path.exists()]
    build_overview(existing, output_dir / "14_overview.png", columns=max(int(columns), 1))
    return existing
