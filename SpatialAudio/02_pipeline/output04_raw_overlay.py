from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heard_direction_overlay.geometry.depth_to_pointcloud import normalize_depth_for_visualization
from heard_direction_overlay.utils.config import PipelineConfig
from heard_direction_overlay.visualization.render_overlay import render_raw_overlay


def _build_depth_colors(pointcloud: dict[str, np.ndarray]) -> np.ndarray:
    depth_values = np.asarray(pointcloud["points"][:, 2], dtype=np.float32)
    normalized_depth, _ = normalize_depth_for_visualization(depth_values)
    colors = plt.cm.turbo(1.0 - normalized_depth)[..., :3]
    return np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)


def generate(
    output_dir: Path,
    rgb: np.ndarray,
    pointcloud: dict[str, np.ndarray],
    config: PipelineConfig,
    gt_annotation: dict[str, Any],
) -> Path:
    output_path = output_dir / "04_raw_pointcloud_overlay.png"
    depth_colors = _build_depth_colors(pointcloud)
    render_raw_overlay(
        rgb=rgb,
        pixels=pointcloud["pixels"],
        colors=depth_colors,
        output_path=output_path,
        point_size=max(float(config.point_filter.raw_point_size), 2.0),
        alpha=max(float(config.point_filter.raw_alpha), 0.65),
        title="Depth-Colored RGB Overlay of Raw Point Cloud",
        gt_pixel_xy=gt_annotation["pixel_xy"],
        gt_in_view=gt_annotation["in_view"],
        out_of_view_reason=gt_annotation["out_of_view_reason"],
    )
    return output_path
