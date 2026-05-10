from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heard_direction_overlay.utils.config import PipelineConfig

from output_common import score_and_render_overlay


def generate(
    output_dir: Path,
    rgb: np.ndarray,
    pointcloud: dict[str, np.ndarray],
    direction_grid: dict[str, np.ndarray],
    beam_result: dict[str, Any],
    gt_annotation: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    return score_and_render_overlay(
        rgb=rgb,
        pointcloud=pointcloud,
        direction_grid=direction_grid,
        direction_map=beam_result["map"],
        output_path=output_dir / "12_beam_filtered_overlay.png",
        title="12. Beam-Power-filtered Point Cloud Overlay",
        subtitle="Visible geometry consistent with beam-power directional evidence",
        config=config,
        gt_annotation=gt_annotation,
    )

