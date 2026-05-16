from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heard_direction_overlay.visualization.render_maps import render_rgb_image


def generate(output_dir: Path, rgb: np.ndarray, gt_annotation: dict[str, Any]) -> Path:
    output_path = output_dir / "01_rgb.png"
    render_rgb_image(
        rgb,
        output_path,
        gt_pixel_xy=gt_annotation["pixel_xy"],
        gt_in_view=gt_annotation["in_view"],
        out_of_view_reason=gt_annotation["out_of_view_reason"],
    )
    return output_path
