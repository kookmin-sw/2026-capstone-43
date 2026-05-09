from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heard_direction_overlay.visualization.render_maps import render_planar_direction_map


def generate(
    output_dir: Path,
    direction_grid: dict[str, Any],
    iv_result: dict[str, Any],
    gt_annotation: dict[str, Any],
    camera_fov: dict[str, Any],
) -> Path:
    output_path = output_dir / "05_intensity_vector_direction_map.png"
    render_planar_direction_map(
        direction_map=iv_result["map"],
        azimuth_deg=direction_grid["azimuth_deg"],
        elevation_deg=direction_grid["elevation_deg"],
        output_path=output_path,
        title="05. Intensity Vector Direction Map",
        peak_direction=iv_result["peak_direction_camera"],
        gt_direction=gt_annotation["direction_camera"],
        camera_fov_boundary_directions=camera_fov["boundary_directions_camera"],
        camera_forward_direction=camera_fov["center_direction_camera"],
        cmap="magma",
        note="Heard directional evidence from FOA active intensity",
    )
    return output_path

