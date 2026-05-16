from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heard_direction_overlay.utils.io import write_ply_ascii
from heard_direction_overlay.visualization.render_overlay import render_raw_pointcloud


def generate(output_dir: Path, pointcloud: dict[str, np.ndarray], gt_annotation: dict[str, Any]) -> dict[str, str]:
    image_path = output_dir / "03_raw_pointcloud.png"
    ply_path = output_dir / "03_raw_pointcloud.ply"
    npz_path = output_dir / "03_raw_pointcloud.npz"
    render_raw_pointcloud(
        pointcloud["points"],
        pointcloud["colors"],
        image_path,
        gt_point_camera=gt_annotation["camera_point"],
    )
    write_ply_ascii(ply_path, pointcloud["points"], pointcloud["colors"])
    np.savez_compressed(
        npz_path,
        points=pointcloud["points"],
        colors=pointcloud["colors"],
        pixels=pointcloud["pixels"],
    )
    return {"image_path": str(image_path), "ply_path": str(ply_path), "npz_path": str(npz_path)}
