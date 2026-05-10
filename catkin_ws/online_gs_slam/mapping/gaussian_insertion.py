from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from online_gs_slam.data.frame import Frame
from online_gs_slam.utils.camera import camera_ray_directions


@dataclass
class GaussianInsertionConfig:
    stride: int = 24
    rgb_only_depth: float = 1.5
    max_insert_per_frame: int = 1200
    initial_scale: float = 0.025
    residual_threshold: float = 0.15
    min_insert_per_frame: int = 32


class GaussianInserter:
    def __init__(self, config: GaussianInsertionConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)

    def propose_from_frame(self, frame: Frame, residual_mask: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intr = frame.intrinsics
        dirs, xs, ys = camera_ray_directions(
            intr.width,
            intr.height,
            intr.fx,
            intr.fy,
            intr.cx,
            intr.cy,
            stride=self.config.stride,
        )
        if residual_mask is not None:
            keep = residual_mask[ys, xs] > self.config.residual_threshold
            if keep.sum() < self.config.min_insert_per_frame and residual_mask.size > 0:
                sampled_residual = residual_mask[ys, xs]
                top_k = min(self.config.min_insert_per_frame, sampled_residual.shape[0])
                top_idx = np.argpartition(sampled_residual, -top_k)[-top_k:]
                keep = np.zeros_like(keep, dtype=bool)
                keep[top_idx] = True
            dirs, xs, ys = dirs[keep], xs[keep], ys[keep]

        if dirs.shape[0] > self.config.max_insert_per_frame:
            choice = np.linspace(0, dirs.shape[0] - 1, self.config.max_insert_per_frame).astype(np.int64)
            dirs, xs, ys = dirs[choice], xs[choice], ys[choice]

        if frame.depth is not None:
            depth = frame.depth[ys, xs].astype(np.float32)
            valid = depth > 0.0
            dirs, xs, ys, depth = dirs[valid], xs[valid], ys[valid], depth[valid]
        else:
            depth = np.full((dirs.shape[0],), self.config.rgb_only_depth, dtype=np.float32)

        points_camera_ros = dirs * depth[:, None]
        # frame.camera_to_world is OpenGL camera-to-world. Convert sampled points from ROS optical to OpenGL camera.
        points_camera_gl = points_camera_ros * np.array([1.0, -1.0, -1.0], dtype=np.float32)
        points_h = np.concatenate([points_camera_gl, np.ones((points_camera_gl.shape[0], 1), dtype=np.float32)], axis=-1)
        points_world = (frame.camera_to_world @ points_h.T).T[:, :3]
        colors = frame.rgb[ys, xs].astype(np.float32) / 255.0
        scales = np.full((points_world.shape[0], 3), self.config.initial_scale, dtype=np.float32)
        return (
            torch.from_numpy(points_world).to(self.device),
            torch.from_numpy(colors).to(self.device),
            torch.from_numpy(scales).to(self.device),
        )
