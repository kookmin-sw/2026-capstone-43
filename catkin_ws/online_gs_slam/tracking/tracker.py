from __future__ import annotations

from dataclasses import dataclass

import torch

from online_gs_slam.data.frame import Frame
from online_gs_slam.mapping.gaussian_map import GaussianMap
from online_gs_slam.rendering.renderer import OnlineGaussianRenderer


@dataclass
class TrackingConfig:
    optimize_pose: bool = False
    iterations: int = 10
    learning_rate: float = 1e-3


class GaussianTracker:
    def __init__(self, config: TrackingConfig, renderer: OnlineGaussianRenderer, gaussian_map: GaussianMap):
        self.config = config
        self.renderer = renderer
        self.gaussian_map = gaussian_map

    def track(self, frame: Frame) -> torch.Tensor:
        initial_pose = torch.from_numpy(frame.camera_to_world).float().to(self.gaussian_map.device)
        if not self.config.optimize_pose or self.gaussian_map.num_gaussians == 0:
            return initial_pose
        # TODO: render current Gaussian map, compute RGB/depth residual, and optimize SE(3) delta.
        return initial_pose

