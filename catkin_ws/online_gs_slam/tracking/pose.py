from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PoseRefinementConfig:
    iterations: int = 10
    learning_rate: float = 1e-3


class SE3Pose:
    """Small placeholder SE(3) wrapper.

    Current skeleton trusts incoming poses. Future work: Lie algebra delta update
    for differentiable pose refinement against rendered Gaussian images.
    """

    def __init__(self, camera_to_world: torch.Tensor):
        self.camera_to_world = camera_to_world

    def matrix(self) -> torch.Tensor:
        return self.camera_to_world

