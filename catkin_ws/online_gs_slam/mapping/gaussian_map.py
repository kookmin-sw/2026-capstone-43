from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch


@dataclass
class GaussianMapConfig:
    material_dim: int = 16
    device: str = "cpu"
    prune_opacity_threshold: float = 0.01
    max_gaussians: int = 200_000


class GaussianMap:
    def __init__(self, config: GaussianMapConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.means = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.scales = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.rotations = torch.empty((0, 4), dtype=torch.float32, device=self.device)
        self.opacity = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        self.colors = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.observation_count = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        self.uncertainty = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        self.material_feature = torch.empty((0, config.material_dim), dtype=torch.float32, device=self.device)

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    def add_gaussians(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        opacity: Optional[torch.Tensor] = None,
        material_feature: Optional[torch.Tensor] = None,
    ) -> None:
        means = means.to(self.device, dtype=torch.float32)
        colors = colors.to(self.device, dtype=torch.float32)
        n = means.shape[0]
        if n == 0:
            return
        if scales is None:
            scales = torch.full((n, 3), 0.02, device=self.device)
        if rotations is None:
            rotations = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).repeat(n, 1)
        if opacity is None:
            opacity = torch.full((n, 1), 0.3, device=self.device)
        if material_feature is None:
            material_feature = torch.zeros((n, self.config.material_dim), device=self.device)

        self.means = torch.cat([self.means, means], dim=0)
        self.colors = torch.cat([self.colors, colors], dim=0)
        self.scales = torch.cat([self.scales, scales.to(self.device)], dim=0)
        self.rotations = torch.cat([self.rotations, rotations.to(self.device)], dim=0)
        self.opacity = torch.cat([self.opacity, opacity.to(self.device)], dim=0)
        self.observation_count = torch.cat([self.observation_count, torch.ones((n, 1), device=self.device)], dim=0)
        self.uncertainty = torch.cat([self.uncertainty, torch.ones((n, 1), device=self.device)], dim=0)
        self.material_feature = torch.cat([self.material_feature, material_feature.to(self.device)], dim=0)

        if self.num_gaussians > self.config.max_gaussians:
            keep = torch.arange(self.num_gaussians, device=self.device)[-self.config.max_gaussians :]
            self._select_in_place(keep)

    def _select_in_place(self, keep: torch.Tensor) -> None:
        self.means = self.means[keep]
        self.scales = self.scales[keep]
        self.rotations = self.rotations[keep]
        self.opacity = self.opacity[keep]
        self.colors = self.colors[keep]
        self.observation_count = self.observation_count[keep]
        self.uncertainty = self.uncertainty[keep]
        self.material_feature = self.material_feature[keep]

    def prune_gaussians(self) -> int:
        if self.num_gaussians == 0:
            return 0
        keep = (self.opacity[:, 0] >= self.config.prune_opacity_threshold).nonzero(as_tuple=False)[:, 0]
        removed = self.num_gaussians - int(keep.numel())
        self._select_in_place(keep)
        return removed

    def query_visible_gaussians(self, camera_to_world: torch.Tensor, radius: float = 5.0) -> torch.Tensor:
        if self.num_gaussians == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        center = camera_to_world[:3, 3].to(self.device)
        dist = torch.linalg.norm(self.means - center[None, :], dim=-1)
        return (dist < radius).nonzero(as_tuple=False)[:, 0]

    def query_local_region(self, center: torch.Tensor, radius: float) -> torch.Tensor:
        if self.num_gaussians == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        center = center.to(self.device)
        dist = torch.linalg.norm(self.means - center[None, :], dim=-1)
        return (dist < radius).nonzero(as_tuple=False)[:, 0]

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.__dict__,
                "means": self.means.detach().cpu(),
                "scales": self.scales.detach().cpu(),
                "rotations": self.rotations.detach().cpu(),
                "opacity": self.opacity.detach().cpu(),
                "colors": self.colors.detach().cpu(),
                "observation_count": self.observation_count.detach().cpu(),
                "uncertainty": self.uncertainty.detach().cpu(),
                "material_feature": self.material_feature.detach().cpu(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], device: Optional[str] = None) -> "GaussianMap":
        data = torch.load(path, map_location=device or "cpu")
        config = GaussianMapConfig(**data["config"])
        if device is not None:
            config.device = device
        gmap = cls(config)
        for key in ["means", "scales", "rotations", "opacity", "colors", "observation_count", "uncertainty", "material_feature"]:
            setattr(gmap, key, data[key].to(gmap.device))
        return gmap
