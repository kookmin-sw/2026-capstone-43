from __future__ import annotations

import torch

from online_gs_slam.mapping.gaussian_map import GaussianMap


def query_gaussians_near_touch(gmap: GaussianMap, touch_position: torch.Tensor, radius: float = 0.05) -> torch.Tensor:
    return gmap.query_local_region(touch_position, radius)


def update_material_features(gmap: GaussianMap, indices: torch.Tensor, material_embedding: torch.Tensor, momentum: float = 0.5) -> None:
    if indices.numel() == 0:
        return
    embedding = material_embedding.to(gmap.device)[None, :]
    gmap.material_feature[indices] = momentum * gmap.material_feature[indices] + (1.0 - momentum) * embedding


def compute_material_uncertainty(gmap: GaussianMap) -> torch.Tensor:
    if gmap.num_gaussians == 0:
        return torch.empty((0, 1), device=gmap.device)
    magnitude = torch.linalg.norm(gmap.material_feature, dim=-1, keepdim=True)
    return torch.exp(-magnitude)

