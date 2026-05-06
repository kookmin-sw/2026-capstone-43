from __future__ import annotations

from typing import Tuple

import torch

from online_gs_slam.mapping.gaussian_map import GaussianMap


def compute_uncertainty(gmap: GaussianMap) -> torch.Tensor:
    if gmap.num_gaussians == 0:
        return gmap.uncertainty
    obs_term = 1.0 / torch.sqrt(gmap.observation_count.clamp_min(1.0))
    opacity_term = 1.0 - gmap.opacity.clamp(0.0, 1.0)
    gmap.uncertainty = (0.7 * obs_term + 0.3 * opacity_term).clamp(0.0, 1.0)
    return gmap.uncertainty


def get_high_uncertainty_regions(gmap: GaussianMap, threshold: float = 0.6, top_k: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    uncertainty = compute_uncertainty(gmap)
    if uncertainty.numel() == 0:
        return torch.empty((0, 3), device=gmap.device), torch.empty((0, 1), device=gmap.device)
    mask = uncertainty[:, 0] >= threshold
    idx = mask.nonzero(as_tuple=False)[:, 0]
    if idx.numel() == 0:
        values, idx = torch.topk(uncertainty[:, 0], k=min(top_k, gmap.num_gaussians))
        return gmap.means[idx], values[:, None]
    if idx.numel() > top_k:
        values, order = torch.topk(uncertainty[idx, 0], k=top_k)
        idx = idx[order]
        return gmap.means[idx], values[:, None]
    return gmap.means[idx], uncertainty[idx]
