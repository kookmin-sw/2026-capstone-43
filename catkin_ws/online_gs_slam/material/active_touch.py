from __future__ import annotations

import torch

from online_gs_slam.mapping.gaussian_map import GaussianMap
from online_gs_slam.mapping.uncertainty import get_high_uncertainty_regions
from online_gs_slam.material.touch_update import compute_material_uncertainty


def suggest_next_touch_region(gmap: GaussianMap, top_k: int = 8) -> torch.Tensor:
    visual_centers, visual_u = get_high_uncertainty_regions(gmap, top_k=top_k)
    if visual_centers.numel() == 0:
        return visual_centers
    material_u = compute_material_uncertainty(gmap)
    # TODO: fuse visual and material uncertainty spatially. Current prototype returns visual uncertainty centers.
    _ = material_u
    return visual_centers

