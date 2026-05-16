from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor


def random_unit_directions(num_sources: int, device: str | torch.device = "cpu") -> Tensor:
    """Generate random unit direction vectors with shape [N, 3]."""
    raw = torch.randn(num_sources, 3, device=device)
    return raw / (torch.linalg.norm(raw, dim=-1, keepdim=True) + 1e-8)


def generate_synthetic_batch(
    batch_size: int,
    num_classes: int,
    kmax: int = 3,
    min_sources: int = 2,
    max_sources: int = 2,
    distance_range: Tuple[float, float] = (1.0, 3.0),
    device: str | torch.device = "cpu",
) -> List[Dict[str, Tensor]]:
    """Generate a list of synthetic PIT targets.

    Args:
        batch_size: Number of samples B.
        num_classes: Number of classes C.
        kmax: Maximum number of source slots K.
        min_sources: Minimum active sources per sample.
        max_sources: Maximum active sources per sample.
        distance_range: Raw meter distance range.
        device: Target tensor device.

    Returns:
        List of B target dicts. Each contains accdoa [N_i, 3], class [N_i],
        distance [N_i, 1] in raw meters.
    """
    if min_sources < 0 or max_sources < min_sources:
        raise ValueError("invalid source count range")
    if max_sources > kmax:
        raise ValueError("max_sources cannot exceed kmax")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    low, high = float(distance_range[0]), float(distance_range[1])
    if low <= 0 or high <= low:
        raise ValueError("distance_range must be positive and increasing")

    targets: List[Dict[str, Tensor]] = []
    for _ in range(batch_size):
        if min_sources == max_sources:
            num_sources = min_sources
        else:
            num_sources = int(torch.randint(min_sources, max_sources + 1, (1,), device=device).item())
        directions = random_unit_directions(num_sources, device=device)
        classes = torch.randint(0, num_classes, (num_sources,), dtype=torch.long, device=device)
        distances = low + (high - low) * torch.rand(num_sources, 1, device=device)
        targets.append(
            {
                "accdoa": directions,
                "class": classes,
                "distance": distances,
            }
        )
    return targets


def generate_toy_inputs(
    batch_size: int,
    input_dim: int,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Generate toy model inputs with shape [B, input_dim]."""
    return torch.randn(batch_size, input_dim, device=device)
