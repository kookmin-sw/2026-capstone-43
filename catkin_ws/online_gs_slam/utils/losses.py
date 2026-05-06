from __future__ import annotations

from typing import Optional

import torch


def rgb_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def depth_l1_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if valid_mask is not None:
        pred = pred[valid_mask]
        target = target[valid_mask]
    return torch.mean(torch.abs(pred - target))
