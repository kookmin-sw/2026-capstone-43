from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from .heads import JointMultiSourceHead


class ToyMultiSourceModel(nn.Module):
    """Tiny model for validating Multi-ACCDOA head/loss without a real encoder.

    Input:
        x: [B, input_dim].

    Internal:
        feat: [B, D]
        slot_queries: [K, D]
        slot_tokens = feat[:, None, :] + slot_queries[None, :, :] -> [B, K, D]

    Output:
        dict from JointMultiSourceHead.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        kmax: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.kmax = int(kmax)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.slot_queries = nn.Parameter(torch.randn(self.kmax, self.hidden_dim) * 0.02)
        self.head = JointMultiSourceHead(
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            kmax=self.kmax,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"x must be [B, {self.input_dim}], got {tuple(x.shape)}")
        feat = self.backbone(x)
        slot_tokens = feat[:, None, :] + self.slot_queries[None, :, :]
        return self.head(slot_tokens)
