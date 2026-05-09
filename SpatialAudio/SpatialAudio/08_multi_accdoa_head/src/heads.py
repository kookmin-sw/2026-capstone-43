from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


class JointMultiSourceHead(nn.Module):
    """Joint source-slot head for Multi-ACCDOA experiments.

    Args:
        hidden_dim: Slot token dimension D.
        num_classes: Number of source classes C.
        kmax: Maximum number of source slots K.

    Input:
        slot_tokens: Tensor with shape [B, K, D].

    Output:
        dict with accdoa [B, K, 3], class_logits [B, K, C],
        distance [B, K, 1]. The distance output is interpreted as log-distance
        by the default loss, but the head itself stays unconstrained.
    """

    def __init__(self, hidden_dim: int, num_classes: int, kmax: int = 3) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if kmax <= 0:
            raise ValueError("kmax must be positive")

        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.kmax = int(kmax)
        self.out_dim = 3 + self.num_classes + 1
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, slot_tokens: Tensor) -> Dict[str, Tensor]:
        """Predict joint slot attributes.

        Args:
            slot_tokens: [B, K, D].

        Returns:
            accdoa: [B, K, 3] raw ACCDOA vectors. Do not normalize here,
                because zero-norm inactive slots must be representable.
            class_logits: [B, K, C].
            distance: [B, K, 1] scalar log-distance prediction by convention.
        """
        if slot_tokens.ndim != 3:
            raise ValueError(f"slot_tokens must be [B, K, D], got {tuple(slot_tokens.shape)}")
        if slot_tokens.shape[1] != self.kmax:
            raise ValueError(f"expected K={self.kmax}, got K={slot_tokens.shape[1]}")
        if slot_tokens.shape[2] != self.hidden_dim:
            raise ValueError(f"expected D={self.hidden_dim}, got D={slot_tokens.shape[2]}")

        joint_out = self.head(slot_tokens)
        accdoa = joint_out[..., 0:3]
        class_logits = joint_out[..., 3 : 3 + self.num_classes]
        distance = joint_out[..., 3 + self.num_classes : 3 + self.num_classes + 1]
        return {
            "accdoa": accdoa,
            "class_logits": class_logits,
            "distance": distance,
        }


def decode_accdoa(
    accdoa: Tensor,
    distance: Tensor,
    class_logits: Tensor,
    activity_threshold: float = 0.5,
    eps: float = 1e-8,
    distance_is_log: bool = True,
) -> Dict[str, Tensor]:
    """Decode raw slot outputs for inference.

    Args:
        accdoa: [B, K, 3] raw ACCDOA vectors.
        distance: [B, K, 1] predicted scalar distance, log-space by default.
        class_logits: [B, K, C].
        activity_threshold: Source activity threshold applied to ||accdoa||.
        eps: Numerical stability constant.
        distance_is_log: If True, return exp(distance) as meters.

    Returns:
        activity: [B, K] ACCDOA norm.
        active_mask: [B, K] boolean active prediction mask.
        direction: [B, K, 3] unit direction where possible.
        class_id: [B, K] argmax class id.
        distance: [B, K, 1] decoded distance.
    """
    if accdoa.ndim != 3 or accdoa.shape[-1] != 3:
        raise ValueError(f"accdoa must be [B, K, 3], got {tuple(accdoa.shape)}")
    activity = torch.linalg.norm(accdoa, dim=-1)
    direction = accdoa / (activity.unsqueeze(-1) + eps)
    active_mask = activity > float(activity_threshold)
    class_id = torch.argmax(class_logits, dim=-1)
    decoded_distance = torch.exp(distance) if distance_is_log else distance
    return {
        "activity": activity,
        "active_mask": active_mask,
        "direction": direction,
        "class_id": class_id,
        "distance": decoded_distance,
    }
