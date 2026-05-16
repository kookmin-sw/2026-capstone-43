from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .pit import IGNORE_INDEX, build_pit_targets


class MultiACCDOALoss(nn.Module):
    """PIT-aligned Multi-ACCDOA loss.

    Args:
        num_classes: Number of source classes C.
        lambda_acc_vec: Outer SmoothL1 loss weight for ACCDOA vector loss.
        lambda_acc_active: Inner SmoothL1 weight for active slots.
        lambda_acc_inactive: Inner SmoothL1 weight for inactive slots.
        lambda_acc_ang: Active-slot angular direction loss weight.
        lambda_cls: Active-slot class CE loss weight.
        lambda_dist: Active-slot distance loss weight.
        distance_is_log: Interpret pred["distance"] as log-distance.
        distance_target_is_log: Set False when target distances are raw meters.
            Set True when target distances are already log-distance.

    Input:
        pred dict with accdoa [B, K, 3], class_logits [B, K, C],
        distance [B, K, 1].
        targets list with B dicts: accdoa [N_i, 3], class [N_i],
        distance [N_i, 1].
    """

    def __init__(
        self,
        num_classes: int,
        lambda_acc_vec: float = 1.0,
        lambda_acc_active: float = 1.0,
        lambda_acc_inactive: float = 1.0,
        lambda_acc_ang: float = 0.2,
        lambda_cls: float = 0.5,
        lambda_dist: float = 0.2,
        distance_is_log: bool = True,
        distance_target_is_log: bool = False,
        target_distance_is_log: Optional[bool] = None,
        pit_weights: Optional[Dict[str, float]] = None,
        ignore_index: int = IGNORE_INDEX,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.lambda_acc_vec = float(lambda_acc_vec)
        self.lambda_acc_active = float(lambda_acc_active)
        self.lambda_acc_inactive = float(lambda_acc_inactive)
        self.lambda_acc_ang = float(lambda_acc_ang)
        self.lambda_cls = float(lambda_cls)
        self.lambda_dist = float(lambda_dist)
        self.distance_is_log = bool(distance_is_log)
        self.distance_target_is_log = (
            bool(target_distance_is_log)
            if target_distance_is_log is not None
            else bool(distance_target_is_log)
        )
        self.pit_weights = pit_weights
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

    def _zero_like_loss(self, reference: Tensor) -> Tensor:
        return reference.sum() * 0.0

    def forward(
        self,
        pred: Dict[str, Tensor],
        targets: Sequence[Dict[str, Tensor]],
    ) -> Dict[str, object]:
        """Compute total loss and return component losses plus PIT targets."""
        matched = build_pit_targets(
            pred,
            targets,
            num_classes=self.num_classes,
            distance_is_log=self.distance_is_log,
            weights=self.pit_weights,
            distance_target_is_log=self.distance_target_is_log,
            ignore_index=self.ignore_index,
            eps=self.eps,
        )
        pred_accdoa = pred["accdoa"]
        pred_class_logits = pred["class_logits"]
        pred_distance = pred["distance"]
        target_accdoa = matched["accdoa"]
        target_class = matched["class"]
        target_distance = matched["distance"]
        active_mask = matched["active_mask"]
        inactive_mask = ~active_mask

        if bool(active_mask.any()):
            loss_acc_active = F.smooth_l1_loss(
                pred_accdoa[active_mask],
                target_accdoa[active_mask],
            )
        else:
            loss_acc_active = self._zero_like_loss(pred_accdoa)

        if bool(inactive_mask.any()):
            loss_acc_inactive = F.smooth_l1_loss(
                pred_accdoa[inactive_mask],
                target_accdoa[inactive_mask],
            )
        else:
            loss_acc_inactive = self._zero_like_loss(pred_accdoa)

        loss_acc_vec = (
            self.lambda_acc_active * loss_acc_active
            + self.lambda_acc_inactive * loss_acc_inactive
        )

        if bool(active_mask.any()):
            active_pred = pred_accdoa[active_mask]
            active_target = target_accdoa[active_mask]
            pred_norm = torch.linalg.norm(active_pred, dim=-1, keepdim=True)
            pred_dir = active_pred / (pred_norm + self.eps)
            cosine = (pred_dir * active_target).sum(dim=-1).clamp(-1.0, 1.0)
            loss_acc_ang = (1.0 - cosine).mean()

            loss_cls = F.cross_entropy(
                pred_class_logits[active_mask],
                target_class[active_mask],
            )
            loss_dist = F.smooth_l1_loss(
                pred_distance[active_mask],
                target_distance[active_mask],
            )
        else:
            loss_acc_ang = self._zero_like_loss(pred_accdoa)
            loss_cls = self._zero_like_loss(pred_class_logits)
            loss_dist = self._zero_like_loss(pred_distance)

        total_loss = (
            self.lambda_acc_vec * loss_acc_vec
            + self.lambda_acc_ang * loss_acc_ang
            + self.lambda_cls * loss_cls
            + self.lambda_dist * loss_dist
        )
        return {
            "loss": total_loss,
            "loss_acc_vec": loss_acc_vec,
            "loss_acc_active": loss_acc_active,
            "loss_acc_inactive": loss_acc_inactive,
            "loss_acc_ang": loss_acc_ang,
            "loss_cls": loss_cls,
            "loss_dist": loss_dist,
            "matched": matched,
        }
