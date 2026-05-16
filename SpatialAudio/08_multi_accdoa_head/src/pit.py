from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


IGNORE_INDEX = -100


@dataclass(frozen=True)
class PITWeights:
    """Weights used only for PIT assignment cost."""

    acc: float = 1.0
    cls: float = 0.2
    dist: float = 0.1


def _weights_from_dict(weights: Optional[Dict[str, float]]) -> PITWeights:
    if weights is None:
        return PITWeights()
    return PITWeights(
        acc=float(weights.get("acc", weights.get("w_acc", 1.0))),
        cls=float(weights.get("cls", weights.get("w_cls", 0.2))),
        dist=float(weights.get("dist", weights.get("w_dist", 0.1))),
    )


def _as_target_tensor(value: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def prepare_distance_target(
    distance: Tensor,
    distance_target_is_log: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """Prepare target distance for the default log-distance regression space.

    Args:
        distance: Target distance tensor, usually [N, 1].
        distance_target_is_log: If False, distance is raw meters and this
            function returns log(distance + eps). If True, distance is already
            log-distance and is returned unchanged.
        eps: Clamp value for raw meter distances before log().

    Returns:
        Log-distance target tensor in the same shape as input.
    """
    if distance_target_is_log:
        return distance
    return torch.log(distance.clamp_min(eps))


def _resolve_distance_target_is_log(
    distance_target_is_log: bool,
    target_distance_is_log: Optional[bool],
) -> bool:
    """Support the old target_distance_is_log name without changing behavior."""
    if target_distance_is_log is not None:
        return bool(target_distance_is_log)
    return bool(distance_target_is_log)


def _normalize_direction(direction: Tensor, eps: float) -> Tensor:
    return direction / (torch.linalg.norm(direction, dim=-1, keepdim=True) + eps)


def _pair_cost_matrix(
    pred_accdoa: Tensor,
    pred_class_logits: Tensor,
    pred_distance: Tensor,
    target_accdoa: Tensor,
    target_class: Tensor,
    target_distance: Tensor,
    weights: PITWeights,
    eps: float,
) -> Tensor:
    """Build active source matching cost matrix [K, N]."""
    pred_dir = _normalize_direction(pred_accdoa, eps)
    gt_dir = _normalize_direction(target_accdoa, eps)
    cosine = torch.matmul(pred_dir, gt_dir.transpose(0, 1)).clamp(-1.0, 1.0)
    acc_cost = 1.0 - cosine

    log_probs = F.log_softmax(pred_class_logits, dim=-1)
    cls_cost = -log_probs[:, target_class.long()]

    pred_distance_2d = pred_distance.reshape(-1, 1)
    target_distance_2d = target_distance.reshape(1, -1)
    dist_cost = F.smooth_l1_loss(
        pred_distance_2d.expand(-1, target_distance_2d.shape[1]),
        target_distance_2d.expand(pred_distance_2d.shape[0], -1),
        reduction="none",
    )

    return weights.acc * acc_cost + weights.cls * cls_cost + weights.dist * dist_cost


def _best_assignment(cost_matrix: Tensor) -> List[Tuple[int, int]]:
    """Return sorted [(pred_slot, gt_index), ...] for the minimum-cost PIT match."""
    kmax, num_sources = cost_matrix.shape
    if num_sources == 0:
        return []
    if num_sources > kmax:
        raise ValueError(f"num_sources={num_sources} exceeds kmax={kmax}")

    best_cost: Optional[float] = None
    best_pairs: List[Tuple[int, int]] = []
    for pred_slots in itertools.permutations(range(kmax), num_sources):
        cost = 0.0
        for gt_index, pred_slot in enumerate(pred_slots):
            cost += float(cost_matrix[pred_slot, gt_index].item())
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_pairs = [(pred_slot, gt_index) for gt_index, pred_slot in enumerate(pred_slots)]
    return sorted(best_pairs, key=lambda pair: pair[0])


def build_pit_targets(
    pred: Dict[str, Tensor],
    targets: Sequence[Dict[str, Tensor]],
    num_classes: int,
    distance_is_log: bool = True,
    weights: Optional[Dict[str, float]] = None,
    distance_target_is_log: bool = False,
    target_distance_is_log: Optional[bool] = None,
    ignore_index: int = IGNORE_INDEX,
    eps: float = 1e-6,
) -> Dict[str, object]:
    """Build slot-aligned targets with exhaustive PIT matching.

    Args:
        pred: Dict containing accdoa [B, K, 3], class_logits [B, K, C],
            distance [B, K, 1].
        targets: List of B dicts. Each dict contains accdoa [N_i, 3],
            class [N_i], distance [N_i, 1].
        num_classes: Number of classes C, used for validation.
        distance_is_log: If True, prediction distance is interpreted as
            log-distance. This is the default and recommended mode.
        weights: Optional assignment cost weights. Keys: acc/cls/dist or
            w_acc/w_cls/w_dist.
        distance_target_is_log: Set False when target["distance"] is raw
            meters. Set True when it is already log-distance. Matched distance
            stores log-distance targets in the default distance_is_log=True mode.
        target_distance_is_log: Backward-compatible alias for
            distance_target_is_log.
        ignore_index: Class target value for inactive slots.
        eps: Numerical stability constant.

    Returns:
        Dict with matched accdoa [B, K, 3], class [B, K], distance [B, K, 1],
        active_mask [B, K], and assignments list of [(pred_slot, gt_index)].
    """
    pred_accdoa = pred["accdoa"]
    pred_class_logits = pred["class_logits"]
    pred_distance = pred["distance"]
    if pred_accdoa.ndim != 3 or pred_accdoa.shape[-1] != 3:
        raise ValueError(f"pred['accdoa'] must be [B, K, 3], got {tuple(pred_accdoa.shape)}")
    if pred_class_logits.ndim != 3 or pred_class_logits.shape[-1] != num_classes:
        raise ValueError(
            "pred['class_logits'] must be [B, K, C], got "
            f"{tuple(pred_class_logits.shape)} with C={num_classes}"
        )
    if pred_distance.ndim != 3 or pred_distance.shape[-1] != 1:
        raise ValueError(f"pred['distance'] must be [B, K, 1], got {tuple(pred_distance.shape)}")

    batch_size, kmax, _ = pred_accdoa.shape
    if len(targets) != batch_size:
        raise ValueError(f"targets length {len(targets)} does not match batch size {batch_size}")

    device = pred_accdoa.device
    matched_accdoa = pred_accdoa.new_zeros(batch_size, kmax, 3)
    matched_distance = pred_distance.new_zeros(batch_size, kmax, 1)
    matched_class = torch.full(
        (batch_size, kmax),
        int(ignore_index),
        dtype=torch.long,
        device=device,
    )
    active_mask = torch.zeros(batch_size, kmax, dtype=torch.bool, device=device)
    assignments: List[List[Tuple[int, int]]] = []
    pit_weights = _weights_from_dict(weights)

    resolved_distance_target_is_log = _resolve_distance_target_is_log(
        distance_target_is_log,
        target_distance_is_log,
    )

    with torch.no_grad():
        for batch_index, target in enumerate(targets):
            target_accdoa = _as_target_tensor(target["accdoa"], device, pred_accdoa.dtype)
            target_class = _as_target_tensor(target["class"], device, torch.long)
            target_distance = _as_target_tensor(target["distance"], device, pred_distance.dtype)

            if target_accdoa.ndim != 2 or target_accdoa.shape[-1] != 3:
                raise ValueError("target accdoa must be [N, 3]")
            if target_class.ndim != 1 or target_class.shape[0] != target_accdoa.shape[0]:
                raise ValueError("target class must be [N]")
            if target_distance.ndim == 1:
                target_distance = target_distance[:, None]
            if target_distance.ndim != 2 or target_distance.shape != (target_accdoa.shape[0], 1):
                raise ValueError("target distance must be [N, 1]")

            num_sources = int(target_accdoa.shape[0])
            if num_sources > kmax:
                raise ValueError(f"target has {num_sources} sources, but kmax is {kmax}")
            if num_sources == 0:
                assignments.append([])
                continue

            target_accdoa = _normalize_direction(target_accdoa, eps)
            if distance_is_log:
                target_distance_space = prepare_distance_target(
                    target_distance,
                    distance_target_is_log=resolved_distance_target_is_log,
                    eps=eps,
                )
            elif resolved_distance_target_is_log:
                target_distance_space = torch.exp(target_distance)
            else:
                target_distance_space = target_distance
            cost_matrix = _pair_cost_matrix(
                pred_accdoa[batch_index].detach(),
                pred_class_logits[batch_index].detach(),
                pred_distance[batch_index].detach().squeeze(-1),
                target_accdoa,
                target_class,
                target_distance_space,
                pit_weights,
                eps,
            )
            pairs = _best_assignment(cost_matrix)
            assignments.append(pairs)
            for pred_slot, gt_index in pairs:
                matched_accdoa[batch_index, pred_slot] = target_accdoa[gt_index]
                matched_class[batch_index, pred_slot] = target_class[gt_index]
                matched_distance[batch_index, pred_slot] = target_distance_space[gt_index]
                active_mask[batch_index, pred_slot] = True

    return {
        "accdoa": matched_accdoa,
        "class": matched_class,
        "distance": matched_distance,
        "active_mask": active_mask,
        "assignments": assignments,
        "ignore_index": ignore_index,
    }
