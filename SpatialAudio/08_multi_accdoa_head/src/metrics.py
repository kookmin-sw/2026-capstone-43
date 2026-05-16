from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .heads import decode_accdoa
from .pit import prepare_distance_target


def _zero_scalar_like(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _unit(vector: Tensor, eps: float = 1e-8) -> Tensor:
    return vector / (torch.linalg.norm(vector, dim=-1, keepdim=True) + eps)


def angular_error_deg(pred_dir: Tensor, target_dir: Tensor, eps: float = 1e-8) -> Tensor:
    """Angular error in degrees.

    Args:
        pred_dir: [..., 3] predicted direction vectors.
        target_dir: [..., 3] target unit direction vectors.

    Returns:
        Tensor with shape [...], angle in degrees.
    """
    pred_unit = _unit(pred_dir, eps)
    target_unit = _unit(target_dir, eps)
    cosine = (pred_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def accdoa_activity(accdoa: Tensor) -> Tensor:
    """Return activity/presence ||accdoa|| with shape [B, K]."""
    return torch.linalg.norm(accdoa, dim=-1)


def decode_predictions(
    pred: Dict[str, Tensor],
    activity_threshold: float = 0.5,
    distance_is_log: bool = True,
) -> Dict[str, Tensor]:
    """Decode model output dict into activity, unit direction, class, distance."""
    return decode_accdoa(
        pred["accdoa"],
        pred["distance"],
        pred["class_logits"],
        activity_threshold=activity_threshold,
        distance_is_log=distance_is_log,
    )


def matched_angular_mae(
    pred: Dict[str, Tensor],
    targets: Optional[object],
    matched: Dict[str, object],
    eps: float = 1e-8,
) -> Tensor:
    """Mean angular error over PIT-matched active slots.

    The targets argument is accepted for a stable external interface; the
    matched dict already contains the aligned target tensors.
    """
    del targets
    active_mask = matched["active_mask"]
    if not bool(active_mask.any()):
        return _zero_scalar_like(pred["accdoa"])
    pred_active = pred["accdoa"][active_mask]
    target_active = matched["accdoa"][active_mask]
    return angular_error_deg(pred_active, target_active, eps=eps).mean()


def distance_mae(
    pred: Dict[str, Tensor],
    targets: Optional[object],
    matched: Dict[str, object],
    distance_is_log: bool = True,
) -> Tensor:
    """Mean absolute distance error over active matched slots."""
    del targets
    active_mask = matched["active_mask"]
    if not bool(active_mask.any()):
        return _zero_scalar_like(pred["distance"])
    pred_distance = pred["distance"][active_mask]
    target_distance = matched["distance"][active_mask]
    if distance_is_log:
        pred_distance = torch.exp(pred_distance)
        target_distance = torch.exp(target_distance)
    return (pred_distance - target_distance).abs().mean()


def class_accuracy(
    pred: Dict[str, Tensor],
    targets: Optional[object],
    matched: Dict[str, object],
) -> Tensor:
    """Class accuracy over active matched slots."""
    del targets
    active_mask = matched["active_mask"]
    if not bool(active_mask.any()):
        return _zero_scalar_like(pred["class_logits"])
    pred_class = torch.argmax(pred["class_logits"], dim=-1)
    return (pred_class[active_mask] == matched["class"][active_mask]).float().mean()


def activity_precision_recall(
    pred_accdoa: Tensor,
    matched_active_mask: Tensor,
    threshold: float = 0.5,
) -> Dict[str, Tensor]:
    """Slot activity precision/recall using ||accdoa|| as activity score."""
    pred_active = accdoa_activity(pred_accdoa) > float(threshold)
    target_active = matched_active_mask.bool()
    tp = (pred_active & target_active).sum().float()
    fp = (pred_active & ~target_active).sum().float()
    fn = (~pred_active & target_active).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _as_tensor(value: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _best_rectangular_assignment(cost_matrix: Tensor) -> List[Tuple[int, int]]:
    """Exhaustively match min(P, N) pairs for a small [P, N] cost matrix."""
    num_pred, num_gt = cost_matrix.shape
    pair_count = min(num_pred, num_gt)
    if pair_count == 0:
        return []

    best_cost: Optional[float] = None
    best_pairs: List[Tuple[int, int]] = []
    for pred_subset in itertools.combinations(range(num_pred), pair_count):
        for gt_perm in itertools.permutations(range(num_gt), pair_count):
            cost = 0.0
            pairs = []
            for pred_index, gt_index in zip(pred_subset, gt_perm):
                cost += float(cost_matrix[pred_index, gt_index].item())
                pairs.append((pred_index, gt_index))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_pairs = pairs
    return best_pairs


def _angular_cost_matrix(pred_accdoa: Tensor, target_accdoa: Tensor) -> Tensor:
    pred_dir = _unit(pred_accdoa)
    target_dir = _unit(target_accdoa)
    return 1.0 - torch.matmul(pred_dir, target_dir.transpose(0, 1)).clamp(-1.0, 1.0)


def _joint_cost_matrix(
    pred: Dict[str, Tensor],
    pred_slots: Tensor,
    target: Dict[str, Tensor],
    distance_target_is_log: bool,
    distance_is_log: bool,
    eps: float,
) -> Tensor:
    """Build angular/class/distance cost for selected slots [P, N]."""
    device = pred["accdoa"].device
    pred_accdoa = pred["accdoa"][pred_slots]
    target_accdoa = _as_tensor(target["accdoa"], device, pred_accdoa.dtype)
    target_accdoa = _unit(target_accdoa, eps)
    cost = _angular_cost_matrix(pred_accdoa, target_accdoa)

    if "class" in target and pred["class_logits"].shape[-1] > 0:
        target_class = _as_tensor(target["class"], device, torch.long)
        log_probs = F.log_softmax(pred["class_logits"][pred_slots], dim=-1)
        cost = cost + 0.2 * (-log_probs[:, target_class])

    if "distance" in target:
        target_distance = _as_tensor(target["distance"], device, pred["distance"].dtype)
        if target_distance.ndim == 1:
            target_distance = target_distance[:, None]
        if distance_is_log:
            target_distance_space = prepare_distance_target(
                target_distance,
                distance_target_is_log=distance_target_is_log,
                eps=eps,
            )
        elif distance_target_is_log:
            target_distance_space = torch.exp(target_distance)
        else:
            target_distance_space = target_distance
        pred_distance = pred["distance"][pred_slots].reshape(-1, 1)
        dist_cost = F.smooth_l1_loss(
            pred_distance.expand(-1, target_distance_space.shape[0]),
            target_distance_space.reshape(1, -1).expand(pred_distance.shape[0], -1),
            reduction="none",
        )
        cost = cost + 0.1 * dist_cost
    return cost


def threshold_sweep_metrics(
    pred: Dict[str, Tensor],
    targets: Sequence[Dict[str, Tensor]],
    thresholds: Sequence[float],
    kmax: Optional[int] = None,
    distance_target_is_log: bool = False,
) -> Dict[float, Dict[str, float]]:
    """Evaluate activity-threshold metrics using ||accdoa|| as activity.

    Args:
        pred: Dict with accdoa [B, K, 3], class_logits [B, K, C],
            distance [B, K, 1].
        targets: List of B target dicts with accdoa [N_i, 3].
        thresholds: Iterable of ACCDOA norm thresholds.
        kmax: Optional expected K. If provided, validates pred slot count.
        distance_target_is_log: Accepted for interface symmetry. Threshold
            sweep currently matches by angular cost only.

    Returns:
        Dict keyed by threshold value. Each value contains count_acc,
        precision, recall, f1, false_positive, miss, false_positive_rate,
        miss_rate, and matched_angular_mae.
    """
    del distance_target_is_log
    pred_accdoa = pred["accdoa"]
    if pred_accdoa.ndim != 3 or pred_accdoa.shape[-1] != 3:
        raise ValueError(f"pred['accdoa'] must be [B, K, 3], got {tuple(pred_accdoa.shape)}")
    batch_size, slot_count, _ = pred_accdoa.shape
    if kmax is not None and int(kmax) != slot_count:
        raise ValueError(f"expected kmax={kmax}, got K={slot_count}")
    if len(targets) != batch_size:
        raise ValueError("targets length must match batch size")

    activity = accdoa_activity(pred_accdoa)
    result: Dict[float, Dict[str, float]] = {}
    for threshold in thresholds:
        count_correct = 0
        total_pred = 0
        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        angular_errors: List[Tensor] = []
        for batch_index, target in enumerate(targets):
            active_slots = torch.nonzero(activity[batch_index] > float(threshold), as_tuple=False).flatten()
            target_accdoa = _as_tensor(target["accdoa"], pred_accdoa.device, pred_accdoa.dtype)
            if target_accdoa.ndim != 2 or target_accdoa.shape[-1] != 3:
                raise ValueError("target accdoa must be [N, 3]")
            pred_count = int(active_slots.numel())
            gt_count = int(target_accdoa.shape[0])
            count_correct += int(pred_count == gt_count)
            total_pred += pred_count
            total_gt += gt_count

            if pred_count > 0 and gt_count > 0:
                cost_matrix = _angular_cost_matrix(pred_accdoa[batch_index, active_slots], target_accdoa)
                pairs = _best_rectangular_assignment(cost_matrix)
            else:
                pairs = []
            matched_count = len(pairs)
            total_tp += matched_count
            total_fp += pred_count - matched_count
            total_fn += gt_count - matched_count
            for local_pred_index, gt_index in pairs:
                slot = active_slots[local_pred_index]
                angular_errors.append(angular_error_deg(pred_accdoa[batch_index, slot], target_accdoa[gt_index]))

        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        angular_mae = (
            float(torch.stack(angular_errors).mean().item())
            if angular_errors
            else float("nan")
        )
        result[float(threshold)] = {
            "count_acc": count_correct / max(batch_size, 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive": float(total_fp),
            "miss": float(total_fn),
            "false_positive_rate": total_fp / max(total_pred, 1),
            "miss_rate": total_fn / max(total_gt, 1),
            "matched_angular_mae": angular_mae,
        }
    return result


def topk_active_slots(accdoa: Tensor, k: int) -> Tensor:
    """Return top-k active slot indices by ||accdoa||.

    Args:
        accdoa: [B, K, 3] raw ACCDOA vectors.
        k: Number of slots to select.

    Returns:
        LongTensor [B, k] with selected slot indices.
    """
    if accdoa.ndim != 3 or accdoa.shape[-1] != 3:
        raise ValueError(f"accdoa must be [B, K, 3], got {tuple(accdoa.shape)}")
    if k <= 0 or k > accdoa.shape[1]:
        raise ValueError(f"k must be in [1, {accdoa.shape[1]}], got {k}")
    return torch.topk(accdoa_activity(accdoa), k=int(k), dim=-1).indices


def topk_matched_metrics(
    pred: Dict[str, Tensor],
    targets: Sequence[Dict[str, Tensor]],
    k: int,
    distance_target_is_log: bool = False,
    distance_is_log: bool = True,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Top-k matched metrics for fixed-N source sanity checks.

    Args:
        pred: Dict with accdoa [B, K, 3], class_logits [B, K, C],
            distance [B, K, 1].
        targets: List of B target dicts with accdoa [N_i, 3], class [N_i],
            distance [N_i, 1].
        k: Select top-k slots by ACCDOA norm for each sample.
        distance_target_is_log: Whether target distances are already log-space.
        distance_is_log: Whether predicted distance is log-space.
        eps: Numerical stability constant.

    Returns:
        Dict with topk_matched_angular_mae, topk_class_acc, topk_distance_mae.
    """
    selected = topk_active_slots(pred["accdoa"], k=k)
    batch_size = pred["accdoa"].shape[0]
    if len(targets) != batch_size:
        raise ValueError("targets length must match batch size")

    angular_errors: List[Tensor] = []
    class_correct: List[Tensor] = []
    distance_errors: List[Tensor] = []
    for batch_index, target in enumerate(targets):
        slots = selected[batch_index]
        target_accdoa = _as_tensor(target["accdoa"], pred["accdoa"].device, pred["accdoa"].dtype)
        if target_accdoa.shape[0] == 0:
            continue
        single_pred = {key: value[batch_index] for key, value in pred.items()}
        cost_matrix = _joint_cost_matrix(
            single_pred,
            slots,
            target,
            distance_target_is_log=distance_target_is_log,
            distance_is_log=distance_is_log,
            eps=eps,
        )
        pairs = _best_rectangular_assignment(cost_matrix)
        target_class = _as_tensor(target["class"], pred["accdoa"].device, torch.long)
        target_distance = _as_tensor(target["distance"], pred["accdoa"].device, pred["distance"].dtype)
        if target_distance.ndim == 1:
            target_distance = target_distance[:, None]
        if distance_is_log:
            target_distance_space = prepare_distance_target(
                target_distance,
                distance_target_is_log=distance_target_is_log,
                eps=eps,
            )
        elif distance_target_is_log:
            target_distance_space = torch.exp(target_distance)
        else:
            target_distance_space = target_distance

        pred_class = torch.argmax(pred["class_logits"][batch_index], dim=-1)
        for local_pred_index, gt_index in pairs:
            slot = slots[local_pred_index]
            angular_errors.append(angular_error_deg(pred["accdoa"][batch_index, slot], target_accdoa[gt_index]))
            class_correct.append((pred_class[slot] == target_class[gt_index]).float())
            pred_distance = pred["distance"][batch_index, slot]
            gt_distance = target_distance_space[gt_index]
            if distance_is_log:
                pred_distance = torch.exp(pred_distance)
                gt_distance = torch.exp(gt_distance)
            distance_errors.append((pred_distance - gt_distance).abs().mean())

    def _mean_or_nan(values: List[Tensor]) -> float:
        return float(torch.stack(values).mean().item()) if values else float("nan")

    return {
        "topk_matched_angular_mae": _mean_or_nan(angular_errors),
        "topk_class_acc": _mean_or_nan(class_correct),
        "topk_distance_mae": _mean_or_nan(distance_errors),
    }
