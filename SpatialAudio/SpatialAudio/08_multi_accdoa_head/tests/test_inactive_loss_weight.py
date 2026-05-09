from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import MultiACCDOALoss


def _pred_with_inactive(nonzero_inactive: bool) -> dict[str, torch.Tensor]:
    inactive_value = 0.5 if nonzero_inactive else 0.0
    class_logits = torch.full((1, 3, 2), -5.0)
    class_logits[0, 0, 0] = 5.0
    return {
        "accdoa": torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [inactive_value, 0.0, 0.0],
                    [0.0, inactive_value, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
        "class_logits": class_logits,
        "distance": torch.zeros(1, 3, 1),
    }


def _one_source_target() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "accdoa": torch.tensor([[1.0, 0.0, 0.0]]),
            "class": torch.tensor([0]),
            "distance": torch.tensor([[1.0]]),
        }
    ]


def test_nonzero_inactive_slot_has_positive_inactive_loss() -> None:
    criterion = MultiACCDOALoss(num_classes=2)
    loss_out = criterion(_pred_with_inactive(nonzero_inactive=True), _one_source_target())

    assert loss_out["loss_acc_inactive"].item() > 0.0


def test_zero_inactive_slot_has_near_zero_inactive_loss() -> None:
    criterion = MultiACCDOALoss(num_classes=2)
    loss_out = criterion(_pred_with_inactive(nonzero_inactive=False), _one_source_target())

    assert loss_out["loss_acc_inactive"].item() < 1e-8


def test_larger_inactive_weight_increases_total_loss() -> None:
    pred = _pred_with_inactive(nonzero_inactive=True)
    targets = _one_source_target()
    low = MultiACCDOALoss(num_classes=2, lambda_acc_inactive=0.1)(pred, targets)["loss"]
    high = MultiACCDOALoss(num_classes=2, lambda_acc_inactive=10.0)(pred, targets)["loss"]

    assert high.item() > low.item()


def test_no_inactive_slots_is_finite() -> None:
    pred = {
        "accdoa": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
        "class_logits": torch.tensor([[[10.0, -1.0], [-1.0, 10.0], [10.0, -1.0]]]),
        "distance": torch.zeros(1, 3, 1),
    }
    targets = [
        {
            "accdoa": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "class": torch.tensor([0, 1, 0]),
            "distance": torch.ones(3, 1),
        }
    ]
    loss_out = MultiACCDOALoss(num_classes=2)(pred, targets)

    assert torch.isfinite(loss_out["loss"])
    assert loss_out["loss_acc_inactive"].item() == 0.0


def test_no_active_sources_is_finite() -> None:
    pred = {
        "accdoa": torch.zeros(1, 3, 3),
        "class_logits": torch.zeros(1, 3, 2),
        "distance": torch.zeros(1, 3, 1),
    }
    targets = [
        {
            "accdoa": torch.empty(0, 3),
            "class": torch.empty(0, dtype=torch.long),
            "distance": torch.empty(0, 1),
        }
    ]
    loss_out = MultiACCDOALoss(num_classes=2)(pred, targets)

    assert torch.isfinite(loss_out["loss"])
    assert loss_out["loss_cls"].item() == 0.0
    assert loss_out["loss_dist"].item() == 0.0
