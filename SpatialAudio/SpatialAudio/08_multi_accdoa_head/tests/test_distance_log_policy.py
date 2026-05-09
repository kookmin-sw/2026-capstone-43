from __future__ import annotations

from pathlib import Path
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import MultiACCDOALoss
from src.pit import build_pit_targets, prepare_distance_target


def _single_source_pred() -> dict[str, torch.Tensor]:
    class_logits = torch.full((1, 3, 2), -5.0)
    class_logits[0, 0, 0] = 5.0
    return {
        "accdoa": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        "class_logits": class_logits,
        "distance": torch.zeros(1, 3, 1),
    }


def _target(distance_value: float) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "accdoa": torch.tensor([[1.0, 0.0, 0.0]]),
            "class": torch.tensor([0]),
            "distance": torch.tensor([[distance_value]], dtype=torch.float32),
        }
    ]


def test_prepare_distance_target_raw_meter_to_log() -> None:
    distance = torch.tensor([[1.0], [math.exp(2.0)]])
    prepared = prepare_distance_target(distance, distance_target_is_log=False)

    assert torch.allclose(prepared[0], torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(prepared[1], torch.tensor([2.0]), atol=1e-5)


def test_prepare_distance_target_already_log_is_not_logged_again() -> None:
    distance = torch.tensor([[2.0]])
    prepared = prepare_distance_target(distance, distance_target_is_log=True)

    assert torch.allclose(prepared, torch.tensor([[2.0]]), atol=1e-6)


def test_pit_stores_log_distance_target_for_raw_meter() -> None:
    pred = _single_source_pred()
    matched = build_pit_targets(
        pred,
        _target(math.exp(2.0)),
        num_classes=2,
        distance_target_is_log=False,
    )

    active_distance = matched["distance"][matched["active_mask"]]
    assert torch.allclose(active_distance, torch.tensor([[2.0]]), atol=1e-5)


def test_loss_and_pit_use_same_distance_policy() -> None:
    pred = _single_source_pred()
    raw_targets = _target(math.exp(2.0))
    matched = build_pit_targets(
        pred,
        raw_targets,
        num_classes=2,
        distance_target_is_log=False,
    )
    criterion = MultiACCDOALoss(num_classes=2, distance_target_is_log=False)
    loss_out = criterion(pred, raw_targets)

    assert torch.allclose(loss_out["matched"]["distance"], matched["distance"])
    assert torch.isfinite(loss_out["loss"])


def test_log_distance_target_is_used_as_is() -> None:
    pred = _single_source_pred()
    matched = build_pit_targets(
        pred,
        _target(2.0),
        num_classes=2,
        distance_target_is_log=True,
    )

    active_distance = matched["distance"][matched["active_mask"]]
    assert torch.allclose(active_distance, torch.tensor([[2.0]]), atol=1e-6)
