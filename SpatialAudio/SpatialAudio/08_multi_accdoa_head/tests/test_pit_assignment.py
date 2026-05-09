from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import MultiACCDOALoss
from src.pit import IGNORE_INDEX, build_pit_targets


def _swapped_prediction() -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    # Coordinate convention: [x_front, y_left, z_up].
    front = torch.tensor([1.0, 0.0, 0.0])
    left = torch.tensor([0.0, 1.0, 0.0])
    num_classes = 4

    pred_accdoa = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0],  # slot0 predicts source B = left
                [0.0, 0.0, 0.0],  # slot1 inactive
                [1.0, 0.0, 0.0],  # slot2 predicts source A = front
            ]
        ],
        dtype=torch.float32,
    )
    class_logits = torch.full((1, 3, num_classes), -10.0)
    class_logits[0, 0, 1] = 10.0
    class_logits[0, 1, 3] = 10.0
    class_logits[0, 2, 0] = 10.0
    distance = torch.log(torch.tensor([[[2.0], [1.5], [1.0]]], dtype=torch.float32))

    targets = [
        {
            "accdoa": torch.stack([front, left], dim=0),
            "class": torch.tensor([0, 1], dtype=torch.long),
            "distance": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        }
    ]
    pred = {
        "accdoa": pred_accdoa,
        "class_logits": class_logits,
        "distance": distance,
    }
    return pred, targets


def test_exhaustive_pit_recovers_swapped_slots() -> None:
    pred, targets = _swapped_prediction()
    matched = build_pit_targets(pred, targets, num_classes=4, distance_is_log=True)

    assert matched["assignments"][0] == [(0, 1), (2, 0)]
    assert matched["active_mask"].tolist() == [[True, False, True]]
    assert matched["class"].tolist() == [[1, IGNORE_INDEX, 0]]
    assert torch.allclose(matched["accdoa"][0, 0], targets[0]["accdoa"][1])
    assert torch.allclose(matched["accdoa"][0, 1], torch.zeros(3))
    assert torch.allclose(matched["accdoa"][0, 2], targets[0]["accdoa"][0])


def test_pit_loss_is_low_and_better_than_fixed_order() -> None:
    pred, targets = _swapped_prediction()
    criterion = MultiACCDOALoss(num_classes=4)
    out = criterion(pred, targets)
    pit_loss = out["loss"]

    # Naive fixed matching assumes slot0->source A, slot1->source B, slot2 inactive.
    naive_accdoa = torch.zeros_like(pred["accdoa"])
    naive_accdoa[0, 0] = targets[0]["accdoa"][0]
    naive_accdoa[0, 1] = targets[0]["accdoa"][1]
    naive_class = torch.tensor([[0, 1, IGNORE_INDEX]], dtype=torch.long)
    naive_distance = torch.zeros_like(pred["distance"])
    naive_distance[0, 0] = torch.log(targets[0]["distance"][0])
    naive_distance[0, 1] = torch.log(targets[0]["distance"][1])
    active = naive_class != IGNORE_INDEX

    naive_vec = F.smooth_l1_loss(pred["accdoa"], naive_accdoa)
    naive_cls = F.cross_entropy(pred["class_logits"][active], naive_class[active])
    naive_dist = F.smooth_l1_loss(pred["distance"][active], naive_distance[active])
    naive_loss = naive_vec + 0.5 * naive_cls + 0.2 * naive_dist

    assert torch.isfinite(pit_loss)
    assert pit_loss.item() < 1e-3
    assert pit_loss.item() < naive_loss.item()
