from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import MultiACCDOALoss
from src.metrics import angular_error_deg, decode_predictions


def main() -> None:
    torch.manual_seed(0)
    num_classes = 4
    front = torch.tensor([1.0, 0.0, 0.0])
    left = torch.tensor([0.0, 1.0, 0.0])

    pred = {
        "accdoa": torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0],  # slot0 = left
                    [0.0, 0.0, 0.0],  # slot1 = inactive
                    [1.0, 0.0, 0.0],  # slot2 = front
                ]
            ],
            dtype=torch.float32,
        ),
        "class_logits": torch.full((1, 3, num_classes), -10.0),
        "distance": torch.log(torch.tensor([[[2.0], [1.5], [1.0]]], dtype=torch.float32)),
    }
    pred["class_logits"][0, 0, 1] = 10.0
    pred["class_logits"][0, 1, 3] = 10.0
    pred["class_logits"][0, 2, 0] = 10.0

    targets = [
        {
            "accdoa": torch.stack([front, left], dim=0),
            "class": torch.tensor([0, 1], dtype=torch.long),
            "distance": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        }
    ]

    criterion = MultiACCDOALoss(num_classes=num_classes)
    loss_out = criterion(pred, targets)
    matched = loss_out["matched"]
    decoded = decode_predictions(pred, activity_threshold=0.5)

    print("GT sources:")
    for index, direction in enumerate(targets[0]["accdoa"]):
        print(
            f"  gt{index}: dir={direction.tolist()} "
            f"class={int(targets[0]['class'][index])} "
            f"distance_m={float(targets[0]['distance'][index])}"
        )

    print("\nPredicted slots:")
    for slot in range(pred["accdoa"].shape[1]):
        print(
            f"  slot{slot}: accdoa={pred['accdoa'][0, slot].tolist()} "
            f"activity={float(decoded['activity'][0, slot]):.3f} "
            f"class={int(decoded['class_id'][0, slot])} "
            f"distance_m={float(decoded['distance'][0, slot]):.3f}"
        )

    print("\nSelected PIT assignment:")
    assigned_slots = {slot for slot, _ in matched["assignments"][0]}
    for slot, gt_index in matched["assignments"][0]:
        err = angular_error_deg(pred["accdoa"][0, slot], targets[0]["accdoa"][gt_index])
        print(f"  slot{slot} -> gt{gt_index}, angular_error={float(err):.3f} deg")
    inactive = [slot for slot in range(pred["accdoa"].shape[1]) if slot not in assigned_slots]
    print(f"  inactive slots: {inactive}")
    print(f"\nTotal loss: {float(loss_out['loss']):.8f}")


if __name__ == "__main__":
    main()
