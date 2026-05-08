from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import threshold_sweep_metrics, topk_active_slots, topk_matched_metrics


def test_topk_active_slots_selects_largest_norms() -> None:
    accdoa = torch.tensor([[[0.1, 0.0, 0.0], [0.0, 0.8, 0.0], [0.9, 0.0, 0.0]]])
    selected = topk_active_slots(accdoa, k=2)

    assert set(selected[0].tolist()) == {1, 2}


def test_topk_matched_metrics_handles_swapped_slots() -> None:
    pred = {
        "accdoa": torch.tensor([[[0.1, 0.0, 0.0], [0.0, 0.8, 0.0], [0.9, 0.0, 0.0]]]),
        "class_logits": torch.tensor([[[0.0, 0.0], [-5.0, 5.0], [5.0, -5.0]]]),
        "distance": torch.log(torch.tensor([[[1.5], [2.0], [1.0]]])),
    }
    targets = [
        {
            "accdoa": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "class": torch.tensor([0, 1]),
            "distance": torch.tensor([[1.0], [2.0]]),
        }
    ]
    metrics = topk_matched_metrics(pred, targets, k=2)

    assert metrics["topk_matched_angular_mae"] < 1e-3
    assert metrics["topk_class_acc"] == 1.0
    assert metrics["topk_distance_mae"] < 1e-5


def test_topk_separates_direction_quality_from_threshold_calibration() -> None:
    pred = {
        "accdoa": torch.tensor([[[0.4, 0.0, 0.0], [0.0, 0.39, 0.0], [0.0, 0.0, 0.0]]]),
        "class_logits": torch.tensor([[[5.0, -5.0], [-5.0, 5.0], [0.0, 0.0]]]),
        "distance": torch.log(torch.tensor([[[1.0], [2.0], [1.5]]])),
    }
    targets = [
        {
            "accdoa": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "class": torch.tensor([0, 1]),
            "distance": torch.tensor([[1.0], [2.0]]),
        }
    ]

    threshold_metrics = threshold_sweep_metrics(pred, targets, thresholds=[0.5], kmax=3)
    topk_metrics = topk_matched_metrics(pred, targets, k=2)

    assert threshold_metrics[0.5]["recall"] == 0.0
    assert topk_metrics["topk_matched_angular_mae"] < 1e-3
