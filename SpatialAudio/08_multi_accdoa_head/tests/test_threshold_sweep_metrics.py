from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import threshold_sweep_metrics


def test_threshold_sweep_counts_and_matching() -> None:
    pred = {
        "accdoa": torch.tensor([[[0.9, 0.0, 0.0], [0.0, 0.8, 0.0], [0.1, 0.0, 0.0]]]),
        "class_logits": torch.zeros(1, 3, 2),
        "distance": torch.zeros(1, 3, 1),
    }
    targets = [
        {
            "accdoa": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "class": torch.tensor([0, 1]),
            "distance": torch.ones(2, 1),
        }
    ]
    metrics = threshold_sweep_metrics(pred, targets, thresholds=[0.05, 0.5, 0.85], kmax=3)

    assert {"count_acc", "precision", "recall", "f1"}.issubset(metrics[0.5].keys())
    assert metrics[0.5]["count_acc"] == 1.0
    assert metrics[0.5]["precision"] == 1.0
    assert metrics[0.5]["recall"] == 1.0
    assert metrics[0.5]["matched_angular_mae"] < 1e-3
    assert metrics[0.85]["recall"] < metrics[0.5]["recall"]
    assert metrics[0.05]["false_positive"] > metrics[0.5]["false_positive"]
