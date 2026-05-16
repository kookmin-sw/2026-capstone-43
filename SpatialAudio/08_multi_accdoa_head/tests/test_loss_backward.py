from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import MultiACCDOALoss
from src.synthetic_data import generate_synthetic_batch, generate_toy_inputs
from src.toy_model import ToyMultiSourceModel


def test_loss_backward_has_gradients() -> None:
    torch.manual_seed(7)
    batch_size = 4
    input_dim = 16
    hidden_dim = 64
    kmax = 3
    num_classes = 5

    model = ToyMultiSourceModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        kmax=kmax,
    )
    x = generate_toy_inputs(batch_size, input_dim)
    targets = generate_synthetic_batch(
        batch_size=batch_size,
        num_classes=num_classes,
        kmax=kmax,
        min_sources=2,
        max_sources=2,
    )
    criterion = MultiACCDOALoss(num_classes=num_classes)

    pred = model(x)
    loss_out = criterion(pred, targets)
    loss = loss_out["loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert model.slot_queries.grad is not None
    assert torch.isfinite(model.slot_queries.grad).all()
    assert model.head.head[-1].weight.grad is not None
    assert model.backbone[0].weight.grad is not None
