from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import MultiACCDOALoss
from src.metrics import class_accuracy, distance_mae, matched_angular_mae
from src.synthetic_data import generate_synthetic_batch, generate_toy_inputs
from src.toy_model import ToyMultiSourceModel


def main() -> None:
    torch.manual_seed(13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    input_dim = 16
    hidden_dim = 64
    kmax = 3
    num_classes = 5
    steps = 300

    model = ToyMultiSourceModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        kmax=kmax,
    ).to(device)
    criterion = MultiACCDOALoss(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    # Fixed toy batch: this script is a pipeline sanity check, not a benchmark.
    x = generate_toy_inputs(batch_size, input_dim, device=device)
    targets = generate_synthetic_batch(
        batch_size=batch_size,
        num_classes=num_classes,
        kmax=kmax,
        min_sources=2,
        max_sources=2,
        distance_range=(1.0, 3.0),
        device=device,
    )

    first_loss = None
    last_loss = None
    print(f"device={device}, steps={steps}, batch_size={batch_size}")
    for step in range(steps + 1):
        pred = model(x)
        loss_out = criterion(pred, targets)
        loss = loss_out["loss"]
        if step == 0:
            first_loss = float(loss.detach().cpu())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

        if step % 50 == 0 or step == steps:
            with torch.no_grad():
                angular_mae = matched_angular_mae(pred, targets, loss_out["matched"])
                dist_err = distance_mae(pred, targets, loss_out["matched"])
                cls_acc = class_accuracy(pred, targets, loss_out["matched"])
            print(
                f"step={step:03d} "
                f"loss={float(loss.detach()):.5f} "
                f"acc_vec={float(loss_out['loss_acc_vec'].detach()):.5f} "
                f"acc_ang={float(loss_out['loss_acc_ang'].detach()):.5f} "
                f"cls={float(loss_out['loss_cls'].detach()):.5f} "
                f"dist={float(loss_out['loss_dist'].detach()):.5f} "
                f"ang_mae={float(angular_mae):.2f}deg "
                f"dist_mae={float(dist_err):.3f}m "
                f"cls_acc={float(cls_acc):.3f}"
            )

    assert first_loss is not None and last_loss is not None
    print(f"\nloss_start={first_loss:.5f}, loss_end={last_loss:.5f}")


if __name__ == "__main__":
    main()
