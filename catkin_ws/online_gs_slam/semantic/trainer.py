from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import HashGridSupervision
from .hash_grid import HashGrid4DConfig, MultiScaleHashGrid4D


@dataclass
class HashGridTrainingConfig:
    steps: int = 2000
    batch_size: int = 8192
    learning_rate: float = 1e-2
    weight_decay: float = 1e-6
    log_every: int = 100


def infer_bounds(supervision: HashGridSupervision, padding: float = 0.05) -> tuple[tuple[float, float, float], tuple[float, float, float], float, float]:
    xyz_min = supervision.xyz.min(dim=0).values
    xyz_max = supervision.xyz.max(dim=0).values
    extent = torch.clamp(xyz_max - xyz_min, min=1e-3)
    xyz_min = xyz_min - extent * padding
    xyz_max = xyz_max + extent * padding
    t_min = float(supervision.time.min().item()) if supervision.time.numel() else 0.0
    t_max = float(supervision.time.max().item()) if supervision.time.numel() else 1.0
    if abs(t_max - t_min) < 1e-6:
        t_max = t_min + 1.0
    return tuple(xyz_min.cpu().tolist()), tuple(xyz_max.cpu().tolist()), t_min, t_max


def train_hash_grid(
    supervision: HashGridSupervision,
    model_config: HashGrid4DConfig,
    train_config: HashGridTrainingConfig,
    output_path: str | Path,
) -> MultiScaleHashGrid4D:
    device = supervision.xyz.device
    model = MultiScaleHashGrid4D(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    n = supervision.xyz.shape[0]
    if n == 0:
        raise ValueError("Cannot train hash grid with zero samples")

    for step in range(1, train_config.steps + 1):
        batch = torch.randint(0, n, (min(train_config.batch_size, n),), device=device)
        logits = model(supervision.xyz[batch], supervision.time[batch])
        per_sample_loss = F.cross_entropy(logits, supervision.labels[batch], reduction="none")
        loss = (per_sample_loss * supervision.weights[batch]).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % train_config.log_every == 0 or step == train_config.steps:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == supervision.labels[batch]).float().mean().item()
            print(f"step={step:05d} loss={loss.item():.5f} batch_acc={acc:.3f}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "state_dict": model.state_dict(),
            "class_names": supervision.class_names,
        },
        output_path,
    )
    print(f"Wrote {output_path}")
    return model


def load_hash_grid_checkpoint(path: str | Path, device: str = "cpu") -> MultiScaleHashGrid4D:
    checkpoint = torch.load(path, map_location=device)
    model = MultiScaleHashGrid4D(HashGrid4DConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)
