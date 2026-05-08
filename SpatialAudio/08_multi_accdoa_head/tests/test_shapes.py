from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.heads import JointMultiSourceHead, decode_accdoa


def test_joint_head_shapes() -> None:
    batch_size = 4
    kmax = 3
    num_classes = 5
    hidden_dim = 64
    slot_tokens = torch.randn(batch_size, kmax, hidden_dim)
    head = JointMultiSourceHead(hidden_dim=hidden_dim, num_classes=num_classes, kmax=kmax)

    out = head(slot_tokens)

    assert out["accdoa"].shape == (batch_size, kmax, 3)
    assert out["class_logits"].shape == (batch_size, kmax, num_classes)
    assert out["distance"].shape == (batch_size, kmax, 1)


def test_decode_shapes() -> None:
    batch_size = 4
    kmax = 3
    num_classes = 5
    out = {
        "accdoa": torch.randn(batch_size, kmax, 3),
        "class_logits": torch.randn(batch_size, kmax, num_classes),
        "distance": torch.randn(batch_size, kmax, 1),
    }

    decoded = decode_accdoa(**out)

    assert decoded["activity"].shape == (batch_size, kmax)
    assert decoded["active_mask"].shape == (batch_size, kmax)
    assert decoded["direction"].shape == (batch_size, kmax, 3)
    assert decoded["class_id"].shape == (batch_size, kmax)
    assert decoded["distance"].shape == (batch_size, kmax, 1)
