"""Minimal Multi-ACCDOA slot-head experiment package."""

from .heads import JointMultiSourceHead, decode_accdoa
from .losses import MultiACCDOALoss
from .toy_model import ToyMultiSourceModel

__all__ = [
    "JointMultiSourceHead",
    "MultiACCDOALoss",
    "ToyMultiSourceModel",
    "decode_accdoa",
]
