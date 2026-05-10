from collections.abc import Iterable

import torch
import torch.nn as nn


def to_2tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(f"Expected length-2 list, got {value}")
        return tuple(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        value = tuple(value)
        if len(value) != 2:
            raise ValueError(f"Expected length-2 iterable, got {value}")
        return value
    return (value, value)


def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(
            x,
            drop_prob=self.drop_prob,
            training=self.training,
            scale_by_keep=self.scale_by_keep,
        )


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
