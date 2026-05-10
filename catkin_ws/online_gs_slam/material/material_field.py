from __future__ import annotations

import torch


class MaterialField:
    def encode_touch(self, tactile_observation: dict) -> torch.Tensor:
        # TODO: replace with learned tactile encoder.
        dim = int(tactile_observation.get("material_dim", 16))
        return torch.zeros((dim,), dtype=torch.float32)

