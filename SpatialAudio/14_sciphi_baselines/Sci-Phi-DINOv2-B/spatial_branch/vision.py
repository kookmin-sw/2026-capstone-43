"""Vision branch helpers for frozen DINOv2 integration."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


DINO_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINO_IMAGE_STD = (0.229, 0.224, 0.225)
DEFAULT_DINO_MODEL_NAME = "dinov2_vitb14"


def build_dinov2_image_transform(image_size: int = 518) -> Callable[[Image.Image], torch.Tensor]:
    if int(image_size) <= 0:
        raise ValueError(f"image_size must be > 0, got {image_size}")

    def _transform(image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = TF.resize(
            image,
            size=[int(image_size), int(image_size)],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, mean=DINO_IMAGE_MEAN, std=DINO_IMAGE_STD)

    return _transform


class LoRALinear(nn.Module):
    """Frozen linear layer with trainable low-rank residual."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)!r}")
        if int(rank) <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.rank = int(rank)
        self.alpha = int(alpha) if alpha is not None else int(rank)
        self.scaling = float(self.alpha) / float(self.rank)

        self.weight = nn.Parameter(base_layer.weight.detach().clone(), requires_grad=False)
        if base_layer.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False)

        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + (update * self.scaling)


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _set_child_module(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if child_name.isdigit():
        parent[int(child_name)] = module  # type: ignore[index]
        return
    setattr(parent, child_name, module)


def _is_dinov2_block_linear(module_name: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) and (
        module_name.startswith("blocks.") or ".blocks." in module_name
    )


def apply_dinov2_lora(
    backbone: nn.Module,
    rank: int,
    alpha: Optional[int] = None,
) -> Dict[str, Any]:
    wrapped_names: List[str] = []
    for module_name, module in list(backbone.named_modules()):
        if not _is_dinov2_block_linear(module_name, module):
            continue
        parent, child_name = _get_parent_module(backbone, module_name)
        _set_child_module(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
        wrapped_names.append(module_name)

    if not wrapped_names:
        raise RuntimeError("No DINOv2 transformer block linear layers were wrapped with LoRA.")

    trainable_params = 0
    for name, param in backbone.named_parameters():
        param.requires_grad = ".lora_" in name
        if param.requires_grad:
            trainable_params += int(param.numel())

    return {
        "rank": int(rank),
        "alpha": int(alpha) if alpha is not None else int(rank),
        "wrapped_modules": tuple(wrapped_names),
        "wrapped_module_count": len(wrapped_names),
        "trainable_lora_params": trainable_params,
    }


def load_dinov2_backbone(model_name: str = DEFAULT_DINO_MODEL_NAME) -> nn.Module:
    return torch.hub.load(
        "facebookresearch/dinov2",
        model_name,
        pretrained=True,
        trust_repo=True,
    )


class FrozenDinoV2VisionBranch(nn.Module):
    """DINOv2 encoder with optional LoRA and lightweight trainable projectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_DINO_MODEL_NAME,
        image_size: int = 518,
        lora_rank: int = 32,
        projector_output_dim: int = 192,
        token_output_dim: Optional[int] = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.projector_output_dim = int(projector_output_dim)
        self.token_output_dim = int(token_output_dim) if token_output_dim is not None else int(projector_output_dim)
        self.freeze_backbone = bool(freeze_backbone)

        self.backbone = load_dinov2_backbone(self.model_name)
        self.transform = build_dinov2_image_transform(self.image_size)
        self.backbone_dim = self._infer_backbone_dim()

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if int(lora_rank) > 0:
            self.lora_info = apply_dinov2_lora(self.backbone, rank=lora_rank, alpha=lora_rank)
        else:
            self.lora_info = {
                "rank": 0,
                "alpha": 0,
                "wrapped_modules": tuple(),
                "wrapped_module_count": 0,
                "trainable_lora_params": 0,
            }
        self.vision_projection = nn.Linear(self.backbone_dim, self.projector_output_dim)
        self.token_projection = (
            nn.Identity()
            if self.token_output_dim == self.projector_output_dim
            else nn.Linear(self.projector_output_dim, self.token_output_dim)
        )

    def _infer_backbone_dim(self) -> int:
        candidates = (
            getattr(self.backbone, "embed_dim", None),
            getattr(self.backbone, "num_features", None),
            getattr(getattr(self.backbone, "head", None), "in_features", None),
        )
        for candidate in candidates:
            if isinstance(candidate, int) and candidate > 0:
                return int(candidate)
        raise RuntimeError("Could not infer DINOv2 backbone feature dimension.")

    def _extract_backbone_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(pixel_values)
        if isinstance(features, dict):
            if isinstance(features.get("x_norm_clstoken"), torch.Tensor):
                return features["x_norm_clstoken"]
            if isinstance(features.get("x_clstoken"), torch.Tensor):
                return features["x_clstoken"]
            if isinstance(features.get("x_norm_patchtokens"), torch.Tensor):
                return features["x_norm_patchtokens"].mean(dim=1)
            tensor_values = [value for value in features.values() if isinstance(value, torch.Tensor)]
            if tensor_values:
                features = tensor_values[0]
            else:
                raise RuntimeError("DINOv2 forward_features returned no tensor values.")
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Unexpected DINOv2 feature type: {type(features)!r}")
        if features.ndim == 3:
            return features[:, 0, :]
        if features.ndim != 2:
            raise RuntimeError(f"Unexpected DINOv2 feature shape: {tuple(features.shape)}")
        return features

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        if pixel_values.ndim != 4:
            raise ValueError(
                f"FrozenDinoV2VisionBranch expects [B, 3, H, W], got {tuple(pixel_values.shape)}"
            )

        backbone_features = self._extract_backbone_features(pixel_values)
        multimodal_features = self.vision_projection(backbone_features)
        token_features = self.token_projection(multimodal_features)
        return {
            "multimodal_features": multimodal_features,
            "token_features": token_features.unsqueeze(1),
        }

    def get_trainable_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state = OrderedDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param.detach().cpu().contiguous()
        return state

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        load_result = self.load_state_dict(state_dict, strict=False)
        return {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }

    def trainable_parameter_summary(self, prefix: str = "vision_branch") -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                summary.append(
                    {
                        "name": f"{prefix}.{name}",
                        "count": int(param.numel()),
                        "shape": tuple(param.shape),
                    }
                )
        return summary


__all__ = [
    "DEFAULT_DINO_MODEL_NAME",
    "DINO_IMAGE_MEAN",
    "DINO_IMAGE_STD",
    "FrozenDinoV2VisionBranch",
    "LoRALinear",
    "apply_dinov2_lora",
    "build_dinov2_image_transform",
    "load_dinov2_backbone",
]
