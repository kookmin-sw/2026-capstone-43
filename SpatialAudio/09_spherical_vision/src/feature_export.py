from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import save_json, save_numpy
from .spherical_projection import CHANNEL_DESCRIPTIONS, FeatureBundle

LOGGER = logging.getLogger(__name__)


def validate_feature_bundle(bundle: FeatureBundle) -> None:
    if not bundle.channel_names:
        raise ValueError("Feature bundle has no channel names.")

    expected_shape = None
    for channel_name in bundle.channel_names:
        if channel_name not in bundle.channels:
            raise KeyError(f"Missing channel '{channel_name}' in feature bundle.")
        current_shape = bundle.channels[channel_name].shape
        if expected_shape is None:
            expected_shape = current_shape
        elif current_shape != expected_shape:
            raise ValueError(
                f"Channel shape mismatch for '{channel_name}': {current_shape} vs expected {expected_shape}."
            )


def build_channels_json(channel_names: list[str]) -> dict[str, Any]:
    return {
        "channel_names": list(channel_names),
        "channel_count": len(channel_names),
        "channels": [
            {
                "index": index,
                "name": channel_name,
                "description": CHANNEL_DESCRIPTIONS.get(channel_name, ""),
            }
            for index, channel_name in enumerate(channel_names)
        ],
    }


def save_feature_bundle(
    output_dir: Path,
    full_bundle: FeatureBundle,
    azimuth_bundle: FeatureBundle,
    meta: dict[str, Any],
    export_pt: bool = False,
) -> dict[str, str | bool]:
    validate_feature_bundle(full_bundle)
    validate_feature_bundle(azimuth_bundle)

    full_tensor = full_bundle.tensor().astype(np.float32)
    azimuth_tensor = azimuth_bundle.tensor().astype(np.float32)

    save_numpy(full_tensor, output_dir / "vision_sphere.npy")
    save_numpy(azimuth_tensor, output_dir / "vision_sphere_azimuth.npy")
    save_json(build_channels_json(full_bundle.channel_names), output_dir / "vision_sphere_channels.json")
    save_json(meta, output_dir / "vision_sphere_meta.json")

    torch_exported = False
    torch_export_path = output_dir / "vision_sphere.pt"
    if export_pt:
        try:
            import torch

            payload = {
                "tensor": torch.from_numpy(full_tensor),
                "channel_names": list(full_bundle.channel_names),
                "meta": meta,
            }
            torch.save(payload, torch_export_path)
            torch_exported = True
        except Exception as exc:  # pragma: no cover - optional export path
            LOGGER.warning("Failed to export vision_sphere.pt: %s", exc)

    return {
        "vision_sphere": str((output_dir / "vision_sphere.npy").resolve()),
        "vision_sphere_azimuth": str((output_dir / "vision_sphere_azimuth.npy").resolve()),
        "vision_sphere_channels": str((output_dir / "vision_sphere_channels.json").resolve()),
        "vision_sphere_meta": str((output_dir / "vision_sphere_meta.json").resolve()),
        "vision_sphere_pt": str(torch_export_path.resolve()) if torch_exported else "",
        "vision_sphere_pt_exported": torch_exported,
    }
