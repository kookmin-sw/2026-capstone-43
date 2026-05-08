from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import ensure_dir, write_json
from .spherical_projection import CHANNEL_DESCRIPTIONS


def validate_audio_tensor(tensor: np.ndarray, channel_names: list[str], name: str = "audio_sphere") -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [E,A,C], got {tensor.shape}.")
    if tensor.shape[-1] != len(channel_names):
        raise ValueError(f"{name} channel mismatch: tensor C={tensor.shape[-1]} names={len(channel_names)}.")
    if not np.all(np.isfinite(tensor)):
        raise ValueError(f"{name} contains non-finite values.")


def validate_azimuth_tensor(tensor: np.ndarray, channel_names: list[str], name: str = "audio_sphere_azimuth") -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [A,C], got {tensor.shape}.")
    if tensor.shape[-1] != len(channel_names):
        raise ValueError(f"{name} channel mismatch: tensor C={tensor.shape[-1]} names={len(channel_names)}.")
    if not np.all(np.isfinite(tensor)):
        raise ValueError(f"{name} contains non-finite values.")


def save_audio_sphere(
    output_dir: str | Path,
    tensor: np.ndarray,
    azimuth_tensor: np.ndarray,
    channel_names: list[str],
    meta: dict[str, Any],
    export_pt: bool = False,
    tensor_max: np.ndarray | None = None,
    azimuth_tensor_max: np.ndarray | None = None,
) -> None:
    output_dir = ensure_dir(output_dir)
    validate_audio_tensor(tensor, channel_names)
    validate_azimuth_tensor(azimuth_tensor, channel_names)

    np.save(output_dir / "audio_sphere.npy", tensor.astype(np.float32))
    np.save(output_dir / "audio_sphere_azimuth.npy", azimuth_tensor.astype(np.float32))

    if tensor_max is not None:
        validate_audio_tensor(tensor_max, channel_names, name="audio_sphere_max")
        np.save(output_dir / "audio_sphere_max.npy", tensor_max.astype(np.float32))
    if azimuth_tensor_max is not None:
        validate_azimuth_tensor(azimuth_tensor_max, channel_names, name="audio_sphere_azimuth_max")
        np.save(output_dir / "audio_sphere_azimuth_max.npy", azimuth_tensor_max.astype(np.float32))

    write_json(
        output_dir / "audio_sphere_channels.json",
        {
            "channel_names": list(channel_names),
            "channel_descriptions": {name: CHANNEL_DESCRIPTIONS.get(name, "") for name in channel_names},
        },
    )
    write_json(output_dir / "audio_sphere_meta.json", meta)

    if export_pt:
        try:
            import torch
        except ImportError:
            print("[EXPORT] torch is not installed; skipped audio_sphere.pt.")
            return
        payload = {
            "tensor": torch.from_numpy(tensor.astype(np.float32)),
            "azimuth_tensor": torch.from_numpy(azimuth_tensor.astype(np.float32)),
            "channel_names": list(channel_names),
            "meta": meta,
        }
        if tensor_max is not None:
            payload["tensor_max"] = torch.from_numpy(tensor_max.astype(np.float32))
        if azimuth_tensor_max is not None:
            payload["azimuth_tensor_max"] = torch.from_numpy(azimuth_tensor_max.astype(np.float32))
        torch.save(payload, output_dir / "audio_sphere.pt")

