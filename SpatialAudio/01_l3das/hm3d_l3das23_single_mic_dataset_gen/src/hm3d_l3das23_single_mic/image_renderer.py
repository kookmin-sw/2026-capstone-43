from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import DatasetGenerationConfig
from .schemas import ImageRenderResult, MicPose, SpeakerProxyPose


def _load_imageio():
    import imageio.v2 as imageio

    return imageio


def _depth_to_png(depth_image: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_image, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16)


def _semantic_to_rgb(semantic_image: np.ndarray) -> np.ndarray:
    semantic = np.asarray(semantic_image, dtype=np.int64)
    semantic = np.clip(semantic, 0, 2**31 - 1)
    r = (semantic * 37) % 255
    g = (semantic * 67) % 255
    b = (semantic * 97) % 255
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def render_sample_images(
    session: Any,
    mic_pose: MicPose,
    layout: dict[str, Path],
    config: DatasetGenerationConfig,
    speaker_proxy_pose: Optional[SpeakerProxyPose] = None,
) -> tuple[ImageRenderResult, dict[str, np.ndarray]]:
    imageio = _load_imageio()
    observations = session.render_visual_observations(
        mic_pose,
        speaker_proxy_pose=speaker_proxy_pose,
    )
    rgb = np.asarray(observations["rgb_sensor"], dtype=np.uint8)[..., :3]
    if config.sensor_rig.write_rgb_png:
        imageio.imwrite(layout["rgb_png"], rgb)

    depth = None
    if config.sensor_rig.enable_depth and "depth_sensor" in observations:
        depth = np.asarray(observations["depth_sensor"], dtype=np.float32)
        if config.sensor_rig.write_depth_png:
            imageio.imwrite(layout["depth_png"], _depth_to_png(depth))
        if config.sensor_rig.write_depth_npy:
            np.save(layout["depth_npy"], depth.astype(np.float32))

    semantic = None
    if config.sensor_rig.enable_semantic and "semantic_sensor" in observations:
        semantic = np.asarray(observations["semantic_sensor"], dtype=np.int32)
        if config.sensor_rig.write_semantic_preview_png:
            imageio.imwrite(layout["semantic_png"], _semantic_to_rgb(semantic))
        if config.sensor_rig.write_instance_mask_npy:
            np.save(layout["instance_mask_npy"], semantic.astype(np.int32))

    return (
        ImageRenderResult(
            rgb_path=layout["rgb_png"],
            depth_path=(
                layout["depth_png"]
                if depth is not None and config.sensor_rig.write_depth_png
                else None
            ),
            depth_npy_path=(
                layout["depth_npy"]
                if depth is not None and config.sensor_rig.write_depth_npy
                else None
            ),
            instance_mask_path=(
                layout["instance_mask_npy"]
                if semantic is not None and config.sensor_rig.write_instance_mask_npy
                else None
            ),
            semantic_preview_path=(
                layout["semantic_png"]
                if semantic is not None and config.sensor_rig.write_semantic_preview_png
                else None
            ),
        ),
        {"rgb": rgb, "depth": depth, "semantic": semantic},
    )
