from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DatasetGenerationConfig
from .schemas import SampleMetadata


def ensure_sample_layout(dataset_root: Path, scene_id: str, sample_id: str) -> dict[str, Path]:
    sample_root = dataset_root / "scenes" / scene_id / "samples" / sample_id
    layout = {
        "sample_root": sample_root,
        "audio_dir": sample_root / "audio",
        "image_dir": sample_root / "image",
        "metadata_dir": sample_root / "metadata",
        "optional_dir": sample_root / "optional",
        "audio_wav": sample_root / "audio" / "foa.wav",
        "audio_wav_librispeech": sample_root / "audio" / "mic_librispeech.wav",
        "rgb_png": sample_root / "image" / "rgb_front.png",
        "metadata_json": sample_root / "metadata" / "sample.json",
        "depth_png": sample_root / "optional" / "depth.png",
        "depth_npy": sample_root / "optional" / "depth.npy",
        "semantic_png": sample_root / "optional" / "semantic_preview.png",
        "instance_mask_npy": sample_root / "optional" / "instance_mask.npy",
        "topdown_debug_png": sample_root / "optional" / "topdown_debug.png",
        "front_overlay_png": sample_root / "optional" / "front_overlay.png",
        "rir_npy": sample_root / "optional" / "rir.npy",
        "rir_wav": sample_root / "optional" / "rir.wav",
    }
    for key in ("audio_dir", "image_dir", "metadata_dir", "optional_dir"):
        layout[key].mkdir(parents=True, exist_ok=True)
    return layout


def sample_is_complete(layout: dict[str, Path], config: DatasetGenerationConfig) -> bool:
    required_paths = [
        layout["audio_wav"],
        layout["metadata_json"],
    ]
    if config.sensor_rig.write_rgb_png:
        required_paths.append(layout["rgb_png"])
    if config.sensor_rig.enable_depth and config.sensor_rig.write_depth_npy:
        required_paths.append(layout["depth_npy"])
    if config.sensor_rig.enable_semantic and config.sensor_rig.write_instance_mask_npy:
        required_paths.append(layout["instance_mask_npy"])
    return all(path.exists() for path in required_paths)


def output_file_map(dataset_root: Path, layout: dict[str, Path]) -> dict[str, str]:
    return {
        "audio_foa_wav": str(layout["audio_wav"].relative_to(dataset_root)),
        "audio_mic_wav": str(layout["audio_wav"].relative_to(dataset_root)),
        "audio_mic_librispeech_wav": str(layout["audio_wav_librispeech"].relative_to(dataset_root))
        if layout["audio_wav_librispeech"].exists()
        else "",
        "rgb_front_png": str(layout["rgb_png"].relative_to(dataset_root))
        if layout["rgb_png"].exists()
        else "",
        "metadata_json": str(layout["metadata_json"].relative_to(dataset_root)),
        "topdown_debug_png": str(layout["topdown_debug_png"].relative_to(dataset_root))
        if layout["topdown_debug_png"].exists()
        else "",
        "front_overlay_png": str(layout["front_overlay_png"].relative_to(dataset_root))
        if layout["front_overlay_png"].exists()
        else "",
        "depth_png": str(layout["depth_png"].relative_to(dataset_root))
        if layout["depth_png"].exists()
        else "",
        "depth_npy": str(layout["depth_npy"].relative_to(dataset_root))
        if layout["depth_npy"].exists()
        else "",
        "semantic_png": str(layout["semantic_png"].relative_to(dataset_root))
        if layout["semantic_png"].exists()
        else "",
        "instance_mask_npy": str(layout["instance_mask_npy"].relative_to(dataset_root))
        if layout["instance_mask_npy"].exists()
        else "",
        "rir_npy": str(layout["rir_npy"].relative_to(dataset_root))
        if layout["rir_npy"].exists()
        else "",
        "rir_wav": str(layout["rir_wav"].relative_to(dataset_root))
        if layout["rir_wav"].exists()
        else "",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_sample_metadata(path: Path, metadata: SampleMetadata) -> None:
    write_json(path, metadata.to_dict())
