from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def discover_images(input_path: Path) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(
                f"Unsupported image file: {input_path}. "
                f"Supported extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )
        return [input_path]

    image_paths = sorted(path for path in input_path.iterdir() if is_image_file(path))
    if not image_paths:
        raise FileNotFoundError(
            f"No jpg/jpeg/png images found in directory: {input_path}"
        )
    return image_paths


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def save_rgb_image(rgb: np.ndarray, output_path: Path) -> Path:
    output_path = output_path.resolve()
    ensure_dir(output_path.parent)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(output_path)
    return output_path


def save_numpy(array: np.ndarray, output_path: Path) -> Path:
    output_path = output_path.resolve()
    ensure_dir(output_path.parent)
    np.save(output_path, array)
    return output_path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable.")


def save_json(data: dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.resolve()
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True, default=_json_default)
    return output_path


def sanitize_stem(text: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    sanitized = sanitized.strip("_")
    return sanitized or "image"


def make_image_output_dir(output_root: Path, image_path: Path, index: int) -> Path:
    safe_stem = sanitize_stem(image_path.stem)
    return ensure_dir(output_root / f"{index:04d}_{safe_stem}")
