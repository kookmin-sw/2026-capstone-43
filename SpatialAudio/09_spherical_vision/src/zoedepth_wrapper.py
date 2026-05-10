from __future__ import annotations

import importlib.metadata as importlib_metadata
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthPrediction:
    depth: np.ndarray
    metadata: dict[str, Any]


def _install_transformers_hub_version_patch() -> None:
    original_version = importlib_metadata.version
    if getattr(importlib_metadata.version, "_zoedepth_patch_installed", False):
        return

    def patched_version(distribution_name: str) -> str:
        version = original_version(distribution_name)
        if distribution_name == "huggingface-hub":
            major = int(version.split(".", maxsplit=1)[0])
            if major >= 1:
                return "0.36.0"
        return version

    patched_version._zoedepth_patch_installed = True  # type: ignore[attr-defined]
    importlib_metadata.version = patched_version


def _load_transformers_zoe_classes() -> tuple[type[Any], type[Any], Any, Any]:
    try:
        from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation

        import torch
        import torch.nn.functional as torch_functional

        return AutoImageProcessor, ZoeDepthForDepthEstimation, torch, torch_functional
    except ImportError as exc:
        message = str(exc)
        if "huggingface-hub" not in message:
            raise
        LOGGER.warning(
            "transformers import failed because of a huggingface-hub version check. "
            "Applying a local compatibility patch and retrying."
        )
        _install_transformers_hub_version_patch()
        from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation

        import torch
        import torch.nn.functional as torch_functional

        return AutoImageProcessor, ZoeDepthForDepthEstimation, torch, torch_functional


def resolve_model_path(model_path: str | Path | None, project_root: Path) -> Path:
    search_candidates: list[Path] = []
    if model_path is not None:
        candidate = Path(model_path).expanduser()
        search_candidates.append(candidate if candidate.is_absolute() else (project_root / candidate))

    try:
        import os

        raw_env_model_path = os.environ.get("ZOEDEPTH_MODEL_PATH")
        if raw_env_model_path:
            search_candidates.append(Path(raw_env_model_path).expanduser())
    except Exception:
        pass

    search_candidates.extend(
        [
            project_root / "pretrained" / "zoedepth-nyu-kitti",
            project_root.parent / "pretrained" / "zoedepth-nyu-kitti",
            Path.cwd() / "pretrained" / "zoedepth-nyu-kitti",
        ]
    )

    hf_cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Intel--zoedepth-nyu-kitti" / "snapshots"
    if hf_cache_root.exists():
        search_candidates.extend(sorted(path for path in hf_cache_root.iterdir() if path.is_dir()))
    raw_cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Intel--zoedepth-nyu-kitti"
    if raw_cache_root.exists():
        search_candidates.append(raw_cache_root)

    normalized_candidates: list[Path] = []
    for candidate in search_candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved not in normalized_candidates:
            normalized_candidates.append(resolved)

    for candidate in normalized_candidates:
        config_path = candidate / "config.json"
        weights_path = candidate / "model.safetensors"
        if config_path.exists() and weights_path.exists():
            return candidate

    searched = "\n".join(f"  - {path}" for path in normalized_candidates)
    raise FileNotFoundError(
        "Could not find a local ZoeDepth model directory with config.json and model.safetensors.\n"
        "Searched:\n"
        f"{searched}\n"
        "Pass --zoe_model_path explicitly or set ZOEDEPTH_MODEL_PATH."
    )


class ZoeDepthWrapper:
    def __init__(
        self,
        model_path: str | Path | None,
        project_root: Path,
        device: str = "auto",
        allow_cpu_fallback: bool = True,
    ) -> None:
        self.project_root = project_root.resolve()
        self.model_path = resolve_model_path(model_path, self.project_root)
        self.device_preference = device
        self.allow_cpu_fallback = allow_cpu_fallback

        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._torch_functional: Any | None = None
        self._current_device: str | None = None
        self._load_notes: list[str] = []

    def _resolve_runtime_device(self, torch_module: Any) -> str:
        if self.device_preference == "cpu":
            return "cpu"
        if self.device_preference == "cuda":
            return "cuda"
        # This MVP prioritizes reproducibility over speed. Auto therefore selects CPU by default,
        # while still allowing an explicit `--device cuda` override when the environment is known-good.
        _ = torch_module
        return "cpu"

    def _load_model(self, force_device: str | None = None, reload: bool = False) -> None:
        if reload:
            self._processor = None
            self._model = None
            self._torch = None
            self._torch_functional = None
            self._current_device = None

        if self._model is not None and self._processor is not None:
            return

        AutoImageProcessor, ZoeDepthForDepthEstimation, torch_module, torch_functional = _load_transformers_zoe_classes()
        runtime_device = force_device or self._resolve_runtime_device(torch_module)

        LOGGER.info("Loading ZoeDepth model from %s", self.model_path)
        processor = AutoImageProcessor.from_pretrained(str(self.model_path), use_fast=False)
        model = ZoeDepthForDepthEstimation.from_pretrained(str(self.model_path))
        model = model.to(runtime_device)
        model.eval()

        self._processor = processor
        self._model = model
        self._torch = torch_module
        self._torch_functional = torch_functional
        self._current_device = runtime_device
        self._load_notes.append(f"loaded_with_transformers:{self.model_path}")
        LOGGER.info("ZoeDepth loaded on device=%s", runtime_device)

    def _run_inference_once(self, rgb_image: np.ndarray) -> np.ndarray:
        if self._processor is None or self._model is None or self._torch is None or self._torch_functional is None:
            raise RuntimeError("ZoeDepth model is not loaded.")

        image_height, image_width = rgb_image.shape[:2]
        image = Image.fromarray(rgb_image.astype(np.uint8), mode="RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._current_device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth
            resized_depth = self._torch_functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(image_height, image_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
        return resized_depth.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def predict(self, rgb_image: np.ndarray) -> DepthPrediction:
        self._load_model()
        assert self._torch is not None
        assert self._model is not None

        try:
            depth_map = self._run_inference_once(rgb_image)
            metadata = {
                "model_path": str(self.model_path),
                "device_requested": self.device_preference,
                "device_used": self._current_device,
                "load_notes": list(self._load_notes),
            }
            return DepthPrediction(depth=depth_map, metadata=metadata)
        except RuntimeError as exc:
            if not self.allow_cpu_fallback or self._current_device != "cuda":
                raise
            LOGGER.warning(
                "ZoeDepth inference failed on CUDA and will retry on CPU. Original error: %s",
                exc,
            )
            self._load_notes.append("cuda_failed_reloading_on_cpu")
            self._load_model(force_device="cpu", reload=True)
            depth_map = self._run_inference_once(rgb_image)
            metadata = {
                "model_path": str(self.model_path),
                "device_requested": self.device_preference,
                "device_used": self._current_device,
                "load_notes": list(self._load_notes),
                "fallback_reason": str(exc),
            }
            return DepthPrediction(depth=depth_map, metadata=metadata)
