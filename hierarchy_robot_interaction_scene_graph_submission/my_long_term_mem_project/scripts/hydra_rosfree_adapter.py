from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_REPO_ROOT = PROJECT_ROOT / "REACT-docker" / "hydra_ws" / "src" / "hydra"


@dataclass
class HydraRosfreeConfig:
    output_dir: Optional[str] = None
    robot_id: int = 0
    config_verbosity: int = 0
    use_step_mode: bool = True
    dataset_config_name: str = "habitat"
    min_range_m: float = 0.1
    max_range_m: float = 4.0


def _require_hydra_python():
    try:
        import hydra_python as hydra  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "hydra_python is not available. Build/install the ROS-free Hydra python bindings first."
        ) from exc
    return hydra


def _config_dir() -> Path:
    return HYDRA_REPO_ROOT / "config"


def _build_label_space_yaml(class_names: Sequence[str]) -> str:
    names = ["void"] + [str(name).strip() for name in class_names]
    label_names = []
    for label_id, name in enumerate(names):
        row = {"label": int(label_id), "name": str(name)}
        if label_id > 0:
            row["name_descriptive"] = f"a {name}"
        label_names.append(row)

    node = {
        "total_semantic_labels": len(names),
        "dynamic_labels": [],
        "invalid_labels": [0],
        "object_labels": list(range(1, len(names))),
        "surface_places_labels": [],
        "label_names": label_names,
    }
    return yaml.safe_dump(node, sort_keys=False)


def _build_pipeline_yaml() -> str:
    return yaml.safe_dump(
        {
            "frontend": {"type": "FrontendModule"},
            "backend": {"type": "BackendModule"},
            "reconstruction": {"type": "ReconstructionModule"},
        },
        sort_keys=False,
    )


def _camera_from_info(hydra, camera_info: Dict[str, float], *, min_range_m: float, max_range_m: float):
    camera = hydra.PythonCamera()
    camera.intrinsics.min_range = float(min_range_m)
    camera.intrinsics.max_range = float(max_range_m)
    camera.intrinsics.width = int(camera_info["width"])
    camera.intrinsics.height = int(camera_info["height"])
    camera.intrinsics.fx = float(camera_info["fx"])
    camera.intrinsics.fy = float(camera_info["fy"])
    camera.intrinsics.cx = float(camera_info["cx"])
    camera.intrinsics.cy = float(camera_info["cy"])
    return camera


class HydraRosfreeAdapter:
    """Thin Python adapter around Hydra C++ core without ROS plumbing.

    This adapter assumes:
    - YOLO+SAM (or another frontend) produces per-instance binary masks and class labels.
    - We assemble a semantic label image from those masks.
    - Hydra consumes `depth + labels + rgb + instance_masks` and returns an object layer snapshot.
    """

    def __init__(
        self,
        *,
        camera_info: Dict[str, float],
        detection_classes: Sequence[str],
        cfg: Optional[HydraRosfreeConfig] = None,
    ):
        self.cfg = cfg or HydraRosfreeConfig()
        self.hydra = _require_hydra_python()
        self.class_names = [str(name).strip() for name in detection_classes]
        self.label_to_id = {name: idx + 1 for idx, name in enumerate(self.class_names)}

        configs = self.hydra.PythonConfig()
        dataset_dir = _config_dir() / self.cfg.dataset_config_name
        configs.add_file(dataset_dir / "frontend_config.yaml", config_ns="frontend")
        configs.add_file(dataset_dir / "backend_config.yaml", config_ns="backend")
        configs.add_file(dataset_dir / "reconstruction_config.yaml", config_ns="reconstruction")
        configs.add_yaml(_build_pipeline_yaml())
        configs.add_yaml(_build_label_space_yaml(self.class_names))

        pipeline_config = self.hydra.PipelineConfig(configs)
        pipeline_config.enable_reconstruction = True
        pipeline_config.enable_lcd = False
        pipeline_config.label_names = {
            0: "void",
            **{idx + 1: name for idx, name in enumerate(self.class_names)},
        }
        if self.cfg.output_dir:
            pipeline_config.logs.log_dir = str(Path(self.cfg.output_dir).expanduser().resolve())

        self.pipeline = self.hydra.HydraPipeline(
            pipeline_config,
            robot_id=int(self.cfg.robot_id),
            config_verbosity=int(self.cfg.config_verbosity),
            use_step_mode=bool(self.cfg.use_step_mode),
        )
        self.pipeline.init(
            configs,
            _camera_from_info(
                self.hydra,
                camera_info,
                min_range_m=self.cfg.min_range_m,
                max_range_m=self.cfg.max_range_m,
            ),
        )

    def make_label_image(self, detections: Sequence[Dict], image_shape_hw: Sequence[int]) -> np.ndarray:
        height = int(image_shape_hw[0])
        width = int(image_shape_hw[1])
        label_image = np.zeros((height, width), dtype=np.int32)
        for det in detections:
            label = str(det["label"]).strip()
            if label not in self.label_to_id:
                continue
            mask = np.asarray(det["mask"])
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                raise ValueError(f"mask must be 2D after squeeze, got {mask.shape}")
            label_image[mask > 0] = int(self.label_to_id[label])
        return label_image

    def make_instance_masks(
        self,
        detections: Sequence[Dict],
        *,
        map_view_id: int,
    ) -> List:
        instance_masks = []
        for det_idx, det in enumerate(detections):
            label = str(det["label"]).strip()
            if label not in self.label_to_id:
                continue
            mask = np.asarray(det["mask"])
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                raise ValueError(f"mask must be 2D after squeeze, got {mask.shape}")
            mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
            instance_masks.append(
                self.hydra.MaskData(
                    int(map_view_id),
                    int(det.get("mask_id", det_idx)),
                    int(self.label_to_id[label]),
                    mask_u8,
                )
            )
        return instance_masks

    def step(
        self,
        *,
        timestamp_ns: int,
        world_t_body_xyz: Sequence[float],
        world_q_wxyz: Sequence[float],
        depth_m: np.ndarray,
        rgb: np.ndarray,
        detections: Sequence[Dict],
        map_view_id: int,
    ) -> List[Dict]:
        label_image = self.make_label_image(detections, depth_m.shape[:2])
        instance_masks = self.make_instance_masks(detections, map_view_id=map_view_id)

        self.pipeline.step_with_masks(
            int(timestamp_ns),
            np.asarray(world_t_body_xyz, dtype=np.float64).reshape(3),
            np.asarray(world_q_wxyz, dtype=np.float64).reshape(4),
            np.asarray(depth_m, dtype=np.float32),
            label_image,
            np.asarray(rgb, dtype=np.uint8),
            int(map_view_id),
            instance_masks,
        )
        return list(self.pipeline.get_object_layer())
