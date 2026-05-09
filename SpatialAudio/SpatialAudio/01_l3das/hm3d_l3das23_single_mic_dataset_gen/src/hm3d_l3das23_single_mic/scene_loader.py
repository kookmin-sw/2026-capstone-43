from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import DatasetGenerationConfig
from .schemas import MicPose, SceneInfo, SpeakerProxyPose
from .speaker_proxy import resolve_humanoid_render_asset_path

LOGGER = logging.getLogger(__name__)


def _vec3_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Cannot convert None to a 3D vector.")
    try:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape == (3,):
            return arr
    except Exception:
        pass
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return np.array([value.x, value.y, value.z], dtype=np.float64)
    return np.array(list(value), dtype=np.float64)


def _normalize_region_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    digits = "".join(ch for ch in text if ch.isdigit() or ch == "-")
    if digits and digits.lstrip("-").isdigit():
        return int(digits)
    return None


def _normalize_category_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if hasattr(value, "name"):
        try:
            result = value.name()
            if isinstance(result, str) and result.strip():
                return result.strip()
        except TypeError:
            result = value.name
            if isinstance(result, str) and result.strip():
                return result.strip()
        except Exception:
            pass
    if hasattr(value, "category"):
        return _normalize_category_name(getattr(value, "category"))
    text = str(value).strip()
    return text or None


def parse_hm3d_semantic_annotation_txt(
    annotation_path: Path,
) -> tuple[dict[int, str], dict[int, Optional[int]]]:
    object_category_by_id: dict[int, str] = {}
    object_room_id_by_id: dict[int, Optional[int]] = {}

    with annotation_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].strip().startswith("HM3D Semantic Annotations"):
                continue
            if len(row) < 4:
                continue
            try:
                object_id = int(row[0])
            except Exception:
                continue
            object_category_by_id[object_id] = str(row[2]).strip()
            object_room_id_by_id[object_id] = _normalize_region_id(row[3])
    return object_category_by_id, object_room_id_by_id


class HabitatSceneSession:
    def __init__(self, config: DatasetGenerationConfig, scene_info: SceneInfo) -> None:
        self.config = config
        self.scene_info = scene_info
        self.sim: Optional[Any] = None
        self.audio_sensor: Optional[Any] = None
        self._speaker_proxy_handle: Optional[str] = None
        self._speaker_proxy_render_asset_path: Optional[Path] = None
        self._speaker_proxy_template_handle: Optional[str] = None
        self._semantic_annotation_path: Optional[Path] = None
        self._object_category_by_id: dict[int, str] = {}
        self._object_room_id_by_id: dict[int, Optional[int]] = {}
        self._room_category_by_id: dict[int, str] = {}
        self._room_probe_cache: dict[tuple[int, int, int], tuple[Optional[int], Optional[str]]] = {}
        self.scene_bounds: tuple[np.ndarray, np.ndarray] = (
            np.array([-1.0, -1.0, -1.0], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
        )

    def __enter__(self) -> "HabitatSceneSession":
        self.open()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def _build_camera_spec(self, habitat_sim: Any, sensor_type: Any, uuid: str) -> Any:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [self.config.sensor_rig.rgb_height, self.config.sensor_rig.rgb_width]
        spec.position = [0.0, 0.0, 0.0]
        if hasattr(spec, "hfov"):
            spec.hfov = float(self.config.sensor_rig.hfov_deg)
        return spec

    def _build_audio_spec(self, habitat_sim: Any) -> Any:
        spec = habitat_sim.AudioSensorSpec()
        spec.uuid = "audio_sensor"
        spec.position = [0.0, 0.0, 0.0]
        if hasattr(spec, "enableMaterials"):
            spec.enableMaterials = bool(self.config.audio.enable_materials)

        channel_layout = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType
        layout_name = str(self.config.audio.channel_layout).lower()
        if layout_name == "ambisonics":
            layout_type = getattr(channel_layout, "Ambisonics")
        elif layout_name == "binaural":
            layout_type = getattr(channel_layout, "Binaural")
        else:
            layout_type = getattr(channel_layout, "Mono")

        if hasattr(spec.channelLayout, "channelType"):
            spec.channelLayout.channelType = layout_type
        elif hasattr(spec.channelLayout, "type"):
            spec.channelLayout.type = layout_type
        spec.channelLayout.channelCount = int(self.config.audio.channel_count)

        acoustics = spec.acousticsConfig
        for field_name, value in {
            "sampleRate": self.config.audio.sample_rate,
            "direct": int(bool(self.config.audio.direct)),
            "indirect": int(bool(self.config.audio.indirect)),
            "diffraction": int(bool(self.config.audio.diffraction)),
            "transmission": int(bool(self.config.audio.transmission)),
            "meshSimplification": int(bool(self.config.audio.mesh_simplification)),
            "temporalCoherence": int(bool(self.config.audio.temporal_coherence)),
            "directSHOrder": self.config.audio.direct_sh_order,
            "indirectSHOrder": self.config.audio.indirect_sh_order,
            "directRayCount": self.config.audio.direct_ray_count,
            "indirectRayCount": self.config.audio.indirect_ray_count,
            "sourceRayCount": self.config.audio.source_ray_count,
            "indirectRayDepth": self.config.audio.indirect_ray_depth,
            "sourceRayDepth": self.config.audio.source_ray_depth,
            "maxDiffractionOrder": self.config.audio.max_diffraction_order,
            "globalVolume": self.config.audio.global_volume,
            "enableMaterials": bool(self.config.audio.enable_materials),
        }.items():
            if hasattr(acoustics, field_name):
                setattr(acoustics, field_name, value)
        if hasattr(acoustics, "irTime"):
            acoustics.irTime = float(self.config.audio.max_ir_length_s)
        elif hasattr(acoustics, "maxIRLength"):
            acoustics.maxIRLength = float(self.config.audio.max_ir_length_s)
        if hasattr(acoustics, "threadCount"):
            acoustics.threadCount = 1
        return spec

    def _resolve_semantic_annots_root(self) -> Optional[Path]:
        explicit_root = self.config.paths.hm3d_semantic_annots_root
        if explicit_root is not None:
            return explicit_root

        dataset_config_parent = self.config.paths.hm3d_scene_dataset_config.parent
        candidate_names = []
        parent_name = dataset_config_parent.name
        if "semantic-configs" in parent_name:
            candidate_names.append(parent_name.replace("semantic-configs", "semantic-annots"))
        hm3d_root_name = self.config.paths.hm3d_root.name
        if "habitat" in hm3d_root_name:
            candidate_names.append(hm3d_root_name.replace("habitat", "semantic-annots"))

        for candidate_name in candidate_names:
            candidate = dataset_config_parent.parent / candidate_name
            if candidate.exists():
                return candidate
            candidate = self.config.paths.hm3d_root.parent / candidate_name
            if candidate.exists():
                return candidate
        return None

    def _resolve_semantic_annotation_path(self) -> Path:
        annot_root = self._resolve_semantic_annots_root()
        if annot_root is None:
            raise FileNotFoundError(
                "Could not resolve HM3D semantic annotation root. "
                "Set paths.hm3d_semantic_annots_root explicitly."
            )
        scene_dir = annot_root / self.scene_info.scene_id
        matches = sorted(scene_dir.glob("*.semantic.txt"))
        if not matches:
            raise FileNotFoundError(
                f"Semantic annotation .txt not found for scene {self.scene_info.scene_id} under {scene_dir}"
            )
        return matches[0]

    def _resolve_backend_scene_id(self) -> str:
        scene_path = self.scene_info.scene_path
        config_root = self.config.paths.hm3d_scene_dataset_config.parent.resolve()
        hm3d_root = self.config.paths.hm3d_root.resolve()
        if config_root == hm3d_root:
            try:
                return str(scene_path.resolve().relative_to(hm3d_root))
            except Exception:
                pass
        return str(scene_path)

    def _load_semantic_annotations(self) -> None:
        annotation_path = self._resolve_semantic_annotation_path()
        self._semantic_annotation_path = annotation_path
        self._object_category_by_id, self._object_room_id_by_id = parse_hm3d_semantic_annotation_txt(
            annotation_path
        )

        semantic_scene = getattr(self.sim, "semantic_scene", None) if self.sim is not None else None
        self._room_category_by_id = {}
        if semantic_scene is None:
            return
        for level in getattr(semantic_scene, "levels", []):
            for region in getattr(level, "regions", []):
                room_id = _normalize_region_id(
                    getattr(region, "id", getattr(region, "region_id", getattr(region, "index", None)))
                )
                if room_id is None:
                    continue
                category = _normalize_category_name(getattr(region, "category", None))
                if category is not None:
                    self._room_category_by_id[room_id] = category

    def open(self) -> None:
        try:
            import habitat_sim  # type: ignore
            import habitat_sim.agent  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "habitat_sim is required to open HM3D scenes. "
                "Use a Habitat-Sim environment with audio support enabled."
            ) from exc

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self._resolve_backend_scene_id()
        backend_cfg.scene_dataset_config_file = str(self.config.paths.hm3d_scene_dataset_config)
        backend_cfg.enable_physics = bool(self.config.simulator.enable_physics)
        backend_cfg.load_semantic_mesh = bool(
            self.config.simulator.load_semantic_mesh
            or self.config.sensor_rig.enable_semantic
            or self.config.audio.enable_materials
        )
        backend_cfg.frustum_culling = bool(self.config.simulator.frustum_culling)
        backend_cfg.gpu_device_id = int(self.config.simulator.gpu_device_id)

        sensor_specs = []
        if self.config.sensor_rig.enable_rgb:
            sensor_specs.append(
                self._build_camera_spec(habitat_sim, habitat_sim.SensorType.COLOR, "rgb_sensor")
            )
        if self.config.sensor_rig.enable_depth:
            sensor_specs.append(
                self._build_camera_spec(habitat_sim, habitat_sim.SensorType.DEPTH, "depth_sensor")
            )
        if self.config.sensor_rig.enable_semantic:
            sensor_specs.append(
                self._build_camera_spec(
                    habitat_sim,
                    habitat_sim.SensorType.SEMANTIC,
                    "semantic_sensor",
                )
            )
        sensor_specs.append(self._build_audio_spec(habitat_sim))

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.height = 0.0
        agent_cfg.radius = 0.1

        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(sim_cfg)
        if not self.sim.pathfinder.is_loaded:
            raise RuntimeError(f"Pathfinder is not loaded for scene {self.scene_info.scene_id}")
        self.audio_sensor = self.sim.get_agent(0)._sensors["audio_sensor"]
        if self.config.paths.audio_materials_json is not None:
            audio_materials_path = Path(self.config.paths.audio_materials_json)
            if not audio_materials_path.exists():
                raise FileNotFoundError(f"audio_materials_json not found: {audio_materials_path}")
            self.audio_sensor.setAudioMaterialsJSON(str(audio_materials_path))

        self.scene_bounds = tuple(
            _vec3_to_numpy(bound) for bound in self.sim.pathfinder.get_bounds()
        )  # type: ignore[assignment]
        self._load_semantic_annotations()
        self._speaker_proxy_render_asset_path = resolve_humanoid_render_asset_path(self.config)
        if self.config.speaker_proxy.enabled:
            if not hasattr(self.sim, "get_object_template_manager"):
                raise RuntimeError(
                    "speaker proxy is enabled but this Habitat-Sim build does not expose "
                    "an object template manager"
                )
            if not hasattr(self.sim, "get_rigid_object_manager"):
                raise RuntimeError(
                    "speaker proxy is enabled but this Habitat-Sim build does not expose "
                    "a rigid object manager"
                )
            if self._speaker_proxy_render_asset_path is None:
                raise RuntimeError(
                    "speaker proxy is enabled but no humanoid render asset path was resolved"
                )
            if not self._speaker_proxy_render_asset_path.exists():
                raise FileNotFoundError(
                    f"speaker proxy render asset not found: {self._speaker_proxy_render_asset_path}"
                )
            self._register_speaker_proxy_template()
        LOGGER.info("Opened scene %s", self.scene_info.scene_id)

    def close(self) -> None:
        if self.sim is not None:
            self._remove_speaker_proxy()
            self.sim.close()
            self.sim = None
            self.audio_sensor = None

    def _yaw_quaternion(self, yaw_rad: float) -> Any:
        from habitat_sim.utils.common import quat_from_angle_axis  # type: ignore

        return quat_from_angle_axis(
            float(yaw_rad),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

    def _yaw_magnum_quaternion(self, yaw_rad: float) -> Any:
        import magnum as mn  # type: ignore

        return mn.Quaternion.rotation(mn.Rad(float(yaw_rad)), mn.Vector3.y_axis())

    def set_microphone_pose(self, mic_pose: MicPose) -> None:
        self._set_agent_pose(mic_pose.position_world, mic_pose.yaw_rad)

    def _set_agent_pose(self, position_world: list[float] | np.ndarray, yaw_rad: float) -> None:
        if self.sim is None:
            raise RuntimeError("Scene session is not open.")
        agent = self.sim.get_agent(0)
        state = agent.get_state()
        quat = self._yaw_quaternion(yaw_rad)
        position = np.asarray(position_world, dtype=np.float32)
        state.position = position
        state.rotation = quat
        for sensor_state in state.sensor_states.values():
            sensor_state.position = position
            sensor_state.rotation = quat
        agent.set_state(state, True)

    def _remove_speaker_proxy(self) -> None:
        if self.sim is None or self._speaker_proxy_handle is None:
            self._speaker_proxy_handle = None
            return
        try:
            manager = self.sim.get_rigid_object_manager()
            manager.remove_object_by_handle(self._speaker_proxy_handle)
        except Exception:
            LOGGER.debug("Failed to remove speaker proxy %s", self._speaker_proxy_handle, exc_info=True)
        finally:
            self._speaker_proxy_handle = None

    def _register_speaker_proxy_template(self) -> None:
        if self.sim is None:
            raise RuntimeError("Scene session is not open.")
        if self._speaker_proxy_render_asset_path is None:
            raise RuntimeError("speaker proxy render asset path is not available")

        template_handle = str(self._speaker_proxy_render_asset_path)
        obj_attr_mgr = self.sim.get_object_template_manager()
        if not obj_attr_mgr.get_library_has_handle(template_handle):
            template = obj_attr_mgr.create_template(template_handle, False)
            if hasattr(template, "scale"):
                template.scale = np.array(
                    [float(self.config.speaker_proxy.global_scale)] * 3,
                    dtype=np.float32,
                )
            if (
                hasattr(template, "orient_up")
                and self.config.speaker_proxy.render_orient_up is not None
            ):
                template.orient_up = tuple(
                    float(value) for value in self.config.speaker_proxy.render_orient_up
                )
            if (
                hasattr(template, "orient_front")
                and self.config.speaker_proxy.render_orient_front is not None
            ):
                template.orient_front = tuple(
                    float(value) for value in self.config.speaker_proxy.render_orient_front
                )
            obj_attr_mgr.register_template(template, template_handle)
        self._speaker_proxy_template_handle = template_handle

    def _spawn_speaker_proxy(self, speaker_proxy_pose: SpeakerProxyPose) -> None:
        if self.sim is None:
            raise RuntimeError("Scene session is not open.")
        if self._speaker_proxy_template_handle is None:
            raise RuntimeError("speaker proxy template handle is not available")
        manager = self.sim.get_rigid_object_manager()
        proxy = manager.add_object_by_template_handle(self._speaker_proxy_template_handle)
        if hasattr(proxy, "motion_type"):
            try:
                import habitat_sim.physics as habitat_physics  # type: ignore

                proxy.motion_type = habitat_physics.MotionType.KINEMATIC
            except Exception:
                LOGGER.debug("Could not switch speaker proxy motion_type to KINEMATIC", exc_info=True)
        proxy.translation = np.asarray(speaker_proxy_pose.root_world, dtype=np.float32)
        proxy.rotation = self._yaw_magnum_quaternion(float(speaker_proxy_pose.yaw_rad))
        self._speaker_proxy_handle = str(proxy.handle)

    def render_visual_observations(
        self,
        mic_pose: MicPose,
        speaker_proxy_pose: Optional[SpeakerProxyPose] = None,
    ) -> dict[str, Any]:
        if self.sim is None:
            raise RuntimeError("Scene session is not open.")
        self.set_microphone_pose(mic_pose)
        if not self.config.speaker_proxy.enabled or speaker_proxy_pose is None:
            return self.sim.get_sensor_observations(0)

        self._remove_speaker_proxy()
        self._spawn_speaker_proxy(speaker_proxy_pose)
        try:
            return self.sim.get_sensor_observations(0)
        finally:
            self._remove_speaker_proxy()

    def render_rir(self, source_position_world: list[float], mic_pose: MicPose) -> np.ndarray:
        if self.sim is None or self.audio_sensor is None:
            raise RuntimeError("Audio sensor is not available.")
        self.set_microphone_pose(mic_pose)
        source = np.asarray(source_position_world, dtype=np.float32)
        self.audio_sensor.setAudioSourceTransform(source)
        if hasattr(self.audio_sensor, "setAudioListenerTransform"):
            listener = np.asarray(mic_pose.position_world, dtype=np.float32)
            quat_wxyz = np.asarray(mic_pose.quaternion_wxyz, dtype=np.float32)
            self.audio_sensor.setAudioListenerTransform(listener, quat_wxyz)
        observations = self.sim.get_sensor_observations(0)
        return np.asarray(observations["audio_sensor"], dtype=np.float32)

    def render_probe_observations(
        self,
        position_world: list[float] | np.ndarray,
        *,
        yaw_rad: float,
    ) -> dict[str, Any]:
        if self.sim is None:
            raise RuntimeError("Scene session is not open.")
        agent = self.sim.get_agent(0)
        original_state = agent.get_state()
        try:
            self._set_agent_pose(position_world, yaw_rad)
            return self.sim.get_sensor_observations(0)
        finally:
            agent.set_state(original_state, True)

    def build_semantic_index(self) -> dict[str, dict[int, Any]]:
        return {
            "object_category_by_id": dict(self._object_category_by_id),
            "object_room_id_by_id": dict(self._object_room_id_by_id),
            "room_category_by_id": dict(self._room_category_by_id),
        }

    def lookup_object_category(self, object_id: Optional[int]) -> Optional[str]:
        if object_id is None:
            return None
        return self._object_category_by_id.get(int(object_id))

    def lookup_room_category(self, room_id: Optional[int]) -> Optional[str]:
        if room_id is None:
            return None
        return self._room_category_by_id.get(int(room_id))

    def visible_object_ids_from_mask(self, semantic_mask: Any) -> list[int]:
        semantic = np.asarray(semantic_mask, dtype=np.int64)
        if semantic.size == 0:
            return []
        visible = sorted(
            int(value)
            for value in np.unique(semantic)
            if int(value) > 0 and int(value) in self._object_category_by_id
        )
        return visible

    def infer_room_info_from_semantic_mask(
        self,
        semantic_mask: Any,
    ) -> tuple[Optional[int], Optional[str]]:
        if semantic_mask is None:
            return None, None
        semantic = np.asarray(semantic_mask, dtype=np.int64)
        if semantic.size == 0:
            return None, None
        values, counts = np.unique(semantic, return_counts=True)
        votes: dict[int, int] = {}
        for value, count in zip(values.tolist(), counts.tolist()):
            object_id = int(value)
            if object_id <= 0:
                continue
            room_id = self._object_room_id_by_id.get(object_id)
            if room_id is None:
                continue
            votes[int(room_id)] = votes.get(int(room_id), 0) + int(count)
        if not votes:
            return None, None
        room_id = max(votes.items(), key=lambda item: (item[1], -abs(item[0])))[0]
        return room_id, self.lookup_room_category(room_id)

    def infer_room_id(self, point_world: list[float]) -> Optional[int]:
        room_id, _ = self.infer_room_info(point_world)
        return room_id

    def infer_room_info(self, point_world: list[float]) -> tuple[Optional[int], Optional[str]]:
        if self.sim is None:
            return None, None
        semantic_scene = getattr(self.sim, "semantic_scene", None)
        if semantic_scene is None:
            return self._infer_room_info_from_semantic_views(point_world)

        point = np.asarray(point_world, dtype=np.float64)
        for level in getattr(semantic_scene, "levels", []):
            for region in getattr(level, "regions", []):
                aabb = getattr(region, "aabb", None)
                if aabb is None:
                    continue
                try:
                    center = _vec3_to_numpy(aabb.center)
                    sizes = _vec3_to_numpy(aabb.sizes)
                except Exception:
                    continue
                if not np.all(np.abs(point - center) <= (sizes * 0.5)):
                    continue
                room_id = _normalize_region_id(
                    getattr(region, "id", getattr(region, "region_id", getattr(region, "index", None)))
                )
                if room_id is None:
                    continue
                room_category = self.lookup_room_category(room_id) or _normalize_category_name(
                    getattr(region, "category", None)
                )
                return room_id, room_category
        return self._infer_room_info_from_semantic_views(point_world)

    def _infer_room_info_from_semantic_views(
        self,
        point_world: list[float],
    ) -> tuple[Optional[int], Optional[str]]:
        if not bool(self.config.sensor_rig.enable_semantic):
            return self._infer_room_info_from_object_hits(point_world)
        cache_key = tuple(int(round(float(coord) * 10.0)) for coord in point_world[:3])
        if cache_key in self._room_probe_cache:
            return self._room_probe_cache[cache_key]
        if self.sim is None or not self._object_room_id_by_id:
            result = (None, None)
            self._room_probe_cache[cache_key] = result
            return result

        base_point = np.asarray(point_world, dtype=np.float64)
        height_offsets = (
            0.5,
            1.0,
            float(self.config.sensor_rig.mic_height_m),
        )
        yaw_candidates = (
            0.0,
            0.5 * math.pi,
            math.pi,
            1.5 * math.pi,
        )
        votes: dict[int, int] = {}
        for height_offset in height_offsets:
            probe_point = base_point.copy()
            probe_point[1] = float(base_point[1]) + float(height_offset)
            for yaw_rad in yaw_candidates:
                try:
                    observations = self.render_probe_observations(
                        probe_point.astype(float).tolist(),
                        yaw_rad=float(yaw_rad),
                    )
                except Exception:
                    continue
                semantic_mask = observations.get("semantic_sensor")
                if semantic_mask is None:
                    continue
                room_id, _ = self.infer_room_info_from_semantic_mask(semantic_mask)
                if room_id is None:
                    continue
                semantic = np.asarray(semantic_mask, dtype=np.int64)
                room_votes = 0
                values, counts = np.unique(semantic, return_counts=True)
                for value, count in zip(values.tolist(), counts.tolist()):
                    object_id = int(value)
                    if object_id <= 0:
                        continue
                    if self._object_room_id_by_id.get(object_id) == room_id:
                        room_votes += int(count)
                votes[int(room_id)] = votes.get(int(room_id), 0) + room_votes

        if votes:
            room_id = max(votes.items(), key=lambda item: (item[1], -abs(item[0])))[0]
            result = (room_id, self.lookup_room_category(room_id))
            self._room_probe_cache[cache_key] = result
            return result

        result = self._infer_room_info_from_object_hits(point_world)
        self._room_probe_cache[cache_key] = result
        return result

    def _infer_room_info_from_object_hits(
        self,
        point_world: list[float],
    ) -> tuple[Optional[int], Optional[str]]:
        if self.sim is None or not self._object_room_id_by_id:
            return None, None
        try:
            import habitat_sim  # type: ignore
        except ImportError:
            return None, None

        point = np.asarray(point_world, dtype=np.float64)
        horizontal_dirs = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            np.array([1.0, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0, -1.0], dtype=np.float64),
            np.array([-1.0, 0.0, 1.0], dtype=np.float64),
            np.array([-1.0, 0.0, -1.0], dtype=np.float64),
        ]
        directions = [direction / max(np.linalg.norm(direction), 1.0e-8) for direction in horizontal_dirs]
        directions.extend(
            [
                np.array([0.0, -1.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
            ]
        )
        origin_offsets = (
            np.array([0.0, 0.05, 0.0], dtype=np.float64),
            np.array([0.0, 0.5, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )

        votes: dict[int, float] = {}
        for offset in origin_offsets:
            origin = point + offset
            for direction in directions:
                ray = habitat_sim.geo.Ray(
                    origin.astype(np.float32),
                    direction.astype(np.float32),
                )
                results = self.sim.cast_ray(ray=ray, max_distance=3.0)
                for hit in getattr(results, "hits", []):
                    try:
                        object_id = int(hit.object_id)
                    except Exception:
                        continue
                    room_id = self._object_room_id_by_id.get(object_id)
                    if room_id is None:
                        continue
                    distance = max(float(hit.ray_distance), 0.05)
                    votes[int(room_id)] = votes.get(int(room_id), 0.0) + (1.0 / distance)
                    break

        if not votes:
            return None, None
        room_id = max(
            votes.items(),
            key=lambda item: (item[1], -abs(item[0])),
        )[0]
        return room_id, self.lookup_room_category(room_id)

    def infer_level_id(self, point_world: list[float]) -> Optional[str]:
        if self.sim is None:
            return None
        semantic_scene = getattr(self.sim, "semantic_scene", None)
        if semantic_scene is None:
            return None

        point = np.asarray(point_world, dtype=np.float64)
        for level in getattr(semantic_scene, "levels", []):
            aabb = getattr(level, "aabb", None)
            if aabb is None:
                continue
            try:
                center = _vec3_to_numpy(aabb.center)
                sizes = _vec3_to_numpy(aabb.sizes)
            except Exception:
                continue
            if np.all(np.abs(point - center) <= (sizes * 0.5)):
                return str(level.id)
        return None
