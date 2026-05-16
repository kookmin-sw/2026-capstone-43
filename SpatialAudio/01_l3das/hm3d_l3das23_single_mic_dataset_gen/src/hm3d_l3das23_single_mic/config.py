from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

import yaml


def _expand_path(value: Optional[Union[str, Path]], base_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


@dataclass
class PathsConfig:
    hm3d_root: Path = Path("data/scene_datasets/hm3d")
    hm3d_scene_glob: str = "**/*.basis.glb"
    hm3d_scene_dataset_config: Path = Path(
        "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    )
    hm3d_semantic_annots_root: Optional[Path] = None
    dataset_root: Path = Path("outputs/hm3d_l3das23_single_mic")
    dry_audio_root: Path = Path("data/dry_audio")
    dry_audio_glob: str = "**/*.wav"
    convolution_dry_audio_root: Optional[Path] = None
    convolution_dry_audio_glob: str = "**/*.flac"
    audio_materials_json: Optional[Path] = None


@dataclass
class SimulatorConfig:
    gpu_device_id: int = 0
    enable_physics: bool = False
    load_semantic_mesh: bool = True
    frustum_culling: bool = True
    silent: bool = True


@dataclass
class SensorRigConfig:
    mic_height_m: float = 1.6
    enable_rgb: bool = True
    rgb_width: int = 512
    rgb_height: int = 512
    hfov_deg: float = 90.0
    enable_depth: bool = True
    enable_semantic: bool = True
    write_rgb_png: bool = True
    write_depth_png: bool = True
    write_depth_npy: bool = True
    write_semantic_preview_png: bool = True
    write_instance_mask_npy: bool = True


@dataclass
class MicSamplingConfig:
    poses_per_scene: int = 32
    candidate_pool_size: int = 4096
    min_clearance_m: float = 0.4
    min_island_radius_m: float = 1.5
    require_navigable_snap_confirmation: bool = True
    navmesh_snap_xy_tolerance_m: float = 0.02
    navmesh_snap_vertical_tolerance_m: float = 0.05
    probe_radius_m: float = 0.25
    min_point_clearance_m: float = 0.12
    probe_ignore_hits_within_m: float = 0.01
    dense_probe_radius_m: float = 0.35
    dense_min_point_clearance_m: float = 0.12
    dense_enclosure_hit_fraction_threshold: float = 0.92
    reject_if_clearance_unknown: bool = True
    yaw_mode: str = "uniform"
    fixed_yaw_deg: float = 0.0


@dataclass
class SourceSamplingConfig:
    rho_min_m: float = 1.0
    rho_max_m: float = 3.0
    rho_step_m: float = 0.5
    z_min_m: float = -1.2
    z_max_m: float = 0.8
    z_step_m: float = 0.4
    theta_sampling_mode: str = "fixed_angle"
    theta_arc_step_m: float = 0.5
    theta_step_deg: float = 15.0
    theta_offset_deg: float = 0.0
    max_sources_per_mic: int = 48
    shuffle_candidates: bool = True
    min_source_distance_m: float = 1.0
    max_source_distance_m: float = 3.05
    scene_bounds_margin_m: float = 0.05
    probe_radius_m: float = 0.2
    min_clearance_m: float = 0.12
    probe_ignore_hits_within_m: float = 0.01
    dense_probe_radius_m: float = 0.35
    dense_min_clearance_m: float = 0.12
    dense_enclosure_hit_fraction_threshold: float = 0.92
    reject_if_clearance_unknown: bool = True
    require_navigable_projection: bool = True
    navmesh_projection_xy_tolerance_m: float = 0.1
    navmesh_projection_vertical_tolerance_m: float = 3.0
    require_same_floor_as_mic: bool = True
    same_floor_vertical_tolerance_m: float = 0.3
    min_anchor_clearance_m: float = 0.3
    min_anchor_island_radius_m: float = 1.5


@dataclass
class LosConfig:
    eps_start_m: float = 0.03
    eps_end_m: float = 0.05
    max_dist_margin_m: float = 0.02
    ignore_hits_within_m: float = 0.01
    reject_endpoint_ambiguity: bool = True
    use_audio_visibility_fallback_when_raycast_empty: bool = True
    bidirectional_consistency_check: bool = True
    conservative_on_bidirectional_disagreement: bool = True
    mark_raycast_empty_unstable_without_fallback: bool = True
    visibility_ratio_enabled: bool = True
    visibility_sphere_radius_m: float = 0.08
    visibility_num_rays: int = 9


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    channel_layout: str = "ambisonics"
    channel_count: int = 4
    enable_materials: bool = False
    direct: bool = True
    indirect: bool = True
    diffraction: bool = True
    transmission: bool = False
    mesh_simplification: bool = False
    temporal_coherence: bool = False
    direct_sh_order: int = 1
    indirect_sh_order: int = 1
    direct_ray_count: int = 500
    indirect_ray_count: int = 5000
    source_ray_count: int = 200
    direct_ray_depth: int = 10
    indirect_ray_depth: int = 200
    source_ray_depth: int = 10
    max_diffraction_order: int = 10
    max_ir_length_s: float = 4.0
    global_volume: float = 0.25
    normalize_peak_dbfs: float = -1.0
    clip_peak: float = 0.999
    silence_rms_threshold: float = 1.0e-6
    write_mic_librispeech_wav: bool = True
    write_rir_npy: bool = False
    write_rir_wav: bool = False


@dataclass
class SpeakerProxyConfig:
    enabled: bool = False
    humanoids_root: Optional[Path] = None
    avatar_name: str = "neutral_0"
    urdf_filename: Optional[str] = None
    render_asset_filename: Optional[str] = None
    render_orient_up: Optional[list[float]] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    render_orient_front: Optional[list[float]] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    fixed_base: bool = True
    global_scale: float = 1.0
    mass_scale: float = 1.0
    force_reload: bool = False
    maintain_link_order: bool = True
    face_microphone: bool = True
    yaw_offset_deg: float = 0.0
    render_root_from_floor_anchor: bool = True
    render_root_height_offset_m: float = 0.9
    body_probe_heights_m: list[float] = field(default_factory=lambda: [0.4, 0.9, 1.4])
    body_probe_radius_m: float = 0.20
    body_probe_ignore_hits_within_m: float = 0.01
    body_min_clearance_m: float = 0.12
    reject_if_body_clearance_unknown: bool = True


@dataclass
class GenerationConfig:
    seed: int = 1337
    audio_only: bool = False
    resume: bool = True
    overwrite: bool = False
    fail_fast: bool = False
    save_topdown_debug: bool = True
    save_front_overlay: bool = True
    splits: list[str] = field(default_factory=lambda: ["train", "val", "test"])
    scene_allowlist: list[str] = field(default_factory=list)
    scene_denylist: list[str] = field(default_factory=list)
    scene_shard_count: int = 1
    scene_shard_index: int = 0
    max_scenes: Optional[int] = None
    shuffle_scene_order: bool = True
    max_valid_samples_per_scene: Optional[int] = None
    target_glos_count: Optional[int] = None
    target_gnlos_count: Optional[int] = None
    allowed_geometry_los: list[str] = field(default_factory=list)
    required_in_fov: Optional[bool] = None
    stop_when_target_reached: bool = True


@dataclass
class SplitConfig:
    seed: int = 1337
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    debug_num_scenes_per_split: int = 1
    debug_max_mics_per_scene: int = 2
    debug_max_sources_per_mic: int = 4


@dataclass
class QCConfig:
    reject_invalid_los: bool = True
    reject_audio_clipped: bool = True
    reject_audio_silent: bool = True
    reject_nan_audio: bool = True
    reject_invalid_projection: bool = True


@dataclass
class DatasetGenerationConfig:
    version: str = "0.1"
    paths: PathsConfig = field(default_factory=PathsConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    sensor_rig: SensorRigConfig = field(default_factory=SensorRigConfig)
    mic_sampling: MicSamplingConfig = field(default_factory=MicSamplingConfig)
    source_sampling: SourceSamplingConfig = field(default_factory=SourceSamplingConfig)
    los: LosConfig = field(default_factory=LosConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    speaker_proxy: SpeakerProxyConfig = field(default_factory=SpeakerProxyConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    qc: QCConfig = field(default_factory=QCConfig)

    @property
    def manifests_dir(self) -> Path:
        return self.paths.dataset_root / "manifests"

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


def _is_path_type(annotation: Any) -> bool:
    if annotation is Path:
        return True
    origin = get_origin(annotation)
    if origin is Union:
        return any(arg is Path for arg in get_args(annotation))
    return False


def _is_list_type(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin in (list, tuple)


def _merge_dataclass(default_obj: Any, payload: dict[str, Any], base_dir: Path) -> Any:
    values: dict[str, Any] = {}
    type_hints = get_type_hints(type(default_obj))
    for field_info in fields(default_obj):
        current_value = getattr(default_obj, field_info.name)
        annotation = type_hints.get(field_info.name, field_info.type)
        if field_info.name not in payload:
            values[field_info.name] = current_value
            continue
        raw_value = payload[field_info.name]
        if is_dataclass(current_value):
            if raw_value is None:
                values[field_info.name] = current_value
            elif not isinstance(raw_value, dict):
                raise TypeError(f"{field_info.name} must be a mapping")
            else:
                values[field_info.name] = _merge_dataclass(current_value, raw_value, base_dir)
            continue
        if _is_path_type(annotation):
            values[field_info.name] = _expand_path(raw_value, base_dir)
            continue
        if _is_list_type(annotation):
            values[field_info.name] = list(raw_value) if raw_value is not None else None
            continue
        values[field_info.name] = raw_value
    return type(default_obj)(**values)


def load_config(config_path: Union[str, Path]) -> DatasetGenerationConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError("Config root must be a mapping")
    defaults = DatasetGenerationConfig()
    config = _merge_dataclass(defaults, payload, path.parent)
    if not isinstance(config, DatasetGenerationConfig):
        raise TypeError("Failed to construct DatasetGenerationConfig")
    return config


def dump_config(config: DatasetGenerationConfig, output_path: Union[str, Path]) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=False)
    return path
