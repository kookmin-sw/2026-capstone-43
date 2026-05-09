from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def _round_value(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _round_value(val, digits) for key, val in value.items()}
    if isinstance(value, list):
        return [_round_value(item, digits) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _round_value(asdict(value), digits)
    return value


@dataclass
class SceneInfo:
    scene_id: str
    scene_path: Path
    split: Optional[str] = None


@dataclass
class PoseRecord:
    position_xyz: list[float]
    quaternion_wxyz: list[float]
    yaw_rad: float
    yaw_deg: float


@dataclass
class MicPose:
    mic_index: int
    floor_point_world: list[float]
    position_world: list[float]
    quaternion_wxyz: list[float]
    yaw_rad: float
    yaw_deg: float

    def to_pose_record(self) -> PoseRecord:
        return PoseRecord(
            position_xyz=list(self.position_world),
            quaternion_wxyz=list(self.quaternion_wxyz),
            yaw_rad=self.yaw_rad,
            yaw_deg=self.yaw_deg,
        )


@dataclass
class CylindricalCoordinate:
    rho: float
    theta_rad: float
    theta_deg: float
    z: float


@dataclass
class SphericalCoordinate:
    distance: float
    azimuth_deg: float
    elevation_deg: float


@dataclass
class SourceCandidate:
    source_index: int
    local_xyz: list[float]
    world_xyz: list[float]
    cylindrical: CylindricalCoordinate
    spherical: SphericalCoordinate


@dataclass
class SpeakerProxyPose:
    avatar_name: str
    reference_world: list[float]
    root_world: list[float]
    floor_anchor_world: list[float]
    yaw_rad: float
    yaw_deg: float


@dataclass
class CameraIntrinsicsRecord:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ProjectionResult:
    in_fov: bool
    pixel_xy: Optional[list[float]]
    depth_cam: Optional[float]
    normalized_xy: Optional[list[float]]
    reason: Optional[str]


@dataclass
class GeometryLosResult:
    geometry_los: str
    first_hit_distance: Optional[float]
    source_distance: float
    hit_object_id: Optional[int]
    stable: bool
    occlusion_hit_distance: Optional[float]
    occluding_object_id: Optional[int]
    occluder_count: int = 0
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioRenderResult:
    rir_generation_status: str
    rendering_status: str
    output_wav_path: Optional[Path]
    rir_npy_path: Optional[Path]
    rir_wav_path: Optional[Path]
    peak_amplitude: Optional[float]
    rms: Optional[float]
    num_samples: int
    output_sample_rate: int
    dry_audio_num_samples: int
    dry_audio_sample_rate: int
    dry_audio_relpath: str
    secondary_output_wav_path: Optional[Path] = None
    secondary_rendering_status: Optional[str] = None
    secondary_peak_amplitude: Optional[float] = None
    secondary_rms: Optional[float] = None
    secondary_num_samples: int = 0
    secondary_dry_audio_num_samples: Optional[int] = None
    secondary_dry_audio_sample_rate: Optional[int] = None
    secondary_dry_audio_relpath: Optional[str] = None
    debug_notes: list[str] = field(default_factory=list)


@dataclass
class ImageRenderResult:
    rgb_path: Path
    depth_path: Optional[Path] = None
    depth_npy_path: Optional[Path] = None
    instance_mask_path: Optional[Path] = None
    semantic_preview_path: Optional[Path] = None
    front_overlay_path: Optional[Path] = None
    debug_notes: list[str] = field(default_factory=list)


@dataclass
class ManifestSource:
    source_id: str
    source_index: int
    source_world_position: list[float]
    source_mic_relative_position: list[float]
    distance_to_mic: float
    continuous_azimuth_deg: float
    continuous_elevation_deg: float
    local_coordinate_frame: str
    azimuth_reference: str
    azimuth_convention: str
    azimuth_continuous_raw_deg: float
    direction_8way: str
    label_8way: str
    ambix_unit_vector_xyz: list[float]
    local_unit_vector_right_front_up: list[float]
    room_id: Optional[int]
    room_category: Optional[str]
    is_los: bool
    is_nlos: bool
    is_in_fov: bool
    is_out_of_fov: bool
    visible_ratio: float
    source_visible_binary: bool
    euclidean_distance: float
    direct_path_length: float
    source_associated_object_id: int
    source_object_instance_id: int
    source_object_category: str
    rir_id: str
    dry_source_id: str
    source_type: str
    snr: Optional[float]
    onset_time: float
    offset_time: float


@dataclass
class ManifestRow:
    sample_id: str
    split: str
    scene_id: str
    room_id: Optional[int]
    source_id: str
    sequence_id_or_episode_id: Optional[str]
    manifest_version: str
    level_id: Optional[str]
    floor_id: Optional[str]
    generation_seed: int
    mic_pose_world: PoseRecord
    camera_pose_world: PoseRecord
    source_pose_world: list[float]
    source_pose_local_xyz: list[float]
    source_pose_cylindrical: CylindricalCoordinate
    source_pose_spherical: SphericalCoordinate
    speaker_proxy_avatar: Optional[str]
    speaker_proxy_root_world: Optional[list[float]]
    speaker_proxy_reference_world: Optional[list[float]]
    speaker_proxy_floor_anchor_world: Optional[list[float]]
    speaker_proxy_yaw_deg: Optional[float]
    geometry_los: str
    geometry_los_stable: bool
    los_definition: str
    in_fov: bool
    projected_pixel_xy: Optional[list[float]]
    projection_depth_cam: Optional[float]
    projection_reason: Optional[str]
    source_distance: float
    azimuth_deg: float
    elevation_deg: float
    occlusion_hit_distance: Optional[float]
    occluding_object_id: Optional[int]
    occluder_count: int
    first_occluder_instance_id: Optional[int]
    first_occluder_category: Optional[str]
    visibility_ratio: float
    visible_ratio: float
    dry_audio_filename: str
    dry_audio_num_samples: int
    dry_audio_sample_rate: int
    secondary_dry_audio_filename: Optional[str]
    secondary_dry_audio_num_samples: Optional[int]
    secondary_dry_audio_sample_rate: Optional[int]
    rir_generation_status: str
    rendering_status: str
    secondary_rendering_status: Optional[str]
    output_files: dict[str, str]
    camera_world_position: list[float]
    camera_world_rotation: list[float]
    mic_world_position: list[float]
    mic_world_rotation: list[float]
    camera_intrinsics: CameraIntrinsicsRecord
    image_width: int
    image_height: int
    horizontal_fov: float
    vertical_fov: float
    camera_mic_axis_aligned: bool
    foa_audio_path: str
    rgb_image_path: str
    depth_image_path: str
    instance_mask_path: str
    metadata_path: str
    audio_format: str
    audio_channel_layout: str
    audio_channel_order: str
    foa_raw_channel_order: str
    foa_canonical_channel_order: str
    foa_canonical_axes: str
    source_world_position: list[float]
    source_mic_relative_position: list[float]
    distance_to_mic: float
    continuous_azimuth_deg: float
    continuous_elevation_deg: float
    local_coordinate_frame: str
    azimuth_reference: str
    azimuth_convention: str
    azimuth_continuous_raw_deg: float
    direction_8way: str
    label_8way: str
    ambix_unit_vector_xyz: list[float]
    local_unit_vector_right_front_up: list[float]
    is_los: bool
    is_nlos: bool
    is_in_fov: bool
    is_out_of_fov: bool
    source_visible_binary: bool
    source_associated_object_id: int
    euclidean_distance: float
    direct_path_length: float
    rir_id: str
    dry_source_id: str
    source_type: str
    num_active_sources: int
    snr: Optional[float]
    onset_time: float
    offset_time: float
    source_object_instance_id: int
    source_object_category: str
    visible_object_ids: list[int]
    distractor_object_ids: list[int]
    mic_room_id: Optional[int]
    source_room_id: Optional[int]
    room_category: Optional[str]
    same_room: bool
    cross_room: bool
    front_boundary: bool
    fov_boundary: bool
    near_far_tag: str
    difficulty_tag: str
    sources: list[ManifestSource]
    debug_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _round_value(asdict(self))


SampleMetadata = ManifestRow
