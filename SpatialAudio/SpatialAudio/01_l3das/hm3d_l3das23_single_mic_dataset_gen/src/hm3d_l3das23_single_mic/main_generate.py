from __future__ import annotations

import argparse
import logging
import math
import random
from hashlib import blake2b
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .audio_renderer import discover_dry_audio_files, render_spatial_audio, select_dry_audio_file
from .config import DatasetGenerationConfig, dump_config, load_config
from .debug_visualizer import save_front_overlay, save_topdown_debug
from .fov_labeler import CameraModel, compute_in_fov
from .geometry import cylindrical_from_local_xyz, stable_int_from_parts, world_to_local
from .image_renderer import render_sample_images
from .los_labeler import compute_geometry_los, compute_visibility_ratio
from .manifest_io import append_manifest_row, dataset_manifest_path, load_manifest_sample_ids
from .metadata_writer import (
    ensure_sample_layout,
    load_json,
    output_file_map,
    sample_is_complete,
    write_sample_metadata,
)
from .pose_sampler import sample_microphone_poses
from .qc import QCAggregator, build_qc_report_from_existing_metadata, write_qc_report
from .scene_loader import HabitatSceneSession
from .speaker_proxy import build_speaker_proxy_pose, validate_speaker_proxy_pose
from .schemas import (
    CameraIntrinsicsRecord,
    CylindricalCoordinate,
    ManifestSource,
    SampleMetadata,
    SceneInfo,
    SphericalCoordinate,
)
from .source_sampler import generate_source_candidates, validate_source_candidate
from .spatial_conventions import (
    AZIMUTH_CONVENTION,
    AZIMUTH_REFERENCE,
    FOA_CANONICAL_AXES,
    FOA_CANONICAL_CHANNEL_ORDER,
    FOA_RAW_CHANNEL_ORDER,
    LOCAL_COORDINATE_FRAME,
    ambix_unit_vector_xyz,
    azimuth_raw_deg,
    direction_8way_from_azimuth,
    local_angles_from_relative_xyz,
    local_unit_vector_right_front_up,
)
from .split_builder import build_scene_splits, discover_hm3d_scenes, write_split_manifests

LOGGER = logging.getLogger(__name__)


def _progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _filter_scenes(config: DatasetGenerationConfig, scenes: list[SceneInfo]) -> list[SceneInfo]:
    allow = set(config.generation.scene_allowlist)
    deny = set(config.generation.scene_denylist)
    shard_count = int(config.generation.scene_shard_count)
    shard_index = int(config.generation.scene_shard_index)
    if shard_count < 1:
        raise ValueError("generation.scene_shard_count must be at least 1.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "generation.scene_shard_index must be in [0, generation.scene_shard_count)."
        )
    filtered = []
    for scene in scenes:
        if allow and scene.scene_id not in allow:
            continue
        if scene.scene_id in deny:
            continue
        filtered.append(scene)
    if shard_count > 1:
        filtered = [
            scene
            for idx, scene in enumerate(filtered)
            if idx % shard_count == shard_index
        ]
    if config.generation.max_scenes is not None:
        filtered = filtered[: int(config.generation.max_scenes)]
    return filtered


def _prepare_split_map(
    config: DatasetGenerationConfig,
    *,
    mode: str,
) -> tuple[dict[str, SceneInfo], dict[str, list[str]]]:
    scenes = _filter_scenes(config, discover_hm3d_scenes(config))
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    split_map = build_scene_splits(
        sorted(scene_by_id),
        train_ratio=config.splits.train_ratio,
        val_ratio=config.splits.val_ratio,
        test_ratio=config.splits.test_ratio,
        seed=config.splits.seed,
    )
    if mode == "debug":
        limit = int(config.splits.debug_num_scenes_per_split)
        split_map = {
            split_name: sorted(scene_ids)[:limit]
            for split_name, scene_ids in split_map.items()
        }
    return scene_by_id, split_map


def _make_sample_id(scene_id: str, mic_index: int, source_index: int, dry_audio_relpath: str) -> str:
    digest = blake2b(dry_audio_relpath.encode("utf-8"), digest_size=4).hexdigest()
    return f"{scene_id}__mic{mic_index:04d}__src{source_index:04d}__dry_{digest}"


def _make_rir_id(sample_id: str) -> str:
    digest = blake2b(sample_id.encode("utf-8"), digest_size=6).hexdigest()
    return f"rir_{digest}"


def _make_dry_source_id(dry_audio_relpath: str) -> str:
    digest = blake2b(str(dry_audio_relpath).encode("utf-8"), digest_size=6).hexdigest()
    return f"dry_{digest}"


def _make_source_object_instance_id(sample_id: str) -> int:
    return -int(stable_int_from_parts("synthetic_speaker_proxy", sample_id))


def _make_source_id(sample_id: str) -> str:
    return f"{sample_id}__source0"


def _vfov_deg(camera_model: CameraModel) -> float:
    return math.degrees(2.0 * math.atan(float(camera_model.tan_half_vfov)))


def _near_far_tag(distance_to_mic: float) -> str:
    if float(distance_to_mic) <= 2.0:
        return "near"
    if float(distance_to_mic) >= 4.0:
        return "far"
    return "mid"


def _difficulty_tag(
    *,
    geometry_los: str,
    in_fov: bool,
    same_room: bool,
    visible_ratio: float,
    fov_boundary: bool,
    distance_to_mic: float,
) -> str:
    if (
        geometry_los == "gLOS"
        and bool(in_fov)
        and bool(same_room)
        and math.isclose(float(visible_ratio), 1.0, abs_tol=1.0e-6)
        and float(distance_to_mic) <= 2.0
    ):
        return "easy"
    if (
        geometry_los == "gNLOS"
        or (not same_room)
        or float(visible_ratio) < 0.5
        or bool(fov_boundary)
        or float(distance_to_mic) >= 4.0
    ):
        return "hard"
    return "medium"


def _geometry_target(config: DatasetGenerationConfig, geometry_los: str) -> Optional[int]:
    if geometry_los == "gLOS":
        value = config.generation.target_glos_count
    elif geometry_los == "gNLOS":
        value = config.generation.target_gnlos_count
    else:
        return None
    if value is None:
        return None
    return max(0, int(value))


def _geometry_label_allowed(config: DatasetGenerationConfig, geometry_los: str) -> bool:
    allowed = {str(label) for label in config.generation.allowed_geometry_los if str(label)}
    if not allowed:
        return True
    return geometry_los in allowed


def _geometry_quota_reached(
    config: DatasetGenerationConfig,
    qc: QCAggregator,
    geometry_los: str,
) -> bool:
    target = _geometry_target(config, geometry_los)
    if target is None:
        return False
    return int(qc.gcounts.get(geometry_los, 0)) >= target


def _all_geometry_targets_reached(
    config: DatasetGenerationConfig,
    qc: QCAggregator,
) -> bool:
    targets = []
    if config.generation.target_glos_count is not None:
        targets.append(("gLOS", max(0, int(config.generation.target_glos_count))))
    if config.generation.target_gnlos_count is not None:
        targets.append(("gNLOS", max(0, int(config.generation.target_gnlos_count))))
    if not targets:
        return False
    for key, target in targets:
        if int(qc.gcounts.get(key, 0)) < target:
            return False
    return True


def _build_sample_metadata(
    *,
    sample_id: str,
    split: str,
    scene_id: str,
    level_id: Optional[str],
    floor_id: Optional[str],
    room_id: Optional[int],
    room_category: Optional[str],
    mic_room_id: Optional[int],
    source_room_id: Optional[int],
    generation_seed: int,
    mic_pose: Any,
    source_candidate: Any,
    render_source_world_position: list[float],
    speaker_proxy_pose: Any,
    los_result: Any,
    visibility_ratio: float,
    projection_result: Any,
    audio_result: Any,
    camera_model: CameraModel,
    visible_object_ids: list[int],
    distractor_object_ids: list[int],
    layout: dict[str, Path],
    dataset_root: Path,
) -> SampleMetadata:
    notes = [
        "gLOS/gNLOS is geometry-based only, not acoustic-path truth.",
        "camera forward is aligned with microphone azimuth 0 degrees.",
        "FOA is remapped from Habitat/RLR world-axis N3D to mic-local AmbiX/ACN/SN3D WYZX before storage.",
    ]
    if speaker_proxy_pose is not None:
        notes.append(
            "LOS/NLOS is evaluated against the speaker-proxy reference point; the humanoid mesh is rendered for visualization only."
        )
    status = los_result.debug.get("status")
    if isinstance(status, str) and status:
        notes.append(status)
    elif isinstance(status, list):
        notes.extend([str(item) for item in status])
    notes.extend(audio_result.debug_notes)
    output_files = output_file_map(dataset_root, layout)
    candidate_source_index = int(getattr(source_candidate, "source_index", 0))
    render_source_local_xyz = world_to_local(
        mic_pose.position_world,
        mic_pose.yaw_rad,
        render_source_world_position,
    )
    continuous_azimuth_deg, continuous_elevation_deg, distance_to_mic = (
        local_angles_from_relative_xyz(render_source_local_xyz)
    )
    azimuth_continuous_raw_deg = azimuth_raw_deg(continuous_azimuth_deg)
    direction_8way = direction_8way_from_azimuth(continuous_azimuth_deg)
    local_unit_vector = local_unit_vector_right_front_up(render_source_local_xyz)
    ambix_unit_vector = ambix_unit_vector_xyz(render_source_local_xyz)
    rho, theta_rad, z_local = cylindrical_from_local_xyz(render_source_local_xyz)
    render_source_cylindrical = CylindricalCoordinate(
        rho=float(rho),
        theta_rad=float(theta_rad),
        theta_deg=math.degrees(float(theta_rad)),
        z=float(z_local),
    )
    render_source_spherical = SphericalCoordinate(
        distance=float(distance_to_mic),
        azimuth_deg=float(continuous_azimuth_deg),
        elevation_deg=float(continuous_elevation_deg),
    )
    source_visible_binary = bool(projection_result.in_fov) and float(visibility_ratio) > 0.0
    source_object_instance_id = _make_source_object_instance_id(sample_id)
    source_object_category = (
        "synthetic_humanoid_speaker"
        if speaker_proxy_pose is not None
        else "point_audio_source"
    )
    source_id = _make_source_id(sample_id)
    rir_id = _make_rir_id(sample_id)
    dry_source_id = _make_dry_source_id(audio_result.dry_audio_relpath)
    euclidean_distance = distance_to_mic
    direct_path_length = distance_to_mic
    horizontal_fov = float(camera_model.hfov_deg)
    vertical_fov = _vfov_deg(camera_model)
    mic_room_id_value = int(mic_room_id) if mic_room_id is not None else None
    source_room_id_value = int(source_room_id) if source_room_id is not None else None
    room_id_value = int(room_id) if room_id is not None else None
    same_room = (
        mic_room_id_value is not None
        and source_room_id_value is not None
        and mic_room_id_value == source_room_id_value
    )
    fov_boundary = abs(abs(continuous_azimuth_deg) - (horizontal_fov * 0.5)) <= 10.0
    row_debug_notes = list(notes)
    source_record = ManifestSource(
        source_id=source_id,
        source_index=candidate_source_index,
        source_world_position=list(render_source_world_position),
        source_mic_relative_position=list(render_source_local_xyz),
        distance_to_mic=distance_to_mic,
        continuous_azimuth_deg=continuous_azimuth_deg,
        continuous_elevation_deg=continuous_elevation_deg,
        local_coordinate_frame=LOCAL_COORDINATE_FRAME,
        azimuth_reference=AZIMUTH_REFERENCE,
        azimuth_convention=AZIMUTH_CONVENTION,
        azimuth_continuous_raw_deg=azimuth_continuous_raw_deg,
        direction_8way=direction_8way,
        label_8way=direction_8way,
        ambix_unit_vector_xyz=ambix_unit_vector,
        local_unit_vector_right_front_up=local_unit_vector,
        room_id=source_room_id_value,
        room_category=room_category,
        is_los=los_result.geometry_los == "gLOS",
        is_nlos=los_result.geometry_los == "gNLOS",
        is_in_fov=bool(projection_result.in_fov),
        is_out_of_fov=not bool(projection_result.in_fov),
        visible_ratio=float(visibility_ratio),
        source_visible_binary=source_visible_binary,
        euclidean_distance=euclidean_distance,
        direct_path_length=direct_path_length,
        source_associated_object_id=source_object_instance_id,
        source_object_instance_id=source_object_instance_id,
        source_object_category=source_object_category,
        rir_id=rir_id,
        dry_source_id=dry_source_id,
        source_type="speech",
        snr=None,
        onset_time=0.0,
        offset_time=float(audio_result.num_samples) / float(audio_result.output_sample_rate),
    )
    return SampleMetadata(
        sample_id=sample_id,
        split=split,
        scene_id=scene_id,
        room_id=room_id_value,
        source_id=source_id,
        sequence_id_or_episode_id=None,
        manifest_version="1.0",
        level_id=level_id,
        floor_id=floor_id,
        generation_seed=generation_seed,
        mic_pose_world=mic_pose.to_pose_record(),
        camera_pose_world=mic_pose.to_pose_record(),
        source_pose_world=list(render_source_world_position),
        source_pose_local_xyz=list(render_source_local_xyz),
        source_pose_cylindrical=render_source_cylindrical,
        source_pose_spherical=render_source_spherical,
        speaker_proxy_avatar=(
            str(speaker_proxy_pose.avatar_name) if speaker_proxy_pose is not None else None
        ),
        speaker_proxy_root_world=(
            list(speaker_proxy_pose.root_world) if speaker_proxy_pose is not None else None
        ),
        speaker_proxy_reference_world=(
            list(speaker_proxy_pose.reference_world) if speaker_proxy_pose is not None else None
        ),
        speaker_proxy_floor_anchor_world=(
            list(speaker_proxy_pose.floor_anchor_world) if speaker_proxy_pose is not None else None
        ),
        speaker_proxy_yaw_deg=(
            float(speaker_proxy_pose.yaw_deg) if speaker_proxy_pose is not None else None
        ),
        geometry_los=los_result.geometry_los,
        geometry_los_stable=los_result.stable,
        in_fov=projection_result.in_fov,
        projected_pixel_xy=projection_result.pixel_xy,
        projection_depth_cam=projection_result.depth_cam,
        projection_reason=projection_result.reason,
        source_distance=distance_to_mic,
        azimuth_deg=continuous_azimuth_deg,
        elevation_deg=continuous_elevation_deg,
        occlusion_hit_distance=los_result.occlusion_hit_distance,
        occluding_object_id=los_result.occluding_object_id,
        occluder_count=int(los_result.occluder_count),
        first_occluder_instance_id=los_result.occluding_object_id,
        first_occluder_category=None,
        visibility_ratio=float(visibility_ratio),
        visible_ratio=float(visibility_ratio),
        dry_audio_filename=audio_result.dry_audio_relpath,
        dry_audio_num_samples=audio_result.dry_audio_num_samples,
        dry_audio_sample_rate=audio_result.dry_audio_sample_rate,
        secondary_dry_audio_filename=audio_result.secondary_dry_audio_relpath,
        secondary_dry_audio_num_samples=audio_result.secondary_dry_audio_num_samples,
        secondary_dry_audio_sample_rate=audio_result.secondary_dry_audio_sample_rate,
        rir_generation_status=audio_result.rir_generation_status,
        rendering_status=audio_result.rendering_status,
        secondary_rendering_status=audio_result.secondary_rendering_status,
        output_files=output_files,
        camera_world_position=list(mic_pose.position_world),
        camera_world_rotation=list(mic_pose.quaternion_wxyz),
        mic_world_position=list(mic_pose.position_world),
        mic_world_rotation=list(mic_pose.quaternion_wxyz),
        camera_intrinsics=CameraIntrinsicsRecord(
            fx=float(camera_model.fx),
            fy=float(camera_model.fy),
            cx=float(camera_model.cx),
            cy=float(camera_model.cy),
        ),
        image_width=int(camera_model.width),
        image_height=int(camera_model.height),
        horizontal_fov=horizontal_fov,
        vertical_fov=vertical_fov,
        camera_mic_axis_aligned=True,
        foa_audio_path=output_files["audio_foa_wav"],
        rgb_image_path=output_files["rgb_front_png"],
        depth_image_path=output_files["depth_npy"],
        instance_mask_path=output_files["instance_mask_npy"],
        metadata_path=output_files["metadata_json"],
        audio_format="foa_ambisonics_4ch",
        audio_channel_layout="ambisonics",
        audio_channel_order=FOA_RAW_CHANNEL_ORDER,
        foa_raw_channel_order=FOA_RAW_CHANNEL_ORDER,
        foa_canonical_channel_order=FOA_CANONICAL_CHANNEL_ORDER,
        foa_canonical_axes=FOA_CANONICAL_AXES,
        source_world_position=list(render_source_world_position),
        source_mic_relative_position=list(render_source_local_xyz),
        distance_to_mic=distance_to_mic,
        continuous_azimuth_deg=continuous_azimuth_deg,
        continuous_elevation_deg=continuous_elevation_deg,
        local_coordinate_frame=LOCAL_COORDINATE_FRAME,
        azimuth_reference=AZIMUTH_REFERENCE,
        azimuth_convention=AZIMUTH_CONVENTION,
        azimuth_continuous_raw_deg=azimuth_continuous_raw_deg,
        direction_8way=direction_8way,
        label_8way=direction_8way,
        ambix_unit_vector_xyz=ambix_unit_vector,
        local_unit_vector_right_front_up=local_unit_vector,
        is_los=los_result.geometry_los == "gLOS",
        is_nlos=los_result.geometry_los == "gNLOS",
        is_in_fov=bool(projection_result.in_fov),
        is_out_of_fov=not bool(projection_result.in_fov),
        source_visible_binary=source_visible_binary,
        source_associated_object_id=source_object_instance_id,
        euclidean_distance=euclidean_distance,
        direct_path_length=direct_path_length,
        los_definition="geometry_defined",
        rir_id=rir_id,
        dry_source_id=dry_source_id,
        source_type="speech",
        num_active_sources=1,
        snr=None,
        onset_time=0.0,
        offset_time=float(audio_result.num_samples) / float(audio_result.output_sample_rate),
        source_object_instance_id=source_object_instance_id,
        source_object_category=source_object_category,
        visible_object_ids=list(visible_object_ids),
        distractor_object_ids=list(distractor_object_ids),
        mic_room_id=mic_room_id_value,
        source_room_id=source_room_id_value,
        room_category=room_category,
        same_room=same_room,
        cross_room=not same_room,
        front_boundary=abs(continuous_azimuth_deg) <= 15.0,
        fov_boundary=fov_boundary,
        near_far_tag=_near_far_tag(distance_to_mic),
        difficulty_tag=_difficulty_tag(
            geometry_los=str(los_result.geometry_los),
            in_fov=bool(projection_result.in_fov),
            same_room=same_room,
            visible_ratio=float(visibility_ratio),
            fov_boundary=fov_boundary,
            distance_to_mic=distance_to_mic,
        ),
        sources=[source_record],
        debug_notes=row_debug_notes,
    )


def _process_scene(
    scene_info: SceneInfo,
    split: str,
    config: DatasetGenerationConfig,
    dry_audio_files: list[Path],
    secondary_dry_audio_files: Optional[list[Path]],
    qc: QCAggregator,
    existing_manifest_sample_ids: set[str],
    *,
    mode: str,
) -> bool:
    mic_limit = config.splits.debug_max_mics_per_scene if mode == "debug" else None
    source_limit = config.splits.debug_max_sources_per_mic if mode == "debug" else None
    max_scene_samples = config.generation.max_valid_samples_per_scene
    scene_valid_samples = 0

    with HabitatSceneSession(config, scene_info) as session:
        mic_seed = stable_int_from_parts(config.generation.seed, scene_info.scene_id, "mics")
        mic_poses, _ = sample_microphone_poses(
            session,
            config,
            seed=mic_seed,
            max_poses=mic_limit,
        )
        qc.record_mic_candidates(len(mic_poses))
        camera_model = CameraModel.from_hfov(
            config.sensor_rig.rgb_width,
            config.sensor_rig.rgb_height,
            config.sensor_rig.hfov_deg,
        )

        for mic_pose in mic_poses:
            if max_scene_samples is not None and scene_valid_samples >= int(max_scene_samples):
                break
            if config.generation.stop_when_target_reached and _all_geometry_targets_reached(config, qc):
                return True
            source_seed = stable_int_from_parts(
                config.generation.seed,
                scene_info.scene_id,
                mic_pose.mic_index,
                "sources",
            )
            source_candidates = generate_source_candidates(
                mic_pose,
                config.source_sampling,
                seed=source_seed,
                max_sources=source_limit,
            )
            for candidate in source_candidates:
                if max_scene_samples is not None and scene_valid_samples >= int(max_scene_samples):
                    break
                qc.record_source_candidate()
                is_valid, reject_reason, _ = validate_source_candidate(
                    session,
                    mic_pose,
                    candidate,
                    config.source_sampling,
                )
                if not is_valid:
                    qc.record_failure(reject_reason or "invalid_source")
                    continue

                speaker_proxy_pose = build_speaker_proxy_pose(
                    session,
                    mic_pose,
                    candidate.world_xyz,
                    config,
                )
                proxy_is_valid, proxy_reject_reason, _ = validate_speaker_proxy_pose(
                    session,
                    speaker_proxy_pose,
                    config,
                )
                if not proxy_is_valid:
                    qc.record_failure(proxy_reject_reason or "invalid_speaker_proxy")
                    continue
                source_room_anchor = (
                    speaker_proxy_pose.floor_anchor_world
                    if speaker_proxy_pose is not None
                    else candidate.world_xyz
                )
                los_target_world = (
                    speaker_proxy_pose.reference_world
                    if speaker_proxy_pose is not None
                    else candidate.world_xyz
                )
                los_result = compute_geometry_los(
                    session.sim,
                    mic_pose.position_world,
                    los_target_world,
                    config.los.eps_start_m,
                    config.los.eps_end_m,
                    config.los.max_dist_margin_m,
                    ignore_hits_within_m=config.los.ignore_hits_within_m,
                    audio_sensor=session.audio_sensor,
                    listener_quat_wxyz=mic_pose.quaternion_wxyz,
                    use_audio_visibility_fallback=config.los.use_audio_visibility_fallback_when_raycast_empty,
                    bidirectional_consistency_check=config.los.bidirectional_consistency_check,
                    conservative_on_bidirectional_disagreement=config.los.conservative_on_bidirectional_disagreement,
                    mark_raycast_empty_unstable_without_fallback=config.los.mark_raycast_empty_unstable_without_fallback,
                )
                if config.qc.reject_invalid_los and config.los.reject_endpoint_ambiguity and not los_result.stable:
                    qc.record_failure("unstable_geometry_los")
                    continue
                if not _geometry_label_allowed(config, los_result.geometry_los):
                    qc.record_failure(f"filtered_geometry_los_{los_result.geometry_los}")
                    continue
                if _geometry_quota_reached(config, qc, los_result.geometry_los):
                    qc.record_failure(f"quota_reached_{los_result.geometry_los}")
                    continue

                projection_result = compute_in_fov(
                    camera_model,
                    mic_pose.position_world,
                    mic_pose.yaw_rad,
                    los_target_world,
                )
                if config.qc.reject_invalid_projection and projection_result.in_fov and projection_result.pixel_xy is None:
                    qc.record_failure("invalid_projection_state")
                    continue
                if (
                    config.generation.required_in_fov is not None
                    and bool(projection_result.in_fov) != bool(config.generation.required_in_fov)
                ):
                    required_label = "true" if bool(config.generation.required_in_fov) else "false"
                    actual_label = "true" if bool(projection_result.in_fov) else "false"
                    qc.record_failure(f"filtered_in_fov_required_{required_label}_got_{actual_label}")
                    continue

                visibility_ratio = None
                if config.los.visibility_ratio_enabled:
                    visibility_ratio = compute_visibility_ratio(
                        session.sim,
                        mic_pose.position_world,
                        los_target_world,
                        sphere_radius_m=config.los.visibility_sphere_radius_m,
                        num_rays=config.los.visibility_num_rays,
                        eps_start=config.los.eps_start_m,
                        eps_end=config.los.eps_end_m,
                        max_dist_margin=config.los.max_dist_margin_m,
                        ignore_hits_within_m=config.los.ignore_hits_within_m,
                    )

                dry_audio_path = select_dry_audio_file(
                    dry_audio_files,
                    selection_key=f"{config.generation.seed}:{split}:{scene_info.scene_id}:{mic_pose.mic_index}:{candidate.source_index}",
                )
                dry_audio_relpath = (
                    dry_audio_path.relative_to(config.paths.dry_audio_root)
                    if dry_audio_path.is_relative_to(config.paths.dry_audio_root)
                    else dry_audio_path.name
                )

                secondary_dry_audio_path: Optional[Path] = None
                secondary_dry_audio_relpath: Optional[str] = None
                if secondary_dry_audio_files and config.audio.write_mic_librispeech_wav:
                    secondary_dry_audio_path = select_dry_audio_file(
                        secondary_dry_audio_files,
                        selection_key=(
                            f"{config.generation.seed}:{split}:{scene_info.scene_id}:"
                            f"{mic_pose.mic_index}:{candidate.source_index}:secondary"
                        ),
                    )
                    secondary_root = config.paths.convolution_dry_audio_root
                    if (
                        secondary_root is not None
                        and secondary_dry_audio_path.is_relative_to(secondary_root)
                    ):
                        secondary_dry_audio_relpath = str(
                            secondary_dry_audio_path.relative_to(secondary_root)
                        )
                    else:
                        secondary_dry_audio_relpath = secondary_dry_audio_path.name

                sample_id = _make_sample_id(
                    scene_info.scene_id,
                    mic_pose.mic_index,
                    candidate.source_index,
                    str(dry_audio_relpath),
                )
                layout = ensure_sample_layout(config.paths.dataset_root, scene_info.scene_id, sample_id)

                if (
                    config.generation.resume
                    and not config.generation.overwrite
                    and sample_is_complete(layout, config)
                ):
                    metadata_row = load_json(layout["metadata_json"])
                    if sample_id not in existing_manifest_sample_ids:
                        append_manifest_row(config.paths.dataset_root, metadata_row)
                        existing_manifest_sample_ids.add(sample_id)
                    qc.record_sample(metadata_row, skipped_existing=True)
                    scene_valid_samples += 1
                    if config.generation.stop_when_target_reached and _all_geometry_targets_reached(config, qc):
                        return True
                    continue

                image_result = None
                image_cache: dict[str, Any] = {}
                if bool(config.generation.audio_only):
                    visible_object_ids = []
                    distractor_object_ids = []
                    mic_room_id, mic_room_category = session.infer_room_info(
                        mic_pose.floor_point_world
                    )
                    source_room_id, source_room_category = session.infer_room_info(
                        source_room_anchor
                    )
                else:
                    image_result, image_cache = render_sample_images(
                        session,
                        mic_pose,
                        layout,
                        config,
                        speaker_proxy_pose=speaker_proxy_pose,
                    )
                    if image_cache.get("depth") is None:
                        qc.record_failure("missing_depth_sensor_output")
                        continue
                    if image_cache.get("semantic") is None:
                        qc.record_failure("missing_semantic_sensor_output")
                        continue
                    visible_object_ids = session.visible_object_ids_from_mask(image_cache["semantic"])
                    if not visible_object_ids:
                        qc.record_failure("empty_visible_object_ids")
                        continue
                    distractor_object_ids = list(visible_object_ids)
                    mic_room_id, mic_room_category = session.infer_room_info_from_semantic_mask(
                        image_cache["semantic"]
                    )
                    if mic_room_id is None:
                        mic_room_id, mic_room_category = session.infer_room_info(
                            mic_pose.floor_point_world
                        )
                    if mic_room_id is None:
                        qc.record_failure("missing_mic_room_id")
                        continue
                    source_room_id, source_room_category = session.infer_room_info(source_room_anchor)
                    if source_room_id is None:
                        qc.record_failure("missing_source_room_id")
                        continue
                audio_result = render_spatial_audio(
                    session,
                    mic_pose,
                    los_target_world,
                    dry_audio_path,
                    layout,
                    config.audio,
                    dry_audio_relpath=str(dry_audio_relpath),
                    secondary_dry_audio_path=secondary_dry_audio_path,
                    secondary_dry_audio_relpath=secondary_dry_audio_relpath,
                )

                if config.qc.reject_nan_audio and audio_result.rendering_status == "nan_audio":
                    qc.record_failure("nan_audio")
                    continue
                if config.qc.reject_audio_clipped and audio_result.rendering_status == "audio_clipped":
                    qc.record_failure("audio_clipped")
                    continue
                if config.qc.reject_audio_silent and audio_result.rendering_status == "audio_silent":
                    qc.record_failure("audio_silent")
                    continue
                if audio_result.rendering_status != "success":
                    qc.record_failure(audio_result.rendering_status)
                    continue
                if (
                    secondary_dry_audio_files
                    and audio_result.secondary_rendering_status not in (None, "success")
                ):
                    qc.record_failure(str(audio_result.secondary_rendering_status))
                    continue

                if config.generation.save_topdown_debug:
                    save_topdown_debug(
                        layout["topdown_debug_png"],
                        mic_pose.position_world,
                        mic_pose.yaw_rad,
                        los_target_world,
                        geometry_los=los_result.geometry_los,
                        in_fov=projection_result.in_fov,
                        title=sample_id,
                        pathfinder=session.sim.pathfinder if session.sim is not None else None,
                        floor_y=(
                            float(mic_pose.floor_point_world[1])
                            if mic_pose.floor_point_world and len(mic_pose.floor_point_world) > 1
                            else None
                        ),
                        source_anchor_world=(
                            speaker_proxy_pose.floor_anchor_world
                            if speaker_proxy_pose is not None
                            else None
                        ),
                    )
                if (
                    config.generation.save_front_overlay
                    and not bool(config.generation.audio_only)
                    and image_result is not None
                    and image_cache.get("rgb") is not None
                ):
                    save_front_overlay(
                        layout["front_overlay_png"],
                        image_cache["rgb"],
                        projected_pixel_xy=projection_result.pixel_xy,
                        geometry_los=los_result.geometry_los,
                        projection_reason=projection_result.reason,
                        title=sample_id,
                    )
                    image_result.front_overlay_path = layout["front_overlay_png"]

                metadata = _build_sample_metadata(
                    sample_id=sample_id,
                    split=split,
                    scene_id=scene_info.scene_id,
                    level_id=session.infer_level_id(candidate.world_xyz),
                    floor_id=session.infer_level_id(mic_pose.position_world),
                    room_id=source_room_id,
                    room_category=source_room_category,
                    mic_room_id=mic_room_id,
                    source_room_id=source_room_id,
                    generation_seed=stable_int_from_parts(config.generation.seed, sample_id),
                    mic_pose=mic_pose,
                    source_candidate=candidate,
                    render_source_world_position=los_target_world,
                    speaker_proxy_pose=speaker_proxy_pose,
                    los_result=los_result,
                    visibility_ratio=float(visibility_ratio or 0.0),
                    projection_result=projection_result,
                    audio_result=audio_result,
                    camera_model=camera_model,
                    visible_object_ids=visible_object_ids,
                    distractor_object_ids=distractor_object_ids,
                    layout=layout,
                    dataset_root=config.paths.dataset_root,
                )
                metadata.first_occluder_category = session.lookup_object_category(
                    metadata.first_occluder_instance_id
                )
                write_sample_metadata(layout["metadata_json"], metadata)
                if sample_id not in existing_manifest_sample_ids:
                    append_manifest_row(config.paths.dataset_root, metadata.to_dict())
                    existing_manifest_sample_ids.add(sample_id)
                qc.record_sample(metadata)
                scene_valid_samples += 1
                if config.generation.stop_when_target_reached and _all_geometry_targets_reached(config, qc):
                    return True

    return False


def build_splits_command(config: DatasetGenerationConfig, *, mode: str) -> dict[str, list[str]]:
    _, split_map = _prepare_split_map(config, mode=mode)
    write_split_manifests(config, split_map)
    return split_map


def generate_command(config: DatasetGenerationConfig, *, mode: str) -> dict[str, Any]:
    # gLOS/gNLOS in this pipeline is defined from scene-geometry visibility.
    # Primary path is ray casting; optional fallback uses audio sensor
    # sourceIsVisible (also geometric visibility) when ray hits are empty.
    # Both rely on collision support from the simulator build. When the
    # humanoid speaker proxy is enabled, the LOS/NLOS target is the proxy
    # reference point, while the humanoid mesh itself is rendered only for RGB.
    if not bool(config.simulator.enable_physics):
        raise ValueError(
            "simulator.enable_physics must be true for geometric gLOS/gNLOS labeling. "
            "Current config has enable_physics=false, which can collapse labels to gLOS."
        )
    audio_only = bool(config.generation.audio_only)
    if not audio_only and not bool(config.sensor_rig.enable_depth):
        raise ValueError("sensor_rig.enable_depth must be true for the FOA+Vision dataset manifest.")
    if not audio_only and not bool(config.sensor_rig.enable_semantic):
        raise ValueError("sensor_rig.enable_semantic must be true for strict room/semantic annotations.")
    if not bool(config.los.visibility_ratio_enabled):
        raise ValueError("los.visibility_ratio_enabled must be true for visible_ratio metadata.")
    if not audio_only and not bool(config.speaker_proxy.enabled):
        raise ValueError("speaker_proxy.enabled must be true because source object semantics use the synthetic speaker proxy.")
    if str(config.audio.channel_layout).lower() != "ambisonics" or int(config.audio.channel_count) != 4:
        raise ValueError(
            "audio.channel_layout must be 'ambisonics' and audio.channel_count must be 4 for FOA output."
        )

    scene_by_id, split_map = _prepare_split_map(config, mode=mode)
    manifest_path = dataset_manifest_path(config.paths.dataset_root)
    if bool(config.generation.overwrite) and manifest_path.exists():
        manifest_path.unlink()
    write_split_manifests(config, split_map)
    dry_audio_files = discover_dry_audio_files(config.paths.dry_audio_root, config.paths.dry_audio_glob)
    if not dry_audio_files:
        raise FileNotFoundError(
            f"No dry audio files found under {config.paths.dry_audio_root} with pattern {config.paths.dry_audio_glob}"
        )
    secondary_dry_audio_files: Optional[list[Path]] = None
    if (
        config.audio.write_mic_librispeech_wav
        and config.paths.convolution_dry_audio_root is not None
    ):
        secondary_dry_audio_files = discover_dry_audio_files(
            config.paths.convolution_dry_audio_root,
            config.paths.convolution_dry_audio_glob,
        )
        if not secondary_dry_audio_files:
            raise FileNotFoundError(
                "No secondary dry audio files found under "
                f"{config.paths.convolution_dry_audio_root} "
                f"with pattern {config.paths.convolution_dry_audio_glob}"
            )

    qc = QCAggregator()
    existing_manifest_sample_ids = load_manifest_sample_ids(config.paths.dataset_root)
    selected_splits = set(config.generation.splits)
    stop_requested = False
    for split in ("train", "val", "test"):
        if split not in selected_splits:
            continue
        scene_ids = list(split_map.get(split, []))
        if bool(config.generation.shuffle_scene_order):
            scene_seed = stable_int_from_parts(config.generation.seed, split, "scene_order")
            random.Random(scene_seed).shuffle(scene_ids)
        for scene_id in _progress(scene_ids, desc=f"{split} scenes", leave=False):
            if config.generation.stop_when_target_reached and _all_geometry_targets_reached(config, qc):
                stop_requested = True
                break
            qc.record_scene_processed()
            try:
                stop_requested = _process_scene(
                    scene_by_id[scene_id],
                    split,
                    config,
                    dry_audio_files,
                    secondary_dry_audio_files,
                    qc,
                    existing_manifest_sample_ids=existing_manifest_sample_ids,
                    mode=mode,
                )
                if stop_requested:
                    break
            except Exception as exc:
                LOGGER.exception("Scene processing failed for %s", scene_id)
                qc.record_scene_failure(scene_id, str(exc))
                if config.generation.fail_fast:
                    raise
        if stop_requested:
            break

    write_split_manifests(config, split_map)
    report = qc.to_dict()
    write_qc_report(config.paths.dataset_root, report)
    return report


def qc_command(dataset_root: Path) -> dict[str, Any]:
    return build_qc_report_from_existing_metadata(dataset_root)


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HM3D single-mic dataset generation pipeline")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_splits = subparsers.add_parser("build-splits", help="Build deterministic scene split manifests.")
    build_splits.add_argument("--config", type=Path, required=True)
    build_splits.add_argument("--mode", choices=["debug", "full"], default="debug")

    generate = subparsers.add_parser("generate", help="Run dataset generation.")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--mode", choices=["debug", "full"], default="debug")

    qc_parser = subparsers.add_parser("qc", help="Build a simple QC summary from existing sample metadata.")
    qc_parser.add_argument("--dataset-root", type=Path, required=True)

    dump = subparsers.add_parser("dump-config", help="Write the resolved config to disk.")
    dump.add_argument("--config", type=Path, required=True)
    dump.add_argument("--output", type=Path, required=True)

    return parser


def main() -> None:
    parser = make_arg_parser()
    args = parser.parse_args()
    _configure_logging(bool(args.verbose))

    if args.command == "qc":
        report = qc_command(args.dataset_root)
        print(report)
        return

    config = load_config(args.config)

    if args.command == "build-splits":
        split_map = build_splits_command(config, mode=args.mode)
        print(split_map)
        return

    if args.command == "generate":
        report = generate_command(config, mode=args.mode)
        print(report)
        return

    if args.command == "dump-config":
        dump_config(config, args.output)
        print({"output": str(args.output)})
        return


if __name__ == "__main__":
    main()
