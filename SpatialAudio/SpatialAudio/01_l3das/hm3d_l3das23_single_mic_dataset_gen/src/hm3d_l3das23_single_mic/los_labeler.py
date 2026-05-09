from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .geometry import distance_m
from .schemas import GeometryLosResult


@dataclass
class RayHit:
    ray_distance: float
    object_id: Optional[int] = None


@dataclass
class RayCastEvaluation:
    result: GeometryLosResult
    hits_raw: list[RayHit]
    hits_filtered: list[RayHit]
    origin: np.ndarray
    direction: np.ndarray
    effective_distance: float


def classify_geometry_los_from_hits(
    hits: Sequence[RayHit],
    *,
    source_distance: float,
    eps_end: float,
    max_dist_margin: float,
    ignore_hits_within: float = 0.0,
) -> GeometryLosResult:
    filtered = sorted(
        (
            hit
            for hit in hits
            if float(hit.ray_distance) > float(ignore_hits_within)
        ),
        key=lambda hit: float(hit.ray_distance),
    )

    if not filtered:
        return GeometryLosResult(
            geometry_los="gLOS",
            first_hit_distance=None,
            source_distance=float(source_distance),
            hit_object_id=None,
            stable=True,
            occlusion_hit_distance=None,
            occluding_object_id=None,
            debug={"status": "no_hit_before_max_distance"},
        )

    first = filtered[0]
    obstruction_limit = max(float(source_distance) - float(eps_end), 0.0)
    endpoint_limit = float(source_distance) + float(max_dist_margin)
    occluders = [
        hit
        for hit in filtered
        if float(hit.ray_distance) < obstruction_limit
    ]

    if float(first.ray_distance) < obstruction_limit:
        return GeometryLosResult(
            geometry_los="gNLOS",
            first_hit_distance=float(first.ray_distance),
            source_distance=float(source_distance),
            hit_object_id=first.object_id,
            stable=True,
            occlusion_hit_distance=float(first.ray_distance),
            occluding_object_id=first.object_id,
            occluder_count=len(occluders),
            debug={"status": "hit_before_source_endpoint"},
        )

    if float(first.ray_distance) <= endpoint_limit:
        return GeometryLosResult(
            geometry_los="gLOS",
            first_hit_distance=float(first.ray_distance),
            source_distance=float(source_distance),
            hit_object_id=first.object_id,
            stable=False,
            occlusion_hit_distance=None,
            occluding_object_id=None,
            occluder_count=0,
            debug={"status": "endpoint_ambiguity"},
        )

    return GeometryLosResult(
        geometry_los="gLOS",
        first_hit_distance=float(first.ray_distance),
        source_distance=float(source_distance),
        hit_object_id=first.object_id,
        stable=True,
        occlusion_hit_distance=None,
        occluding_object_id=None,
        occluder_count=0,
        debug={"status": "first_hit_beyond_source"},
    )


def _cast_single_geometry_los_ray(
    sim: Any,
    start_pos_world: np.ndarray,
    end_pos_world: np.ndarray,
    *,
    eps_start: float,
    eps_end: float,
    max_dist_margin: float,
    ignore_hits_within_m: float,
) -> RayCastEvaluation:
    try:
        import habitat_sim  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "habitat_sim is required for geometric LOS ray casting."
        ) from exc

    total_distance = distance_m(start_pos_world, end_pos_world)
    if total_distance <= 0.0:
        raise ValueError("start and end positions must be distinct")

    direction = (end_pos_world - start_pos_world) / total_distance
    origin = start_pos_world + (direction * float(eps_start))
    effective_distance = max(total_distance - float(eps_start), 0.0)

    ray = habitat_sim.geo.Ray(
        origin.astype(np.float32),
        direction.astype(np.float32),
    )
    results = sim.cast_ray(
        ray=ray,
        max_distance=float(effective_distance + float(max_dist_margin)),
    )

    hits_raw = [
        RayHit(ray_distance=float(hit.ray_distance), object_id=int(hit.object_id))
        for hit in getattr(results, "hits", [])
    ]
    hits_filtered = sorted(
        [
            hit
            for hit in hits_raw
            if float(hit.ray_distance) > float(ignore_hits_within_m)
        ],
        key=lambda item: float(item.ray_distance),
    )
    result = classify_geometry_los_from_hits(
        hits_raw,
        source_distance=effective_distance,
        eps_end=float(eps_end),
        max_dist_margin=float(max_dist_margin),
        ignore_hits_within=float(ignore_hits_within_m),
    )
    return RayCastEvaluation(
        result=result,
        hits_raw=hits_raw,
        hits_filtered=hits_filtered,
        origin=origin,
        direction=direction,
        effective_distance=float(effective_distance),
    )


def reconcile_bidirectional_los(
    forward: GeometryLosResult,
    reverse: Optional[GeometryLosResult],
    *,
    conservative_on_disagreement: bool = True,
) -> GeometryLosResult:
    if reverse is None:
        return forward

    forward_status = str(forward.debug.get("status", ""))
    reverse_status = str(reverse.debug.get("status", ""))

    if forward.geometry_los != reverse.geometry_los:
        if conservative_on_disagreement:
            forward.geometry_los = "gNLOS"
            if forward.occlusion_hit_distance is None:
                forward.occlusion_hit_distance = reverse.occlusion_hit_distance
                forward.occluding_object_id = reverse.occluding_object_id
        forward.stable = False
        forward.debug.update(
            {
                "status": "forward_reverse_disagreement",
                "forward_status": forward_status,
                "reverse_status": reverse_status,
            }
        )
        return forward

    if (not forward.stable) or (not reverse.stable):
        forward.stable = False
        forward.debug.update(
            {
                "status": "forward_reverse_endpoint_ambiguity",
                "forward_status": forward_status,
                "reverse_status": reverse_status,
            }
        )
    return forward


def compute_geometry_los(
    sim: Any,
    mic_pos_world: Iterable[float],
    source_pos_world: Iterable[float],
    eps_start: float,
    eps_end: float,
    max_dist_margin: float,
    *,
    ignore_hits_within_m: float = 0.0,
    audio_sensor: Optional[Any] = None,
    listener_quat_wxyz: Optional[Iterable[float]] = None,
    use_audio_visibility_fallback: bool = False,
    bidirectional_consistency_check: bool = True,
    conservative_on_bidirectional_disagreement: bool = True,
    mark_raycast_empty_unstable_without_fallback: bool = True,
) -> GeometryLosResult:
    """
    gLOS/gNLOS is determined by scene-mesh visibility only.
    This is not a claim about acoustic direct-path dominance.
    """
    mic = np.asarray(list(mic_pos_world), dtype=np.float64)
    source = np.asarray(list(source_pos_world), dtype=np.float64)
    forward_eval = _cast_single_geometry_los_ray(
        sim,
        mic,
        source,
        eps_start=float(eps_start),
        eps_end=float(eps_end),
        max_dist_margin=float(max_dist_margin),
        ignore_hits_within_m=float(ignore_hits_within_m),
    )
    output = forward_eval.result
    output.debug.update(
        {
            "ray_origin": forward_eval.origin.astype(float).tolist(),
            "ray_direction": forward_eval.direction.astype(float).tolist(),
            "effective_distance": float(forward_eval.effective_distance),
            "num_hits_raw": len(forward_eval.hits_raw),
            "num_hits_filtered": len(forward_eval.hits_filtered),
        }
    )

    reverse_eval: Optional[RayCastEvaluation] = None
    if bool(bidirectional_consistency_check):
        reverse_eval = _cast_single_geometry_los_ray(
            sim,
            source,
            mic,
            eps_start=float(eps_start),
            eps_end=float(eps_end),
            max_dist_margin=float(max_dist_margin),
            ignore_hits_within_m=float(ignore_hits_within_m),
        )
        output = reconcile_bidirectional_los(
            output,
            reverse_eval.result,
            conservative_on_disagreement=bool(conservative_on_bidirectional_disagreement),
        )
        output.debug.update(
            {
                "reverse_ray_origin": reverse_eval.origin.astype(float).tolist(),
                "reverse_ray_direction": reverse_eval.direction.astype(float).tolist(),
                "reverse_effective_distance": float(reverse_eval.effective_distance),
                "reverse_num_hits_raw": len(reverse_eval.hits_raw),
                "reverse_num_hits_filtered": len(reverse_eval.hits_filtered),
            }
        )

    raycast_empty = len(forward_eval.hits_filtered) == 0 and (
        reverse_eval is None or len(reverse_eval.hits_filtered) == 0
    )
    output.debug["raycast_empty"] = bool(raycast_empty)

    # Fallback for environments where Simulator.cast_ray returns empty hits
    # consistently (e.g., missing physics collision backend). We still keep
    # semantics geometric: sourceIsVisible checks direct geometric visibility.
    if (
        raycast_empty
        and use_audio_visibility_fallback
        and audio_sensor is not None
        and hasattr(audio_sensor, "sourceIsVisible")
    ):
        try:
            src_np = np.asarray(source, dtype=np.float32)
            mic_np = np.asarray(mic, dtype=np.float32)
            audio_sensor.setAudioSourceTransform(src_np)
            if hasattr(audio_sensor, "setAudioListenerTransform"):
                if listener_quat_wxyz is None:
                    quat_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    quat_np = np.asarray(list(listener_quat_wxyz), dtype=np.float32)
                audio_sensor.setAudioListenerTransform(mic_np, quat_np)
            visible = bool(audio_sensor.sourceIsVisible())
            output.geometry_los = "gLOS" if visible else "gNLOS"
            output.stable = True
            if not visible:
                output.occlusion_hit_distance = None
                output.occluding_object_id = None
                output.occluder_count = 0
            output.debug["status"] = (
                "raycast_empty_fallback_audio_visible"
                if visible
                else "raycast_empty_fallback_audio_occluded"
            )
        except Exception as exc:
            output.debug["fallback_error"] = str(exc)

    if (
        raycast_empty
        and not use_audio_visibility_fallback
        and bool(mark_raycast_empty_unstable_without_fallback)
    ):
        output.stable = False
        output.debug["status"] = "raycast_empty_without_fallback"

    return output


def compute_visibility_ratio(
    sim: Any,
    mic_pos_world: Iterable[float],
    source_pos_world: Iterable[float],
    *,
    sphere_radius_m: float,
    num_rays: int,
    eps_start: float,
    eps_end: float,
    max_dist_margin: float,
    ignore_hits_within_m: float = 0.0,
) -> float:
    source = np.asarray(list(source_pos_world), dtype=np.float64)
    if num_rays <= 1 or sphere_radius_m <= 0.0:
        result = compute_geometry_los(
            sim,
            mic_pos_world,
            source_pos_world,
            eps_start,
            eps_end,
            max_dist_margin,
            ignore_hits_within_m=ignore_hits_within_m,
        )
        return 1.0 if result.geometry_los == "gLOS" else 0.0

    directions = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for idx in range(max(0, num_rays - 1)):
        y = 1.0 - (2.0 * idx) / max(1, num_rays - 2)
        radius = max(0.0, 1.0 - y * y) ** 0.5
        theta = golden_angle * idx
        directions.append(
            np.array(
                [np.cos(theta) * radius, y, np.sin(theta) * radius],
                dtype=np.float64,
            )
        )

    visible = 0
    for direction in directions[:num_rays]:
        target = source + (direction * float(sphere_radius_m))
        result = compute_geometry_los(
            sim,
            mic_pos_world,
            target,
            eps_start,
            eps_end,
            max_dist_margin,
            ignore_hits_within_m=ignore_hits_within_m,
        )
        if result.geometry_los == "gLOS":
            visible += 1
    return float(visible) / float(num_rays)
