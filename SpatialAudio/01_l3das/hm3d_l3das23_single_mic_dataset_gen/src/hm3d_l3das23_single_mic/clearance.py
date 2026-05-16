from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class PointClearanceResult:
    min_hit_distance: float
    max_hit_distance: float
    num_rays_with_hits: int
    num_rays_tested: int

    @property
    def hit_fraction(self) -> float:
        if self.num_rays_tested <= 0:
            return 0.0
        return float(self.num_rays_with_hits) / float(self.num_rays_tested)


_PROBE_DIRECTIONS = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

_DENSE_PROBE_DIRECTIONS = np.array(
    [
        [float(x), float(y), float(z)]
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
        if not (x == 0 and y == 0 and z == 0)
    ],
    dtype=np.float64,
)
_DENSE_PROBE_DIRECTIONS = _DENSE_PROBE_DIRECTIONS / np.linalg.norm(
    _DENSE_PROBE_DIRECTIONS,
    axis=1,
    keepdims=True,
)


def dense_probe_directions() -> np.ndarray:
    return _DENSE_PROBE_DIRECTIONS.copy()


def probe_point_clearance(
    sim: Any,
    point_world: Iterable[float],
    *,
    probe_radius_m: float,
    ignore_hits_within_m: float,
    directions: Iterable[Iterable[float]] | None = None,
) -> PointClearanceResult:
    try:
        import habitat_sim  # type: ignore
    except ImportError as exc:
        raise ImportError("habitat_sim is required for point-clearance probing.") from exc

    point = np.asarray(list(point_world), dtype=np.float64)
    min_hit = float(probe_radius_m)
    max_hit = 0.0
    hit_rays = 0
    probe_directions = np.asarray(
        list(directions) if directions is not None else _PROBE_DIRECTIONS,
        dtype=np.float64,
    )
    if probe_directions.ndim != 2 or probe_directions.shape[1] != 3:
        raise ValueError("Probe directions must have shape (N, 3).")

    for direction in probe_directions:
        ray = habitat_sim.geo.Ray(
            point.astype(np.float32),
            direction.astype(np.float32),
        )
        results = sim.cast_ray(ray=ray, max_distance=float(probe_radius_m))
        valid_hits = [
            float(hit.ray_distance)
            for hit in getattr(results, "hits", [])
            if float(hit.ray_distance) > float(ignore_hits_within_m)
        ]
        if not valid_hits:
            continue
        hit_rays += 1
        local_min = min(valid_hits)
        local_max = max(valid_hits)
        min_hit = min(min_hit, local_min)
        max_hit = max(max_hit, local_max)

    return PointClearanceResult(
        min_hit_distance=float(min_hit),
        max_hit_distance=float(max_hit),
        num_rays_with_hits=int(hit_rays),
        num_rays_tested=int(probe_directions.shape[0]),
    )
