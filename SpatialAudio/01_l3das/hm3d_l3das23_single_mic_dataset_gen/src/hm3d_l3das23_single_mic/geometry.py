from __future__ import annotations

import math
from hashlib import blake2b
from typing import Iterable

import numpy as np


def stable_int_from_parts(*parts: object, modulus: int = 2**31 - 1) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int(blake2b(payload, digest_size=8).hexdigest(), 16) % modulus


def linspace_with_step(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    values: list[float] = []
    count = int(round((stop - start) / step))
    for idx in range(count + 1):
        values.append(round(start + idx * step, 10))
    if not math.isclose(values[-1], stop, abs_tol=1.0e-8):
        values.append(round(stop, 10))
    return values


def yaw_deg_to_rad(yaw_deg: float) -> float:
    return math.radians(float(yaw_deg))


def yaw_rad_to_deg(yaw_rad: float) -> float:
    return math.degrees(float(yaw_rad))


def yaw_rad_to_quaternion_wxyz(yaw_rad: float) -> list[float]:
    half = 0.5 * float(yaw_rad)
    return [math.cos(half), 0.0, math.sin(half), 0.0]


def basis_from_yaw(yaw_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local microphone frame:
    - +X: right
    - +Y: front
    - +Z: up

    Azimuth values follow the AmbiX horizontal sign convention:
    positive angles rotate toward listener-left, so azimuth is atan2(left, front)
    = atan2(-local_x_right, local_y_front).

    Habitat world frame is Y-up. The microphone forward direction follows the
    camera convention used by Habitat pinhole sensors, which face world -Z at yaw 0.
    """
    yaw = float(yaw_rad)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float64)
    forward = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return right, forward, up


def local_xyz_from_cylindrical(rho: float, theta_rad: float, z: float) -> np.ndarray:
    return np.array(
        [
            -float(rho) * math.sin(float(theta_rad)),
            float(rho) * math.cos(float(theta_rad)),
            float(z),
        ],
        dtype=np.float64,
    )


def cylindrical_from_local_xyz(local_xyz: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = np.asarray(list(local_xyz), dtype=np.float64)
    rho = math.hypot(float(x), float(y))
    theta = math.atan2(float(-x), float(y))
    return rho, theta, float(z)


def spherical_from_local_xyz(local_xyz: Iterable[float]) -> tuple[float, float, float]:
    vec = np.asarray(list(local_xyz), dtype=np.float64)
    distance = float(np.linalg.norm(vec))
    horizontal = max(math.hypot(float(vec[0]), float(vec[1])), 1.0e-12)
    azimuth_deg = yaw_rad_to_deg(math.atan2(float(-vec[0]), float(vec[1])))
    elevation_deg = yaw_rad_to_deg(math.atan2(float(vec[2]), horizontal))
    return distance, azimuth_deg, elevation_deg


def local_to_world(
    mic_position_world: Iterable[float],
    yaw_rad: float,
    local_xyz: Iterable[float],
) -> np.ndarray:
    mic = np.asarray(list(mic_position_world), dtype=np.float64)
    local = np.asarray(list(local_xyz), dtype=np.float64)
    right, forward, up = basis_from_yaw(yaw_rad)
    return mic + (local[0] * right) + (local[1] * forward) + (local[2] * up)


def world_to_local(
    mic_position_world: Iterable[float],
    yaw_rad: float,
    point_world: Iterable[float],
) -> np.ndarray:
    mic = np.asarray(list(mic_position_world), dtype=np.float64)
    point = np.asarray(list(point_world), dtype=np.float64)
    delta = point - mic
    right, forward, up = basis_from_yaw(yaw_rad)
    return np.array(
        [
            float(np.dot(delta, right)),
            float(np.dot(delta, forward)),
            float(np.dot(delta, up)),
        ],
        dtype=np.float64,
    )


def camera_xyz_from_local(local_xyz: Iterable[float]) -> np.ndarray:
    local = np.asarray(list(local_xyz), dtype=np.float64)
    return np.array([local[0], local[2], local[1]], dtype=np.float64)


def distance_m(point_a: Iterable[float], point_b: Iterable[float]) -> float:
    a = np.asarray(list(point_a), dtype=np.float64)
    b = np.asarray(list(point_b), dtype=np.float64)
    return float(np.linalg.norm(a - b))


def point_in_aabb(
    point: Iterable[float],
    aabb_min: Iterable[float],
    aabb_max: Iterable[float],
    *,
    margin: float = 0.0,
) -> bool:
    p = np.asarray(list(point), dtype=np.float64)
    lo = np.asarray(list(aabb_min), dtype=np.float64) + float(margin)
    hi = np.asarray(list(aabb_max), dtype=np.float64) - float(margin)
    return bool(np.all(p >= lo) and np.all(p <= hi))


def farthest_point_sampling(
    points: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if points.size == 0 or n_target <= 0:
        return np.empty((0, 3), dtype=np.float64)
    if n_target >= points.shape[0]:
        return points.copy()

    first_idx = int(rng.integers(0, points.shape[0]))
    selected = [first_idx]
    min_dist = np.linalg.norm(points - points[first_idx], axis=1)

    for _ in range(1, int(n_target)):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        next_dist = np.linalg.norm(points - points[next_idx], axis=1)
        min_dist = np.minimum(min_dist, next_dist)

    return points[np.asarray(selected, dtype=np.int64)]
