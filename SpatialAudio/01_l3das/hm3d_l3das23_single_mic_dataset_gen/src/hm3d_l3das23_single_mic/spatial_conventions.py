from __future__ import annotations

import math
from typing import Iterable

import numpy as np


LOCAL_COORDINATE_FRAME = "mic_local_right_front_up"
AZIMUTH_REFERENCE = "mic_local_source_relative"
AZIMUTH_CONVENTION = "ambix_acn_sn3d_positive_left"

# Habitat/RLR exposes raw world-axis N3D FOA. The audio renderer remaps it
# before writing files, so stored waveforms are mic-local AmbiX/ACN/SN3D WYZX.
FOA_RAW_CHANNEL_ORDER = "WYZX"
FOA_CANONICAL_CHANNEL_ORDER = "WYZX"
FOA_CANONICAL_AXES = "AmbiX_ACN_SN3D_W,Y_left,Z_up,X_front"

EIGHT_WAY_ORDER = (
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
)


def normalize_azimuth_deg(azimuth_deg: float) -> float:
    """Normalize azimuth to [-180, 180)."""
    return ((float(azimuth_deg) + 180.0) % 360.0) - 180.0


def azimuth_raw_deg(azimuth_deg: float) -> float:
    """Normalize azimuth to [0, 360)."""
    return float(azimuth_deg) % 360.0


def local_angles_from_relative_xyz(relative_xyz: Iterable[float]) -> tuple[float, float, float]:
    """
    Convert mic-local [right, front, up] source vector to AmbiX-style angles.

    Positive azimuth points to listener-left, so front=0, left=+90,
    right=-90, and back is +/-180.
    """
    right, front, up = [float(value) for value in relative_xyz]
    horizontal = max(math.hypot(right, front), 1.0e-12)
    azimuth_deg = math.degrees(math.atan2(-right, front))
    elevation_deg = math.degrees(math.atan2(up, horizontal))
    distance_m = math.sqrt(right * right + front * front + up * up)
    return azimuth_deg, elevation_deg, distance_m


def direction_8way_from_azimuth(azimuth_deg: float) -> str:
    azimuth_norm = normalize_azimuth_deg(azimuth_deg)
    if -22.5 <= azimuth_norm < 22.5:
        return "front"
    if 22.5 <= azimuth_norm < 67.5:
        return "front-left"
    if 67.5 <= azimuth_norm < 112.5:
        return "left"
    if 112.5 <= azimuth_norm < 157.5:
        return "back-left"
    if azimuth_norm >= 157.5 or azimuth_norm < -157.5:
        return "back"
    if -157.5 <= azimuth_norm < -112.5:
        return "back-right"
    if -112.5 <= azimuth_norm < -67.5:
        return "right"
    return "front-right"


def local_unit_vector_right_front_up(relative_xyz: Iterable[float]) -> list[float]:
    vec = np.asarray(list(relative_xyz), dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12:
        return [0.0, 0.0, 0.0]
    unit = vec / norm
    return [float(value) for value in unit]


def ambix_unit_vector_xyz(relative_xyz: Iterable[float]) -> list[float]:
    """
    Map mic-local [right, front, up] to AmbiX axes [X_front, Y_left, Z_up].
    """
    right, front, up = local_unit_vector_right_front_up(relative_xyz)
    return [float(front), float(-right), float(up)]


def rounded_float_list(values: Iterable[float], *, digits: int = 6) -> list[float]:
    return [round(float(value), int(digits)) for value in values]
