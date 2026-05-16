from __future__ import annotations

import math

import numpy as np

from hm3d_l3das23_single_mic.audio_renderer import (
    remap_rir_world_n3d_to_mic_ambix_sn3d_wyzx,
)
from hm3d_l3das23_single_mic.geometry import basis_from_yaw


def _synthetic_rlr_world_n3d(direction_world: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction_world, dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1.0e-8)
    rir = np.zeros((4, 1), dtype=np.float32)
    rir[0, 0] = 1.0
    rir[1, 0] = math.sqrt(3.0) * float(direction[1])
    rir[2, 0] = math.sqrt(3.0) * float(direction[2])
    rir[3, 0] = math.sqrt(3.0) * float(direction[0])
    return rir


def test_rlr_world_n3d_remaps_to_mic_local_ambix_sn3d_wyzx_for_multiple_yaws() -> None:
    for yaw_deg in (0.0, 90.0, 180.0, 270.0, 37.0):
        yaw_rad = math.radians(yaw_deg)
        right, front, up = basis_from_yaw(yaw_rad)

        front_foa = remap_rir_world_n3d_to_mic_ambix_sn3d_wyzx(
            _synthetic_rlr_world_n3d(front),
            yaw_rad,
        )
        right_foa = remap_rir_world_n3d_to_mic_ambix_sn3d_wyzx(
            _synthetic_rlr_world_n3d(right),
            yaw_rad,
        )
        left_foa = remap_rir_world_n3d_to_mic_ambix_sn3d_wyzx(
            _synthetic_rlr_world_n3d(-right),
            yaw_rad,
        )
        back_foa = remap_rir_world_n3d_to_mic_ambix_sn3d_wyzx(
            _synthetic_rlr_world_n3d(-front),
            yaw_rad,
        )
        up_foa = remap_rir_world_n3d_to_mic_ambix_sn3d_wyzx(
            _synthetic_rlr_world_n3d(up),
            yaw_rad,
        )

        np.testing.assert_allclose(front_foa[:, 0], [1.0, 0.0, 0.0, 1.0], atol=1.0e-6)
        np.testing.assert_allclose(right_foa[:, 0], [1.0, -1.0, 0.0, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(left_foa[:, 0], [1.0, 1.0, 0.0, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(back_foa[:, 0], [1.0, 0.0, 0.0, -1.0], atol=1.0e-6)
        np.testing.assert_allclose(up_foa[:, 0], [1.0, 0.0, 1.0, 0.0], atol=1.0e-6)
