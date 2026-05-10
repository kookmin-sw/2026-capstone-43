from __future__ import annotations

import numpy as np


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def ros_pose_to_camera_to_world(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert ROS optical camera pose to Nerfstudio/OpenGL-style camera-to-world.

    ROS optical frame: +X right, +Y down, +Z forward.
    OpenGL camera frame: +X right, +Y up, +Z backward.
    """

    ros_optical_from_gl_camera = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    rot_world_from_ros = quat_to_rot(qx, qy, qz, qw)
    rot_world_from_gl = rot_world_from_ros @ ros_optical_from_gl_camera
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot_world_from_gl
    mat[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return mat


def camera_ray_directions(width: int, height: int, fx: float, fy: float, cx: float, cy: float, stride: int = 16) -> np.ndarray:
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    x = (xs.astype(np.float32) - cx) / fx
    y = (ys.astype(np.float32) - cy) / fy
    z = np.ones_like(x, dtype=np.float32)
    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8
    return dirs.reshape(-1, 3), xs.reshape(-1), ys.reshape(-1)

