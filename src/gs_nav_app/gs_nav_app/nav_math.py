"""NumPy-only navigation math shared by GUI code.

This module deliberately does not import OpenCV. The pip OpenCV wheel bundles a
different Qt plugin tree which conflicts with Ubuntu's PyQt5 xcb plugin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RenderStyle:
    ribbon_width_m: float = 0.95
    horizon_ratio: float = 0.43
    camera_height_m: float = 0.72
    virtual_fx_ratio: float = 0.90
    virtual_fy_ratio: float = 0.78


def image_to_bgr(msg) -> np.ndarray:
    """Convert common ROS Image encodings using NumPy only."""
    encoding = msg.encoding.lower()
    channels = {
        "mono8": 1,
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
    }.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    packed = rows[:, :msg.width * channels].reshape(
        msg.height, msg.width, channels)
    if encoding == "mono8":
        return np.repeat(packed, 3, axis=2)
    if encoding == "bgr8":
        return packed.copy()
    if encoding == "rgb8":
        return packed[:, :, ::-1].copy()
    if encoding == "bgra8":
        return packed[:, :, :3].copy()
    return packed[:, :, :3][:, :, ::-1].copy()


def transform_matrix(transform) -> np.ndarray:
    translation = transform.translation
    quaternion = transform.rotation
    x, y, z, w = (
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    )
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        rotation = np.eye(3)
    else:
        scale = 2.0 / norm
        rotation = np.array([
            [
                1 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
            ],
            [
                scale * (x * y + z * w),
                1 - scale * (x * x + z * z),
                scale * (y * z - x * w),
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1 - scale * (x * x + y * y),
            ],
        ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (
        translation.x,
        translation.y,
        translation.z,
    )
    return matrix


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def cumulative_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def make_ribbon(points: np.ndarray, width_m: float):
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return points.copy(), points.copy()
    tangents = np.empty((len(points), 2), dtype=np.float32)
    tangents[0] = points[1, :2] - points[0, :2]
    tangents[-1] = points[-1, :2] - points[-2, :2]
    if len(points) > 2:
        tangents[1:-1] = points[2:, :2] - points[:-2, :2]
    lengths = np.linalg.norm(tangents, axis=1)
    lengths[lengths < 1e-6] = 1.0
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    normals /= lengths[:, None]
    offsets = normals * width_m * 0.5
    left = points.copy()
    right = points.copy()
    left[:, :2] += offsets
    right[:, :2] -= offsets
    return left, right


def project_ground(
    points_base: np.ndarray,
    image_shape: Tuple[int, ...],
    style: RenderStyle,
):
    height, width = image_shape[:2]
    points = np.asarray(points_base, dtype=np.float32)
    forward = points[:, 0]
    left = points[:, 1]
    valid = forward > -0.35
    depth = np.maximum(forward + 0.60, 0.25)
    u = width * 0.5 - width * style.virtual_fx_ratio * left / depth
    v = (
        height * style.horizon_ratio
        + height * style.virtual_fy_ratio * style.camera_height_m / depth
    )
    return np.column_stack((u, v)), valid


def project_optical(points_camera: np.ndarray, camera_matrix: np.ndarray):
    points = np.asarray(points_camera, dtype=np.float64)
    depth = points[:, 2]
    valid = depth > 0.10
    safe_depth = np.where(valid, depth, 1.0)
    u = camera_matrix[0, 0] * points[:, 0] / safe_depth + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * points[:, 1] / safe_depth + camera_matrix[1, 2]
    return np.column_stack((u, v)).astype(np.float32), valid


def occupancy_to_bgr(occupancy: np.ndarray) -> np.ndarray:
    occupancy = np.asarray(occupancy)
    result = np.empty((*occupancy.shape, 3), dtype=np.uint8)
    result[:] = (58, 63, 66)
    known = occupancy >= 0
    gray = (215 - np.clip(occupancy, 0, 100) * 1.85).astype(np.uint8)
    result[known] = np.column_stack((gray[known], gray[known], gray[known]))
    return result
