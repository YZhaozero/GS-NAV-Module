"""OpenCV drawing helpers used by the AR navigation node.

This module intentionally has no ROS imports so the visual layer can be tested
and tuned on a laptop without starting a ROS graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class RenderStyle:
    route_bgr: Tuple[int, int, int] = (238, 165, 0)
    route_edge_bgr: Tuple[int, int, int] = (255, 235, 55)
    route_alpha: float = 0.58
    ribbon_width_m: float = 0.95
    horizon_ratio: float = 0.43
    camera_height_m: float = 0.72
    virtual_fx_ratio: float = 0.90
    virtual_fy_ratio: float = 0.78


def cumulative_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def make_ribbon(points: np.ndarray, width_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return left/right boundaries for an XY polyline."""
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
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0])) / lengths[:, None]

    offsets = normals * (width_m * 0.5)
    left = points.copy()
    right = points.copy()
    left[:, :2] += offsets
    right[:, :2] -= offsets
    return left, right


def project_ground(
    points_base: np.ndarray,
    image_shape: Tuple[int, ...],
    style: RenderStyle,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project base_link ground points with a tunable virtual pinhole camera.

    ROS base convention is X forward and Y left. This fallback is useful before
    a real CameraInfo/extrinsic calibration is available.
    """
    h, w = image_shape[:2]
    points = np.asarray(points_base, dtype=np.float32)
    forward = points[:, 0]
    left = points[:, 1]
    valid = forward > -0.35
    depth = np.maximum(forward + 0.60, 0.25)
    fx = w * style.virtual_fx_ratio
    fy = h * style.virtual_fy_ratio
    u = w * 0.5 - fx * left / depth
    v = h * style.horizon_ratio + fy * style.camera_height_m / depth
    return np.column_stack((u, v)), valid


def project_optical(
    points_camera: np.ndarray,
    camera_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project points expressed in a ROS optical frame (X right/Y down/Z front)."""
    points = np.asarray(points_camera, dtype=np.float64)
    z = points[:, 2]
    valid = z > 0.10
    safe_z = np.where(valid, z, 1.0)
    u = camera_matrix[0, 0] * points[:, 0] / safe_z + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * points[:, 1] / safe_z + camera_matrix[1, 2]
    return np.column_stack((u, v)).astype(np.float32), valid


def _visible(point: np.ndarray, w: int, h: int, margin: int = 160) -> bool:
    return -margin <= point[0] < w + margin and -margin <= point[1] < h + margin


def draw_route(
    image: np.ndarray,
    center_px: np.ndarray,
    left_px: np.ndarray,
    right_px: np.ndarray,
    valid: np.ndarray,
    style: RenderStyle,
) -> None:
    """Draw the AR route as separately clipped quadrilaterals."""
    h, w = image.shape[:2]
    overlay = image.copy()
    for i in range(len(center_px) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        quad = np.array(
            [left_px[i], left_px[i + 1], right_px[i + 1], right_px[i]],
            dtype=np.int32,
        )
        if any(_visible(p, w, h) for p in quad):
            cv2.fillConvexPoly(
                overlay, quad, style.route_bgr, lineType=cv2.LINE_AA)
    cv2.addWeighted(
        overlay, style.route_alpha, image, 1.0 - style.route_alpha, 0, image)

    for boundary in (left_px, right_px):
        segments = []
        for i in range(len(boundary) - 1):
            if valid[i] and valid[i + 1]:
                segments.append(np.array([boundary[i], boundary[i + 1]], dtype=np.int32))
        if segments:
            cv2.polylines(image, segments, False, style.route_edge_bgr, 3, cv2.LINE_AA)

    visible_indices = [i for i in range(1, len(center_px) - 1) if valid[i]]
    if visible_indices:
        idx = visible_indices[min(len(visible_indices) // 3, len(visible_indices) - 1)]
        p = center_px[idx].astype(np.int32)
        q = center_px[min(idx + 1, len(center_px) - 1)].astype(np.int32)
        direction = q - p
        norm = float(np.linalg.norm(direction))
        if norm > 2 and _visible(p, w, h, 40):
            direction = direction / norm
            tip = p + (direction * 30).astype(np.int32)
            cv2.arrowedLine(
                image, tuple(p), tuple(tip), (255, 255, 255), 7,
                cv2.LINE_AA, tipLength=0.45)


def _rounded_panel(
    image: np.ndarray,
    rect: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    alpha: float,
    radius: int = 18,
) -> None:
    x0, y0, x1, y1 = rect
    overlay = image.copy()
    cv2.rectangle(overlay, (x0 + radius, y0), (x1 - radius, y1), color, -1)
    cv2.rectangle(overlay, (x0, y0 + radius), (x1, y1 - radius), color, -1)
    for center in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                   (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)):
        cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def occupancy_to_bgr(occupancy: np.ndarray) -> np.ndarray:
    """Convert ROS OccupancyGrid values (-1..100) to a compact map image."""
    occupancy = np.asarray(occupancy)
    result = np.empty((*occupancy.shape, 3), dtype=np.uint8)
    result[:] = (58, 63, 66)
    known = occupancy >= 0
    gray = 215 - np.clip(occupancy, 0, 100) * 1.85
    gray = gray.astype(np.uint8)
    result[known] = np.column_stack((gray[known], gray[known], gray[known]))
    return result


def draw_minimap(
    image: np.ndarray,
    occupancy: Optional[np.ndarray],
    map_origin_xy: np.ndarray,
    map_resolution: float,
    map_origin_yaw: float,
    path_xy: Optional[np.ndarray],
    robot_xy: Optional[np.ndarray],
    robot_yaw: float = 0.0,
) -> None:
    h, w = image.shape[:2]
    panel_w = max(190, int(w * 0.28))
    panel_h = max(170, int(h * 0.31))
    x0, y0 = 24, h - panel_h - 28
    x1, y1 = x0 + panel_w, y0 + panel_h
    _rounded_panel(image, (x0, y0, x1, y1), (20, 35, 38), 0.78, 20)

    if occupancy is None or not occupancy.size or map_resolution <= 0.0:
        cv2.putText(image, "WAITING FOR MAP", (x0 + 14, y0 + panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 190, 190), 1, cv2.LINE_AA)
        return

    map_h, map_w = occupancy.shape
    content_x0, content_y0 = x0 + 10, y0 + 32
    content_w, content_h = panel_w - 20, panel_h - 42
    scale = min(content_w / map_w, content_h / map_h)
    draw_w = max(1, int(round(map_w * scale)))
    draw_h = max(1, int(round(map_h * scale)))
    offset_x = content_x0 + (content_w - draw_w) // 2
    offset_y = content_y0 + (content_h - draw_h) // 2
    grid_image = cv2.resize(
        np.flipud(occupancy_to_bgr(occupancy)),
        (draw_w, draw_h), interpolation=cv2.INTER_NEAREST)
    image[offset_y:offset_y + draw_h, offset_x:offset_x + draw_w] = grid_image

    cos_yaw = np.cos(map_origin_yaw)
    sin_yaw = np.sin(map_origin_yaw)

    def to_px(xy: np.ndarray) -> np.ndarray:
        relative = np.asarray(xy, dtype=np.float32)[:, :2] - map_origin_xy
        grid_x = (cos_yaw * relative[:, 0] + sin_yaw * relative[:, 1])
        grid_y = (-sin_yaw * relative[:, 0] + cos_yaw * relative[:, 1])
        grid_x /= map_resolution
        grid_y /= map_resolution
        return np.column_stack((
            offset_x + grid_x * scale,
            offset_y + (map_h - 1 - grid_y) * scale,
        )).astype(np.int32)

    if path_xy is not None and len(path_xy) > 1:
        cv2.polylines(image, [to_px(path_xy)], False, (255, 210, 30), 5, cv2.LINE_AA)
        goal = to_px(path_xy[-1:].reshape(1, 2))[0]
        cv2.circle(image, tuple(goal), 8, (55, 70, 235), -1, cv2.LINE_AA)
        cv2.circle(image, tuple(goal), 8, (255, 255, 255), 2, cv2.LINE_AA)

    if robot_xy is not None:
        robot_center = to_px(np.asarray(robot_xy).reshape(1, 2))[0]
        robot = np.array([[11, 0], [-8, -7], [-5, 0], [-8, 7]], dtype=np.float32)
        display_yaw = -(robot_yaw - map_origin_yaw)
        rotation = np.array([
            [np.cos(display_yaw), -np.sin(display_yaw)],
            [np.sin(display_yaw), np.cos(display_yaw)],
        ], dtype=np.float32)
        robot = (robot @ rotation.T + robot_center).astype(np.int32)
        cv2.fillConvexPoly(image, robot, (255, 255, 255), cv2.LINE_AA)
        cv2.polylines(image, [robot], True, (238, 165, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "MAP", (x0 + 12, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (225, 235, 235), 2, cv2.LINE_AA)


def draw_hud(image: np.ndarray, remaining_m: float, status: str) -> None:
    h, w = image.shape[:2]
    panel_w = min(430, w - 40)
    x0 = (w - panel_w) // 2
    _rounded_panel(image, (x0, 18, x0 + panel_w, 92), (7, 16, 22), 0.72, 20)
    cv2.arrowedLine(image, (x0 + 38, 72), (x0 + 38, 37), (255, 255, 255),
                    5, cv2.LINE_AA, tipLength=0.4)
    distance_text = (
        f"{remaining_m:.0f} m"
        if remaining_m < 1000
        else f"{remaining_m / 1000:.1f} km"
    )
    cv2.putText(image, distance_text, (x0 + 70, 55), cv2.FONT_HERSHEY_DUPLEX,
                0.83, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Follow the route", (x0 + 70, 79), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (205, 220, 225), 1, cv2.LINE_AA)
    color = (80, 230, 125) if status == "LIVE" else (40, 190, 255)
    cv2.circle(image, (x0 + panel_w - 42, 51), 7, color, -1, cv2.LINE_AA)
    cv2.putText(image, status, (x0 + panel_w - 83, 77), cv2.FONT_HERSHEY_SIMPLEX,
                0.37, color, 1, cv2.LINE_AA)


def synthetic_camera_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a neutral road-like frame for the no-hardware demo launch."""
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    road = np.array([55, 60, 62], dtype=np.float32)
    horizon = int(height * 0.43)
    image = np.empty((height, width, 3), dtype=np.uint8)
    for channel in range(3):
        top = 135 + 30 * y[:horizon]
        image[:horizon, :, channel] = np.repeat(top, width, axis=1).astype(np.uint8)
        ground_gradient = road[channel] + 45 * y[horizon:]
        image[horizon:, :, channel] = np.repeat(ground_gradient, width, axis=1).astype(np.uint8)
    cv2.rectangle(image, (0, horizon - 45), (width, horizon), (55, 78, 62), -1)
    for x in range(0, width, 110):
        cv2.line(image, (x, horizon - 80), (x + 20, horizon), (76, 105, 80), 5)
    cv2.line(image, (0, horizon), (width, horizon), (185, 190, 180), 3)
    cv2.putText(image, "GS NAV CAMERA DEMO", (26, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (240, 240, 240), 1, cv2.LINE_AA)
    return image


def waiting_camera_frame(
    camera_topic: str, width: int = 1280, height: int = 720
) -> np.ndarray:
    """Create the initial UI shown while the first camera frame is pending."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (22, 27, 31)
    for y in range(height):
        shade = int(18 * y / max(height - 1, 1))
        image[y, :, :] += shade

    cx, cy = width // 2, height // 2
    cv2.circle(image, (cx, cy - 42), 37, (42, 53, 60), 3, cv2.LINE_AA)
    cv2.rectangle(
        image, (cx - 20, cy - 57), (cx + 14, cy - 31),
        (115, 205, 235), 3, cv2.LINE_AA)
    lens = np.array([
        [cx + 14, cy - 51], [cx + 29, cy - 59],
        [cx + 29, cy - 29], [cx + 14, cy - 37],
    ], dtype=np.int32)
    cv2.polylines(image, [lens], True, (115, 205, 235), 3, cv2.LINE_AA)
    cv2.putText(
        image, "WAITING FOR CAMERA", (cx - 153, cy + 40),
        cv2.FONT_HERSHEY_DUPLEX, 0.85, (225, 232, 235), 2, cv2.LINE_AA)
    cv2.putText(
        image, camera_topic, (cx - min(210, len(camera_topic) * 5), cy + 73),
        cv2.FONT_HERSHEY_SIMPLEX, 0.53, (135, 155, 165), 1, cv2.LINE_AA)
    return image
