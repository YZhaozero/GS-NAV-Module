"""Interactive Qt dashboard for camera and Nav2 navigation control."""

from __future__ import annotations

import math
import os
import re
import signal
import sys
from pathlib import Path as FilePath
from typing import Optional, Tuple

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateThroughPoses
from nav_msgs.msg import OccupancyGrid, Path
from PyQt5.QtCore import (
    QObject,
    QPointF,
    QProcess,
    QRectF,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from .map_processing import (
    GridMap,
    PointCloudMap,
    load_grid_map,
    load_pointcloud,
    pointcloud2_to_xyz,
    pointcloud_to_grid,
    save_grid_map,
    save_pcd,
)

from .nav_math import (
    RenderStyle,
    apply_transform,
    cumulative_distance,
    image_to_bgr,
    make_ribbon,
    occupancy_to_bgr,
    project_ground,
    project_optical,
    transform_matrix,
)


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def yaw_quaternion(yaw: float) -> Tuple[float, float]:
    return math.sin(yaw * 0.5), math.cos(yaw * 0.5)


class CameraPanel(QWidget):
    """Raw camera display with a resolution-independent Qt vector overlay."""

    def __init__(
        self, show_status: bool = True, compact: bool = False,
    ) -> None:
        super().__init__()
        self._show_status = show_status
        if compact:
            self.setMinimumSize(280, 160)
        else:
            self.setMinimumSize(640, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap: Optional[QPixmap] = None
        self._source_size = (1280, 720)
        self._route = None
        self._distance = 0.0
        self._camera_live = False

    def set_frame(
        self,
        frame_bgr: Optional[np.ndarray],
        route,
        remaining_m: float,
    ) -> None:
        if frame_bgr is not None:
            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            h, w = rgb.shape[:2]
            qimage = QImage(
                rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
            self._pixmap = QPixmap.fromImage(qimage)
            self._source_size = (w, h)
            self._camera_live = True
        self._route = route
        self._distance = remaining_m
        self.update()

    def _video_rect(self) -> QRectF:
        source_w, source_h = self._source_size
        scale = min(self.width() / source_w, self.height() / source_h)
        width = source_w * scale
        height = source_h * scale
        return QRectF(
            (self.width() - width) * 0.5,
            (self.height() - height) * 0.5,
            width,
            height,
        )

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#090d12"))
        target = self._video_rect()
        if self._pixmap is not None:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
            painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        else:
            painter.setPen(QColor("#e5ebef"))
            painter.setFont(QFont("Sans Serif", 20, QFont.DemiBold))
            painter.drawText(target, Qt.AlignCenter, "等待相机 /color/image_raw")

        if self._route is not None:
            self._draw_route(painter, target)
        if self._show_status:
            self._draw_status(painter, target)

    def _map_video_point(self, point: np.ndarray, target: QRectF) -> QPointF:
        source_w, source_h = self._source_size
        return QPointF(
            target.left() + float(point[0]) * target.width() / source_w,
            target.top() + float(point[1]) * target.height() / source_h,
        )

    def _draw_route(self, painter: QPainter, target: QRectF) -> None:
        center, left, right, valid = self._route
        painter.setPen(QPen(QColor(74, 222, 255, 225), 2.0))
        painter.setBrush(QColor(0, 150, 230, 112))
        for index in range(len(center) - 1):
            if not (valid[index] and valid[index + 1]):
                continue
            polygon = QPolygonF([
                self._map_video_point(left[index], target),
                self._map_video_point(left[index + 1], target),
                self._map_video_point(right[index + 1], target),
                self._map_video_point(right[index], target),
            ])
            painter.drawPolygon(polygon)

    def _draw_status(self, painter: QPainter, target: QRectF) -> None:
        card = QRectF(target.left() + 22, target.top() + 20, 235, 66)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(7, 12, 18, 205))
        painter.drawRoundedRect(card, 12, 12)
        painter.setPen(QColor("#f4f7f9"))
        painter.setFont(QFont("Sans Serif", 17, QFont.DemiBold))
        painter.drawText(
            card.adjusted(17, 8, -10, -28),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"{self._distance:.1f} m",
        )
        painter.setPen(QColor("#94a5b2"))
        painter.setFont(QFont("Sans Serif", 10))
        state = "相机在线 · 全局路径" if self._camera_live else "等待相机数据"
        painter.drawText(
            card.adjusted(17, 34, -10, -7), Qt.AlignLeft | Qt.AlignVCenter, state)


class SensorLaunchProcess(QObject):
    """Own one ros2 launch process and expose its combined output to Qt."""

    log_received = pyqtSignal(str)
    state_changed = pyqtSignal(str)

    def __init__(self, display_name: str, parent=None) -> None:
        super().__init__(parent)
        self.display_name = display_name
        self._requested_stop = False
        self._process_group_id: Optional[int] = None
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.started.connect(self._on_started)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.NotRunning

    def start_launch(self, arguments) -> bool:
        if self.running:
            self.log_received.emit(f"{self.display_name}已经在运行\n")
            return False
        command = "ros2 " + " ".join(str(value) for value in arguments)
        self.log_received.emit(f"$ {command}\n")
        self._requested_stop = False
        self._process_group_id = None
        self.state_changed.emit("starting")
        # Start every launch in its own session.  Signalling only the outer
        # ``ros2 launch`` process leaves driver nodes behind; a dedicated
        # process group lets Stop reliably reach the whole launch tree.
        self.process.start(
            "setsid", ["ros2", *[str(value) for value in arguments]])
        return True

    def stop_launch(self) -> None:
        if not self.running:
            self.log_received.emit(f"{self.display_name}当前未运行\n")
            self.state_changed.emit("stopped")
            return
        self.log_received.emit(f"正在停止{self.display_name}…\n")
        self._requested_stop = True
        self.state_changed.emit("stopping")
        self._capture_process_group()
        self._signal_process_group(signal.SIGINT)
        QTimer.singleShot(1800, self._terminate_if_alive)

    def shutdown(self) -> None:
        if not self.running and not self._group_alive():
            return
        self._requested_stop = True
        self._capture_process_group()
        self._signal_process_group(signal.SIGINT)
        if not self.process.waitForFinished(1500):
            self._signal_process_group(signal.SIGTERM)
            self.process.waitForFinished(700)
        if self._group_alive():
            self._signal_process_group(signal.SIGKILL)
            self.process.waitForFinished(500)
        self._finish_stopping()

    def _capture_process_group(self) -> None:
        if self._process_group_id is None:
            process_id = int(self.process.processId())
            if process_id > 0:
                self._process_group_id = process_id

    def _on_started(self) -> None:
        self._capture_process_group()
        self.state_changed.emit("running")

    def _group_alive(self) -> bool:
        group_id = self._process_group_id
        if group_id is None:
            return False
        group_seen = False
        try:
            for entry in os.scandir("/proc"):
                if not entry.name.isdigit():
                    continue
                try:
                    stat = FilePath(entry.path, "stat").read_text()
                    fields = stat[stat.rfind(")") + 2:].split()
                    if len(fields) > 2 and int(fields[2]) == group_id:
                        group_seen = True
                        if fields[0] not in ("Z", "X"):
                            return True
                except (OSError, ValueError):
                    continue
            if group_seen:
                # Zombies have already stopped executing and only await reaping.
                return False
        except OSError:
            pass
        try:
            os.killpg(group_id, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _signal_process_group(self, signal_number: int) -> None:
        self._capture_process_group()
        if self._process_group_id is not None:
            try:
                os.killpg(self._process_group_id, signal_number)
                return
            except ProcessLookupError:
                return
            except PermissionError:
                pass
        if self.running:
            if signal_number == signal.SIGKILL:
                self.process.kill()
            else:
                self.process.terminate()

    def _terminate_if_alive(self) -> None:
        if not self._requested_stop:
            return
        if self._group_alive() or self.running:
            self.log_received.emit(
                f"{self.display_name}仍在退出，正在终止整个进程组…\n")
            self._signal_process_group(signal.SIGTERM)
        QTimer.singleShot(1500, self._kill_if_alive)

    def _kill_if_alive(self) -> None:
        if not self._requested_stop:
            return
        if self._group_alive() or self.running:
            self.log_received.emit(
                f"{self.display_name}仍有残留进程，正在强制结束整个进程组…\n")
            self._signal_process_group(signal.SIGKILL)
        QTimer.singleShot(150, self._finish_stopping)

    def _finish_stopping(self) -> None:
        if not self._requested_stop:
            return
        if self._group_alive():
            self.log_received.emit(
                f"{self.display_name}进程组尚未完全退出，将再次强制清理…\n")
            self._signal_process_group(signal.SIGKILL)
            QTimer.singleShot(200, self._finish_stopping)
            return
        self._requested_stop = False
        self._process_group_id = None
        self.log_received.emit(f"\n{self.display_name}进程已完全停止\n")
        self.state_changed.emit("stopped")

    def _read_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace")
        if data:
            self.log_received.emit(data)

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        self._read_output()
        if self._requested_stop:
            if not self._group_alive():
                self._finish_stopping()
            return
        if self._group_alive():
            self.log_received.emit(
                f"\n{self.display_name}主进程已结束，但检测到残留子进程，正在清理…\n")
            self._requested_stop = True
            self.state_changed.emit("stopping")
            self._signal_process_group(signal.SIGTERM)
            QTimer.singleShot(1000, self._kill_if_alive)
            return
        self._process_group_id = None
        self.log_received.emit(
            f"\n{self.display_name}进程已结束（退出码 {exit_code}）\n")
        self.state_changed.emit("stopped")

    def _on_error(self, error) -> None:
        if self._requested_stop and error == QProcess.Crashed:
            return
        self.log_received.emit(
            f"{self.display_name}进程错误：{self.process.errorString()} ({error})\n")
        self.state_changed.emit("error")


class MapPanel(QWidget):
    """Grid/top-down cloud view with an optional interactive 3-D cloud mode."""

    point_selected = pyqtSignal(float, float)
    edit_started = pyqtSignal()
    edit_requested = pyqtSignal(str, str, float, float, float)

    def __init__(self, cloud_3d: bool = False) -> None:
        super().__init__()
        self.setMinimumSize(260, 200)
        self.setCursor(Qt.CrossCursor)
        self._mode = "grid"
        self._grid_pixmap: Optional[QPixmap] = None
        self._grid_shape = (0, 0)
        self._grid_origin = np.zeros(2, dtype=np.float32)
        self._grid_resolution = 0.0
        self._grid_origin_yaw = 0.0
        self._cloud_pixmap: Optional[QPixmap] = None
        self._cloud_shape = (0, 0)
        self._cloud_origin = np.zeros(2, dtype=np.float32)
        self._cloud_resolution = 0.0
        self._cloud_3d_enabled = bool(cloud_3d)
        self._cloud_source_points = np.empty((0, 3), dtype=np.float32)
        self._cloud_source_colors: Optional[np.ndarray] = None
        self._cloud_source_splat_scales: Optional[np.ndarray] = None
        self._cloud_source_splat_rotations: Optional[np.ndarray] = None
        self._cloud_source_splat_opacities: Optional[np.ndarray] = None
        self._cloud_total_points = 0
        self._cloud_points = np.empty((0, 3), dtype=np.float32)
        self._cloud_colors = np.empty((0, 3), dtype=np.uint8)
        self._cloud_display_matrix = np.eye(3, dtype=np.float32)
        self._cloud_display_offset = np.zeros(3, dtype=np.float32)
        self._cloud_ground_reference_z = 0.0
        self._cloud_alignment = "auto"
        self._cloud_axis_order = "XYZ"
        self._cloud_axis_flips = (False, False, False)
        self._cloud_projection = "perspective"
        self._cloud_color_mode = "rgb"
        self._cloud_center = np.zeros(3, dtype=np.float32)
        self._cloud_extent = np.ones(3, dtype=np.float32)
        self._cloud_bounds_low = np.zeros(3, dtype=np.float32)
        self._cloud_level_angle_degrees = 0.0
        self._cloud_view_target = np.zeros(3, dtype=np.float32)
        self._cloud_azimuth = math.radians(-45.0)
        self._cloud_elevation = math.radians(32.0)
        self._cloud_zoom = 1.0
        self._view_drag_button = Qt.NoButton
        self._last_mouse_position = None
        self._view_drag_distance = 0.0
        self._path: Optional[np.ndarray] = None
        self._robot_xy: Optional[np.ndarray] = None
        self._robot_yaw = 0.0
        self._waypoints = []
        self._selected_waypoint_index: Optional[int] = None
        self._edit_tool = "navigate"
        self._brush_radius = 0.20
        self._painting = False

    @property
    def display_mode(self) -> str:
        return self._mode

    def set_display_mode(self, mode: str) -> None:
        if mode not in ("grid", "cloud"):
            raise ValueError(f"unknown map display mode: {mode}")
        self._mode = mode
        self._update_cursor()
        self.update()

    def set_edit_tool(self, tool: str, brush_radius: float) -> None:
        self._edit_tool = tool
        self._brush_radius = max(0.01, float(brush_radius))
        self._update_cursor()

    def _update_cursor(self) -> None:
        if self._cloud_3d_enabled and self._mode == "cloud":
            cursor = Qt.OpenHandCursor if self._edit_tool == "navigate" else Qt.CrossCursor
        else:
            cursor = Qt.CrossCursor if self._edit_tool != "navigate" else Qt.PointingHandCursor
        self.setCursor(cursor)

    def reset_cloud_view(self) -> None:
        """Restore an isometric view fitted around the current cloud."""
        self._cloud_view_target = self._cloud_center.copy()
        self._cloud_azimuth = math.radians(-45.0)
        self._cloud_elevation = math.radians(32.0)
        self._cloud_zoom = 1.0
        self.update()

    @property
    def cloud_has_rgb(self) -> bool:
        return self._cloud_source_colors is not None

    def set_cloud_display_options(
        self,
        alignment: str,
        axis_order: str,
        axis_flips,
        projection: str,
        color_mode: str,
    ) -> None:
        if alignment not in ("original", "auto"):
            raise ValueError(f"unknown cloud alignment: {alignment}")
        if sorted(axis_order) != ["X", "Y", "Z"]:
            raise ValueError(f"invalid cloud axis order: {axis_order}")
        if projection not in ("orthographic", "perspective"):
            raise ValueError(f"unknown cloud projection: {projection}")
        if color_mode not in ("rgb", "height"):
            raise ValueError(f"unknown cloud color mode: {color_mode}")
        self._cloud_alignment = alignment
        self._cloud_axis_order = axis_order
        self._cloud_axis_flips = tuple(bool(value) for value in axis_flips)
        self._cloud_projection = projection
        self._cloud_color_mode = color_mode
        self._rebuild_cloud_3d()
        self.reset_cloud_view()

    def set_map(
        self,
        occupancy: Optional[np.ndarray],
        origin: np.ndarray,
        resolution: float,
        origin_yaw: float,
    ) -> None:
        if occupancy is None:
            self._grid_pixmap = None
            self.update()
            return
        colored = occupancy_to_bgr(occupancy)[:, :, ::-1]
        colored = np.ascontiguousarray(np.flipud(colored))
        h, w = colored.shape[:2]
        qimage = QImage(
            colored.data, w, h, colored.strides[0], QImage.Format_RGB888).copy()
        self._grid_pixmap = QPixmap.fromImage(qimage)
        self._grid_shape = (h, w)
        self._grid_origin = np.asarray(origin, dtype=np.float32)[:2]
        self._grid_resolution = float(resolution)
        self._grid_origin_yaw = float(origin_yaw)
        self.update()

    def set_pointcloud(
        self,
        points: Optional[np.ndarray],
        colors: Optional[np.ndarray] = None,
        splat_scales: Optional[np.ndarray] = None,
        splat_rotations: Optional[np.ndarray] = None,
        splat_opacities: Optional[np.ndarray] = None,
    ) -> None:
        if points is None or not len(points):
            self._cloud_pixmap = None
            self._cloud_source_points = np.empty((0, 3), dtype=np.float32)
            self._cloud_source_colors = None
            self._cloud_source_splat_scales = None
            self._cloud_source_splat_rotations = None
            self._cloud_source_splat_opacities = None
            self._cloud_total_points = 0
            self._cloud_points = np.empty((0, 3), dtype=np.float32)
            self._cloud_colors = np.empty((0, 3), dtype=np.uint8)
            self.update()
            return
        source_xyz = np.asarray(points, dtype=np.float32)
        finite = np.all(np.isfinite(source_xyz[:, :3]), axis=1)
        xyz = source_xyz[:, :3] if np.all(finite) else source_xyz[finite, :3]
        if not len(xyz):
            self._cloud_pixmap = None
            self._cloud_source_points = np.empty((0, 3), dtype=np.float32)
            self._cloud_source_colors = None
            self._cloud_source_splat_scales = None
            self._cloud_source_splat_rotations = None
            self._cloud_source_splat_opacities = None
            self._cloud_total_points = 0
            self._cloud_points = np.empty((0, 3), dtype=np.float32)
            self._cloud_colors = np.empty((0, 3), dtype=np.uint8)
            self.update()
            return
        was_empty = not len(self._cloud_points)
        if colors is not None:
            source_colors = np.asarray(colors, dtype=np.uint8)
            if source_colors.shape != (len(points), 3):
                source_colors = None
            elif not np.all(finite):
                source_colors = source_colors[finite]
        else:
            source_colors = None
        is_splat = all(value is not None for value in (
            splat_scales, splat_rotations, splat_opacities))
        if is_splat:
            source_scales = np.asarray(splat_scales, dtype=np.float32)
            source_rotations = np.asarray(splat_rotations, dtype=np.float32)
            source_opacities = np.asarray(splat_opacities, dtype=np.float32).reshape(-1)
            if (
                source_scales.shape != (len(points), 3)
                or source_rotations.shape != (len(points), 4)
                or len(source_opacities) != len(points)
            ):
                is_splat = False
            elif not np.all(finite):
                source_scales = source_scales[finite]
                source_rotations = source_rotations[finite]
                source_opacities = source_opacities[finite]
        self._cloud_total_points = len(xyz)
        display_limit = 60000 if is_splat else 120000
        if len(xyz) > display_limit and is_splat:
            top_count = int(display_limit * 0.75)
            uniform_count = display_limit - top_count
            importance = source_opacities * np.square(
                np.max(source_scales, axis=1))
            top = np.argpartition(importance, -top_count)[-top_count:]
            uniform = np.linspace(
                0, len(xyz) - 1, uniform_count, dtype=np.int64)
            sample_indices = np.unique(np.concatenate((top, uniform)))
        elif len(xyz) > display_limit:
            sample_step = int(math.ceil(len(xyz) / display_limit))
            sample_indices = np.arange(0, len(xyz), sample_step)
        else:
            sample_indices = np.arange(len(xyz))
        self._cloud_source_points = np.ascontiguousarray(xyz[sample_indices, :3])
        self._cloud_source_colors = (
            None if source_colors is None
            else np.ascontiguousarray(source_colors[sample_indices]))
        if is_splat:
            self._cloud_source_splat_scales = np.ascontiguousarray(
                source_scales[sample_indices])
            self._cloud_source_splat_rotations = np.ascontiguousarray(
                source_rotations[sample_indices])
            self._cloud_source_splat_opacities = np.ascontiguousarray(
                source_opacities[sample_indices])
        else:
            self._cloud_source_splat_scales = None
            self._cloud_source_splat_rotations = None
            self._cloud_source_splat_opacities = None
        render_xyz = self._cloud_source_points
        lower = render_xyz[:, :2].min(axis=0)
        upper = render_xyz[:, :2].max(axis=0)
        extent = np.maximum(upper - lower, 0.05)
        resolution = max(float(np.max(extent)) / 1100.0, 0.01)
        width, height = np.ceil(extent / resolution).astype(int) + 1
        image = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        image[:, :] = (8, 15, 21)
        pixels = np.floor((render_xyz[:, :2] - lower) / resolution).astype(int)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
        z = render_xyz[:, 2]
        z_low, z_high = np.percentile(z, [3, 97]) if len(z) > 10 else (z.min(), z.max())
        normalized = np.clip((z - z_low) / max(float(z_high - z_low), 1e-6), 0.0, 1.0)
        colors = np.column_stack([
            35 + 210 * normalized,
            205 + 45 * normalized,
            245 - 110 * normalized,
        ]).astype(np.uint8)
        # Draw high points last, making walls and obstacles easier to distinguish.
        order = np.argsort(z)
        ordered_x = pixels[order, 0]
        ordered_y = pixels[order, 1]
        ordered_colors = colors[order]
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            draw_x = np.clip(ordered_x + dx, 0, width - 1)
            draw_y = np.clip(ordered_y + dy, 0, height - 1)
            image[draw_y, draw_x] = ordered_colors
        image = np.ascontiguousarray(np.flipud(image))
        qimage = QImage(
            image.data, image.shape[1], image.shape[0], image.strides[0],
            QImage.Format_RGB888).copy()
        self._cloud_pixmap = QPixmap.fromImage(qimage)
        self._cloud_shape = (int(height), int(width))
        self._cloud_origin = lower.astype(np.float32)
        self._cloud_resolution = resolution
        self._rebuild_cloud_3d()
        if was_empty:
            self.reset_cloud_view()
        self.update()

    @staticmethod
    def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        source /= max(float(np.linalg.norm(source)), 1e-12)
        target /= max(float(np.linalg.norm(target)), 1e-12)
        cross = np.cross(source, target)
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        cross_norm_sq = float(np.dot(cross, cross))
        if cross_norm_sq < 1e-12:
            if dot > 0.0:
                return np.eye(3, dtype=np.float32)
            helper = np.array([1.0, 0.0, 0.0])
            if abs(float(source[0])) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
            axis = np.cross(source, helper)
            axis /= np.linalg.norm(axis)
            return (-np.eye(3) + 2.0 * np.outer(axis, axis)).astype(np.float32)
        skew = np.array([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ])
        rotation = np.eye(3) + skew + skew @ skew * ((1.0 - dot) / cross_norm_sq)
        return rotation.astype(np.float32)

    def _rebuild_cloud_3d(self) -> None:
        if not len(self._cloud_source_points):
            self._cloud_points = np.empty((0, 3), dtype=np.float32)
            self._cloud_colors = np.empty((0, 3), dtype=np.uint8)
            return

        indices = ["XYZ".index(axis) for axis in self._cloud_axis_order]
        signs = np.array([
            -1.0 if flipped else 1.0 for flipped in self._cloud_axis_flips
        ], dtype=np.float32)
        axis_matrix = np.zeros((3, 3), dtype=np.float32)
        for output_axis, source_axis in enumerate(indices):
            axis_matrix[output_axis, source_axis] = signs[output_axis]
        mapped = self._cloud_source_points @ axis_matrix.T

        rotation = np.eye(3, dtype=np.float32)
        offset = np.zeros(3, dtype=np.float32)
        self._cloud_level_angle_degrees = 0.0
        self._cloud_ground_reference_z = 0.0
        if self._cloud_alignment == "auto" and len(mapped) >= 6:
            lower, upper = np.percentile(mapped, [1.0, 99.0], axis=0)
            robust = mapped[np.all((mapped >= lower) & (mapped <= upper), axis=1)]
            if len(robust) >= 6:
                level_center = np.median(robust, axis=0).astype(np.float32)
                covariance = np.cov((robust - level_center).T)
                _values, vectors = np.linalg.eigh(covariance)
                normal = vectors[:, 0]
                if normal[2] < 0.0:
                    normal = -normal
                self._cloud_level_angle_degrees = math.degrees(
                    math.acos(float(np.clip(normal[2], -1.0, 1.0))))
                rotation = self._rotation_between(
                    normal, np.array([0.0, 0.0, 1.0]))
                offset = level_center - rotation @ level_center
                # The leveling rotation used to be preview-only and retained
                # the source cloud's arbitrary Z offset.  Establish a stable
                # floor reference as well, so UI height limits such as
                # -0.20..1.50 m describe height above the leveled ground.
                leveled_robust = robust @ rotation.T + offset
                self._cloud_ground_reference_z = float(
                    np.percentile(leveled_robust[:, 2], 2.0))
                offset[2] -= self._cloud_ground_reference_z

        self._cloud_display_matrix = rotation @ axis_matrix
        self._cloud_display_offset = offset.astype(np.float32)
        displayed = (
            self._cloud_source_points @ self._cloud_display_matrix.T
            + self._cloud_display_offset
        )
        self._cloud_points = np.ascontiguousarray(displayed.astype(np.float32))
        percentile = [0.5, 99.5] if len(displayed) > 200 else [0.0, 100.0]
        bounds_low, bounds_high = np.percentile(displayed, percentile, axis=0)
        self._cloud_center = ((bounds_low + bounds_high) * 0.5).astype(np.float32)
        self._cloud_extent = np.maximum(
            bounds_high - bounds_low,
            np.array([0.1, 0.1, 0.1], dtype=np.float32),
        ).astype(np.float32)
        self._cloud_bounds_low = bounds_low.astype(np.float32)

        if self._cloud_color_mode == "rgb" and self._cloud_source_colors is not None:
            self._cloud_colors = self._cloud_source_colors.copy()
        else:
            height = displayed[:, 2]
            z_low, z_high = np.percentile(height, [3, 97])
            normalized = np.clip(
                (height - z_low) / max(float(z_high - z_low), 1e-6),
                0.0, 1.0,
            )
            self._cloud_colors = np.ascontiguousarray(np.column_stack([
                35 + 210 * normalized,
                205 + 45 * normalized,
                245 - 110 * normalized,
            ]).astype(np.uint8))

    def cloud_display_description(self) -> str:
        alignment = (
            f"自动找平 {self._cloud_level_angle_degrees:.1f}°/地面Z=0"
            if self._cloud_alignment == "auto" else "原始坐标")
        projection = "透视" if self._cloud_projection == "perspective" else "正交"
        if self._cloud_color_mode == "rgb" and self.cloud_has_rgb:
            color = (
                "GS 基础颜色" if self._cloud_source_splat_scales is not None
                else "原始 RGB")
        else:
            color = "高度着色"
        flips = [
            f"反{axis}" for axis, enabled
            in zip("XYZ", self._cloud_axis_flips) if enabled
        ]
        axes = self._cloud_axis_order + (" " + "/".join(flips) if flips else "")
        kind = "Gaussian Splat" if self._cloud_source_splat_scales is not None else "点云"
        return f"{kind} · {alignment} · {axes} · {projection} · {color}"

    def transform_cloud_points(self, points: np.ndarray) -> np.ndarray:
        """Apply the preview transform to a copy of full-resolution XYZ data."""
        xyz = np.asarray(points, dtype=np.float32)[:, :3]
        return np.ascontiguousarray(
            xyz @ self._cloud_display_matrix.T + self._cloud_display_offset)

    def _cloud_view_basis(self):
        azimuth = self._cloud_azimuth
        elevation = self._cloud_elevation
        right = np.array(
            [-math.sin(azimuth), math.cos(azimuth), 0.0], dtype=np.float32)
        up = np.array([
            -math.sin(elevation) * math.cos(azimuth),
            -math.sin(elevation) * math.sin(azimuth),
            math.cos(elevation),
        ], dtype=np.float32)
        toward_camera = np.array([
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ], dtype=np.float32)
        return right, up, toward_camera

    def _cloud_view_scale(self) -> float:
        longest = max(float(np.linalg.norm(self._cloud_extent)), 0.1)
        minimum_size = max(1.0, min(self.width(), self.height()))
        if self._cloud_projection == "perspective":
            focal_length = minimum_size * 0.82 * self._cloud_zoom
            camera_distance = longest * 1.25
            return focal_length / camera_distance
        fitted = max(1.0, minimum_size * 0.82 / longest)
        return fitted * self._cloud_zoom

    def _project_cloud_points(self, points: np.ndarray):
        right, up, depth_axis = self._cloud_view_basis()
        relative = np.asarray(points, dtype=np.float32) - self._cloud_view_target
        depth = relative @ depth_axis
        if self._cloud_projection == "perspective":
            longest = max(float(np.linalg.norm(self._cloud_extent)), 0.1)
            camera_distance = longest * 1.25
            focal_length = max(
                1.0, min(self.width(), self.height()) * 0.82 * self._cloud_zoom)
            camera_depth = camera_distance - depth
            safe_depth = np.where(camera_depth > longest * 0.02, camera_depth, np.nan)
            screen_x = self.width() * 0.5 + (relative @ right) * focal_length / safe_depth
            screen_y = self.height() * 0.5 - (relative @ up) * focal_length / safe_depth
        else:
            scale = self._cloud_view_scale()
            screen_x = self.width() * 0.5 + relative @ right * scale
            screen_y = self.height() * 0.5 - relative @ up * scale
        return screen_x, screen_y, depth

    @staticmethod
    def _quaternion_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
        """Convert Gaussian Splat quaternions (w, x, y, z) to matrices."""
        q = np.asarray(quaternions, dtype=np.float32)
        w, x, y, z = q.T
        matrices = np.empty((len(q), 3, 3), dtype=np.float32)
        matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        matrices[:, 0, 1] = 2.0 * (x * y - z * w)
        matrices[:, 0, 2] = 2.0 * (x * z + y * w)
        matrices[:, 1, 0] = 2.0 * (x * y + z * w)
        matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        matrices[:, 1, 2] = 2.0 * (y * z - x * w)
        matrices[:, 2, 0] = 2.0 * (x * z - y * w)
        matrices[:, 2, 1] = 2.0 * (y * z + x * w)
        matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        return matrices

    def _gaussian_screen_radii(self, indices: np.ndarray):
        scales = self._cloud_source_splat_scales[indices]
        rotations = self._quaternion_rotation_matrices(
            self._cloud_source_splat_rotations[indices])
        scaled_axes = rotations * scales[:, None, :]
        display_axes = np.einsum(
            "ij,njk->nik", self._cloud_display_matrix, scaled_axes)
        points = self._cloud_points[indices]
        relative = points - self._cloud_view_target
        right, up, depth_axis = self._cloud_view_basis()
        if self._cloud_projection == "perspective":
            longest = max(float(np.linalg.norm(self._cloud_extent)), 0.1)
            camera_distance = longest * 1.25
            focal = max(
                1.0, min(self.width(), self.height()) * 0.82 * self._cloud_zoom)
            camera_depth = np.maximum(
                camera_distance - relative @ depth_axis, longest * 0.02)
            camera_x = relative @ right
            camera_y = relative @ up
            jacobian_x = (
                right[None, :] * (focal / camera_depth)[:, None]
                + depth_axis[None, :]
                * (focal * camera_x / np.square(camera_depth))[:, None]
            )
            jacobian_y = -(
                up[None, :] * (focal / camera_depth)[:, None]
                + depth_axis[None, :]
                * (focal * camera_y / np.square(camera_depth))[:, None]
            )
        else:
            scale = self._cloud_view_scale()
            jacobian_x = np.broadcast_to(right * scale, (len(indices), 3))
            jacobian_y = np.broadcast_to(-up * scale, (len(indices), 3))
        projected_x = np.einsum("ni,nik->nk", jacobian_x, display_axes)
        projected_y = np.einsum("ni,nik->nk", jacobian_y, display_axes)
        radius_x = np.clip(
            3.0 * np.sqrt(np.sum(np.square(projected_x), axis=1)), 0.7, 32.0)
        radius_y = np.clip(
            3.0 * np.sqrt(np.sum(np.square(projected_y), axis=1)), 0.7, 32.0)
        return radius_x, radius_y

    def _cloud3d_widget_to_world(self, point: QPointF) -> Optional[np.ndarray]:
        """Pick a map-frame XY location on the displayed cloud ground."""
        if not len(self._cloud_points):
            return None
        right, up, depth_axis = self._cloud_view_basis()
        ground_z = (
            0.0 if self._cloud_alignment == "auto"
            else float(self._cloud_bounds_low[2]))
        relative_z = ground_z - float(self._cloud_view_target[2])
        if self._cloud_projection == "perspective":
            longest = max(float(np.linalg.norm(self._cloud_extent)), 0.1)
            camera_distance = longest * 1.25
            focal_length = max(
                1.0, min(self.width(), self.height()) * 0.82 * self._cloud_zoom)
            normalized_x = (point.x() - self.width() * 0.5) / focal_length
            normalized_y = (self.height() * 0.5 - point.y()) / focal_length
            matrix = np.array([
                right[:2] + normalized_x * depth_axis[:2],
                up[:2] + normalized_y * depth_axis[:2],
            ])
            screen = np.array([
                normalized_x * camera_distance
                - (right[2] + normalized_x * depth_axis[2]) * relative_z,
                normalized_y * camera_distance
                - (up[2] + normalized_y * depth_axis[2]) * relative_z,
            ])
        else:
            scale = self._cloud_view_scale()
            screen = np.array([
                (point.x() - self.width() * 0.5) / scale
                - right[2] * relative_z,
                (self.height() * 0.5 - point.y()) / scale
                - up[2] * relative_z,
            ])
            matrix = np.array([
                [right[0], right[1]],
                [up[0], up[1]],
            ])
        if abs(float(np.linalg.det(matrix))) < 1e-5:
            return None
        relative_xy = np.linalg.solve(matrix, screen)
        displayed = np.array([
            self._cloud_view_target[0] + relative_xy[0],
            self._cloud_view_target[1] + relative_xy[1],
            ground_z,
        ], dtype=np.float32)
        # Display transforms are rigid axis/level rotations, so their inverse
        # is the transpose.  Return the original map-frame XY used by Nav2.
        source = (
            displayed - self._cloud_display_offset
        ) @ self._cloud_display_matrix
        return source[:2]

    def _orbit_cloud_view(self, dx: float, dy: float) -> None:
        self._cloud_azimuth -= float(dx) * 0.008
        self._cloud_elevation = float(np.clip(
            self._cloud_elevation - float(dy) * 0.008,
            math.radians(8.0), math.radians(82.0),
        ))
        self.update()

    def _pan_cloud_view(self, dx: float, dy: float) -> None:
        right, up, _depth = self._cloud_view_basis()
        scale = self._cloud_view_scale()
        self._cloud_view_target += (-right * float(dx) + up * float(dy)) / scale
        self.update()

    def _zoom_cloud_view(self, wheel_steps: float) -> None:
        self._cloud_zoom = float(np.clip(
            self._cloud_zoom * math.exp(float(wheel_steps) * 0.16),
            0.08, 25.0,
        ))
        self.update()

    def set_navigation_state(
        self,
        path: Optional[np.ndarray],
        robot_xy: Optional[np.ndarray],
        robot_yaw: float,
    ) -> None:
        self._path = path
        self._robot_xy = robot_xy
        self._robot_yaw = robot_yaw
        self.update()

    def set_waypoints(self, waypoints) -> None:
        self._waypoints = [np.asarray(point).copy() for point in waypoints]
        if (
            self._selected_waypoint_index is not None
            and self._selected_waypoint_index >= len(self._waypoints)
        ):
            self._selected_waypoint_index = None
        self.update()

    def set_waypoint_selection(self, index: Optional[int]) -> None:
        self._selected_waypoint_index = (
            int(index) if index is not None and 0 <= index < len(self._waypoints)
            else None)
        self.update()

    def _view(self):
        if self._mode == "cloud":
            return (
                self._cloud_pixmap, self._cloud_shape, self._cloud_origin,
                self._cloud_resolution, 0.0)
        return (
            self._grid_pixmap, self._grid_shape, self._grid_origin,
            self._grid_resolution, self._grid_origin_yaw)

    def _map_rect(self) -> QRectF:
        pixmap, _shape, _origin, _resolution, _yaw = self._view()
        if pixmap is None:
            return QRectF()
        margin = 12.0
        available_w = max(1.0, self.width() - margin * 2)
        available_h = max(1.0, self.height() - margin * 2)
        scale = min(
            available_w / pixmap.width(),
            available_h / pixmap.height(),
        )
        width = pixmap.width() * scale
        height = pixmap.height() * scale
        return QRectF(
            (self.width() - width) * 0.5,
            (self.height() - height) * 0.5,
            width,
            height,
        )

    def world_to_widget(self, xy: np.ndarray) -> QPointF:
        target = self._map_rect()
        _pixmap, (h, w), origin, resolution, origin_yaw = self._view()
        relative = np.asarray(xy) - origin
        cos_yaw = math.cos(origin_yaw)
        sin_yaw = math.sin(origin_yaw)
        grid_x = (cos_yaw * relative[0] + sin_yaw * relative[1]) / resolution
        grid_y = (-sin_yaw * relative[0] + cos_yaw * relative[1]) / resolution
        return QPointF(
            target.left() + grid_x * target.width() / w,
            target.top() + (h - 1 - grid_y) * target.height() / h,
        )

    def widget_to_world(self, point: QPointF) -> Optional[np.ndarray]:
        if self._cloud_3d_enabled and self._mode == "cloud":
            return self._cloud3d_widget_to_world(point)
        target = self._map_rect()
        pixmap, (h, w), origin, resolution, origin_yaw = self._view()
        if (
            pixmap is None
            or resolution <= 0.0
            or not target.contains(point)
        ):
            return None
        grid_x = (point.x() - target.left()) * w / target.width()
        display_y = (point.y() - target.top()) * h / target.height()
        grid_y = h - 1 - display_y
        local = np.array(
            [grid_x * resolution, grid_y * resolution])
        cos_yaw = math.cos(origin_yaw)
        sin_yaw = math.sin(origin_yaw)
        return origin + np.array([
            cos_yaw * local[0] - sin_yaw * local[1],
            sin_yaw * local[0] + cos_yaw * local[1],
        ])

    def mousePressEvent(self, event) -> None:
        if self._cloud_3d_enabled and self._mode == "cloud":
            is_orbit = event.button() == Qt.LeftButton and self._edit_tool == "navigate"
            is_pan = event.button() in (Qt.RightButton, Qt.MiddleButton)
            if is_orbit or is_pan:
                self._view_drag_button = event.button()
                self._last_mouse_position = event.pos()
                self._view_drag_distance = 0.0
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
        if event.button() == Qt.LeftButton:
            world = self.widget_to_world(event.localPos())
            if world is not None:
                if self._edit_tool == "navigate":
                    self.point_selected.emit(float(world[0]), float(world[1]))
                else:
                    self._painting = True
                    self.edit_started.emit()
                    self.edit_requested.emit(
                        self._mode, self._edit_tool, float(world[0]),
                        float(world[1]), self._brush_radius)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._view_drag_button != Qt.NoButton and self._last_mouse_position is not None:
            delta = event.pos() - self._last_mouse_position
            self._last_mouse_position = event.pos()
            self._view_drag_distance += abs(delta.x()) + abs(delta.y())
            if self._view_drag_distance <= 4.0:
                event.accept()
                return
            if self._view_drag_button == Qt.LeftButton:
                self._orbit_cloud_view(delta.x(), delta.y())
            else:
                self._pan_cloud_view(delta.x(), delta.y())
            event.accept()
            return
        if self._painting and event.buttons() & Qt.LeftButton:
            world = self.widget_to_world(event.localPos())
            if world is not None:
                self.edit_requested.emit(
                    self._mode, self._edit_tool, float(world[0]),
                    float(world[1]), self._brush_radius)
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == self._view_drag_button:
            select_point = (
                event.button() == Qt.LeftButton
                and self._edit_tool == "navigate"
                and self._view_drag_distance <= 4.0
            )
            self._view_drag_button = Qt.NoButton
            self._last_mouse_position = None
            self._view_drag_distance = 0.0
            self._update_cursor()
            if select_point:
                world = self.widget_to_world(event.localPos())
                if world is not None:
                    self.point_selected.emit(float(world[0]), float(world[1]))
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self._painting = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        if self._cloud_3d_enabled and self._mode == "cloud":
            delta = event.angleDelta().y()
            if delta:
                self._zoom_cloud_view(delta / 120.0)
                event.accept()
                return
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._cloud_3d_enabled and self._mode == "cloud":
            self.reset_cloud_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#111820"))
        if self._cloud_3d_enabled and self._mode == "cloud":
            self._paint_cloud_3d(painter)
            return
        pixmap, _shape, _origin, _resolution, _yaw = self._view()
        if pixmap is None:
            painter.setPen(QColor("#8999a5"))
            painter.setFont(QFont("Sans Serif", 13))
            waiting = "尚未加载点云地图" if self._mode == "cloud" else "等待栅格地图 /map"
            painter.drawText(self.rect(), Qt.AlignCenter, waiting)
            return
        target = self._map_rect()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
        painter.setClipRect(target)
        self._draw_global_path(painter)
        self._draw_robot(painter)
        for index, waypoint in enumerate(self._waypoints):
            is_last = index == len(self._waypoints) - 1
            if index == self._selected_waypoint_index:
                color = QColor("#4ed8ff")
            else:
                color = QColor("#ff5f67") if is_last else QColor("#ffc857")
            self._draw_marker(painter, waypoint, color, str(index + 1))

    def _paint_cloud_3d(self, painter: QPainter) -> None:
        if not len(self._cloud_points):
            painter.setPen(QColor("#8999a5"))
            painter.setFont(QFont("Sans Serif", 13))
            painter.drawText(self.rect(), Qt.AlignCenter, "尚未加载点云地图")
            return

        width, height = max(1, self.width()), max(1, self.height())
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:, :] = (8, 14, 20)
        screen_x, screen_y, depth = self._project_cloud_points(self._cloud_points)
        render_gaussians = (
            self._cloud_source_splat_scales is not None
            and self._view_drag_button == Qt.NoButton
        )
        projectable = np.isfinite(screen_x) & np.isfinite(screen_y)
        pixel_x = np.rint(screen_x[projectable]).astype(np.int32)
        pixel_y = np.rint(screen_y[projectable]).astype(np.int32)
        projected_depth = depth[projectable]
        projected_colors = self._cloud_colors[projectable]
        visible = (
            (pixel_x >= 0) & (pixel_x < width)
            & (pixel_y >= 0) & (pixel_y < height)
        )
        if np.any(visible) and not render_gaussians:
            pixel_x = pixel_x[visible]
            pixel_y = pixel_y[visible]
            visible_depth = projected_depth[visible]
            colors = projected_colors[visible].astype(np.float32)
            depth_span = max(float(np.ptp(visible_depth)), 1e-6)
            light = 0.60 + 0.40 * (
                (visible_depth - float(visible_depth.min())) / depth_span)
            colors = np.clip(colors * light[:, None], 0, 255).astype(np.uint8)
            order = np.argsort(visible_depth)
            ordered_x = pixel_x[order]
            ordered_y = pixel_y[order]
            ordered_colors = colors[order]
            point_size = 2 if len(ordered_x) < 90000 else 1
            for offset_x in range(point_size):
                for offset_y in range(point_size):
                    draw_x = np.clip(ordered_x + offset_x, 0, width - 1)
                    draw_y = np.clip(ordered_y + offset_y, 0, height - 1)
                    image[draw_y, draw_x] = ordered_colors

        qimage = QImage(
            image.data, width, height, image.strides[0],
            QImage.Format_RGB888,
        ).copy()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(self.rect(), qimage)
        if render_gaussians:
            self._draw_gaussian_splats(
                painter, screen_x, screen_y, depth, projectable)
        self._draw_cloud_3d_reference(painter)
        self._draw_cloud_3d_navigation(painter)

        compact = width < 520 or height < 360
        painter.setPen(QColor("#c4d3dc"))
        painter.setFont(QFont("Sans Serif", 10, QFont.DemiBold))
        kind = "Gaussian Splat" if self._cloud_source_splat_scales is not None else "三维点云"
        painter.drawText(
            QRectF(18, 14, width - 36, 28),
            Qt.AlignLeft | Qt.AlignVCenter,
            (
                f"{kind} · {len(self._cloud_points):,} 点"
                if compact else
                f"{kind} · 渲染 {len(self._cloud_points):,} / "
                f"{self._cloud_total_points:,}"
            ),
        )
        if not compact:
            painter.setPen(QColor("#71bed8"))
            painter.setFont(QFont("Sans Serif", 9))
            painter.drawText(
                QRectF(18, 40, width - 36, 24),
                Qt.AlignLeft | Qt.AlignVCenter,
                self.cloud_display_description(),
            )
        painter.setPen(QColor("#91a5b2"))
        painter.setFont(QFont("Sans Serif", 9))
        painter.drawText(
            QRectF(18, height - 42, width - 36, 28),
            Qt.AlignLeft | Qt.AlignVCenter,
            (
                "拖动旋转 · 右键平移 · 滚轮缩放"
                if compact else
                "左键拖动旋转  ·  右键/中键拖动平移  ·  "
                "滚轮缩放  ·  双击复位"
            ),
        )

    def _draw_gaussian_splats(
        self,
        painter: QPainter,
        screen_x: np.ndarray,
        screen_y: np.ndarray,
        depth: np.ndarray,
        projectable: np.ndarray,
    ) -> None:
        margin = 34.0
        visible = (
            projectable
            & (screen_x >= -margin) & (screen_x < self.width() + margin)
            & (screen_y >= -margin) & (screen_y < self.height() + margin)
            & (self._cloud_source_splat_opacities >= 0.01)
        )
        indices = np.flatnonzero(visible)
        if not len(indices):
            return
        radius_x, radius_y = self._gaussian_screen_radii(indices)
        order = np.argsort(depth[indices])
        indices = indices[order]
        radius_x = radius_x[order]
        radius_y = radius_y[order]
        colors = self._cloud_colors[indices]
        opacities = self._cloud_source_splat_opacities[indices]
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(Qt.NoPen)
        for index, rx, ry, color, opacity in zip(
            indices, radius_x, radius_y, colors, opacities,
        ):
            alpha = int(np.clip(float(opacity) * 205.0, 2.0, 245.0))
            painter.setBrush(QColor(
                int(color[0]), int(color[1]), int(color[2]), alpha))
            painter.drawEllipse(QRectF(
                float(screen_x[index] - rx), float(screen_y[index] - ry),
                float(rx * 2.0), float(ry * 2.0),
            ))

    def _draw_cloud_3d_reference(self, painter: QPainter) -> None:
        """Draw a ground grid and XYZ axes to make depth unambiguous."""
        half = max(float(self._cloud_extent[0]), float(self._cloud_extent[1])) * 0.55
        half = max(half, 0.5)
        ground_z = float(self._cloud_bounds_low[2])
        center_x, center_y = self._cloud_center[:2]
        ticks = np.linspace(-half, half, 11)
        painter.setPen(QPen(QColor(87, 112, 126, 70), 1.0))
        for tick in ticks:
            lines = (
                np.array([
                    [center_x - half, center_y + tick, ground_z],
                    [center_x + half, center_y + tick, ground_z],
                ]),
                np.array([
                    [center_x + tick, center_y - half, ground_z],
                    [center_x + tick, center_y + half, ground_z],
                ]),
            )
            for line in lines:
                x, y, _depth = self._project_cloud_points(line)
                painter.drawLine(QPointF(x[0], y[0]), QPointF(x[1], y[1]))

        origin = np.array([center_x, center_y, ground_z], dtype=np.float32)
        axis_length = max(half * 0.28, float(self._cloud_extent[2]), 0.3)
        axes = (
            (np.array([axis_length, 0.0, 0.0]), QColor("#ff5f67"), "X"),
            (np.array([0.0, axis_length, 0.0]), QColor("#66e09a"), "Y"),
            (np.array([0.0, 0.0, axis_length]), QColor("#4ed8ff"), "Z"),
        )
        for vector, color, label in axes:
            x, y, _depth = self._project_cloud_points(
                np.vstack((origin, origin + vector)))
            painter.setPen(QPen(color, 2.2))
            painter.drawLine(QPointF(x[0], y[0]), QPointF(x[1], y[1]))
            painter.drawText(QPointF(x[1] + 4, y[1] - 3), label)

    def _cloud_ground_points_from_xy(self, xy: np.ndarray) -> np.ndarray:
        """Place map-frame XY coordinates on the leveled cloud ground."""
        points_xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        ground_z = (
            0.0 if self._cloud_alignment == "auto"
            else float(self._cloud_bounds_low[2]))
        matrix = self._cloud_display_matrix
        denominator = float(matrix[2, 2])
        if abs(denominator) < 1e-6:
            return np.column_stack((
                points_xy,
                np.full(len(points_xy), ground_z + 0.04, dtype=np.float32),
            ))
        source_z = (
            ground_z - self._cloud_display_offset[2]
            - points_xy[:, 0] * matrix[2, 0]
            - points_xy[:, 1] * matrix[2, 1]
        ) / denominator
        source = np.column_stack((points_xy, source_z)).astype(np.float32)
        displayed = self.transform_cloud_points(source)
        displayed[:, 2] += 0.04
        return displayed

    def _draw_cloud_3d_navigation(self, painter: QPainter) -> None:
        """Overlay the active global path and robot on the 3-D ground plane."""
        if self._path is not None and len(self._path) >= 2:
            path_points = self._cloud_ground_points_from_xy(self._path[:, :2])
            screen_x, screen_y, _depth = self._project_cloud_points(path_points)
            finite = np.isfinite(screen_x) & np.isfinite(screen_y)
            valid_indices = np.flatnonzero(finite)
            if len(valid_indices) >= 2:
                path = QPainterPath(QPointF(
                    float(screen_x[valid_indices[0]]),
                    float(screen_y[valid_indices[0]]),
                ))
                for index in valid_indices[1:]:
                    path.lineTo(QPointF(
                        float(screen_x[index]), float(screen_y[index])))
                painter.setPen(QPen(
                    QColor("#00d5ff"), 3.2, Qt.SolidLine, Qt.RoundCap))
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)

        for index, waypoint in enumerate(self._waypoints):
            point_3d = self._cloud_ground_points_from_xy(
                np.asarray(waypoint, dtype=np.float32)[:2])
            screen_x, screen_y, _depth = self._project_cloud_points(point_3d)
            if not np.isfinite(screen_x[0]) or not np.isfinite(screen_y[0]):
                continue
            point = QPointF(float(screen_x[0]), float(screen_y[0]))
            is_last = index == len(self._waypoints) - 1
            if index == self._selected_waypoint_index:
                color = QColor("#4ed8ff")
            else:
                color = QColor("#ff5f67") if is_last else QColor("#ffc857")
            painter.setPen(QPen(QColor("#ffffff"), 1.8))
            painter.setBrush(color)
            painter.drawEllipse(point, 6, 6)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(point + QPointF(8, -7), str(index + 1))

        if self._robot_xy is None:
            return
        direction = np.array([
            math.cos(self._robot_yaw), math.sin(self._robot_yaw)],
            dtype=np.float32)
        robot_points = self._cloud_ground_points_from_xy(np.vstack((
            self._robot_xy[:2], self._robot_xy[:2] + direction,
        )))
        screen_x, screen_y, _depth = self._project_cloud_points(robot_points)
        if not np.all(np.isfinite(screen_x)) or not np.all(np.isfinite(screen_y)):
            return
        center = np.array([screen_x[0], screen_y[0]], dtype=float)
        heading = np.array(
            [screen_x[1] - screen_x[0], screen_y[1] - screen_y[0]],
            dtype=float)
        length = float(np.linalg.norm(heading))
        if length < 1e-5:
            heading = np.array([0.0, -1.0])
        else:
            heading /= length
        side = np.array([-heading[1], heading[0]])
        arrow = QPolygonF([
            QPointF(*(center + heading * 10.0)),
            QPointF(*(center - heading * 7.0 + side * 6.0)),
            QPointF(*(center - heading * 4.0)),
            QPointF(*(center - heading * 7.0 - side * 6.0)),
        ])
        painter.setPen(QPen(QColor("#073346"), 1.8))
        painter.setBrush(QColor("#ffffff"))
        painter.drawPolygon(arrow)

    def _draw_global_path(self, painter: QPainter) -> None:
        if self._path is None or len(self._path) < 2:
            return
        path = QPainterPath(self.world_to_widget(self._path[0]))
        for point in self._path[1:]:
            path.lineTo(self.world_to_widget(point))
        painter.setPen(QPen(QColor("#00b9f2"), 3.2, Qt.SolidLine, Qt.RoundCap))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

    def _draw_robot(self, painter: QPainter) -> None:
        if self._robot_xy is None:
            return
        center = self.world_to_widget(self._robot_xy)
        _pixmap, _shape, _origin, _resolution, origin_yaw = self._view()
        heading = -(self._robot_yaw - origin_yaw)
        base = np.array([[12, 0], [-8, -7], [-5, 0], [-8, 7]], dtype=float)
        rotation = np.array([
            [math.cos(heading), -math.sin(heading)],
            [math.sin(heading), math.cos(heading)],
        ])
        polygon = QPolygonF([
            QPointF(center.x() + p[0], center.y() + p[1])
            for p in base @ rotation.T
        ])
        painter.setPen(QPen(QColor("#053347"), 2))
        painter.setBrush(QColor("#ffffff"))
        painter.drawPolygon(polygon)

    def _draw_marker(
        self,
        painter: QPainter,
        xy: Optional[np.ndarray],
        color: QColor,
        label: str,
    ) -> None:
        if xy is None:
            return
        point = self.world_to_widget(xy)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.setBrush(color)
        painter.drawEllipse(point, 7, 7)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Sans Serif", 9, QFont.DemiBold))
        painter.drawText(point + QPointF(10, -8), label)


class QtNavRosNode(Node):
    """ROS-facing state and commands consumed by the Qt widgets."""

    def __init__(self) -> None:
        super().__init__("gs_qt_nav")
        self.declare_parameter("camera_topic", "/color/image_raw")
        self.declare_parameter("camera_info_topic", "/color/camera_info")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("path_topic", "/global_plan")
        self.declare_parameter("pointcloud_topic", "")
        self.declare_parameter("pointcloud_map_path", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("projection_mode", "auto")
        self.declare_parameter("route_width_m", 0.95)
        self.declare_parameter("camera_height_m", 0.72)
        self.declare_parameter("horizon_ratio", 0.43)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.camera_frame_override = str(self.get_parameter("camera_frame").value)
        self.projection_mode = str(self.get_parameter("projection_mode").value)
        self.style = RenderStyle(
            ribbon_width_m=float(self.get_parameter("route_width_m").value),
            camera_height_m=float(self.get_parameter("camera_height_m").value),
            horizon_ratio=float(self.get_parameter("horizon_ratio").value),
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(
            Image, str(self.get_parameter("camera_topic").value),
            self._on_image, sensor_qos)
        self.create_subscription(
            CameraInfo, str(self.get_parameter("camera_info_topic").value),
            self._on_camera_info, sensor_qos)
        self.create_subscription(
            OccupancyGrid, str(self.get_parameter("map_topic").value),
            self._on_map, map_qos)
        self.create_subscription(
            Path, str(self.get_parameter("path_topic").value),
            self._on_path, reliable_qos)
        pointcloud_topic = str(self.get_parameter("pointcloud_topic").value)
        if pointcloud_topic:
            self.create_subscription(
                PointCloud2, pointcloud_topic, self._on_pointcloud, sensor_qos)
        self.navigation_client = ActionClient(
            self, NavigateThroughPoses, "/navigate_through_poses")

        self.latest_image: Optional[np.ndarray] = None
        self.camera_frame = ""
        self.camera_matrix: Optional[np.ndarray] = None
        self.camera_info_size: Optional[Tuple[int, int]] = None
        self.path = np.empty((0, 3), dtype=np.float32)
        self.path_frame = self.map_frame
        self.occupancy: Optional[np.ndarray] = None
        self.grid_frame = self.map_frame
        self.map_origin = np.zeros(2, dtype=np.float32)
        self.map_origin_yaw = 0.0
        self.map_resolution = 0.0
        self.map_revision = 0
        self.latest_pointcloud: Optional[np.ndarray] = None
        self.pointcloud_frame = self.map_frame
        self.pointcloud_revision = 0
        self.navigation_status = "请在地图上添加至少一个途径点"
        self.navigation_active = False
        self.navigation_result_status: Optional[int] = None
        self.navigation_result_revision = 0
        self.goal_handle = None
        self.get_logger().info(
            "Qt navigation ready: camera=/color/image_raw, map=/map, "
            "path=/global_plan, action=/navigate_through_poses")

    def _on_image(self, msg: Image) -> None:
        try:
            self.latest_image = image_to_bgr(msg)
            self.camera_frame = self.camera_frame_override or msg.header.frame_id
        except (ValueError, TypeError) as exc:
            self.navigation_status = f"相机格式错误: {exc}"

    def _on_camera_info(self, msg: CameraInfo) -> None:
        matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if matrix[0, 0] > 0 and matrix[1, 1] > 0:
            self.camera_matrix = matrix
            self.camera_info_size = (msg.width, msg.height)

    def _on_path(self, msg: Path) -> None:
        if not self.navigation_active:
            self.clear_navigation_path()
            return
        self.path = np.array([
            (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
            for pose in msg.poses
        ], dtype=np.float32).reshape(-1, 3)
        self.path_frame = (
            msg.header.frame_id
            or (msg.poses[0].header.frame_id if msg.poses else "")
            or self.map_frame
        )

    def _on_map(self, msg: OccupancyGrid) -> None:
        expected = int(msg.info.width * msg.info.height)
        data = np.asarray(msg.data, dtype=np.int8)
        if expected == 0 or len(data) != expected:
            return
        self.occupancy = data.reshape(msg.info.height, msg.info.width).copy()
        self.grid_frame = msg.header.frame_id or self.map_frame
        self.map_resolution = float(msg.info.resolution)
        origin = msg.info.origin
        self.map_origin = np.array(
            [origin.position.x, origin.position.y], dtype=np.float32)
        q = origin.orientation
        self.map_origin_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.map_revision += 1

    def _on_pointcloud(self, msg: PointCloud2) -> None:
        try:
            self.latest_pointcloud = pointcloud2_to_xyz(msg)
            self.pointcloud_frame = msg.header.frame_id or self.map_frame
            self.pointcloud_revision += 1
        except (ValueError, TypeError) as exc:
            self.navigation_status = f"点云格式错误: {exc}"

    def lookup_matrix(self, target: str, source: str) -> Optional[np.ndarray]:
        if target == source:
            return np.eye(4)
        try:
            transform = self.tf_buffer.lookup_transform(target, source, Time())
            return transform_matrix(transform.transform)
        except TransformException:
            return None

    def projected_route(self, image_shape):
        if len(self.path) < 2:
            return None
        left, right = make_ribbon(self.path, self.style.ribbon_width_m)
        if (
            self.projection_mode in ("auto", "calibrated")
            and self.camera_matrix is not None
            and self.camera_frame
        ):
            matrix = self.lookup_matrix(self.camera_frame, self.path_frame)
            if matrix is not None:
                k = self.camera_matrix.copy()
                if self.camera_info_size:
                    k[0, :] *= image_shape[1] / max(self.camera_info_size[0], 1)
                    k[1, :] *= image_shape[0] / max(self.camera_info_size[1], 1)
                center_px, center_valid = project_optical(
                    apply_transform(self.path, matrix), k)
                left_px, left_valid = project_optical(
                    apply_transform(left, matrix), k)
                right_px, right_valid = project_optical(
                    apply_transform(right, matrix), k)
                return (
                    center_px, left_px, right_px,
                    center_valid & left_valid & right_valid,
                )
        if self.projection_mode == "calibrated":
            return None
        matrix = self.lookup_matrix(self.base_frame, self.path_frame)
        if matrix is None:
            return None
        center_px, center_valid = project_ground(
            apply_transform(self.path, matrix), image_shape, self.style)
        left_px, left_valid = project_ground(
            apply_transform(left, matrix), image_shape, self.style)
        right_px, right_valid = project_ground(
            apply_transform(right, matrix), image_shape, self.style)
        return (
            center_px, left_px, right_px,
            center_valid & left_valid & right_valid,
        )

    def path_in_frame(self, target: str) -> Optional[np.ndarray]:
        if not len(self.path):
            return self.path.copy()
        matrix = self.lookup_matrix(target, self.path_frame)
        return None if matrix is None else apply_transform(self.path, matrix)

    def map_navigation_state(self):
        path = self.path_in_frame(self.grid_frame)
        path_xy = None if path is None or not len(path) else path[:, :2]
        matrix = self.lookup_matrix(self.grid_frame, self.base_frame)
        if matrix is None:
            return path_xy, None, 0.0
        robot_xy = matrix[:2, 3].copy()
        robot_yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        return path_xy, robot_xy, robot_yaw

    def remaining_distance(self) -> float:
        path = self.path_in_frame(self.base_frame)
        if path is None or not len(path):
            return cumulative_distance(self.path)
        nearest = int(np.argmin(np.linalg.norm(path[:, :2], axis=1)))
        return cumulative_distance(path[nearest:])

    def clear_navigation_path(self) -> None:
        self.path = np.empty((0, 3), dtype=np.float32)
        self.path_frame = self.map_frame

    def send_navigation_waypoints(self, waypoints) -> bool:
        if not waypoints:
            self.navigation_status = "请先添加途径点"
            return False
        if not self.navigation_client.server_is_ready():
            self.navigation_status = "导航服务 /navigate_through_poses 尚未就绪"
            return False
        self.clear_navigation_path()
        self.navigation_active = True
        self.navigation_result_status = None
        goal_msg = NavigateThroughPoses.Goal()
        for index, waypoint in enumerate(waypoints):
            if index + 1 < len(waypoints):
                direction = waypoints[index + 1] - waypoint
            elif index > 0:
                direction = waypoint - waypoints[index - 1]
            else:
                direction = np.array([1.0, 0.0])
            heading = math.atan2(float(direction[1]), float(direction[0]))
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = self.grid_frame
            pose.pose.position.x = float(waypoint[0])
            pose.pose.position.y = float(waypoint[1])
            pose.pose.orientation.z, pose.pose.orientation.w = yaw_quaternion(heading)
            goal_msg.poses.append(pose)
        self.navigation_status = f"正在提交 {len(waypoints)} 个途径点"
        future = self.navigation_client.send_goal_async(
            goal_msg, feedback_callback=self._on_navigation_feedback)
        future.add_done_callback(self._on_goal_response)
        return True

    def _on_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # ROS future propagates transport errors here.
            self._set_navigation_terminal(
                -1, f"导航目标发送失败: {exc}")
            return
        if not goal_handle.accepted:
            self._set_navigation_terminal(-2, "导航目标被 Nav2 拒绝")
            return
        # The user may have exited while the asynchronous request was pending.
        if not self.navigation_active:
            goal_handle.cancel_goal_async()
            self.clear_navigation_path()
            return
        self.goal_handle = goal_handle
        self.navigation_status = "导航目标已接受"
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self._on_navigation_result)

    def _on_navigation_feedback(self, feedback) -> None:
        if not self.navigation_active:
            return
        current = feedback.feedback.number_of_poses_remaining
        self.navigation_status = f"导航中 · 剩余 {current} 个途径点"

    def _on_navigation_result(self, future) -> None:
        try:
            status = future.result().status
        except Exception as exc:
            self._set_navigation_terminal(-3, f"读取导航结果失败: {exc}")
            return
        labels = {
            4: "导航成功",
            5: "导航已取消",
            6: "导航失败",
        }
        self._set_navigation_terminal(
            status, labels.get(status, f"导航结束，状态码 {status}"))

    def _set_navigation_terminal(self, status: int, text: str) -> None:
        """Publish one terminal-state revision for the Qt page controller."""
        self.navigation_status = text
        self.navigation_active = False
        self.clear_navigation_path()
        self.goal_handle = None
        self.navigation_result_status = int(status)
        self.navigation_result_revision += 1

    def cancel_navigation(self) -> None:
        self.navigation_active = False
        self.clear_navigation_path()
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
            self.navigation_status = "已退出并取消整个导航任务"
        else:
            self.navigation_status = "已退出导航"


class ActiveNavigationPage(QWidget):
    """Distraction-free camera view used while a task is active."""

    def __init__(self, exit_callback, mode_callback=None) -> None:
        super().__init__()
        self._mode_callback = mode_callback
        self.camera_panel = CameraPanel(show_status=True)
        self.camera_panel.setParent(self)
        self.map_panel = MapPanel(cloud_3d=True)
        self.map_panel.setObjectName("activeMap")
        self.map_panel.set_edit_tool("navigate", 0.20)
        self.map_panel.setParent(self)
        self.grid_mode_button = QPushButton("栅格")
        self.grid_mode_button.setObjectName("mapModeButton")
        self.grid_mode_button.setCheckable(True)
        self.grid_mode_button.setChecked(True)
        self.grid_mode_button.clicked.connect(lambda: self.set_map_mode("grid"))
        self.grid_mode_button.setParent(self)
        self.cloud_mode_button = QPushButton("点云")
        self.cloud_mode_button.setObjectName("mapModeButton")
        self.cloud_mode_button.setCheckable(True)
        self.cloud_mode_button.clicked.connect(lambda: self.set_map_mode("cloud"))
        self.cloud_mode_button.setParent(self)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.grid_mode_button)
        self.mode_group.addButton(self.cloud_mode_button)
        self.exit_button = QPushButton("退出导航")
        self.exit_button.setObjectName("exitNavigationButton")
        self.exit_button.clicked.connect(exit_callback)
        self.exit_button.setParent(self)
        self.result_overlay = QLabel(self)
        self.result_overlay.setObjectName("navigationResultOverlay")
        self.result_overlay.setAlignment(Qt.AlignCenter)
        self.result_overlay.setWordWrap(True)
        self.result_overlay.hide()

    def show_navigation_result(self, status: Optional[int], text: str) -> None:
        if status == 4:
            title = "✓ 已到达目的地"
            detail = "导航任务完成，即将返回地图"
            border = "#43d17d"
            background = "rgba(10, 42, 27, 235)"
        elif status == 5:
            title = "导航已取消"
            detail = "即将返回地图"
            border = "#ffc857"
            background = "rgba(49, 39, 13, 235)"
        else:
            title = "导航已结束"
            detail = text
            border = "#ff737b"
            background = "rgba(50, 19, 24, 235)"
        self.result_overlay.setText(f"{title}\n{detail}")
        self.result_overlay.setStyleSheet(
            "QLabel#navigationResultOverlay {"
            f"background: {background}; border: 2px solid {border};"
            "border-radius: 16px; color: #f5f8fa;"
            "font-size: 18px; font-weight: 700; padding: 18px;"
            "}")
        self.result_overlay.show()
        self.result_overlay.raise_()
        self.exit_button.setText("立即返回地图")

    def clear_navigation_result(self) -> None:
        self.result_overlay.hide()
        self.result_overlay.clear()
        self.exit_button.setText("退出导航")

    def set_map_mode(self, mode: str, notify: bool = True) -> None:
        self.map_panel.set_display_mode(mode)
        self.grid_mode_button.setChecked(mode == "grid")
        self.cloud_mode_button.setChecked(mode == "cloud")
        if notify and self._mode_callback is not None:
            self._mode_callback(mode)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.camera_panel.setGeometry(self.rect())
        map_w = min(370, max(280, int(self.width() * 0.25)))
        map_h = int(map_w * 0.70)
        map_x = 26
        map_y = self.height() - map_h - 88
        self.map_panel.setGeometry(map_x, map_y, map_w, map_h)
        self.grid_mode_button.setGeometry(map_x + 12, map_y + 10, 62, 28)
        self.cloud_mode_button.setGeometry(map_x + 78, map_y + 10, 62, 28)
        button_w, button_h = 190, 46
        self.exit_button.setGeometry(
            (self.width() - button_w) // 2,
            self.height() - button_h - 20,
            button_w,
            button_h,
        )
        overlay_w = min(500, max(320, self.width() - 80))
        overlay_h = 126
        self.result_overlay.setGeometry(
            (self.width() - overlay_w) // 2,
            max(74, int(self.height() * 0.18)),
            overlay_w,
            overlay_h,
        )
        self.map_panel.raise_()
        self.grid_mode_button.raise_()
        self.cloud_mode_button.raise_()
        self.exit_button.raise_()
        if not self.result_overlay.isHidden():
            self.result_overlay.raise_()


class NavigationWindow(QMainWindow):
    def __init__(self, node: QtNavRosNode) -> None:
        super().__init__()
        self.node = node
        self.waypoints = []
        self.waypoint_edit_index: Optional[int] = None
        self.last_map_revision = -1
        self.last_pointcloud_revision = -1
        self.last_navigation_result_revision = int(
            getattr(node, "navigation_result_revision", 0))
        self.local_grid: Optional[GridMap] = None
        self.local_cloud: Optional[PointCloudMap] = None
        self._grid_original: Optional[np.ndarray] = None
        self._cloud_original: Optional[np.ndarray] = None
        self._cloud_original_colors: Optional[np.ndarray] = None
        self._grid_edit_revision = 0
        self._cloud_edit_revision = 0
        self._rendered_grid_edit_revision = -1
        self._rendered_cloud_edit_revision = -1
        self._undo_stack = []
        self.sensor_processes = {}
        self.sensor_state_labels = {}
        self.sensor_start_buttons = {}
        self.sensor_stop_buttons = {}
        self.sensor_log_views = {}
        self.setWindowTitle("GS Navigation Console")
        self.resize(1500, 900)
        self.setMinimumSize(1100, 680)
        self._build_ui()
        self._load_configured_pointcloud()
        self.pending_navigation_result_status: Optional[int] = None
        self.navigation_return_timer = QTimer(self)
        self.navigation_return_timer.setSingleShot(True)
        self.navigation_return_timer.setInterval(2200)
        self.navigation_return_timer.timeout.connect(
            self._finish_navigation_return)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh_timer.start(33)

    def _build_ui(self) -> None:
        """Build three distinct workspaces: navigation, map tools, active nav."""
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(14)

        header = QHBoxLayout()
        title = QLabel("GS NAVIGATION")
        title.setObjectName("title")
        subtitle = QLabel("实时相机 · 栅格地图 · Nav2 全局导航")
        subtitle.setObjectName("subtitle")
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        self.open_sensor_tools_button = QPushButton("传感器管理")
        self.open_sensor_tools_button.setObjectName("workspaceButton")
        self.open_sensor_tools_button.clicked.connect(self.show_sensor_tools)
        header.addWidget(self.open_sensor_tools_button)
        self.open_map_tools_button = QPushButton("地图处理")
        self.open_map_tools_button.setObjectName("workspaceButton")
        self.open_map_tools_button.clicked.connect(self.show_map_tools)
        header.addWidget(self.open_map_tools_button)
        self.ros_status = QLabel("ROS 2 ONLINE")
        self.ros_status.setObjectName("onlineChip")
        header.addWidget(self.ros_status)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        map_card = QFrame()
        map_card.setObjectName("sidePanel")
        map_layout = QVBoxLayout(map_card)
        map_layout.setContentsMargins(16, 16, 16, 16)
        map_layout.setSpacing(10)
        map_header = QHBoxLayout()
        map_title = QLabel("导航地图")
        map_title.setObjectName("sectionTitle")
        map_header.addWidget(map_title)
        map_header.addStretch(1)
        self.grid_mode_button = QPushButton("栅格")
        self.grid_mode_button.setObjectName("mapModeButton")
        self.grid_mode_button.setCheckable(True)
        self.grid_mode_button.setChecked(True)
        self.cloud_mode_button = QPushButton("点云")
        self.cloud_mode_button.setObjectName("mapModeButton")
        self.cloud_mode_button.setCheckable(True)
        self.map_mode_group = QButtonGroup(self)
        self.map_mode_group.setExclusive(True)
        self.map_mode_group.addButton(self.grid_mode_button)
        self.map_mode_group.addButton(self.cloud_mode_button)
        self.grid_mode_button.clicked.connect(
            lambda: self.set_navigation_map_mode("grid"))
        self.cloud_mode_button.clicked.connect(
            lambda: self.set_navigation_map_mode("cloud"))
        map_header.addWidget(self.grid_mode_button)
        map_header.addWidget(self.cloud_mode_button)
        map_layout.addLayout(map_header)
        self.map_panel = MapPanel(cloud_3d=True)
        self.map_panel.setObjectName("mapPanel")
        self.map_panel.point_selected.connect(self.on_map_point)
        self.map_panel.set_edit_tool("navigate", 0.20)
        map_layout.addWidget(self.map_panel, 1)

        hint = QLabel(
            "单击添加途径点；点云模式可左键拖动旋转、右键平移、滚轮缩放")
        hint.setObjectName("hint")
        map_layout.addWidget(hint)
        splitter.addWidget(map_card)

        side = QFrame()
        side.setObjectName("sidePanel")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(10)
        camera_title = QLabel("实时相机")
        camera_title.setObjectName("sectionTitle")
        side_layout.addWidget(camera_title)
        self.camera_panel = CameraPanel(show_status=False, compact=True)
        self.camera_panel.setObjectName("card")
        self.camera_panel.setMinimumHeight(170)
        self.camera_panel.setMaximumHeight(260)
        side_layout.addWidget(self.camera_panel)

        self.waypoint_title = QLabel("途径点（尚未添加）")
        self.waypoint_title.setObjectName("sectionTitle")
        side_layout.addWidget(self.waypoint_title)
        self.waypoint_list = QListWidget()
        self.waypoint_list.setObjectName("waypointList")
        self.waypoint_list.setMinimumHeight(130)
        self.waypoint_list.currentRowChanged.connect(
            self._on_waypoint_selection_changed)
        self.waypoint_list.itemDoubleClicked.connect(
            self.begin_waypoint_reselection)
        side_layout.addWidget(self.waypoint_list, 1)

        waypoint_actions = QHBoxLayout()
        self.edit_waypoint_button = QPushButton("重新选点")
        self.edit_waypoint_button.setObjectName("secondaryButton")
        self.edit_waypoint_button.setEnabled(False)
        self.edit_waypoint_button.clicked.connect(
            self.begin_waypoint_reselection)
        self.delete_waypoint_button = QPushButton("删除选中点")
        self.delete_waypoint_button.setObjectName("dangerButton")
        self.delete_waypoint_button.setEnabled(False)
        self.delete_waypoint_button.clicked.connect(
            self.delete_selected_waypoint)
        waypoint_actions.addWidget(self.edit_waypoint_button)
        waypoint_actions.addWidget(self.delete_waypoint_button)
        side_layout.addLayout(waypoint_actions)

        buttons = QHBoxLayout()
        self.clear_button = QPushButton("清空途径点")
        self.clear_button.setObjectName("secondaryButton")
        self.clear_button.clicked.connect(self.clear_selection)
        self.start_button = QPushButton("开始导航")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_navigation)
        buttons.addWidget(self.clear_button)
        buttons.addWidget(self.start_button)
        side_layout.addLayout(buttons)
        self.nav_status = QLabel(self.node.navigation_status)
        self.nav_status.setWordWrap(True)
        self.nav_status.setObjectName("statusBar")
        side_layout.addWidget(self.nav_status)

        splitter.addWidget(side)
        splitter.setSizes([1080, 380])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        layout.addWidget(splitter, 1)
        self.setup_page = root
        self.active_page = ActiveNavigationPage(
            self.exit_navigation, self.set_navigation_map_mode)
        self.map_tools_page = self._build_map_tools_page()
        self.sensor_tools_page = self._build_sensor_tools_page()
        self.pages = QStackedWidget()
        self.pages.addWidget(self.setup_page)
        self.pages.addWidget(self.map_tools_page)
        self.pages.addWidget(self.sensor_tools_page)
        self.pages.addWidget(self.active_page)
        self.setCentralWidget(self.pages)
        self.setStyleSheet(self._style_sheet())

    def _build_sensor_tools_page(self) -> QWidget:
        """Build the local driver launcher and its live log console."""
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(14)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("GS SENSOR MANAGER")
        title.setObjectName("title")
        subtitle = QLabel("雷达与相机驱动 · 启动参数 · 实时进程日志")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        back_button = QPushButton("返回导航")
        back_button.setObjectName("workspaceButton")
        back_button.clicked.connect(self.show_navigation_setup)
        header.addWidget(back_button)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        controls = QFrame()
        controls.setObjectName("sidePanel")
        controls.setMinimumWidth(440)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(12)
        control_title = QLabel("驱动控制")
        control_title.setObjectName("sectionTitle")
        controls_layout.addWidget(control_title)
        self.sensor_control_tabs = QTabWidget()
        self.sensor_control_tabs.addTab(self._build_lidar_controls(), "雷达")
        self.sensor_control_tabs.addTab(self._build_camera_controls(), "相机")
        controls_layout.addWidget(self.sensor_control_tabs, 1)
        splitter.addWidget(controls)

        log_card = QFrame()
        log_card.setObjectName("sidePanel")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_header = QHBoxLayout()
        log_title = QLabel("启动日志")
        log_title.setObjectName("sectionTitle")
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        clear_button = QPushButton("清空日志")
        clear_button.setObjectName("secondaryButton")
        clear_button.clicked.connect(self._clear_sensor_logs)
        log_header.addWidget(clear_button)
        log_layout.addLayout(log_header)
        self.sensor_log_tabs = QTabWidget()
        for key, label in (("all", "全部"), ("lidar", "雷达"), ("camera", "相机")):
            view = QPlainTextEdit()
            view.setObjectName("sensorLog")
            view.setReadOnly(True)
            view.setLineWrapMode(QPlainTextEdit.NoWrap)
            view.document().setMaximumBlockCount(3000)
            self.sensor_log_views[key] = view
            self.sensor_log_tabs.addTab(view, label)
        log_layout.addWidget(self.sensor_log_tabs, 1)
        splitter.addWidget(log_card)
        splitter.setSizes([500, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        for key, display_name in (("lidar", "雷达"), ("camera", "相机")):
            controller = SensorLaunchProcess(display_name, self)
            controller.log_received.connect(
                lambda text, sensor=key: self._append_sensor_log(sensor, text))
            controller.state_changed.connect(
                lambda state, sensor=key: self._handle_sensor_state(sensor, state))
            self.sensor_processes[key] = controller
            self._handle_sensor_state(key, "stopped")
        return root

    def _sensor_action_row(self, key: str, start_text: str) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        status = QLabel("未启动")
        status.setObjectName("sensorStatus")
        start = QPushButton(start_text)
        start.setObjectName("primaryButton")
        stop = QPushButton("停止")
        stop.setObjectName("dangerButton")
        start.clicked.connect(lambda: self._start_sensor(key))
        stop.clicked.connect(lambda: self._stop_sensor(key))
        layout.addWidget(status)
        layout.addStretch(1)
        layout.addWidget(stop)
        layout.addWidget(start)
        self.sensor_state_labels[key] = status
        self.sensor_start_buttons[key] = start
        self.sensor_stop_buttons[key] = stop
        return widget

    def _build_lidar_controls(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lidar_xfer_format = QComboBox()
        self.lidar_xfer_format.addItem("Livox CustomMsg", 4)
        self.lidar_xfer_format.addItem("PointCloud2", 0)
        form.addRow("输出格式", self.lidar_xfer_format)
        self.lidar_publish_frequency = QDoubleSpinBox()
        self.lidar_publish_frequency.setRange(1.0, 100.0)
        self.lidar_publish_frequency.setDecimals(1)
        self.lidar_publish_frequency.setValue(10.0)
        self.lidar_publish_frequency.setSuffix(" Hz")
        form.addRow("发布频率", self.lidar_publish_frequency)
        self.lidar_frame_id = QLineEdit("livox_frame")
        form.addRow("坐标系 frame_id", self.lidar_frame_id)
        self.lidar_multi_topic = QCheckBox("每台雷达使用独立话题")
        form.addRow("多话题", self.lidar_multi_topic)
        config_row = QWidget()
        config_layout = QHBoxLayout(config_row)
        config_layout.setContentsMargins(0, 0, 0, 0)
        self.lidar_config_path = QLineEdit()
        self.lidar_config_path.setPlaceholderText("留空使用 MID360_config.json")
        browse = QPushButton("选择")
        browse.setObjectName("secondaryButton")
        browse.clicked.connect(self._choose_lidar_config)
        config_layout.addWidget(self.lidar_config_path, 1)
        config_layout.addWidget(browse)
        form.addRow("配置文件", config_row)
        layout.addLayout(form)
        hint = QLabel(
            "对应 livox_ros_driver2/msg_MID360_launch.py。参数会作为 ROS 2 "
            "launch 参数传入，不会修改驱动配置文件。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        layout.addWidget(self._sensor_action_row("lidar", "雷达启动"))
        return root

    def _build_camera_controls(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.camera_name_input = QLineEdit("camera")
        self.camera_namespace_input = QLineEdit("camera")
        self.camera_serial_input = QLineEdit()
        self.camera_serial_input.setPlaceholderText("留空自动选择设备")
        form.addRow("相机名称", self.camera_name_input)
        form.addRow("命名空间", self.camera_namespace_input)
        form.addRow("设备序列号", self.camera_serial_input)
        self.camera_enable_color = QCheckBox("彩色图像")
        self.camera_enable_depth = QCheckBox("深度图像")
        self.camera_enable_gyro = QCheckBox("陀螺仪")
        self.camera_enable_accel = QCheckBox("加速度计")
        for checkbox in (
            self.camera_enable_color, self.camera_enable_depth,
            self.camera_enable_gyro, self.camera_enable_accel,
        ):
            checkbox.setChecked(True)
        stream_row = QWidget()
        stream_layout = QVBoxLayout(stream_row)
        stream_layout.setContentsMargins(0, 0, 0, 0)
        stream_layout.setSpacing(4)
        first_stream_row = QHBoxLayout()
        first_stream_row.addWidget(self.camera_enable_color)
        first_stream_row.addWidget(self.camera_enable_depth)
        first_stream_row.addStretch(1)
        second_stream_row = QHBoxLayout()
        second_stream_row.addWidget(self.camera_enable_gyro)
        second_stream_row.addWidget(self.camera_enable_accel)
        second_stream_row.addStretch(1)
        stream_layout.addLayout(first_stream_row)
        stream_layout.addLayout(second_stream_row)
        form.addRow("数据流", stream_row)
        self.camera_enable_sync = QCheckBox("同步彩色与深度")
        self.camera_enable_sync.setChecked(True)
        self.camera_align_depth = QCheckBox("深度对齐到彩色图")
        self.camera_align_depth.setChecked(True)
        self.camera_pointcloud = QCheckBox("由相机生成点云")
        form.addRow("同步", self.camera_enable_sync)
        form.addRow("深度对齐", self.camera_align_depth)
        form.addRow("点云", self.camera_pointcloud)
        self.camera_unite_imu = QComboBox()
        self.camera_unite_imu.addItem("不合并 (0)", "0")
        self.camera_unite_imu.addItem("复制插值 (1)", "1")
        self.camera_unite_imu.addItem("线性插值 (2)", "2")
        self.camera_unite_imu.setCurrentIndex(2)
        form.addRow("IMU 合并方式", self.camera_unite_imu)
        layout.addLayout(form)
        hint = QLabel(
            "对应 realsense2_camera/d435i.launch.py。默认开启 RGB、深度与 IMU，"
            "发布的彩色图可供导航界面订阅。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        layout.addWidget(self._sensor_action_row("camera", "相机启动"))
        return root

    @staticmethod
    def _launch_bool(value: bool) -> str:
        return "true" if value else "false"

    def _lidar_launch_arguments(self):
        arguments = [
            "launch", "livox_ros_driver2", "msg_MID360_launch.py",
            f"xfer_format:={self.lidar_xfer_format.currentData()}",
            f"multi_topic:={int(self.lidar_multi_topic.isChecked())}",
            f"publish_freq:={self.lidar_publish_frequency.value():g}",
            f"frame_id:={self.lidar_frame_id.text().strip() or 'livox_frame'}",
        ]
        config_path = self.lidar_config_path.text().strip()
        if config_path:
            arguments.append(f"user_config_path:={config_path}")
        return arguments

    def _camera_launch_arguments(self):
        arguments = [
            "launch", "realsense2_camera", "d435i.launch.py",
            f"camera_name:={self.camera_name_input.text().strip() or 'camera'}",
            f"camera_namespace:={self.camera_namespace_input.text().strip() or 'camera'}",
            f"enable_color:={self._launch_bool(self.camera_enable_color.isChecked())}",
            f"enable_depth:={self._launch_bool(self.camera_enable_depth.isChecked())}",
            f"enable_gyro:={self._launch_bool(self.camera_enable_gyro.isChecked())}",
            f"enable_accel:={self._launch_bool(self.camera_enable_accel.isChecked())}",
            f"unite_imu_method:={self.camera_unite_imu.currentData()}",
            f"enable_sync:={self._launch_bool(self.camera_enable_sync.isChecked())}",
            f"align_depth.enable:={self._launch_bool(self.camera_align_depth.isChecked())}",
            f"pointcloud.enable:={self._launch_bool(self.camera_pointcloud.isChecked())}",
        ]
        serial = self.camera_serial_input.text().strip()
        if serial:
            arguments.append(f"serial_no:={serial}")
        return arguments

    def _choose_lidar_config(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "选择 Livox 配置", self.lidar_config_path.text(),
            "JSON 配置 (*.json);;所有文件 (*)")
        if path:
            self.lidar_config_path.setText(path)

    def _start_sensor(self, key: str) -> None:
        builder = (
            self._lidar_launch_arguments
            if key == "lidar" else self._camera_launch_arguments)
        self.sensor_processes[key].start_launch(builder())

    def _stop_sensor(self, key: str) -> None:
        self.sensor_processes[key].stop_launch()

    def _handle_sensor_state(self, key: str, state: str) -> None:
        labels = {
            "starting": ("启动中…", "#ffd166"),
            "running": ("运行中", "#66e09a"),
            "stopping": ("停止中…", "#ffd166"),
            "stopped": ("未启动", "#91a2ad"),
            "error": ("启动失败", "#ff7f88"),
        }
        text, color = labels.get(state, (state, "#91a2ad"))
        label = self.sensor_state_labels.get(key)
        if label is not None:
            label.setText(f"● {text}")
            label.setStyleSheet(f"color: {color}; font-weight: 700;")
        running = state in ("starting", "running", "stopping")
        if key in self.sensor_start_buttons:
            self.sensor_start_buttons[key].setEnabled(not running)
        if key in self.sensor_stop_buttons:
            self.sensor_stop_buttons[key].setEnabled(
                state in ("starting", "running"))

    def _append_sensor_log(self, key: str, text: str) -> None:
        if not text:
            return
        clean = ANSI_ESCAPE_RE.sub("", text).replace("\r", "").rstrip("\n")
        if not clean:
            return
        self.sensor_log_views[key].appendPlainText(clean)
        prefix = "雷达" if key == "lidar" else "相机"
        self.sensor_log_views["all"].appendPlainText(
            "\n".join(f"[{prefix}] {line}" for line in clean.splitlines()))

    def _clear_sensor_logs(self) -> None:
        for view in self.sensor_log_views.values():
            view.clear()

    def _build_map_tools_page(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(14)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("GS MAP STUDIO")
        title.setObjectName("title")
        subtitle = QLabel("本地点云与栅格地图处理 · 不启动 Web 服务")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        back_button = QPushButton("返回导航")
        back_button.setObjectName("workspaceButton")
        back_button.clicked.connect(self.show_navigation_setup)
        header.addWidget(back_button)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        self.editor_map_panel = MapPanel(cloud_3d=True)
        self.editor_map_panel.setObjectName("mapEditorPanel")
        self.editor_map_panel.edit_started.connect(self.snapshot_map_edit)
        self.editor_map_panel.edit_requested.connect(self.edit_map_at)
        splitter.addWidget(self.editor_map_panel)

        controls = QFrame()
        controls.setObjectName("sidePanel")
        controls.setMinimumWidth(360)
        controls.setMaximumWidth(440)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setSpacing(13)

        mode_header = QHBoxLayout()
        mode_title = QLabel("地图类型")
        mode_title.setObjectName("sectionTitle")
        mode_header.addWidget(mode_title)
        mode_header.addStretch(1)
        self.editor_grid_mode_button = QPushButton("栅格地图")
        self.editor_grid_mode_button.setObjectName("mapModeButton")
        self.editor_grid_mode_button.setCheckable(True)
        self.editor_grid_mode_button.setChecked(True)
        self.editor_cloud_mode_button = QPushButton("点云地图")
        self.editor_cloud_mode_button.setObjectName("mapModeButton")
        self.editor_cloud_mode_button.setCheckable(True)
        self.editor_mode_group = QButtonGroup(self)
        self.editor_mode_group.setExclusive(True)
        self.editor_mode_group.addButton(self.editor_grid_mode_button)
        self.editor_mode_group.addButton(self.editor_cloud_mode_button)
        self.editor_grid_mode_button.clicked.connect(
            lambda: self.set_editor_map_mode("grid"))
        self.editor_cloud_mode_button.clicked.connect(
            lambda: self.set_editor_map_mode("cloud"))
        mode_header.addWidget(self.editor_grid_mode_button)
        mode_header.addWidget(self.editor_cloud_mode_button)
        controls_layout.addLayout(mode_header)
        self.reset_cloud_view_button = QPushButton("重置三维视角")
        self.reset_cloud_view_button.setObjectName("secondaryButton")
        self.reset_cloud_view_button.clicked.connect(
            self.editor_map_panel.reset_cloud_view)
        controls_layout.addWidget(self.reset_cloud_view_button)
        cloud_view_hint = QLabel(
            "点云模式：左键旋转，右键/中键平移，滚轮缩放，双击复位")
        cloud_view_hint.setObjectName("hint")
        cloud_view_hint.setWordWrap(True)
        controls_layout.addWidget(cloud_view_hint)

        display_title = QLabel("点云显示（不修改原始数据）")
        display_title.setObjectName("toolGroupTitle")
        controls_layout.addWidget(display_title)
        alignment_row = QHBoxLayout()
        alignment_row.addWidget(QLabel("姿态"))
        self.cloud_alignment_combo = QComboBox()
        self.cloud_alignment_combo.addItem("自动找平地面", "auto")
        self.cloud_alignment_combo.addItem("原始坐标", "original")
        alignment_row.addWidget(self.cloud_alignment_combo, 1)
        controls_layout.addLayout(alignment_row)

        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("轴顺序"))
        self.cloud_axis_combo = QComboBox()
        for order in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
            self.cloud_axis_combo.addItem(order, order)
        axis_row.addWidget(self.cloud_axis_combo, 1)
        self.cloud_flip_x = QCheckBox("反X")
        self.cloud_flip_y = QCheckBox("反Y")
        self.cloud_flip_z = QCheckBox("反Z")
        axis_row.addWidget(self.cloud_flip_x)
        axis_row.addWidget(self.cloud_flip_y)
        axis_row.addWidget(self.cloud_flip_z)
        controls_layout.addLayout(axis_row)

        render_row = QHBoxLayout()
        render_row.addWidget(QLabel("投影"))
        self.cloud_projection_combo = QComboBox()
        self.cloud_projection_combo.addItem("透视", "perspective")
        self.cloud_projection_combo.addItem("正交", "orthographic")
        render_row.addWidget(self.cloud_projection_combo, 1)
        render_row.addWidget(QLabel("颜色"))
        self.cloud_color_combo = QComboBox()
        self.cloud_color_combo.addItem("原始/GS颜色", "rgb")
        self.cloud_color_combo.addItem("高度", "height")
        render_row.addWidget(self.cloud_color_combo, 1)
        controls_layout.addLayout(render_row)

        self.cloud_alignment_combo.currentIndexChanged.connect(
            self.update_cloud_display_options)
        self.cloud_axis_combo.currentIndexChanged.connect(
            self.update_cloud_display_options)
        self.cloud_projection_combo.currentIndexChanged.connect(
            self.update_cloud_display_options)
        self.cloud_color_combo.currentIndexChanged.connect(
            self.update_cloud_display_options)
        self.cloud_flip_x.toggled.connect(self.update_cloud_display_options)
        self.cloud_flip_y.toggled.connect(self.update_cloud_display_options)
        self.cloud_flip_z.toggled.connect(self.update_cloud_display_options)

        source_title = QLabel("文件")
        source_title.setObjectName("toolGroupTitle")
        controls_layout.addWidget(source_title)
        file_buttons = QHBoxLayout()
        self.load_grid_button = QPushButton("加载 YAML/PGM")
        self.load_grid_button.setObjectName("secondaryButton")
        self.load_grid_button.clicked.connect(self.load_grid_dialog)
        self.load_cloud_button = QPushButton("加载 PCD/PLY")
        self.load_cloud_button.setObjectName("secondaryButton")
        self.load_cloud_button.clicked.connect(self.load_pointcloud_dialog)
        file_buttons.addWidget(self.load_grid_button)
        file_buttons.addWidget(self.load_cloud_button)
        controls_layout.addLayout(file_buttons)
        self.save_map_button = QPushButton("保存当前地图")
        self.save_map_button.setObjectName("primaryButton")
        self.save_map_button.clicked.connect(self.save_current_map)
        controls_layout.addWidget(self.save_map_button)

        convert_title = QLabel("点云转栅格")
        convert_title.setObjectName("toolGroupTitle")
        controls_layout.addWidget(convert_title)
        convert_frame_row = QHBoxLayout()
        convert_frame_row.addWidget(QLabel("转换坐标"))
        self.convert_coordinates_combo = QComboBox()
        self.convert_coordinates_combo.addItem(
            "自动找平/当前显示坐标", "display")
        self.convert_coordinates_combo.addItem("原始 XYZ", "original")
        convert_frame_row.addWidget(self.convert_coordinates_combo, 1)
        controls_layout.addLayout(convert_frame_row)
        resolution_row = QHBoxLayout()
        resolution_row.addWidget(QLabel("栅格分辨率"))
        self.grid_resolution_spin = QDoubleSpinBox()
        self.grid_resolution_spin.setRange(0.01, 1.0)
        self.grid_resolution_spin.setSingleStep(0.01)
        self.grid_resolution_spin.setDecimals(2)
        self.grid_resolution_spin.setValue(0.05)
        self.grid_resolution_spin.setSuffix(" m")
        resolution_row.addWidget(self.grid_resolution_spin)
        controls_layout.addLayout(resolution_row)
        height_row = QHBoxLayout()
        height_row.addWidget(QLabel("保留高度 Z"))
        self.z_min_spin = QDoubleSpinBox()
        self.z_min_spin.setRange(-1000.0, 1000.0)
        self.z_min_spin.setValue(-0.20)
        self.z_min_spin.setDecimals(2)
        self.z_max_spin = QDoubleSpinBox()
        self.z_max_spin.setRange(-1000.0, 1000.0)
        self.z_max_spin.setValue(1.50)
        self.z_max_spin.setDecimals(2)
        height_row.addWidget(self.z_min_spin)
        height_row.addWidget(QLabel("至"))
        height_row.addWidget(self.z_max_spin)
        controls_layout.addLayout(height_row)
        self.convert_button = QPushButton("转换为栅格地图")
        self.convert_button.setObjectName("primaryButton")
        self.convert_button.clicked.connect(self.convert_pointcloud)
        controls_layout.addWidget(self.convert_button)

        edit_title = QLabel("地图编辑")
        edit_title.setObjectName("toolGroupTitle")
        controls_layout.addWidget(edit_title)
        self.edit_tool_combo = QComboBox()
        self.edit_tool_combo.addItem("浏览（不修改）", "navigate")
        self.edit_tool_combo.addItem("栅格：绘制障碍物", "occupied")
        self.edit_tool_combo.addItem("栅格：绘制自由区域", "free")
        self.edit_tool_combo.addItem("栅格：绘制未知区域", "unknown")
        self.edit_tool_combo.addItem("点云：添加障碍点", "cloud_add")
        self.edit_tool_combo.addItem("点云：擦除点", "cloud_erase")
        self.edit_tool_combo.currentIndexChanged.connect(self.update_edit_tool)
        controls_layout.addWidget(self.edit_tool_combo)
        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("画笔半径"))
        self.brush_radius_spin = QDoubleSpinBox()
        self.brush_radius_spin.setRange(0.02, 5.0)
        self.brush_radius_spin.setSingleStep(0.05)
        self.brush_radius_spin.setValue(0.20)
        self.brush_radius_spin.setSuffix(" m")
        self.brush_radius_spin.valueChanged.connect(self.update_edit_tool)
        brush_row.addWidget(self.brush_radius_spin)
        controls_layout.addLayout(brush_row)
        edit_buttons = QHBoxLayout()
        self.undo_button = QPushButton("撤销上一笔")
        self.undo_button.setObjectName("secondaryButton")
        self.undo_button.clicked.connect(self.undo_map_edit)
        self.reset_map_button = QPushButton("恢复加载版本")
        self.reset_map_button.setObjectName("secondaryButton")
        self.reset_map_button.clicked.connect(self.reset_map_edits)
        edit_buttons.addWidget(self.undo_button)
        edit_buttons.addWidget(self.reset_map_button)
        controls_layout.addLayout(edit_buttons)

        self.map_status = QLabel(
            "地图处理与导航任务相互独立。编辑保存后返回导航使用。")
        self.map_status.setObjectName("mapStatus")
        self.map_status.setWordWrap(True)
        controls_layout.addWidget(self.map_status)
        controls_layout.addStretch(1)
        self.update_cloud_display_options(show_status=False)

        splitter.addWidget(controls)
        splitter.setSizes([1050, 400])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        layout.addWidget(splitter, 1)
        return root

    @staticmethod
    def _style_sheet() -> str:
        return """
        QMainWindow, QWidget { background: #0b1016; color: #eaf0f4; }
        QLabel#title { font-size: 25px; font-weight: 700; letter-spacing: 2px; }
        QLabel#subtitle { color: #728493; font-size: 12px; }
        QLabel#onlineChip {
            color: #66e09a; background: #10291d; border: 1px solid #245d3c;
            border-radius: 12px; padding: 7px 14px; font-weight: 700;
        }
        QWidget#card, QFrame#sidePanel {
            background: #101720; border: 1px solid #26323d; border-radius: 14px;
        }
        QWidget#mapPanel, QWidget#mapEditorPanel {
            border: 1px solid #293946; border-radius: 9px;
        }
        QWidget#mapEditorPanel { background: #0d141b; }
        QWidget#activeMap {
            background: #111820; border: 2px solid rgba(255, 255, 255, 75);
            border-radius: 12px;
        }
        QLabel#sectionTitle { font-size: 18px; font-weight: 650; }
        QLabel#toolGroupTitle {
            color: #7fcce6; font-size: 12px; font-weight: 700;
            padding-top: 5px;
        }
        QLabel#hint { color: #8ca0ae; font-size: 12px; }
        QLabel#coordinate {
            color: #d6e0e6; background: #151f29; border-radius: 7px; padding: 8px;
        }
        QLabel#statusBar {
            color: #8ed9f4; background: #0d1d27; border-radius: 7px; padding: 10px;
        }
        QLabel#mapStatus {
            color: #9fb2bf; background: #111b24; border-radius: 6px; padding: 6px;
            font-size: 11px;
        }
        QPushButton {
            min-height: 38px; border-radius: 7px; padding: 0 15px;
            font-size: 13px; font-weight: 650;
        }
        QPushButton#primaryButton { background: #00a9df; color: #041018; border: none; }
        QPushButton#primaryButton:hover { background: #25c2f2; }
        QPushButton#primaryButton:disabled { background: #28404b; color: #71838c; }
        QPushButton#secondaryButton {
            background: #18232d; border: 1px solid #354653; color: #dbe4e9;
        }
        QPushButton#workspaceButton {
            background: #17232d; border: 1px solid #3b5666; color: #bcecff;
        }
        QPushButton#workspaceButton:hover { background: #203441; }
        QPushButton#mapModeButton {
            min-height: 24px; max-height: 28px; min-width: 48px; padding: 0 8px;
            background: rgba(12, 20, 27, 220); border: 1px solid #40525e;
            color: #aebdc6; font-size: 11px;
        }
        QPushButton#mapModeButton:checked {
            background: #00a9df; border-color: #32c5ef; color: #031016;
        }
        QComboBox, QDoubleSpinBox, QLineEdit {
            min-height: 30px; background: #17222c; border: 1px solid #344652;
            border-radius: 5px; padding: 0 6px; color: #dbe4e9;
        }
        QGroupBox {
            border: 1px solid #2c3b46; border-radius: 8px;
            margin-top: 9px; padding-top: 8px; font-weight: 650;
        }
        QTabWidget::pane {
            border: 1px solid #2c3b46; border-radius: 7px; background: #0e161e;
        }
        QTabBar::tab {
            min-width: 82px; min-height: 30px; padding: 2px 12px;
            background: #141f28; color: #9eb0bb;
            border: 1px solid #2c3b46;
        }
        QTabBar::tab:selected { background: #174a60; color: #ffffff; }
        QPlainTextEdit#sensorLog {
            background: #080d12; color: #b9d9c5; border: none;
            font-family: Monospace; font-size: 12px; padding: 8px;
            selection-background-color: #245b70;
        }
        QListWidget#waypointList {
            background: #111b24; border: 1px solid #2d3e4a;
            border-radius: 7px; padding: 5px; color: #dbe4e9;
            outline: none;
        }
        QListWidget#waypointList::item {
            min-height: 34px; border-radius: 5px; padding: 2px 8px;
        }
        QListWidget#waypointList::item:selected {
            background: #174a60; color: #ffffff;
        }
        QCheckBox { color: #dbe4e9; spacing: 4px; }
        QCheckBox::indicator {
            width: 15px; height: 15px; border: 1px solid #48606e;
            border-radius: 3px; background: #111a22;
        }
        QCheckBox::indicator:checked { background: #00a9df; }
        QPushButton#dangerButton {
            background: transparent; border: 1px solid #6d3540; color: #ff8d96;
        }
        QPushButton#exitNavigationButton {
            background: rgba(8, 13, 19, 225); border: 1px solid #ff626c;
            color: #ff8a92; font-size: 15px; font-weight: 700;
        }
        QPushButton#exitNavigationButton:hover { background: #5b252d; color: white; }
        QSplitter::handle { background: transparent; width: 12px; }
        """

    def _load_configured_pointcloud(self) -> None:
        if not hasattr(self.node, "get_parameter"):
            return
        configured = str(self.node.get_parameter("pointcloud_map_path").value)
        if not configured:
            return
        path = FilePath(configured).expanduser()
        if not path.exists():
            self.map_status.setText(f"配置的点云文件不存在：{path}")
            return
        try:
            self._set_local_cloud(load_pointcloud(path), reset_original=True)
            self.map_status.setText(
                f"已自动加载点云：{path.name} · {len(self.local_cloud.points):,} 点 · "
                f"{self.editor_map_panel.cloud_display_description()}")
        except (OSError, ValueError) as exc:
            self.map_status.setText(f"自动加载点云失败：{exc}")

    def show_map_tools(self) -> None:
        """Open the standalone map processing workspace."""
        self.pages.setCurrentWidget(self.map_tools_page)

    def show_sensor_tools(self) -> None:
        """Open the standalone sensor driver workspace."""
        self.pages.setCurrentWidget(self.sensor_tools_page)

    def show_navigation_setup(self) -> None:
        """Return to navigation without starting or cancelling a task."""
        self.pages.setCurrentWidget(self.setup_page)

    def set_navigation_map_mode(self, mode: str) -> None:
        """Set map type for navigation setup and the active mini-map only."""
        self.map_panel.set_display_mode(mode)
        self.active_page.set_map_mode(mode, notify=False)
        self.grid_mode_button.setChecked(mode == "grid")
        self.cloud_mode_button.setChecked(mode == "cloud")

    def set_editor_map_mode(self, mode: str) -> None:
        """Set map type in Map Studio without changing navigation views."""
        self.editor_map_panel.set_display_mode(mode)
        self.editor_grid_mode_button.setChecked(mode == "grid")
        self.editor_cloud_mode_button.setChecked(mode == "cloud")
        self.update_edit_tool()

    def set_map_mode(self, mode: str) -> None:
        """Backward-compatible alias for the navigation map selector."""
        self.set_navigation_map_mode(mode)

    def update_cloud_display_options(
        self, *_args, show_status: bool = True,
    ) -> None:
        display_options = dict(
            alignment=str(self.cloud_alignment_combo.currentData()),
            axis_order=str(self.cloud_axis_combo.currentData()),
            axis_flips=(
                self.cloud_flip_x.isChecked(),
                self.cloud_flip_y.isChecked(),
                self.cloud_flip_z.isChecked(),
            ),
            projection=str(self.cloud_projection_combo.currentData()),
            color_mode=str(self.cloud_color_combo.currentData()),
        )
        self.editor_map_panel.set_cloud_display_options(**display_options)
        self.map_panel.set_cloud_display_options(**display_options)
        self.active_page.map_panel.set_cloud_display_options(**display_options)
        display_transform_active = (
            self.cloud_alignment_combo.currentData() == "auto"
            or self.cloud_axis_combo.currentData() != "XYZ"
            or self.cloud_flip_x.isChecked()
            or self.cloud_flip_y.isChecked()
            or self.cloud_flip_z.isChecked()
        )
        if display_transform_active:
            # Keep grid generation in the same coordinate system the user is
            # inspecting.  The original cloud remains untouched and can still
            # be selected after returning to an untransformed display.
            display_index = self.convert_coordinates_combo.findData("display")
            self.convert_coordinates_combo.setCurrentIndex(display_index)
        if show_status and hasattr(self, "map_status"):
            suffix = ""
            if (
                hasattr(self, "edit_tool_combo")
                and self.edit_tool_combo.currentData()
                in ("cloud_add", "cloud_erase")
                and (
                    self.cloud_alignment_combo.currentData() != "original"
                    or self.cloud_axis_combo.currentData() != "XYZ"
                    or self.cloud_flip_x.isChecked()
                    or self.cloud_flip_y.isChecked()
                    or self.cloud_flip_z.isChecked()
                )
            ):
                self.edit_tool_combo.setCurrentIndex(0)
                suffix += "；已退出点云画笔，防止在显示变换下误编辑"
            if (
                self.cloud_color_combo.currentData() == "rgb"
                and not self.editor_map_panel.cloud_has_rgb
            ):
                suffix += "；文件没有 RGB 字段，已回退到高度着色"
            self.map_status.setText(
                "点云显示已更新（原始点云未修改）· "
                + self.editor_map_panel.cloud_display_description() + suffix)

    def update_edit_tool(self, *_args) -> None:
        tool = str(self.edit_tool_combo.currentData())
        radius = float(self.brush_radius_spin.value())
        if (
            tool in ("cloud_add", "cloud_erase")
            and self.local_cloud is not None
            and self.local_cloud.is_gaussian_splat
        ):
            self.edit_tool_combo.blockSignals(True)
            self.edit_tool_combo.setCurrentIndex(0)
            self.edit_tool_combo.blockSignals(False)
            tool = "navigate"
            self.map_status.setText(
                "Gaussian Splat 包含尺度/旋转等属性，当前版本仅支持浏览和转栅格，"
                "不允许按普通 XYZ 点云画笔修改")
        if tool in ("cloud_add", "cloud_erase") and (
            self.cloud_alignment_combo.currentData() != "original"
            or self.cloud_axis_combo.currentData() != "XYZ"
            or self.cloud_flip_x.isChecked()
            or self.cloud_flip_y.isChecked()
            or self.cloud_flip_z.isChecked()
        ):
            widgets = (
                self.cloud_alignment_combo, self.cloud_axis_combo,
                self.cloud_flip_x, self.cloud_flip_y, self.cloud_flip_z,
            )
            for widget in widgets:
                widget.blockSignals(True)
            self.cloud_alignment_combo.setCurrentIndex(
                self.cloud_alignment_combo.findData("original"))
            self.cloud_axis_combo.setCurrentIndex(
                self.cloud_axis_combo.findData("XYZ"))
            self.cloud_flip_x.setChecked(False)
            self.cloud_flip_y.setChecked(False)
            self.cloud_flip_z.setChecked(False)
            for widget in widgets:
                widget.blockSignals(False)
            self.update_cloud_display_options(show_status=False)
            self.map_status.setText(
                "点云画笔按原始 XY 坐标编辑，已临时切回原始坐标显示")
        self.editor_map_panel.set_edit_tool(tool, radius)

    def _apply_gaussian_display_preset(self) -> None:
        widgets = (
            self.cloud_alignment_combo, self.cloud_axis_combo,
            self.cloud_flip_x, self.cloud_flip_y, self.cloud_flip_z,
            self.cloud_projection_combo, self.cloud_color_combo,
            self.convert_coordinates_combo,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self.cloud_alignment_combo.setCurrentIndex(
            self.cloud_alignment_combo.findData("original"))
        # SuperSplat uses Y-up. Convert that to this renderer's Z-up after
        # applying the object's Z rotation = 180 degrees from the reference.
        self.cloud_axis_combo.setCurrentIndex(
            self.cloud_axis_combo.findData("XZY"))
        self.cloud_flip_x.setChecked(True)
        self.cloud_flip_y.setChecked(True)
        self.cloud_flip_z.setChecked(True)
        self.cloud_projection_combo.setCurrentIndex(
            self.cloud_projection_combo.findData("perspective"))
        self.cloud_color_combo.setCurrentIndex(
            self.cloud_color_combo.findData("rgb"))
        self.convert_coordinates_combo.setCurrentIndex(
            self.convert_coordinates_combo.findData("display"))
        for widget in widgets:
            widget.blockSignals(False)
        self.update_cloud_display_options(show_status=False)

    @staticmethod
    def _set_panel_cloud(panel: MapPanel, cloud: PointCloudMap) -> None:
        panel.set_pointcloud(
            cloud.points,
            cloud.colors,
            cloud.splat_scales,
            cloud.splat_rotations,
            cloud.splat_opacities,
        )

    def _set_local_cloud(
        self, cloud: PointCloudMap, reset_original: bool = False,
    ) -> None:
        self.local_cloud = cloud
        if reset_original:
            self._cloud_original = cloud.points.copy()
            self._cloud_original_colors = (
                None if cloud.colors is None else cloud.colors.copy())
            self._undo_stack.clear()
            if cloud.is_gaussian_splat:
                self._apply_gaussian_display_preset()
        self._cloud_edit_revision += 1
        self._rendered_cloud_edit_revision = -1
        for panel in (
            self.map_panel, self.editor_map_panel, self.active_page.map_panel,
        ):
            self._set_panel_cloud(panel, cloud)
        if reset_original:
            self.editor_map_panel.reset_cloud_view()

    def _set_local_grid(
        self, grid: GridMap, reset_original: bool = False,
    ) -> None:
        self.local_grid = grid
        if reset_original:
            self._grid_original = grid.occupancy.copy()
            self._undo_stack.clear()
        self._grid_edit_revision += 1
        self._rendered_grid_edit_revision = -1
        self._render_grid(grid)

    def _render_grid(self, grid: GridMap) -> None:
        for panel in (
            self.map_panel, self.editor_map_panel, self.active_page.map_panel,
        ):
            panel.set_map(
                grid.occupancy, grid.origin[:2], grid.resolution,
                float(grid.origin[2]))

    def load_pointcloud_dialog(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "加载点云地图", "",
            "Point Cloud (*.pcd *.ply);;PCD (*.pcd);;PLY (*.ply);;All Files (*)")
        if not path:
            return
        try:
            cloud = load_pointcloud(path)
            self._set_local_cloud(cloud, reset_original=True)
            self.set_editor_map_mode("cloud")
            self.map_status.setText(
                f"点云已加载：{FilePath(path).name} · {len(cloud.points):,} 点 · "
                f"{self.editor_map_panel.cloud_display_description()}")
        except (OSError, ValueError) as exc:
            self.map_status.setText(f"点云加载失败：{exc}")

    def load_grid_dialog(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "加载栅格地图", "",
            "Nav2 Map (*.yaml *.yml);;Portable Graymap (*.pgm)")
        if not path:
            return
        try:
            grid = load_grid_map(path, self.grid_resolution_spin.value())
            self._set_local_grid(grid, reset_original=True)
            self.node.grid_frame = grid.frame_id
            self.set_editor_map_mode("grid")
            h, w = grid.occupancy.shape
            self.map_status.setText(
                f"栅格已加载：{FilePath(path).name} · {w}×{h} · {grid.resolution:.3f} m")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self.map_status.setText(f"栅格加载失败：{exc}")

    def convert_pointcloud(self) -> None:
        cloud = self.local_cloud
        if cloud is None and getattr(self.node, "latest_pointcloud", None) is not None:
            cloud = PointCloudMap(
                self.node.latest_pointcloud.copy(),
                frame_id=self.node.pointcloud_frame)
        if cloud is None or not len(cloud.points):
            self.map_status.setText("请先加载 PCD/PLY 或配置 pointcloud_topic")
            return
        try:
            display_transform_active = (
                self.cloud_alignment_combo.currentData() == "auto"
                or self.cloud_axis_combo.currentData() != "XYZ"
                or self.cloud_flip_x.isChecked()
                or self.cloud_flip_y.isChecked()
                or self.cloud_flip_z.isChecked()
            )
            if display_transform_active:
                self.convert_coordinates_combo.setCurrentIndex(
                    self.convert_coordinates_combo.findData("display"))
            use_display_coordinates = display_transform_active or (
                self.convert_coordinates_combo.currentData() == "display")
            conversion_points = (
                self.editor_map_panel.transform_cloud_points(cloud.points)
                if use_display_coordinates else cloud.points)
            grid = pointcloud_to_grid(
                conversion_points,
                resolution=float(self.grid_resolution_spin.value()),
                z_min=float(self.z_min_spin.value()),
                z_max=float(self.z_max_spin.value()),
            )
            grid.frame_id = cloud.frame_id
            self._set_local_grid(grid, reset_original=True)
            self.node.grid_frame = grid.frame_id
            self.set_editor_map_mode("grid")
            h, w = grid.occupancy.shape
            coordinates = (
                "自动找平/当前显示坐标（地面 Z=0）"
                if use_display_coordinates else "原始 XYZ")
            self.map_status.setText(
                f"转换完成：{w}×{h}（使用{coordinates}），可继续编辑并保存 YAML/PGM")
        except ValueError as exc:
            self.map_status.setText(f"点云转栅格失败：{exc}")

    def save_current_map(self) -> None:
        try:
            if self.editor_map_panel.display_mode == "cloud":
                if self.local_cloud is None:
                    self.map_status.setText("当前没有可保存的点云")
                    return
                suggested_path = self.local_cloud.path or FilePath("edited_map.pcd")
                suggested = str(FilePath(suggested_path).with_suffix(".pcd"))
                path, _selected = QFileDialog.getSaveFileName(
                    self, "保存点云地图", suggested, "Point Cloud (*.pcd)")
                if not path:
                    return
                saved = save_pcd(
                    path, self.local_cloud.points, self.local_cloud.colors)
                self.local_cloud.path = saved
                self._cloud_original = self.local_cloud.points.copy()
                self._cloud_original_colors = (
                    None if self.local_cloud.colors is None
                    else self.local_cloud.colors.copy())
                self.map_status.setText(f"点云已保存：{saved}")
            else:
                grid = self._ensure_local_grid()
                if grid is None:
                    self.map_status.setText("当前没有可保存的栅格地图")
                    return
                suggested = str(grid.path or "edited_map.yaml")
                path, _selected = QFileDialog.getSaveFileName(
                    self, "保存 Nav2 栅格地图", suggested, "Nav2 Map (*.yaml)")
                if not path:
                    return
                yaml_path, pgm_path = save_grid_map(path, grid)
                grid.path = yaml_path
                self._grid_original = grid.occupancy.copy()
                self.map_status.setText(
                    f"栅格已保存：{yaml_path.name} / {pgm_path.name}")
        except (OSError, ValueError) as exc:
            self.map_status.setText(f"地图保存失败：{exc}")

    def _ensure_local_grid(self) -> Optional[GridMap]:
        if self.local_grid is not None:
            return self.local_grid
        occupancy = getattr(self.node, "occupancy", None)
        resolution = float(getattr(self.node, "map_resolution", 0.0))
        if occupancy is None or resolution <= 0.0:
            return None
        origin_xy = np.asarray(self.node.map_origin, dtype=float)
        origin = np.array([
            origin_xy[0], origin_xy[1], float(self.node.map_origin_yaw)])
        grid = GridMap(
            occupancy.copy(), resolution, origin,
            frame_id=str(getattr(self.node, "grid_frame", "map")))
        self.local_grid = grid
        self._grid_original = grid.occupancy.copy()
        return grid

    def _ensure_local_cloud(self) -> Optional[PointCloudMap]:
        if self.local_cloud is not None:
            return self.local_cloud
        points = getattr(self.node, "latest_pointcloud", None)
        if points is None:
            return None
        cloud = PointCloudMap(
            points.copy(), frame_id=str(self.node.pointcloud_frame))
        self.local_cloud = cloud
        self._cloud_original = cloud.points.copy()
        self._cloud_original_colors = None
        return cloud

    def snapshot_map_edit(self) -> None:
        mode = self.editor_map_panel.display_mode
        if mode == "grid":
            grid = self._ensure_local_grid()
            if grid is not None:
                self._undo_stack.append(("grid", grid.occupancy.copy()))
        else:
            cloud = self._ensure_local_cloud()
            if cloud is not None:
                self._undo_stack.append(("cloud", (
                    cloud.points.copy(),
                    None if cloud.colors is None else cloud.colors.copy(),
                )))
        self._undo_stack = self._undo_stack[-12:]

    def edit_map_at(
        self, mode: str, tool: str, x: float, y: float, radius: float,
    ) -> None:
        if mode == "grid":
            if tool not in ("occupied", "free", "unknown"):
                self.map_status.setText("栅格模式请选择障碍物、自由区域或未知区域画笔")
                return
            grid = self._ensure_local_grid()
            if grid is None:
                self.map_status.setText("尚无栅格地图，无法编辑")
                return
            relative = np.array([x, y]) - grid.origin[:2]
            yaw = float(grid.origin[2])
            local = np.array([
                math.cos(yaw) * relative[0] + math.sin(yaw) * relative[1],
                -math.sin(yaw) * relative[0] + math.cos(yaw) * relative[1],
            ])
            center_x, center_y = np.floor(local / grid.resolution).astype(int)
            cells = max(1, int(math.ceil(radius / grid.resolution)))
            height, width = grid.occupancy.shape
            x0, x1 = max(0, center_x - cells), min(width, center_x + cells + 1)
            y0, y1 = max(0, center_y - cells), min(height, center_y + cells + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= cells ** 2
            value = {"occupied": 100, "free": 0, "unknown": -1}[tool]
            patch = grid.occupancy[y0:y1, x0:x1]
            patch[mask] = value
            self._grid_edit_revision += 1
            self._render_grid(grid)
            self.map_status.setText("栅格地图已修改（尚未保存）")
            return

        if tool not in ("cloud_add", "cloud_erase"):
            self.map_status.setText("点云模式请选择添加点或擦除点工具")
            return
        cloud = self._ensure_local_cloud()
        if cloud is None:
            self.map_status.setText("尚无点云地图，无法编辑")
            return
        if tool == "cloud_erase":
            distance_sq = (
                (cloud.points[:, 0] - x) ** 2 + (cloud.points[:, 1] - y) ** 2)
            keep = distance_sq > radius ** 2
            cloud.points = np.ascontiguousarray(cloud.points[keep])
            if cloud.colors is not None:
                cloud.colors = np.ascontiguousarray(cloud.colors[keep])
        else:
            angles = np.linspace(0.0, 2.0 * math.pi, 18, endpoint=False)
            radial = np.array([0.0, radius * 0.5, radius])
            z_low, z_high = sorted((self.z_min_spin.value(), self.z_max_spin.value()))
            heights = np.linspace(z_low, z_high, 6)
            additions = []
            for z_value in heights:
                for ring in radial:
                    additions.extend([
                        [x + ring * math.cos(angle), y + ring * math.sin(angle), z_value]
                        for angle in angles
                    ])
            additions = np.asarray(additions, dtype=np.float32)
            cloud.points = np.vstack([cloud.points, additions])
            if cloud.colors is not None:
                added_colors = np.tile(
                    np.array([[0, 210, 255]], dtype=np.uint8),
                    (len(additions), 1))
                cloud.colors = np.vstack([cloud.colors, added_colors])
        self._cloud_edit_revision += 1
        for panel in (
            self.map_panel, self.editor_map_panel, self.active_page.map_panel,
        ):
            self._set_panel_cloud(panel, cloud)
        self.map_status.setText(
            f"点云地图已修改 · 当前 {len(cloud.points):,} 点（尚未保存）")

    def undo_map_edit(self) -> None:
        if not self._undo_stack:
            self.map_status.setText("没有可撤销的地图编辑")
            return
        mode, data = self._undo_stack.pop()
        if mode == "grid" and self.local_grid is not None:
            self.local_grid.occupancy = data
            self._grid_edit_revision += 1
            self._render_grid(self.local_grid)
            self.set_editor_map_mode("grid")
        elif mode == "cloud" and self.local_cloud is not None:
            points, colors = data
            self.local_cloud.points = points
            self.local_cloud.colors = colors
            self._cloud_edit_revision += 1
            for panel in (
                self.map_panel, self.editor_map_panel,
                self.active_page.map_panel,
            ):
                self._set_panel_cloud(panel, self.local_cloud)
            self.set_editor_map_mode("cloud")
        self.map_status.setText("已撤销上一笔地图编辑")

    def reset_map_edits(self) -> None:
        if self.editor_map_panel.display_mode == "grid":
            if self.local_grid is None or self._grid_original is None:
                self.map_status.setText("当前栅格地图没有可恢复的本地版本")
                return
            self.local_grid.occupancy = self._grid_original.copy()
            self._grid_edit_revision += 1
            self._render_grid(self.local_grid)
        else:
            if self.local_cloud is None or self._cloud_original is None:
                self.map_status.setText("当前点云地图没有可恢复的本地版本")
                return
            self.local_cloud.points = self._cloud_original.copy()
            self.local_cloud.colors = (
                None if self._cloud_original_colors is None
                else self._cloud_original_colors.copy())
            self._cloud_edit_revision += 1
            for panel in (
                self.map_panel, self.editor_map_panel,
                self.active_page.map_panel,
            ):
                self._set_panel_cloud(panel, self.local_cloud)
        self._undo_stack.clear()
        self.map_status.setText("已恢复到最近一次加载/保存的地图")

    def on_map_point(self, x: float, y: float) -> None:
        point = np.array([x, y], dtype=np.float32)
        edit_index = self.waypoint_edit_index
        if edit_index is not None and 0 <= edit_index < len(self.waypoints):
            self.waypoints[edit_index] = point
            self.waypoint_edit_index = None
            self.waypoint_list.setEnabled(True)
            self.node.navigation_status = f"已更新途径点 {edit_index + 1}"
            self.nav_status.setText(self.node.navigation_status)
            self._sync_selection_ui(edit_index)
            return
        self.waypoints.append(point)
        self.node.navigation_status = f"已添加途径点 {len(self.waypoints)}"
        self.nav_status.setText(self.node.navigation_status)
        self._sync_selection_ui(len(self.waypoints) - 1)

    def clear_selection(self) -> None:
        self.waypoints = []
        self.waypoint_edit_index = None
        self.waypoint_list.setEnabled(True)
        self._sync_selection_ui()

    def _on_waypoint_selection_changed(self, row: int) -> None:
        valid = 0 <= row < len(self.waypoints)
        editing = self.waypoint_edit_index is not None
        self.edit_waypoint_button.setEnabled(valid or editing)
        self.edit_waypoint_button.setText(
            "取消重新选点" if editing else "重新选点")
        self.delete_waypoint_button.setEnabled(valid and not editing)
        self.start_button.setEnabled(bool(self.waypoints) and not editing)
        highlighted = self.waypoint_edit_index if editing else (row if valid else None)
        self.map_panel.set_waypoint_selection(highlighted)

    def begin_waypoint_reselection(self, *_args) -> None:
        if self.waypoint_edit_index is not None:
            previous = self.waypoint_edit_index
            self.waypoint_edit_index = None
            self.waypoint_list.setEnabled(True)
            self.node.navigation_status = "已取消重新选点"
            self.nav_status.setText(self.node.navigation_status)
            self._sync_selection_ui(previous)
            return
        row = self.waypoint_list.currentRow()
        if not 0 <= row < len(self.waypoints):
            self.node.navigation_status = "请先在列表中选择一个途径点"
            self.nav_status.setText(self.node.navigation_status)
            return
        self.waypoint_edit_index = row
        self.waypoint_list.setEnabled(False)
        self.node.navigation_status = (
            f"正在修改途径点 {row + 1}：请在地图上短按新的位置")
        self.nav_status.setText(self.node.navigation_status)
        self._on_waypoint_selection_changed(row)

    def delete_selected_waypoint(self) -> None:
        row = self.waypoint_list.currentRow()
        if not 0 <= row < len(self.waypoints):
            self.node.navigation_status = "请先在列表中选择要删除的途径点"
            self.nav_status.setText(self.node.navigation_status)
            return
        self.waypoints.pop(row)
        self.waypoint_edit_index = None
        self.waypoint_list.setEnabled(True)
        self.node.navigation_status = f"已删除途径点 {row + 1}"
        self.nav_status.setText(self.node.navigation_status)
        next_row = min(row, len(self.waypoints) - 1)
        self._sync_selection_ui(next_row)

    def _sync_selection_ui(self, preferred_index: Optional[int] = None) -> None:
        current_row = (
            self.waypoint_list.currentRow()
            if preferred_index is None else preferred_index)
        self.waypoint_list.blockSignals(True)
        self.waypoint_list.clear()
        for index, point in enumerate(self.waypoints):
            self.waypoint_list.addItem(
                f"{index + 1}.   X {point[0]:.2f} m    Y {point[1]:.2f} m")
        if self.waypoints:
            current_row = int(np.clip(current_row, 0, len(self.waypoints) - 1))
            self.waypoint_list.setCurrentRow(current_row)
        else:
            current_row = -1
        self.waypoint_list.blockSignals(False)
        self.waypoint_title.setText(
            f"途径点（{len(self.waypoints)}）"
            if self.waypoints else "途径点（尚未添加）")
        self.map_panel.set_waypoints(self.waypoints)
        self._on_waypoint_selection_changed(current_row)

    def start_navigation(self) -> None:
        if self.waypoint_edit_index is not None:
            self.node.navigation_status = "请先在地图上完成途径点重新选择"
            self.nav_status.setText(self.node.navigation_status)
            return
        self.navigation_return_timer.stop()
        self.pending_navigation_result_status = None
        self.active_page.clear_navigation_result()
        self.last_navigation_result_revision = int(
            getattr(self.node, "navigation_result_revision", 0))
        if self.node.send_navigation_waypoints(self.waypoints):
            self.active_page.map_panel.set_waypoints([])
            self.pages.setCurrentWidget(self.active_page)

    def exit_navigation(self) -> None:
        if self.pending_navigation_result_status is not None:
            self.navigation_return_timer.stop()
            self._finish_navigation_return()
            return
        self.node.cancel_navigation()
        self.active_page.clear_navigation_result()
        self.clear_selection()
        self.pages.setCurrentWidget(self.setup_page)

    def _handle_navigation_terminal_state(self) -> None:
        revision = int(getattr(
            self.node, "navigation_result_revision",
            self.last_navigation_result_revision))
        if revision == self.last_navigation_result_revision:
            return
        self.last_navigation_result_revision = revision
        if self.pages.currentWidget() is not self.active_page:
            return
        status = getattr(self.node, "navigation_result_status", None)
        self.pending_navigation_result_status = status
        self.active_page.show_navigation_result(
            status, str(self.node.navigation_status))
        self.navigation_return_timer.start()

    def _finish_navigation_return(self) -> None:
        status = self.pending_navigation_result_status
        self.pending_navigation_result_status = None
        self.clear_selection()
        self.active_page.map_panel.set_navigation_state(None, None, 0.0)
        self.active_page.clear_navigation_result()
        self.pages.setCurrentWidget(self.setup_page)
        if status == 4:
            self.node.navigation_status = "导航成功，已自动返回地图"

    def refresh(self) -> None:
        self._handle_navigation_terminal_state()
        image = self.node.latest_image
        route = None
        if image is not None:
            route = self.node.projected_route(image.shape)
        self.camera_panel.set_frame(
            image, route, self.node.remaining_distance())
        self.active_page.camera_panel.set_frame(
            image, route, self.node.remaining_distance())
        if self.local_grid is not None:
            if self._rendered_grid_edit_revision != self._grid_edit_revision:
                self._render_grid(self.local_grid)
                self._rendered_grid_edit_revision = self._grid_edit_revision
        elif self.node.map_revision != self.last_map_revision:
            for panel in (
                self.map_panel, self.editor_map_panel,
                self.active_page.map_panel,
            ):
                panel.set_map(
                    self.node.occupancy,
                    self.node.map_origin,
                    self.node.map_resolution,
                    self.node.map_origin_yaw,
                )
            self.last_map_revision = self.node.map_revision
        if self.local_cloud is not None:
            if self._rendered_cloud_edit_revision != self._cloud_edit_revision:
                for panel in (
                    self.map_panel, self.editor_map_panel,
                    self.active_page.map_panel,
                ):
                    self._set_panel_cloud(panel, self.local_cloud)
                self._rendered_cloud_edit_revision = self._cloud_edit_revision
        elif self.node.pointcloud_revision != self.last_pointcloud_revision:
            for panel in (
                self.map_panel, self.editor_map_panel,
                self.active_page.map_panel,
            ):
                panel.set_pointcloud(self.node.latest_pointcloud)
            self.last_pointcloud_revision = self.node.pointcloud_revision
        path, robot_xy, robot_yaw = self.node.map_navigation_state()
        self.map_panel.set_navigation_state(path, robot_xy, robot_yaw)
        self.active_page.map_panel.set_navigation_state(
            path, robot_xy, robot_yaw)
        self.nav_status.setText(self.node.navigation_status)

    def closeEvent(self, event) -> None:
        """Stop child launch processes before closing the desktop app."""
        self.refresh_timer.stop()
        self.navigation_return_timer.stop()
        for controller in self.sensor_processes.values():
            controller.shutdown()
        super().closeEvent(event)


def main(args=None) -> None:
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    app.setApplicationName("GS Navigation Console")
    node = QtNavRosNode()
    window = NavigationWindow(node)
    window.show()
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())

    ros_timer = QTimer()
    ros_timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
    ros_timer.start(5)
    try:
        app.exec_()
    finally:
        ros_timer.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
