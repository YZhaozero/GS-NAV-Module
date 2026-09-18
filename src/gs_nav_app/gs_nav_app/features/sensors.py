"""Sensor driver lifecycle and live ROS topic preview adapters."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu, PointCloud2

from ..map_processing import pointcloud2_to_xyz
from ..nav_math import image_to_bgr
from ..ros_launch_process import RosLaunchProcess

try:
    from livox_ros_driver2.msg import CustomMsg as LivoxCustomMsg
except ImportError:
    LivoxCustomMsg = None


class SensorPreviewRosAdapter:
    """Own temporary subscriptions used by the sensor inspection page."""

    def __init__(self, node) -> None:
        self.node = node
        self.image: Optional[np.ndarray] = None
        self.cloud: Optional[np.ndarray] = None
        self.imu = None
        self.revisions = {"camera": 0, "lidar": 0, "imu": 0}
        self.topics = {"camera": "", "lidar": "", "imu": ""}
        self.errors = {"camera": "", "lidar": "", "imu": ""}
        self._subscriptions = {"camera": None, "lidar": None, "imu": None}

    def available_topics(self):
        type_to_kind = {
            "sensor_msgs/msg/PointCloud2": "lidar",
            "livox_ros_driver2/msg/CustomMsg": "lidar",
            "sensor_msgs/msg/Image": "camera",
            "sensor_msgs/msg/Imu": "imu",
        }
        grouped = {"lidar": [], "camera": [], "imu": []}
        for topic_name, topic_types in self.node.get_topic_names_and_types():
            for topic_type in topic_types:
                kind = type_to_kind.get(topic_type)
                if kind is not None:
                    grouped[kind].append(topic_name)
                    break
        for topics in grouped.values():
            topics.sort()
        return grouped

    def start(self, kind: str, topic: str):
        message_types = {
            "lidar": PointCloud2,
            "camera": Image,
            "imu": Imu,
        }
        callbacks = {
            "lidar": self._on_cloud,
            "camera": self._on_image,
            "imu": self._on_imu,
        }
        if kind not in message_types:
            return False, f"不支持的传感器类型：{kind}"
        topic = str(topic).strip()
        if not topic:
            return False, "请选择或输入话题"
        if kind == "lidar":
            topic_types = []
            try:
                topic_types = dict(
                    self.node.get_topic_names_and_types()).get(topic, [])
            except RuntimeError:
                pass
            if "livox_ros_driver2/msg/CustomMsg" in topic_types:
                if LivoxCustomMsg is None:
                    return False, "当前环境缺少 livox_ros_driver2/CustomMsg"
                message_types[kind] = LivoxCustomMsg
                callbacks[kind] = self._on_livox
        self.stop(kind)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        try:
            subscription = self.node.create_subscription(
                message_types[kind], topic, callbacks[kind], qos)
        except (RuntimeError, ValueError) as exc:
            self.errors[kind] = str(exc)
            return False, f"订阅失败：{exc}"
        self._subscriptions[kind] = subscription
        self.topics[kind] = topic
        self.errors[kind] = ""
        return True, f"正在订阅 {topic}"

    def stop(self, kind: Optional[str] = None) -> None:
        kinds = tuple(self._subscriptions) if kind is None else (kind,)
        for sensor_kind in kinds:
            subscription = self._subscriptions.get(sensor_kind)
            if subscription is not None:
                self.node.destroy_subscription(subscription)
                self._subscriptions[sensor_kind] = None
            if sensor_kind in self.topics:
                self.topics[sensor_kind] = ""

    def _on_image(self, msg: Image) -> None:
        try:
            self.image = image_to_bgr(msg)
            self.errors["camera"] = ""
            self.revisions["camera"] += 1
        except (ValueError, TypeError) as exc:
            self.errors["camera"] = f"图像格式错误：{exc}"

    def _on_cloud(self, msg: PointCloud2) -> None:
        try:
            self.cloud = pointcloud2_to_xyz(msg)
            self.errors["lidar"] = ""
            self.revisions["lidar"] += 1
        except (ValueError, TypeError) as exc:
            self.errors["lidar"] = f"点云格式错误：{exc}"

    def _on_livox(self, msg) -> None:
        try:
            self.cloud = np.asarray([
                (point.x, point.y, point.z) for point in msg.points
            ], dtype=np.float32).reshape(-1, 3)
            self.errors["lidar"] = ""
            self.revisions["lidar"] += 1
        except (AttributeError, TypeError, ValueError) as exc:
            self.errors["lidar"] = f"Livox 点云格式错误：{exc}"

    def _on_imu(self, msg: Imu) -> None:
        self.imu = {
            "frame_id": msg.header.frame_id,
            "stamp": (msg.header.stamp.sec, msg.header.stamp.nanosec),
            "orientation": (
                msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w,
            ),
            "angular_velocity": (
                msg.angular_velocity.x, msg.angular_velocity.y,
                msg.angular_velocity.z,
            ),
            "linear_acceleration": (
                msg.linear_acceleration.x, msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ),
            "orientation_covariance": tuple(msg.orientation_covariance),
            "angular_velocity_covariance": tuple(
                msg.angular_velocity_covariance),
            "linear_acceleration_covariance": tuple(
                msg.linear_acceleration_covariance),
        }
        self.errors["imu"] = ""
        self.revisions["imu"] += 1


class UnavailableSensorPreviewAdapter:
    """Inert preview source used by UI-only tests and mock nodes."""

    image = None
    cloud = None
    imu = None
    revisions = {"camera": 0, "lidar": 0, "imu": 0}
    topics = {"camera": "", "lidar": "", "imu": ""}
    errors = {"camera": "", "lidar": "", "imu": ""}

    def available_topics(self):
        return {"camera": [], "lidar": [], "imu": []}

    def start(self, _kind: str, _topic: str):
        return False, "当前节点未启用传感器监看接口"

    def stop(self, _kind: Optional[str] = None) -> None:
        pass


class SensorDriverController(QObject):
    """Own driver launch processes and expose keyed state/log signals."""

    log_received = pyqtSignal(str, str)
    state_changed = pyqtSignal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.processes = {}
        for key, label in (("lidar", "雷达"), ("camera", "相机")):
            process = RosLaunchProcess(label, self)
            process.log_received.connect(
                lambda text, sensor=key: self.log_received.emit(sensor, text))
            process.state_changed.connect(
                lambda state, sensor=key: self.state_changed.emit(sensor, state))
            self.processes[key] = process

    @staticmethod
    def lidar_launch_arguments(
        xfer_format,
        multi_topic: bool,
        publish_frequency: float,
        frame_id: str,
        config_path: str = "",
    ):
        arguments = [
            "launch", "livox_ros_driver2", "msg_MID360_launch.py",
            f"xfer_format:={xfer_format}",
            f"multi_topic:={int(multi_topic)}",
            f"publish_freq:={publish_frequency:g}",
            f"frame_id:={frame_id.strip() or 'livox_frame'}",
        ]
        if config_path.strip():
            arguments.append(f"user_config_path:={config_path.strip()}")
        return arguments

    @staticmethod
    def camera_launch_arguments(
        camera_name: str,
        camera_namespace: str,
        serial: str,
        enable_color: bool,
        enable_depth: bool,
        enable_gyro: bool,
        enable_accel: bool,
        unite_imu_method,
        enable_sync: bool,
        align_depth: bool,
        pointcloud: bool,
    ):
        def launch_bool(value: bool) -> str:
            return "true" if value else "false"
        arguments = [
            "launch", "realsense2_camera", "d435i.launch.py",
            f"camera_name:={camera_name.strip() or 'camera'}",
            f"camera_namespace:={camera_namespace.strip() or 'camera'}",
            f"enable_color:={launch_bool(enable_color)}",
            f"enable_depth:={launch_bool(enable_depth)}",
            f"enable_gyro:={launch_bool(enable_gyro)}",
            f"enable_accel:={launch_bool(enable_accel)}",
            f"unite_imu_method:={unite_imu_method}",
            f"enable_sync:={launch_bool(enable_sync)}",
            f"align_depth.enable:={launch_bool(align_depth)}",
            f"pointcloud.enable:={launch_bool(pointcloud)}",
        ]
        if serial.strip():
            arguments.append(f"serial_no:={serial.strip()}")
        return arguments

    def start(self, key: str, arguments) -> bool:
        return self.processes[key].start_launch(arguments)

    def stop(self, key: str) -> None:
        self.processes[key].stop_launch()

    def shutdown(self) -> None:
        for process in self.processes.values():
            process.shutdown()
