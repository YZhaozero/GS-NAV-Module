"""Mapping backends, ROS adapter and lifecycle controller.

The UI depends on :class:`MappingController` instead of knowing how DLIO is
launched, which ROS types it consumes, or how its map is saved. A new mapping
algorithm can be registered by adding a :class:`MappingBackend` and a matching
save adapter when required.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from ..map_processing import pointcloud2_to_xyz
from ..ros_launch_process import RosLaunchProcess

try:
    from direct_lidar_inertial_odometry.srv import SavePCD as DlioSavePCD
except ImportError:  # UI remains available before the DLIO package is built.
    DlioSavePCD = None


@dataclass(frozen=True)
class MappingBackend:
    """Declarative contract between one SLAM package and the desktop app."""

    key: str
    label: str
    description: str
    package: str
    launch_file: str
    launch_arguments: Dict[str, str]
    default_pointcloud_topic: str
    default_imu_topic: str
    default_map_topic: str
    save_adapter: str = ""
    save_service: str = ""


MAPPING_BACKENDS = {
    "dlio": MappingBackend(
        key="dlio",
        label="DLIO",
        description="Direct LiDAR-Inertial Odometry（激光雷达 + IMU）",
        package="direct_lidar_inertial_odometry",
        launch_file="dlio.launch.py",
        launch_arguments={
            "pointcloud": "pointcloud_topic",
            "imu": "imu_topic",
            "rviz": "rviz",
            "use_sim_time": "use_sim_time",
        },
        default_pointcloud_topic="/livox/lidar/pointcloud",
        default_imu_topic="/livox/imu",
        default_map_topic="/dlio/map_node/map",
        save_adapter="dlio_save_pcd",
        save_service="/save_pcd",
    ),
}


def unique_map_path(
    directory: Path, map_name: str, now: Optional[datetime] = None,
) -> Path:
    """Return a timestamped PCD path that never replaces an existing map."""
    raw_name = Path(str(map_name).strip()).stem
    safe_name = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in raw_name
    ).strip("_-") or "gs_map"
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    candidate = directory / f"{safe_name}_{timestamp}.pcd"
    suffix = 1
    while candidate.exists():
        candidate = directory / f"{safe_name}_{timestamp}_{suffix:02d}.pcd"
        suffix += 1
    return candidate


class MappingRosAdapter:
    """Own all ROS subscriptions and services required by mapping features."""

    def __init__(self, node) -> None:
        self.node = node
        self.cloud: Optional[np.ndarray] = None
        self.cloud_frame = ""
        self.cloud_topic = ""
        self.cloud_error = ""
        self.cloud_revision = 0
        self.save_status = ""
        self.save_revision = 0
        self.save_in_progress = False
        self.last_saved_path: Optional[Path] = None
        self._pending_save_path: Optional[Path] = None
        self._cloud_subscription = None
        self._dlio_save_client = (
            node.create_client(DlioSavePCD, "/save_pcd")
            if DlioSavePCD is not None else None
        )

    def available_topics(self):
        grouped = {"pointcloud": [], "imu": []}
        for topic_name, topic_types in self.node.get_topic_names_and_types():
            if "sensor_msgs/msg/PointCloud2" in topic_types:
                grouped["pointcloud"].append(topic_name)
            if "sensor_msgs/msg/Imu" in topic_types:
                grouped["imu"].append(topic_name)
        for topics in grouped.values():
            topics.sort()
        return grouped

    def validate_inputs(self, pointcloud_topic: str, imu_topic: str):
        graph = dict(self.node.get_topic_names_and_types())
        expected = (
            (pointcloud_topic, "sensor_msgs/msg/PointCloud2", "雷达"),
            (imu_topic, "sensor_msgs/msg/Imu", "IMU"),
        )
        for topic, message_type, label in expected:
            published_types = graph.get(topic, [])
            if published_types and message_type not in published_types:
                actual = ", ".join(published_types)
                return False, (
                    f"{label}话题 {topic} 类型不兼容：{actual}；"
                    f"当前算法需要 {message_type}")
        return True, "输入话题类型检查通过"

    def start_preview(self, topic: str):
        topic = str(topic).strip()
        if not topic:
            return False, "地图输出话题不能为空"
        self.stop_preview()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        try:
            self._cloud_subscription = self.node.create_subscription(
                PointCloud2, topic, self._on_cloud, qos)
        except (RuntimeError, ValueError) as exc:
            self.cloud_error = str(exc)
            return False, f"地图点云订阅失败：{exc}"
        self.cloud = None
        self.cloud_frame = ""
        self.cloud_revision += 1
        self.cloud_topic = topic
        self.cloud_error = ""
        return True, f"等待建图数据 {topic}"

    def stop_preview(self) -> None:
        if self._cloud_subscription is not None:
            self.node.destroy_subscription(self._cloud_subscription)
            self._cloud_subscription = None
        self.cloud_topic = ""

    def _on_cloud(self, msg: PointCloud2) -> None:
        try:
            self.cloud = pointcloud2_to_xyz(msg)
            self.cloud_frame = msg.header.frame_id
            self.cloud_error = ""
            self.cloud_revision += 1
        except (ValueError, TypeError) as exc:
            self.cloud_error = f"建图点云格式错误：{exc}"

    def save_map(
        self,
        backend: MappingBackend,
        save_path: str,
        map_name: str,
        leaf_size: float,
    ):
        if backend.save_adapter != "dlio_save_pcd":
            return False, "当前建图算法尚未配置地图保存接口"
        if DlioSavePCD is None or self._dlio_save_client is None:
            return False, "未找到 DLIO SavePCD 服务类型，请先构建并 source 工作空间"
        if self.save_in_progress:
            return False, "上一次地图仍在保存，请稍候"
        if self.cloud is None or len(self.cloud) == 0:
            return False, "当前还没有可保存的地图点云，请等待 DLIO 产生关键帧"
        raw_path = str(save_path).strip()
        if not raw_path:
            return False, "请选择地图保存目录"
        target = Path(raw_path).expanduser()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return False, f"无法创建保存目录：{exc}"
        if (
            not self._dlio_save_client.service_is_ready()
            and not self._dlio_save_client.wait_for_service(timeout_sec=0.25)
        ):
            return False, (
                f"保存服务 {backend.save_service} 尚未就绪，请先启动建图")
        output_path = unique_map_path(target.resolve(), map_name)
        request = DlioSavePCD.Request()
        request.leaf_size = float(leaf_size)
        request.save_path = str(output_path)
        self._pending_save_path = output_path
        self.save_in_progress = True
        self.save_status = f"正在保存：{output_path.name}"
        try:
            future = self._dlio_save_client.call_async(request)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.save_in_progress = False
            self._pending_save_path = None
            return False, f"调用地图保存服务失败：{exc}"
        future.add_done_callback(self._on_save_done)
        return True, self.save_status

    def _on_save_done(self, future) -> None:
        output_path = self._pending_save_path
        try:
            response = future.result()
            if (
                response is not None
                and response.success
                and output_path is not None
                and output_path.is_file()
                and output_path.stat().st_size > 0
            ):
                self.last_saved_path = output_path
                self.save_status = f"地图保存完成：{output_path}"
            elif response is not None and response.success:
                self.save_status = (
                    "DLIO 返回成功，但没有找到有效文件："
                    f"{output_path or '(unknown)'}")
            else:
                self.save_status = (
                    f"DLIO 返回保存失败：{output_path or '(unknown)'}")
        except Exception as exc:  # rclpy future transports errors here.
            self.save_status = f"地图保存失败：{exc}"
        self.save_in_progress = False
        self._pending_save_path = None
        self.save_revision += 1


class UnavailableMappingRosAdapter:
    """Inert adapter for previews/tests that do not construct a ROS node."""

    cloud = None
    cloud_frame = ""
    cloud_topic = ""
    cloud_error = ""
    cloud_revision = 0
    save_status = ""
    save_revision = 0
    save_in_progress = False
    last_saved_path = None

    def available_topics(self):
        return {"pointcloud": [], "imu": []}

    def validate_inputs(self, _pointcloud_topic: str, _imu_topic: str):
        return False, "当前节点未启用建图 ROS 接口"

    def start_preview(self, _topic: str):
        return False, "当前节点未启用建图 ROS 接口"

    def stop_preview(self) -> None:
        pass

    def save_map(
        self, _backend, _save_path: str, _map_name: str, _leaf_size: float,
    ):
        return False, "当前节点未启用建图保存接口"


class MappingController(QObject):
    """UI-independent mapping use cases and process lifecycle."""

    log_received = pyqtSignal(str)
    state_changed = pyqtSignal(str)

    def __init__(
        self,
        ros_adapter: MappingRosAdapter,
        backend_key: str = "dlio",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.ros = ros_adapter
        self.backend_key = backend_key
        self.process = RosLaunchProcess(
            f"{self.backend.label} 建图", self)
        self.process.log_received.connect(self.log_received.emit)
        self.process.state_changed.connect(self._on_process_state)

    @property
    def backend(self) -> MappingBackend:
        return MAPPING_BACKENDS[self.backend_key]

    def select_backend(self, backend_key: str) -> MappingBackend:
        if backend_key not in MAPPING_BACKENDS:
            raise KeyError(f"Unknown mapping backend: {backend_key}")
        if self.process.running:
            raise RuntimeError("建图运行时不能切换算法")
        self.backend_key = backend_key
        self.process.display_name = f"{self.backend.label} 建图"
        return self.backend

    def available_topics(self):
        return self.ros.available_topics()

    def build_launch_arguments(
        self,
        pointcloud_topic: str,
        imu_topic: str,
        open_rviz: bool,
        use_sim_time: bool,
    ):
        backend = self.backend
        values = {
            "pointcloud": pointcloud_topic.strip(),
            "imu": imu_topic.strip(),
            "rviz": "true" if open_rviz else "false",
            "use_sim_time": "true" if use_sim_time else "false",
        }
        arguments = ["launch", backend.package, backend.launch_file]
        for role, launch_name in backend.launch_arguments.items():
            if launch_name and values.get(role, ""):
                arguments.append(f"{launch_name}:={values[role]}")
        return arguments

    def start(
        self,
        pointcloud_topic: str,
        imu_topic: str,
        map_topic: str,
        open_rviz: bool = False,
        use_sim_time: bool = False,
    ):
        pointcloud_topic = pointcloud_topic.strip()
        imu_topic = imu_topic.strip()
        map_topic = map_topic.strip()
        if not pointcloud_topic or not imu_topic or not map_topic:
            return False, "雷达、IMU 和地图输出话题都不能为空"
        compatible, reason = self.ros.validate_inputs(
            pointcloud_topic, imu_topic)
        if not compatible:
            self.log_received.emit(reason + "\n")
            return False, reason
        success, status = self.ros.start_preview(map_topic)
        if not success:
            return False, status
        started = self.process.start_launch(self.build_launch_arguments(
            pointcloud_topic, imu_topic, open_rviz, use_sim_time))
        if not started:
            self.ros.stop_preview()
            return False, "建图进程已经在运行"
        return True, status

    def stop(self) -> None:
        self.process.stop_launch()

    def save_map(self, save_path: str, map_name: str, leaf_size: float):
        return self.ros.save_map(
            self.backend, save_path, map_name, leaf_size)

    def shutdown(self) -> None:
        self.process.shutdown()
        self.ros.stop_preview()

    def _on_process_state(self, state: str) -> None:
        if state in ("stopped", "error"):
            self.ros.stop_preview()
        self.state_changed.emit(state)
