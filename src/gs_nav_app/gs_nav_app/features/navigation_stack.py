"""Ordered lifecycle control for the complete navigation runtime stack."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Dict, List, Sequence, Tuple

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from ..ros_launch_process import RosLaunchProcess
from .navigation_backends import (
    LOCALIZATION_BACKENDS,
    NAVIGATION_BACKENDS,
    launch_bool,
)


COMPONENT_LABELS = {
    "lidar": "Livox 雷达",
    "camera": "RealSense 相机",
    "dlio": "DLIO 里程计",
    "laserscan": "点云转 LaserScan",
    "localizer": "定位后端",
    "nav2": "导航后端",
}

START_ORDER = ("lidar", "camera", "dlio", "laserscan", "localizer", "nav2")


@dataclass
class NavigationStackConfig:
    """User-editable inputs required to launch the navigation stack."""

    navigation_map: str
    localization_map: str
    localization_backend: str = "pointcloud_localizer"
    navigation_backend: str = "nav2"
    lidar_arguments: Sequence[str] = field(default_factory=tuple)
    camera_arguments: Sequence[str] = field(default_factory=tuple)
    use_sim_time: bool = False
    autostart: bool = True
    dlio_rviz: bool = False
    dlio_pointcloud_topic: str = "/livox/lidar/pointcloud"
    dlio_imu_topic: str = "/livox/imu"
    laser_cloud_topic: str = "/livox/lidar/pointcloud"
    scan_topic: str = "/scan"
    laser_target_frame: str = "livox_frame"
    min_height: float = -1.0
    max_height: float = 0.1
    range_min: float = 0.45
    range_max: float = 10.0
    localizer_cloud_topic: str = "/dlio/odom_node/pointcloud/deskewed"
    localizer_odom_topic: str = "/dlio/odom_node/odom"
    localizer_config: str = ""
    nav2_params: str = ""
    nav2_rviz: bool = False
    custom_localization_package: str = ""
    custom_localization_launch: str = ""
    custom_localization_arguments: str = ""
    custom_navigation_package: str = ""
    custom_navigation_launch: str = ""
    custom_navigation_arguments: str = ""
    scan_navi_mode: int = 1
    scan_sensor_type: str = "lidar"
    scan_body_pose_topic: str = "/dlio/odom_node/odom"
    scan_sensor_pose_topic: str = "/dlio/odom_node/odom"
    scan_cloud_topic: str = "/livox/lidar/pointcloud"
    scan_depth_topic: str = "/camera/aligned_depth_to_color/image_raw"
    scan_goal_topic: str = "/move_base_simple/goal"
    scan_initial_path_topic: str = "/initial_path"
    scan_cmd_vel_topic: str = "/cmd_vel"
    scan_start_controller: bool = True
    scan_cloud_is_world: bool = False
    scan_need_extrinsic: bool = False
    scan_planner_params: str = ""
    scan_controller_params: str = ""
    scan_keypoints_file: str = ""
    scan_reference_path_file: str = ""
    startup_interval_ms: int = 1000
    sensor_timeout_s: float = 12.0


def validate_navigation_stack(config: NavigationStackConfig) -> Tuple[bool, str]:
    """Validate files and coupled numeric parameters before any process starts."""
    localization = LOCALIZATION_BACKENDS.get(config.localization_backend)
    if localization is None:
        return False, f"未知定位后端：{config.localization_backend}"
    navigation = NAVIGATION_BACKENDS.get(config.navigation_backend)
    if navigation is None:
        return False, f"未知导航后端：{config.navigation_backend}"
    valid, message = localization.validate(config)
    if not valid:
        return valid, message
    valid, message = navigation.validate(config)
    if not valid:
        return valid, message
    if config.min_height >= config.max_height:
        return False, "LaserScan 最低高度必须小于最高高度"
    if config.range_min < 0.0 or config.range_min >= config.range_max:
        return False, "LaserScan 最小量程必须小于最大量程"
    if not config.lidar_arguments or not config.camera_arguments:
        return False, "雷达或相机启动参数不完整"
    return True, "启动参数检查通过"


def build_navigation_launches(
    config: NavigationStackConfig,
) -> Dict[str, List[str]]:
    """Build the exact ros2 launch argument vector for every component."""
    launches = {
        "lidar": list(config.lidar_arguments),
        "camera": list(config.camera_arguments),
        "dlio": [
            "launch", "direct_lidar_inertial_odometry", "dlio.launch.py",
            f"rviz:={launch_bool(config.dlio_rviz)}",
            f"use_sim_time:={launch_bool(config.use_sim_time)}",
            f"pointcloud_topic:={config.dlio_pointcloud_topic.strip()}",
            f"imu_topic:={config.dlio_imu_topic.strip()}",
        ],
        "laserscan": [
            "launch", "pointcloud_to_laserscan",
            "pointcloud_to_laserscan_launch.py",
            f"use_sim_time:={launch_bool(config.use_sim_time)}",
            f"cloud_topic:={config.laser_cloud_topic.strip()}",
            f"scan_topic:={config.scan_topic.strip()}",
            f"target_frame:={config.laser_target_frame.strip()}",
            f"min_height:={config.min_height:g}",
            f"max_height:={config.max_height:g}",
            f"range_min:={config.range_min:g}",
            f"range_max:={config.range_max:g}",
        ],
    }
    localization = LOCALIZATION_BACKENDS[config.localization_backend]
    localization_launch = localization.launch_arguments(config)
    if localization_launch:
        launches["localizer"] = localization_launch
    navigation = NAVIGATION_BACKENDS[config.navigation_backend]
    navigation_launch = navigation.launch_arguments(config)
    if navigation_launch:
        launches["nav2"] = navigation_launch
    return launches


class NavigationStackController(QObject):
    """Start dependent launch files in order and stop managed processes safely."""

    log_received = pyqtSignal(str, str)
    component_state_changed = pyqtSignal(str, str)
    overall_state_changed = pyqtSignal(str)

    def __init__(
        self,
        sensor_controller,
        sensor_active: Callable[[str], bool] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.sensor_controller = sensor_controller
        self.sensor_active = sensor_active or (lambda _key: False)
        self.processes: Dict[str, RosLaunchProcess] = {}
        self.states = {key: "stopped" for key in START_ORDER}
        self.overall_state = "stopped"
        self._launches: Dict[str, List[str]] = {}
        self._queue: List[str] = []
        self._waiting_key = ""
        self._owned_components = set()
        self._stop_targets = set()
        self._stopping = False
        self._startup_interval_ms = 1000
        self._sensor_timeout_s = 12.0
        self._sensor_wait_deadline = 0.0
        self.log_events = []

        for key in ("dlio", "laserscan", "localizer", "nav2"):
            process = RosLaunchProcess(COMPONENT_LABELS[key], self)
            process.log_received.connect(
                lambda text, component=key: self._record_log(
                    component, text))
            process.state_changed.connect(
                lambda state, component=key: self._handle_state(
                    component, state))
            self.processes[key] = process
        self.sensor_controller.log_received.connect(self._record_log)
        self.sensor_controller.state_changed.connect(self._handle_state)

    def start(self, config: NavigationStackConfig) -> Tuple[bool, str]:
        if self._queue or self._waiting_key or self.overall_state in (
            "starting", "running", "stopping",
        ):
            return False, "导航系统已经在启动或运行"
        valid, message = validate_navigation_stack(config)
        if not valid:
            return False, message
        self._launches = build_navigation_launches(config)
        self._queue = [key for key in START_ORDER if key in self._launches]
        self._waiting_key = ""
        self._owned_components.clear()
        self._stopping = False
        self._startup_interval_ms = max(0, int(config.startup_interval_ms))
        self._sensor_timeout_s = max(1.0, float(config.sensor_timeout_s))
        for key in START_ORDER:
            if key not in self._launches:
                self.states[key] = "disabled"
                self.component_state_changed.emit(key, "disabled")
        self._set_overall_state("starting")
        self._record_log("system", "开始按依赖顺序启动导航系统…\n")
        self._start_next()
        return True, "导航系统开始启动"

    def _component_running(self, key: str) -> bool:
        if key in ("lidar", "camera"):
            return bool(self.sensor_active(key))
        return self.processes[key].running

    def _start_next(self) -> None:
        if self._stopping or self.overall_state != "starting":
            return
        if not self._queue:
            self._waiting_key = ""
            self._record_log("system", "导航系统全部组件已启动。\n")
            self._set_overall_state("running")
            QTimer.singleShot(1000, self._monitor_sensor_health)
            return
        key = self._queue.pop(0)
        if self._component_running(key):
            self.states[key] = "running"
            self.component_state_changed.emit(key, "running")
            message = f"{COMPONENT_LABELS[key]}已有真实数据，跳过重复启动。\n"
            self._record_log("system", message)
            if key in ("lidar", "camera"):
                self._record_sensor_log(key, message)
            QTimer.singleShot(self._startup_interval_ms, self._start_next)
            return

        if key in ("lidar", "camera"):
            if self.sensor_controller.processes[key].running:
                self._waiting_key = key
                self._begin_sensor_wait(key, started_here=False)
                return

        self._waiting_key = key
        self._record_log(
            "system", f"正在启动 {COMPONENT_LABELS[key]}…\n")
        if key in ("lidar", "camera"):
            started = self.sensor_controller.start(key, self._launches[key])
        else:
            started = self.processes[key].start_launch(self._launches[key])
        if started:
            self._owned_components.add(key)
            return
        self._waiting_key = ""
        self._queue.clear()
        self._record_log(
            "system", f"{COMPONENT_LABELS[key]}未能启动，启动流程中止。\n")
        self._set_overall_state("error")

    def _handle_state(self, key: str, state: str) -> None:
        if key not in self.states:
            return
        if all((
            key == self._waiting_key,
            key in ("lidar", "camera"),
            state == "running",
        )):
            self._begin_sensor_wait(key, started_here=True)
            return
        self.states[key] = state
        self.component_state_changed.emit(key, state)
        if self._stopping:
            if state in ("stopped", "error"):
                self._finish_stop_if_ready()
            return
        if key == self._waiting_key and state == "running":
            self._waiting_key = ""
            QTimer.singleShot(self._startup_interval_ms, self._start_next)
        elif key == self._waiting_key and state in ("error", "stopped"):
            self._waiting_key = ""
            self._queue.clear()
            self._record_log(
                "system", f"{COMPONENT_LABELS[key]}启动失败，流程中止。\n")
            self._set_overall_state("error")
        elif all((
            self.overall_state in ("starting", "running"),
            key in self._owned_components,
            state in ("error", "stopped"),
        )):
            self._queue.clear()
            self._waiting_key = ""
            self._record_log(
                "system", f"{COMPONENT_LABELS[key]}意外退出。\n")
            self._set_overall_state("error")

    def _begin_sensor_wait(self, key: str, started_here: bool) -> None:
        self.states[key] = "waiting"
        self.component_state_changed.emit(key, "waiting")
        self._sensor_wait_deadline = time.monotonic() + self._sensor_timeout_s
        origin = "驱动已启动" if started_here else "检测到已有驱动进程"
        message = f"{COMPONENT_LABELS[key]}{origin}，等待真实传感器数据…\n"
        self._record_log("system", message)
        self._record_sensor_log(key, message)
        QTimer.singleShot(100, lambda: self._poll_sensor_data(key))

    def _poll_sensor_data(self, key: str) -> None:
        if any((
            self._stopping,
            self.overall_state != "starting",
            self._waiting_key != key,
        )):
            return
        if self.sensor_active(key):
            self._waiting_key = ""
            self.states[key] = "running"
            self.component_state_changed.emit(key, "running")
            message = f"{COMPONENT_LABELS[key]}已收到真实数据。\n"
            self._record_log("system", message)
            self._record_sensor_log(key, message)
            QTimer.singleShot(self._startup_interval_ms, self._start_next)
            return
        if time.monotonic() >= self._sensor_wait_deadline:
            self._waiting_key = ""
            self._queue.clear()
            self.states[key] = "error"
            self.component_state_changed.emit(key, "error")
            message = (
                f"{COMPONENT_LABELS[key]}在 {self._sensor_timeout_s:g} 秒内"
                "没有收到数据，请检查供电、网络和设备连接。\n")
            self._record_log("system", message)
            self._record_sensor_log(key, message)
            self._set_overall_state("error")
            if key in self._owned_components:
                self.sensor_controller.stop(key)
            return
        QTimer.singleShot(200, lambda: self._poll_sensor_data(key))

    def _monitor_sensor_health(self) -> None:
        if self.overall_state != "running":
            return
        for key in ("lidar", "camera"):
            if not self.sensor_active(key):
                self.states[key] = "error"
                self.component_state_changed.emit(key, "error")
                message = f"{COMPONENT_LABELS[key]}数据已中断。\n"
                self._record_log("system", message)
                self._record_sensor_log(key, message)
                self._set_overall_state("error")
                return
        QTimer.singleShot(1000, self._monitor_sensor_health)

    def stop(self) -> None:
        self._queue.clear()
        self._waiting_key = ""
        targets = set(self._owned_components)
        for key in START_ORDER:
            process = self._component_process(key)
            if self._process_active(process):
                targets.add(key)
        self._stop_targets = targets
        if not targets:
            self._record_log("system", "没有仍在运行的受管组件需要停止。\n")
            self._set_overall_state("stopped")
            return
        self._stopping = True
        self._set_overall_state("stopping")
        self._record_log("system", "正在按反向顺序停止导航系统…\n")
        for key in reversed(START_ORDER):
            if key not in targets:
                continue
            if key in ("lidar", "camera"):
                self.sensor_controller.stop(key)
            else:
                self.processes[key].stop_launch()
        self._finish_stop_if_ready()

    def _finish_stop_if_ready(self) -> None:
        if not self._stopping:
            return
        if any(
            self._process_active(process)
            for process in (
                self._component_process(key) for key in self._stop_targets)
        ):
            return
        self._owned_components.clear()
        stopped_targets = set(self._stop_targets)
        self._stop_targets.clear()
        self._stopping = False
        for key in stopped_targets:
            self.states[key] = "stopped"
            self.component_state_changed.emit(key, "stopped")
        self._record_log("system", "导航系统进程已全部停止。\n")
        self._set_overall_state("stopped")

    def _component_process(self, key: str):
        if key in ("lidar", "camera"):
            return self.sensor_controller.processes[key]
        return self.processes[key]

    @staticmethod
    def _process_active(process) -> bool:
        """Support real process wrappers and lightweight test doubles."""
        return bool(getattr(process, "active", getattr(process, "running", False)))

    def _set_overall_state(self, state: str) -> None:
        self.overall_state = state
        self.overall_state_changed.emit(state)

    def _record_log(self, key: str, text: str) -> None:
        if not text:
            return
        self.log_events.append((key, text))
        if len(self.log_events) > 10000:
            del self.log_events[:2000]
        self.log_received.emit(key, text)

    def _record_sensor_log(self, key: str, text: str) -> None:
        """Mirror diagnostics into the shared sensor page when supported."""
        append_log = getattr(self.sensor_controller, "append_log", None)
        if callable(append_log):
            append_log(key, text)
        else:
            self._record_log(key, text)

    def shutdown(self) -> None:
        self._queue.clear()
        self._waiting_key = ""
        for process in self.processes.values():
            process.shutdown()
