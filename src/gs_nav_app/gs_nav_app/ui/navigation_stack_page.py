"""Qt page for configuring and launching the complete navigation stack."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..features.navigation_stack import (
    COMPONENT_LABELS,
    START_ORDER,
    NavigationStackConfig,
    NavigationStackController,
)


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def latest_map(directory: Path, suffixes) -> str:
    candidates = []
    for suffix in suffixes:
        candidates.extend(directory.glob(f"*{suffix}"))
    files = [path for path in candidates if path.is_file()]
    return str(max(files, key=lambda path: path.stat().st_mtime)) if files else ""


class NavigationStackPage(QWidget):
    """Parameter editor, ordered launcher, process status and combined logs."""

    return_requested = pyqtSignal()

    def __init__(
        self,
        controller: NavigationStackController,
        map_storage_dir: Path,
        lidar_arguments: Callable[[], list],
        camera_arguments: Callable[[], list],
    ) -> None:
        super().__init__()
        self.controller = controller
        self.map_storage_dir = Path(map_storage_dir)
        self.lidar_arguments = lidar_arguments
        self.camera_arguments = camera_arguments
        self.component_status = {}
        self._build_ui()
        self.controller.log_received.connect(self._append_log)
        self.controller.component_state_changed.connect(
            self._handle_component_state)
        self.controller.overall_state_changed.connect(
            self._handle_overall_state)
        self._handle_overall_state("stopped")
        for key in START_ORDER:
            self._handle_component_state(key, "stopped")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(14)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("GS NAVIGATION SYSTEM")
        title.setObjectName("title")
        subtitle = QLabel("传感器 · DLIO · LaserScan · Localizer · Nav2 顺序启动")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        back = QPushButton("返回 AR 导航")
        back.setObjectName("workspaceButton")
        back.clicked.connect(self.return_requested.emit)
        header.addWidget(back)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        controls = QFrame()
        controls.setObjectName("sidePanel")
        controls.setMinimumWidth(560)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(10)
        parameters_title = QLabel("启动参数")
        parameters_title.setObjectName("sectionTitle")
        controls_layout.addWidget(parameters_title)
        tabs = QTabWidget()
        tabs.addTab(self._build_map_tab(), "地图与定位")
        tabs.addTab(self._build_dlio_tab(), "DLIO")
        tabs.addTab(self._build_laserscan_tab(), "LaserScan")
        tabs.addTab(self._build_general_tab(), "通用")
        controls_layout.addWidget(tabs, 1)

        sensor_hint = QLabel(
            "雷达与相机使用“传感器管理”页面中的参数；若对应驱动进程或 ROS "
            "话题已经存在，本流程会跳过重复启动。")
        sensor_hint.setObjectName("hint")
        sensor_hint.setWordWrap(True)
        controls_layout.addWidget(sensor_hint)
        action_row = QHBoxLayout()
        self.overall_status = QLabel("● 未启动")
        self.overall_status.setObjectName("sensorStatus")
        action_row.addWidget(self.overall_status)
        action_row.addStretch(1)
        self.stop_button = QPushButton("停止整套系统")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.clicked.connect(self.controller.stop)
        self.start_button = QPushButton("按顺序启动")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_stack)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.start_button)
        controls_layout.addLayout(action_row)
        self.action_status = QLabel("请选择导航地图和定位点云地图")
        self.action_status.setObjectName("statusBar")
        self.action_status.setWordWrap(True)
        controls_layout.addWidget(self.action_status)
        splitter.addWidget(controls)

        monitor = QFrame()
        monitor.setObjectName("sidePanel")
        monitor_layout = QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(16, 16, 16, 16)
        status_title = QLabel("启动顺序与状态")
        status_title.setObjectName("sectionTitle")
        monitor_layout.addWidget(status_title)
        for index, key in enumerate(START_ORDER, start=1):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{index}. {COMPONENT_LABELS[key]}"))
            row.addStretch(1)
            status = QLabel("● 未启动")
            status.setObjectName("sensorStatus")
            self.component_status[key] = status
            row.addWidget(status)
            monitor_layout.addLayout(row)
        log_header = QHBoxLayout()
        log_title = QLabel("启动日志")
        log_title.setObjectName("sectionTitle")
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        clear = QPushButton("清空")
        clear.setObjectName("secondaryButton")
        clear.clicked.connect(self._clear_logs)
        log_header.addWidget(clear)
        monitor_layout.addLayout(log_header)
        self.log_views = {}
        self.log_tabs = QTabWidget()
        for key, label in (
            ("all", "全部"),
            ("system", "系统"),
            *((component, COMPONENT_LABELS[component])
              for component in START_ORDER),
        ):
            view = QPlainTextEdit()
            view.setObjectName("sensorLog")
            view.setReadOnly(True)
            view.setLineWrapMode(QPlainTextEdit.NoWrap)
            view.document().setMaximumBlockCount(3000)
            self.log_views[key] = view
            self.log_tabs.addTab(view, label)
        monitor_layout.addWidget(self.log_tabs, 1)
        splitter.addWidget(monitor)
        splitter.setSizes([620, 780])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

    def _build_map_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.navigation_map = QLineEdit(latest_map(
            self.map_storage_dir, (".yaml", ".yml")))
        form.addRow("Nav2 导航地图", self._path_row(
            self.navigation_map, self._choose_navigation_map))
        self.localization_map = QLineEdit(latest_map(
            self.map_storage_dir, (".pcd",)))
        form.addRow("Localizer 定位地图", self._path_row(
            self.localization_map, self._choose_localization_map))
        self.nav2_params = QLineEdit()
        self.nav2_params.setPlaceholderText("留空使用 nav2_bringup 默认参数")
        form.addRow("Nav2 参数文件", self._path_row(
            self.nav2_params, self._choose_nav2_params))
        self.localizer_config = QLineEdit()
        self.localizer_config.setPlaceholderText("留空使用 localizer 包内配置")
        form.addRow("Localizer 配置", self._path_row(
            self.localizer_config, self._choose_localizer_config))
        self.localizer_cloud_topic = QLineEdit(
            "/dlio/odom_node/pointcloud/deskewed")
        self.localizer_odom_topic = QLineEdit("/dlio/odom_node/odom")
        form.addRow("定位点云话题", self.localizer_cloud_topic)
        form.addRow("定位里程计话题", self.localizer_odom_topic)
        return tab

    def _build_dlio_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.dlio_pointcloud_topic = QLineEdit("/livox/lidar/pointcloud")
        self.dlio_imu_topic = QLineEdit("/livox/imu")
        self.dlio_rviz = QCheckBox("启动 DLIO RViz")
        form.addRow("点云话题", self.dlio_pointcloud_topic)
        form.addRow("IMU 话题", self.dlio_imu_topic)
        form.addRow("RViz", self.dlio_rviz)
        return tab

    def _build_laserscan_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.laser_cloud_topic = QLineEdit("/livox/lidar/pointcloud")
        self.scan_topic = QLineEdit("/scan")
        self.laser_target_frame = QLineEdit("livox_frame")
        form.addRow("输入点云", self.laser_cloud_topic)
        form.addRow("输出扫描", self.scan_topic)
        form.addRow("目标坐标系", self.laser_target_frame)
        self.min_height = self._number(-20.0, 20.0, -1.0)
        self.max_height = self._number(-20.0, 20.0, 0.1)
        self.range_min = self._number(0.0, 1000.0, 0.45)
        self.range_max = self._number(0.01, 1000.0, 10.0)
        form.addRow("最低高度", self.min_height)
        form.addRow("最高高度", self.max_height)
        form.addRow("最小量程", self.range_min)
        form.addRow("最大量程", self.range_max)
        return tab

    def _build_general_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.use_sim_time = QCheckBox("使用 /clock 仿真时间")
        self.autostart = QCheckBox("Nav2 自动进入 Active")
        self.autostart.setChecked(True)
        self.nav2_rviz = QCheckBox("同时启动 Nav2 RViz")
        self.startup_interval = self._number(0.0, 10.0, 1.0)
        self.startup_interval.setSuffix(" s")
        self.sensor_timeout = self._number(1.0, 120.0, 12.0)
        self.sensor_timeout.setSuffix(" s")
        form.addRow("时间源", self.use_sim_time)
        form.addRow("Nav2", self.autostart)
        form.addRow("可视化", self.nav2_rviz)
        form.addRow("组件启动间隔", self.startup_interval)
        form.addRow("传感器数据超时", self.sensor_timeout)
        return tab

    @staticmethod
    def _number(minimum: float, maximum: float, value: float):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(2)
        box.setSingleStep(0.05)
        box.setValue(value)
        box.setSuffix(" m")
        return box

    @staticmethod
    def _path_row(field: QLineEdit, callback) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        choose = QPushButton("选择")
        choose.setObjectName("secondaryButton")
        choose.clicked.connect(callback)
        layout.addWidget(field, 1)
        layout.addWidget(choose)
        return row

    def _choose_navigation_map(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 Nav2 导航地图", str(self.map_storage_dir),
            "Nav2 Map (*.yaml *.yml)")
        if path:
            self.navigation_map.setText(path)

    def _choose_localization_map(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 Localizer 定位地图", str(self.map_storage_dir),
            "Point Cloud (*.pcd)")
        if path:
            self.localization_map.setText(path)

    def _choose_nav2_params(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 Nav2 参数文件", self.nav2_params.text(),
            "YAML (*.yaml *.yml)")
        if path:
            self.nav2_params.setText(path)

    def _choose_localizer_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 Localizer 配置", self.localizer_config.text(),
            "YAML (*.yaml *.yml)")
        if path:
            self.localizer_config.setText(path)

    def configuration(self) -> NavigationStackConfig:
        return NavigationStackConfig(
            navigation_map=self.navigation_map.text().strip(),
            localization_map=self.localization_map.text().strip(),
            lidar_arguments=self.lidar_arguments(),
            camera_arguments=self.camera_arguments(),
            use_sim_time=self.use_sim_time.isChecked(),
            autostart=self.autostart.isChecked(),
            dlio_rviz=self.dlio_rviz.isChecked(),
            dlio_pointcloud_topic=self.dlio_pointcloud_topic.text().strip(),
            dlio_imu_topic=self.dlio_imu_topic.text().strip(),
            laser_cloud_topic=self.laser_cloud_topic.text().strip(),
            scan_topic=self.scan_topic.text().strip(),
            laser_target_frame=self.laser_target_frame.text().strip(),
            min_height=self.min_height.value(),
            max_height=self.max_height.value(),
            range_min=self.range_min.value(),
            range_max=self.range_max.value(),
            localizer_cloud_topic=self.localizer_cloud_topic.text().strip(),
            localizer_odom_topic=self.localizer_odom_topic.text().strip(),
            localizer_config=self.localizer_config.text().strip(),
            nav2_params=self.nav2_params.text().strip(),
            nav2_rviz=self.nav2_rviz.isChecked(),
            startup_interval_ms=int(self.startup_interval.value() * 1000),
            sensor_timeout_s=self.sensor_timeout.value(),
        )

    def start_stack(self) -> None:
        success, status = self.controller.start(self.configuration())
        self.action_status.setText(status)
        if not success:
            self._append_log("system", status + "\n")

    def activate(self) -> None:
        """Rebuild all tabs from persistent controller history."""
        self._clear_logs()
        for key, text in self.controller.log_events:
            self._append_log(key, text)
        for key, state in self.controller.states.items():
            self._handle_component_state(key, state)
        self._handle_overall_state(self.controller.overall_state)

    def _append_log(self, key: str, text: str) -> None:
        clean = ANSI_ESCAPE_RE.sub("", text).replace("\r", "").rstrip("\n")
        if not clean:
            return
        prefix = "系统" if key == "system" else COMPONENT_LABELS.get(key, key)
        own_view = self.log_views.get(key)
        if own_view is not None:
            own_view.appendPlainText(clean)
        self.log_views["all"].appendPlainText(
            "\n".join(f"[{prefix}] {line}" for line in clean.splitlines()))

    def _clear_logs(self) -> None:
        for view in self.log_views.values():
            view.clear()

    def _handle_component_state(self, key: str, state: str) -> None:
        labels = {
            "starting": ("启动中", "#ffd166"),
            "waiting": ("等待数据", "#ffd166"),
            "running": ("运行中", "#66e09a"),
            "stopping": ("停止中", "#ffd166"),
            "stopped": ("未启动", "#91a2ad"),
            "error": ("失败", "#ff7f88"),
        }
        text, color = labels.get(state, (state, "#91a2ad"))
        status = self.component_status.get(key)
        if status is not None:
            status.setText(f"● {text}")
            status.setStyleSheet(f"color: {color}; font-weight: 700;")

    def _handle_overall_state(self, state: str) -> None:
        labels = {
            "starting": ("系统启动中", "#ffd166"),
            "running": ("系统运行中", "#66e09a"),
            "stopping": ("系统停止中", "#ffd166"),
            "stopped": ("系统未启动", "#91a2ad"),
            "error": ("启动失败", "#ff7f88"),
        }
        text, color = labels.get(state, (state, "#91a2ad"))
        self.overall_status.setText(f"● {text}")
        self.overall_status.setStyleSheet(
            f"color: {color}; font-weight: 700;")
        busy = state in ("starting", "stopping")
        self.start_button.setEnabled(not busy and state != "running")
        self.stop_button.setEnabled(state in ("starting", "running", "error"))
        if state == "running":
            self.action_status.setText("导航系统已完成顺序启动，可以进入 AR 导航")
        elif state == "error":
            self.action_status.setText("导航系统启动失败，请查看右侧日志")
