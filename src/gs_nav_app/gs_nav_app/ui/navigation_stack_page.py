"""Qt page for configuring and launching the complete navigation stack."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
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
from ..features.navigation_backends import (
    LOCALIZATION_BACKENDS,
    NAVIGATION_BACKENDS,
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

        header = QVBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("GS NAVIGATION SYSTEM")
        title.setObjectName("title")
        subtitle = QLabel("传感器 · 里程计 · 定位后端 · 导航后端顺序启动")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header_actions = QHBoxLayout()
        header_actions.addStretch(1)
        back = QPushButton("返回 AR 导航")
        back.setObjectName("workspaceButton")
        back.clicked.connect(self.return_requested.emit)
        header_actions.addWidget(back)
        header.addLayout(header_actions)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        self.splitter = splitter
        controls = QFrame()
        controls.setObjectName("sidePanel")
        controls.setMinimumWidth(560)
        self.controls = controls
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(10)
        parameters_title = QLabel("启动参数")
        parameters_title.setObjectName("sectionTitle")
        controls_layout.addWidget(parameters_title)
        self.parameter_tabs = QTabWidget()
        self.parameter_tabs.addTab(self._build_map_tab(), "地图")
        self.parameter_tabs.addTab(self._build_localization_tab(), "定位")
        self.parameter_tabs.addTab(self._build_navigation_tab(), "导航")
        self.parameter_tabs.addTab(self._build_dlio_tab(), "DLIO")
        self.parameter_tabs.addTab(self._build_laserscan_tab(), "LaserScan")
        self.parameter_tabs.addTab(self._build_general_tab(), "通用")
        controls_layout.addWidget(self.parameter_tabs, 1)

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
        self.action_status = QLabel("请选择地图、定位后端和导航后端")
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

    def set_compact_mode(self, compact: bool) -> None:
        """Stack controls and logs when the screen cannot fit two columns."""
        compact = bool(compact)
        self.layout().setContentsMargins(
            *(8, 6, 8, 8) if compact else (22, 18, 22, 22))
        self.layout().setSpacing(7 if compact else 14)
        self.splitter.setOrientation(Qt.Vertical if compact else Qt.Horizontal)
        self.controls.setMinimumWidth(0 if compact else 560)
        self.splitter.setSizes([400, 250] if compact else [620, 780])
        for label in self.findChildren(QLabel):
            if label.objectName() == "subtitle":
                label.setVisible(not compact)

    def _build_map_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.navigation_map = QLineEdit(latest_map(
            self.map_storage_dir, (".yaml", ".yml")))
        form.addRow("导航地图", self._path_row(
            self.navigation_map, self._choose_navigation_map))
        self.localization_map = QLineEdit(latest_map(
            self.map_storage_dir, (".pcd",)))
        form.addRow("定位地图", self._path_row(
            self.localization_map, self._choose_localization_map))
        hint = QLabel(
            "地图只作为独立资源配置：Nav2 使用 YAML 栅格地图，当前点云 "
            "Localizer 使用 PCD；不需要地图的后端会忽略对应文件。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_localization_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        selector = QFormLayout()
        selector.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.localization_backend = QComboBox()
        for key, backend in LOCALIZATION_BACKENDS.items():
            self.localization_backend.addItem(backend.label, key)
        selector.addRow("定位后端", self.localization_backend)
        layout.addLayout(selector)
        self.localization_backend_description = QLabel()
        self.localization_backend_description.setObjectName("hint")
        self.localization_backend_description.setWordWrap(True)
        layout.addWidget(self.localization_backend_description)
        self.localization_backend_stack = QStackedWidget()
        self.localization_backend_stack.addWidget(
            self._build_pointcloud_localizer_panel())
        self.localization_backend_stack.addWidget(
            self._build_disabled_localization_panel())
        self.localization_backend_stack.addWidget(
            self._build_custom_localization_panel())
        layout.addWidget(self.localization_backend_stack, 1)
        self.localization_backend.currentIndexChanged.connect(
            self._update_localization_backend)
        self._update_localization_backend(0)
        return tab

    def _build_pointcloud_localizer_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.localizer_config = QLineEdit()
        self.localizer_config.setPlaceholderText("留空使用 localizer 包内配置")
        form.addRow("Localizer 配置", self._path_row(
            self.localizer_config, self._choose_localizer_config))
        self.localizer_cloud_topic = QLineEdit(
            "/dlio/odom_node/pointcloud/deskewed")
        self.localizer_odom_topic = QLineEdit("/dlio/odom_node/odom")
        form.addRow("定位点云话题", self.localizer_cloud_topic)
        form.addRow("定位里程计话题", self.localizer_odom_topic)
        return panel

    @staticmethod
    def _build_disabled_localization_panel() -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        message = QLabel(
            "定位阶段将从启动顺序中移除。适合直接使用 LIO 里程计，或定位由外部系统"
            "提供的导航方案。")
        message.setObjectName("hint")
        message.setWordWrap(True)
        layout.addWidget(message)
        layout.addStretch(1)
        return panel

    def _build_custom_localization_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.custom_localization_package = QLineEdit()
        self.custom_localization_launch = QLineEdit()
        self.custom_localization_arguments = QLineEdit()
        self.custom_localization_arguments.setPlaceholderText(
            "map:={map} config:={config} use_sim_time:={use_sim_time}")
        form.addRow("ROS 包名", self.custom_localization_package)
        form.addRow("Launch 文件", self.custom_localization_launch)
        form.addRow("附加参数", self.custom_localization_arguments)
        hint = QLabel("支持占位符：{map}、{config}、{use_sim_time}")
        hint.setObjectName("hint")
        form.addRow("", hint)
        return panel

    def _build_navigation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        selector = QFormLayout()
        selector.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.navigation_backend = QComboBox()
        for key, backend in NAVIGATION_BACKENDS.items():
            self.navigation_backend.addItem(backend.label, key)
        selector.addRow("导航后端", self.navigation_backend)
        layout.addLayout(selector)
        self.navigation_backend_description = QLabel()
        self.navigation_backend_description.setObjectName("hint")
        self.navigation_backend_description.setWordWrap(True)
        layout.addWidget(self.navigation_backend_description)
        self.navigation_backend_stack = QStackedWidget()
        self.navigation_backend_stack.addWidget(self._build_nav2_panel())
        self.navigation_backend_stack.addWidget(
            self._scrollable(self._build_scan_planner_panel()))
        self.navigation_backend_stack.addWidget(
            self._build_custom_navigation_panel())
        layout.addWidget(self.navigation_backend_stack, 1)
        self.navigation_backend.currentIndexChanged.connect(
            self._update_navigation_backend)
        self._update_navigation_backend(0)
        return tab

    def _build_nav2_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.nav2_params = QLineEdit()
        self.nav2_params.setPlaceholderText("留空使用 nav2_bringup 默认参数")
        form.addRow("Nav2 参数文件", self._path_row(
            self.nav2_params, self._choose_nav2_params))
        self.autostart = QCheckBox("Nav2 自动进入 Active")
        self.autostart.setChecked(True)
        self.nav2_rviz = QCheckBox("同时启动 Nav2 RViz")
        form.addRow("生命周期", self.autostart)
        form.addRow("可视化", self.nav2_rviz)
        return panel

    def _build_scan_planner_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.scan_navi_mode = QComboBox()
        self.scan_navi_mode.addItem("1 · 单目标", 1)
        self.scan_navi_mode.addItem("2 · 参数途径点", 2)
        self.scan_navi_mode.addItem("3 · 外部参考路径", 3)
        self.scan_sensor_type = QComboBox()
        self.scan_sensor_type.addItem("激光点云", "lidar")
        self.scan_sensor_type.addItem("深度图", "depth")
        self.scan_body_pose_topic = QLineEdit("/dlio/odom_node/odom")
        self.scan_sensor_pose_topic = QLineEdit("/dlio/odom_node/odom")
        self.scan_cloud_topic = QLineEdit("/livox/lidar/pointcloud")
        self.scan_depth_topic = QLineEdit(
            "/camera/aligned_depth_to_color/image_raw")
        self.scan_goal_topic = QLineEdit("/move_base_simple/goal")
        self.scan_initial_path_topic = QLineEdit("/initial_path")
        self.scan_cmd_vel_topic = QLineEdit("/cmd_vel")
        self.scan_start_controller = QCheckBox("启动闭环速度控制器")
        self.scan_start_controller.setChecked(True)
        self.scan_cloud_is_world = QCheckBox("输入点云已经位于世界坐标系")
        self.scan_need_extrinsic = QCheckBox("应用 SCAN 内置传感器外参")
        self.scan_planner_params = QLineEdit()
        self.scan_controller_params = QLineEdit()
        self.scan_keypoints_file = QLineEdit()
        self.scan_reference_path_file = QLineEdit()
        form.addRow("导航模式", self.scan_navi_mode)
        form.addRow("感知类型", self.scan_sensor_type)
        form.addRow("机身里程计", self.scan_body_pose_topic)
        form.addRow("传感器位姿", self.scan_sensor_pose_topic)
        form.addRow("点云话题", self.scan_cloud_topic)
        form.addRow("深度图话题", self.scan_depth_topic)
        form.addRow("目标话题", self.scan_goal_topic)
        form.addRow("参考路径话题", self.scan_initial_path_topic)
        form.addRow("速度输出", self.scan_cmd_vel_topic)
        form.addRow("控制器", self.scan_start_controller)
        form.addRow("点云坐标", self.scan_cloud_is_world)
        form.addRow("传感器外参", self.scan_need_extrinsic)
        form.addRow("规划器参数", self._path_row(
            self.scan_planner_params,
            lambda: self._choose_yaml(
                self.scan_planner_params, "选择 SCAN-Planner 参数")))
        form.addRow("控制器参数", self._path_row(
            self.scan_controller_params,
            lambda: self._choose_yaml(
                self.scan_controller_params, "选择 SCAN 控制器参数")))
        form.addRow("模式 2 途径点", self._path_row(
            self.scan_keypoints_file,
            lambda: self._choose_yaml(
                self.scan_keypoints_file, "选择 SCAN 途径点")))
        form.addRow("模式 3 参考路径", self._path_row(
            self.scan_reference_path_file,
            lambda: self._choose_yaml(
                self.scan_reference_path_file, "选择 SCAN 参考路径")))
        return panel

    def _build_custom_navigation_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.custom_navigation_package = QLineEdit()
        self.custom_navigation_launch = QLineEdit()
        self.custom_navigation_arguments = QLineEdit()
        self.custom_navigation_arguments.setPlaceholderText(
            "map:={map} params_file:={params} use_sim_time:={use_sim_time}")
        form.addRow("ROS 包名", self.custom_navigation_package)
        form.addRow("Launch 文件", self.custom_navigation_launch)
        form.addRow("附加参数", self.custom_navigation_arguments)
        hint = QLabel("支持占位符：{map}、{params}、{use_sim_time}")
        hint.setObjectName("hint")
        form.addRow("", hint)
        return panel

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
        self.startup_interval = self._number(0.0, 10.0, 1.0)
        self.startup_interval.setSuffix(" s")
        self.sensor_timeout = self._number(1.0, 120.0, 12.0)
        self.sensor_timeout.setSuffix(" s")
        form.addRow("时间源", self.use_sim_time)
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

    @staticmethod
    def _scrollable(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(widget)
        return area

    def _update_localization_backend(self, index: int) -> None:
        self.localization_backend_stack.setCurrentIndex(index)
        key = self.localization_backend.itemData(index)
        backend = LOCALIZATION_BACKENDS.get(key)
        self.localization_backend_description.setText(
            backend.description if backend is not None else "")

    def _update_navigation_backend(self, index: int) -> None:
        self.navigation_backend_stack.setCurrentIndex(index)
        key = self.navigation_backend.itemData(index)
        backend = NAVIGATION_BACKENDS.get(key)
        self.navigation_backend_description.setText(
            backend.description if backend is not None else "")

    def _choose_navigation_map(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择导航地图", str(self.map_storage_dir),
            "Map Files (*.yaml *.yml *.pcd *.ply *.pgm);;All Files (*)")
        if path:
            self.navigation_map.setText(path)

    def _choose_localization_map(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择定位地图", str(self.map_storage_dir),
            "Map Files (*.pcd *.ply *.yaml *.yml *.pgm);;All Files (*)")
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

    def _choose_yaml(self, field: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, title, field.text() or str(self.map_storage_dir),
            "YAML (*.yaml *.yml);;All Files (*)")
        if path:
            field.setText(path)

    def configuration(self) -> NavigationStackConfig:
        return NavigationStackConfig(
            navigation_map=self.navigation_map.text().strip(),
            localization_map=self.localization_map.text().strip(),
            localization_backend=self.localization_backend.currentData(),
            navigation_backend=self.navigation_backend.currentData(),
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
            custom_localization_package=(
                self.custom_localization_package.text().strip()),
            custom_localization_launch=(
                self.custom_localization_launch.text().strip()),
            custom_localization_arguments=(
                self.custom_localization_arguments.text().strip()),
            custom_navigation_package=(
                self.custom_navigation_package.text().strip()),
            custom_navigation_launch=(
                self.custom_navigation_launch.text().strip()),
            custom_navigation_arguments=(
                self.custom_navigation_arguments.text().strip()),
            scan_navi_mode=self.scan_navi_mode.currentData(),
            scan_sensor_type=self.scan_sensor_type.currentData(),
            scan_body_pose_topic=self.scan_body_pose_topic.text().strip(),
            scan_sensor_pose_topic=self.scan_sensor_pose_topic.text().strip(),
            scan_cloud_topic=self.scan_cloud_topic.text().strip(),
            scan_depth_topic=self.scan_depth_topic.text().strip(),
            scan_goal_topic=self.scan_goal_topic.text().strip(),
            scan_initial_path_topic=(
                self.scan_initial_path_topic.text().strip()),
            scan_cmd_vel_topic=self.scan_cmd_vel_topic.text().strip(),
            scan_start_controller=self.scan_start_controller.isChecked(),
            scan_cloud_is_world=self.scan_cloud_is_world.isChecked(),
            scan_need_extrinsic=self.scan_need_extrinsic.isChecked(),
            scan_planner_params=self.scan_planner_params.text().strip(),
            scan_controller_params=self.scan_controller_params.text().strip(),
            scan_keypoints_file=self.scan_keypoints_file.text().strip(),
            scan_reference_path_file=(
                self.scan_reference_path_file.text().strip()),
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
            "disabled": ("已禁用", "#71828e"),
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
