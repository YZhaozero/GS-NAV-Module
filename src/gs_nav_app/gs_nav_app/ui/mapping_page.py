"""Standalone mapping page.

This module only renders state and forwards user actions to
``MappingController``. ROS messages, launch commands and service calls stay in
the feature layer.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
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
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..features.mapping import MAPPING_BACKENDS, MappingController
from ..map_storage import default_map_directory


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class MappingPage(QWidget):
    """Interactive view for one replaceable mapping controller."""

    return_requested = pyqtSignal()

    def __init__(
        self, controller: MappingController, map_panel_factory,
        map_storage_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.map_storage_dir = default_map_directory(str(
            map_storage_dir or ""))
        self.process_state = "stopped"
        self.last_cloud_revision = controller.ros.cloud_revision
        self.last_save_revision = controller.ros.save_revision
        self._map_panel_factory = map_panel_factory
        self._build_ui()
        self.controller.log_received.connect(self._append_log)
        self.controller.state_changed.connect(self._handle_state)
        self._backend_changed()
        self._handle_state("stopped")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(14)

        header = QVBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("GS MAPPING STUDIO")
        title.setObjectName("title")
        subtitle = QLabel("独立在线建图 · DLIO · 可扩展算法后端")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header_actions = QHBoxLayout()
        header_actions.addStretch(1)
        refresh_button = QPushButton("刷新输入话题")
        refresh_button.setObjectName("secondaryButton")
        refresh_button.clicked.connect(self.refresh_topics)
        header_actions.addWidget(refresh_button)
        back_button = QPushButton("返回 AR 导航")
        back_button.setObjectName("workspaceButton")
        back_button.clicked.connect(self.return_requested.emit)
        header_actions.addWidget(back_button)
        header.addLayout(header_actions)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        self.splitter = splitter
        map_card = QFrame()
        map_card.setObjectName("sidePanel")
        map_layout = QVBoxLayout(map_card)
        map_layout.setContentsMargins(14, 14, 14, 14)
        map_header = QHBoxLayout()
        map_title = QLabel("实时建图点云")
        map_title.setObjectName("sectionTitle")
        map_header.addWidget(map_title)
        map_header.addStretch(1)
        reset_view = QPushButton("重置三维视角")
        reset_view.setObjectName("secondaryButton")
        map_header.addWidget(reset_view)
        map_layout.addLayout(map_header)
        self.map_panel = self._map_panel_factory(cloud_3d=True)
        self.map_panel.setObjectName("mappingMapPanel")
        self.map_panel.set_display_mode("cloud")
        reset_view.clicked.connect(self.map_panel.reset_cloud_view)
        map_layout.addWidget(self.map_panel, 1)
        self.cloud_status = QLabel(
            "尚未开始建图 · 等待 PointCloud2 地图输出")
        self.cloud_status.setObjectName("statusBar")
        map_layout.addWidget(self.cloud_status)
        hint = QLabel("左键拖动旋转 · 右键/中键平移 · 滚轮缩放 · 双击恢复视角")
        hint.setObjectName("hint")
        map_layout.addWidget(hint)
        splitter.addWidget(map_card)

        controls = QFrame()
        controls.setObjectName("sidePanel")
        controls.setMinimumWidth(390)
        controls.setMaximumWidth(500)
        self.controls = controls
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(10)
        backend_title = QLabel("建图算法")
        backend_title.setObjectName("sectionTitle")
        controls_layout.addWidget(backend_title)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.algorithm_combo = QComboBox()
        for key, backend in MAPPING_BACKENDS.items():
            self.algorithm_combo.addItem(backend.label, key)
        self.algorithm_combo.currentIndexChanged.connect(
            self._backend_changed)
        form.addRow("算法", self.algorithm_combo)
        self.pointcloud_combo = self._editable_combo(
            "选择或输入 PointCloud2 话题")
        self.imu_combo = self._editable_combo("选择或输入 IMU 话题")
        self.map_topic = QLineEdit()
        form.addRow("雷达 PointCloud2", self.pointcloud_combo)
        form.addRow("IMU", self.imu_combo)
        form.addRow("地图输出", self.map_topic)
        controls_layout.addLayout(form)
        self.backend_description = QLabel()
        self.backend_description.setObjectName("hint")
        self.backend_description.setWordWrap(True)
        controls_layout.addWidget(self.backend_description)

        options_row = QHBoxLayout()
        self.use_sim_time = QCheckBox("使用仿真时钟")
        self.open_rviz = QCheckBox("同时启动 RViz")
        options_row.addWidget(self.use_sim_time)
        options_row.addWidget(self.open_rviz)
        options_row.addStretch(1)
        controls_layout.addLayout(options_row)

        action_row = QHBoxLayout()
        self.state_label = QLabel("● 未启动")
        self.state_label.setObjectName("sensorStatus")
        action_row.addWidget(self.state_label)
        action_row.addStretch(1)
        self.stop_button = QPushButton("停止建图")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.clicked.connect(self.controller.stop)
        self.start_button = QPushButton("开始建图")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_mapping)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.start_button)
        controls_layout.addLayout(action_row)

        save_title = QLabel("保存点云地图")
        save_title.setObjectName("sectionTitle")
        controls_layout.addWidget(save_title)
        save_form = QFormLayout()
        save_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.map_name = QLineEdit("gs_map")
        self.map_name.setPlaceholderText("例如：factory_floor_1")
        save_form.addRow("地图名称", self.map_name)
        save_row = QWidget()
        save_row_layout = QHBoxLayout(save_row)
        save_row_layout.setContentsMargins(0, 0, 0, 0)
        self.save_path = QLineEdit(str(self.map_storage_dir))
        choose_save_path = QPushButton("选择")
        choose_save_path.setObjectName("secondaryButton")
        choose_save_path.clicked.connect(self._choose_save_path)
        save_row_layout.addWidget(self.save_path, 1)
        save_row_layout.addWidget(choose_save_path)
        save_form.addRow("保存目录", save_row)
        self.leaf_size = QDoubleSpinBox()
        self.leaf_size.setRange(0.01, 2.0)
        self.leaf_size.setDecimals(2)
        self.leaf_size.setSingleStep(0.05)
        self.leaf_size.setValue(0.20)
        self.leaf_size.setSuffix(" m")
        save_form.addRow("降采样体素", self.leaf_size)
        controls_layout.addLayout(save_form)
        self.save_button = QPushButton("保存当前 PCD")
        self.save_button.setObjectName("primaryButton")
        self.save_button.clicked.connect(self.save_mapping_map)
        controls_layout.addWidget(self.save_button)
        self.save_status = QLabel()
        self.save_status.setObjectName("hint")
        self.save_status.setWordWrap(True)
        controls_layout.addWidget(self.save_status)

        log_header = QHBoxLayout()
        log_title = QLabel("建图日志")
        log_title.setObjectName("sectionTitle")
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        clear_log = QPushButton("清空")
        clear_log.setObjectName("secondaryButton")
        clear_log.clicked.connect(self.log_view_clear)
        log_header.addWidget(clear_log)
        controls_layout.addLayout(log_header)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("sensorLog")
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log_view.document().setMaximumBlockCount(3000)
        controls_layout.addWidget(self.log_view, 1)
        controls_scroll = QScrollArea()
        controls_scroll.setObjectName("transparentScroll")
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)
        splitter.setSizes([1000, 440])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        layout.addWidget(splitter, 1)

    def set_compact_mode(self, compact: bool) -> None:
        """Use a vertically scrollable layout on narrow portrait screens."""
        compact = bool(compact)
        self.layout().setContentsMargins(
            *(8, 6, 8, 8) if compact else (22, 18, 22, 22))
        self.layout().setSpacing(7 if compact else 14)
        self.splitter.setOrientation(Qt.Vertical if compact else Qt.Horizontal)
        self.controls.setMinimumWidth(0 if compact else 390)
        self.controls.setMaximumWidth(16777215 if compact else 500)
        self.map_panel.setMinimumSize(
            180 if compact else 260, 170 if compact else 200)
        self.splitter.setSizes([350, 300] if compact else [1000, 440])
        for label in self.findChildren(QLabel):
            if label.objectName() == "subtitle":
                label.setVisible(not compact)

    @staticmethod
    def _editable_combo(placeholder: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.lineEdit().setPlaceholderText(placeholder)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return combo

    def activate(self) -> None:
        self.refresh_topics()

    def _backend_changed(self, *_args) -> None:
        key = str(self.algorithm_combo.currentData() or "dlio")
        try:
            backend = self.controller.select_backend(key)
        except RuntimeError as exc:
            self.cloud_status.setText(str(exc))
            return
        self.backend_description.setText(
            backend.description
            + "。新增算法时只需注册 launch、参数映射、地图话题和保存适配器。")
        self.pointcloud_combo.setCurrentText(
            backend.default_pointcloud_topic)
        self.imu_combo.setCurrentText(backend.default_imu_topic)
        self.map_topic.setText(backend.default_map_topic)
        self.save_status.setText(
            f"地图保存服务：{backend.save_service or '未配置'}；"
            "文件名会自动附加时间，不会覆盖旧地图")

    def refresh_topics(self) -> None:
        try:
            grouped = self.controller.available_topics()
        except (RuntimeError, TypeError, ValueError) as exc:
            self.cloud_status.setText(f"读取建图输入话题失败：{exc}")
            return
        backend = self.controller.backend
        selectors = (
            (self.pointcloud_combo, grouped.get("pointcloud", []),
             backend.default_pointcloud_topic),
            (self.imu_combo, grouped.get("imu", []),
             backend.default_imu_topic),
        )
        for combo, discovered_topics, default in selectors:
            current = combo.currentText().strip()
            candidates = list(discovered_topics)
            for candidate in (current, default):
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(sorted(candidates))
            combo.setCurrentText(current or default)
            combo.blockSignals(False)
        self.cloud_status.setText(
            f"发现 {len(grouped.get('pointcloud', []))} 个 PointCloud2、"
            f"{len(grouped.get('imu', []))} 个 IMU 话题")

    def launch_arguments(self):
        return self.controller.build_launch_arguments(
            self.pointcloud_combo.currentText(),
            self.imu_combo.currentText(),
            self.open_rviz.isChecked(),
            self.use_sim_time.isChecked(),
        )

    def start_mapping(self) -> None:
        self.map_panel.set_pointcloud(np.empty((0, 3), dtype=np.float32))
        self.map_panel.set_display_mode("cloud")
        success, status = self.controller.start(
            self.pointcloud_combo.currentText(),
            self.imu_combo.currentText(),
            self.map_topic.text(),
            self.open_rviz.isChecked(),
            self.use_sim_time.isChecked(),
        )
        self.cloud_status.setText(status)
        if not success:
            self._append_log(status + "\n")

    def save_mapping_map(self) -> None:
        success, status = self.controller.save_map(
            self.save_path.text().strip(),
            self.map_name.text().strip(),
            self.leaf_size.value(),
        )
        self.save_status.setText(status)
        if success:
            self.save_button.setEnabled(False)
        else:
            self._append_log(status + "\n")

    def refresh(self) -> None:
        ros = self.controller.ros
        if ros.cloud_error:
            self.cloud_status.setText(ros.cloud_error)
        if ros.cloud_revision != self.last_cloud_revision:
            self.last_cloud_revision = ros.cloud_revision
            if ros.cloud is not None:
                self.map_panel.set_pointcloud(ros.cloud)
                self.map_panel.set_display_mode("cloud")
                self.cloud_status.setText(
                    f"实时地图 {ros.cloud_topic} · {len(ros.cloud):,} 点 · "
                    f"frame={ros.cloud_frame or '(empty)'}")
        if ros.save_revision != self.last_save_revision:
            self.last_save_revision = ros.save_revision
            if ros.save_status:
                self.save_status.setText(ros.save_status)
                self._append_log(ros.save_status + "\n")
            self.save_button.setEnabled(self.process_state == "running")

    def _handle_state(self, state: str) -> None:
        self.process_state = state
        labels = {
            "starting": ("启动中…", "#ffd166"),
            "running": ("建图中", "#66e09a"),
            "stopping": ("停止中…", "#ffd166"),
            "stopped": ("未启动", "#91a2ad"),
            "error": ("启动失败", "#ff7f88"),
        }
        text, color = labels.get(state, (state, "#91a2ad"))
        self.state_label.setText(f"● {text}")
        self.state_label.setStyleSheet(
            f"color: {color}; font-weight: 700;")
        active = state in ("starting", "running", "stopping")
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(state in ("starting", "running"))
        self.save_button.setEnabled(
            state == "running" and not self.controller.ros.save_in_progress)
        self.algorithm_combo.setEnabled(not active)
        if state in ("stopped", "error") and self.controller.ros.cloud is not None:
            self.cloud_status.setText(
                "建图已停止 · 保留最后一帧地图，可继续查看")

    def _append_log(self, text: str) -> None:
        clean = ANSI_ESCAPE_RE.sub("", text).replace("\r", "").rstrip("\n")
        if clean:
            self.log_view.appendPlainText(clean)

    def log_view_clear(self) -> None:
        self.log_view.clear()

    def _choose_save_path(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "选择地图保存目录", self.save_path.text())
        if selected:
            self.save_path.setText(selected)
