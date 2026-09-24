"""Localizer maintenance tools used by the navigation desktop app.

This module deliberately keeps process execution and ROS service calls out of
the Qt page.  Additional localization backends can provide another adapter
without changing the page layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal

from ..ros_launch_process import RosLaunchProcess

try:
    from std_srvs.srv import Trigger
except ImportError:  # Keep the desktop UI importable before ROS is sourced.
    Trigger = None


DEFAULT_GLOBAL_RELOCALIZE_SERVICE = (
    "/localizer/localizer_node/global_relocalize"
)


def scan_context_path(pcd_path: str) -> Path:
    """Return the file produced by ``localizer generate_sc_tool``."""
    return Path(str(Path(pcd_path).expanduser().resolve()) + ".sc")


def validate_scan_context_source(pcd_path: str) -> Tuple[bool, str]:
    """Validate an SC source map without mutating it."""
    raw_path = str(pcd_path).strip()
    if not raw_path:
        return False, "请先在“地图”页选择定位 PCD 地图"
    path = Path(raw_path).expanduser()
    if path.suffix.lower() != ".pcd":
        return False, "Scan Context 只能从 PCD 点云地图生成"
    if not path.is_file():
        return False, f"定位地图不存在：{path}"
    return True, str(path.resolve())


class LocalizationRosAdapter:
    """ROS-facing implementation of localizer runtime commands."""

    def __init__(self, node) -> None:
        self.node = node
        self._trigger_clients: Dict[str, object] = {}

    def trigger_global_relocalization(
        self,
        service_name: str,
        done: Callable[[bool, str], None],
    ) -> Tuple[bool, str]:
        service_name = str(service_name).strip()
        if not service_name:
            return False, "全局重定位服务名不能为空"
        if Trigger is None:
            return False, "未找到 std_srvs/Trigger，请构建并 source ROS 工作空间"
        client = self._trigger_clients.get(service_name)
        if client is None:
            try:
                client = self.node.create_client(Trigger, service_name)
            except (RuntimeError, TypeError, ValueError) as exc:
                return False, f"创建全局重定位客户端失败：{exc}"
            self._trigger_clients[service_name] = client
        if (
            not client.service_is_ready()
            and not client.wait_for_service(timeout_sec=0.1)
        ):
            return False, f"服务尚未就绪：{service_name}"
        try:
            future = client.call_async(Trigger.Request())
        except (RuntimeError, TypeError, ValueError) as exc:
            return False, f"调用全局重定位服务失败：{exc}"
        future.add_done_callback(
            lambda result: self._finish_trigger(result, done))
        return True, f"已请求全局重定位：{service_name}"

    @staticmethod
    def _finish_trigger(future, done: Callable[[bool, str], None]) -> None:
        try:
            response = future.result()
            if response is None:
                done(False, "全局重定位服务没有返回结果")
                return
            message = str(response.message).strip()
            if response.success:
                done(True, message or "全局重定位已触发")
            else:
                done(False, message or "Localizer 拒绝了全局重定位请求")
        except Exception as exc:  # rclpy future transports errors here.
            done(False, f"全局重定位服务调用失败：{exc}")


class UnavailableLocalizationRosAdapter:
    """Fallback used by UI tests and non-ROS previews."""

    def trigger_global_relocalization(
        self,
        _service_name: str,
        _done: Callable[[bool, str], None],
    ) -> Tuple[bool, str]:
        return False, "当前节点未启用定位 ROS 接口"


class LocalizationToolsController(QObject):
    """Own SC generation and global-relocalization operation state."""

    log_received = pyqtSignal(str)
    operation_state_changed = pyqtSignal(str, str)
    status_changed = pyqtSignal(str)

    def __init__(self, ros_adapter, parent=None) -> None:
        super().__init__(parent)
        self.ros_adapter = ros_adapter
        self.sc_process = RosLaunchProcess("Scan Context 地图生成", self)
        self.sc_process.log_received.connect(self.log_received.emit)
        self.sc_process.state_changed.connect(self._on_sc_state_changed)
        self._expected_sc_path: Optional[Path] = None
        self._global_relocalize_pending = False
        self._shutting_down = False
        self.status = "可生成 SC 地图或触发全局重定位"

    @property
    def sc_generation_active(self) -> bool:
        return self.sc_process.active

    @property
    def global_relocalize_pending(self) -> bool:
        return self._global_relocalize_pending

    def generate_scan_context(self, pcd_path: str) -> Tuple[bool, str]:
        if self.sc_process.active:
            return False, "Scan Context 地图正在生成，请稍候"
        valid, value = validate_scan_context_source(pcd_path)
        if not valid:
            self._set_status(value)
            self._log(value)
            return False, value
        resolved_path = value
        self._expected_sc_path = scan_context_path(resolved_path)
        message = f"开始为 {Path(resolved_path).name} 生成 SC 地图"
        self._set_status(message)
        self._log(message)
        started = self.sc_process.start_launch([
            "run", "localizer", "generate_sc_tool", resolved_path,
        ])
        if not started:
            message = "Scan Context 生成工具未能启动"
            self._set_status(message)
            self._log(message)
            return False, message
        return True, message

    def trigger_global_relocalization(
        self, service_name: str = DEFAULT_GLOBAL_RELOCALIZE_SERVICE,
    ) -> Tuple[bool, str]:
        if self._global_relocalize_pending:
            return False, "全局重定位请求正在执行，请稍候"
        service_name = str(service_name).strip()
        self._global_relocalize_pending = True
        self.operation_state_changed.emit("global_relocalize", "starting")
        accepted, message = self.ros_adapter.trigger_global_relocalization(
            service_name, self._on_global_relocalize_done)
        if not accepted:
            self._global_relocalize_pending = False
            self.operation_state_changed.emit("global_relocalize", "error")
            self._set_status(message)
            self._log(message)
            return False, message
        self._set_status(message)
        self._log(message)
        return True, message

    def _on_sc_state_changed(self, state: str) -> None:
        if self._shutting_down:
            return
        if state in ("starting", "running"):
            self.operation_state_changed.emit("scan_context", state)
            return
        if state == "error":
            message = "SC 地图生成失败，请查看定位日志"
            self.operation_state_changed.emit("scan_context", "error")
        elif (
            state == "stopped"
            and self._expected_sc_path is not None
            and self._expected_sc_path.is_file()
            and self._expected_sc_path.stat().st_size > 0
        ):
            message = f"SC 地图已生成：{self._expected_sc_path}"
            self.operation_state_changed.emit("scan_context", "success")
        else:
            message = "SC 生成进程已结束，但没有找到有效的 .pcd.sc 文件"
            self.operation_state_changed.emit("scan_context", "error")
        self._set_status(message)
        self._log(message)

    def _on_global_relocalize_done(self, success: bool, message: str) -> None:
        if self._shutting_down:
            return
        self._global_relocalize_pending = False
        state = "success" if success else "error"
        self.operation_state_changed.emit("global_relocalize", state)
        self._set_status(message)
        self._log(message)

    def _set_status(self, message: str) -> None:
        self.status = str(message)
        self.status_changed.emit(self.status)

    def _log(self, message: str) -> None:
        self.log_received.emit(str(message).rstrip("\n") + "\n")

    def shutdown(self) -> None:
        self._shutting_down = True
        self.sc_process.shutdown()

