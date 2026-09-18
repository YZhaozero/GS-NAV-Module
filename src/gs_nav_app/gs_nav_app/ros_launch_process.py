"""Reusable lifecycle wrapper for a complete ``ros2 launch`` process tree."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QObject, QProcess, QTimer, pyqtSignal


class RosLaunchProcess(QObject):
    """Start, log and reliably stop one ROS 2 launch process group."""

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

    @property
    def active(self) -> bool:
        """Include orphaned descendants whose launch parent already exited."""
        return self.running or self._group_alive()

    def start_launch(self, arguments) -> bool:
        if self.running:
            self.log_received.emit(f"{self.display_name}已经在运行\n")
            return False
        command = "ros2 " + " ".join(str(value) for value in arguments)
        self.log_received.emit(f"$ {command}\n")
        self._requested_stop = False
        self._process_group_id = None
        self.state_changed.emit("starting")
        self.process.start(
            "setsid", ["ros2", *[str(value) for value in arguments]])
        return True

    def stop_launch(self) -> None:
        if not self.active:
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
                    stat = Path(entry.path, "stat").read_text()
                    fields = stat[stat.rfind(")") + 2:].split()
                    if len(fields) > 2 and int(fields[2]) == group_id:
                        group_seen = True
                        if fields[0] not in ("Z", "X"):
                            return True
                except (OSError, ValueError):
                    continue
            if group_seen:
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
        self.state_changed.emit("stopped" if exit_code == 0 else "error")

    def _on_error(self, error) -> None:
        if self._requested_stop and error == QProcess.Crashed:
            return
        self.log_received.emit(
            f"{self.display_name}进程错误：{self.process.errorString()} ({error})\n")
        self.state_changed.emit("error")


# Compatibility name for code and downstream tests written before the module
# was split. New features should use RosLaunchProcess directly.
SensorLaunchProcess = RosLaunchProcess
