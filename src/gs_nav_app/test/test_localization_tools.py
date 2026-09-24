import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QObject, pyqtSignal  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import gs_nav_app.features.localization_tools as tools_module  # noqa: E402
from gs_nav_app.features.localization_tools import (  # noqa: E402
    DEFAULT_GLOBAL_RELOCALIZE_SERVICE,
    LocalizationToolsController,
    scan_context_path,
    validate_scan_context_source,
)


class FakeProcess(QObject):
    log_received = pyqtSignal(str)
    state_changed = pyqtSignal(str)

    def __init__(self, _label, parent=None):
        super().__init__(parent)
        self.active = False
        self.started = []

    def start_launch(self, arguments):
        self.started.append(list(arguments))
        self.active = True
        self.state_changed.emit("running")
        return True

    def finish(self, state="stopped"):
        self.active = False
        self.state_changed.emit(state)

    def shutdown(self):
        self.active = False


class FakeRosAdapter:
    def __init__(self):
        self.calls = []
        self.done = None

    def trigger_global_relocalization(self, service_name, done):
        self.calls.append(service_name)
        self.done = done
        return True, f"已请求全局重定位：{service_name}"


def test_scan_context_source_requires_existing_pcd(tmp_path):
    valid, message = validate_scan_context_source("")
    assert not valid
    assert "定位 PCD" in message

    ply = tmp_path / "map.ply"
    ply.write_bytes(b"ply")
    valid, message = validate_scan_context_source(str(ply))
    assert not valid
    assert "PCD" in message

    pcd = tmp_path / "map.pcd"
    pcd.write_bytes(b"pcd")
    valid, resolved = validate_scan_context_source(str(pcd))
    assert valid
    assert resolved == str(pcd.resolve())
    assert scan_context_path(resolved) == Path(str(pcd.resolve()) + ".sc")


def test_controller_runs_generate_sc_tool_and_verifies_output(
    tmp_path, monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    monkeypatch.setattr(tools_module, "RosLaunchProcess", FakeProcess)
    pcd = tmp_path / "location.pcd"
    pcd.write_bytes(b"pcd")
    controller = LocalizationToolsController(FakeRosAdapter())
    states = []
    controller.operation_state_changed.connect(
        lambda operation, state: states.append((operation, state)))

    success, _message = controller.generate_scan_context(str(pcd))
    assert success
    assert controller.sc_process.started == [[
        "run", "localizer", "generate_sc_tool", str(pcd.resolve())]]

    scan_context_path(str(pcd)).write_bytes(b"scan-context")
    controller.sc_process.finish()
    assert states[-1] == ("scan_context", "success")
    assert ".pcd.sc" in controller.status


def test_controller_calls_configurable_global_relocalize_service(monkeypatch):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    monkeypatch.setattr(tools_module, "RosLaunchProcess", FakeProcess)
    adapter = FakeRosAdapter()
    controller = LocalizationToolsController(adapter)
    states = []
    controller.operation_state_changed.connect(
        lambda operation, state: states.append((operation, state)))

    success, _message = controller.trigger_global_relocalization(
        DEFAULT_GLOBAL_RELOCALIZE_SERVICE)
    assert success
    assert adapter.calls == [DEFAULT_GLOBAL_RELOCALIZE_SERVICE]
    assert controller.global_relocalize_pending

    adapter.done(True, "重定位任务已开始")
    assert not controller.global_relocalize_pending
    assert states[-1] == ("global_relocalize", "success")
    assert controller.status == "重定位任务已开始"
