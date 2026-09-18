import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QObject, pyqtSignal  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import gs_nav_app.features.navigation_stack as stack_module  # noqa: E402
from gs_nav_app.features.navigation_stack import (  # noqa: E402
    NavigationStackConfig,
    NavigationStackController,
    build_navigation_launches,
    validate_navigation_stack,
)


def make_config(tmp_path: Path, **changes) -> NavigationStackConfig:
    navigation_map = tmp_path / "navigation.yaml"
    localization_map = tmp_path / "localization.pcd"
    navigation_map.write_text("image: navigation.pgm\n")
    localization_map.write_bytes(b"pcd")
    values = {
        "navigation_map": str(navigation_map),
        "localization_map": str(localization_map),
        "lidar_arguments": [
            "launch", "livox_ros_driver2", "msg_MID360_launch.py",
            "xfer_format:=4",
        ],
        "camera_arguments": [
            "launch", "realsense2_camera", "d435i.launch.py",
        ],
        "startup_interval_ms": 0,
    }
    values.update(changes)
    return NavigationStackConfig(**values)


def test_complete_launch_arguments_include_both_map_types(tmp_path):
    config = make_config(tmp_path)
    valid, _message = validate_navigation_stack(config)
    assert valid
    launches = build_navigation_launches(config)
    assert launches["lidar"][-1] == "xfer_format:=4"
    assert "rviz:=false" in launches["dlio"]
    assert "use_sim_time:=false" in launches["dlio"]
    assert "cloud_topic:=/livox/lidar/pointcloud" in launches["laserscan"]
    assert f"map:={Path(config.localization_map).resolve()}" in (
        launches["localizer"])
    assert f"map:={Path(config.navigation_map).resolve()}" in launches["nav2"]
    assert "autostart:=true" in launches["nav2"]
    assert "rviz:=false" in launches["nav2"]


def test_invalid_or_swapped_maps_are_rejected(tmp_path):
    config = make_config(tmp_path)
    config.navigation_map = config.localization_map
    valid, message = validate_navigation_stack(config)
    assert not valid
    assert "YAML" in message


def test_scan_planner_can_run_without_map_or_localization_backend(tmp_path):
    config = make_config(
        tmp_path,
        navigation_map="",
        localization_map="",
        localization_backend="disabled",
        navigation_backend="scan_planner",
    )
    valid, message = validate_navigation_stack(config)
    assert valid, message
    launches = build_navigation_launches(config)
    assert "localizer" not in launches
    assert launches["nav2"][:3] == [
        "launch", "scan_planner", "run.launch.py"]
    assert "navi_mode:=1" in launches["nav2"]
    assert "cloud_topic:=/livox/lidar/pointcloud" in launches["nav2"]


def test_custom_backends_expand_resource_placeholders(tmp_path):
    config = make_config(
        tmp_path,
        localization_backend="custom",
        navigation_backend="custom",
        custom_localization_package="alternate_localizer",
        custom_localization_launch="start.launch.py",
        custom_localization_arguments="map:={map} clock:={use_sim_time}",
        custom_navigation_package="alternate_navigation",
        custom_navigation_launch="navigation.launch.py",
        custom_navigation_arguments="map:={map} params:={params}",
    )
    valid, message = validate_navigation_stack(config)
    assert valid, message
    launches = build_navigation_launches(config)
    assert launches["localizer"][:3] == [
        "launch", "alternate_localizer", "start.launch.py"]
    assert f"map:={config.localization_map}" in launches["localizer"]
    assert launches["nav2"][:3] == [
        "launch", "alternate_navigation", "navigation.launch.py"]


def test_ordered_start_skips_sensors_that_are_already_online(
    tmp_path, monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeProcess(QObject):
        log_received = pyqtSignal(str)
        state_changed = pyqtSignal(str)

        def __init__(self, label, parent=None):
            super().__init__(parent)
            self.label = label
            self.running = False
            self.started = []

        def start_launch(self, arguments):
            self.started.append(list(arguments))
            self.running = True
            self.state_changed.emit("running")
            return True

        def stop_launch(self):
            self.running = False
            self.state_changed.emit("stopped")

        def shutdown(self):
            self.running = False

    class FakeSensorController(QObject):
        log_received = pyqtSignal(str, str)
        state_changed = pyqtSignal(str, str)

        def __init__(self):
            super().__init__()
            self.processes = {
                "lidar": FakeProcess("lidar"),
                "camera": FakeProcess("camera"),
            }
            self.start_calls = []

        def start(self, key, arguments):
            self.start_calls.append((key, arguments))
            return self.processes[key].start_launch(arguments)

        def stop(self, key):
            self.processes[key].stop_launch()

    monkeypatch.setattr(stack_module, "RosLaunchProcess", FakeProcess)
    sensor = FakeSensorController()
    controller = NavigationStackController(
        sensor, sensor_active=lambda key: key in ("lidar", "camera"))
    states = []
    controller.overall_state_changed.connect(states.append)
    success, _status = controller.start(make_config(tmp_path))
    assert success
    for _index in range(20):
        app.processEvents()
        if controller.overall_state == "running":
            break

    assert sensor.start_calls == []
    assert controller.overall_state == "running"
    assert all(controller.processes[key].started for key in (
        "dlio", "laserscan", "localizer", "nav2"))
    assert states[0] == "starting"
    assert states[-1] == "running"


def test_sensor_process_is_not_running_until_real_data_arrives(
    tmp_path, monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeProcess(QObject):
        log_received = pyqtSignal(str)
        state_changed = pyqtSignal(str)

        def __init__(self, _label, parent=None):
            super().__init__(parent)
            self.running = False
            self.started = []

        def start_launch(self, arguments):
            self.started.append(list(arguments))
            self.running = True
            self.state_changed.emit("running")
            return True

        def stop_launch(self):
            self.running = False
            self.state_changed.emit("stopped")

        def shutdown(self):
            self.running = False

    class FakeSensorController(QObject):
        log_received = pyqtSignal(str, str)
        state_changed = pyqtSignal(str, str)

        def __init__(self):
            super().__init__()
            self.processes = {
                "lidar": FakeProcess("lidar"),
                "camera": FakeProcess("camera"),
            }

        def start(self, key, arguments):
            process = self.processes[key]
            process.started.append(list(arguments))
            process.running = True
            self.state_changed.emit(key, "running")
            return True

        def stop(self, key):
            self.processes[key].stop_launch()

    monkeypatch.setattr(stack_module, "RosLaunchProcess", FakeProcess)
    live_data = {"lidar": False, "camera": False}
    controller = NavigationStackController(
        FakeSensorController(), sensor_active=live_data.__getitem__)
    success, _status = controller.start(make_config(tmp_path))
    assert success
    assert controller.states["lidar"] == "waiting"
    assert controller.states["dlio"] == "stopped"

    live_data["lidar"] = True
    controller._poll_sensor_data("lidar")
    app.processEvents()
    assert controller.states["lidar"] == "running"
    assert controller.states["camera"] == "waiting"
    assert controller.states["dlio"] == "stopped"


def test_component_exit_during_start_aborts_remaining_queue(
    tmp_path, monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeProcess(QObject):
        log_received = pyqtSignal(str)
        state_changed = pyqtSignal(str)

        def __init__(self, _label, parent=None):
            super().__init__(parent)
            self.running = False

        def shutdown(self):
            pass

    class FakeSensorController(QObject):
        log_received = pyqtSignal(str, str)
        state_changed = pyqtSignal(str, str)

        def __init__(self):
            super().__init__()
            self.processes = {
                "lidar": FakeProcess("lidar"),
                "camera": FakeProcess("camera"),
            }

    monkeypatch.setattr(stack_module, "RosLaunchProcess", FakeProcess)
    controller = NavigationStackController(FakeSensorController())
    controller.overall_state = "starting"
    controller._owned_components.add("laserscan")
    controller._queue = ["localizer", "nav2"]
    controller._handle_state("laserscan", "error")
    assert controller.overall_state == "error"
    assert controller._queue == []


def test_stop_cleans_every_managed_process_even_when_not_owned_by_stack(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeProcess(QObject):
        log_received = pyqtSignal(str)
        state_changed = pyqtSignal(str)

        def __init__(self, _label, parent=None):
            super().__init__(parent)
            self.running = False
            self.stop_calls = 0

        @property
        def active(self):
            return self.running

        def stop_launch(self):
            self.stop_calls += 1
            self.running = False
            self.state_changed.emit("stopped")

        def shutdown(self):
            self.running = False

    class FakeSensorController(QObject):
        log_received = pyqtSignal(str, str)
        state_changed = pyqtSignal(str, str)

        def __init__(self):
            super().__init__()
            self.processes = {
                "lidar": FakeProcess("lidar"),
                "camera": FakeProcess("camera"),
            }

        def stop(self, key):
            self.processes[key].stop_launch()
            self.state_changed.emit(key, "stopped")

    monkeypatch.setattr(stack_module, "RosLaunchProcess", FakeProcess)
    sensor = FakeSensorController()
    controller = NavigationStackController(sensor)
    for process in (*sensor.processes.values(), *controller.processes.values()):
        process.running = True
    controller.overall_state = "error"

    controller.stop()
    app.processEvents()

    assert controller.overall_state == "stopped"
    assert all(
        process.stop_calls == 1
        for process in (*sensor.processes.values(), *controller.processes.values())
    )
    assert all(state == "stopped" for state in controller.states.values())
