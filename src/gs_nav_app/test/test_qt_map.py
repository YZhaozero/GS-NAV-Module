import os
import time
from pathlib import Path as FilePath

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Path  # noqa: E402
from PyQt5.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QMouseEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from gs_nav_app.qt_nav_node import (  # noqa: E402
    MapPanel,
    NavigationWindow,
    QtNavRosNode,
    SensorLaunchProcess,
)
from gs_nav_app.map_processing import GridMap, PointCloudMap  # noqa: E402
from gs_nav_app.map_processing import pointcloud_to_grid  # noqa: E402


def test_map_click_coordinate_round_trip():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    panel = MapPanel()
    panel.resize(500, 400)
    grid = np.zeros((100, 200), dtype=np.int8)
    origin = np.array([1.5, -2.0], dtype=np.float32)
    resolution = 0.05
    origin_yaw = 0.3
    panel.set_map(grid, origin, resolution, origin_yaw)

    local = np.array([3.0, 2.0])
    rotation = np.array([
        [np.cos(origin_yaw), -np.sin(origin_yaw)],
        [np.sin(origin_yaw), np.cos(origin_yaw)],
    ])
    world = origin + rotation @ local
    pixel = panel.world_to_widget(world)
    recovered = panel.widget_to_world(pixel)
    assert recovered is not None
    assert np.allclose(recovered, world, atol=resolution * 1.5)


def test_pointcloud_view_coordinate_round_trip():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    panel = MapPanel()
    panel.resize(500, 400)
    points = np.array([
        [-2.0, -1.0, 0.0],
        [3.0, 4.0, 1.0],
        [1.0, 2.0, 0.5],
    ], dtype=np.float32)
    panel.set_pointcloud(points)
    panel.set_display_mode("cloud")
    world = np.array([1.0, 2.0])
    recovered = panel.widget_to_world(panel.world_to_widget(world))
    assert recovered is not None
    assert np.allclose(recovered, world, atol=0.03)


def test_map_studio_cloud_has_independent_3d_camera_controls():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    panel = MapPanel(cloud_3d=True)
    panel.resize(800, 600)
    points = np.array([
        [-2.0, -1.0, 0.0],
        [3.0, 4.0, 2.0],
        [1.0, 2.0, 1.2],
        [2.0, -1.0, 0.6],
    ], dtype=np.float32)
    panel.set_pointcloud(points)
    panel.set_display_mode("cloud")

    before_x, before_y, _ = panel._project_cloud_points(points[[2]])
    panel._orbit_cloud_view(35.0, -18.0)
    after_x, after_y, _ = panel._project_cloud_points(points[[2]])
    assert not np.allclose([before_x, before_y], [after_x, after_y])

    target_before = panel._cloud_view_target.copy()
    panel._pan_cloud_view(20.0, -12.0)
    assert not np.allclose(panel._cloud_view_target, target_before)
    zoom_before = panel._cloud_zoom
    panel._zoom_cloud_view(1.0)
    assert panel._cloud_zoom > zoom_before
    panel.reset_cloud_view()
    assert panel._cloud_zoom == 1.0
    assert np.allclose(panel._cloud_view_target, panel._cloud_center)


def test_cloud_auto_level_is_display_only():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    panel = MapPanel(cloud_3d=True)
    x, y = np.meshgrid(np.linspace(-4.0, 4.0, 30), np.linspace(-2.0, 2.0, 20))
    original = np.column_stack((x.ravel(), y.ravel(), 0.45 * x.ravel())).astype(
        np.float32)
    original_copy = original.copy()
    panel.set_pointcloud(original)
    panel.set_cloud_display_options(
        "auto", "XYZ", (False, False, False), "perspective", "height")
    assert np.std(panel._cloud_points[:, 2]) < 1e-4
    assert abs(float(np.percentile(panel._cloud_points[:, 2], 2.0))) < 1e-4
    assert panel._cloud_level_angle_degrees > 20.0
    assert np.array_equal(original, original_copy)
    assert np.array_equal(panel._cloud_source_points, original_copy)
    assert np.allclose(panel.transform_cloud_points(original), panel._cloud_points)

    leveled_grid = pointcloud_to_grid(
        panel.transform_cloud_points(original), resolution=0.2,
        z_min=-0.2, z_max=0.2, padding=0.0, inflation_radius=0.0)
    assert leveled_grid.occupancy.shape[1] >= 40
    assert leveled_grid.occupancy.shape[0] >= 20

    panel.set_cloud_display_options(
        "original", "XZY", (True, False, False), "orthographic", "height")
    assert np.allclose(panel._cloud_points[:, 0], -original[:, 0])
    assert np.allclose(panel._cloud_points[:, 1], original[:, 2])
    assert np.allclose(panel._cloud_points[:, 2], original[:, 1])


def test_waypoints_switch_to_active_navigation_page():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"

        def __init__(self):
            self.sent = []
            self.cancelled = False

        def send_navigation_waypoints(self, waypoints):
            self.sent = [point.copy() for point in waypoints]
            return True

        def cancel_navigation(self):
            self.cancelled = True

    node = FakeNode()
    window = NavigationWindow(node)
    window.refresh_timer.stop()
    window.on_map_point(1.0, 2.0)
    window.on_map_point(3.0, 4.0)
    window.on_map_point(5.0, 6.0)
    window.start_navigation()
    assert len(node.sent) == 3
    assert window.pages.currentWidget() is window.active_page

    window.exit_navigation()
    assert node.cancelled
    assert window.pages.currentWidget() is window.setup_page
    assert window.waypoints == []


def test_individual_waypoint_can_be_reselected_and_deleted():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    window = NavigationWindow(FakeNode())
    window.refresh_timer.stop()
    window.on_map_point(1.0, 2.0)
    window.on_map_point(3.0, 4.0)
    window.on_map_point(5.0, 6.0)
    assert window.waypoint_list.count() == 3

    window.waypoint_list.setCurrentRow(1)
    window.begin_waypoint_reselection()
    assert window.waypoint_edit_index == 1
    assert not window.start_button.isEnabled()
    window.on_map_point(30.0, 40.0)
    assert len(window.waypoints) == 3
    assert np.allclose(window.waypoints[1], [30.0, 40.0])
    assert window.waypoint_list.currentRow() == 1
    assert "X 30.00" in window.waypoint_list.currentItem().text()

    window.waypoint_list.setCurrentRow(0)
    window.delete_selected_waypoint()
    assert len(window.waypoints) == 2
    assert np.allclose(window.waypoints[0], [30.0, 40.0])
    assert window.waypoint_list.count() == 2
    assert window.waypoint_list.item(0).text().startswith("1.")


def test_navigation_setup_cloud_is_draggable_and_clickable():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    window = NavigationWindow(FakeNode())
    window.refresh_timer.stop()
    panel = window.map_panel
    panel.resize(760, 560)
    x, y = np.meshgrid(np.linspace(-4.0, 4.0, 8), np.linspace(-3.0, 3.0, 8))
    panel.set_pointcloud(np.column_stack((
        x.ravel(), y.ravel(), np.zeros(x.size))), None)
    panel.set_display_mode("cloud")
    selected = []
    panel.point_selected.connect(lambda px, py: selected.append((px, py)))

    center = QPointF(panel.width() * 0.5, panel.height() * 0.5)
    panel.mousePressEvent(QMouseEvent(
        QEvent.MouseButtonPress, center, Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))
    panel.mouseReleaseEvent(QMouseEvent(
        QEvent.MouseButtonRelease, center, Qt.LeftButton,
        Qt.NoButton, Qt.NoModifier))
    assert len(selected) == 1

    azimuth_before = panel._cloud_azimuth
    panel.mousePressEvent(QMouseEvent(
        QEvent.MouseButtonPress, center, Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))
    moved = center + QPointF(60.0, 25.0)
    panel.mouseMoveEvent(QMouseEvent(
        QEvent.MouseMove, moved, Qt.NoButton,
        Qt.LeftButton, Qt.NoModifier))
    panel.mouseReleaseEvent(QMouseEvent(
        QEvent.MouseButtonRelease, moved, Qt.LeftButton,
        Qt.NoButton, Qt.NoModifier))
    assert panel._cloud_azimuth != azimuth_before
    assert len(selected) == 1


def test_successful_navigation_shows_transition_before_returning():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"
        navigation_result_revision = 0
        navigation_result_status = None

        def __init__(self):
            self.cancelled = False

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            self.cancelled = True

    node = FakeNode()
    window = NavigationWindow(node)
    window.refresh_timer.stop()
    window.on_map_point(1.0, 2.0)
    window.start_navigation()
    assert window.pages.currentWidget() is window.active_page

    node.navigation_result_status = 4
    node.navigation_result_revision += 1
    node.navigation_status = "导航成功"
    window._handle_navigation_terminal_state()

    assert window.pages.currentWidget() is window.active_page
    assert window.navigation_return_timer.isActive()
    assert window.pending_navigation_result_status == 4
    assert not window.active_page.result_overlay.isHidden()
    assert "已到达目的地" in window.active_page.result_overlay.text()
    assert window.active_page.exit_button.text() == "立即返回地图"
    assert len(window.waypoints) == 1

    window.exit_navigation()
    assert window.pages.currentWidget() is window.setup_page
    assert window.waypoints == []
    assert not node.cancelled
    assert node.navigation_status == "导航成功，已自动返回地图"


def test_sensor_manager_has_independent_page_and_launch_arguments():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    window = NavigationWindow(FakeNode())
    window.refresh_timer.stop()
    window.show_sensor_tools()
    assert window.pages.currentWidget() is window.sensor_tools_page

    lidar = window._lidar_launch_arguments()
    assert lidar[:3] == [
        "launch", "livox_ros_driver2", "msg_MID360_launch.py"]
    assert "xfer_format:=4" in lidar
    assert "publish_freq:=10" in lidar
    assert "frame_id:=livox_frame" in lidar

    window.lidar_xfer_format.setCurrentIndex(1)
    window.lidar_multi_topic.setChecked(True)
    window.lidar_config_path.setText("/tmp/MID360 test.json")
    lidar = window._lidar_launch_arguments()
    assert "xfer_format:=0" in lidar
    assert "multi_topic:=1" in lidar
    assert "user_config_path:=/tmp/MID360 test.json" in lidar

    camera = window._camera_launch_arguments()
    assert camera[:3] == [
        "launch", "realsense2_camera", "d435i.launch.py"]
    assert "enable_color:=true" in camera
    assert "enable_depth:=true" in camera
    assert "unite_imu_method:=2" in camera
    assert "pointcloud.enable:=false" in camera
    assert not any(value.startswith("serial_no:=") for value in camera)

    window.camera_serial_input.setText("123456")
    window.camera_pointcloud.setChecked(True)
    camera = window._camera_launch_arguments()
    assert "serial_no:=123456" in camera
    assert "pointcloud.enable:=true" in camera


def test_sensor_logs_are_split_and_process_state_updates_controls():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    window = NavigationWindow(FakeNode())
    window.refresh_timer.stop()
    window._append_sensor_log(
        "lidar", "\x1b[32mdriver ready\x1b[0m\npoint published\r\n")
    assert "driver ready" in window.sensor_log_views["lidar"].toPlainText()
    assert "\x1b" not in window.sensor_log_views["lidar"].toPlainText()
    assert "[雷达] driver ready" in window.sensor_log_views["all"].toPlainText()
    assert window.sensor_log_views["camera"].toPlainText() == ""

    window._handle_sensor_state("lidar", "running")
    assert not window.sensor_start_buttons["lidar"].isEnabled()
    assert window.sensor_stop_buttons["lidar"].isEnabled()
    assert "运行中" in window.sensor_state_labels["lidar"].text()
    window._handle_sensor_state("lidar", "stopped")
    assert window.sensor_start_buttons["lidar"].isEnabled()
    assert not window.sensor_stop_buttons["lidar"].isEnabled()

    window._clear_sensor_logs()
    assert window.sensor_log_views["all"].toPlainText() == ""
    assert window.sensor_log_views["lidar"].toPlainText() == ""


def test_sensor_stop_terminates_the_complete_launch_process_group(
    tmp_path, monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    child_pid_file = tmp_path / "child.pid"
    fake_ros2 = fake_bin / "ros2"
    fake_ros2.write_text(
        "#!/bin/sh\n"
        "sleep 60 &\n"
        "echo $! > \"$FAKE_SENSOR_CHILD_PID\"\n"
        "wait\n")
    fake_ros2.chmod(0o755)
    monkeypatch.setenv("FAKE_SENSOR_CHILD_PID", str(child_pid_file))
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    controller = SensorLaunchProcess("测试传感器")
    states = []
    controller.state_changed.connect(states.append)
    controller.start_launch(["launch", "fake_driver", "fake.launch.py"])

    def wait_until(predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        return False

    assert wait_until(child_pid_file.exists, 2.0)
    child_pid = int(child_pid_file.read_text().strip())
    assert FilePath(f"/proc/{child_pid}").exists()
    controller.stop_launch()
    assert wait_until(
        lambda: (
            not controller.running
            and not controller._group_alive()
            and states[-1] == "stopped"),
        5.0,
    )
    assert "stopping" in states
    if FilePath(f"/proc/{child_pid}/stat").exists():
        stat = FilePath(f"/proc/{child_pid}/stat").read_text()
        assert stat[stat.rfind(")") + 2:].split()[0] in ("Z", "X")


def test_navigation_and_map_processing_are_separate_workspaces():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    window = NavigationWindow(FakeNode())
    window.refresh_timer.stop()
    assert window.convert_coordinates_combo.currentData() == "display"
    assert window.map_panel._cloud_3d_enabled
    assert window.camera_panel.maximumHeight() == 260
    assert not window.camera_panel._show_status
    assert window.active_page.camera_panel._show_status
    assert window.active_page.map_panel._cloud_3d_enabled
    assert not window.active_page.map_panel.testAttribute(
        Qt.WA_TransparentForMouseEvents)
    assert window.pages.currentWidget() is window.setup_page
    assert window.setup_page.isAncestorOf(window.map_panel)
    assert not window.setup_page.isAncestorOf(window.load_cloud_button)
    assert window.map_tools_page.isAncestorOf(window.load_cloud_button)

    window.show_map_tools()
    assert window.pages.currentWidget() is window.map_tools_page
    window.set_navigation_map_mode("cloud")
    assert window.map_panel.display_mode == "cloud"
    assert window.active_page.map_panel.display_mode == "cloud"
    assert window.editor_map_panel.display_mode == "grid"

    window.set_editor_map_mode("cloud")
    window.set_navigation_map_mode("grid")
    assert window.editor_map_panel.display_mode == "cloud"
    assert window.map_panel.display_mode == "grid"
    window.show_navigation_setup()
    assert window.pages.currentWidget() is window.setup_page


def test_grid_conversion_follows_auto_leveled_coordinates():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"
        latest_pointcloud = None

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    x, y = np.meshgrid(
        np.linspace(-4.0, 4.0, 30), np.linspace(-2.0, 2.0, 20))
    points = np.column_stack(
        (x.ravel(), y.ravel(), 0.45 * x.ravel())).astype(np.float32)
    window = NavigationWindow(FakeNode())
    window.refresh_timer.stop()
    window._set_local_cloud(PointCloudMap(points), reset_original=True)
    window.grid_resolution_spin.setValue(0.2)
    window.z_min_spin.setValue(-0.2)
    window.z_max_spin.setValue(0.2)

    # Even a stale/manual original selection must not silently bypass the
    # active auto-level transform.
    window.convert_coordinates_combo.setCurrentIndex(
        window.convert_coordinates_combo.findData("original"))
    window.convert_pointcloud()

    assert window.convert_coordinates_combo.currentData() == "display"
    assert window.local_grid.occupancy.shape[1] >= 40
    assert window.local_grid.occupancy.shape[0] >= 20
    assert "地面 Z=0" in window.map_status.text()


def test_gaussian_splat_uses_supersplat_orientation_and_distinct_colors():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    points = np.array([
        [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 1.0, 0.5],
    ], dtype=np.float32)
    colors = np.array([[240, 20, 30], [10, 220, 40], [30, 40, 230]], dtype=np.uint8)
    cloud = PointCloudMap(
        points,
        colors=colors,
        splat_scales=np.full((3, 3), 0.1, dtype=np.float32),
        splat_rotations=np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (3, 1)),
        splat_opacities=np.full(3, 0.8, dtype=np.float32),
    )
    window = NavigationWindow(FakeNode())
    window.refresh_timer.stop()
    window._set_local_cloud(cloud, reset_original=True)
    assert window.cloud_alignment_combo.currentData() == "original"
    assert window.cloud_axis_combo.currentData() == "XZY"
    assert window.cloud_flip_x.isChecked()
    assert window.cloud_flip_y.isChecked()
    assert window.cloud_flip_z.isChecked()
    assert window.convert_coordinates_combo.currentData() == "display"
    assert window.editor_map_panel.cloud_has_rgb
    transformed = window.editor_map_panel.transform_cloud_points(points)
    assert np.allclose(transformed[:, 0], -points[:, 0])
    assert np.allclose(transformed[:, 1], -points[:, 2])
    assert np.allclose(transformed[:, 2], -points[:, 1])
    rgb = window.editor_map_panel._cloud_colors.copy()
    window.cloud_color_combo.setCurrentIndex(
        window.cloud_color_combo.findData("height"))
    assert not np.array_equal(window.editor_map_panel._cloud_colors, rgb)


def test_grid_brush_edit_and_undo():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    window = NavigationWindow(FakeNode())
    window.refresh_timer.stop()
    grid = GridMap(
        np.zeros((20, 20), dtype=np.int8), 0.1, np.array([0.0, 0.0, 0.0]))
    window._set_local_grid(grid, reset_original=True)
    window.snapshot_map_edit()
    window.edit_map_at("grid", "occupied", 1.0, 1.0, 0.2)
    assert np.count_nonzero(window.local_grid.occupancy == 100) > 1
    window.undo_map_edit()
    assert np.count_nonzero(window.local_grid.occupancy) == 0


def test_global_path_is_ignored_after_navigation_exit():
    class FakeNode:
        navigation_active = False
        path = np.ones((4, 3), dtype=np.float32)
        path_frame = "map"
        map_frame = "map"

        def clear_navigation_path(self):
            self.path = np.empty((0, 3), dtype=np.float32)
            self.path_frame = self.map_frame

    message = Path()
    message.header.frame_id = "map"
    message.poses.append(PoseStamped())
    node = FakeNode()
    QtNavRosNode._on_path(node, message)
    assert node.path.shape == (0, 3)
