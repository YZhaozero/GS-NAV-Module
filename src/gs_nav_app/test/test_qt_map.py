import os
import time
from datetime import datetime
from pathlib import Path as FilePath

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Path  # noqa: E402
from PyQt5.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QMouseEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from gs_nav_app.qt_nav_node import (  # noqa: E402
    CameraPanel,
    MapPanel,
    NavigationWindow,
    QtNavRosNode,
)
from gs_nav_app.features.mapping import (  # noqa: E402
    MAPPING_BACKENDS,
    MappingRosAdapter,
    unique_map_path,
)
import gs_nav_app.features.mapping as mapping_feature  # noqa: E402
from gs_nav_app.features.sensors import SensorPreviewRosAdapter  # noqa: E402
from gs_nav_app.ros_launch_process import RosLaunchProcess  # noqa: E402
from gs_nav_app.map_processing import (  # noqa: E402
    GridMap,
    PointCloudMap,
    save_pcd,
)
from gs_nav_app.map_processing import pointcloud_to_grid  # noqa: E402


def test_navigation_yaml_matches_the_runtime_node_name():
    config_path = FilePath(__file__).parents[1] / "config" / "gs_nav.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert "/gs_qt_nav" in config
    parameters = config["/gs_qt_nav"]["ros__parameters"]
    assert parameters["camera_topic"] == "/camera/camera/color/image_raw"
    assert parameters["camera_info_topic"] == "/camera/camera/color/camera_info"
    assert parameters["path_topic"] == "/plan"
    assert "/global_plan" in parameters["path_topic_fallbacks"]
    assert parameters["fullscreen_on_small_screen"] is True


def test_window_switches_all_workspaces_to_480x800_portrait_layout():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"
        camera_topic = "/camera/camera/color/image_raw"

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    window = NavigationWindow(FakeNode())
    window.refresh_timer.stop()
    window.resize(480, 800)
    window.show()
    app.processEvents()

    assert window.minimumWidth() <= 480
    assert window.minimumHeight() <= 800
    assert window._compact_mode
    assert window.setup_splitter.orientation() == Qt.Vertical
    assert window.map_tools_splitter.orientation() == Qt.Vertical
    assert window.sensor_tools_splitter.orientation() == Qt.Vertical
    assert window.mapping_page.splitter.orientation() == Qt.Vertical
    assert window.navigation_stack_page.splitter.orientation() == Qt.Vertical
    assert window.map_tools_controls.maximumWidth() > 480
    assert window.camera_panel.maximumHeight() == 150

    window.pages.setCurrentWidget(window.active_page)
    app.processEvents()
    assert window.active_page.camera_panel.width() > (
        window.active_page.map_panel.width())
    assert window.active_page.camera_panel.height() > (
        window.active_page.map_panel.height())
    assert window.active_page.camera_panel._fill
    video_rect = window.active_page.camera_panel._video_rect()
    assert video_rect.width() >= window.active_page.camera_panel.width()
    assert video_rect.height() >= window.active_page.camera_panel.height()
    window.close()


def test_window_uses_compact_landscape_layout_for_rotated_800x480_display():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"
        camera_topic = "/camera/camera/color/image_raw"

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    window = NavigationWindow(FakeNode())
    window.refresh_timer.stop()
    window.resize(800, 480)
    window.show()
    app.processEvents()

    assert window.size().height() == 480
    assert window.minimumHeight() <= 480
    assert window.setup_page.minimumSizeHint().width() <= 800
    assert window._layout_profile == "compact_landscape"
    assert window.setup_splitter.orientation() == Qt.Horizontal
    assert window.map_tools_splitter.orientation() == Qt.Horizontal
    assert window.sensor_tools_splitter.orientation() == Qt.Horizontal
    assert window.mapping_page.splitter.orientation() == Qt.Horizontal
    assert window.navigation_stack_page.splitter.orientation() == Qt.Horizontal
    assert window.camera_panel.maximumHeight() == 86
    window.close()


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


def test_camera_panel_converts_each_ros_frame_only_once():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    panel = CameraPanel()
    first = np.zeros((48, 64, 3), dtype=np.uint8)
    second = np.full((48, 64, 3), 255, dtype=np.uint8)

    panel.set_frame(first, None, 0.0, frame_revision=1)
    first_key = panel._pixmap.cacheKey()
    panel.set_frame(second, None, 0.0, frame_revision=1)
    assert panel._pixmap.cacheKey() == first_key

    panel.set_frame(second, None, 0.0, frame_revision=2)
    assert panel._pixmap.cacheKey() != first_key


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


def test_ar_cloud_uses_source_intensity_colors_instead_of_height():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    panel = MapPanel(cloud_3d=True)
    points = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [2.0, 0.0, 1.0],
    ], dtype=np.float32)
    source_colors = np.array([
        [30, 30, 30], [220, 220, 220], [120, 120, 120],
    ], dtype=np.uint8)
    panel.set_pointcloud(
        points, colors=source_colors, color_source="intensity")
    assert np.array_equal(panel._cloud_colors, source_colors)
    assert "原始强度" in panel.cloud_display_description()


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


def test_sensor_topic_discovery_filters_supported_message_types():
    class FakeNode:
        def get_topic_names_and_types(self):
            return [
                ("/points", ["sensor_msgs/msg/PointCloud2"]),
                ("/livox/lidar", ["livox_ros_driver2/msg/CustomMsg"]),
                ("/color/image_raw", ["sensor_msgs/msg/Image"]),
                ("/imu/data", ["sensor_msgs/msg/Imu"]),
                ("/scan", ["sensor_msgs/msg/LaserScan"]),
            ]

    class Adapter:
        node = FakeNode()

    grouped = SensorPreviewRosAdapter.available_topics(Adapter())
    assert grouped == {
        "lidar": ["/livox/lidar", "/points"],
        "camera": ["/color/image_raw"],
        "imu": ["/imu/data"],
    }


def test_mapping_topic_discovery_only_accepts_dlio_message_types():
    class FakeNode:
        def get_topic_names_and_types(self):
            return [
                ("/points", ["sensor_msgs/msg/PointCloud2"]),
                ("/livox/custom", ["livox_ros_driver2/msg/CustomMsg"]),
                ("/imu/data", ["sensor_msgs/msg/Imu"]),
                ("/image", ["sensor_msgs/msg/Image"]),
            ]

    class Adapter:
        node = FakeNode()

    grouped = MappingRosAdapter.available_topics(Adapter())
    assert grouped == {
        "pointcloud": ["/points"],
        "imu": ["/imu/data"],
    }
    compatible, message = MappingRosAdapter.validate_inputs(
        Adapter(), "/livox/custom", "/imu/data")
    assert not compatible
    assert "PointCloud2" in message
    compatible, _message = MappingRosAdapter.validate_inputs(
        Adapter(), "/points", "/imu/data")
    assert compatible


def test_mapping_save_uses_unique_timestamped_file_and_verifies_output(
    tmp_path, monkeypatch,
):
    fixed_time = datetime(2026, 9, 18, 12, 34, 56, 789000)
    first = unique_map_path(tmp_path, "factory map", fixed_time)
    assert first.name == "factory_map_20260918_123456_789.pcd"
    first.write_bytes(b"existing map")
    second = unique_map_path(tmp_path, "factory map", fixed_time)
    assert second.name == "factory_map_20260918_123456_789_01.pcd"

    class FakeSavePCD:
        class Request:
            leaf_size = 0.0
            save_path = ""

    class Response:
        success = True

    class Future:
        def result(self):
            return Response()

        def add_done_callback(self, callback):
            callback(self)

    class Client:
        request = None

        def service_is_ready(self):
            return True

        def wait_for_service(self, timeout_sec):
            del timeout_sec
            return True

        def call_async(self, request):
            self.request = request
            FilePath(request.save_path).write_bytes(b"valid pcd data")
            return Future()

    monkeypatch.setattr(mapping_feature, "DlioSavePCD", FakeSavePCD)
    adapter = MappingRosAdapter.__new__(MappingRosAdapter)
    adapter.cloud = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    adapter.save_status = ""
    adapter.save_revision = 0
    adapter.save_in_progress = False
    adapter.last_saved_path = None
    adapter._pending_save_path = None
    adapter._dlio_save_client = Client()
    success, status = adapter.save_map(
        MAPPING_BACKENDS["dlio"], str(tmp_path), "factory map", 0.2)
    assert success
    assert adapter.last_saved_path is not None
    assert adapter.last_saved_path.is_file()
    assert adapter.last_saved_path.name.startswith("factory_map_")
    assert adapter.last_saved_path != first
    assert str(adapter.last_saved_path) in status
    assert adapter.save_revision == 1


def test_mapping_workspace_builds_dlio_launch_and_controls_preview():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"

        def __init__(self):
            self.mapping = FakeMappingAdapter()

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    class FakeMappingAdapter:
        cloud_revision = 0
        save_revision = 0
        cloud = None
        cloud_error = ""
        cloud_frame = ""
        cloud_topic = ""
        save_status = ""
        save_in_progress = False

        def __init__(self):
            self.preview_started = []
            self.preview_stopped = 0

        def available_topics(self):
            return {
                "pointcloud": ["/robot/lidar_points"],
                "imu": ["/robot/imu"],
            }

        def validate_inputs(self, pointcloud_topic, imu_topic):
            return True, f"{pointcloud_topic} + {imu_topic}"

        def start_preview(self, topic):
            self.preview_started.append(topic)
            self.cloud_topic = topic
            return True, f"等待建图数据 {topic}"

        def stop_preview(self):
            self.preview_stopped += 1

        def save_map(self, _backend, _path, _name, _leaf_size):
            return True, "saving"

    class FakeProcess:
        def __init__(self):
            self.started = []
            self.stopped = False

        def start_launch(self, arguments):
            self.started.append(arguments)
            return True

        def stop_launch(self):
            self.stopped = True

    node = FakeNode()
    window = NavigationWindow(node)
    window.refresh_timer.stop()
    window.show_mapping()
    assert window.pages.currentWidget() is window.mapping_page
    page = window.mapping_page
    assert page.map_panel._cloud_3d_enabled
    assert MAPPING_BACKENDS["dlio"].package == (
        "direct_lidar_inertial_odometry")
    assert page.pointcloud_combo.findText(
        "/robot/lidar_points") >= 0
    assert page.imu_combo.findText("/robot/imu") >= 0

    page.pointcloud_combo.setCurrentText("/robot/lidar_points")
    page.imu_combo.setCurrentText("/robot/imu")
    page.map_topic.setText("/robot/dlio/map")
    page.use_sim_time.setChecked(True)
    arguments = page.launch_arguments()
    assert arguments[:3] == [
        "launch", "direct_lidar_inertial_odometry", "dlio.launch.py"]
    assert "pointcloud_topic:=/robot/lidar_points" in arguments
    assert "imu_topic:=/robot/imu" in arguments
    assert "use_sim_time:=true" in arguments
    assert "rviz:=false" in arguments

    process = FakeProcess()
    window.mapping_controller.process = process
    page.start_mapping()
    assert node.mapping.preview_started[-1] == "/robot/dlio/map"
    assert process.started[-1] == arguments
    window.mapping_controller.stop()
    assert process.stopped


def test_livox_custom_message_is_converted_for_the_cloud_monitor():
    class Point:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class Message:
        points = [Point(1.0, 2.0, 3.0), Point(-1.0, 0.5, 0.25)]

    class FakeAdapter:
        cloud = None
        errors = {"lidar": "old error"}
        revisions = {"lidar": 4}

    adapter = FakeAdapter()
    SensorPreviewRosAdapter._on_livox(adapter, Message())
    assert adapter.cloud.shape == (2, 3)
    assert np.allclose(adapter.cloud[1], [-1.0, 0.5, 0.25])
    assert adapter.errors["lidar"] == ""
    assert adapter.revisions["lidar"] == 5


def test_sensor_monitor_page_selects_topics_and_renders_live_data():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakePreviewAdapter:
        def __init__(self):
            self.started = []
            self.stopped = []
            self.revisions = {"lidar": 0, "camera": 0, "imu": 0}
            self.topics = {"lidar": "", "camera": "", "imu": ""}
            self.errors = {"lidar": "", "camera": "", "imu": ""}
            self.cloud = None
            self.image = None
            self.imu = None

        def available_topics(self):
            return {
                "lidar": ["/livox/lidar"],
                "camera": ["/camera/camera/color/image_raw"],
                "imu": ["/camera/camera/imu"],
            }

        def start(self, kind, topic):
            self.started.append((kind, topic))
            self.topics[kind] = topic
            return True, f"正在订阅 {topic}"

        def stop(self, kind=None):
            self.stopped.append(kind)

    class FakeNode:
        navigation_status = "ready"
        camera_topic = "/camera/camera/color/image_raw"
        pointcloud_topic = ""

        def __init__(self):
            self.sensor_preview = FakePreviewAdapter()

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    node = FakeNode()
    window = NavigationWindow(node)
    window.refresh_timer.stop()
    window.show_sensor_tools()
    window.show_sensor_monitor()
    assert window.pages.currentWidget() is window.sensor_monitor_page
    assert window.sensor_monitor_tabs.count() == 3
    assert window.sensor_monitor_tabs.tabText(0) == "雷达点云"
    assert window.sensor_monitor_tabs.tabText(1) == "相机视频"
    assert window.sensor_monitor_tabs.tabText(2) == "IMU"
    assert window.sensor_topic_combos["lidar"].findText(
        "/livox/lidar") >= 0

    window.sensor_topic_combos["lidar"].setCurrentText("/livox/lidar")
    window._start_sensor_preview("lidar")
    assert node.sensor_preview.started[-1] == ("lidar", "/livox/lidar")
    node.sensor_preview.cloud = np.array([
        [0.0, 0.0, 0.0], [1.0, 2.0, 0.5],
    ], dtype=np.float32)
    node.sensor_preview.revisions["lidar"] += 1
    window._refresh_sensor_previews()
    assert len(window.sensor_preview_cloud_panel._cloud_source_points) == 2
    assert "2 点" in window.sensor_preview_status_labels["lidar"].text()

    window.sensor_topic_combos["imu"].setCurrentText("/camera/camera/imu")
    window._start_sensor_preview("imu")
    node.sensor_preview.imu = {
        "frame_id": "camera_imu_optical_frame",
        "stamp": (12, 345),
        "orientation": (0.0, 0.0, 0.0, 1.0),
        "angular_velocity": (0.1, 0.2, 0.3),
        "linear_acceleration": (1.0, 2.0, 9.8),
        "orientation_covariance": (0.0,) * 9,
        "angular_velocity_covariance": (0.0,) * 9,
        "linear_acceleration_covariance": (0.0,) * 9,
    }
    node.sensor_preview.revisions["imu"] += 1
    window._refresh_sensor_previews()
    imu_text = window.sensor_preview_imu_text.toPlainText()
    assert "camera_imu_optical_frame" in imu_text
    assert "Angular velocity" in imu_text
    assert "Linear acceleration" in imu_text

    window.leave_sensor_monitor()
    assert node.sensor_preview.stopped[-1] is None
    assert window.pages.currentWidget() is window.sensor_tools_page


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

    controller = RosLaunchProcess("测试传感器")
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
        camera_topic = "/camera/camera/color/image_raw"

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    window = NavigationWindow(FakeNode())
    window.refresh_timer.stop()
    assert window.convert_coordinates_combo.currentData() == "display"
    assert window.map_panel._cloud_3d_enabled
    assert window.camera_panel.maximumHeight() <= 260
    assert window.camera_panel._camera_topic == FakeNode.camera_topic
    assert not window.camera_panel._show_status
    assert window.active_page.camera_panel._show_status
    assert window.active_page.camera_panel._camera_topic == FakeNode.camera_topic
    assert window.active_page.map_panel._cloud_3d_enabled
    assert not window.active_page.map_panel.testAttribute(
        Qt.WA_TransparentForMouseEvents)
    assert window.pages.currentWidget() is window.setup_page
    assert window.setup_page.isAncestorOf(window.map_panel)
    assert window.setup_page.isAncestorOf(window.exit_app_button)
    assert window.exit_app_button.text() in ("退出", "退出程序")
    assert not window.setup_page.isAncestorOf(window.load_cloud_button)
    assert window.map_tools_page.isAncestorOf(window.load_cloud_button)
    assert window.open_navigation_stack_button.text() in ("导航", "导航系统")
    assert window.navigation_stack_page.log_tabs.count() == 8
    assert [
        window.navigation_stack_page.parameter_tabs.tabText(index)
        for index in range(window.navigation_stack_page.parameter_tabs.count())
    ] == ["定位", "导航", "DLIO", "LaserScan", "通用"]
    assert window.navigation_stack_page.parameter_tabs.widget(0).isAncestorOf(
        window.navigation_stack_page.localization_map)
    assert window.navigation_stack_page.parameter_tabs.widget(1).isAncestorOf(
        window.navigation_stack_page.navigation_map)
    assert window.navigation_stack_page.localization_backend.currentData() == (
        "pointcloud_localizer")
    assert window.navigation_stack_page.navigation_backend.currentData() == "nav2"
    assert window.navigation_stack_page.navigation_backend.findData(
        "scan_planner") >= 0
    assert not window.navigation_stack_page.nav2_rviz.isChecked()

    window.sensor_driver_controller.append_log("lidar", "雷达数据等待诊断\n")
    window.show_sensor_tools()
    assert "雷达数据等待诊断" in (
        window.sensor_log_views["lidar"].toPlainText())

    window.show_navigation_stack()
    assert window.pages.currentWidget() is window.navigation_stack_page
    assert "雷达数据等待诊断" in (
        window.navigation_stack_page.log_views["lidar"].toPlainText())
    window.show_navigation_setup()

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


def test_selected_localization_pcd_is_loaded_into_ar_navigation(tmp_path):
    app = QApplication.instance() or QApplication([])
    assert app is not None

    class FakeNode:
        navigation_status = "ready"
        camera_topic = "/camera/camera/color/image_raw"

        def send_navigation_waypoints(self, waypoints):
            return bool(waypoints)

        def cancel_navigation(self):
            pass

    points = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.5, 0.2], [2.0, 1.0, 0.4],
    ], dtype=np.float32)
    colors = np.array([
        [240, 20, 30], [10, 220, 40], [30, 40, 230],
    ], dtype=np.uint8)
    path = save_pcd(tmp_path / "localization_map.pcd", points, colors)
    window = NavigationWindow(FakeNode())
    window.refresh_timer.stop()

    window.navigation_stack_page.localization_map.setText(str(path))
    window.navigation_stack_page.localization_map.editingFinished.emit()

    assert window.local_cloud is not None
    assert window.local_cloud.path == path.resolve()
    assert np.array_equal(window.map_panel._cloud_source_colors, colors)
    assert np.array_equal(
        window.active_page.map_panel._cloud_source_colors, colors)
    assert window.map_panel.display_mode == "cloud"
    assert "AR 导航已加载定位地图" in window.map_status.text()


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


def test_nav2_plan_topic_is_preferred_over_legacy_global_plan():
    class FakeNode:
        navigation_active = True
        path_topic = "/plan"
        path_topics = ("/plan", "/global_plan")
        active_path_topic = ""
        path_message_count = 0
        map_frame = "map"

    def path_message(x: float) -> Path:
        message = Path()
        message.header.frame_id = "map"
        pose = PoseStamped()
        pose.pose.position.x = x
        message.poses.append(pose)
        return message

    node = FakeNode()
    QtNavRosNode._on_path(node, path_message(1.0), "/global_plan")
    assert node.active_path_topic == "/global_plan"
    assert node.path[0, 0] == 1.0

    QtNavRosNode._on_path(node, path_message(2.0), "/plan")
    assert node.active_path_topic == "/plan"
    assert node.path[0, 0] == 2.0

    QtNavRosNode._on_path(node, path_message(3.0), "/global_plan")
    assert node.active_path_topic == "/plan"
    assert node.path[0, 0] == 2.0
