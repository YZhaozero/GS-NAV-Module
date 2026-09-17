import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Path  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from gs_nav_demo.qt_nav_node import (  # noqa: E402
    MapPanel,
    NavigationWindow,
    QtNavRosNode,
)
from gs_nav_demo.map_processing import GridMap, PointCloudMap  # noqa: E402


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
    assert panel._cloud_level_angle_degrees > 20.0
    assert np.array_equal(original, original_copy)
    assert np.array_equal(panel._cloud_source_points, original_copy)

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
