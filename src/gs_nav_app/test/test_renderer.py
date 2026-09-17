import numpy as np

from gs_nav_app.renderer import (
    RenderStyle,
    draw_route,
    make_ribbon,
    occupancy_to_bgr,
    project_ground,
    synthetic_camera_frame,
    waiting_camera_frame,
)


def test_ribbon_has_requested_width():
    path = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
    left, right = make_ribbon(path, 1.0)
    assert np.allclose(np.linalg.norm(left[:, :2] - right[:, :2], axis=1), 1.0)


def test_ground_projection_converges_on_horizon():
    style = RenderStyle()
    points = np.array([[1.0, 0.0, 0.0], [12.0, 0.0, 0.0]], dtype=np.float32)
    pixels, valid = project_ground(points, (720, 1280, 3), style)
    assert valid.all()
    assert pixels[1, 1] < pixels[0, 1]


def test_route_changes_camera_frame():
    image = synthetic_camera_frame(640, 360)
    before = image.copy()
    path = np.column_stack(
        (np.linspace(1.0, 14.0, 25), np.zeros(25), np.zeros(25)))
    left, right = make_ribbon(path, 0.9)
    center_px, valid = project_ground(path, image.shape, RenderStyle())
    left_px, _ = project_ground(left, image.shape, RenderStyle())
    right_px, _ = project_ground(right, image.shape, RenderStyle())
    draw_route(image, center_px, left_px, right_px, valid, RenderStyle())
    assert np.count_nonzero(image != before) > 1000


def test_waiting_camera_frame_has_requested_size():
    image = waiting_camera_frame("/camera/image_raw", 800, 450)
    assert image.shape == (450, 800, 3)
    assert image.mean() > 10


def test_occupancy_grid_rendering():
    grid = np.zeros((80, 120), dtype=np.int8)
    grid[:, 0] = 100
    grid[20:30, 30:70] = 100
    rendered = occupancy_to_bgr(grid)
    assert rendered.shape == (80, 120, 3)
    assert rendered[25, 40, 0] < rendered[40, 40, 0]
