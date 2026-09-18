"""Launch SCAN-Planner against real odometry and perception topics."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(value):
    return value.lower() in ("1", "true", "yes", "on")


def _required_file(path, description):
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"{description} must reference an existing file: {path!r}")
    return path


def _setup(context):
    scan_share = get_package_share_directory("scan_planner")
    planner_params_file = LaunchConfiguration("planner_params_file").perform(context)
    controller_params_file = LaunchConfiguration("controller_params_file").perform(context)
    keypoints_file = LaunchConfiguration("keypoints_file").perform(context)
    reference_path_file = LaunchConfiguration("reference_path_file").perform(context)
    sensor_type = LaunchConfiguration("sensor_type").perform(context)
    navi_mode = int(LaunchConfiguration("navi_mode").perform(context))
    start_controller = _as_bool(LaunchConfiguration("start_controller").perform(context))

    if not planner_params_file:
        planner_params_file = os.path.join(scan_share, "config", "planner.yaml")
    if not controller_params_file:
        controller_params_file = os.path.join(scan_share, "config", "controllers.yaml")
    _required_file(planner_params_file, "planner_params_file")
    if start_controller:
        _required_file(controller_params_file, "controller_params_file")

    if sensor_type not in ("lidar", "depth"):
        raise RuntimeError("sensor_type must be 'lidar' or 'depth'")
    if navi_mode not in (1, 2, 3):
        raise RuntimeError("navi_mode must be 1, 2, or 3")
    if navi_mode == 2:
        _required_file(keypoints_file, "keypoints_file for navi_mode=2")
    if reference_path_file:
        if navi_mode != 3:
            raise RuntimeError("reference_path_file is only valid when navi_mode=3")
        _required_file(reference_path_file, "reference_path_file")

    body_pose_topic = LaunchConfiguration("body_pose_topic").perform(context)
    sensor_pose_topic = LaunchConfiguration("sensor_pose_topic").perform(context)
    cloud_topic = LaunchConfiguration("cloud_topic").perform(context)
    depth_topic = LaunchConfiguration("depth_topic").perform(context)
    goal_topic = LaunchConfiguration("goal_topic").perform(context)
    initial_path_topic = LaunchConfiguration("initial_path_topic").perform(context)
    bspline_topic = LaunchConfiguration("bspline_topic").perform(context)
    data_display_topic = LaunchConfiguration("data_display_topic").perform(context)
    execution_frozen_topic = LaunchConfiguration("execution_frozen_topic").perform(context)
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic").perform(context)

    planner_parameters = [
        planner_params_file,
        *([keypoints_file] if keypoints_file else []),
        {
            "fsm.navi_mode": navi_mode,
            "grid_map.sensor_type": sensor_type,
            "grid_map.cloud_is_world": _as_bool(
                LaunchConfiguration("cloud_is_world").perform(context)
            ),
            "grid_map.need_extrinsic": _as_bool(
                LaunchConfiguration("need_extrinsic").perform(context)
            ),
        },
    ]

    actions = [
        Node(
            package="scan_planner",
            executable="scan_planner_node",
            name="scan_planner_node",
            output="screen",
            parameters=planner_parameters,
            remappings=[
                ("body_pose", body_pose_topic),
                ("sensor_pose", sensor_pose_topic),
                ("cloud", cloud_topic),
                ("depth", depth_topic),
                ("move_base_simple/goal", goal_topic),
                ("initial_path", initial_path_topic),
                ("planning/bspline", bspline_topic),
                ("planning/data_display", data_display_topic),
                ("planning/execution_frozen", execution_frozen_topic),
            ],
        )
    ]

    if start_controller:
        actions.append(
            Node(
                package="scan_planner",
                executable="closed_loop_controller",
                name="closed_loop_controller",
                output="screen",
                parameters=[controller_params_file],
                remappings=[
                    ("body_pose", body_pose_topic),
                    ("planning/bspline", bspline_topic),
                    ("planning/execution_frozen", execution_frozen_topic),
                    ("cmd_vel", cmd_vel_topic),
                ],
            )
        )

    if reference_path_file:
        actions.append(
            Node(
                package="scan_planner",
                executable="reference_path_publisher.py",
                name="reference_path_publisher",
                output="screen",
                parameters=[reference_path_file],
                remappings=[
                    ("body_pose", body_pose_topic),
                    ("initial_path", initial_path_topic),
                ],
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "planner_params_file", default_value="",
                description="Planner parameter YAML; empty selects the packaged default",
            ),
            DeclareLaunchArgument(
                "controller_params_file", default_value="",
                description="Closed-loop controller YAML; empty selects the packaged default",
            ),
            DeclareLaunchArgument(
                "navi_mode", default_value="1",
                description="Goal source: 1=single goal, 2=parameter waypoints, 3=reference path",
            ),
            DeclareLaunchArgument(
                "sensor_type", default_value="lidar",
                description="Occupancy input type: lidar or depth",
            ),
            DeclareLaunchArgument(
                "keypoints_file", default_value="",
                description="Waypoint parameter YAML required by navi_mode=2",
            ),
            DeclareLaunchArgument(
                "reference_path_file", default_value="",
                description="Optional fixed reference-path YAML for navi_mode=3",
            ),
            DeclareLaunchArgument(
                "start_controller", default_value="true",
                description="Start the B-spline to cmd_vel closed-loop adapter",
            ),
            DeclareLaunchArgument(
                "cloud_is_world", default_value="false",
                description="Whether incoming lidar points are already in grid_map.frame_id",
            ),
            DeclareLaunchArgument(
                "need_extrinsic", default_value="true",
                description="Apply the built-in sensor extrinsic to sensor_pose",
            ),
            DeclareLaunchArgument(
                "body_pose_topic", default_value="/LIO/odom_vehicle",
                description="Robot body odometry input",
            ),
            DeclareLaunchArgument(
                "sensor_pose_topic", default_value="/LIO/odom_imu",
                description="Lidar/camera pose input",
            ),
            DeclareLaunchArgument(
                "cloud_topic", default_value="/LIO/clouds_lidar",
                description="PointCloud2 input used in lidar mode",
            ),
            DeclareLaunchArgument(
                "depth_topic", default_value="/camera/aligned_depth_to_color/image_raw",
                description="Depth image input used in depth mode",
            ),
            DeclareLaunchArgument(
                "goal_topic", default_value="/move_base_simple/goal",
                description="PoseStamped goal input used by navi_mode=1",
            ),
            DeclareLaunchArgument(
                "initial_path_topic", default_value="/initial_path",
                description="Path input used by navi_mode=3",
            ),
            DeclareLaunchArgument(
                "bspline_topic", default_value="/planning/bspline",
                description="Planned B-spline output",
            ),
            DeclareLaunchArgument(
                "data_display_topic", default_value="/planning/data_display",
                description="Planner diagnostic heartbeat output",
            ),
            DeclareLaunchArgument(
                "execution_frozen_topic", default_value="/planning/execution_frozen",
                description="Trajectory-clock freeze feedback from the controller",
            ),
            DeclareLaunchArgument(
                "cmd_vel_topic", default_value="/cmd_vel",
                description="Twist output from the optional closed-loop controller",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
