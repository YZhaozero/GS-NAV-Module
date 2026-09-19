from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_share = get_package_share_directory("gs_nav_app")
    config = os.path.join(package_share, "config", "gs_nav.yaml")
    default_pointcloud = os.path.join(
        package_share, "maps", "dilo_map.pcd")
    return LaunchDescription([
        DeclareLaunchArgument(
            "pointcloud_map_path",
            default_value=default_pointcloud,
            description="PCD/PLY loaded when the navigation UI starts; empty disables it",
        ),
        DeclareLaunchArgument(
            "fullscreen_on_small_screen",
            default_value="true",
            description="Use fullscreen mode automatically below 700 px width",
        ),
        Node(
            package="gs_nav_app",
            executable="qt_nav_node",
            name="gs_qt_nav",
            output="screen",
            parameters=[config, {
                "pointcloud_map_path": LaunchConfiguration(
                    "pointcloud_map_path"),
                "fullscreen_on_small_screen": LaunchConfiguration(
                    "fullscreen_on_small_screen"),
            }],
        ),
    ])
