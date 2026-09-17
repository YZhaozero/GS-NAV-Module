from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("gs_nav_demo"), "config", "gs_nav.yaml")
    return LaunchDescription([
        Node(
            package="gs_nav_demo",
            executable="qt_nav_node",
            name="gs_qt_nav",
            output="screen",
            parameters=[config],
        ),
    ])
