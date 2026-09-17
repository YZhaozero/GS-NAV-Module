from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("show_window", default_value="true"),
        Node(
            package="gs_nav_app",
            executable="ar_nav_node",
            name="gs_ar_nav",
            output="screen",
            parameters=[{
                "demo_mode": True,
                "show_window": ParameterValue(
                    LaunchConfiguration("show_window"), value_type=bool),
                "projection_mode": "ground",
            }],
        ),
    ])
