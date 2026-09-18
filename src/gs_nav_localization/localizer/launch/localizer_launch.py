import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config_path = PathJoinSubstitution(
        [FindPackageShare("localizer"), "config", "localizer.yaml"]
    )

    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path", default_value=default_config_path,
                description="Localizer YAML configuration file",
            ),
            DeclareLaunchArgument(
                "map", default_value="",
                description="PCD localization map; empty uses localizer.yaml",
            ),
            DeclareLaunchArgument(
                "cloud_topic", default_value="",
                description="Input cloud override; empty uses localizer.yaml",
            ),
            DeclareLaunchArgument(
                "odom_topic", default_value="",
                description="Odometry override; empty uses localizer.yaml",
            ),
            launch_ros.actions.Node(
                package="localizer",
                namespace="localizer",
                executable="localizer_node",
                name="localizer_node",
                output="screen",
                parameters=[
                    {
                        "config_path": LaunchConfiguration("config_path"),
                        "default_map_path": ParameterValue(
                            LaunchConfiguration("map"), value_type=str),
                        "cloud_topic": ParameterValue(
                            LaunchConfiguration("cloud_topic"), value_type=str),
                        "odom_topic": ParameterValue(
                            LaunchConfiguration("odom_topic"), value_type=str),
                    }
                ],
            ),

        ]
    )
