from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

package_share = get_package_share_directory('livox_ros_driver2')
user_config_path = join(package_share, 'config', 'MID360_config.json')

def generate_launch_description():
    livox_driver = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[{
            "xfer_format": ParameterValue(
                LaunchConfiguration("xfer_format"), value_type=int),
            "multi_topic": ParameterValue(
                LaunchConfiguration("multi_topic"), value_type=int),
            "publish_freq": ParameterValue(
                LaunchConfiguration("publish_freq"), value_type=float),
            "frame_id": LaunchConfiguration("frame_id"),
            "user_config_path": LaunchConfiguration("user_config_path"),
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument("xfer_format", default_value="4"),
        DeclareLaunchArgument("multi_topic", default_value="0"),
        DeclareLaunchArgument("publish_freq", default_value="10.0"),
        DeclareLaunchArgument("frame_id", default_value="livox_frame"),
        DeclareLaunchArgument(
            "user_config_path", default_value=user_config_path),
        livox_driver,
    ])
