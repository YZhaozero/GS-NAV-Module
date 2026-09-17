from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

package_share = get_package_share_directory('livox_ros_driver2')
rviz_config_path = join(package_share, 'config', 'display_point_cloud.rviz')
user_config_path = join(package_share, 'config', 'MID360_config.json')

livox_ros2_params = [
    {"xfer_format": 0},
    {"multi_topic": 0},
    {"publish_freq": 10.0},
    {"frame_id": "livox_frame"},
    {"user_config_path": user_config_path},
]


def generate_launch_description():
    livox_driver = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=livox_ros2_params
    )

    livox_rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['--display-config', rviz_config_path]
    )

    return LaunchDescription([
        livox_driver,
        livox_rviz,
    ])
