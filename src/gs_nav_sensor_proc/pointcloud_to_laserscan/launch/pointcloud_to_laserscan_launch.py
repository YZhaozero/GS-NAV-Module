from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            name='scanner', default_value='scanner',
            description='Namespace for sample topics'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (bag) clock if true'
        ),
        DeclareLaunchArgument(
            'cloud_topic', default_value='/livox/lidar/pointcloud'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('target_frame', default_value='livox_frame'),
        DeclareLaunchArgument('min_height', default_value='-1.0'),
        DeclareLaunchArgument('max_height', default_value='0.1'),
        DeclareLaunchArgument('range_min', default_value='0.45'),
        DeclareLaunchArgument('range_max', default_value='10.0'),

        Node(
            package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
            remappings=[('cloud_in', LaunchConfiguration('cloud_topic')),
                        ('scan', LaunchConfiguration('scan_topic'))],
            parameters=[{
                'target_frame': LaunchConfiguration('target_frame'),
                'transform_tolerance': 0.01,
                'min_height': ParameterValue(
                    LaunchConfiguration('min_height'), value_type=float),
                'max_height': ParameterValue(
                    LaunchConfiguration('max_height'), value_type=float),
                'angle_min': -3.14159,  # -M_PI/2
                'angle_max': 3.14159,  # M_PI/2
                'angle_increment': 0.0043,  # M_PI/360.0
                'scan_time': 0.3333,
                'range_min': ParameterValue(
                    LaunchConfiguration('range_min'), value_type=float),
                'range_max': ParameterValue(
                    LaunchConfiguration('range_max'), value_type=float),
                'use_inf': True,
                'inf_epsilon': 1.0,
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'), value_type=bool)
            }],
            name='pointcloud_to_laserscan',
            output='screen'
        )
    ])
