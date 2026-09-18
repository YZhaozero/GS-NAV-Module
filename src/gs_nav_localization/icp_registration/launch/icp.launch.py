from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
import sys
from ament_index_python.packages import get_package_share_directory
sys.path.append(os.path.join(get_package_share_directory('icp_registration'), 'launch'))

def generate_launch_description():
  from launch_ros.actions import Node
  from launch import LaunchDescription

  use_sim_time_launch_arg = DeclareLaunchArgument(
      'use_sim_time',
      default_value='false',
      description='Use simulation (bag) clock if true'
  )
  params = os.path.join(get_package_share_directory('icp_registration'), 'config', 'icp.yaml')

  icp_node = Node(
    package='icp_registration',
    executable='icp_registration_node',
    output='screen',
    parameters=[params, {'use_sim_time': LaunchConfiguration('use_sim_time')}]
  )

  tf_node = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='map_tf_pub',
    arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
  )
  
  return LaunchDescription([use_sim_time_launch_arg, icp_node, tf_node])
