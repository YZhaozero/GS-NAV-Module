# Copyright 2026 GS-NAV contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Launch an Intel RealSense D435i with RGB, depth, and IMU enabled."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    camera_launch = PythonLaunchDescriptionSource(
        [get_package_share_directory("realsense2_camera"), "/launch/rs_launch.py"]
    )

    forwarded_arguments = {
        "camera_name": LaunchConfiguration("camera_name"),
        "camera_namespace": LaunchConfiguration("camera_namespace"),
        "serial_no": LaunchConfiguration("serial_no"),
        "device_type": "d435i",
        "enable_color": LaunchConfiguration("enable_color"),
        "enable_depth": LaunchConfiguration("enable_depth"),
        "enable_gyro": LaunchConfiguration("enable_gyro"),
        "enable_accel": LaunchConfiguration("enable_accel"),
        "unite_imu_method": LaunchConfiguration("unite_imu_method"),
        "enable_sync": LaunchConfiguration("enable_sync"),
        "align_depth.enable": LaunchConfiguration("align_depth.enable"),
        "pointcloud.enable": LaunchConfiguration("pointcloud.enable"),
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument("camera_name", default_value="camera"),
            DeclareLaunchArgument("camera_namespace", default_value="camera"),
            DeclareLaunchArgument("serial_no", default_value="''"),
            DeclareLaunchArgument("enable_color", default_value="true"),
            DeclareLaunchArgument("enable_depth", default_value="true"),
            DeclareLaunchArgument("enable_gyro", default_value="true"),
            DeclareLaunchArgument("enable_accel", default_value="true"),
            DeclareLaunchArgument("unite_imu_method", default_value="2"),
            DeclareLaunchArgument("enable_sync", default_value="true"),
            DeclareLaunchArgument("align_depth.enable", default_value="true"),
            DeclareLaunchArgument("pointcloud.enable", default_value="false"),
            IncludeLaunchDescription(
                camera_launch,
                launch_arguments=forwarded_arguments.items(),
            ),
        ]
    )
