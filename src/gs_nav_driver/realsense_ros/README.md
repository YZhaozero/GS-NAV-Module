# RealSense D435i ROS 2 driver

This directory contains the D435i-relevant subset of the official
[`realsenseai/realsense-ros`](https://github.com/realsenseai/realsense-ros)
repository, pinned to tag `4.55.1` (commit
`8a86cb88a428bdefa204759c899b84adc81606ae`). The upstream Git history and
files for CI, tests, examples, development utilities, and other camera models
have intentionally been removed so this workspace can manage the sources
directly.

The shared `realsense2_camera` implementation is retained in full because the
D435i uses the common D400 runtime. The following ROS 2 packages are required:

- `realsense2_camera`: camera node and D435i launch file
- `realsense2_camera_msgs`: messages and services used by the camera node
- `realsense2_description`: D435/D435i meshes and xacro files

## Requirements

- ROS 2 Humble
- `librealsense2` and its development files, version 2.55.1 or newer

The current workspace was validated with `ros-humble-librealsense2` 2.55.1.

## Build

From the `GS-NAV-Module` workspace root:

```bash
./build.sh --packages-up-to realsense2_camera realsense2_description
source install/setup.bash
```

The workspace build script removes stale entries from a previously sourced
`install/` directory and explicitly allows these source packages to override
the matching ROS Humble binary packages.

## Run D435i

Connect the camera to a USB 3 port, then run:

```bash
ros2 launch realsense2_camera d435i.launch.py
```

This preset enables color, depth, accelerometer, gyroscope, RGB/depth sync,
depth-to-color alignment, and linearly interpolated unified IMU output. Point
cloud output is optional:

```bash
ros2 launch realsense2_camera d435i.launch.py pointcloud.enable:=true
```

For multiple cameras, choose one by serial number:

```bash
ros2 launch realsense2_camera d435i.launch.py serial_no:="'_YOUR_SERIAL_'"
```

The upstream Apache 2.0 license and notices are preserved in this directory.
