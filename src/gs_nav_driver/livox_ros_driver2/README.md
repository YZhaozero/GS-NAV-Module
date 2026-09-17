# Livox MID-360 ROS 2 Driver

本功能包是面向 ROS 2 Humble 的 MID-360 专用驱动，只保留 MID-360 的配置、点云、IMU 和 RViz 启动入口。

## 当前网络与外参

- 主机网卡 IP：`192.168.1.100`
- MID-360 IP：`192.168.1.161`
- 安装角：`roll=0°`、`pitch=30°`、`yaw=0°`
- 配置文件：`src/config/MID360_config.json`

平移外参 `x/y/z` 的单位为毫米，旋转外参 `roll/pitch/yaw` 的单位为度。

## 构建

先安装 Livox-SDK2，再在工作空间根目录运行：

```bash
colcon build --symlink-install --packages-select livox_ros_driver2
source install/setup.bash
```

## 启动

同时发布 Livox CustomMsg、PointCloud2 和 IMU：

```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

发布 PointCloud2、IMU 并启动 RViz：

```bash
ros2 launch livox_ros_driver2 rviz_MID360_launch.py
```

## 话题

| 话题 | 消息类型 |
| --- | --- |
| `/livox/lidar` | `livox_ros_driver2/msg/CustomMsg` |
| `/livox/lidar/pointcloud` | `sensor_msgs/msg/PointCloud2` |
| `/livox/imu` | `sensor_msgs/msg/Imu` |

RViz 启动方式只发布 `/livox/lidar`（`sensor_msgs/msg/PointCloud2`）和 `/livox/imu`。

## 输出格式

`xfer_format` 支持以下值：

- `0`：PointCloud2（字段为 `x/y/z/intensity/tag/line/timestamp`）
- `1`：Livox CustomMsg
- `4`：同时发布 CustomMsg 和 PointCloud2

MID-360 每个原始数据包包含 4 条扫描线。点坐标和 IMU 数据都会应用 `MID360_config.json` 中的安装外参。
