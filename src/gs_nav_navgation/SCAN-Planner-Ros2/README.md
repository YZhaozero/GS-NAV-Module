# SCAN-Planner ROS 2

面向四足机器人长程导航的空间碰撞感知局部规划器。当前目录只保留规划、局部建图、
闭环速度控制和调试可视化，不包含 Gazebo、地图生成、传感器渲染、运动学仿真或机器人
模型。

本项目是 [wuyi2121/SCAN-Planner](https://github.com/wuyi2121/SCAN-Planner) 的 ROS 2
移植，适配 Ubuntu 22.04、ROS 2 Humble 和 C++17。署名与许可证见 [NOTICE](NOTICE)
和 [LICENSE](LICENSE)。

## 数据流

```text
目标（单点 / 预设点 / 参考路径）
                  \
里程计 + 传感器位姿 + 点云或深度图 --> scan_planner_node --> planning/bspline
                                                     \
                                                      +--> 栅格和 Marker 调试输出

planning/bspline + 里程计 --> closed_loop_controller --> cmd_vel
```

`scan_planner_node` 是规划器本体。`closed_loop_controller` 是可选的平面速度控制适配器；
如果底盘已有轨迹跟踪器，可设置 `start_controller:=false`，直接消费 B-spline。

## 构建与启动

```bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

ros2 launch scan_planner run.launch.py \
  navi_mode:=1 sensor_type:=lidar \
  body_pose_topic:=/LIO/odom_vehicle \
  sensor_pose_topic:=/LIO/odom_imu \
  cloud_topic:=/LIO/clouds_lidar \
  cmd_vel_topic:=/cmd_vel
```

RViz 可单独启动：

```bash
ros2 launch scan_planner rviz.launch.py
```

所有 launch 话题参数都可通过 `ros2 launch scan_planner run.launch.py --show-args`
查看。默认参数在：

- `src/planner/plan_manage/config/planner.yaml`
- `src/planner/plan_manage/config/controllers.yaml`

可用 `planner_params_file:=/absolute/path/planner.yaml` 和
`controller_params_file:=/absolute/path/controllers.yaml` 替换默认参数文件。

## 规划器输入接口

以下名称是节点内部的相对话题名；`run.launch.py` 已将其映射为可配置的 launch 参数。

| 内部话题 | 类型 | 条件 | 数据约定 |
| --- | --- | --- | --- |
| `body_pose` | `nav_msgs/msg/Odometry` | 必需 | 机身在 `grid_map.frame_id` 下的位置、姿态和世界系速度；同时驱动 FSM 和滑动地图中心。默认映射 `/LIO/odom_vehicle`。 |
| `sensor_pose` | `nav_msgs/msg/Odometry` | 必需 | 传感器在 `grid_map.frame_id` 下的位姿。激光模式使用最新位姿；深度模式与图像做近似时间同步。默认映射 `/LIO/odom_imu`。 |
| `cloud` | `sensor_msgs/msg/PointCloud2` | `sensor_type=lidar` | `cloud_is_world=false` 时点坐标必须在传感器坐标系，规划器用 `sensor_pose` 转到世界系；为 `true` 时点已在世界系。 |
| `depth` | `sensor_msgs/msg/Image` | `sensor_type=depth` | 对齐深度图，支持 `16UC1`（按 `grid_map.k_depth_scaling_factor`）和 `32FC1`；相机内参由 `grid_map.fx/fy/cx/cy` 提供。 |
| `move_base_simple/goal` | `geometry_msgs/msg/PoseStamped` | `navi_mode=1` | 单目标模式，只采用目标 x/y；机身目标高度沿用收到的第一帧 `body_pose.z`。 |
| `initial_path` | `nav_msgs/msg/Path` | `navi_mode=3` | 至少两个 xyz 点。输入 z 表示地面/路线高度，规划器会加上 `grid_map.body_height`；路径按 0.5 m 三维间距降采样并保留终点。 |
| `planning/execution_frozen` | `std_msgs/msg/Bool` | 可选 | 控制器原地调头时暂停规划轨迹时钟。自带闭环控制器会自动发布。 |

`navi_mode=2` 不订阅目标话题，而是从参数 `fsm.waypoints: [x0, y0, z0, ...]`
读取机身目标点，收到首帧里程计后开始。启动时必须传入 `keypoints_file`：

```bash
ros2 launch scan_planner run.launch.py \
  navi_mode:=2 keypoints_file:=/absolute/path/keypoints.yaml
```

`navi_mode=3` 的参考路径通常由上游全局规划器发布。仓库内的
`reference_path_publisher.py` 只是从 YAML 发布固定路线的辅助工具，可通过
`reference_path_file` 启动。

## 规划器输出接口

| 内部话题 | 类型 | 用途 |
| --- | --- | --- |
| `planning/bspline` | `scan_planner_msgs/msg/Bspline` | 核心规划结果。`header.frame_id` 为 `grid_map.frame_id`，`start_time` 为执行起点，`order`、`knots` 和 `pos_pts` 完整定义位置 B-spline；当前不生成 `yaw_pts/yaw_dt`。 |
| `planning/data_display` | `scan_planner_msgs/msg/DataDisp` | 兼容性诊断心跳；当前只更新 `header.stamp`，`a` 至 `e` 未赋业务含义。 |
| `self_inflation` | `visualization_msgs/msg/Marker` | 机器人双圆柱碰撞包络。 |
| `grid_map/occupancy` | `sensor_msgs/msg/PointCloud2` | 当前占用体素，仅有订阅者时生成。 |
| `grid_map/occupancy_inflate` | `sensor_msgs/msg/PointCloud2` | 膨胀后的占用体素，仅有订阅者时生成。 |
| `grid_map/unknown` | `sensor_msgs/msg/PointCloud2` | 未知体素调试输出。 |
| `grid_map/depth_cloud` | `sensor_msgs/msg/PointCloud2` | 深度图投影后的世界系点云，仅深度模式有效。 |
| `grid_map/sensor_pose_extrinsic` | `nav_msgs/msg/Odometry` | 应用内置外参后的传感器位姿，便于校验。 |
| `grid_map/sliding_map_bbox` | `visualization_msgs/msg/Marker` | 滑动局部地图边界。 |
| `/tf` | `geometry_msgs/msg/TransformStamped` | `grid_map.frame_id` 到 `grid_map.sliding_map_frame_id` 的平移变换。 |
| `goal_point`, `global_list`, `init_list`, `optimal_list`, `a_star_list` | `visualization_msgs/msg/Marker` | RViz 规划调试显示。 |

如果启用 `closed_loop_controller`，系统还会输出：

| 话题 | 类型 | 用途 |
| --- | --- | --- |
| `cmd_vel` | `geometry_msgs/msg/Twist` | 底盘平面速度指令，只使用 `linear.x`、`linear.y`、`angular.z`。 |
| `planning/execution_frozen` | `std_msgs/msg/Bool` | 偏航误差过大、仅原地转向时为 `true`。 |

## 三个导航模式

- `navi_mode:=1`：外部发送单个 `PoseStamped` 目标。
- `navi_mode:=2`：依次执行参数文件中的 xyz 目标点。
- `navi_mode:=3`：沿外部 `nav_msgs/Path` 做局部规划和避障。

记录模式 2 路点：

```bash
ros2 run scan_planner keypoint_recorder.py \
  --odom /LIO/odom_vehicle --output keypoints.yaml
```

## 接入注意事项

1. `body_pose`、`sensor_pose`、目标和感知数据必须落在同一世界坐标约定下；代码不通过
   TF 自动纠正输入帧。
2. 使用激光原始点云时设置 `cloud_is_world:=false`；若 LIO 已输出世界系点云则设为
   `true`。
3. `need_extrinsic:=true` 会在 `sensor_pose` 上再应用源码中的固定传感器外参。若上游
   位姿已经是传感器光心/雷达中心，应设为 `false`，避免重复变换。
4. 深度模式必须按实际相机修改 `grid_map.fx/fy/cx/cy` 和深度尺度参数。
5. 自带闭环控制器只跟踪 x/y/yaw，不执行 z 方向控制；真正需要三维机身高度执行时，
   应关闭它并由底盘控制器直接消费 `planning/bspline`。
