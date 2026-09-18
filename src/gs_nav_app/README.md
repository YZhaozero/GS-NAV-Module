# GS AR Navigation App

一个 ROS 2 Humble 交互式导航界面：

- 订阅配置文件指定的彩色相机话题，作为全屏背景；
- 订阅导航路径 `nav_msgs/msg/Path`，在相机画面上绘制半透明导航带；
- 订阅导航 `map_server` 的 `/map` 栅格地图，在主地图和导航小窗中绘制全局路径；
- 使用 Qt 直接显示原始相机画面，导航带使用独立矢量图层绘制，不再压缩成 `/ar_image`；
- 地图支持添加任意数量的有序途径点，调用 Nav2 `/navigate_through_poses`；
- 支持查看导航反馈、剩余距离和取消当前导航；
- 支持加载 ASCII/binary/binary_compressed PCD，以及 ASCII/大小端 binary PLY 点云；
- 自动识别 3D Gaussian Splat PLY，解码球谐基础颜色、透明度、尺度和旋转属性；
- 地图处理页以三维方式显示点云，支持鼠标旋转、平移、缩放和视角复位；
- 支持原始坐标/自动地面找平、XYZ 轴交换与翻转、透视/正交投影及 RGB/高度着色；
- 支持加载和保存 Nav2 PGM/YAML 栅格地图，并把 PCD/PLY 按高度范围转换为栅格；
- 支持用画笔修改栅格障碍/自由/未知区域，或对俯视点云添加、擦除点；
- 导航和地图处理使用两个独立工作区，地图编辑工具不会混入导航选点界面；
- 提供独立的传感器管理页，可配置、启动和停止 Livox MID360 与 RealSense D435i；
- 雷达、相机及合并日志实时显示在界面中，关闭软件时自动回收已启动的驱动进程；
- 传感器数据页可按消息类型选择 ROS 2 话题，监看三维点云、相机视频和 IMU 数值；
- 提供独立建图页，当前可直接启动/停止 DLIO、查看实时三维地图并调用服务保存 PCD；
- 导航设置页以地图为主、相机为辅，可在栅格地图与交互式三维点云之间即时切换；
- 有相机内参和 TF 时执行真实针孔投影，尚未标定时自动回退到可调地面投影。

## 无硬件预览

```bash
cd ~/ws/GS-NAV-Module
colcon build --packages-select gs_nav_app --symlink-install
source install/setup.bash
ros2 launch gs_nav_app demo.launch.py
```

无桌面环境时关闭窗口，合成图仍会发布：

```bash
ros2 launch gs_nav_app demo.launch.py show_window:=false
```

## 接入小车

先按实际话题修改 `config/gs_nav.yaml`，然后运行：

```bash
ros2 launch gs_nav_app ar_nav.launch.py
```

默认点云已随软件放在包内，不依赖原来的 `car_simple_sim` 路径。也可在启动时换成其他
PCD/PLY，或传入空值取消自动加载：

```bash
ros2 launch gs_nav_app ar_nav.launch.py pointcloud_map_path:=/path/to/map.pcd
ros2 launch gs_nav_app ar_nav.launch.py pointcloud_map_path:=""
```

该命令会立即打开 Qt 导航控制台；相机尚未发布图像时显示等待页面，收到第一帧后自动
切换到实时画面。设置页的小相机只显示预览，不显示距离状态卡。主地图每次点击会依次
追加一个途径点；右侧列表可单独选择某个途径点并重新选点或删除，删除后会自动重新
编号。机器人从当前 TF 位置出发，
按编号顺序前往各点。点击“开始导航”后进入纯导航页面，仅保留相机、AR 路线、左下角
小地图和底部“退出导航”按钮，同时在全屏相机中显示剩余距离状态卡。到达最后一个目标
点后先显示“已到达目的地”的完成提示，短暂停留后自动返回地图页，也可以点击“立即
返回地图”跳过等待；手动退出会取消
整个 `/navigate_through_poses` 任务。如果通过 SSH 启动，需要保证图形转发可用并且环境
中存在 `DISPLAY`。

## 传感器管理

导航页右上角点击“传感器管理”进入独立的驱动控制页面。该页面直接运行本机 ROS 2
launch，不启动 HTTP 服务：

- “雷达”页启动 `livox_ros_driver2/msg_MID360_launch.py`，可设置输出消息格式、发布
  频率、`frame_id`、多话题模式和 MID360 JSON 配置路径；
- “相机”页启动 `realsense2_camera/d435i.launch.py`，可设置设备名称、命名空间、
  序列号、RGB/深度/IMU 数据流、同步、深度对齐、相机点云和 IMU 合并方式；
- 右侧日志可在“全部 / 雷达 / 相机”之间切换，启动失败、驱动输出和退出码都会实时
  显示。返回导航页面不会停止传感器，点击对应“停止”按钮或关闭整个软件会清理该
  launch 的完整进程组，避免驱动子进程和话题残留。

点击传感器管理页右上角的“传感器数据”进入实时监看页面：

- “雷达点云”自动列出 `sensor_msgs/msg/PointCloud2` 和
  `livox_ros_driver2/msg/CustomMsg`，支持三维旋转、平移和缩放；
- “相机视频”自动列出 `sensor_msgs/msg/Image` 并显示原始视频帧；
- “IMU”自动列出 `sensor_msgs/msg/Imu`，显示 frame、时间戳、姿态四元数、角速度、
  线加速度和协方差；
- 三种选择框均可手动输入完整话题名称。返回传感器管理页时，会释放这些临时监看订阅，
  不影响已经启动的驱动。

传感器驱动随工作空间构建后，仍使用同一个入口打开界面：

```bash
source install/setup.bash
ros2 launch gs_nav_app ar_nav.launch.py
```

## 独立建图

导航页右上角点击“建图”进入 `GS MAPPING STUDIO`。建图和导航、地图后处理分别处于
独立工作区；返回导航不会终止正在运行的建图任务，必须点击“停止建图”或关闭软件。
当前后端为工作空间内的 `direct_lidar_inertial_odometry`（DLIO）：

- 雷达输入只列出 `sensor_msgs/msg/PointCloud2`，IMU 输入只列出
  `sensor_msgs/msg/Imu`，也可以手动填写话题；
- 默认输入为 `/livox/lidar/pointcloud` 和 `/livox/imu`，默认实时地图为
  `/dlio/map_node/map`；
- 页面中央实时显示 DLIO 输出的三维点云，支持鼠标旋转、平移、缩放；
- 可选择 `use_sim_time` 和是否额外启动 RViz，右侧显示完整 launch 日志；
- “保存当前 PCD”调用 `/save_pcd`，按设置的体素大小和地图名称保存；文件名自动附加
  毫秒级时间，例如 `gs_map_20260918_123456_789.pcd`，连续保存不会覆盖旧地图；
- 保存前会检查是否已经收到有效地图点云，服务返回后还会验证目标文件真实存在且非空，
  最终完整路径会显示在页面和建图日志中。

DLIO 本身要求 PointCloud2。若 Livox 驱动当前使用 `CustomMsg`，请先在“传感器管理”
中把雷达“输出格式”切到 `PointCloud2`，再启动建图。界面会阻止已知类型不匹配的话题，
避免进程正常启动却始终收不到点云。

建图算法入口集中在 `features/mapping.py` 的 `MAPPING_BACKENDS`。后续接入其他 SLAM 时，
注册包名、launch 文件、参数名、地图输出话题与保存适配器即可复用同一套页面、进程组
回收、日志和三维预览，不需要改动导航或地图处理页面。

## 代码结构

界面外壳不再直接实现新功能的 ROS 和进程逻辑，主要模块划分如下：

- `qt_nav_node.py`：导航 ROS 节点、主窗口和页面切换；
- `features/mapping.py`：建图后端定义、ROS 点云/保存适配器和建图控制器；
- `ui/mapping_page.py`：建图页面，只渲染控制器状态并转发用户操作；
- `features/sensors.py`：传感器驱动控制、启动参数生成和实时话题订阅；
- `ros_launch_process.py`：所有功能共用的完整 launch 进程组启停与日志管理；
- `map_processing.py`：PCD/PLY、栅格转换及地图文件处理；
- `nav_math.py`：导航路径投影与几何计算。

新增功能时应建立自己的功能控制器；新增页面只依赖该控制器，不在主窗口中直接创建
ROS 订阅、服务客户端或子进程。这样算法、ROS 接口和 Qt 页面可以分别测试和替换。

## 本地地图处理

地图处理完全在 Qt 进程内完成，不会启动 HTTP 服务。导航页右上角点击“地图处理”会
进入独立的 `GS MAP STUDIO` 页面；该页面负责加载 `.pcd/.ply` 或 Nav2 `.yaml/.pgm`、点云
转栅格、地图画笔编辑和保存，页面内没有导航选点或启动导航按钮。点云转栅格时可设置
分辨率和保留的 Z 高度范围；“保存当前地图”会将点云写为 binary PCD，或将栅格写为
配套的 YAML/PGM 文件。点击“返回导航”回到相机与途径点导航页面。两个页面各自维护
点云/栅格显示模式，地图处理页切换显示类型不会改变导航小地图当前类型。

地图处理页切换到“点云地图”后，左键拖动旋转三维视角，右键或中键拖动平移，滚轮
放大缩小；双击点云区域或点击“重置三维视角”恢复默认等轴视角。导航设置页和开始导航
后的点云小窗也使用可交互三维视图：短按左键添加途径点，左键拖动旋转，右键拖动平移，
滚轮缩放；三维视图会叠加途径点、全局路径与机器人朝向。

普通点云显示默认使用“自动找平地面 + 透视 + 原始 RGB”。自动找平通过点云主平面估计
显示方向；没有 RGB 字段时自动回退到高度着色。姿态、轴顺序、反向开关、投影和颜色
都只影响三维预览，不会改写内存中的原始 XYZ。加载 PCD/PLY 后均可直接执行点云转
栅格；自动找平或轴变换启用时，转换会自动使用与预览一致的坐标副本，并把估计地面
设为 Z=0，因此界面的 Z 高度范围表示相对地面的高度。原始点云不会被旋转或覆盖。
保存当前点云时统一输出 binary PCD，并保留已加载的 RGB。只有切回完全未变换的显示
后才能选择原始 XYZ 转换。选择点云添加/擦除画笔时，界面会自动回到原始 XYZ 显示，
避免在显示变换下误编辑。

对于包含 `f_dc_* / opacity / scale_* / rot_*` 的 Gaussian Splat PLY，程序会自动切换
到 Gaussian 泼溅预览，并按 SuperSplat 的 Y-up 坐标及参考中的 Z=180° 姿态设置显示。
静止时使用颜色、透明度和尺度绘制高斯，拖动期间临时使用快速点预览。此预设只改变
显示矩阵；转栅格默认选择“自动找平/当前显示坐标”，原始 PLY 不会被旋转或覆盖。

也可以直接通过 remap 临时接入，例如：

```bash
ros2 run gs_nav_app ar_nav_node --ros-args \
  -p camera_topic:=/color/image_raw \
  -p camera_info_topic:=/color/camera_info \
  -p map_topic:=/map \
  -p path_topic:=/global_plan
```

## 投影模式

- `projection_mode: auto`：优先使用 `CameraInfo` 和 TF，数据不全时回退；
- `projection_mode: calibrated`：只使用真实标定，适合最终部署；
- `projection_mode: ground`：始终使用简化地面投影，适合初期联调。

真实投影要求 TF 树中存在 `路径坐标系 -> 相机 optical frame` 的变换。回退投影要求存在
`路径坐标系 -> base_link` 的变换。路径带偏高或偏低时，先调整 `camera_height_m` 和
`horizon_ratio`；最终效果应以相机内参和相机到车体的外参标定为准。
