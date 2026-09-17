# GS AR Navigation App

一个 ROS 2 Humble 交互式导航界面：

- 订阅小车彩色相机 `/color/image_raw`，作为全屏背景；
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
