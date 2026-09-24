# GS-NAV Ubuntu 22.04 部署说明

本文档面向已经安装好 Ubuntu 22.04 和 ROS 2 Humble 的设备。工作空间可以放在任意路径，以下命令不要求用户名为 `zy`，也不要求目录固定为 `~/ws/GS-NAV-Module`。

## 1. 部署结构

建议把项目和第三方依赖分开：

```text
任意位置/GS-NAV-Module/       # 本项目源码及 build/install/log
~/gs_nav_dependencies/
├── src/                     # Sophus、Livox-SDK2、TEASER++ 源码
├── build/                   # 上述原生库的构建目录
├── install/                 # 上述原生库的用户级安装前缀
├── ros2_ws/                 # Scan Context、FastGICP、NDT-OMP underlay
└── setup.bash               # 统一环境入口
```

安装脚本会用 `apt` 安装 Ubuntu/ROS 二进制包，并按项目要求添加 BorgLab 的 GTSAM 4.2 PPA。GTSAM 属于系统级安装；从源码构建的 Sophus、Livox-SDK2、TEASER++ 等库不会写入 `/usr/local`，也不会写入当前 GS-NAV 工作空间。

部署脚本固定或指定的关键第三方依赖为：

| 依赖 | 固定版本 | 用途 |
| --- | --- | --- |
| Livox-SDK2 | `v1.2.5` | `livox_ros_driver2` 的硬依赖 |
| Sophus | `1.22.10` | FAST-LIVO2 和 vikit 的硬依赖 |
| TEASER++ | `e2f6f17` | `localizer` 全局点云配准的硬依赖 |
| GTSAM | BorgLab PPA `4.2` | `localizer` 位姿图、iSAM2 和 SC 生成工具的硬编译依赖 |
| Scan Context ROS 2 | 仓库内适配版本 | 外部 ROS underlay；定位数据库/兼容工具 |
| FastGICP、NDT-OMP | 仓库内适配版本 | 独立定位算法包，当前默认 Localizer 不直接链接它们 |

说明：当前 `localizer` 已在自身源码中实现 Scan Context 描述子、数据库加载和匹配，并未链接 `scancontext_ros2` 动态库。外部构建该包是为了保留完整定位工具链和后续数据库工具兼容。`localizer_node` 和 `generate_sc_tool` 都直接链接 GTSAM 与 TEASER++，因此即使运行时暂不启用位姿图，也必须在编译前安装 GTSAM。

若项目用于商业交付，还应单独复核各第三方项目许可证。仓库内 Scan Context README 标注了 CC BY-NC-SA 4.0 条款，不能只因为脚本能够编译就默认满足商业使用条件。

## 2. 一键安装依赖

在项目根目录执行：

```bash
./scripts/install_dependencies.sh
```

脚本会调用 `sudo apt-get`、`sudo add-apt-repository`、`rosdep` 和 `git`，因此需要管理员密码及可访问 Ubuntu、ROS、Launchpad 和 GitHub 的网络。默认使用全部 CPU 核心；Jetson 或内存较小的设备建议限制并行数：

```bash
./scripts/install_dependencies.sh --jobs 2
```

如需换依赖目录：

```bash
./scripts/install_dependencies.sh --deps-root "$HOME/robot_libs/gs_nav"
export GS_NAV_DEPS_ROOT="$HOME/robot_libs/gs_nav"
```

后续只重编源码、不再运行 apt/rosdep：

```bash
./scripts/install_dependencies.sh --skip-apt --jobs 2
```

`--skip-apt` 会同时跳过 GTSAM PPA 配置和 GTSAM 安装，只能在已经成功安装 `libgtsam-dev`、`libgtsam-unstable-dev` 的机器上使用。

脚本可重复运行。它只同步 `~/gs_nav_dependencies/ros2_ws/src` 中由本项目管理的三个包；如果指定目录已经存在但没有 `.gs_nav_dependencies` 标记，脚本会停止，避免覆盖用户文件。

### 手动安装 GTSAM 4.2

如果不使用一键脚本，必须先按以下顺序安装 GTSAM：

```bash
sudo apt update
sudo apt install software-properties-common

sudo add-apt-repository ppa:borglab/gtsam-release-4.2
sudo apt update

sudo apt install libgtsam-dev libgtsam-unstable-dev
```

然后确认两个开发包的来源和版本：

```bash
apt-cache policy libgtsam-dev libgtsam-unstable-dev
dpkg-query -W 'libgtsam*'
```

候选版本应来自 `ppa.launchpadcontent.net/borglab/gtsam-release-4.2`。不要再额外安装 `ros-humble-gtsam`：项目的 `package.xml` 虽然声明了 `gtsam`，但一键脚本会让 `rosdep` 跳过该键，统一使用 BorgLab PPA 版本，避免两套 GTSAM 头文件、CMake 配置或动态库相互覆盖。

GTSAM 安装到系统标准路径，CMake 会直接通过 `find_package(GTSAM REQUIRED)` 找到它，不需要把 GTSAM 路径加入 `~/gs_nav_dependencies/setup.bash`。

## 3. 一键编译工作空间

依赖部署完成后执行：

```bash
./build.sh
```

`build.sh` 会自动加载 ROS 2 Humble 和 `~/gs_nav_dependencies/setup.bash`，并忽略工作空间内的 `fast_gicp`、`ndt_omp_ros2`、`scancontext_ros2` 副本，实际使用外部 underlay。编译产物仍位于本项目的 `build/`、`install/`、`log/`。

如果这个工作空间以前已经编译过上述三个包，`install/fast_gicp`、`install/ndt_omp_ros2`、`install/scancontext_ros2` 可能仍是旧产物。迁移到外部 underlay 时先把这三个精确目录移动到工作空间之外备份，再运行 `./build.sh`；不要删除整个 `install/`，除非确定不需要保留其他构建结果。

内存较小的设备可执行：

```bash
./build.sh --parallel-workers 2
```

需要刷新 CMake 缓存时：

```bash
./build.sh --cmake-clean-cache --parallel-workers 2
```

只有离线调试、明确希望在主工作空间构建三套随仓库提供的第三方 ROS 包时，才使用：

```bash
GS_NAV_BUILD_BUNDLED_THIRDPARTY=1 ./build.sh
```

不要使用 `sudo colcon build`，否则构建产物会变成 root 所有，后续普通用户无法增量编译。

每个新终端运行程序前需要加载两层环境：

```bash
source "${GS_NAV_DEPS_ROOT:-$HOME/gs_nav_dependencies}/setup.bash"
source ./install/setup.bash
```

如果工作空间不在当前目录，把第二行换成该工作空间的绝对路径。不要把主工作空间 `install/setup.bash` 永久写死到 `~/.bashrc`，否则换分支或清理工作空间后容易污染下一次编译环境。

## 4. Livox MID-360 网络配置（每台机器必须核对）

仓库默认配置为：

- 上位机有线网卡：`192.168.1.100`；
- 雷达：`192.168.1.161`；
- 数据端口：`56100`～`56501` 范围内的配置端口。

这两个 IP 不是通用值。换工控机、Jetson、网卡或雷达后，需要同时设置主机网卡和 JSON，不能只改其中一个。

先给连接雷达的有线网卡配置静态 IPv4。假设接口名为 `enp3s0`、主机地址为 `192.168.1.100/24`：

```bash
ip -br address
sudo ip address add 192.168.1.100/24 dev enp3s0
sudo ip link set enp3s0 up
ping -c 3 192.168.1.161
```

`ip address add` 重启后会失效，量产部署请用 Ubuntu“网络”设置或 Netplan 固化，并避免给无线网卡和雷达网卡配置重叠子网。

不要直接把每台设备的 IP 提交回公共配置。建议复制一份机器专用配置：

```bash
mkdir -p "$HOME/.config/gs_nav"
cp src/gs_nav_driver/livox_ros_driver2/src/config/MID360_config.json \
  "$HOME/.config/gs_nav/MID360_config.json"
vim "$HOME/.config/gs_nav/MID360_config.json"
```

需要修改：

- `host_net_info` 下所有非空的 `*_ip`：连接雷达的本机网卡 IP；
- `lidar_configs[].ip`：雷达实际 IP；
- 多雷达时，每台雷达使用独立 `lidar_configs` 项并检查端口冲突；
- `extrinsic_parameter`：雷达安装姿态和位置。当前示例 `pitch` 为 `30.0` 度，不能默认认为适用于新车。

单独验证雷达：

```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py \
  user_config_path:="$HOME/.config/gs_nav/MID360_config.json" \
  xfer_format:=4
ros2 topic hz /livox/lidar/pointcloud
ros2 topic hz /livox/imu
```

`xfer_format:=4` 发布 `sensor_msgs/msg/PointCloud2`，适合当前 DLIO、LaserScan 和默认导航链路；FAST-LIVO2 的 `mid360_live.launch.py` 使用 `CustomMsg` 时需要 `xfer_format:=1`。不要让同一个雷达驱动被界面和命令行重复启动。

若能 `ping` 通但没有数据，依次检查：JSON 中主机 IP、雷达 IP、网卡路由、防火墙和端口占用。首次联调可临时检查 `sudo ufw status`，正式设备应按安全策略放行 JSON 中使用的 UDP 端口，而不是长期关闭防火墙。

## 5. RealSense D435i

相机应连接 USB 3.x 端口。先验证系统是否识别：

```bash
lsusb | rg -i 'intel|realsense'
ros2 launch realsense2_camera d435i.launch.py
ros2 topic list | rg '^/camera/'
```

无权限或设备反复断开时，优先检查 librealsense udev 规则、USB 线材、供电和 USB 带宽。多相机部署时在界面或 launch 参数中填写序列号，不要仅依赖自动发现顺序。

## 6. 定位、地图、GTSAM 和 Scan Context 配置

`src/gs_nav_localization/localizer/config/localizer.yaml` 当前仍在 `default_map_path` 中保存了开发机绝对路径。通过 GS-NAV 界面启动定位时，所选 PCD 会通过 `map:=...` 覆盖它，所以推荐在“导航系统 → 地图”中选择实际 PCD。

命令行启动示例：

```bash
ros2 launch localizer localizer_launch.py \
  map:="/absolute/path/to/localization_map.pcd" \
  cloud_topic:=/dlio/odom_node/pointcloud/deskewed \
  odom_topic:=/dlio/odom_node/odom
```

如果需要保存本机专用的定位参数，复制 YAML 后修改 `default_map_path`、SC 阈值和采样分辨率，再显式传入配置：

```bash
mkdir -p "$HOME/.config/gs_nav"
cp src/gs_nav_localization/localizer/config/localizer.yaml \
  "$HOME/.config/gs_nav/localizer.yaml"
vim "$HOME/.config/gs_nav/localizer.yaml"

ros2 launch localizer localizer_launch.py \
  config_path:="$HOME/.config/gs_nav/localizer.yaml" \
  map:="/absolute/path/to/localization_map.pcd"
```

当前版本不再从 `sc_db/poses.txt + data/*.sc` 目录加载定位数据库。启用 `use_scan_context: true` 后，Localizer 会使用与 PCD 同目录、同文件名追加 `.sc` 后缀的二进制缓存，例如：

```text
/data/maps/site_a.pcd
/data/maps/site_a.pcd.sc
```

首次加载 PCD 时，如果缓存不存在，Localizer 会根据 `sc_grid_resolution` 和 `sc_max_radius` 自动生成并写入；若缓存读取失败，也会重新生成。地图目录必须对运行 GS-NAV 的普通用户可写，否则应提前离线生成缓存。仓库中的 `src/gs_nav_localization/localizer/sc_db/` 属于上一版格式，当前默认加载路径不会使用它。

更新后的代码还提供了离线 SC 生成工具，输入 PCD 后会在同目录生成 `<地图文件>.pcd.sc`：

```bash
ros2 run localizer generate_sc_tool /absolute/path/to/localization_map.pcd
```

工具发现目标 `.sc` 已存在时会直接跳过。PCD 内容发生变化，或修改 `sc_grid_resolution`、`sc_max_radius` 等影响数据库结构的启动参数后，应先把旧 `.sc` 移走备份，再重新生成，避免地图与缓存不匹配。

`generate_sc_tool` 与 `localizer_node` 共用定位源码，因此也会链接 GTSAM。当前 GTSAM 位姿图管理器采用 iSAM2，并已进入编译链路；运行时位姿图开关目前默认为关闭，且优化结果尚未直接应用到 TF，但这不影响 GTSAM 已成为必装的编译和动态链接依赖。

定位还需注意：

- PCD 地图与实时点云必须使用一致的尺度、坐标约定和外参；
- `map_frame`、`local_frame`、DLIO 的 `odom`/点云话题必须一致；
- TEASER++ 全局配准依赖有效几何特征，空旷或重复结构场景需要重新调 `global_voxel_size`、`global_feature_radius`；
- `use_sim_time` 真机运行应为 `false`。界面的完整链路默认是 `false`，但 `src/gs_nav_app/config/gs_nav.yaml` 目前为 `true`，直接使用该 YAML 启动 AR 界面时应按实际时钟修改。

## 7. 首次启动建议顺序

不要第一次就直接启动整套系统。按下列顺序确认，能快速定位问题：

1. 启动 MID-360，确认 `/livox/lidar/pointcloud` 和 `/livox/imu` 有稳定频率；
2. 启动 D435i，确认彩色、深度和 IMU 话题；
3. 启动 DLIO，确认 `/dlio/odom_node/odom` 与去畸变点云；
4. 启动 Localizer，检查地图成功加载、TEASER++/Scan Context 日志和 TF；
5. 启动 Nav2，确认地图、TF、LaserScan、代价地图和 `/cmd_vel`；
6. 最后启动图形界面统一管理。

图形界面入口：

```bash
ros2 launch gs_nav_app ar_nav.launch.py
```

远程 SSH 启动 GUI 时需要可用的 `DISPLAY`/X11 转发；无图形环境只适合分别启动后端节点。

## 8. 编译与运行自检

```bash
ros2 pkg prefix livox_ros_driver2
ros2 pkg prefix localizer
ros2 pkg prefix scancontext_ros2
ros2 pkg prefix fast_livo

dpkg-query -W 'libgtsam*'
ldd install/localizer/lib/localizer/localizer_node | rg 'not found' || true
ldd install/localizer/lib/localizer/generate_sc_tool | rg 'not found' || true
```

`livox_ros_driver2`、`localizer`、`fast_livo` 应指向当前项目 `install`，`scancontext_ros2` 应指向 `~/gs_nav_dependencies/ros2_ws/install`。`dpkg-query` 应列出 `libgtsam-dev` 和 `libgtsam-unstable-dev`；两次 `ldd` 均不应输出 `not found`。如果提示找不到 `libteaser_registration.so` 或 Livox SDK 动态库，通常是当前终端漏掉了：

```bash
source "${GS_NAV_DEPS_ROOT:-$HOME/gs_nav_dependencies}/setup.bash"
source ./install/setup.bash
```

常见编译问题：

| 现象 | 处理 |
| --- | --- |
| `Could not find GTSAM` / `GTSAMConfig.cmake` | 按本文添加 BorgLab 4.2 PPA并安装两个 GTSAM 开发包，然后用 `--cmake-clean-cache` 重编 |
| `libgtsam.so` 或 `libgtsam_unstable.so` 找不到 | 用 `apt-cache policy` 检查 PPA 来源；若混装了 `ros-humble-gtsam`，先确认没有其他项目依赖它，再移除冲突版本、重装 PPA 包并运行 `sudo ldconfig` |
| `add-apt-repository: command not found` | 先安装 `software-properties-common` |
| `Could not find teaserpp` | 先运行依赖脚本，并确认 `CMAKE_PREFIX_PATH` 包含外部 `install` |
| `Could not find Sophus` | 同上；不要混用系统中其他不兼容 Sophus 版本 |
| `livox_lidar_api.h` 或 SDK 库找不到 | 检查外部 `install/include`、`install/lib`，重新加载 `setup.bash` |
| 构建进程被系统杀死 | 使用 `--parallel-workers 1` 或 `2`，Jetson 同时确认 swap/内存 |
| 编译时出现旧工作空间路径 | 新终端重新执行 `./build.sh --cmake-clean-cache`，不要 source 旧工作空间 |
| 真机节点全部等待时间 | 关闭 `use_sim_time`，并确认没有等待不存在的 `/clock` |

## 9. 部署检查清单

- [ ] Ubuntu 22.04、ROS 2 Humble 架构与目标设备一致；
- [ ] BorgLab GTSAM 4.2 PPA 已配置，两个 GTSAM 开发包已安装且未混装 `ros-humble-gtsam`；
- [ ] `~/gs_nav_dependencies/setup.bash` 存在且可加载；
- [ ] 主机网卡 IP 与 MID-360 JSON 完全一致；
- [ ] 雷达 IP、端口、安装外参已按本车修改；
- [ ] 真机使用 `PointCloud2` 的链路已设 `xfer_format:=4`；
- [ ] RealSense USB、权限、序列号已验证；
- [ ] PCD 定位地图和 Nav2 YAML/PGM 栅格地图没有混用；
- [ ] Localizer 的 `<定位地图>.pcd.sc` 与当前 PCD/SC 参数匹配，或已明确关闭 `use_scan_context`；
- [ ] 真机 `use_sim_time=false`；
- [ ] TF 树包含 `map -> odom -> base_link` 及传感器外参；
- [ ] `/cmd_vel` 接口、底盘控制和急停策略已单独验证后再联调整套导航。
