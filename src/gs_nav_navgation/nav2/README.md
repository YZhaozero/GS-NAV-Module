# GS-NAV Navigation

本目录包含用于 ROS 2 Humble 的精简版 Nav2 源码。

- 上游仓库：https://github.com/ros-navigation/navigation2
- 上游版本：`1.1.19`
- 上游提交：`bf40e4d5dfdd02d24e05e75f14a770cb9e349be9`
- 许可证：Apache-2.0（见 `LICENSE`）

为便于项目内代码管理，已移除上游 Git/CI/容器配置、文档站点、测试、基准工具、
演示媒体以及 TurtleBot/Gazebo 示例资源；保留导航运行、二次开发和基础启动所需的
源码、接口、插件描述、行为树、参数与 RViz 配置。

`nav2_route` 未包含在 Nav2 1.1.19 的 `navigation2` 元包依赖闭包中，并额外依赖
`nanoflann`，因此未纳入本精简版本。可选的 `nav2_graceful_controller` 已保留。

构建整个工作区：

```bash
./build.sh
```

只构建 Nav2：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to nav2_bringup
```
