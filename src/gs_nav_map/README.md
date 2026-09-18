# GS NAV maps

`gs_nav_app` 默认将在线建图生成的 PCD，以及地图处理页面保存的
PCD、YAML 和 PGM 文件放在此目录。默认文件名包含毫秒级时间戳，避免覆盖旧地图。

可通过 ROS 参数 `map_storage_dir` 或环境变量 `GS_NAV_MAP_DIR` 修改默认位置。
