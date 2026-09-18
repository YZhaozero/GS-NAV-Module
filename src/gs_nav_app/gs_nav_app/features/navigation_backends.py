"""Pluggable localization and navigation launch backends."""

from __future__ import annotations

from pathlib import Path
import shlex
from typing import Dict, List, Tuple


def launch_bool(value: bool) -> str:
    return "true" if value else "false"


def existing_optional_file(path: str, label: str) -> Tuple[bool, str]:
    if path.strip() and not Path(path).expanduser().is_file():
        return False, f"{label}不存在：{path}"
    return True, ""


def expanded_or_empty(path: str) -> str:
    return str(Path(path).expanduser()) if path.strip() else ""


class LaunchBackend:
    """Interface implemented by a selectable ROS 2 launch backend."""

    key = ""
    label = ""
    description = ""

    def validate(self, _config) -> Tuple[bool, str]:
        return True, ""

    def launch_arguments(self, _config) -> List[str]:
        raise NotImplementedError


class PointCloudLocalizerBackend(LaunchBackend):
    key = "pointcloud_localizer"
    label = "点云 Localizer"
    description = "使用 PCD 定位地图、DLIO 点云和里程计进行全局定位。"

    def validate(self, config) -> Tuple[bool, str]:
        map_path = Path(config.localization_map).expanduser()
        if not map_path.is_file():
            return False, f"定位点云地图不存在：{map_path}"
        if map_path.suffix.lower() != ".pcd":
            return False, "点云 Localizer 的定位地图必须是 PCD 文件"
        return existing_optional_file(
            config.localizer_config, "Localizer 配置文件")

    def launch_arguments(self, config) -> List[str]:
        arguments = [
            "launch", "localizer", "localizer_launch.py",
            f"map:={Path(config.localization_map).expanduser().resolve()}",
            f"cloud_topic:={config.localizer_cloud_topic.strip()}",
            f"odom_topic:={config.localizer_odom_topic.strip()}",
        ]
        if config.localizer_config.strip():
            arguments.append(
                "config_path:="
                f"{Path(config.localizer_config).expanduser().resolve()}")
        return arguments


class DisabledLocalizationBackend(LaunchBackend):
    key = "disabled"
    label = "不启动定位"
    description = "适用于导航后端直接使用 LIO 里程计的场景。"

    def launch_arguments(self, _config) -> List[str]:
        return []


class CustomLocalizationBackend(LaunchBackend):
    key = "custom"
    label = "自定义定位 Launch"
    description = "启动任意定位包；附加参数支持地图和配置文件占位符。"

    def validate(self, config) -> Tuple[bool, str]:
        if not config.custom_localization_package.strip():
            return False, "请填写自定义定位包名"
        if not config.custom_localization_launch.strip():
            return False, "请填写自定义定位 Launch 文件"
        try:
            self._extra_arguments(config)
        except (KeyError, ValueError) as exc:
            return False, f"自定义定位参数格式错误：{exc}"
        return True, ""

    @staticmethod
    def _extra_arguments(config) -> List[str]:
        values = {
            "map": expanded_or_empty(config.localization_map),
            "config": expanded_or_empty(config.localizer_config),
            "use_sim_time": launch_bool(config.use_sim_time),
        }
        return shlex.split(
            config.custom_localization_arguments.format(**values))

    def launch_arguments(self, config) -> List[str]:
        return [
            "launch",
            config.custom_localization_package.strip(),
            config.custom_localization_launch.strip(),
            *self._extra_arguments(config),
        ]


class Nav2Backend(LaunchBackend):
    key = "nav2"
    label = "Nav2"
    description = "使用栅格地图和 Nav2 Bringup 进行全局导航。"

    def validate(self, config) -> Tuple[bool, str]:
        map_path = Path(config.navigation_map).expanduser()
        if not map_path.is_file():
            return False, f"Nav2 导航地图不存在：{map_path}"
        if map_path.suffix.lower() not in (".yaml", ".yml"):
            return False, "Nav2 导航地图必须是 YAML 文件"
        return existing_optional_file(config.nav2_params, "Nav2 参数文件")

    def launch_arguments(self, config) -> List[str]:
        arguments = [
            "launch", "nav2_bringup", "bringup_launch.py",
            f"map:={Path(config.navigation_map).expanduser().resolve()}",
            f"use_sim_time:={launch_bool(config.use_sim_time)}",
            f"autostart:={launch_bool(config.autostart)}",
            f"rviz:={launch_bool(config.nav2_rviz)}",
        ]
        if config.nav2_params.strip():
            arguments.append(
                f"params_file:={Path(config.nav2_params).expanduser().resolve()}")
        return arguments


class ScanPlannerBackend(LaunchBackend):
    key = "scan_planner"
    label = "SCAN-Planner"
    description = "直接消费里程计和点云进行局部空间规划，不需要栅格地图。"

    def validate(self, config) -> Tuple[bool, str]:
        if config.scan_navi_mode not in (1, 2, 3):
            return False, "SCAN-Planner 导航模式必须是 1、2 或 3"
        if config.scan_sensor_type not in ("lidar", "depth"):
            return False, "SCAN-Planner 传感器类型必须是 lidar 或 depth"
        for path, label in (
            (config.scan_planner_params, "SCAN-Planner 参数文件"),
            (config.scan_controller_params, "SCAN 控制器参数文件"),
            (config.scan_keypoints_file, "SCAN 途径点文件"),
            (config.scan_reference_path_file, "SCAN 参考路径文件"),
        ):
            valid, message = existing_optional_file(path, label)
            if not valid:
                return valid, message
        if config.scan_navi_mode == 2 and not config.scan_keypoints_file.strip():
            return False, "SCAN-Planner 模式 2 必须选择途径点 YAML"
        if config.scan_reference_path_file.strip() and config.scan_navi_mode != 3:
            return False, "SCAN 参考路径文件只能用于导航模式 3"
        return True, ""

    def launch_arguments(self, config) -> List[str]:
        arguments = [
            "launch", "scan_planner", "run.launch.py",
            f"navi_mode:={config.scan_navi_mode}",
            f"sensor_type:={config.scan_sensor_type}",
            f"body_pose_topic:={config.scan_body_pose_topic.strip()}",
            f"sensor_pose_topic:={config.scan_sensor_pose_topic.strip()}",
            f"cloud_topic:={config.scan_cloud_topic.strip()}",
            f"depth_topic:={config.scan_depth_topic.strip()}",
            f"goal_topic:={config.scan_goal_topic.strip()}",
            f"initial_path_topic:={config.scan_initial_path_topic.strip()}",
            f"cmd_vel_topic:={config.scan_cmd_vel_topic.strip()}",
            f"start_controller:={launch_bool(config.scan_start_controller)}",
            f"cloud_is_world:={launch_bool(config.scan_cloud_is_world)}",
            f"need_extrinsic:={launch_bool(config.scan_need_extrinsic)}",
        ]
        for name, path in (
            ("planner_params_file", config.scan_planner_params),
            ("controller_params_file", config.scan_controller_params),
            ("keypoints_file", config.scan_keypoints_file),
            ("reference_path_file", config.scan_reference_path_file),
        ):
            if path.strip():
                arguments.append(
                    f"{name}:={Path(path).expanduser().resolve()}")
        return arguments


class CustomNavigationBackend(LaunchBackend):
    key = "custom"
    label = "自定义导航 Launch"
    description = "启动任意导航包；附加参数支持地图和参数文件占位符。"

    def validate(self, config) -> Tuple[bool, str]:
        if not config.custom_navigation_package.strip():
            return False, "请填写自定义导航包名"
        if not config.custom_navigation_launch.strip():
            return False, "请填写自定义导航 Launch 文件"
        try:
            self._extra_arguments(config)
        except (KeyError, ValueError) as exc:
            return False, f"自定义导航参数格式错误：{exc}"
        return True, ""

    @staticmethod
    def _extra_arguments(config) -> List[str]:
        values = {
            "map": expanded_or_empty(config.navigation_map),
            "params": expanded_or_empty(config.nav2_params),
            "use_sim_time": launch_bool(config.use_sim_time),
        }
        return shlex.split(config.custom_navigation_arguments.format(**values))

    def launch_arguments(self, config) -> List[str]:
        return [
            "launch",
            config.custom_navigation_package.strip(),
            config.custom_navigation_launch.strip(),
            *self._extra_arguments(config),
        ]


LOCALIZATION_BACKENDS: Dict[str, LaunchBackend] = {
    backend.key: backend for backend in (
        PointCloudLocalizerBackend(),
        DisabledLocalizationBackend(),
        CustomLocalizationBackend(),
    )
}

NAVIGATION_BACKENDS: Dict[str, LaunchBackend] = {
    backend.key: backend for backend in (
        Nav2Backend(),
        ScanPlannerBackend(),
        CustomNavigationBackend(),
    )
}
