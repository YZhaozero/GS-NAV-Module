#!/usr/bin/env python3
"""FAST-LIVO2 live 启动 — Jetson Orin Nano + Livox MID-360 + RealSense D435i.

依赖启动环境(顺序 source):
    source /opt/ros/humble/setup.bash
    source ~/ws_livox/install/setup.bash
    source ~/ws_fastlivo2/install/setup.bash

组合:
    livox_ros_driver2_node  (CustomMsg xfer_format=1, MID360_config.json)
    parameter_blackboard    (载入 D435 相机参数, fast_livo 经此读取)
    fastlivo_mapping        (laserMapping, mid360_live.yaml)
    realsense2_camera_node  (img_en=1 时才启动; 彩色 640x480 remap 到 /left_camera/image)

MID360_config.json 定位(优先级从高到低):
    lidar_config_path:=<路径>  >  环境变量 MID360_CONFIG  >  自动探测
    自动探测顺序: ~/ws_livox 源码树 -> livox_ros_driver2 安装目录(share/)
    ⚠ 该 json 里的 host_net_info / lidar ip 逐台机器不同, 换机器要改它, 不是改这里

用法:
    # stage A —— 先只 lidar+imu, 验证 MID-360 live 路径
    ros2 launch fast_livo mid360_live.launch.py img_en:=0
    # stage B —— 加 D435i 视觉
    ros2 launch fast_livo mid360_live.launch.py img_en:=1
    # 真机 rviz(仅 Nano 有显示器时)
    ... rviz:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# 包目录
_FASTLIVO_SHARE = get_package_share_directory("fast_livo")


def _livox_driver_node(lidar_config_path, publish_freq, frame_id):
    """Livox 驱动, CustomMsg 模式(xfer_format=1), 与 FAST-LIO2 live 相同。"""
    livox_ros2_params = [
        {"xfer_format": 1},            # 1 = CustomMsg (livox_ros_driver2/msg/CustomMsg)
        {"multi_topic": 0},            # 所有 LiDAR 共用一个 topic
        {"data_src": 0},               # 0 = live lidar
        {"publish_freq": publish_freq},
        {"output_data_type": 0},
        {"frame_id": frame_id},
        {"lvx_file_path": "/home/livox/livox_test.lvx"},
        {"user_config_path": lidar_config_path},
        {"cmdline_input_bd_code": "livox0000000001"},
    ]
    return Node(
        package="livox_ros_driver2",
        executable="livox_ros_driver2_node",
        name="livox_lidar_publisher",
        output="screen",
        parameters=livox_ros2_params,
    )


def _realsense_node():
    """D435i 彩色流; depth/imu 关掉省 CPU/USB 带宽。"""
    params = [
        {"enable_color": True, "enable_depth": False},
        {"enable_infra1": False, "enable_infra2": False},
        {"enable_gyro": False, "enable_accel": False},   # LIVO 用 MID-360 内 IMU
        # 2026-09-16 实测: 本版 realsense2_camera(4.58.3) 不认扁平参数 color_width/
        #   color_height/color_fps, 开关是 rgb_camera.color_profile 这个字符串。
        #   D435i 挂在 USB 2.0 口上, 带宽把可选 profile 限成:
        #     1280x720x{6,10,15} / 1920x1080x8 / 640x480x{6,15,30} / 424x240x{6,15,30,60}
        #   (1280x720x30 是 USB3 专享, 传了会被拒并回落到 640x480x15)
        #   选 1280x720x15: HFOV 70.2°, 而 640x480 只有 55.6° —— 差的 14.6° 是
        #   4:3 把水平裁窄造成的, VFOV 两者都是 43.1°(垂直没丢)。白赚的视场
        #   让更多雷达点落进图像, 直接改善建图覆盖与 exp2 的统计密度。
        # ⚠ 改这里必须同步改 config/camera_pinhole_d435.yaml 的 cam_*, 否则
        #   "Frame: provided image has not the same size as the camera model" 崩溃。
        {"rgb_camera.color_profile": "1280x720x15"},
    ]
    return Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="realsense_camera",
        namespace="camera",
        output="screen",
        parameters=params,
        # realsense2_camera 会把 node 名也拼进话题路径
        # (实际出图 /camera/realsense_camera/color/image_raw),
        # 所以 remap 必须用绝对路径才能匹配上; 相对名 "color/image_raw" 匹配不到。
        remappings=[
            ("/camera/realsense_camera/color/image_raw", "/left_camera/image"),
            ("/camera/realsense_camera/color/camera_info", "/left_camera/camera_info"),
        ],
        respawn=True,
        respawn_delay=3.0,
    )


def _rviz_node():
    rviz_cfg = PathJoinSubstitution(
        [FindPackageShare("fast_livo"), "rviz_cfg", "fast_livo2.rviz"]
    )
    return Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_cfg],
    )


def _make_fastlivo_and_optional(context):
    """OpaqueFunction: 把 launch arg img_en 转成真正的 int 参数叠加到 yaml 之上。"""
    img_en = int(context.perform_substitution(LaunchConfiguration("img_en")))
    fastlivo_params_file = context.perform_substitution(LaunchConfiguration("fastlivo_params_file"))
    use_rviz = context.perform_substitution(LaunchConfiguration("rviz")).lower() in ("1", "true", "yes")
    # 只有在这里才拿得到最终生效值(可能被 lidar_config_path:= 覆盖),
    # 在 generate_launch_description 里打印只会打出"自动探测的默认值", 会误导。
    lidar_cfg_effective = context.perform_substitution(LaunchConfiguration("lidar_config_path"))
    print(f"[mid360_live] {_MID360_CONFIG_NAME} = {lidar_cfg_effective}")

    nodes = [
        Node(
            package="fast_livo",
            executable="fastlivo_mapping",
            name="laserMapping",
            parameters=[fastlivo_params_file, {"common.img_en": img_en}],
            output="screen",
            respawn=True,
            respawn_delay=2.0,   # 防冷启动时 parameter_blackboard 未就绪导致 throw
        ),
    ]
    if img_en == 1:
        nodes.append(_realsense_node())
    if use_rviz:
        nodes.append(_rviz_node())
    return nodes


_MID360_CONFIG_NAME = "MID360_config.json"


def _resolve_lidar_config():
    """定位 MID360_config.json, 返回路径(用作 lidar_config_path 的默认值)。

    优先级: 环境变量 MID360_CONFIG > ~/ws_livox 源码树 > livox_ros_driver2 安装目录。
    更上层的 launch 参数 lidar_config_path:= 显式传入时优先级最高。
    """
    env_path = os.environ.get("MID360_CONFIG", "").strip()
    if env_path:
        path = os.path.expanduser(env_path)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"环境变量 MID360_CONFIG 指向的文件不存在: {path}\n"
                f"    请修正该变量, 或用 lidar_config_path:=<路径> 显式指定"
            )
        return path

    candidates = [
        (f"~/ws_livox/src/livox_ros_driver2/config/{_MID360_CONFIG_NAME}", "ws_livox 源码树"),
    ]
    try:
        share = get_package_share_directory("livox_ros_driver2")
        candidates.append(
            (os.path.join(share, "config", _MID360_CONFIG_NAME), "livox_ros_driver2 安装目录")
        )
    except Exception:
        # 没装 / 没 source livox_ros_driver2 时查不到包, 属正常情况, 继续试其它候选
        pass

    for raw, _source in candidates:
        path = os.path.expanduser(raw)
        if os.path.isfile(path):
            return path

    tried = "\n".join(f"    - {os.path.expanduser(raw)}  ({source})" for raw, source in candidates)
    raise RuntimeError(
        f"找不到 {_MID360_CONFIG_NAME}, 已尝试:\n{tried}\n"
        f"    请用 lidar_config_path:=<路径> 指定, 或设环境变量 MID360_CONFIG"
    )


def generate_launch_description():
    # 默认参数文件/路径
    fastlivo_cfg = os.path.join(_FASTLIVO_SHARE, "config", "mid360_live.yaml")
    camera_cfg = os.path.join(_FASTLIVO_SHARE, "config", "camera_pinhole_d435.yaml")
    lidar_cfg = _resolve_lidar_config()

    args = [
        DeclareLaunchArgument(
            "img_en",
            default_value="1",
            description="stage A: 0 = lidar+imu 不带相机; stage B: 1 = 含 D435i 视觉",
        ),
        DeclareLaunchArgument("rviz", default_value="false", description="Nano 有显示器时开 rviz"),
        DeclareLaunchArgument(
            "fastlivo_params_file", default_value=fastlivo_cfg,
            description="fast_livo 主参数 yaml",
        ),
        DeclareLaunchArgument(
            "camera_params_file", default_value=camera_cfg,
            description="相机(vikit Pinhole)参数 yaml, 载入 parameter_blackboard",
        ),
        DeclareLaunchArgument(
            "lidar_config_path", default_value=lidar_cfg,
            description="Livox MID360_config.json 路径(默认自动探测, 见文件头注释)",
        ),
        DeclareLaunchArgument("publish_freq", default_value="10.0"),
        DeclareLaunchArgument("frame_id", default_value="livox_frame"),
    ]

    # 相机参数经 parameter_blackboard 全局发布, fast_livo 的
    # loadFromRosNs(node, "parameter_blackboard", cam) 读取
    blackboard = Node(
        package="demo_nodes_cpp",
        executable="parameter_blackboard",
        name="parameter_blackboard",
        parameters=[LaunchConfiguration("camera_params_file")],
        output="screen",
    )

    return LaunchDescription(
        args
        + [
            blackboard,
            _livox_driver_node(
                LaunchConfiguration("lidar_config_path"),
                LaunchConfiguration("publish_freq"),
                LaunchConfiguration("frame_id"),
            ),
            OpaqueFunction(function=_make_fastlivo_and_optional),
        ]
    )
