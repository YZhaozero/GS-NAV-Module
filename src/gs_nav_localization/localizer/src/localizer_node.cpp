/**
 * @file localizer_node.cpp
 * @brief 基于ICP的定位节点实现（支持 initialpose map->base_link -> map->odom 计算）
 */

#include <queue>
#include <deque>
#include <mutex>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <sstream>

// ROS2核心头文件
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// PCL和TF相关头文件
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/pose_stamped.hpp>

// 本地头文件
#include "localizers/commons.h"
#include "localizers/icp_localizer.h"
#include "localizers/pose_graph_manager.h" // 引入PoseGraph
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;

struct NodeConfig
{
    std::string cloud_topic = "/fastlio2/body_cloud";
    std::string odom_topic = "/fastlio2/lio_odom";
    std::string map_frame = "map";
    std::string local_frame = "base_link";
    double update_hz = 1.0;
    std::string default_map_path = "";
    bool save_clouds = false;  // 🔥 新增：是否保存定位时的点云
    std::string save_clouds_dir = "/tmp/localizer_clouds/";  // 保存目录
};

// 在NodeState结构体中添加匹配控制标志
struct NodeState
{
    std::mutex message_mutex;
    bool message_received = false;
    bool service_received = false;
    bool localize_success = false;
    bool map_loaded = false;
    bool need_icp_match = true; // 添加标志：是否需要进行ICP匹配
    bool skip_global_registration = false; // 添加标志：是否跳过全局配准（TEASER++）
    bool force_sc_relocalization = false; // 标志：是否强制进行SC全局重定位
    int icp_failure_count = 0;  // ICP失败计数器
    const int max_failure_attempts = 3;  // 最大失败尝试次数
    rclcpp::Time last_send_tf_time = rclcpp::Clock().now();
    builtin_interfaces::msg::Time last_message_time;

    CloudType::Ptr last_cloud = std::make_shared<CloudType>();
    M3D last_r = M3D::Identity();  // 🔧 修复：初始化为单位矩阵
    V3D last_t = V3D::Zero();      // 🔧 修复：初始化为零向量

    // 🔥 定位过程中的 odom 位移补偿
    M3D odom_start_r = M3D::Identity();  // 定位开始时的 odom 旋转
    V3D odom_start_t = V3D::Zero();      // 定位开始时的 odom 位置
    bool odom_start_recorded = false;   // 是否已记录定位开始时的 odom

    // 原 map->odom
    M3D last_offset_r = M3D::Identity();
    V3D last_offset_t = V3D::Zero();

    // 初始位姿 map->base_link
    M3D map_base_r = M3D::Identity();
    V3D map_base_t = V3D::Zero();

    M4F initial_guess = M4F::Identity();
};

class LocalizerNode : public rclcpp::Node
{
public:
    LocalizerNode()
        : Node("localizer_node"),
          max_time_diff_s(0.3)
    {
        RCLCPP_INFO(this->get_logger(), "Localizer Node Started (initialpose topic, max_time_diff=%.3f s)", max_time_diff_s);
        loadParameters();

        rclcpp::QoS qos = rclcpp::SensorDataQoS();

        m_cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            m_config.cloud_topic, qos,
            std::bind(&LocalizerNode::cloudCB, this, std::placeholders::_1));

        m_odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
            m_config.odom_topic, qos,
            std::bind(&LocalizerNode::odomCB, this, std::placeholders::_1));

        m_initialpose_sub = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 10,
            std::bind(&LocalizerNode::initialPoseCB, this, std::placeholders::_1));
        m_global_relocalize_service = this->create_service<std_srvs::srv::Trigger>(
            "~/global_relocalize",
            std::bind(
                &LocalizerNode::globalRelocalizeCB, this,
                std::placeholders::_1, std::placeholders::_2));

        m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

        m_localizer = std::make_shared<ICPLocalizer>(m_localizer_config);

        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "📍 地图配置信息:");
        RCLCPP_INFO(this->get_logger(), "  配置文件路径: %s", m_config.default_map_path.c_str());

        if (!m_config.default_map_path.empty() && std::filesystem::exists(m_config.default_map_path))
        {
            RCLCPP_INFO(this->get_logger(), "  地图文件状态: ✅ 存在");
            if (m_localizer->loadMap(m_config.default_map_path))
            {
                RCLCPP_INFO(this->get_logger(), "  加载结果: ✅ 成功");
                RCLCPP_INFO(this->get_logger(), "🗺️  当前使用地图: %s", m_config.default_map_path.c_str());
                m_state.map_loaded = true;
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "  加载结果: ❌ 失败");
                RCLCPP_ERROR(this->get_logger(), "❌ 地图加载失败: %s", m_config.default_map_path.c_str());
            }
        }
        else
        {
            if (m_config.default_map_path.empty())
            {
                RCLCPP_WARN(this->get_logger(), "  地图文件状态: ⚠️  配置为空");
                RCLCPP_WARN(this->get_logger(), "⚠️  未配置地图路径！");
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "  地图文件状态: ❌ 不存在");
                RCLCPP_ERROR(this->get_logger(), "❌ 地图文件不存在: %s", m_config.default_map_path.c_str());
            }
        }
        RCLCPP_INFO(this->get_logger(), "========================================");

        m_map_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("map_cloud", 10);

        // 添加匹配过程可视化的发布器（添加localizer命名空间）
        m_teaser_source_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("localizer/teaser_source_cloud", 10);
        m_teaser_target_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("localizer/teaser_target_cloud", 10);
        m_teaser_corr_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("localizer/teaser_correspondences", 10);
        m_rough_source_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("localizer/rough_source_cloud", 10);
        m_rough_target_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("localizer/rough_target_cloud", 10);
        m_refine_source_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("localizer/refine_source_cloud", 10);
        m_refine_target_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("localizer/refine_target_cloud", 10);

        RCLCPP_INFO(this->get_logger(), "发布话题:");
        RCLCPP_INFO(this->get_logger(), "  - /map_cloud: 地图点云");
        RCLCPP_INFO(this->get_logger(), "  - /teaser_aligned_cloud: TEASER++全局配准结果");
        RCLCPP_INFO(this->get_logger(), "  - /rough_aligned_cloud: GICP粗匹配结果");
        RCLCPP_INFO(this->get_logger(), "  - /refine_aligned_cloud: GICP精匹配结果");

        auto period = std::chrono::milliseconds(10);
        m_timer = this->create_wall_timer(period, std::bind(&LocalizerNode::timerCB, this));
    }

    ~LocalizerNode() = default;

    void loadParameters()
    {
        this->declare_parameter("config_path", "");
        this->declare_parameter("default_map_path", "");
        this->declare_parameter("cloud_topic", "");
        this->declare_parameter("odom_topic", "");
        std::string config_path;
        this->get_parameter<std::string>("config_path", config_path);

        if (config_path.empty())
        {
            RCLCPP_WARN(this->get_logger(), "config_path not provided; using defaults");
            return;
        }

        YAML::Node config = YAML::LoadFile(config_path);
        if (!config)
        {
            RCLCPP_WARN(this->get_logger(), "FAIL TO LOAD YAML FILE!");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "LOAD FROM YAML CONFIG PATH: %s", config_path.c_str());

        if (config["cloud_topic"]) m_config.cloud_topic = config["cloud_topic"].as<std::string>();
        if (config["odom_topic"]) m_config.odom_topic = config["odom_topic"].as<std::string>();
        if (config["map_frame"]) m_config.map_frame = config["map_frame"].as<std::string>();
        if (config["local_frame"]) m_config.local_frame = config["local_frame"].as<std::string>();
        if (config["update_hz"]) m_config.update_hz = config["update_hz"].as<double>();
        if (config["default_map_path"]) m_config.default_map_path = config["default_map_path"].as<std::string>();

        // ICP参数
        if (config["rough_scan_resolution"]) m_localizer_config.rough_scan_resolution = config["rough_scan_resolution"].as<double>();
        if (config["rough_map_resolution"]) m_localizer_config.rough_map_resolution = config["rough_map_resolution"].as<double>();
        if (config["rough_max_iteration"]) m_localizer_config.rough_max_iteration = config["rough_max_iteration"].as<int>();
        if (config["rough_score_thresh"]) m_localizer_config.rough_score_thresh = config["rough_score_thresh"].as<double>();
        if (config["refine_scan_resolution"]) m_localizer_config.refine_scan_resolution = config["refine_scan_resolution"].as<double>();
        if (config["refine_map_resolution"]) m_localizer_config.refine_map_resolution = config["refine_map_resolution"].as<double>();
        if (config["refine_max_iteration"]) m_localizer_config.refine_max_iteration = config["refine_max_iteration"].as<int>();
        if (config["refine_score_thresh"]) m_localizer_config.refine_score_thresh = config["refine_score_thresh"].as<double>();

        // GICP特有参数
        if (config["correspondence_randomness"]) m_localizer_config.correspondence_randomness = config["correspondence_randomness"].as<int>();
        if (config["maximum_optimizer_iterations"]) m_localizer_config.maximum_optimizer_iterations = config["maximum_optimizer_iterations"].as<int>();

        // 全局配准参数
        if (config["use_global_registration"]) m_localizer_config.use_global_registration = config["use_global_registration"].as<bool>();
        if (config["global_voxel_size"]) m_localizer_config.global_voxel_size = config["global_voxel_size"].as<double>();
        if (config["global_feature_radius"]) m_localizer_config.global_feature_radius = config["global_feature_radius"].as<double>();
        if (config["global_score_thresh"]) m_localizer_config.global_score_thresh = config["global_score_thresh"].as<double>();

        // Scan Context 参数
        if (config["use_scan_context"]) m_localizer_config.use_scan_context = config["use_scan_context"].as<bool>();
        if (config["sc_dist_thresh"]) m_localizer_config.sc_dist_thresh = config["sc_dist_thresh"].as<double>();
        if (config["sc_max_radius"]) m_localizer_config.sc_max_radius = config["sc_max_radius"].as<double>();
        if (config["sc_grid_resolution"]) m_localizer_config.sc_grid_resolution = config["sc_grid_resolution"].as<double>();
        if (config["sc_similar_candidates_thresh"]) m_localizer_config.sc_similar_candidates_thresh = config["sc_similar_candidates_thresh"].as<double>();
        if (config["sc_max_candidates"]) m_localizer_config.sc_max_candidates = config["sc_max_candidates"].as<int>();
        if (config["sc_enable_early_exit"]) m_localizer_config.sc_enable_early_exit = config["sc_enable_early_exit"].as<bool>();
        if (config["enable_ground_z_correction"]) m_localizer_config.enable_ground_z_correction = config["enable_ground_z_correction"].as<bool>();
        if (config["map_ground_z"]) m_localizer_config.map_ground_z = config["map_ground_z"].as<double>();
        if (config["max_ground_z_correction"]) m_localizer_config.max_ground_z_correction = config["max_ground_z_correction"].as<double>();

        // 🔥 新增：点云保存参数
        if (config["save_clouds"]) m_config.save_clouds = config["save_clouds"].as<bool>();
        if (config["save_clouds_dir"]) m_config.save_clouds_dir = config["save_clouds_dir"].as<std::string>();

        // 创建保存目录
        if (m_config.save_clouds) {
            std::filesystem::create_directories(m_config.save_clouds_dir);
            RCLCPP_INFO(this->get_logger(), "💾 点云保存已启用，保存目录: %s", m_config.save_clouds_dir.c_str());
        }

        // 输出全局配准状态
        RCLCPP_INFO(this->get_logger(), "全局配准: %s", m_localizer_config.use_global_registration ? "启用" : "禁用");
        if (m_localizer_config.use_global_registration) {
            RCLCPP_INFO(this->get_logger(), "  - 体素大小: %.2f m", m_localizer_config.global_voxel_size);
            RCLCPP_INFO(this->get_logger(), "  - 特征半径: %.2f m", m_localizer_config.global_feature_radius);
            RCLCPP_INFO(this->get_logger(), "  - 评分阈值: %.2f", m_localizer_config.global_score_thresh);
        }

        // 输出 Scan Context 状态
        RCLCPP_INFO(this->get_logger(), "Scan Context: %s", m_localizer_config.use_scan_context ? "启用" : "禁用");
        if (m_localizer_config.use_scan_context) {
            RCLCPP_INFO(this->get_logger(), "  - 匹配阈值: %.2f", m_localizer_config.sc_dist_thresh);
            RCLCPP_INFO(this->get_logger(), "  - 最大半径: %.2f m", m_localizer_config.sc_max_radius);
            RCLCPP_INFO(this->get_logger(), "  - 网格分辨率: %.2f m", m_localizer_config.sc_grid_resolution);
            RCLCPP_INFO(this->get_logger(), "  - 最大候选数: %d", m_localizer_config.sc_max_candidates);
            RCLCPP_INFO(this->get_logger(), "  - 提前退出: %s", m_localizer_config.sc_enable_early_exit ? "启用" : "禁用");
        }

        std::string override_map_path;
        std::string override_cloud_topic;
        std::string override_odom_topic;
        this->get_parameter("default_map_path", override_map_path);
        this->get_parameter("cloud_topic", override_cloud_topic);
        this->get_parameter("odom_topic", override_odom_topic);
        if (!override_map_path.empty()) m_config.default_map_path = override_map_path;
        if (!override_cloud_topic.empty()) m_config.cloud_topic = override_cloud_topic;
        if (!override_odom_topic.empty()) m_config.odom_topic = override_odom_topic;

        RCLCPP_INFO(this->get_logger(), "Localization map: %s", m_config.default_map_path.c_str());
    }

    // 修改timerCB函数中的匹配逻辑
    void timerCB()
    {
        processBuffers();

        if (!m_state.message_received)
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            size_t cloud_buf_size = cloud_buf.size();
            size_t odom_buf_size = odom_buf.size();
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "NO MESSAGE RECEIVED YET! 缓冲区状态: 点云=%zu, 里程计=%zu",
                cloud_buf_size, odom_buf_size);
            return;
        }

        rclcpp::Duration diff = rclcpp::Clock().now() - m_state.last_send_tf_time;
        bool update_tf = diff.seconds() > (1.0 / m_config.update_hz) && m_state.message_received;

        if (!update_tf)
        {
            sendBroadCastTF(m_state.last_message_time);
            return;
        }
        m_state.last_send_tf_time = rclcpp::Clock().now();

        M4F initial_guess = M4F::Identity();
        bool manual_initial_pose_this_time = false;
        {
            std::lock_guard<std::mutex> lock(m_state.message_mutex);

            // ======= initialpose map->base_link 处理 =======
            if (m_state.service_received)
            {
                // 检查是否为手动位姿模式（skip_global=true 且 force_sc=false）
                if (m_state.skip_global_registration && !m_state.force_sc_relocalization) {
                    // 手动位姿只作为局部GICP的初值；匹配成功前不更新TF。
                    manual_initial_pose_this_time = true;
                    m_state.need_icp_match = true;
                    m_state.icp_failure_count = 0;
                    m_state.service_received = false;

                    RCLCPP_INFO(this->get_logger(), "🎯 手动位姿模式：以 /initialpose 为初值执行局部GICP");
                } else {
                    // 🌍 全局重定位模式：需要进行匹配（SC或TEASER++）
                    // 🔥 修复：全局重定位模式下不基于 (0,0,0) 更新 last_offset
                    // 保持当前的 last_offset 不变，让 SC/GICP 定位后再更新
                    m_state.need_icp_match = true;
                    m_state.service_received = false;
                    RCLCPP_INFO(this->get_logger(), "🌍 全局重定位模式：保持当前TF，等待SC/GICP定位结果后更新");
                }
            }

            // ICP初始猜测
            Eigen::Matrix3d R_double = m_state.last_offset_r * m_state.last_r.transpose();
            Eigen::Vector3d t_double = -R_double * m_state.last_t + m_state.last_offset_t;

            initial_guess.block<3,3>(0,0) = R_double.cast<float>();
            initial_guess.block<3,1>(0,3) = t_double.cast<float>();

            if (manual_initial_pose_this_time) {
                initial_guess.block<3,3>(0,0) = m_state.map_base_r.cast<float>();
                initial_guess.block<3,1>(0,3) = m_state.map_base_t.cast<float>();
            }

            m_localizer->setInput(m_state.last_cloud);
        }

        // 只有在需要匹配时才执行ICP
        if (m_state.need_icp_match && m_state.map_loaded)
        {
            // 🔥 关键修复：首次定位时（启动导航后），强制使用全局重定位
            bool force_sc_this_time = m_state.force_sc_relocalization;
            // 🔥 修改：只在真正首次定位时（且未成功过）才强制SC，避免多轮重复测试
            if (!manual_initial_pose_this_time && !m_state.localize_success &&
                m_state.icp_failure_count == 0) {
                // 首次定位 且 没有收到手动initialpose 且 还没有失败过，强制全局重定位
                force_sc_this_time = true;
                RCLCPP_INFO(this->get_logger(), "🌍 首次定位：自动进入全局重定位模式（Scan Context）");
            } else if (m_state.icp_failure_count > 0) {
                // 如果已经失败过，不再强制SC，避免重复测试所有432个候选
                force_sc_this_time = false;
                RCLCPP_INFO(this->get_logger(), "ℹ️  已测试过所有SC候选，跳过重复测试");
            }

            // 🔥 保存定位时的点云（仅在全局定位时保存，用于离线测试）
            if (m_config.save_clouds && force_sc_this_time) {
                static int cloud_counter = 0;
                std::stringstream ss;
                ss << m_config.save_clouds_dir << "/query_cloud_"
                   << std::setfill('0') << std::setw(6) << cloud_counter++ << ".pcd";
                pcl::io::savePCDFileBinary(ss.str(), *m_state.last_cloud);
                RCLCPP_INFO(this->get_logger(), "💾 已保存全局定位查询点云: %s (%zu 点)",
                           ss.str().c_str(), m_state.last_cloud->size());
            }

            // 🔥 记录定位开始时的 odom 位姿，用于后续补偿机器人移动
            {
                std::lock_guard<std::mutex> lock(m_state.message_mutex);
                m_state.odom_start_r = m_state.last_r;
                m_state.odom_start_t = m_state.last_t;
                m_state.odom_start_recorded = true;
                RCLCPP_INFO(this->get_logger(), "📍 记录定位开始时的 odom 位姿: [%.3f, %.3f, %.3f]",
                           m_state.odom_start_t.x(), m_state.odom_start_t.y(), m_state.odom_start_t.z());
            }

            // 如果设置了跳过全局配准标志，临时禁用TEASER++
            bool original_use_global = m_localizer_config.use_global_registration;
            if (m_state.skip_global_registration)
            {
                m_localizer_config.use_global_registration = false;
                // 🔍 调试：打印更新配置前的 sc_max_candidates 值
                RCLCPP_INFO(this->get_logger(), "🔍 更新配置前: sc_max_candidates=%d", m_localizer_config.sc_max_candidates);
                m_localizer->updateConfig(m_localizer_config);
                // 🔍 调试：打印更新配置后的 sc_max_candidates 值
                RCLCPP_INFO(this->get_logger(), "🔍 更新配置后: sc_max_candidates=%d", m_localizer->config().sc_max_candidates);
                RCLCPP_INFO(this->get_logger(), "ℹ️  临时禁用TEASER++，使用手动位姿进行局部GICP");
            }

            bool result = m_localizer->align(initial_guess, force_sc_this_time);

            // 如果强制SC成功，清除标志
            if (m_state.force_sc_relocalization) {
                m_state.force_sc_relocalization = false;
            }

            // 恢复全局配准设置
            if (m_state.skip_global_registration)
            {
                m_localizer_config.use_global_registration = original_use_global;
                // 🔍 调试：打印恢复配置前的 sc_max_candidates 值
                RCLCPP_INFO(this->get_logger(), "🔍 恢复配置前: sc_max_candidates=%d", m_localizer_config.sc_max_candidates);
                m_localizer->updateConfig(m_localizer_config);
                // 🔍 调试：打印恢复配置后的 sc_max_candidates 值
                RCLCPP_INFO(this->get_logger(), "🔍 恢复配置后: sc_max_candidates=%d", m_localizer->config().sc_max_candidates);
                m_state.skip_global_registration = false; // 只跳过一次
            }

            if (result)
            {
                M3D map_body_r = initial_guess.block<3,3>(0,0).cast<double>();
                V3D map_body_t = initial_guess.block<3,1>(0,3).cast<double>();

                std::lock_guard<std::mutex> lock(m_state.message_mutex);

                RCLCPP_INFO(this->get_logger(), "🔍 定位成功，检查是否需要补偿机器人移动...");
                RCLCPP_INFO(this->get_logger(), "   odom_start_recorded = %s", m_state.odom_start_recorded ? "true" : "false");

                // 将定位结果从点云采集时刻补偿到当前时刻，同时补偿旋转和平移。
                if (m_state.odom_start_recorded) {
                    M3D map_odom_r_start = map_body_r * m_state.odom_start_r.transpose();
                    V3D map_odom_t_start =
                        -map_odom_r_start * m_state.odom_start_t + map_body_t;

                    M3D compensated_r = map_odom_r_start * m_state.last_r;
                    V3D compensated_t =
                        map_odom_r_start * m_state.last_t + map_odom_t_start;

                    const double translation_delta =
                        (compensated_t - map_body_t).norm();
                    Eigen::AngleAxisd rotation_delta(
                        compensated_r * map_body_r.transpose());
                    const double rotation_delta_deg =
                        std::abs(rotation_delta.angle()) * 180.0 / M_PI;

                    map_body_r = compensated_r;
                    map_body_t = compensated_t;
                    m_state.odom_start_recorded = false;
                    RCLCPP_INFO(
                        this->get_logger(),
                        "✅ 已补偿定位期间运动：平移 %.3f m，旋转 %.2f°",
                        translation_delta, rotation_delta_deg);
                } else {
                    RCLCPP_WARN(
                        this->get_logger(),
                        "⚠️  odom_start_recorded 为 false，跳过补偿计算");
                }

                m_state.last_offset_r = map_body_r * m_state.last_r.transpose();
                m_state.last_offset_t = -map_body_r * m_state.last_r.transpose() * m_state.last_t + map_body_t;

                m_state.localize_success = true;
                m_state.need_icp_match = false; // 匹配成功后不再匹配
                m_state.icp_failure_count = 0;  // 重置失败计数器

                RCLCPP_INFO(this->get_logger(), "✅ ICP匹配成功！匹配得分: %.4f", m_localizer->getFitnessScore());

                // Pose Graph Update
                if (m_use_pose_graph && m_pose_graph) {
                    double timestamp = rclcpp::Time(m_state.last_message_time).seconds();
                    m_pose_graph->addKeyFrame(timestamp, initial_guess, m_state.last_cloud);

                    // 获取优化后的位姿 (目前仅作为记录，真正应用回环修正需要更复杂的TF逻辑)
                    // Eigen::Matrix4f optimized_pose = m_pose_graph->getOptimizedPose();
                }
            }
            else
            {
                m_state.icp_failure_count++;
                RCLCPP_WARN(this->get_logger(), "❌ ICP匹配失败！（%d/%d）",
                           m_state.icp_failure_count, m_state.max_failure_attempts);

                if (m_state.icp_failure_count >= m_state.max_failure_attempts)
                {
                    m_state.need_icp_match = false;  // 停止自动尝试
                    RCLCPP_ERROR(this->get_logger(),
                                "⛔ ICP匹配失败%d次，停止自动定位。请使用手工配准（/initialpose）重新初始化。",
                                m_state.max_failure_attempts);
                }
            }
        }

        sendBroadCastTF(m_state.last_message_time);
        publishMapCloud(m_state.last_message_time);
        publishMatchingResults(m_state.last_message_time);
    }

    void cloudCB(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        cloud_buf.push_back(msg);
        if (cloud_buf.size() > max_buffer_size) cloud_buf.pop_front();
    }

    void odomCB(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        odom_buf.push_back(msg);
        if (odom_buf.size() > max_buffer_size) odom_buf.pop_front();
    }

    void processBuffers()
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        static int sync_fail_count = 0;
        static rclcpp::Time last_warn_time = rclcpp::Clock().now();

        while (!cloud_buf.empty() && !odom_buf.empty())
        {
            auto cloud_msg = cloud_buf.front();
            auto odom_msg = odom_buf.front();

            double cloud_t = rclcpp::Time(cloud_msg->header.stamp).seconds();
            double odom_t = rclcpp::Time(odom_msg->header.stamp).seconds();
            double diff = std::fabs(cloud_t - odom_t);

            if (diff <= max_time_diff_s)
            {
                sync_fail_count = 0;
                cloud_buf.pop_front();
                odom_buf.pop_front();
                syncCB(std::const_pointer_cast<const sensor_msgs::msg::PointCloud2>(cloud_msg),
                       std::const_pointer_cast<const nav_msgs::msg::Odometry>(odom_msg));
            }
            else
            {
                sync_fail_count++;
                // 每5秒警告一次时间戳不同步问题
                rclcpp::Time now = rclcpp::Clock().now();
                if ((now - last_warn_time).seconds() > 5.0)
                {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                        "时间戳不同步: 点云=%.3f, 里程计=%.3f, 差值=%.3f秒 (阈值=%.3f秒), 失败次数=%d",
                        cloud_t, odom_t, diff, max_time_diff_s, sync_fail_count);
                    last_warn_time = now;
                }
                if (cloud_t < odom_t) cloud_buf.pop_front();
                else odom_buf.pop_front();
            }
        }
    }

    void syncCB(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg,
                const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg)
    {
        std::lock_guard<std::mutex> lock(m_state.message_mutex);
        pcl::fromROSMsg(*cloud_msg, *m_state.last_cloud);
        m_state.last_r = Eigen::Quaterniond(
            odom_msg->pose.pose.orientation.w,
            odom_msg->pose.pose.orientation.x,
            odom_msg->pose.pose.orientation.y,
            odom_msg->pose.pose.orientation.z
        ).toRotationMatrix();
        m_state.last_t = V3D(
            odom_msg->pose.pose.position.x,
            odom_msg->pose.pose.position.y,
            odom_msg->pose.pose.position.z
        );
        m_state.last_message_time = cloud_msg->header.stamp;

        if (!m_state.message_received)
        {
            m_state.message_received = true;
            m_config.local_frame = odom_msg->header.frame_id;
        }
    }

    void globalRelocalizeCB(
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        std::lock_guard<std::mutex> lock(m_state.message_mutex);
        if (!m_state.map_loaded) {
            response->success = false;
            response->message = "map is not loaded";
            return;
        }

        m_state.force_sc_relocalization = true;
        m_state.skip_global_registration = false;
        m_state.need_icp_match = true;
        m_state.icp_failure_count = 0;
        response->success = true;
        response->message = "global relocalization queued";
        RCLCPP_INFO(
            this->get_logger(),
            "🌍 收到全局重定位请求：Scan Context + GICP，TEASER++ 兜底");
    }

    void initialPoseCB(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        if (!m_state.map_loaded)
        {
            RCLCPP_WARN(this->get_logger(), "Map not loaded, cannot relocalize");
            return;
        }

        Eigen::Quaterniond q(msg->pose.pose.orientation.w,
                             msg->pose.pose.orientation.x,
                             msg->pose.pose.orientation.y,
                             msg->pose.pose.orientation.z);
        Eigen::Vector3d t(msg->pose.pose.position.x,
                          msg->pose.pose.position.y,
                          msg->pose.pose.position.z);

        std::lock_guard<std::mutex> lock(m_state.message_mutex);
        m_state.map_base_r = q.toRotationMatrix();
        m_state.map_base_t = t;
        m_state.service_received = true;
        m_state.need_icp_match = true; // 收到initialpose后需要重新匹配
        m_state.icp_failure_count = 0; // 重置失败计数器

        // /initialpose 始终表示人工先验，只执行局部GICP，不再用“接近原点”猜测操作意图。
        m_state.skip_global_registration = true;
        m_state.force_sc_relocalization = false;
        RCLCPP_INFO(
            this->get_logger(),
            "🎯 手动位姿模式：使用用户先验 (%.2f, %.2f, %.2f) 执行局部GICP",
            t.x(), t.y(), t.z());
    }

    void sendBroadCastTF(builtin_interfaces::msg::Time &time)
    {
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.frame_id = m_config.map_frame;
        transformStamped.child_frame_id = m_config.local_frame;
        transformStamped.header.stamp = time;

        Eigen::Quaterniond q(m_state.last_offset_r);
        V3D t = m_state.last_offset_t;

        transformStamped.transform.translation.x = t.x();
        transformStamped.transform.translation.y = t.y();
        transformStamped.transform.translation.z = t.z();

        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        transformStamped.transform.rotation.w = q.w();

        m_tf_broadcaster->sendTransform(transformStamped);
    }

    void publishMapCloud(builtin_interfaces::msg::Time &time)
    {
        if (m_map_cloud_pub->get_subscription_count() < 1) return;
        CloudType::Ptr map_cloud = m_localizer->refineMap();
        if (!map_cloud || map_cloud->size() < 1) return;

        sensor_msgs::msg::PointCloud2 map_cloud_msg;
        pcl::toROSMsg(*map_cloud, map_cloud_msg);
        map_cloud_msg.header.frame_id = m_config.map_frame;
        map_cloud_msg.header.stamp = time;
        m_map_cloud_pub->publish(map_cloud_msg);
    }

    void publishMatchingResults(builtin_interfaces::msg::Time &time)
    {
        // 发布TEASER++的source和target点云
        if (m_localizer->hasTeaserResult())
        {
            CloudType::Ptr teaser_src = m_localizer->getTeaserSourceCloud();
            CloudType::Ptr teaser_tgt = m_localizer->getTeaserTargetCloud();
            M4F teaser_transform = m_localizer->getTeaserTransform();

            if (teaser_src && teaser_src->size() > 0)
            {
                // 将source变换到map frame，这样才能和target对齐可视化
                CloudType::Ptr teaser_src_transformed(new CloudType);
                pcl::transformPointCloud(*teaser_src, *teaser_src_transformed, teaser_transform);

                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*teaser_src_transformed, msg);
                msg.header.frame_id = m_config.map_frame;
                msg.header.stamp = time;
                m_teaser_source_pub->publish(msg);
            }

            if (teaser_tgt && teaser_tgt->size() > 0)
            {
                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*teaser_tgt, msg);
                msg.header.frame_id = m_config.map_frame;
                msg.header.stamp = time;
                m_teaser_target_pub->publish(msg);

                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "发布TEASER++点云: source=%zu, target=%zu",
                    teaser_src->size(), teaser_tgt->size());
            }

            // 发布对应关系连线
            auto correspondences = m_localizer->getTeaserCorrespondences();
            if (!correspondences.empty() && teaser_src && teaser_tgt)
            {
                visualization_msgs::msg::MarkerArray marker_array;

                // 创建连线
                for (size_t i = 0; i < correspondences.size(); ++i)
                {
                    auto [src_idx, tgt_idx] = correspondences[i];

                    if (src_idx < 0 || tgt_idx < 0 ||
                        static_cast<std::size_t>(src_idx) >= teaser_src->size() ||
                        static_cast<std::size_t>(tgt_idx) >= teaser_tgt->size())
                        continue;

                    // 计算对应点对之间的距离
                    const auto& src_pt = teaser_src->points[src_idx];
                    const auto& tgt_pt = teaser_tgt->points[tgt_idx];
                    float dist = std::sqrt(
                        std::pow(src_pt.x - tgt_pt.x, 2) +
                        std::pow(src_pt.y - tgt_pt.y, 2) +
                        std::pow(src_pt.z - tgt_pt.z, 2));

                    visualization_msgs::msg::Marker line_marker;
                    line_marker.header.frame_id = m_config.map_frame;
                    line_marker.header.stamp = time;
                    line_marker.ns = "teaser_correspondences";
                    line_marker.id = i;
                    line_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
                    line_marker.action = visualization_msgs::msg::Marker::ADD;
                    line_marker.scale.x = 0.02;  // 线宽

                    // 根据距离设置颜色
                    if (dist < 1.0)
                    {
                        // 绿色：好的匹配
                        line_marker.color.r = 0.0;
                        line_marker.color.g = 1.0;
                        line_marker.color.b = 0.0;
                        line_marker.color.a = 0.8;
                    }
                    else if (dist < 2.0)
                    {
                        // 黄色：一般的匹配
                        line_marker.color.r = 1.0;
                        line_marker.color.g = 1.0;
                        line_marker.color.b = 0.0;
                        line_marker.color.a = 0.6;
                    }
                    else
                    {
                        // 红色：差的匹配
                        line_marker.color.r = 1.0;
                        line_marker.color.g = 0.0;
                        line_marker.color.b = 0.0;
                        line_marker.color.a = 0.3;
                    }

                    // 添加source点（配准后的位置）
                    geometry_msgs::msg::Point p_src;
                    Eigen::Vector4f src_vec(src_pt.x, src_pt.y, src_pt.z, 1.0f);
                    Eigen::Vector4f src_transformed = teaser_transform * src_vec;
                    p_src.x = src_transformed[0];
                    p_src.y = src_transformed[1];
                    p_src.z = src_transformed[2];

                    // 添加target点
                    geometry_msgs::msg::Point p_tgt;
                    p_tgt.x = tgt_pt.x;
                    p_tgt.y = tgt_pt.y;
                    p_tgt.z = tgt_pt.z;

                    line_marker.points.push_back(p_src);
                    line_marker.points.push_back(p_tgt);

                    marker_array.markers.push_back(line_marker);
                }

                m_teaser_corr_pub->publish(marker_array);

                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "发布TEASER++对应关系连线: %zu 对", correspondences.size());
            }
        }

        // 发布GICP粗匹配的source和target点云
        if (m_localizer->hasRoughResult())
        {
            CloudType::Ptr rough_src = m_localizer->getRoughSourceCloud();
            CloudType::Ptr rough_tgt = m_localizer->getRoughTargetCloud();
            M4F rough_transform = m_localizer->getRoughTransform();

            if (rough_src && rough_src->size() > 0)
            {
                // 将source变换到map frame
                CloudType::Ptr rough_src_transformed(new CloudType);
                pcl::transformPointCloud(*rough_src, *rough_src_transformed, rough_transform);

                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*rough_src_transformed, msg);
                msg.header.frame_id = m_config.map_frame;
                msg.header.stamp = time;
                m_rough_source_pub->publish(msg);
            }

            if (rough_tgt && rough_tgt->size() > 0)
            {
                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*rough_tgt, msg);
                msg.header.frame_id = m_config.map_frame;
                msg.header.stamp = time;
                m_rough_target_pub->publish(msg);

                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "发布GICP粗匹配点云: source=%zu, target=%zu",
                    rough_src->size(), rough_tgt->size());
            }
        }

        // 发布GICP精匹配的source和target点云
        if (m_localizer->hasRefineResult())
        {
            CloudType::Ptr refine_src = m_localizer->getRefineSourceCloud();
            CloudType::Ptr refine_tgt = m_localizer->getRefineTargetCloud();
            M4F refine_transform = m_localizer->getRefineTransform();

            if (refine_src && refine_src->size() > 0)
            {
                // 将source变换到map frame
                CloudType::Ptr refine_src_transformed(new CloudType);
                pcl::transformPointCloud(*refine_src, *refine_src_transformed, refine_transform);

                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*refine_src_transformed, msg);
                msg.header.frame_id = m_config.map_frame;
                msg.header.stamp = time;
                m_refine_source_pub->publish(msg);
            }

            if (refine_tgt && refine_tgt->size() > 0)
            {
                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*refine_tgt, msg);
                msg.header.frame_id = m_config.map_frame;
                msg.header.stamp = time;
                m_refine_target_pub->publish(msg);

                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "发布GICP精匹配点云: source=%zu, target=%zu",
                    refine_src->size(), refine_tgt->size());
            }
        }
    }

private:
    NodeConfig m_config;
    NodeState m_state;

    ICPConfig m_localizer_config;
    std::shared_ptr<ICPLocalizer> m_localizer;
    std::shared_ptr<PoseGraphManager> m_pose_graph; // Pose Graph Manager
    bool m_use_pose_graph = false;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_cloud_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr m_odom_sub;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr m_initialpose_sub;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr m_global_relocalize_service;

    std::deque<sensor_msgs::msg::PointCloud2::SharedPtr> cloud_buf;
    std::deque<nav_msgs::msg::Odometry::SharedPtr> odom_buf;
    std::mutex buffer_mutex;
    const size_t max_buffer_size = 10000;  // 增加缓冲区大小，保留更多定位日志
    const double max_time_diff_s;

    rclcpp::TimerBase::SharedPtr m_timer;
    std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_map_cloud_pub;

    // 匹配过程可视化的发布器
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_teaser_source_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_teaser_target_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr m_teaser_corr_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_rough_source_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_rough_target_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_refine_source_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_refine_target_pub;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocalizerNode>());
    rclcpp::shutdown();
    return 0;
}
