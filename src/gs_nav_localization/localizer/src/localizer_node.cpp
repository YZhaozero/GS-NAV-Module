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

// ROS2核心头文件
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

// PCL和TF相关头文件
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>

// 本地头文件
#include "localizers/commons.h"
#include "localizers/icp_localizer.h"
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
    rclcpp::Time last_send_tf_time = rclcpp::Clock().now();
    builtin_interfaces::msg::Time last_message_time;

    CloudType::Ptr last_cloud = std::make_shared<CloudType>();
    M3D last_r;
    V3D last_t;

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
          max_time_diff_s(0.3),
          teaser_only(false)  // 初始化成员变量
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

        m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
        m_localizer = std::make_shared<ICPLocalizer>(m_localizer_config);
        
        // 设置TEASER++模式
        m_localizer->setUseTeaserOnly(teaser_only);
        
        if (!m_config.default_map_path.empty() && std::filesystem::exists(m_config.default_map_path))
        {
            if (m_localizer->loadMap(m_config.default_map_path))
            {
                RCLCPP_INFO(this->get_logger(), "Default map loaded from: %s", m_config.default_map_path.c_str());
                m_state.map_loaded = true;
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Failed to load default map from: %s", m_config.default_map_path.c_str());
            }
        }

        m_map_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("map_cloud", 10);
        
        // 添加TEASER++匹配点云发布器
        m_teaser_src_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("teaser_source_cloud", 10);
        m_teaser_tgt_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("teaser_target_cloud", 10);
        m_teaser_corr_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("teaser_correspondence_cloud", 10);
        m_teaser_aligned_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("teaser_aligned_cloud", 10);
        m_teaser_corr_marker_pub = this->create_publisher<visualization_msgs::msg::Marker>("teaser_corr_marker", 10);
        
        // 添加GICP匹配点云发布器
        m_gicp_rough_src_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("gicp_rough_source_cloud", 10);
        m_gicp_rough_tgt_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("gicp_rough_target_cloud", 10);
        m_gicp_refine_src_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("gicp_refine_source_cloud", 10);
        m_gicp_refine_tgt_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("gicp_refine_target_cloud", 10);
        
        m_sc_pose_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("sc_initial_pose", 10);
        
        auto period = std::chrono::milliseconds(10);
        m_timer = this->create_wall_timer(period, std::bind(&LocalizerNode::timerCB, this));

    }

    ~LocalizerNode() = default;

    void loadParameters()
    {
        this->declare_parameter("config_path", "");
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
        
        // ======= 新增 TEASER++ 匹配参数读取 =======
        if (config["teaser_use_absolute_scale"]) m_localizer_config.teaser_use_absolute_scale = config["teaser_use_absolute_scale"].as<bool>();
        if (config["teaser_use_crosscheck"]) m_localizer_config.teaser_use_crosscheck = config["teaser_use_crosscheck"].as<bool>();
        if (config["teaser_use_tuple_test"]) m_localizer_config.teaser_use_tuple_test = config["teaser_use_tuple_test"].as<bool>();
        if (config["teaser_tuple_scale"]) m_localizer_config.teaser_tuple_scale = config["teaser_tuple_scale"].as<double>();
        
        // 全局配准参数
        if (config["use_global_registration"]) m_localizer_config.use_global_registration = config["use_global_registration"].as<bool>();
        if (config["global_voxel_size"]) m_localizer_config.global_voxel_size = config["global_voxel_size"].as<double>();
        if (config["global_feature_radius"]) m_localizer_config.global_feature_radius = config["global_feature_radius"].as<double>();
        if (config["global_score_thresh"]) m_localizer_config.global_score_thresh = config["global_score_thresh"].as<double>();
        
        // ======= 新增 Scan Context 参数读取 =======
        if (config["use_scan_context"]) m_localizer_config.use_scan_context = config["use_scan_context"].as<bool>();
        if (config["sc_dist_thresh"]) m_localizer_config.sc_dist_thresh = config["sc_dist_thresh"].as<double>();
        if (config["sc_max_radius"]) m_localizer_config.sc_max_radius = config["sc_max_radius"].as<double>();

        if (config["sc_database_path"]) m_localizer_config.sc_database_path = config["sc_database_path"].as<std::string>();
        RCLCPP_INFO(this->get_logger(), "SC DB Path: %s", m_localizer_config.sc_database_path.c_str());
        
        // 添加仅使用TEASER++模式参数
        teaser_only = false;  // 使用成员变量，而不是局部变量
        if (config["teaser_only"]) teaser_only = config["teaser_only"].as<bool>();
        
        // 输出全局配准状态
        RCLCPP_INFO(this->get_logger(), "全局配准: %s", m_localizer_config.use_global_registration ? "启用" : "禁用");
        if (m_localizer_config.use_global_registration) {
            RCLCPP_INFO(this->get_logger(), "  - 体素大小: %.2f m", m_localizer_config.global_voxel_size);
            RCLCPP_INFO(this->get_logger(), "  - 特征半径: %.2f m", m_localizer_config.global_feature_radius);
            RCLCPP_INFO(this->get_logger(), "  - 评分阈值: %.2f", m_localizer_config.global_score_thresh);
            RCLCPP_INFO(this->get_logger(), "  - 匹配参数: AbsScale=%d, CrossCheck=%d, TupleTest=%d, TupleScale=%.2f",
                m_localizer_config.teaser_use_absolute_scale,
                m_localizer_config.teaser_use_crosscheck,
                m_localizer_config.teaser_use_tuple_test,
                m_localizer_config.teaser_tuple_scale);
            RCLCPP_INFO(this->get_logger(), "  - 仅使用TEASER++: %s", teaser_only ? "是" : "否");
        }
        
        if (m_localizer_config.use_scan_context) {
             RCLCPP_INFO(this->get_logger(), "Scan Context 回环检测: 启用");
             RCLCPP_INFO(this->get_logger(), "  - 距离阈值: %.2f", m_localizer_config.sc_dist_thresh);
             RCLCPP_INFO(this->get_logger(), "  - 最大半径: %.2f m", m_localizer_config.sc_max_radius);
        }
    }

    // 修改timerCB函数中的匹配逻辑
    void timerCB()
    {
        processBuffers();
    
        if (!m_state.message_received)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "NO MESSAGE RECEIVED YET!");
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
        {
            std::lock_guard<std::mutex> lock(m_state.message_mutex);
    
            // ======= initialpose map->base_link 处理 =======
            if (m_state.service_received)
            {
                // 计算 map->odom
                Eigen::Matrix3d odom_base_r = m_state.last_r;
                Eigen::Vector3d odom_base_t = m_state.last_t;
    
                Eigen::Matrix3d map_odom_r = m_state.map_base_r * odom_base_r.transpose();
                Eigen::Vector3d map_odom_t = -map_odom_r * odom_base_t + m_state.map_base_t;
    
                m_state.last_offset_r = map_odom_r;
                m_state.last_offset_t = map_odom_t;
    
                m_state.service_received = false;
                m_state.need_icp_match = true; // 收到initialpose后需要重新匹配
    
                RCLCPP_INFO(this->get_logger(), "Applied initialpose to map->odom");
            }
    
            // ICP初始猜测
            Eigen::Matrix3d R_double = m_state.last_offset_r * m_state.last_r.transpose();
            Eigen::Vector3d t_double = -R_double * m_state.last_t + m_state.last_offset_t;
    
            initial_guess.block<3,3>(0,0) = R_double.cast<float>();
            initial_guess.block<3,1>(0,3) = t_double.cast<float>();
    
            m_localizer->setInput(m_state.last_cloud);
        }
    
        // 只有在需要匹配时才执行ICP
        if (m_state.need_icp_match && m_state.map_loaded)
        {
            bool result = m_localizer->align(initial_guess);
            if (result || teaser_only) {
                geometry_msgs::msg::PoseStamped pose_msg;
                pose_msg.header.stamp = m_state.last_message_time;
                pose_msg.header.frame_id = m_config.map_frame;
                Eigen::Vector3f t = initial_guess.block<3,1>(0,3);
                Eigen::Quaternionf q(initial_guess.block<3,3>(0,0));
                pose_msg.pose.position.x = t.x();
                pose_msg.pose.position.y = t.y();
                pose_msg.pose.position.z = t.z();
                pose_msg.pose.orientation.x = q.x();
                pose_msg.pose.orientation.y = q.y();
                pose_msg.pose.orientation.z = q.z();
                pose_msg.pose.orientation.w = q.w();
                m_sc_pose_pub->publish(pose_msg); // 这里复用 sc_pose_pub 发布最终结果
            }

            if (result)
            {
                M3D map_body_r = initial_guess.block<3,3>(0,0).cast<double>();
                V3D map_body_t = initial_guess.block<3,1>(0,3).cast<double>();
    
                std::lock_guard<std::mutex> lock(m_state.message_mutex);
                m_state.last_offset_r = map_body_r * m_state.last_r.transpose();
                m_state.last_offset_t = -map_body_r * m_state.last_r.transpose() * m_state.last_t + map_body_t;
    
                m_state.localize_success = true;
                m_state.need_icp_match = false; // 匹配成功后不再匹配
                
                RCLCPP_INFO(this->get_logger(), "ICP匹配成功！匹配得分: %.4f", m_localizer->getFitnessScore());
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "ICP匹配失败！");
            }
        }
    
        sendBroadCastTF(m_state.last_message_time);
        publishMapCloud(m_state.last_message_time);
        publishTeaserClouds(m_state.last_message_time); // 添加发布TEASER++点云
        publishGICPClouds(m_state.last_message_time); // 添加发布GICP点云
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
        while (!cloud_buf.empty() && !odom_buf.empty())
        {
            auto cloud_msg = cloud_buf.front();
            auto odom_msg = odom_buf.front();

            double cloud_t = rclcpp::Time(cloud_msg->header.stamp).seconds();
            double odom_t = rclcpp::Time(odom_msg->header.stamp).seconds();
            double diff = std::fabs(cloud_t - odom_t);

            if (diff <= max_time_diff_s)
            {
                cloud_buf.pop_front();
                odom_buf.pop_front();
                syncCB(std::const_pointer_cast<const sensor_msgs::msg::PointCloud2>(cloud_msg),
                       std::const_pointer_cast<const nav_msgs::msg::Odometry>(odom_msg));
            }
            else
            {
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

        RCLCPP_INFO(this->get_logger(), "Received initialpose (map->base_link)");
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
    void publishGICPClouds(builtin_interfaces::msg::Time &time)
        {
            // 发布GICP粗匹配源点云
            if (m_gicp_rough_src_pub->get_subscription_count() > 0) {
                CloudType::Ptr src_cloud = m_localizer->gicpRoughSourceCloud();
                if (src_cloud && src_cloud->size() > 0) {
                    // 创建一个新的点云，设置所有点的强度值为100
                    CloudType::Ptr colored_src_cloud(new CloudType);
                    for (const auto& pt : src_cloud->points) {
                        PointType colored_pt;
                        colored_pt.x = pt.x;
                        colored_pt.y = pt.y;
                        colored_pt.z = pt.z;
                        colored_pt.intensity = 100.0; // 设置统一的强度值
                        colored_src_cloud->push_back(colored_pt);
                    }
                    
                    sensor_msgs::msg::PointCloud2 src_cloud_msg;
                    pcl::toROSMsg(*colored_src_cloud, src_cloud_msg);
                    src_cloud_msg.header.frame_id = m_config.map_frame;
                    src_cloud_msg.header.stamp = time;
                    m_gicp_rough_src_pub->publish(src_cloud_msg);
                }
            }
            
            // 发布GICP粗匹配目标点云
            if (m_gicp_rough_tgt_pub->get_subscription_count() > 0) {
                CloudType::Ptr tgt_cloud = m_localizer->gicpRoughTargetCloud();
                if (tgt_cloud && tgt_cloud->size() > 0) {
                    // 创建一个新的点云，设置所有点的强度值为200
                    CloudType::Ptr colored_tgt_cloud(new CloudType);
                    for (const auto& pt : tgt_cloud->points) {
                        PointType colored_pt;
                        colored_pt.x = pt.x;
                        colored_pt.y = pt.y;
                        colored_pt.z = pt.z;
                        colored_pt.intensity = 200.0; // 设置统一的强度值
                        colored_tgt_cloud->push_back(colored_pt);
                    }
                    
                    sensor_msgs::msg::PointCloud2 tgt_cloud_msg;
                    pcl::toROSMsg(*colored_tgt_cloud, tgt_cloud_msg);
                    tgt_cloud_msg.header.frame_id = m_config.map_frame;
                    tgt_cloud_msg.header.stamp = time;
                    m_gicp_rough_tgt_pub->publish(tgt_cloud_msg);
                }
            }
            
            // 发布GICP精匹配源点云
            if (m_gicp_refine_src_pub->get_subscription_count() > 0) {
                CloudType::Ptr src_cloud = m_localizer->gicpRefineSourceCloud();
                if (src_cloud && src_cloud->size() > 0) {
                    // 创建一个新的点云，设置所有点的强度值为150
                    CloudType::Ptr colored_src_cloud(new CloudType);
                    for (const auto& pt : src_cloud->points) {
                        PointType colored_pt;
                        colored_pt.x = pt.x;
                        colored_pt.y = pt.y;
                        colored_pt.z = pt.z;
                        colored_pt.intensity = 150.0; // 设置统一的强度值
                        colored_src_cloud->push_back(colored_pt);
                    }
                    
                    sensor_msgs::msg::PointCloud2 src_cloud_msg;
                    pcl::toROSMsg(*colored_src_cloud, src_cloud_msg);
                    src_cloud_msg.header.frame_id = m_config.map_frame;
                    src_cloud_msg.header.stamp = time;
                    m_gicp_refine_src_pub->publish(src_cloud_msg);
                }
            }
            
            // 发布GICP精匹配目标点云
            if (m_gicp_refine_tgt_pub->get_subscription_count() > 0) {
                CloudType::Ptr tgt_cloud = m_localizer->gicpRefineTargetCloud();
                if (tgt_cloud && tgt_cloud->size() > 0) {
                    // 创建一个新的点云，设置所有点的强度值为250
                    CloudType::Ptr colored_tgt_cloud(new CloudType);
                    for (const auto& pt : tgt_cloud->points) {
                        PointType colored_pt;
                        colored_pt.x = pt.x;
                        colored_pt.y = pt.y;
                        colored_pt.z = pt.z;
                        colored_pt.intensity = 250.0; // 设置统一的强度值
                        colored_tgt_cloud->push_back(colored_pt);
                    }
                    
                    sensor_msgs::msg::PointCloud2 tgt_cloud_msg;
                    pcl::toROSMsg(*colored_tgt_cloud, tgt_cloud_msg);
                    tgt_cloud_msg.header.frame_id = m_config.map_frame;
                    tgt_cloud_msg.header.stamp = time;
                    m_gicp_refine_tgt_pub->publish(tgt_cloud_msg);
                }
            }
        }
void publishTeaserClouds(builtin_interfaces::msg::Time &time)
    {
        // 1. 发布原始的源点云和目标点云 (保持不变)
        if (m_teaser_src_pub->get_subscription_count() > 0) {
            CloudType::Ptr src_cloud = m_localizer->teaserSourceCloud();
            if (src_cloud && src_cloud->size() > 0) {
                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*src_cloud, msg);
                msg.header.frame_id = "base_link"; // 源点云其实是在雷达系，为了方便看原始形状
                // 如果想看它相对于map的位置，需要变换，下面 aligned_cloud 会做这件事
                // 这里我们还是发到 map frame 方便对比，虽然它会显示在原点
                msg.header.frame_id = m_config.map_frame; 
                msg.header.stamp = time;
                m_teaser_src_pub->publish(msg);
            }
        }
        
        if (m_teaser_tgt_pub->get_subscription_count() > 0) {
            CloudType::Ptr tgt_cloud = m_localizer->teaserTargetCloud();
            if (tgt_cloud && tgt_cloud->size() > 0) {
                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*tgt_cloud, msg);
                msg.header.frame_id = m_config.map_frame;
                msg.header.stamp = time;
                m_teaser_tgt_pub->publish(msg);
            }
        }

        // 2. 【关键新增】发布 "配准后" 的点云 (Aligned Cloud)
        if (m_teaser_aligned_pub->get_subscription_count() > 0) {
            CloudType::Ptr src_cloud = m_localizer->teaserSourceCloud();
            if (src_cloud && src_cloud->size() > 0) {
                CloudType::Ptr aligned_cloud(new CloudType);
                // 获取 TEASER 算出的变换矩阵
                Eigen::Matrix4f transform = m_localizer->getTeaserTransform();
                
                // 执行变换： Source * Transform -> Aligned
                pcl::transformPointCloud(*src_cloud, *aligned_cloud, transform);
                
                // 颜色设置为醒目的红色/粉色
                for(auto& p : aligned_cloud->points) p.intensity = 255.0;

                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*aligned_cloud, msg);
                msg.header.frame_id = m_config.map_frame;
                msg.header.stamp = time;
                m_teaser_aligned_pub->publish(msg);
            }
        }

        // 3. 【关键新增】发布匹配连线 (Correspondence Lines)
        // 画线连接：变换后的源点 <---> 目标点
        // 线越短越好，线如果乱飞说明匹配错了
        if (m_teaser_corr_marker_pub->get_subscription_count() > 0) {
            CloudType::Ptr corr_cloud = m_localizer->teaserCorrespondenceCloud();
            if (corr_cloud && corr_cloud->size() > 0 && corr_cloud->size() % 2 == 0) {
                visualization_msgs::msg::Marker line_strip;
                line_strip.header.frame_id = m_config.map_frame;
                line_strip.header.stamp = time;
                line_strip.ns = "teaser_lines";
                line_strip.action = visualization_msgs::msg::Marker::ADD;
                line_strip.pose.orientation.w = 1.0;
                line_strip.id = 0;
                line_strip.type = visualization_msgs::msg::Marker::LINE_LIST;
                line_strip.scale.x = 0.05; // 线宽
                line_strip.color.r = 0.0;
                line_strip.color.g = 1.0; // 绿色线条
                line_strip.color.b = 0.0;
                line_strip.color.a = 0.8;

                Eigen::Matrix4f transform = m_localizer->getTeaserTransform();

                // corr_cloud 存储结构是: [src_pt_1, tgt_pt_1, src_pt_2, tgt_pt_2, ...]
                for (size_t i = 0; i < corr_cloud->size(); i += 2) {
                    PointType src_pt_raw = corr_cloud->points[i];   // 原始源点
                    PointType tgt_pt = corr_cloud->points[i+1];     // 目标点

                    // 将源点变换到配准后的位置，这样连线才有意义
                    Eigen::Vector4f src_vec(src_pt_raw.x, src_pt_raw.y, src_pt_raw.z, 1.0);
                    Eigen::Vector4f src_transformed = transform * src_vec;

                    geometry_msgs::msg::Point p_start, p_end;
                    p_start.x = src_transformed.x();
                    p_start.y = src_transformed.y();
                    p_start.z = src_transformed.z();
                    
                    p_end.x = tgt_pt.x;
                    p_end.y = tgt_pt.y;
                    p_end.z = tgt_pt.z;

                    line_strip.points.push_back(p_start);
                    line_strip.points.push_back(p_end);
                }
                m_teaser_corr_marker_pub->publish(line_strip);
            }
        }
    }

private:
    NodeConfig m_config;
    NodeState m_state;

    ICPConfig m_localizer_config;
    std::shared_ptr<ICPLocalizer> m_localizer;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_cloud_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr m_odom_sub;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr m_initialpose_sub;

    std::deque<sensor_msgs::msg::PointCloud2::SharedPtr> cloud_buf;
    std::deque<nav_msgs::msg::Odometry::SharedPtr> odom_buf;
    std::mutex buffer_mutex;
    const size_t max_buffer_size = 200;
    const double max_time_diff_s;
    
    // 添加teaser_only成员变量
    bool teaser_only;

    bool m_teaser_only = false;

    rclcpp::TimerBase::SharedPtr m_timer;
    std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_map_cloud_pub;
    
    // 添加TEASER++匹配点云发布器
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_teaser_src_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_teaser_tgt_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_teaser_corr_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_teaser_aligned_pub; // 配准后的点云
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr m_teaser_corr_marker_pub; // 匹配连线
    
    // 添加GICP点云发布器
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_gicp_rough_src_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_gicp_rough_tgt_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_gicp_refine_src_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_gicp_refine_tgt_pub;

    // 添加Scan Context初始位姿发布器
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_sc_pose_pub; 
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocalizerNode>());
    rclcpp::shutdown();
    return 0;
}