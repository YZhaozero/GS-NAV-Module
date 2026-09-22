#pragma once
#include "commons.h"
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
// 添加GICP头文件
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
// 全局配准相关头文件 - TEASER++
#include <teaser/registration.h>
#include <teaser/matcher.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/transforms.h>
#include "scan_context.h"

struct ICPConfig
{
    double refine_scan_resolution = 0.1;
    double refine_map_resolution = 0.1;
    double refine_score_thresh = 0.1;
    int refine_max_iteration = 10;

    double rough_scan_resolution = 0.25;
    double rough_map_resolution = 0.25;
    double rough_score_thresh = 0.2;
    int rough_max_iteration = 5;

    // 添加GICP特有参数
    int correspondence_randomness = 20;    // 随机采样对应点的数量
    int maximum_optimizer_iterations = 20; // 优化器最大迭代次数

    // 全局配准参数
    bool use_global_registration = false;
    double global_voxel_size = 0.5;
    double global_feature_radius = 1.0;
    double global_score_thresh = 20.0;

    // Scan Context 参数
    bool use_scan_context = true;
    double sc_dist_thresh = 0.15; // SC匹配阈值
    double sc_max_radius = 40.0;  // SC最大半径
    double sc_grid_resolution = 2.0; // SC数据库网格分辨率(米)
    double sc_similar_candidates_thresh = 0.1; // 相似候选的评分阈值（相对于最小评分的增量）
    int sc_max_candidates = 10; // SC候选点最大数量，用于ICP匹配
    bool sc_enable_early_exit = false; // 默认完整评估候选，避免并行提前退出遗漏更优解

    // 地图相关的Z轴修正必须显式启用，禁止把单张地图的地面高度用于其他地图。
    bool enable_ground_z_correction = false;
    double map_ground_z = 0.0;
    double max_ground_z_correction = 0.5;

    // 安全检查参数 - 防止位姿突然发散
    bool enable_safety_check = true;           // 是否启用安全检查
    double max_translation_per_frame = 5.0;    // 单帧最大平移距离(m)
    double max_rotation_per_frame = 30.0;      // 单帧最大旋转角度(度)
    double max_velocity = 10.0;                // 最大速度(m/s)
    double max_angular_velocity = 90.0;        // 最大角速度(度/s)
    double max_odom_jump = 10.0;               // odom最大跳变(m)
};

struct SCDatabaseEntry {
    Eigen::Vector3f pose; // x, y, z
    Eigen::MatrixXd descriptor;
    Eigen::VectorXd ringkey;
};

class ICPLocalizer
{
public:
    ICPLocalizer(const ICPConfig &config);

    bool loadMap(const std::string &path);

    void setInput(const CloudType::Ptr &cloud);

    bool align(M4F &guess, bool force_sc = false); // 添加 force_sc 参数
    double getFitnessScore() { return m_fitness_score; } // 添加此方法获取适应度评分
    ICPConfig &config() { return m_config; }
    void updateConfig(const ICPConfig &config) {
        // 🔍 调试：打印配置更新前后的 sc_max_candidates 值
        std::cout << "🔍 ICPLocalizer::updateConfig: sc_max_candidates " << m_config.sc_max_candidates
                  << " -> " << config.sc_max_candidates << std::endl;
        PCL_INFO_STREAM("ICPLocalizer::updateConfig: sc_max_candidates " << m_config.sc_max_candidates
                       << " -> " << config.sc_max_candidates << std::endl);
        m_config = config;
    } // 动态更新配置
    CloudType::Ptr roughMap() { return m_rough_tgt; }
    CloudType::Ptr refineMap() { return m_refine_tgt; }

    // 添加设置GICP参数的方法
    void setUseGICP(bool use_gicp) { m_use_gicp = use_gicp; }

    // 获取中间匹配结果 - 用于可视化匹配过程
    // TEASER++
    CloudType::Ptr getTeaserSourceCloud() { return m_teaser_source_cloud; }
    CloudType::Ptr getTeaserTargetCloud() { return m_teaser_target_cloud; }
    // GICP粗匹配
    CloudType::Ptr getRoughSourceCloud() { return m_rough_inp; }
    CloudType::Ptr getRoughTargetCloud() { return m_rough_tgt; }
    // GICP精匹配
    CloudType::Ptr getRefineSourceCloud() { return m_refine_inp; }
    CloudType::Ptr getRefineTargetCloud() { return m_refine_tgt; }

    M4F getTeaserTransform() { return m_teaser_transform; }
    M4F getRoughTransform() { return m_rough_transform; }
    M4F getRefineTransform() { return m_refine_transform; }
    bool hasTeaserResult() { return m_has_teaser_result; }
    bool hasRoughResult() { return m_has_rough_result; }
    bool hasRefineResult() { return m_has_refine_result; }

    // 获取TEASER++对应关系
    std::vector<std::pair<int, int>> getTeaserCorrespondences() { return m_teaser_correspondences; }

    // 安全检查相关方法
    void updateTimestamp(double timestamp);
    bool checkTransformSafety(const M4F &current_transform, const M4F &previous_transform,
                              double dt, const std::string &stage_name);

private:
    // 全局配准相关方法
    bool globalRegistration(M4F &guess);
    void computeFPFH(CloudType::Ptr cloud,
                     pcl::PointCloud<pcl::FPFHSignature33>::Ptr features);

    // ICP 辅助方法：对单个初始位姿执行完整的 ICP 流程
    // 返回：是否成功，粗匹配评分，细匹配评分
    bool performICPForCandidate(const M4F& initial_guess, M4F& result_pose,
                                double& rough_score, double& refine_score);

    // 线程安全的 ICP 方法：使用独立的 ICP 对象和点云副本
    bool performICPForCandidateThreadSafe(const M4F& initial_guess,
                                         const CloudType::Ptr& rough_inp,
                                         const CloudType::Ptr& rough_tgt,
                                         const CloudType::Ptr& refine_inp,
                                         const CloudType::Ptr& refine_tgt,
                                         M4F& result_pose,
                                         double& rough_score,
                                         double& refine_score);

    // Scan Context 相关方法
    void buildScanContextDatabase(const CloudType::Ptr& map_cloud);
    bool saveScanContextDatabase(const std::string& path);
    bool loadScanContextDatabase(const std::string& path);
    bool getInitialPoseFromScanContext(const CloudType::Ptr& scan, M4F& result_pose);
    bool getInitialPosesFromScanContext(
        const CloudType::Ptr& scan, std::vector<M4F>& result_poses);

    ICPConfig m_config;
    pcl::VoxelGrid<PointType> m_voxel_filter;
    // 将ICP对象替换为GICP对象
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> m_refine_icp;
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> m_rough_icp;
    // pcl::IterativeClosestPoint<PointType, PointType> m_refine_icp;
    // pcl::IterativeClosestPoint<PointType, PointType> m_rough_icp;
    CloudType::Ptr m_refine_inp;
    CloudType::Ptr m_rough_inp;
    CloudType::Ptr m_refine_tgt;
    CloudType::Ptr m_rough_tgt;
    std::string m_pcd_path;
    double m_fitness_score; // 添加此成员变量存储适应度评分
    bool m_use_gicp = true; // 控制是否使用GICP算法的标志

    // Scan Context Manager
    std::shared_ptr<ScanContextManager> m_sc_manager;
    std::vector<SCDatabaseEntry> m_sc_database;
    bool m_sc_database_loaded = false;

    // 存储中间匹配结果 - 用于可视化
    CloudType::Ptr m_teaser_source_cloud;  // TEASER++输入点云
    CloudType::Ptr m_teaser_target_cloud;  // TEASER++目标点云（降采样后的地图）
    M4F m_teaser_transform;
    M4F m_rough_transform;
    M4F m_refine_transform;
    bool m_has_teaser_result = false;
    bool m_has_rough_result = false;
    bool m_has_refine_result = false;

    // TEASER++对应关系 - source索引 -> target索引
    std::vector<std::pair<int, int>> m_teaser_correspondences;

    // 安全检查相关成员变量
    M4F m_last_transform = M4F::Identity();     // 上一帧的变换
    double m_last_timestamp = -1.0;             // 上一帧的时间戳
    bool m_has_last_transform = false;          // 是否有历史数据
    M4F m_last_guess = M4F::Identity();         // 上一帧的输入guess（odom）
    bool m_has_last_guess = false;
};
