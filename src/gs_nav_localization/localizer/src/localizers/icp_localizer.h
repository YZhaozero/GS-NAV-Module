#pragma once
#include "commons.h"
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <teaser/registration.h>
#include <teaser/matcher.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/transforms.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream> 

// 定义关键帧结构体，用于存储数据库
struct SCKeyframe {
    Eigen::MatrixXd sc;   // Scan Context 矩阵
    Eigen::Matrix4f pose; // 该帧对应的位姿
    int index;            // 索引
};

struct ICPConfig
{
    // 粗/精配准分辨率
    double refine_scan_resolution = 0.1;
    double refine_map_resolution = 0.1;
    double refine_score_thresh = 0.1;
    int refine_max_iteration = 10;

    double rough_scan_resolution = 0.25;
    double rough_map_resolution = 0.25;
    double rough_score_thresh = 0.2;
    int rough_max_iteration = 5;
    
    // GICP参数
    int correspondence_randomness = 20;
    int maximum_optimizer_iterations = 20;
    
    // 全局配准 (TEASER++) 参数
    bool use_global_registration = false;
    double global_voxel_size = 0.5;
    double global_feature_radius = 1.0;
    double global_score_thresh = 20.0;

    // TEASER++ 匹配参数
    bool teaser_use_absolute_scale = false;
    bool teaser_use_crosscheck = true;
    bool teaser_use_tuple_test = true;
    double teaser_tuple_scale = 0.95;

    // ======= Scan Context 参数 =======
    bool use_scan_context = true;            // 是否启用 Scan Context
    std::string sc_database_path = "";       // 【新增】数据库路径
    double sc_dist_thresh = 0.4;             // SC 距离阈值
    double sc_max_radius = 80.0;             // SC 最大半径
    int sc_num_rings = 20;                   // 环数
    int sc_num_sectors = 60;                 // 扇区数
};

class ICPLocalizer
{
public:
    ICPLocalizer(const ICPConfig &config);
    
    bool loadMap(const std::string &path);
    // 【新增】加载 SC 数据库
    bool loadSCDatabase(const std::string &db_path);
    
    void setInput(const CloudType::Ptr &cloud);

    // 核心对齐函数
    bool align(M4F &guess);
    
    double getFitnessScore() { return m_fitness_score; }
    ICPConfig &config() { return m_config; }
    CloudType::Ptr roughMap() { return m_rough_tgt; }
    CloudType::Ptr refineMap() { return m_refine_tgt; }

    // TEASER++ 相关获取函数
    Eigen::Matrix4f getTeaserTransform() { return m_teaser_transform; }
    CloudType::Ptr teaserSourceCloud() { return m_teaser_src_cloud; }
    CloudType::Ptr teaserTargetCloud() { return m_teaser_tgt_cloud; }
    CloudType::Ptr teaserCorrespondenceCloud() { return m_teaser_corr_cloud; }
    
    // GICP 相关获取函数
    CloudType::Ptr gicpRoughSourceCloud() { return m_gicp_rough_src_cloud; }
    CloudType::Ptr gicpRoughTargetCloud() { return m_gicp_rough_tgt_cloud; }
    CloudType::Ptr gicpRefineSourceCloud() { return m_gicp_refine_src_cloud; }
    CloudType::Ptr gicpRefineTargetCloud() { return m_gicp_refine_tgt_cloud; }
    
    // 模式设置
    void setUseGICP(bool use_gicp) { m_use_gicp = use_gicp; }
    void setUseTeaserOnly(bool teaser_only) { m_teaser_only = teaser_only; }

private:
    // TEASER++ 全局配准
    bool globalRegistration(M4F &guess);
    void computeFPFH(CloudType::Ptr cloud, pcl::PointCloud<pcl::FPFHSignature33>::Ptr features);

    // ======= Scan Context 核心算法 =======
    Eigen::MatrixXd makeScanContext(CloudType::Ptr cloud);
    double distDirectSC(const Eigen::MatrixXd &sc1, const Eigen::MatrixXd &sc2);
    std::pair<double, int> getSCScoreAndShift(const Eigen::MatrixXd &sc1, const Eigen::MatrixXd &sc2);
    Eigen::MatrixXd circShift(const Eigen::MatrixXd &mat, int shift);
    
    // 【新增】基于数据库的重定位
    bool relocalizeWithSC(const Eigen::MatrixXd &curr_sc, Eigen::Matrix4f &result_pose);

    ICPConfig m_config;
    pcl::VoxelGrid<PointType> m_voxel_filter;
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> m_refine_icp;
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> m_rough_icp;
    
    CloudType::Ptr m_refine_inp;
    CloudType::Ptr m_rough_inp;
    CloudType::Ptr m_refine_tgt;
    CloudType::Ptr m_rough_tgt;
    
    double m_fitness_score; 
    bool m_use_gicp = true; 
    bool m_teaser_only = false; 

    Eigen::Matrix4f m_teaser_transform = Eigen::Matrix4f::Identity();

    // TEASER++ Cloud Data
    CloudType::Ptr m_teaser_src_cloud;
    CloudType::Ptr m_teaser_tgt_cloud;
    CloudType::Ptr m_teaser_corr_cloud;
    
    // GICP Cloud Data
    CloudType::Ptr m_gicp_rough_src_cloud;
    CloudType::Ptr m_gicp_rough_tgt_cloud;
    CloudType::Ptr m_gicp_refine_src_cloud;
    CloudType::Ptr m_gicp_refine_tgt_cloud;

    // 【新增】数据库存储
    std::vector<SCKeyframe> m_keyframes;
};
