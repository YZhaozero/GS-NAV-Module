#include "icp_localizer.h"
#include <iomanip>

ICPLocalizer::ICPLocalizer(const ICPConfig &config) : m_config(config)
{
    m_refine_inp.reset(new CloudType);
    m_refine_tgt.reset(new CloudType);
    m_rough_inp.reset(new CloudType);
    m_rough_tgt.reset(new CloudType);
    
    // 初始化点云指针
    m_teaser_src_cloud.reset(new CloudType);
    m_teaser_tgt_cloud.reset(new CloudType);
    m_teaser_corr_cloud.reset(new CloudType);
    
    m_gicp_rough_src_cloud.reset(new CloudType);
    m_gicp_rough_tgt_cloud.reset(new CloudType);
    m_gicp_refine_src_cloud.reset(new CloudType);
    m_gicp_refine_tgt_cloud.reset(new CloudType);

    // 配置 GICP
    m_rough_icp.setCorrespondenceRandomness(m_config.correspondence_randomness);
    m_rough_icp.setMaximumOptimizerIterations(m_config.maximum_optimizer_iterations);
    m_refine_icp.setCorrespondenceRandomness(m_config.correspondence_randomness);
    m_refine_icp.setMaximumOptimizerIterations(m_config.maximum_optimizer_iterations);
    
    m_rough_icp.setTransformationEpsilon(1e-8);
    m_rough_icp.setEuclideanFitnessEpsilon(1e-6);
    m_refine_icp.setTransformationEpsilon(1e-9);
    m_refine_icp.setEuclideanFitnessEpsilon(1e-7);

    // 【新增】加载 SC 数据库
    if (m_config.use_scan_context && !m_config.sc_database_path.empty()) {
        loadSCDatabase(m_config.sc_database_path);
    }
}

// =================== SC 数据库管理 ===================

bool ICPLocalizer::loadSCDatabase(const std::string &db_path)
{
    m_keyframes.clear();
    std::filesystem::path root(db_path);
    if (!std::filesystem::exists(root)) {
        PCL_WARN_STREAM("SC数据库目录不存在: " << db_path << std::endl);
        return false;
    }

    PCL_INFO_STREAM("📂 正在加载 Scan Context 数据库: " << db_path << std::endl);

    // 读取 poses.txt
    std::ifstream pose_file(root / "poses.txt");
    if (!pose_file.is_open()) {
        PCL_ERROR_STREAM("无法打开 poses.txt" << std::endl);
        return false;
    }

    double t, x, y, z, qx, qy, qz, qw;
    int idx = 0;
    // 假设格式: timestamp x y z qx qy qz qw
    while (pose_file >> t >> x >> y >> z >> qx >> qy >> qz >> qw) {
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << idx << ".sc";
        std::string sc_filename = ss.str();
        std::filesystem::path sc_path = root / "data" / sc_filename;

        if (!std::filesystem::exists(sc_path)) break;

        // 读取 SC 矩阵
        Eigen::MatrixXd sc(m_config.sc_num_rings, m_config.sc_num_sectors);
        std::ifstream sc_fs(sc_path);
        for(int r=0; r<m_config.sc_num_rings; ++r) {
            for(int s=0; s<m_config.sc_num_sectors; ++s) {
                sc_fs >> sc(r, s);
            }
        }
        sc_fs.close();

        // 构建位姿
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        Eigen::Quaternionf q(qw, qx, qy, qz);
        pose.block<3,3>(0,0) = q.toRotationMatrix();
        pose.block<3,1>(0,3) = Eigen::Vector3f(x, y, z);

        SCKeyframe kf;
        kf.sc = sc;
        kf.pose = pose;
        kf.index = idx;
        m_keyframes.push_back(kf);

        idx++;
    }
    
    PCL_INFO_STREAM("✅ 数据库加载完成，共 " << m_keyframes.size() << " 个关键帧。" << std::endl);
    return true;
}

bool ICPLocalizer::relocalizeWithSC(const Eigen::MatrixXd &curr_sc, Eigen::Matrix4f &result_pose)
{
    if (m_keyframes.empty()) {
        PCL_WARN_STREAM("⚠️ SC 数据库为空，跳过重定位。" << std::endl);
        return false;
    }

    int best_idx = -1;
    double min_dist = 1000.0;
    int best_shift = 0;

    // 线性搜索 (对于几千帧以内很快)
    for (const auto &kf : m_keyframes) {
        auto res = getSCScoreAndShift(curr_sc, kf.sc);
        if (res.first < min_dist) {
            min_dist = res.first;
            best_shift = res.second;
            best_idx = kf.index;
        }
    }

    if (best_idx == -1) return false;

    // 输出调试信息
    double angle_res_deg = 360.0 / m_config.sc_num_sectors;
    double yaw_diff_deg = angle_res_deg * best_shift;
    
    PCL_INFO_STREAM("🔍 SC 最佳匹配: Frame " << best_idx 
                   << " | Dist: " << min_dist 
                   << " (阈值: " << m_config.sc_dist_thresh << ")"
                   << " | Yaw Shift: " << yaw_diff_deg << "°" << std::endl);

    if (min_dist < m_config.sc_dist_thresh) {
        const auto &match_kf = m_keyframes[best_idx];
        
        // 计算偏航角修正
        // SC 的 shift 是 "当前Scan" 还需要旋转多少度才能对齐 "Keyframe"
        // 或者是 "当前Scan" 相当于 "Keyframe" 旋转了多少度
        // 通常：Keyframe SC 也就是 map 坐标系下的 view
        // 这里的 shift 含义是：column shift。如果 shift=10，意味着 current 需要右移10列才能和 database 匹配。
        // 这意味着 current 的 0度 对应 database 的 10度。
        // 所以 current 相对 database 旋转了 +shift 角度。
        
        double yaw_diff = (2.0 * M_PI / m_config.sc_num_sectors) * best_shift;
        
        Eigen::Matrix4f yaw_rot = Eigen::Matrix4f::Identity();
        yaw_rot.block<3,3>(0,0) = Eigen::AngleAxisf(yaw_diff, Eigen::Vector3f::UnitZ()).toRotationMatrix();
        
        // 最终位姿 = 匹配帧位姿 * 旋转修正
        result_pose = match_kf.pose * yaw_rot;
        return true;
    }

    return false;
}

// =================== 核心 Align 函数 ===================

bool ICPLocalizer::align(M4F &guess)
{
    bool sc_success = false;
    M4F sc_pose = M4F::Identity();

    // 1. 优先尝试 SC 数据库重定位
    if (m_config.use_scan_context && !m_keyframes.empty()) {
        PCL_INFO_STREAM("🔄 [SC Reloc] 正在数据库中搜索当前位置..." << std::endl);
        Eigen::MatrixXd curr_sc = makeScanContext(m_rough_inp);
        
        if (relocalizeWithSC(curr_sc, sc_pose)) {
            PCL_INFO_STREAM("✅ SC 重定位成功！直接使用匹配位姿作为初值。" << std::endl);
            PCL_INFO_STREAM("   - Pose: " << sc_pose(0,3) << ", " << sc_pose(1,3) << std::endl);
            guess = sc_pose; // 【关键】直接覆盖外部传入的 guess
            sc_success = true;
        } else {
            PCL_WARN_STREAM("❌ SC 重定位失败 (未找到相似帧)。" << std::endl);
        }
    }

    // 2. 如果 SC 失败，且启用了全局配准 (TEASER)，则尝试 TEASER
    if (!sc_success && m_config.use_global_registration) {
        PCL_INFO_STREAM("尝试 TEASER++ 全局配准..." << std::endl);
        M4F teaser_guess = M4F::Identity();
        if (globalRegistration(teaser_guess)) {
            // TEASER 后处理：强制平面约束
            teaser_guess(2, 3) = 0.0f; 
            Eigen::Matrix3f rotation = teaser_guess.block<3,3>(0,0);
            float yaw = std::atan2(rotation(1,0), rotation(0,0));
            teaser_guess.block<3,3>(0,0) = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()).toRotationMatrix();
            
            guess = teaser_guess;
            PCL_INFO_STREAM("🛡️ TEASER++ 配准成功，使用其结果作为初值。" << std::endl);
        }
    }

    // 如果处于 "仅TEASER模式" 且 SC 也失败了，则退出
    if (m_teaser_only && !sc_success && !m_config.use_global_registration) {
         PCL_WARN_STREAM("TEASER Only 模式下未获得有效初值。" << std::endl);
         return false;
    }
    
    // 如果 TEASER Only 模式下获得了初值（SC 或 TEASER），直接返回，不跑 ICP
    if (m_teaser_only) {
        PCL_INFO_STREAM("🎉 [调试模式] 仅获取初值 (SC/TEASER)，跳过 GICP。" << std::endl);
        return true;
    }

    // 3. 运行 GICP (粗 + 精)
    if (m_refine_tgt->size() == 0 || m_rough_tgt->size() == 0) {
        std::cerr << "ICP失败：目标点云为空！" << std::endl;
        return false;
    }

    CloudType::Ptr aligned_cloud(new CloudType);
    
    // GICP 粗配准
    m_rough_icp.setMaxCorrespondenceDistance(2.0); 
    m_rough_icp.setMaximumIterations(m_config.rough_max_iteration);
    m_rough_icp.setInputSource(m_rough_inp);
    m_rough_icp.setInputTarget(m_rough_tgt);
    
    PCL_INFO_STREAM("🔵 开始 GICP 粗配准... Iter: " << m_config.rough_max_iteration << std::endl);
    m_rough_icp.align(*aligned_cloud, guess);

    if (!m_rough_icp.hasConverged()) {
        PCL_WARN_STREAM("❌ GICP(粗) 未收敛。" << std::endl);
        return false;
    }
    M4F rough_transform = m_rough_icp.getFinalTransformation();

    // GICP 精配准
    m_refine_icp.setMaxCorrespondenceDistance(1.0); 
    m_refine_icp.setMaximumIterations(m_config.refine_max_iteration);
    m_refine_icp.setInputSource(m_refine_inp);
    m_refine_icp.setInputTarget(m_refine_tgt);
    
    PCL_INFO_STREAM("🟢 开始 GICP 精配准..." << std::endl);
    m_refine_icp.align(*m_refine_inp, rough_transform);

    m_fitness_score = m_refine_icp.getFitnessScore();
    guess = m_refine_icp.getFinalTransformation();

    PCL_INFO_STREAM("🎉 最终定位成功！Score: " << m_fitness_score << std::endl);
    return m_refine_icp.hasConverged();
}

// =================== 辅助函数保持原样 ===================

bool ICPLocalizer::loadMap(const std::string &path) {
    if (!std::filesystem::exists(path)) return false;
    pcl::PCDReader reader;
    CloudType::Ptr cloud(new CloudType);
    reader.read(path, *cloud);
    
    if (m_config.refine_map_resolution > 0) {
        m_voxel_filter.setLeafSize(m_config.refine_map_resolution, m_config.refine_map_resolution, m_config.refine_map_resolution);
        m_voxel_filter.setInputCloud(cloud);
        m_voxel_filter.filter(*m_refine_tgt);
    } else pcl::copyPointCloud(*cloud, *m_refine_tgt);

    if (m_config.rough_map_resolution > 0) {
        m_voxel_filter.setLeafSize(m_config.rough_map_resolution, m_config.rough_map_resolution, m_config.rough_map_resolution);
        m_voxel_filter.setInputCloud(cloud);
        m_voxel_filter.filter(*m_rough_tgt);
    } else pcl::copyPointCloud(*cloud, *m_rough_tgt);
    return true;
}

void ICPLocalizer::setInput(const CloudType::Ptr &cloud) {
    if (m_config.refine_scan_resolution > 0) {
        m_voxel_filter.setLeafSize(m_config.refine_scan_resolution, m_config.refine_scan_resolution, m_config.refine_scan_resolution);
        m_voxel_filter.setInputCloud(cloud);
        m_voxel_filter.filter(*m_refine_inp);
    } else pcl::copyPointCloud(*cloud, *m_refine_inp);

    if (m_config.rough_scan_resolution > 0) {
        m_voxel_filter.setLeafSize(m_config.rough_scan_resolution, m_config.rough_scan_resolution, m_config.rough_scan_resolution);
        m_voxel_filter.setInputCloud(cloud);
        m_voxel_filter.filter(*m_rough_inp);
    } else pcl::copyPointCloud(*cloud, *m_rough_inp);
}

Eigen::MatrixXd ICPLocalizer::makeScanContext(CloudType::Ptr cloud) {
    int num_ring = m_config.sc_num_rings;
    int num_sector = m_config.sc_num_sectors;
    double max_radius = m_config.sc_max_radius;
    double gap_ring = max_radius / num_ring;
    double gap_sector = 2.0 * M_PI / num_sector;
    
    Eigen::MatrixXd sc = Eigen::MatrixXd::Zero(num_ring, num_sector);
    
    for (const auto& pt : cloud->points) {
        float r = std::sqrt(pt.x*pt.x + pt.y*pt.y);
        if (r >= max_radius || r <= 0) continue;
        float ang = std::atan2(pt.y, pt.x) + M_PI;
        if (ang >= 2.0 * M_PI) ang -= 2.0 * M_PI;
        
        int ring_idx = std::max(0, std::min(num_ring - 1, (int)(r / gap_ring)));
        int sector_idx = std::max(0, std::min(num_sector - 1, (int)(ang / gap_sector)));
        if (pt.z > sc(ring_idx, sector_idx)) sc(ring_idx, sector_idx) = pt.z;
    }
    return sc;
}

Eigen::MatrixXd ICPLocalizer::circShift(const Eigen::MatrixXd &mat, int shift) {
    if (shift == 0) return mat;
    Eigen::MatrixXd shifted = Eigen::MatrixXd::Zero(mat.rows(), mat.cols());
    int cols = mat.cols();
    int n = shift % cols;
    shifted.rightCols(n) = mat.leftCols(n);
    shifted.leftCols(cols - n) = mat.rightCols(cols - n);
    return shifted;
}

double ICPLocalizer::distDirectSC(const Eigen::MatrixXd &sc1, const Eigen::MatrixXd &sc2) {
    int num_cols = sc1.cols();
    double dist_sum = 0.0;
    int valid_cols = 0;
    for (int j = 0; j < num_cols; ++j) {
        Eigen::VectorXd col1 = sc1.col(j);
        Eigen::VectorXd col2 = sc2.col(j);
        double norm1 = col1.norm();
        double norm2 = col2.norm();
        if (norm1 == 0 || norm2 == 0) continue; 
        double cos_sim = col1.dot(col2) / (norm1 * norm2);
        dist_sum += (1.0 - cos_sim);
        valid_cols++;
    }
    return (valid_cols == 0) ? 1.0 : (dist_sum / valid_cols);
}

std::pair<double, int> ICPLocalizer::getSCScoreAndShift(const Eigen::MatrixXd &sc1, const Eigen::MatrixXd &sc2) {
    double min_dist = 1e9;
    int best_shift = 0;
    for (int i = 0; i < sc1.cols(); ++i) {
        Eigen::MatrixXd sc1_shifted = circShift(sc1, i);
        double dist = distDirectSC(sc1_shifted, sc2);
        if (dist < min_dist) {
            min_dist = dist;
            best_shift = i;
        }
    }
    return {min_dist, best_shift};
}

// globalRegistration 和 computeFPFH 函数内容保持原样，篇幅原因这里省略，请务必保留原文件中的实现！
// 请将原 icp_localizer.cpp 中 computeFPFH 和 globalRegistration 的完整实现复制到这里。
// ... (保留 TEASER++ 实现) ...

void ICPLocalizer::computeFPFH(CloudType::Ptr cloud,
                                pcl::PointCloud<pcl::FPFHSignature33>::Ptr features)
{
    float normal_radius = m_config.global_voxel_size * 2.5;
    if (normal_radius >= m_config.global_feature_radius) {
        normal_radius = m_config.global_feature_radius * 0.5;
    }
    pcl::NormalEstimationOMP<PointType, pcl::Normal> nest;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    nest.setRadiusSearch(normal_radius); 
    nest.setInputCloud(cloud);
    nest.compute(*normals);
    
    pcl::FPFHEstimationOMP<PointType, pcl::Normal, pcl::FPFHSignature33> fest;
    fest.setRadiusSearch(m_config.global_feature_radius);
    fest.setInputCloud(cloud);
    fest.setInputNormals(normals);
    fest.compute(*features);
}

bool ICPLocalizer::globalRegistration(M4F &guess)
{
    // ... 请复制原文件中的 TEASER++ 实现代码 ...
    // 这里为了简洁我写了关键部分，你需要把原文件这段完整贴回来
    
    CloudType::Ptr src_sparse(new CloudType);
    CloudType::Ptr tgt_sparse(new CloudType);
    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(m_config.global_voxel_size, m_config.global_voxel_size, m_config.global_voxel_size);
    vg.setInputCloud(m_rough_inp); vg.filter(*src_sparse);
    vg.setInputCloud(m_rough_tgt); vg.filter(*tgt_sparse);
    
    pcl::copyPointCloud(*src_sparse, *m_teaser_src_cloud);
    pcl::copyPointCloud(*tgt_sparse, *m_teaser_tgt_cloud);
    
    if (src_sparse->size() < 10 || tgt_sparse->size() < 10) return false;

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr tgt_features(new pcl::PointCloud<pcl::FPFHSignature33>);
    computeFPFH(src_sparse, src_features);
    computeFPFH(tgt_sparse, tgt_features);
    
    teaser::PointCloud src_teaser, tgt_teaser;
    for (const auto& p : src_sparse->points) src_teaser.push_back({p.x, p.y, p.z});
    for (const auto& p : tgt_sparse->points) tgt_teaser.push_back({p.x, p.y, p.z});

    teaser::Matcher matcher;
    auto correspondences = matcher.calculateCorrespondences(
            src_teaser, tgt_teaser, *src_features, *tgt_features,
            m_config.teaser_use_absolute_scale, m_config.teaser_use_crosscheck, 
            m_config.teaser_use_tuple_test, m_config.teaser_tuple_scale);
            
    // ... 保留高度过滤逻辑 ...
    std::vector<std::pair<int, int>> filtered_correspondences;
    for (const auto& corr : correspondences) {
        if (std::abs(src_teaser[corr.first].z - tgt_teaser[corr.second].z) < 0.5) {
            filtered_correspondences.push_back(corr);
        }
    }
    correspondences = filtered_correspondences;
    
    // Visualization data
    m_teaser_corr_cloud->clear();
    for (const auto& corr : correspondences) {
        PointType p1; p1.x=src_teaser[corr.first].x; p1.y=src_teaser[corr.first].y; p1.z=src_teaser[corr.first].z;
        PointType p2; p2.x=tgt_teaser[corr.second].x; p2.y=tgt_teaser[corr.second].y; p2.z=tgt_teaser[corr.second].z;
        m_teaser_corr_cloud->push_back(p1);
        m_teaser_corr_cloud->push_back(p2);
    }

    if (correspondences.size() < 3) return false;

    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = m_config.global_voxel_size;
    params.cbar2 = 1.0;
    params.estimate_scaling = false;
    params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_gnc_factor = 1.4;
    params.rotation_max_iterations = 100;
    params.rotation_cost_threshold = 1e-6;

    teaser::RobustRegistrationSolver solver(params);
    solver.solve(src_teaser, tgt_teaser, correspondences);
    auto solution = solver.getSolution();
    
    if (!solution.valid) return false;
    
    Eigen::Matrix4d res = Eigen::Matrix4d::Identity();
    res.block<3,3>(0,0) = solution.rotation;
    res.block<3,1>(0,3) = solution.translation;
    guess = res.cast<float>();
    m_teaser_transform = guess;
    return true;
}