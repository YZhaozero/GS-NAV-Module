#include "icp_localizer.h"
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <fstream>
#include <omp.h>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <cstdio>  // 用于文件输出

ICPLocalizer::ICPLocalizer(const ICPConfig &config) : m_config(config)
{
    m_refine_inp.reset(new CloudType);
    m_refine_tgt.reset(new CloudType);
    m_rough_inp.reset(new CloudType);
    m_rough_tgt.reset(new CloudType);

    // 🔍 调试：打印 sc_max_candidates 配置值
    std::cout << "🔍 ICPLocalizer 初始化: sc_max_candidates=" << m_config.sc_max_candidates << std::endl;
    PCL_INFO_STREAM("ICPLocalizer 初始化: sc_max_candidates=" << m_config.sc_max_candidates << std::endl);

    // 初始化Scan Context管理器
    SCConfig sc_cfg;
    sc_cfg.max_radius = m_config.sc_max_radius;
    sc_cfg.dist_thresh = m_config.sc_dist_thresh;
    m_sc_manager = std::make_shared<ScanContextManager>(sc_cfg);

    // 初始化中间结果点云 - 用于可视化
    m_teaser_source_cloud.reset(new CloudType);
    m_teaser_target_cloud.reset(new CloudType);
    m_teaser_transform = M4F::Identity();
    m_rough_transform = M4F::Identity();
    m_refine_transform = M4F::Identity();

    // 配置GICP特有参数
    m_rough_icp.setCorrespondenceRandomness(m_config.correspondence_randomness);
    m_rough_icp.setMaximumOptimizerIterations(m_config.maximum_optimizer_iterations);
    m_refine_icp.setCorrespondenceRandomness(m_config.correspondence_randomness);
    m_refine_icp.setMaximumOptimizerIterations(m_config.maximum_optimizer_iterations);

    // 🔥 关键修复：设置最大对应点距离 (MaxCorrespondenceDistance)
    // 防止GICP匹配到过远的点导致发散
    // 粗匹配：允许较大误差 (例如 2.0m)
    m_rough_icp.setMaxCorrespondenceDistance(m_config.rough_scan_resolution * 10.0);
    // 🔥 L型走廊场景调参：精匹配对应距离设为固定值1.2m（0.8~1.5m推荐范围）
    // 太大容易跑飞，太小对应点太少容易提前失败
    m_refine_icp.setMaxCorrespondenceDistance(1.2f);

    // 设置变换收敛阈值和适应度阈值
    m_rough_icp.setTransformationEpsilon(1e-8);
    m_rough_icp.setEuclideanFitnessEpsilon(1e-6);
    // 🔥 L型走廊场景调参：放宽收敛阈值（调大一个数量级），避免在平坦谷底过早停止
    // TransformationEpsilon: 1e-7 -> 1e-6, EuclideanFitnessEpsilon: 1e-5 -> 1e-4
    m_refine_icp.setTransformationEpsilon(1e-6);
    m_refine_icp.setEuclideanFitnessEpsilon(1e-4);
}

bool ICPLocalizer::loadMap(const std::string &path)
{
    if (!std::filesystem::exists(path))
    {
        std::cerr << "地图文件不存在: " << path << std::endl;
        return false;
    }

    CloudType::Ptr cloud(new CloudType);
    if (pcl::io::loadPCDFile(path, *cloud) == -1)
    {
        std::cerr << "加载地图失败: " << path << std::endl;
        return false;
    }

    std::cout << "成功加载地图: " << path << " (" << cloud->size() << " 点)" << std::endl;

    if (m_config.refine_map_resolution > 0)
    {
        m_voxel_filter.setLeafSize(m_config.refine_map_resolution, m_config.refine_map_resolution, m_config.refine_map_resolution);
        m_voxel_filter.setInputCloud(cloud);
        m_voxel_filter.filter(*m_refine_tgt);
    }
    else
    {
        pcl::copyPointCloud(*cloud, *m_refine_tgt);
    }

    if (m_config.rough_map_resolution > 0)
    {
        m_voxel_filter.setLeafSize(m_config.rough_map_resolution, m_config.rough_map_resolution, m_config.rough_map_resolution);
        m_voxel_filter.setInputCloud(cloud);
        m_voxel_filter.filter(*m_rough_tgt);
    }
    else
    {
        pcl::copyPointCloud(*cloud, *m_rough_tgt);
    }

    // 构建或加载Scan Context数据库
    if (m_config.use_scan_context) {
        std::string sc_cache_path = path + ".sc";
        if (std::filesystem::exists(sc_cache_path)) {
            PCL_INFO_STREAM("Found Scan Context cache: " << sc_cache_path << std::endl);
            if (!loadScanContextDatabase(sc_cache_path)) {
                PCL_WARN_STREAM("Failed to load SC cache, rebuilding..." << std::endl);
                buildScanContextDatabase(cloud);
                saveScanContextDatabase(sc_cache_path);
            }
        } else {
            PCL_INFO_STREAM("Building Scan Context database from map..." << std::endl);
            buildScanContextDatabase(cloud);
            saveScanContextDatabase(sc_cache_path);
        }
    }

    return true;
}

void ICPLocalizer::buildScanContextDatabase(const CloudType::Ptr& map_cloud) {
    // 1. 计算地图边界
    PointType min_pt, max_pt;
    pcl::getMinMax3D(*map_cloud, min_pt, max_pt);

    PCL_INFO_STREAM("Map bounds: [" << min_pt.x << ", " << max_pt.x << "] x ["
                                    << min_pt.y << ", " << max_pt.y << "]" << std::endl);

    // 2. 生成采样网格点
    std::vector<Eigen::Vector3f> grid_points;
    double step = m_config.sc_grid_resolution; // 2.0m

    for (double x = min_pt.x; x <= max_pt.x; x += step) {
        for (double y = min_pt.y; y <= max_pt.y; y += step) {
            // 简单过滤：如果该点附近没有点云，则跳过（避免在空旷区域采样）
            // 这里为了速度，我们稍后在crop时判断
            grid_points.push_back(Eigen::Vector3f(x, y, 0.0f));
        }
    }

    PCL_INFO_STREAM("Generated " << grid_points.size() << " grid points. Generating SC descriptors..." << std::endl);

    // 3. 并行生成SC
    m_sc_database.resize(grid_points.size());
    std::atomic<int> valid_sc_count(0);

    // 使用OpenMP加速
    #pragma omp parallel for
    for (size_t i = 0; i < grid_points.size(); ++i) {
        Eigen::Vector3f center = grid_points[i];

        // Crop Box: 截取该网格点周围 sc_max_radius 范围内的点云
        pcl::CropBox<PointType> crop;
        crop.setMin(Eigen::Vector4f(center.x() - m_config.sc_max_radius, center.y() - m_config.sc_max_radius, -100.0f, 1.0f));
        crop.setMax(Eigen::Vector4f(center.x() + m_config.sc_max_radius, center.y() + m_config.sc_max_radius, 100.0f, 1.0f));
        crop.setInputCloud(map_cloud);

        CloudType::Ptr submap(new CloudType);
        crop.filter(*submap);

        // 如果点云太少，视为无效区域
        if (submap->size() < 100) {
            m_sc_database[i].pose = Eigen::Vector3f::Zero(); // Mark as invalid
            continue;
        }

        // 将submap变换到以center为原点
        CloudType::Ptr local_submap(new CloudType);
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translation() << -center.x(), -center.y(), 0.0f;
        pcl::transformPointCloud(*submap, *local_submap, transform);

        // 生成SC
        Eigen::MatrixXd sc = m_sc_manager->makeScanContext(local_submap);
        Eigen::VectorXd rk = m_sc_manager->makeRingkeyFromScancontext(sc);

        m_sc_database[i].pose = center;
        m_sc_database[i].descriptor = sc;
        m_sc_database[i].ringkey = rk;

        valid_sc_count++;
    }

    // 移除无效条目
    auto it = std::remove_if(
        m_sc_database.begin(), m_sc_database.end(),
        [](const SCDatabaseEntry& entry) { return entry.descriptor.size() == 0; });
    m_sc_database.erase(it, m_sc_database.end());

    m_sc_database_loaded = true;
    PCL_INFO_STREAM("Scan Context Database built. " << m_sc_database.size() << " valid frames." << std::endl);
}

bool ICPLocalizer::saveScanContextDatabase(const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return false;

    size_t size = m_sc_database.size();
    ofs.write((char*)&size, sizeof(size));

    for (const auto& entry : m_sc_database) {
        ofs.write((char*)entry.pose.data(), sizeof(float) * 3);

        int rows = entry.descriptor.rows();
        int cols = entry.descriptor.cols();
        ofs.write((char*)&rows, sizeof(int));
        ofs.write((char*)&cols, sizeof(int));
        ofs.write((char*)entry.descriptor.data(), sizeof(double) * rows * cols);
        // Ringkey can be recomputed
    }
    ofs.close();
    return true;
}

bool ICPLocalizer::loadScanContextDatabase(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;

    size_t size;
    ifs.read((char*)&size, sizeof(size));
    m_sc_database.clear();
    m_sc_database.reserve(size);

    for (size_t i = 0; i < size; ++i) {
        SCDatabaseEntry entry;
        ifs.read((char*)entry.pose.data(), sizeof(float) * 3);

        int rows, cols;
        ifs.read((char*)&rows, sizeof(int));
        ifs.read((char*)&cols, sizeof(int));

        entry.descriptor = Eigen::MatrixXd(rows, cols);
        ifs.read((char*)entry.descriptor.data(), sizeof(double) * rows * cols);

        entry.ringkey = m_sc_manager->makeRingkeyFromScancontext(entry.descriptor);
        m_sc_database.push_back(entry);
    }
    ifs.close();
    m_sc_database_loaded = true;
    PCL_INFO_STREAM("Loaded " << m_sc_database.size() << " SC frames from cache." << std::endl);
    return true;
}

bool ICPLocalizer::getInitialPoseFromScanContext(const CloudType::Ptr& scan, M4F& result_pose) {
    if (!m_sc_database_loaded || m_sc_database.empty()) {
        PCL_WARN_STREAM("SC Database empty!" << std::endl);
        return false;
    }

    // 1. Compute query SC
    Eigen::MatrixXd query_sc = m_sc_manager->makeScanContext(scan);
    Eigen::VectorXd query_rk = m_sc_manager->makeRingkeyFromScancontext(query_sc);

    // 2. Search (Ring Key based fast candidates)
    std::vector<std::pair<double, int>> candidates; // distance, index

    // RingKey 粗筛 (此处简单起见，遍历所有，实际可KD-Tree)
    // 但由于RingKey是向量，直接遍历计算距离也很快
    for (size_t i = 0; i < m_sc_database.size(); ++i) {
        double dist = (m_sc_database[i].ringkey - query_rk).norm();
        candidates.push_back({dist, (int)i});
    }

    std::sort(candidates.begin(), candidates.end());

    // 3. Fine matching (Scan Context distance)
    int top_k = std::min((size_t)50, candidates.size()); // Check top 50 candidates
    double min_dist = 1000.0;
    int best_idx = -1;
    int best_yaw_idx = 0;

    #pragma omp parallel for
    for (int k = 0; k < top_k; ++k) {
        int idx = candidates[k].second;
        auto [dist, yaw_idx] = m_sc_manager->distanceBtnScanContext(query_sc, m_sc_database[idx].descriptor);

        #pragma omp critical
        {
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = idx;
                best_yaw_idx = yaw_idx;
            }
        }
    }

    PCL_INFO_STREAM("SC Search: Min Dist = " << min_dist << " (Thresh: " << m_config.sc_dist_thresh << ")" << std::endl);

    if (min_dist < m_config.sc_dist_thresh && best_idx != -1) {
        // Found a match!
        Eigen::Vector3f best_pose = m_sc_database[best_idx].pose;

        // Calculate yaw
        // SC yaw index corresponds to 2*PI / num_sectors
        double yaw_step = 2.0 * M_PI / 60.0; // default config
        double yaw_correction = best_yaw_idx * yaw_step;

        PCL_INFO_STREAM("SC Match Found! Pose: [" << best_pose.transpose() << "], Yaw Shift: " << yaw_correction * 180.0/M_PI << " deg" << std::endl);

        result_pose = M4F::Identity();
        result_pose.block<3,1>(0,3) = best_pose;

        // Set Rotation (Yaw only)
        Eigen::AngleAxisf rot(yaw_correction, Eigen::Vector3f::UnitZ());
        result_pose.block<3,3>(0,0) = rot.toRotationMatrix();

        return true;
    }

    return false;
}

bool ICPLocalizer::getInitialPosesFromScanContext(
    const CloudType::Ptr& scan, std::vector<M4F>& result_poses) {
    result_poses.clear();

    if (!m_sc_database_loaded || m_sc_database.empty()) {
        PCL_WARN_STREAM("SC Database empty!" << std::endl);
        return false;
    }

    // 1. Compute query SC
    Eigen::MatrixXd query_sc = m_sc_manager->makeScanContext(scan);
    Eigen::VectorXd query_rk = m_sc_manager->makeRingkeyFromScancontext(query_sc);

    // 2. Search (Ring Key based fast candidates)
    std::vector<std::pair<double, int>> candidates; // distance, index

    // RingKey 粗筛
    for (size_t i = 0; i < m_sc_database.size(); ++i) {
        double dist = (m_sc_database[i].ringkey - query_rk).norm();
        candidates.push_back({dist, (int)i});
    }

    std::sort(candidates.begin(), candidates.end());

    // 3. Fine matching (Scan Context distance) - 对所有候选点进行 Fine matching
    // 🔥 修改：先对所有候选点进行 Fine matching，然后排序取前 sc_max_candidates 个
    std::vector<std::tuple<double, int, int>> sc_results; // dist, idx, yaw_idx

    PCL_INFO_STREAM("SC Search: 配置 sc_max_candidates=" << m_config.sc_max_candidates
                   << ", Ring Key 候选数=" << candidates.size()
                   << ", 将对所有候选点进行 Fine matching..." << std::endl);

    // 🔥 对所有候选点进行 Fine matching
    #pragma omp parallel for
    for (size_t k = 0; k < candidates.size(); ++k) {
        int idx = candidates[k].second;
        auto [dist, yaw_idx] = m_sc_manager->distanceBtnScanContext(query_sc, m_sc_database[idx].descriptor);

        #pragma omp critical
        {
            sc_results.push_back({dist, idx, yaw_idx});
        }
    }

    // 按评分排序
    std::sort(sc_results.begin(), sc_results.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

    PCL_INFO_STREAM("SC Search: Fine matching 完成，收集到 " << sc_results.size() << " 个候选结果" << std::endl);

    if (sc_results.empty()) {
        return false;
    }

    double min_dist = std::get<0>(sc_results[0]);
    PCL_INFO_STREAM("SC Search: Min Dist = " << min_dist << " (Thresh: " << m_config.sc_dist_thresh << ")" << std::endl);

    if (min_dist > m_config.sc_dist_thresh) {
        PCL_WARN_STREAM("SC Search: 最优候选仍超过绝对阈值，拒绝本次SC结果" << std::endl);
        return false;
    }

    const double relative_limit = min_dist + m_config.sc_similar_candidates_thresh;
    sc_results.erase(
        std::remove_if(
            sc_results.begin(), sc_results.end(),
            [&](const auto& result) {
                const double score = std::get<0>(result);
                return score > m_config.sc_dist_thresh || score > relative_limit;
            }),
        sc_results.end());

    const size_t configured_max = static_cast<size_t>(std::max(1, m_config.sc_max_candidates));
    const size_t max_candidates = std::min(sc_results.size(), configured_max);
    sc_results.resize(max_candidates);
    PCL_INFO_STREAM("SC Search: 阈值过滤后保留 " << max_candidates
                    << " 个候选（绝对阈值=" << m_config.sc_dist_thresh
                    << ", 相对阈值=" << relative_limit << ")" << std::endl);

    // 打印所有候选的评分分布
    size_t print_count = std::min((size_t)20, sc_results.size());
    PCL_INFO_STREAM("SC Search: 所有候选评分分布（前" << print_count << "个）:" << std::endl);
    for (size_t i = 0; i < print_count; ++i) {
        double dist = std::get<0>(sc_results[i]);
        int idx = std::get<1>(sc_results[i]);
        std::string status = (dist < m_config.sc_dist_thresh) ? " ✅" : " ❌";
        PCL_INFO_STREAM("  候选 #" << (i+1) << ": 评分=" << dist
                       << ", 数据库索引=" << idx << status << std::endl);
    }
    if (sc_results.size() > print_count) {
        PCL_INFO_STREAM("  ... (共 " << sc_results.size() << " 个候选，仅显示前 " << print_count << " 个)" << std::endl);
    }

    // 4. 🔥 返回前 sc_max_candidates 个候选（已在上一步完成截取）
    std::vector<std::tuple<double, int, int>> selected_candidates = sc_results;
    PCL_INFO_STREAM("SC Search: 🔥 返回前 " << selected_candidates.size() << " 个候选（按 Fine matching 评分从低到高）" << std::endl);

    if (selected_candidates.empty()) {
        return false;
    }

    // 5. 为每个候选生成位姿
    double yaw_step = 2.0 * M_PI / 60.0; // default config

    PCL_INFO_STREAM("📋 所有候选点详细信息：" << std::endl);
    PCL_INFO_STREAM("ℹ️  注意：候选点坐标是网格中心（网格分辨率: " << m_config.sc_grid_resolution
                   << " m），不是精确传感器位置。ICP会从网格中心开始匹配到实际位置。" << std::endl);
    for (size_t i = 0; i < selected_candidates.size(); ++i) {
        const auto& result = selected_candidates[i];
        double dist = std::get<0>(result);
        int idx = std::get<1>(result);
        int yaw_idx = std::get<2>(result);

        Eigen::Vector3f pose = m_sc_database[idx].pose;
        double yaw_correction = yaw_idx * yaw_step;

        // 计算最终位姿的 yaw 角
        Eigen::Matrix3f rot_matrix = Eigen::AngleAxisf(yaw_correction, Eigen::Vector3f::UnitZ()).toRotationMatrix();
        float final_yaw = std::atan2(rot_matrix(1,0), rot_matrix(0,0)) * 180.0 / M_PI;

        M4F candidate_pose = M4F::Identity();
        candidate_pose.block<3,1>(0,3) = pose;
        candidate_pose.block<3,3>(0,0) = rot_matrix;

        result_poses.push_back(candidate_pose);

        PCL_INFO_STREAM("  [" << (i+1) << "] 候选点 #" << (i+1) << ":" << std::endl
                       << "      - 坐标 (x, y, z): [" << pose.x() << ", " << pose.y() << ", " << pose.z() << "] m" << std::endl
                       << "      - 航向角 (yaw): " << final_yaw << " deg" << std::endl
                       << "      - Yaw 修正: " << yaw_correction * 180.0/M_PI << " deg" << std::endl
                       << "      - SC 评分: " << dist << std::endl
                       << "      - 数据库索引: " << idx << std::endl);
    }
    PCL_INFO_STREAM("📋 共 " << result_poses.size() << " 个候选点" << std::endl);

    return !result_poses.empty();
}

bool ICPLocalizer::performICPForCandidate(const M4F& initial_guess, M4F& result_pose,
                                          double& rough_score, double& refine_score) {
    // 检查点云是否为空
    if (m_refine_tgt->size() == 0 || m_rough_tgt->size() == 0) {
        PCL_WARN_STREAM("ICP候选验证失败: 点云为空" << std::endl);
        return false;
    }

    CloudType::Ptr aligned_cloud(new CloudType);

    // 🔥 关键修复：候选验证阶段使用更大的对应距离，因为初始位姿误差可能很大（8-27m）
    // 保存原始对应距离
    float original_rough_corr_dist = m_rough_icp.getMaxCorrespondenceDistance();
    float original_refine_corr_dist = m_refine_icp.getMaxCorrespondenceDistance();

    // 临时增加对应距离（允许更大的初始误差）
    // 🔥 改进：候选验证阶段使用更宽松的参数
    m_rough_icp.setMaxCorrespondenceDistance(std::max(5.0f, static_cast<float>(m_config.rough_scan_resolution * 25.0))); // 增加到5m或25倍分辨率
    // 🔥 L型走廊场景调参：候选验证阶段精匹配对应距离设为1.5m（推荐范围上限）
    // 候选验证时初始位姿误差可能较大，但精匹配仍不应太大以免跑飞
    m_refine_icp.setMaxCorrespondenceDistance(1.5f);

    // 粗匹配
    m_rough_icp.setMaximumIterations(m_config.rough_max_iteration);
    m_rough_icp.setInputSource(m_rough_inp);
    m_rough_icp.setInputTarget(m_rough_tgt);
    m_rough_icp.align(*aligned_cloud, initial_guess);

    if (!m_rough_icp.hasConverged()) {
        // 恢复原始对应距离
        m_rough_icp.setMaxCorrespondenceDistance(original_rough_corr_dist);
        m_refine_icp.setMaxCorrespondenceDistance(original_refine_corr_dist);
        PCL_WARN_STREAM("ICP候选验证失败: 粗匹配未收敛 (对应距离: "
                       << std::max(5.0f, static_cast<float>(m_config.rough_scan_resolution * 25.0)) << " m)" << std::endl);
        return false;
    }

    rough_score = m_rough_icp.getFitnessScore();

    // 🔥 计算粗匹配的位置变化（用于动态调整精匹配对应距离）
    Eigen::Vector3f initial_pos = initial_guess.block<3,1>(0,3);
    M4F rough_transform_result = m_rough_icp.getFinalTransformation();
    Eigen::Vector3f rough_trans = rough_transform_result.block<3,1>(0,3);
    float position_change = (rough_trans - initial_pos).norm();

    // 🔍 验证变换矩阵的正确性（检查是否有异常）
    Eigen::Matrix3f rough_rot = rough_transform_result.block<3,3>(0,0);
    float rot_determinant = rough_rot.determinant();
    Eigen::Matrix3f rot_orthogonality_check = rough_rot * rough_rot.transpose();
    Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();
    float orthogonality_error = (rot_orthogonality_check - identity).norm();

    // 检查旋转矩阵的缩放（应该是1.0）
    Eigen::Vector3f rot_scale;
    rot_scale(0) = rough_rot.col(0).norm();
    rot_scale(1) = rough_rot.col(1).norm();
    rot_scale(2) = rough_rot.col(2).norm();
    float max_scale_deviation = std::max({std::abs(rot_scale(0) - 1.0f),
                                          std::abs(rot_scale(1) - 1.0f),
                                          std::abs(rot_scale(2) - 1.0f)});

    // 🔍 计算粗匹配输入点云的几何中心（用于诊断）
    Eigen::Vector4f rough_inp_centroid_before;
    pcl::compute3DCentroid(*m_rough_inp, rough_inp_centroid_before);
    Eigen::Vector3f rough_inp_center_before = rough_inp_centroid_before.head<3>();

    // 🔍 计算粗匹配输入点云经过粗匹配变换后的几何中心
    CloudType::Ptr rough_inp_transformed(new CloudType);
    pcl::transformPointCloud(*m_rough_inp, *rough_inp_transformed, rough_transform_result);
    Eigen::Vector4f rough_inp_centroid_after;
    pcl::compute3DCentroid(*rough_inp_transformed, rough_inp_centroid_after);
    Eigen::Vector3f rough_inp_center_after = rough_inp_centroid_after.head<3>();

    // 如果点云中心接近原点，变换后的中心应该接近位姿位置
    float center_to_origin_dist = rough_inp_center_before.norm();
    float transformed_center_to_pose_dist = (rough_inp_center_after - rough_trans).norm();

    // 🔍 计算旋转对点云中心的影响
    Eigen::Vector3f center_rotation_effect = rough_rot * rough_inp_center_before;
    Eigen::Vector3f expected_transformed_center = center_rotation_effect + rough_trans;
    float expected_vs_actual_diff = (expected_transformed_center - rough_inp_center_after).norm();

    // 🔥 打印粗匹配结果（无论成功或失败都打印）
    float rough_yaw = std::atan2(rough_rot(1,0), rough_rot(0,0)) * 180.0 / M_PI;
    PCL_INFO_STREAM("    🔵 粗匹配: 评分=" << std::fixed << std::setprecision(4) << rough_score
                   << " | 位置=[" << std::setprecision(2) << rough_trans.x() << ", "
                   << rough_trans.y() << ", " << rough_trans.z() << "] m | 航向="
                   << std::setprecision(1) << rough_yaw << "° | 位置变化="
                   << std::setprecision(2) << position_change << " m" << std::endl
                   << "    🔍 变换矩阵验证:" << std::endl
                   << "       • 旋转矩阵行列式: " << std::setprecision(6) << rot_determinant
                   << " (应该接近1.0，异常值可能表示矩阵错误)" << std::endl
                   << "       • 旋转矩阵正交性误差: " << std::setprecision(6) << orthogonality_error
                   << " (应该接近0，异常值可能表示矩阵错误)" << std::endl
                   << "       • 旋转矩阵列向量长度: [" << std::setprecision(3)
                   << rot_scale(0) << ", " << rot_scale(1) << ", " << rot_scale(2) << "] (应该都是1.0)" << std::endl
                   << "       • 最大缩放偏差: " << std::setprecision(6) << max_scale_deviation
                   << " (应该接近0，异常值可能表示矩阵错误)" << std::endl
                   << "    🔍 粗匹配点云诊断:" << std::endl
                   << "       • 粗匹配输入点云原始几何中心: [" << std::setprecision(3)
                   << rough_inp_center_before(0) << ", " << rough_inp_center_before(1) << ", " << rough_inp_center_before(2) << "] m" << std::endl
                   << "       • 原始中心到原点距离: " << std::setprecision(3) << center_to_origin_dist << " m" << std::endl
                   << "       • 粗匹配输入点云变换后几何中心: [" << std::setprecision(3)
                   << rough_inp_center_after(0) << ", " << rough_inp_center_after(1) << ", " << rough_inp_center_after(2) << "] m" << std::endl
                   << "       • 变换后几何中心与粗匹配位姿位置差异: " << std::setprecision(3)
                   << transformed_center_to_pose_dist << " m" << std::endl
                   << "       • 旋转对中心的影响: [" << std::setprecision(3)
                   << center_rotation_effect(0) << ", " << center_rotation_effect(1) << ", " << center_rotation_effect(2) << "] m" << std::endl
                   << "       • 预期变换后中心: [" << std::setprecision(3)
                   << expected_transformed_center(0) << ", " << expected_transformed_center(1) << ", " << expected_transformed_center(2) << "] m" << std::endl
                   << "       • 预期vs实际差异: " << std::setprecision(3) << expected_vs_actual_diff << " m (应该接近0)" << std::endl);

    // ⚠️ 如果点云坐标异常大（距离原点很远），旋转会产生巨大位移
    if (center_to_origin_dist > 50.0f) {
        PCL_WARN_STREAM("    ⚠️  点云坐标异常大！原始中心距离原点 " << center_to_origin_dist << " m" << std::endl
                       << "       • 这可能导致旋转时产生巨大位移" << std::endl
                       << "       • 建议检查点云坐标系是否正确" << std::endl);
    }

    // ⚠️ 如果变换后中心与位姿位置差异很大，说明变换矩阵应用可能有问题
    if (transformed_center_to_pose_dist > 10.0f) {
        PCL_WARN_STREAM("    ⚠️  变换后点云中心与位姿位置差异过大！差异=" << transformed_center_to_pose_dist << " m" << std::endl
                       << "       • 这可能导致精匹配失败" << std::endl
                       << "       • 可能原因：点云坐标异常大，旋转产生巨大位移" << std::endl);
    }

    // ⚠️ 如果变换矩阵异常，直接返回失败
    if (std::abs(rot_determinant - 1.0f) > 0.1f || orthogonality_error > 0.1f || max_scale_deviation > 0.1f) {
        PCL_WARN_STREAM("    ⚠️  变换矩阵异常！行列式=" << rot_determinant
                       << ", 正交性误差=" << orthogonality_error
                       << ", 缩放偏差=" << max_scale_deviation << std::endl
                       << "       • 这可能导致点云变换后位置错误" << std::endl
                       << "       • 变换矩阵:\n" << rough_transform_result << std::endl);
        // 不直接返回，继续执行看看精匹配能否修复
    }

    const float candidate_rough_thresh = m_config.rough_score_thresh;
    if (rough_score > candidate_rough_thresh) {
        // 恢复原始对应距离
        m_rough_icp.setMaxCorrespondenceDistance(original_rough_corr_dist);
        m_refine_icp.setMaxCorrespondenceDistance(original_refine_corr_dist);
        PCL_WARN_STREAM("ICP候选验证失败: 粗匹配评分过高 (" << rough_score
                       << " > " << candidate_rough_thresh << ", 正常阈值: " << m_config.rough_score_thresh << ")" << std::endl);
        return false;
    }

    // 🔥 L型走廊场景调参：根据粗匹配评分动态调整精匹配对应距离
    // 对应距离控制在0.8~1.5m范围内，避免跑飞
    // 太大容易跑飞，太小对应点太少容易提前失败
    float adaptive_refine_corr_dist;
    if (rough_score < 0.05f) {
        // 粗匹配评分非常好，使用推荐范围上限1.5m
        adaptive_refine_corr_dist = 1.5f;
    } else if (rough_score < 0.1f) {
        // 粗匹配评分很好，使用1.3m
        adaptive_refine_corr_dist = 1.3f;
    } else if (rough_score < 0.5f) {
        // 粗匹配评分较好，使用1.1m
        adaptive_refine_corr_dist = 1.1f;
    } else {
        // 粗匹配评分较差，使用推荐范围下限0.9m
        adaptive_refine_corr_dist = 0.9f;
    }

    // 设置动态调整后的精匹配对应距离
    m_refine_icp.setMaxCorrespondenceDistance(adaptive_refine_corr_dist);
    PCL_INFO_STREAM("    🔧 精匹配对应距离: " << std::fixed << std::setprecision(2) << adaptive_refine_corr_dist
                   << " m (粗匹配评分=" << rough_score << ", 位置变化=" << position_change << " m)" << std::endl);

    M4F rough_result = m_rough_icp.getFinalTransformation();

    // 🔍 计算点云几何中心（用于诊断）
    Eigen::Vector4f refine_inp_centroid_before;
    pcl::compute3DCentroid(*m_refine_inp, refine_inp_centroid_before);
    Eigen::Vector3f refine_inp_center_before = refine_inp_centroid_before.head<3>();

    // 🔥 关键修复：先将精匹配输入点云变换到粗匹配后的坐标系
    // 这样精匹配的初始猜测就是Identity矩阵，避免坐标不一致导致的错误
    CloudType::Ptr refine_inp_transformed_by_rough(new CloudType);
    pcl::transformPointCloud(*m_refine_inp, *refine_inp_transformed_by_rough, rough_result);

    // 🔍 计算使用粗匹配变换后的精匹配输入点云几何中心
    Eigen::Vector4f refine_inp_centroid_after_rough;
    pcl::compute3DCentroid(*refine_inp_transformed_by_rough, refine_inp_centroid_after_rough);
    Eigen::Vector3f refine_inp_center_after_rough = refine_inp_centroid_after_rough.head<3>();

    // ⚠️ 检查点云坐标差异，如果差异过大，说明下采样导致坐标不一致
    float center_diff = (rough_inp_center_before - refine_inp_center_before).norm();
    if (center_diff > 5.0f) {
        PCL_WARN_STREAM("    ⚠️  粗匹配和精匹配点云原始中心差异过大: " << center_diff << " m" << std::endl
                       << "       • 这可能导致精匹配初始猜测错误" << std::endl
                       << "       • 已先将精匹配点云变换到粗匹配坐标系" << std::endl);
    }

    // 🔍 打印中心点诊断信息（对比粗匹配和精匹配的点云）
    Eigen::Matrix3f rough_rot_for_yaw = rough_result.block<3,3>(0,0);
    float rough_yaw_deg = std::atan2(rough_rot_for_yaw(1,0), rough_rot_for_yaw(0,0)) * 180.0 / M_PI;
    PCL_INFO_STREAM("    🔍 位姿验证（关键诊断）:" << std::endl
                   << "       • 粗匹配后位姿位置: [" << std::fixed << std::setprecision(3)
                   << rough_trans(0) << ", " << rough_trans(1) << ", " << rough_trans(2) << "] m" << std::endl
                   << "       • 粗匹配后位姿航向: " << std::setprecision(2) << rough_yaw_deg << "°" << std::endl
                   << "       📊 粗匹配点云（m_rough_inp）:" << std::endl
                   << "          - 原始几何中心: [" << std::setprecision(3)
                   << rough_inp_center_before(0) << ", " << rough_inp_center_before(1) << ", " << rough_inp_center_before(2) << "] m" << std::endl
                   << "          - 变换后几何中心: [" << std::setprecision(3)
                   << rough_inp_center_after(0) << ", " << rough_inp_center_after(1) << ", " << rough_inp_center_after(2) << "] m" << std::endl
                   << "          - 变换后中心与位姿位置差异: " << std::setprecision(3)
                   << (rough_inp_center_after - rough_trans).norm() << " m" << std::endl
                   << "       📊 精匹配点云（m_refine_inp）:" << std::endl
                   << "          - 原始几何中心: [" << std::setprecision(3)
                   << refine_inp_center_before(0) << ", " << refine_inp_center_before(1) << ", " << refine_inp_center_before(2) << "] m" << std::endl
                   << "          - 变换后几何中心: [" << std::setprecision(3)
                   << refine_inp_center_after_rough(0) << ", " << refine_inp_center_after_rough(1) << ", " << refine_inp_center_after_rough(2) << "] m" << std::endl
                   << "          - 变换后中心与位姿位置差异: " << std::setprecision(3)
                   << (refine_inp_center_after_rough - rough_trans).norm() << " m" << std::endl
                   << "       ⚠️  关键对比:" << std::endl
                   << "          - 粗匹配点云原始中心 vs 精匹配点云原始中心差异: " << std::setprecision(3)
                   << (rough_inp_center_before - refine_inp_center_before).norm() << " m" << std::endl
                   << "          - 粗匹配点云变换后中心 vs 精匹配点云变换后中心差异: " << std::setprecision(3)
                   << (rough_inp_center_after - refine_inp_center_after_rough).norm() << " m" << std::endl);

    // 🔥 精匹配：使用已变换到粗匹配坐标系的点云，初始猜测为Identity
    m_refine_icp.setMaximumIterations(m_config.refine_max_iteration);
    m_refine_icp.setInputSource(refine_inp_transformed_by_rough);  // 使用变换后的点云
    m_refine_icp.setInputTarget(m_refine_tgt);
    CloudType::Ptr aligned_refine_cloud(new CloudType);
    M4F identity_guess = M4F::Identity();  // 初始猜测为Identity（因为点云已经变换过了）
    m_refine_icp.align(*aligned_refine_cloud, identity_guess);

    // 🔥 计算最终变换矩阵：粗匹配变换 * 精匹配变换
    M4F final_refine_transform_candidate = m_refine_icp.getFinalTransformation();
    M4F combined_transform = final_refine_transform_candidate * rough_result;  // 组合变换

    // 🔍 计算对应点统计信息（用于诊断）
    // 注意：使用组合变换矩阵（粗匹配 + 精匹配）
    CloudType::Ptr transformed_cloud_candidate(new CloudType);
    pcl::transformPointCloud(*m_refine_inp, *transformed_cloud_candidate, combined_transform);

    // 使用KD-tree查找最近邻，计算对应点统计
    pcl::KdTreeFLANN<PointType> kdtree_candidate;
    kdtree_candidate.setInputCloud(m_refine_tgt);

    size_t num_correspondences_candidate = 0;
    size_t num_inliers_candidate = 0;
    double sum_squared_dist_candidate = 0.0;
    double sum_dist_candidate = 0.0;
    double max_dist_candidate = 0.0;
    float corr_dist_threshold_candidate = adaptive_refine_corr_dist;

    std::vector<int> indices_candidate(1);
    std::vector<float> sqr_dists_candidate(1);

    for (size_t i = 0; i < transformed_cloud_candidate->size(); ++i) {
        if (kdtree_candidate.nearestKSearch(transformed_cloud_candidate->points[i], 1, indices_candidate, sqr_dists_candidate) > 0) {
            double dist = std::sqrt(sqr_dists_candidate[0]);
            num_correspondences_candidate++;
            sum_squared_dist_candidate += sqr_dists_candidate[0];
            sum_dist_candidate += dist;
            max_dist_candidate = std::max(max_dist_candidate, dist);
            if (dist < corr_dist_threshold_candidate) {
                num_inliers_candidate++;
            }
        }
    }

    double avg_dist_candidate = num_correspondences_candidate > 0 ? sum_dist_candidate / num_correspondences_candidate : 0.0;
    double rmse_candidate = num_correspondences_candidate > 0 ? std::sqrt(sum_squared_dist_candidate / num_correspondences_candidate) : 0.0;
    double inlier_ratio_candidate = num_correspondences_candidate > 0 ? (100.0 * num_inliers_candidate / num_correspondences_candidate) : 0.0;

    // 🔍 计算变换矩阵的变化量（相对于粗匹配结果）
    M4F transform_diff_candidate = final_refine_transform_candidate.inverse() * identity_guess;
    Eigen::Vector3f transform_trans_diff_candidate = transform_diff_candidate.block<3,1>(0,3);
    float transform_translation_change_candidate = transform_trans_diff_candidate.norm();
    Eigen::Matrix3f transform_rot_diff_candidate = transform_diff_candidate.block<3,3>(0,0);
    float transform_rotation_change_candidate = (transform_rot_diff_candidate - Eigen::Matrix3f::Identity()).norm();

    // 🔥 如果精匹配未收敛，使用更大的对应距离多次重试
    if (!m_refine_icp.hasConverged()) {
        PCL_WARN_STREAM("    ⚠️  精匹配未收敛！详细分析:" << std::endl
                       << "      📊 收敛状态: hasConverged()=false" << std::endl
                       << "      📈 最终适应度评分: " << std::fixed << std::setprecision(6)
                       << m_refine_icp.getFitnessScore() << std::endl
                       << "      📐 变换矩阵平移变化: " << std::setprecision(6)
                       << transform_translation_change_candidate << " m (阈值: 1e-6)" << std::endl
                       << "      📐 变换矩阵旋转变化: " << std::setprecision(6)
                       << transform_rotation_change_candidate << " (阈值: 1e-6)" << std::endl
                       << "      📊 对应点统计（诊断信息）:" << std::endl
                       << "         • 输入点数: " << refine_inp_transformed_by_rough->size() << std::endl
                       << "         • 目标点数（地图）: " << m_refine_tgt->size() << std::endl
                       << "         • 找到对应点数: " << num_correspondences_candidate << " / " << refine_inp_transformed_by_rough->size()
                       << " (" << std::fixed << std::setprecision(1) << (100.0 * num_correspondences_candidate / refine_inp_transformed_by_rough->size()) << "%)" << std::endl
                       << "         • Inliers（距离 < " << std::setprecision(2) << corr_dist_threshold_candidate << "m）: "
                       << num_inliers_candidate << " / " << num_correspondences_candidate
                       << " (" << std::setprecision(1) << inlier_ratio_candidate << "%)" << std::endl
                       << "         • 平均对应距离: " << std::setprecision(3) << avg_dist_candidate << " m" << std::endl
                       << "         • 最大对应距离: " << std::setprecision(3) << max_dist_candidate << " m" << std::endl
                       << "         • RMSE: " << std::setprecision(3) << rmse_candidate << " m" << std::endl
                       << "         • Fitness Score (平方误差和): " << std::setprecision(2) << m_refine_icp.getFitnessScore() << std::endl);

        // 🔥 L型走廊场景调参：第一次重试，但不超过1.5m（推荐范围上限）
        float retry_corr_dist = std::min(1.5f, adaptive_refine_corr_dist * 1.2f);
        m_refine_icp.setMaxCorrespondenceDistance(retry_corr_dist);
        PCL_WARN_STREAM("      🔄 重试1: 对应距离=" << std::fixed << std::setprecision(2)
                       << retry_corr_dist << " m (原: " << adaptive_refine_corr_dist << " m)" << std::endl);
        m_refine_icp.align(*aligned_refine_cloud, identity_guess);  // 使用Identity初始猜测

        // 重新计算变换变化（相对于Identity）
        M4F retry1_final_transform = m_refine_icp.getFinalTransformation();
        M4F retry1_combined_transform = retry1_final_transform * rough_result;  // 组合变换
        M4F retry1_transform_diff = retry1_final_transform.inverse() * identity_guess;
        Eigen::Vector3f retry1_trans_diff = retry1_transform_diff.block<3,1>(0,3);
        float retry1_translation_change = retry1_trans_diff.norm();
        Eigen::Matrix3f retry1_rot_diff = retry1_transform_diff.block<3,3>(0,0);
        float retry1_rotation_change = (retry1_rot_diff - Eigen::Matrix3f::Identity()).norm();

        if (!m_refine_icp.hasConverged()) {
            PCL_WARN_STREAM("      ❌ 重试1失败: 适应度=" << std::fixed << std::setprecision(6)
                           << m_refine_icp.getFitnessScore() << ", 平移变化="
                           << std::setprecision(6) << retry1_translation_change << " m, 旋转变化="
                           << std::setprecision(6) << retry1_rotation_change << std::endl);

            // 🔥 L型走廊场景调参：第二次重试，但不超过1.5m（推荐范围上限）
            retry_corr_dist = std::min(1.5f, retry_corr_dist * 1.1f);
            m_refine_icp.setMaxCorrespondenceDistance(retry_corr_dist);
            PCL_WARN_STREAM("      🔄 重试2: 对应距离=" << std::fixed << std::setprecision(2)
                           << retry_corr_dist << " m" << std::endl);
            m_refine_icp.align(*aligned_refine_cloud, identity_guess);  // 使用Identity初始猜测

            // 重新计算变换变化（相对于Identity）
            M4F retry2_final_transform = m_refine_icp.getFinalTransformation();
            M4F retry2_combined_transform = retry2_final_transform * rough_result;  // 组合变换
            M4F retry2_transform_diff = retry2_final_transform.inverse() * identity_guess;
            Eigen::Vector3f retry2_trans_diff = retry2_transform_diff.block<3,1>(0,3);
            float retry2_translation_change = retry2_trans_diff.norm();
            Eigen::Matrix3f retry2_rot_diff = retry2_transform_diff.block<3,3>(0,0);
            float retry2_rotation_change = (retry2_rot_diff - Eigen::Matrix3f::Identity()).norm();

            if (!m_refine_icp.hasConverged()) {
                // 恢复原始对应距离
                m_rough_icp.setMaxCorrespondenceDistance(original_rough_corr_dist);
                m_refine_icp.setMaxCorrespondenceDistance(original_refine_corr_dist);
                PCL_WARN_STREAM("      ❌ 重试2失败: 适应度=" << std::fixed << std::setprecision(6)
                               << m_refine_icp.getFitnessScore() << ", 平移变化="
                               << std::setprecision(6) << retry2_translation_change << " m, 旋转变化="
                               << std::setprecision(6) << retry2_rotation_change << std::endl
                               << "ICP候选验证失败: 精匹配未收敛 (对应距离: " << std::fixed << std::setprecision(2)
                               << adaptive_refine_corr_dist << " m -> " << retry_corr_dist << " m, 粗匹配评分="
                               << rough_score << ", 位置变化=" << position_change << " m)" << std::endl);
                return false;
            } else {
                // 更新组合变换和适应度评分
                combined_transform = retry2_combined_transform;
                refine_score = m_refine_icp.getFitnessScore();
                PCL_INFO_STREAM("      ✅ 重试2成功: 适应度=" << std::fixed << std::setprecision(6)
                               << m_refine_icp.getFitnessScore() << ", 平移变化="
                               << std::setprecision(6) << retry2_translation_change << " m, 旋转变化="
                               << std::setprecision(6) << retry2_rotation_change << std::endl);
            }
        } else {
            // 更新组合变换和适应度评分
            combined_transform = retry1_combined_transform;
            refine_score = m_refine_icp.getFitnessScore();
            PCL_INFO_STREAM("      ✅ 重试1成功: 适应度=" << std::fixed << std::setprecision(6)
                           << m_refine_icp.getFitnessScore() << ", 平移变化="
                           << std::setprecision(6) << retry1_translation_change << " m, 旋转变化="
                           << std::setprecision(6) << retry1_rotation_change << std::endl);
        }
    } else {
        PCL_INFO_STREAM("    ✅ 精匹配收敛成功: 适应度=" << std::fixed << std::setprecision(6)
                       << m_refine_icp.getFitnessScore() << ", 平移变化="
                       << std::setprecision(6) << transform_translation_change_candidate << " m, 旋转变化="
                       << std::setprecision(6) << transform_rotation_change_candidate << std::endl
                       << "      📊 对应点统计（诊断信息）:" << std::endl
                       << "         • 输入点数: " << refine_inp_transformed_by_rough->size() << std::endl
                       << "         • 目标点数（地图）: " << m_refine_tgt->size() << std::endl
                       << "         • 找到对应点数: " << num_correspondences_candidate << " / " << refine_inp_transformed_by_rough->size()
                       << " (" << std::fixed << std::setprecision(1) << (100.0 * num_correspondences_candidate / refine_inp_transformed_by_rough->size()) << "%)" << std::endl
                       << "         • Inliers（距离 < " << std::setprecision(2) << corr_dist_threshold_candidate << "m）: "
                       << num_inliers_candidate << " / " << num_correspondences_candidate
                       << " (" << std::setprecision(1) << inlier_ratio_candidate << "%)" << std::endl
                       << "         • 平均对应距离: " << std::setprecision(3) << avg_dist_candidate << " m" << std::endl
                       << "         • 最大对应距离: " << std::setprecision(3) << max_dist_candidate << " m" << std::endl
                       << "         • RMSE: " << std::setprecision(3) << rmse_candidate << " m" << std::endl);
    }

    refine_score = m_refine_icp.getFitnessScore();

    // 🔥 打印精匹配结果（无论成功或失败都打印）
    // 注意：使用组合变换矩阵（粗匹配 + 精匹配）
    Eigen::Vector3f refine_trans = combined_transform.block<3,1>(0,3);
    Eigen::Matrix3f refine_rot = combined_transform.block<3,3>(0,0);
    float refine_yaw = std::atan2(refine_rot(1,0), refine_rot(0,0)) * 180.0 / M_PI;
    PCL_INFO_STREAM("    🟢 精匹配: 评分=" << std::fixed << std::setprecision(4) << refine_score
                   << " | 位置=[" << std::setprecision(2) << refine_trans.x() << ", "
                   << refine_trans.y() << ", " << refine_trans.z() << "] m | 航向="
                   << std::setprecision(1) << refine_yaw << "°" << std::endl);
    const float candidate_refine_thresh = m_config.refine_score_thresh;
    if (refine_score > candidate_refine_thresh) {
        // 恢复原始对应距离
        m_rough_icp.setMaxCorrespondenceDistance(original_rough_corr_dist);
        m_refine_icp.setMaxCorrespondenceDistance(original_refine_corr_dist);
        PCL_WARN_STREAM("ICP候选验证失败: 精匹配评分过高 (" << refine_score
                       << " > " << candidate_refine_thresh << ", 正常阈值: " << m_config.refine_score_thresh << ")" << std::endl);
        return false;
    }

    // 恢复原始对应距离
    m_rough_icp.setMaxCorrespondenceDistance(original_rough_corr_dist);
    m_refine_icp.setMaxCorrespondenceDistance(original_refine_corr_dist);

    // 🔥 返回组合变换矩阵（粗匹配 + 精匹配）
    result_pose = combined_transform;
    return true;
}

bool ICPLocalizer::performICPForCandidateThreadSafe(const M4F& initial_guess,
                                                    const CloudType::Ptr& rough_inp,
                                                    const CloudType::Ptr& rough_tgt,
                                                    const CloudType::Ptr& refine_inp,
                                                    const CloudType::Ptr& refine_tgt,
                                                    M4F& result_pose,
                                                    double& rough_score,
                                                    double& refine_score) {
    // 检查点云是否为空
    if (refine_tgt->size() == 0 || rough_tgt->size() == 0) {
        return false;
    }

    // 🔥 为每个线程创建独立的 ICP 对象（线程安全）
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> rough_icp;
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> refine_icp;

    // 配置 ICP 参数（与成员变量相同）
    rough_icp.setCorrespondenceRandomness(m_config.correspondence_randomness);
    rough_icp.setMaximumOptimizerIterations(m_config.maximum_optimizer_iterations);
    refine_icp.setCorrespondenceRandomness(m_config.correspondence_randomness);
    refine_icp.setMaximumOptimizerIterations(m_config.maximum_optimizer_iterations);

    // 设置对应距离
    rough_icp.setMaxCorrespondenceDistance(std::max(5.0f, static_cast<float>(m_config.rough_scan_resolution * 25.0)));
    refine_icp.setMaxCorrespondenceDistance(1.5f);

    // 设置收敛阈值
    rough_icp.setTransformationEpsilon(1e-8);
    rough_icp.setEuclideanFitnessEpsilon(1e-6);
    refine_icp.setTransformationEpsilon(1e-9);
    refine_icp.setEuclideanFitnessEpsilon(1e-7);

    CloudType::Ptr aligned_cloud(new CloudType);

    // 粗匹配
    rough_icp.setMaximumIterations(m_config.rough_max_iteration);
    rough_icp.setInputSource(rough_inp);
    rough_icp.setInputTarget(rough_tgt);
    rough_icp.align(*aligned_cloud, initial_guess);

    if (!rough_icp.hasConverged()) {
        return false;
    }

    rough_score = rough_icp.getFitnessScore();

    // 检查粗匹配评分
    const float candidate_rough_thresh = m_config.rough_score_thresh;
    if (rough_score > candidate_rough_thresh) {
        return false;
    }

    M4F rough_result = rough_icp.getFinalTransformation();

    // 精匹配
    refine_icp.setMaximumIterations(m_config.refine_max_iteration);

    // 将精匹配输入点云变换到粗匹配坐标系
    CloudType::Ptr refine_inp_transformed(new CloudType);
    pcl::transformPointCloud(*refine_inp, *refine_inp_transformed, rough_result);

    refine_icp.setInputSource(refine_inp_transformed);
    refine_icp.setInputTarget(refine_tgt);
    CloudType::Ptr aligned_refine_cloud(new CloudType);
    M4F identity_guess = M4F::Identity();
    refine_icp.align(*aligned_refine_cloud, identity_guess);

    if (!refine_icp.hasConverged()) {
        return false;
    }

    refine_score = refine_icp.getFitnessScore();

    // 检查精匹配评分
    const float candidate_refine_thresh = m_config.refine_score_thresh;
    if (refine_score > candidate_refine_thresh) {
        return false;
    }

    // 计算最终变换矩阵：粗匹配变换 * 精匹配变换
    M4F final_refine_transform = refine_icp.getFinalTransformation();
    result_pose = final_refine_transform * rough_result;

    return true;
}

void ICPLocalizer::setInput(const CloudType::Ptr &cloud)
{
    // 🔥 改进：添加点云预处理（去噪）
    CloudType::Ptr preprocessed_cloud(new CloudType);

    // 1. 统计离群点移除（去除噪声点）
    if (cloud->size() > 50) {  // 只有点云足够大时才去噪
        pcl::StatisticalOutlierRemoval<PointType> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(20);  // 每个点考虑20个邻居
        sor.setStddevMulThresh(2.0);  // 标准差倍数阈值
        sor.filter(*preprocessed_cloud);

        if (preprocessed_cloud->size() < cloud->size() * 0.5) {
            // 如果去噪后点云减少太多，说明参数可能不合适，使用原始点云
            PCL_WARN_STREAM("去噪后点云减少过多 (" << preprocessed_cloud->size()
                           << " < " << cloud->size() * 0.5 << ")，使用原始点云" << std::endl);
            pcl::copyPointCloud(*cloud, *preprocessed_cloud);
        }
    } else {
        pcl::copyPointCloud(*cloud, *preprocessed_cloud);
    }

    // 2. 体素下采样（精匹配）
    if (m_config.refine_scan_resolution > 0)
    {
        m_voxel_filter.setLeafSize(m_config.refine_scan_resolution, m_config.refine_scan_resolution, m_config.refine_scan_resolution);
        m_voxel_filter.setInputCloud(preprocessed_cloud);
        m_voxel_filter.filter(*m_refine_inp);
    }
    else
    {
        pcl::copyPointCloud(*preprocessed_cloud, *m_refine_inp);
    }

    // 3. 体素下采样（粗匹配）
    if (m_config.rough_scan_resolution > 0)
    {
        m_voxel_filter.setLeafSize(m_config.rough_scan_resolution, m_config.rough_scan_resolution, m_config.rough_scan_resolution);
        m_voxel_filter.setInputCloud(preprocessed_cloud);
        m_voxel_filter.filter(*m_rough_inp);
    }
    else
    {
        pcl::copyPointCloud(*preprocessed_cloud, *m_rough_inp);
    }
}

// 修改align函数中的日志代码
bool ICPLocalizer::align(M4F &guess, bool force_sc)
{
    // 策略：总是尝试TEASER++（如果启用），作为对先验的验证
    // 如果TEASER++与先验差异大或没有先验，优先使用TEASER++结果

    bool has_valid_prior = !guess.isApprox(M4F::Identity(), 0.01f);

    // 如果强制 SC，则忽略先验
    if (force_sc) {
        has_valid_prior = false;
        PCL_INFO_STREAM("🌍 强制触发全局重定位 (Force SC)" << std::endl);
    }

    M4F original_guess = guess;

    // 辅助lambda：从矩阵提取坐标和偏航角
    auto getPoseStr = [](const M4F& pose) -> std::string {
        Eigen::Vector3f t = pose.block<3,1>(0,3);
        Eigen::Vector3f euler = pose.block<3,3>(0,0).eulerAngles(0, 1, 2); // Roll, Pitch, Yaw
        char buf[100];
        sprintf(buf, "(x=%.2f, y=%.2f, z=%.2f, yaw=%.1f°)", t.x(), t.y(), t.z(), euler[2]*180.0/M_PI);
        return std::string(buf);
    };

    PCL_INFO_STREAM("🏁 初始猜测: " << getPoseStr(guess) << std::endl);

    if (has_valid_prior) {
        Eigen::Vector3f prior_t = guess.block<3,1>(0,3);
        float prior_translation = prior_t.norm();
        PCL_INFO_STREAM("📍 检测到先验位姿（平移=" << prior_translation << "m）" << std::endl);
    } else {
        PCL_INFO_STREAM("ℹ️  无先验位姿" << std::endl);
    }

    // 🔥 关键修改：总是尝试TEASER++作为验证

    // 0. 优先尝试 Scan Context 进行全局定位 (如果启用 且 强制SC模式)
    if (m_config.use_scan_context && force_sc) {
        // 🌍 全局重定位模式：强制使用 Scan Context
        PCL_INFO_STREAM("🔍 全局重定位：启动 Scan Context 全局定位（测试所有候选模式）..." << std::endl);
        std::vector<M4F> sc_candidates;

        // 🔥 统一测试所有候选（432个）
        if (getInitialPosesFromScanContext(m_rough_inp, sc_candidates)) {
            PCL_INFO_STREAM("✅ Scan Context 找到 " << sc_candidates.size() << " 个候选位姿" << std::endl);
            PCL_INFO_STREAM("🔬 开始对所有候选执行GICP匹配（共 " << sc_candidates.size() << " 个）..." << std::endl);

            // 🔥 改进：保存所有候选的ICP结果用于分析
            struct CandidateResult {
                size_t index = 0;
                bool icp_success = false;
                double rough_score = std::numeric_limits<double>::infinity();
                double refine_score = std::numeric_limits<double>::infinity();
                M4F initial_pose = M4F::Identity();
                M4F result_pose = M4F::Identity();
                double position_change = -1.0;
                long duration_ms = 0;
            };
            std::vector<CandidateResult> all_results;

            // 🔥 并行处理：对每个候选分别进行 ICP，选择最佳结果
            double best_fitness = std::numeric_limits<double>::max();
            M4F best_pose = M4F::Identity();
            M4F best_initial_pose = M4F::Identity();
            bool found_valid_candidate = false;
            std::atomic<bool> early_exit{false};

            // 初始化结果向量
            all_results.resize(sc_candidates.size());
            for (size_t i = 0; i < sc_candidates.size(); ++i) {
                all_results[i].index = i;
                all_results[i].initial_pose = sc_candidates[i];
            }

            // 🔥 使用 OpenMP 并行处理候选点
            auto overall_start_time = std::chrono::high_resolution_clock::now();
            PCL_INFO_STREAM("🚀 开始并行处理 " << sc_candidates.size() << " 个候选点（使用 " << omp_get_max_threads() << " 个线程）..." << std::endl);
            PCL_INFO_STREAM("⚙️  提前退出配置: " << (m_config.sc_enable_early_exit ? "启用" : "禁用") << std::endl);

            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t i = 0; i < sc_candidates.size(); ++i) {
                // 跳过已找到优秀候选的情况（提前退出）
                if (early_exit.load(std::memory_order_acquire) &&
                    m_config.sc_enable_early_exit) {
                    continue;
                }

                CandidateResult& result = all_results[i];

                M4F candidate_result;
                double candidate_rough_score, candidate_refine_score;

                auto start_time = std::chrono::high_resolution_clock::now();
                // 🔥 使用线程安全的 ICP 函数
                bool icp_success = performICPForCandidateThreadSafe(
                    sc_candidates[i],
                    m_rough_inp, m_rough_tgt,
                    m_refine_inp, m_refine_tgt,
                    candidate_result,
                    candidate_rough_score,
                    candidate_refine_score
                );
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                result.duration_ms = duration.count();

                result.icp_success = icp_success;
                result.rough_score = candidate_rough_score;
                result.refine_score = candidate_refine_score;
                result.result_pose = candidate_result;

                if (icp_success) {
                    Eigen::Vector3f init_pos = sc_candidates[i].block<3,1>(0,3);
                    Eigen::Vector3f result_pos = candidate_result.block<3,1>(0,3);
                    result.position_change = (result_pos - init_pos).norm();

                    // 🔥 如果粗匹配和精匹配都 < 0.1，且启用了提前返回，标记为优秀候选并提前退出
                    if (m_config.sc_enable_early_exit && candidate_rough_score < 0.1 && candidate_refine_score < 0.1) {
                        #pragma omp critical
                        {
                            if (!early_exit.load(std::memory_order_relaxed) ||
                                candidate_refine_score < best_fitness) {
                                best_fitness = candidate_refine_score;
                                best_pose = candidate_result;
                                best_initial_pose = sc_candidates[i];
                                found_valid_candidate = true;
                                early_exit.store(true, std::memory_order_release);
                                PCL_INFO_STREAM("  🎯 [线程 " << omp_get_thread_num() << "] 候选 " << (i+1)
                                               << " 评分优秀（粗匹配=" << candidate_rough_score
                                               << ", 精匹配=" << candidate_refine_score << "），直接选用！" << std::endl
                                               << "     ⚙️  提前退出已启用，将跳过剩余候选点" << std::endl);
                            }
                        }
                        continue;  // 提前退出，跳过后续处理
                    }

                    // 更新最佳候选（无论是否启用提前退出，都要更新最佳候选）
                    #pragma omp critical
                    {
                        if (candidate_refine_score < best_fitness) {
                            best_fitness = candidate_refine_score;
                            best_pose = candidate_result;
                            best_initial_pose = sc_candidates[i];
                            found_valid_candidate = true;
                            if (candidate_rough_score < 0.1 && candidate_refine_score < 0.1) {
                                PCL_INFO_STREAM("  ⭐ [线程 " << omp_get_thread_num() << "] 候选 " << (i+1)
                                               << " 评分优秀（粗匹配=" << candidate_rough_score
                                               << ", 精匹配=" << candidate_refine_score << "），成为当前最佳" << std::endl
                                               << "     ⚙️  提前退出已禁用，继续处理其他候选" << std::endl);
                            } else {
                                PCL_INFO_STREAM("  🏆 [线程 " << omp_get_thread_num() << "] 候选 " << (i+1)
                                               << " 成为当前最佳（评分: " << best_fitness << "）" << std::endl);
                            }
                        }
                    }
                } else {
                    result.position_change = -1.0; // 标记失败
                }
            }

            auto overall_end_time = std::chrono::high_resolution_clock::now();
            auto overall_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time);
            PCL_INFO_STREAM("✅ 并行处理完成，总耗时: " << overall_duration.count() << " ms" << std::endl);

            // 打印所有成功候选的详细信息
            for (size_t i = 0; i < all_results.size(); ++i) {
                const CandidateResult& result = all_results[i];
                if (result.icp_success) {
                    Eigen::Vector3f init_pos = sc_candidates[i].block<3,1>(0,3);
                    Eigen::Matrix3f init_rot = sc_candidates[i].block<3,3>(0,0);
                    float init_yaw = std::atan2(init_rot(1,0), init_rot(0,0)) * 180.0 / M_PI;
                    Eigen::Vector3f result_pos = result.result_pose.block<3,1>(0,3);
                    Eigen::Matrix3f result_rot = result.result_pose.block<3,3>(0,0);
                    float result_yaw = std::atan2(result_rot(1,0), result_rot(0,0)) * 180.0 / M_PI;

                    PCL_INFO_STREAM("  ✅ 候选 " << (i+1) << " ICP成功" << std::endl
                                   << "    初始坐标: [" << init_pos.x() << ", " << init_pos.y() << ", " << init_pos.z() << "] m, yaw: " << init_yaw << " deg" << std::endl
                                   << "    结果坐标: [" << result_pos.x() << ", " << result_pos.y() << ", " << result_pos.z() << "] m, yaw: " << result_yaw << " deg" << std::endl
                                   << "    位置变化: " << result.position_change << " m" << std::endl
                                   << "    粗匹配评分: " << std::fixed << std::setprecision(4) << result.rough_score << std::endl
                                   << "    精匹配评分: " << std::fixed << std::setprecision(4) << result.refine_score << std::endl
                                   << "    耗时: " << result.duration_ms << " ms" << std::endl);
                }
            }

            if (early_exit.load(std::memory_order_acquire)) {
                PCL_INFO_STREAM("\n⚡ 提前退出：找到评分优秀的候选点（粗匹配和精匹配均 < 0.1）" << std::endl);
            }

            // 🔥 输出所有候选的汇总结果
            PCL_INFO_STREAM("\n" << std::string(60, '=') << std::endl);
            PCL_INFO_STREAM("📊 所有候选GICP匹配结果汇总：" << std::endl);
            PCL_INFO_STREAM(std::string(60, '=') << std::endl);

            // 按ICP评分排序（成功的在前）
            std::sort(all_results.begin(), all_results.end(),
                     [](const CandidateResult& a, const CandidateResult& b) {
                         if (a.icp_success != b.icp_success) {
                             return a.icp_success > b.icp_success; // 成功的在前
                         }
                         return a.refine_score < b.refine_score; // 精匹配评分低的在前
                     });

            int success_count = 0;
            long total_time_ms = 0;
            for (const auto& res : all_results) {
                total_time_ms += res.duration_ms;
                if (res.icp_success) {
                    success_count++;
                    Eigen::Vector3f result_pos = res.result_pose.block<3,1>(0,3);
                    PCL_INFO_STREAM("✅ 候选 #" << (res.index+1)
                                   << " | 粗匹配评分: " << std::fixed << std::setprecision(4) << res.rough_score
                                   << " | 精匹配评分: " << std::setprecision(4) << res.refine_score
                                   << " | 位置: [" << std::setprecision(2) << result_pos.x() << ", "
                                   << result_pos.y() << ", " << result_pos.z() << "] m"
                                   << " | 位置变化: " << res.position_change << " m" << std::endl);
                }
            }

            PCL_INFO_STREAM("\n📈 统计信息：" << std::endl);
            PCL_INFO_STREAM("  - 总候选数: " << all_results.size() << std::endl);
            PCL_INFO_STREAM("  - ICP成功: " << success_count << " (" << std::fixed << std::setprecision(1)
                           << (100.0 * success_count / all_results.size()) << "%)" << std::endl);
            PCL_INFO_STREAM("  - ICP失败: " << (all_results.size() - success_count) << std::endl);
            PCL_INFO_STREAM("  - 总耗时: " << total_time_ms << " ms (" << (total_time_ms / 1000.0) << " s)" << std::endl);
            PCL_INFO_STREAM("  - 平均耗时: " << (total_time_ms / all_results.size()) << " ms/候选" << std::endl);

            if (success_count > 0) {
                // 找出最佳的几个结果
                PCL_INFO_STREAM("\n🏆 最佳候选（按ICP评分排序，前10个）：" << std::endl);
                int top_n = std::min(10, success_count);
                for (int i = 0; i < top_n; ++i) {
                    const auto& res = all_results[i];
                    Eigen::Vector3f result_pos = res.result_pose.block<3,1>(0,3);
                    Eigen::Matrix3f result_rot = res.result_pose.block<3,3>(0,0);
                    float result_yaw = std::atan2(result_rot(1,0), result_rot(0,0)) * 180.0 / M_PI;
                    PCL_INFO_STREAM("  [" << (i+1) << "] 候选 #" << (res.index+1)
                                   << " | 粗匹配评分: " << std::fixed << std::setprecision(4) << res.rough_score
                                   << " | 精匹配评分: " << std::setprecision(4) << res.refine_score
                                   << " | 位置: [" << std::setprecision(2) << result_pos.x() << ", "
                                   << result_pos.y() << ", " << result_pos.z() << "] m"
                                   << " | 航向: " << std::setprecision(1) << result_yaw << "°" << std::endl);
                }
            }
            PCL_INFO_STREAM(std::string(60, '=') << std::endl);

            if (found_valid_candidate) {
                PCL_INFO_STREAM("✅ Scan Context 多候选定位成功！最佳候选评分: " << best_fitness << std::endl);
                PCL_INFO_STREAM("📍 SC最佳结果: " << getPoseStr(best_pose) << std::endl);

                // 🔥 关键修复：performICPForCandidate 已经在候选验证阶段执行过，结果已经保存在成员变量中
                // 不需要重新执行，直接使用候选验证阶段的结果
                guess = best_pose;
                m_fitness_score = best_fitness;
                PCL_INFO_STREAM("💾 使用候选验证阶段的 ICP 结果（已保存在成员变量中）" << std::endl);

                // SC成功后直接返回，跳过后续的粗匹配和精匹配，避免重复匹配
                PCL_INFO_STREAM("🚀 SC定位成功，直接返回结果，跳过后续ICP匹配" << std::endl);
                return true;
            } else {
                PCL_WARN_STREAM("❌ 所有 Scan Context 候选的 ICP 都失败" << std::endl);
                // SC失败，继续尝试TEASER++
            }
        } else {
            PCL_WARN_STREAM("❌ Scan Context 定位失败 (无匹配)" << std::endl);
            // SC失败，继续尝试TEASER++
        }
    } else if (has_valid_prior && !force_sc) {
        // 🎯 手动位姿模式：有先验且非强制SC，直接跳过SC
        PCL_INFO_STREAM("🎯 手动位姿模式：已有先验 (" << getPoseStr(guess) << ")，跳过 Scan Context" << std::endl);
    }

    if (m_config.use_global_registration) {
        PCL_INFO_STREAM("🎯 启动TEASER++全局配准（验证/补充先验）..." << std::endl);
        M4F teaser_guess = M4F::Identity();

        if (globalRegistration(teaser_guess)) {
            // TEASER++成功
            PCL_INFO_STREAM("📍 TEASER++结果: " << getPoseStr(teaser_guess) << std::endl);

            if (has_valid_prior) {
                // 对比TEASER++与先验的差异
                Eigen::Vector3f prior_t = original_guess.block<3,1>(0,3);
                Eigen::Vector3f teaser_t = teaser_guess.block<3,1>(0,3);
                float translation_diff = (teaser_t - prior_t).norm();

                Eigen::Matrix3f prior_R = original_guess.block<3,3>(0,0);
                Eigen::Matrix3f teaser_R = teaser_guess.block<3,3>(0,0);
                Eigen::Matrix3f diff_R = teaser_R * prior_R.transpose();
                Eigen::AngleAxisf diff_aa(diff_R);
                float rotation_diff_deg = std::abs(diff_aa.angle()) * 180.0f / M_PI;

                PCL_INFO_STREAM("🔍 TEASER++与先验对比：" << std::endl
                               << "  - 先验: " << getPoseStr(original_guess) << std::endl
                               << "  - TEASER: " << getPoseStr(teaser_guess) << std::endl
                               << "  - 平移差异: " << translation_diff << " m" << std::endl
                               << "  - 旋转差异: " << rotation_diff_deg << "°" << std::endl);

                // 如果差异显著，说明先验不可靠，优先使用TEASER++
                const float significant_translation_diff = 2.0f;  // 2米
                const float significant_rotation_diff = 15.0f;    // 15度

                if (translation_diff > significant_translation_diff ||
                    rotation_diff_deg > significant_rotation_diff) {
                    PCL_WARN_STREAM("⚠️  先验与TEASER++差异显著，使用TEASER++结果" << std::endl);
                    guess = teaser_guess;
                } else {
                    PCL_INFO_STREAM("✅ 先验与TEASER++一致，保持先验（更精确）" << std::endl);
                    // guess保持原先验不变
                }
            } else {
                // 无先验，直接使用TEASER++
                PCL_INFO_STREAM("✅ TEASER++成功，使用TEASER++结果" << std::endl);
                guess = teaser_guess;
            }
        } else {
            // TEASER++失败
            if (has_valid_prior) {
                PCL_WARN_STREAM("⚠️  TEASER++失败，使用原始先验" << std::endl);
                // guess保持不变
            } else {
                PCL_ERROR_STREAM("❌ 无先验且TEASER++失败，无法初始化" << std::endl);
                return false;
            }
        }
    } else {
        if (!has_valid_prior) {
            PCL_ERROR_STREAM("❌ 未启用全局配准且无先验，无法初始化" << std::endl);
            return false;
        }
        PCL_INFO_STREAM("ℹ️  全局配准未启用，使用先验位姿" << std::endl);
    }

    CloudType::Ptr aligned_cloud(new CloudType);
    // 在icp_localizer.cpp中的align函数添加
    if (m_refine_tgt->size() == 0 || m_rough_tgt->size() == 0)
    {
        std::cerr << "ICP失败：目标点云为空！" << std::endl;
        return false;
    }

    // 粗匹配前 - 显示初始猜测
    Eigen::Vector3f init_trans = guess.block<3,1>(0,3);
    Eigen::Matrix3f init_rot = guess.block<3,3>(0,0);
    Eigen::AngleAxisf init_angle(init_rot);
    float init_yaw_deg = std::atan2(init_rot(1,0), init_rot(0,0)) * 180.0f / M_PI;

    PCL_INFO_STREAM("🔵 开始粗匹配（GICP）" << std::endl
                   << "  - 输入点云大小: " << m_rough_inp->size() << std::endl
                   << "  - 目标点云大小: " << m_rough_tgt->size() << std::endl
                   << "  - 最大迭代次数: " << m_config.rough_max_iteration << std::endl
                   << "  - 评分阈值: " << m_config.rough_score_thresh << std::endl
                   << "  - 初始位置: [" << init_trans(0) << ", " << init_trans(1)
                   << ", " << init_trans(2) << "] m" << std::endl
                   << "  - 初始航向: " << init_yaw_deg << "°" << std::endl);

    // 设置粗匹配参数并执行匹配
    m_rough_icp.setMaximumIterations(m_config.rough_max_iteration);
    m_rough_icp.setInputSource(m_rough_inp);
    m_rough_icp.setInputTarget(m_rough_tgt);
    m_rough_icp.align(*aligned_cloud, guess);

    // 粗匹配后
    if (!m_rough_icp.hasConverged()) {
        PCL_WARN_STREAM("❌ 粗匹配未收敛！" << std::endl
                       << "  - 迭代次数: " << m_config.rough_max_iteration << std::endl
                       << "  - 最终变换矩阵：\n" << m_rough_icp.getFinalTransformation() << std::endl);
        return false;
    }

    float rough_score = m_rough_icp.getFitnessScore();
    M4F rough_transform = m_rough_icp.getFinalTransformation();
    Eigen::Vector3f rough_trans = rough_transform.block<3,1>(0,3);
    Eigen::Matrix3f rough_rot = rough_transform.block<3,3>(0,0);
    float rough_yaw_deg = std::atan2(rough_rot(1,0), rough_rot(0,0)) * 180.0f / M_PI;

    // 🔍 计算变换后当前帧的几何中心（用于验证）
    CloudType::Ptr transformed_rough_cloud(new CloudType);
    pcl::transformPointCloud(*m_rough_inp, *transformed_rough_cloud, rough_transform);
    Eigen::Vector4f rough_cloud_centroid;
    pcl::compute3DCentroid(*transformed_rough_cloud, rough_cloud_centroid);
    Eigen::Vector3f rough_cloud_center = rough_cloud_centroid.head<3>();

    PCL_INFO_STREAM("✅ 粗匹配完成" << std::endl
                   << "  - 评分: " << rough_score
                   << " (阈值: " << m_config.rough_score_thresh << ")" << std::endl
                   << "  📍 粗匹配后位姿:" << std::endl
                   << "     • 位置 (x, y, z): [" << std::fixed << std::setprecision(3)
                   << rough_trans(0) << ", " << rough_trans(1) << ", " << rough_trans(2) << "] m" << std::endl
                   << "     • 航向 (yaw): " << std::setprecision(2) << rough_yaw_deg << "°" << std::endl
                   << "     • 位置修正: " << std::setprecision(3) << (rough_trans - init_trans).norm() << " m" << std::endl
                   << "  📊 变换后当前帧几何中心:" << std::endl
                   << "     • 几何中心 (x, y, z): [" << std::setprecision(3)
                   << rough_cloud_center(0) << ", " << rough_cloud_center(1) << ", " << rough_cloud_center(2) << "] m" << std::endl
                   << "     • 与位姿位置差异: " << std::setprecision(3)
                   << (rough_cloud_center - rough_trans).norm() << " m" << std::endl
                   << "  📋 粗匹配变换矩阵:" << std::endl
                   << rough_transform << std::endl);

    // 保存GICP粗匹配变换矩阵用于可视化
    m_rough_transform = rough_transform;
    m_has_rough_result = true;
    PCL_INFO_STREAM("💾 已保存GICP粗匹配结果用于可视化（source: "
                   << m_rough_inp->size() << " 点, target: " << m_rough_tgt->size() << " 点）" << std::endl);

    if (rough_score > m_config.rough_score_thresh)
    {
        PCL_WARN_STREAM("❌ ICP粗匹配失败：评分过高" << std::endl
                       << "  - 实际评分: " << rough_score << std::endl
                       << "  - 评分阈值: " << m_config.rough_score_thresh << std::endl);
        return false;
    }

    PCL_INFO_STREAM("📋 粗匹配变换矩阵：\n" << rough_transform << std::endl);

    // 精匹配设置
    float position_correction = (rough_trans - init_trans).norm();

    // 🔍 记录精匹配前的初始状态
    M4F initial_refine_transform = m_rough_icp.getFinalTransformation();

    // 🔍 计算精匹配输入点云的几何中心（变换前）
    Eigen::Vector4f refine_inp_centroid_before;
    pcl::compute3DCentroid(*m_refine_inp, refine_inp_centroid_before);
    Eigen::Vector3f refine_inp_center_before = refine_inp_centroid_before.head<3>();

    // 🔍 计算使用粗匹配变换后的精匹配输入点云几何中心
    M4F rough_result = m_rough_icp.getFinalTransformation();
    CloudType::Ptr refine_inp_transformed_by_rough(new CloudType);
    pcl::transformPointCloud(*m_refine_inp, *refine_inp_transformed_by_rough, rough_result);
    Eigen::Vector4f refine_inp_centroid_after_rough;
    pcl::compute3DCentroid(*refine_inp_transformed_by_rough, refine_inp_centroid_after_rough);
    Eigen::Vector3f refine_inp_center_after_rough = refine_inp_centroid_after_rough.head<3>();

    PCL_INFO_STREAM("🟢 开始精匹配（GICP）" << std::endl
                   << "  - 输入点云大小: " << m_refine_inp->size() << std::endl
                   << "  - 目标点云大小: " << m_refine_tgt->size() << std::endl
                   << "  - 最大迭代次数: " << m_config.refine_max_iteration << std::endl
                   << "  - 评分阈值: " << m_config.refine_score_thresh << std::endl
                   << "  - 粗匹配位置修正: " << position_correction << " m" << std::endl
                   << "  📋 GICP收敛规则:" << std::endl
                   << "     • TransformationEpsilon: 1e-6 (变换矩阵变化量阈值，放宽以避免过早停止)" << std::endl
                   << "     • EuclideanFitnessEpsilon: 1e-4 (适应度评分变化量阈值，放宽以避免过早停止)" << std::endl
                   << "     • MaximumOptimizerIterations: " << m_config.maximum_optimizer_iterations
                   << " (优化器迭代次数，让优化器在平坦区域有更多步数)" << std::endl
                   << "     • 当连续两次迭代的变换变化 < 1e-6 且适应度变化 < 1e-4 时，认为收敛" << std::endl
                   << "  📊 子图信息:" << std::endl
                   << "     • 是否使用子图: 否（使用全局地图）" << std::endl
                   << "     • 地图总点数: " << m_refine_tgt->size() << std::endl
                   << "     • 子图提取中心: 无（使用全局地图）" << std::endl
                   << "  🔍 位姿验证（关键诊断）:" << std::endl
                   << "     • 粗匹配后位姿位置: [" << std::fixed << std::setprecision(3)
                   << rough_trans(0) << ", " << rough_trans(1) << ", " << rough_trans(2) << "] m" << std::endl
                   << "     • 粗匹配后位姿航向: " << std::setprecision(2) << rough_yaw_deg << "°" << std::endl
                   << "     • 精匹配输入点云原始几何中心: [" << std::setprecision(3)
                   << refine_inp_center_before(0) << ", " << refine_inp_center_before(1) << ", " << refine_inp_center_before(2) << "] m" << std::endl
                   << "     • 精匹配输入点云经粗匹配变换后几何中心: [" << std::setprecision(3)
                   << refine_inp_center_after_rough(0) << ", " << refine_inp_center_after_rough(1) << ", " << refine_inp_center_after_rough(2) << "] m" << std::endl
                   << "     • 变换后几何中心与粗匹配位姿位置差异: " << std::setprecision(3)
                   << (refine_inp_center_after_rough - rough_trans).norm() << " m" << std::endl
                   << "     • 粗匹配变换矩阵（作为精匹配初始猜测）:" << std::endl
                   << rough_result << std::endl);

    // 🔥 动态调整精匹配对应距离：如果粗匹配修正较大，使用更大的对应距离
    float original_refine_corr_dist = m_refine_icp.getMaxCorrespondenceDistance();
    float adaptive_refine_corr_dist = original_refine_corr_dist;

    // 🔥 L型走廊场景调参：如果位置修正 > 0.1m，适当增加对应距离，但不超过1.5m（推荐范围上限）
    if (position_correction > 0.1f) {
        // 对应距离至少为位置修正的2倍，但不超过1.5m（避免跑飞）
        adaptive_refine_corr_dist = std::min(1.5f, std::max(original_refine_corr_dist, position_correction * 2.0f));
        m_refine_icp.setMaxCorrespondenceDistance(adaptive_refine_corr_dist);
        PCL_INFO_STREAM("  - 调整精匹配对应距离: " << original_refine_corr_dist << " m -> "
                       << adaptive_refine_corr_dist << " m (位置修正: " << position_correction << " m)" << std::endl);
    } else {
        PCL_INFO_STREAM("  - 精匹配对应距离: " << original_refine_corr_dist << " m" << std::endl);
    }

    m_refine_icp.setMaximumIterations(m_config.refine_max_iteration);

    // 使用粗匹配结果作为精匹配的初始位姿（rough_result已在上面声明）

    // 精配准
    m_refine_icp.setInputSource(m_refine_inp);
    m_refine_icp.setInputTarget(m_refine_tgt);
    m_refine_icp.align(*m_refine_inp, rough_result);

    // 存储适应度评分
    m_fitness_score = m_refine_icp.getFitnessScore();

    // 🔍 计算对应点统计信息（用于诊断）
    M4F final_refine_transform = m_refine_icp.getFinalTransformation();
    CloudType::Ptr transformed_cloud(new CloudType);
    pcl::transformPointCloud(*m_refine_inp, *transformed_cloud, final_refine_transform);

    // 使用KD-tree查找最近邻，计算对应点统计
    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(m_refine_tgt);

    size_t num_correspondences = 0;
    size_t num_inliers = 0;
    double sum_squared_dist = 0.0;
    double sum_dist = 0.0;
    double max_dist = 0.0;
    float corr_dist_threshold = adaptive_refine_corr_dist;

    std::vector<int> indices(1);
    std::vector<float> sqr_dists(1);

    for (size_t i = 0; i < transformed_cloud->size(); ++i) {
        if (kdtree.nearestKSearch(transformed_cloud->points[i], 1, indices, sqr_dists) > 0) {
            double dist = std::sqrt(sqr_dists[0]);
            num_correspondences++;
            sum_squared_dist += sqr_dists[0];
            sum_dist += dist;
            max_dist = std::max(max_dist, dist);
            if (dist < corr_dist_threshold) {
                num_inliers++;
            }
        }
    }

    double avg_dist = num_correspondences > 0 ? sum_dist / num_correspondences : 0.0;
    double rmse = num_correspondences > 0 ? std::sqrt(sum_squared_dist / num_correspondences) : 0.0;
    double inlier_ratio = num_correspondences > 0 ? (100.0 * num_inliers / num_correspondences) : 0.0;

    // 🔍 计算变换矩阵的变化量（用于分析收敛情况）
    M4F transform_diff = final_refine_transform.inverse() * initial_refine_transform;
    Eigen::Vector3f transform_trans_diff = transform_diff.block<3,1>(0,3);
    float transform_translation_change = transform_trans_diff.norm();

    // 计算旋转变化（使用旋转矩阵的Frobenius范数）
    Eigen::Matrix3f transform_rot_diff = transform_diff.block<3,3>(0,0);
    float transform_rotation_change = (transform_rot_diff - Eigen::Matrix3f::Identity()).norm();

    if (!m_refine_icp.hasConverged())
    {
        // 🔍 详细记录未收敛的原因
        PCL_WARN_STREAM("❌ 精匹配未收敛！详细分析:" << std::endl
                       << "  📊 收敛状态检查:" << std::endl
                       << "     • hasConverged(): false" << std::endl
                       << "     • 实际迭代次数: " << m_config.refine_max_iteration << " (可能达到最大迭代次数)" << std::endl
                       << "  📈 最终状态:" << std::endl
                       << "     • 最终适应度评分: " << std::fixed << std::setprecision(6) << m_fitness_score << std::endl
                       << "     • 变换矩阵平移变化: " << std::setprecision(6) << transform_translation_change
                       << " m (阈值: 1e-6)" << std::endl
                       << "     • 变换矩阵旋转变化: " << std::setprecision(6) << transform_rotation_change
                       << " (阈值: 1e-6)" << std::endl
                       << "  ⚙️  参数设置:" << std::endl
                       << "     • 对应距离: " << std::setprecision(2) << adaptive_refine_corr_dist << " m" << std::endl
                       << "     • 粗匹配位置修正: " << position_correction << " m" << std::endl
                       << "  📊 对应点统计（诊断信息）:" << std::endl
                       << "     • 输入点数: " << m_refine_inp->size() << std::endl
                       << "     • 目标点数（地图）: " << m_refine_tgt->size() << std::endl
                       << "     • 找到对应点数: " << num_correspondences << " / " << m_refine_inp->size()
                       << " (" << std::fixed << std::setprecision(1) << (100.0 * num_correspondences / m_refine_inp->size()) << "%)" << std::endl
                       << "     • Inliers（距离 < " << std::setprecision(2) << corr_dist_threshold << "m）: "
                       << num_inliers << " / " << num_correspondences
                       << " (" << std::setprecision(1) << inlier_ratio << "%)" << std::endl
                       << "     • 平均对应距离: " << std::setprecision(3) << avg_dist << " m" << std::endl
                       << "     • 最大对应距离: " << std::setprecision(3) << max_dist << " m" << std::endl
                       << "     • RMSE: " << std::setprecision(3) << rmse << " m" << std::endl
                       << "     • Fitness Score (平方误差和): " << std::setprecision(2) << m_fitness_score << std::endl);

        // 🔥 L型走廊场景调参：如果第一次精匹配失败且对应距离较小，尝试使用更大的对应距离重试
        // 但不超过1.5m（推荐范围上限），避免跑飞
        if (adaptive_refine_corr_dist < 0.5f && position_correction > 0.1f) {
            float retry_corr_dist = std::min(1.5f, position_correction * 3.0f);
            m_refine_icp.setMaxCorrespondenceDistance(retry_corr_dist);
            PCL_WARN_STREAM("  🔄 重试策略: 使用更大对应距离重试" << std::endl
                           << "     • 新对应距离: " << retry_corr_dist << " m (原: " << adaptive_refine_corr_dist << " m)" << std::endl);
            m_refine_icp.align(*m_refine_inp, rough_result);

            // 重新计算变换变化
            M4F retry_final_transform = m_refine_icp.getFinalTransformation();
            M4F retry_transform_diff = retry_final_transform.inverse() * initial_refine_transform;
            Eigen::Vector3f retry_trans_diff = retry_transform_diff.block<3,1>(0,3);
            float retry_translation_change = retry_trans_diff.norm();
            Eigen::Matrix3f retry_rot_diff = retry_transform_diff.block<3,3>(0,0);
            float retry_rotation_change = (retry_rot_diff - Eigen::Matrix3f::Identity()).norm();

            if (m_refine_icp.hasConverged()) {
                m_fitness_score = m_refine_icp.getFitnessScore();
                PCL_INFO_STREAM("✅ 精匹配重试成功！" << std::endl
                               << "  📊 收敛状态:" << std::endl
                               << "     • hasConverged(): true" << std::endl
                               << "     • 最终适应度评分: " << std::fixed << std::setprecision(6) << m_fitness_score << std::endl
                               << "     • 变换矩阵平移变化: " << std::setprecision(6) << retry_translation_change << " m" << std::endl
                               << "     • 变换矩阵旋转变化: " << std::setprecision(6) << retry_rotation_change << std::endl);
            } else {
                // 恢复原始对应距离
                m_refine_icp.setMaxCorrespondenceDistance(original_refine_corr_dist);
                PCL_WARN_STREAM("❌ 精匹配未收敛（重试后仍失败）！" << std::endl
                               << "  📊 重试后状态:" << std::endl
                               << "     • 对应距离: " << std::setprecision(2) << retry_corr_dist << " m" << std::endl
                               << "     • 最终适应度评分: " << std::fixed << std::setprecision(6) << m_refine_icp.getFitnessScore() << std::endl
                               << "     • 变换矩阵平移变化: " << std::setprecision(6) << retry_translation_change << " m" << std::endl
                               << "     • 变换矩阵旋转变化: " << std::setprecision(6) << retry_rotation_change << std::endl
                               << "     • 粗匹配位置修正: " << position_correction << " m" << std::endl
                               << "  💡 可能原因:" << std::endl
                               << "     • 点云质量不足或噪声过大" << std::endl
                               << "     • 初始位姿误差仍然过大" << std::endl
                               << "     • 需要进一步增大对应距离或迭代次数" << std::endl);
                return false;
            }
        } else {
            // 恢复原始对应距离
            if (position_correction > 0.1f) {
                m_refine_icp.setMaxCorrespondenceDistance(original_refine_corr_dist);
            }
            PCL_WARN_STREAM("  💡 未重试原因:" << std::endl
                           << "     • 对应距离已较大 (" << adaptive_refine_corr_dist << " m >= 0.5 m)" << std::endl
                           << "     • 或粗匹配位置修正较小 (" << position_correction << " m <= 0.1 m)" << std::endl);
            return false;
        }
    } else {
        // 🔍 记录收敛成功的详细信息
        PCL_INFO_STREAM("  ✅ 收敛成功！详细状态:" << std::endl
                       << "     • hasConverged(): true" << std::endl
                       << "     • 最终适应度评分: " << std::fixed << std::setprecision(6) << m_fitness_score << std::endl
                       << "     • 变换矩阵平移变化: " << std::setprecision(6) << transform_translation_change
                       << " m (满足 < 1e-6 阈值)" << std::endl
                       << "     • 变换矩阵旋转变化: " << std::setprecision(6) << transform_rotation_change
                       << " (满足 < 1e-6 阈值)" << std::endl
                       << "  📊 对应点统计（诊断信息）:" << std::endl
                       << "     • 输入点数: " << m_refine_inp->size() << std::endl
                       << "     • 目标点数（地图）: " << m_refine_tgt->size() << std::endl
                       << "     • 找到对应点数: " << num_correspondences << " / " << m_refine_inp->size()
                       << " (" << std::fixed << std::setprecision(1) << (100.0 * num_correspondences / m_refine_inp->size()) << "%)" << std::endl
                       << "     • Inliers（距离 < " << std::setprecision(2) << corr_dist_threshold << "m）: "
                       << num_inliers << " / " << num_correspondences
                       << " (" << std::setprecision(1) << inlier_ratio << "%)" << std::endl
                       << "     • 平均对应距离: " << std::setprecision(3) << avg_dist << " m" << std::endl
                       << "     • 最大对应距离: " << std::setprecision(3) << max_dist << " m" << std::endl
                       << "     • RMSE: " << std::setprecision(3) << rmse << " m" << std::endl);
    }

    // 恢复原始对应距离（如果之前调整过）
    if (position_correction > 0.1f && adaptive_refine_corr_dist != original_refine_corr_dist) {
        m_refine_icp.setMaxCorrespondenceDistance(original_refine_corr_dist);
    }

    M4F refine_transform = m_refine_icp.getFinalTransformation();
    Eigen::Vector3f refine_trans = refine_transform.block<3,1>(0,3);
    Eigen::Matrix3f refine_rot = refine_transform.block<3,3>(0,0);
    float refine_yaw_deg = std::atan2(refine_rot(1,0), refine_rot(0,0)) * 180.0f / M_PI;

    PCL_INFO_STREAM("✅ 精匹配完成" << std::endl
                   << "  - 评分: " << m_fitness_score
                   << " (阈值: " << m_config.refine_score_thresh << ")" << std::endl
                   << "  - 结果位置: [" << refine_trans(0) << ", " << refine_trans(1)
                   << ", " << refine_trans(2) << "] m" << std::endl
                   << "  - 结果航向: " << refine_yaw_deg << "°" << std::endl
                   << "  - 相对粗匹配修正: " << (refine_trans - rough_trans).norm() << " m" << std::endl);

    // 保存GICP精匹配变换矩阵用于可视化
    m_refine_transform = refine_transform;
    m_has_refine_result = true;
    PCL_INFO_STREAM("💾 已保存GICP精匹配结果用于可视化（source: "
                   << m_refine_inp->size() << " 点, target: " << m_refine_tgt->size() << " 点）" << std::endl);

    if (m_fitness_score > m_config.refine_score_thresh)
    {
        PCL_WARN_STREAM("❌ ICP精匹配失败：评分过高" << std::endl
                       << "  - 实际评分: " << m_fitness_score << std::endl
                       << "  - 评分阈值: " << m_config.refine_score_thresh << std::endl);
        return false;
    }

    guess = m_refine_icp.getFinalTransformation();

    if (m_config.enable_ground_z_correction) {
    // 地面机器人Z轴修正：仅在配置明确启用时执行。
    // 问题：之前用的是odom坐标系的点云，导致地面高度检测错误
    // 解决：先将点云转换到map坐标系，再检测地面

    // 将当前扫描点云转换到map坐标系
    CloudType::Ptr scan_in_map(new CloudType);
    pcl::transformPointCloud(*m_refine_inp, *scan_in_map, guess);

    // 从转换后的点云中检测地面
    std::vector<float> z_values;
    z_values.reserve(scan_in_map->size());
    for (const auto& pt : scan_in_map->points) {
        z_values.push_back(pt.z);
    }

    if (z_values.size() > 100) {  // 确保有足够的点
        // 排序找到最低的10%的点
        std::sort(z_values.begin(), z_values.end());
        size_t ground_sample_size = std::min(static_cast<size_t>(z_values.size() * 0.1), static_cast<size_t>(1000));

        // 计算地面点的平均高度
        float ground_height_sum = 0.0f;
        for (size_t i = 0; i < ground_sample_size; ++i) {
            ground_height_sum += z_values[i];
        }
        float detected_ground_z = ground_height_sum / ground_sample_size;

        const float map_ground_z = static_cast<float>(m_config.map_ground_z);

        // 当前GICP得到的Z轴平移
        Eigen::Vector3f final_trans = guess.block<3,1>(0,3);

        // 根据地面检测修正Z轴
        float z_correction = map_ground_z - detected_ground_z;
        float corrected_z = final_trans(2) + z_correction;

        PCL_INFO_STREAM("🌍 地面检测（map坐标系）:" << std::endl
                       << "  - 检测到的地面高度: " << detected_ground_z << " m" << std::endl
                       << "  - 地图地面高度: " << map_ground_z << " m" << std::endl
                       << "  - Z轴修正量: " << z_correction << " m" << std::endl
                       << "  - 原始Z: " << final_trans(2) << " m → 修正后Z: " << corrected_z << " m" << std::endl);

        // 应用修正（但限制修正幅度，防止地面检测错误）
        const float max_correction =
            static_cast<float>(m_config.max_ground_z_correction);
        if (std::abs(z_correction) < max_correction) {
            guess(2, 3) = corrected_z;
            PCL_INFO_STREAM("✅ 应用地面修正" << std::endl);
        } else {
            PCL_WARN_STREAM("⚠️  地面修正过大(" << z_correction << " m)，可能检测错误，保持GICP结果" << std::endl);
        }
    } else {
        PCL_WARN_STREAM("⚠️  点云太少，跳过地面检测" << std::endl);
    }
    }

    PCL_INFO_STREAM("📋 最终变换矩阵（地面修正后）：\n" << guess << std::endl);
    PCL_INFO_STREAM("🎉 配准成功！总体位置修正: " << (refine_trans - init_trans).norm() << " m" << std::endl);

    return true;
}

// 辅助函数：计算FPFH特征
void ICPLocalizer::computeFPFH(CloudType::Ptr cloud,
                               pcl::PointCloud<pcl::FPFHSignature33>::Ptr features)
{
    // 1. 估计法向量
    pcl::NormalEstimationOMP<PointType, pcl::Normal> nest;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    nest.setRadiusSearch(m_config.global_feature_radius * 0.5);
    nest.setInputCloud(cloud);
    nest.compute(*normals);

    // 2. 计算FPFH特征
    pcl::FPFHEstimationOMP<PointType, pcl::Normal, pcl::FPFHSignature33> fest;
    fest.setRadiusSearch(m_config.global_feature_radius);
    fest.setInputCloud(cloud);
    fest.setInputNormals(normals);
    fest.compute(*features);
}

bool ICPLocalizer::globalRegistration(M4F &guess)
{
    // 记录输入的初始猜测（用于后续验证TEASER++结果）
    M4F input_guess = guess;
    bool has_prior = !guess.isApprox(M4F::Identity(), 0.01f);

    Eigen::Vector3f prior_t(0, 0, 0);
    Eigen::Matrix3f prior_R = Eigen::Matrix3f::Identity();
    float prior_rotation_deg = 0.0f;

    if (has_prior) {
        prior_t = input_guess.block<3,1>(0,3);
        prior_R = input_guess.block<3,3>(0,0);
        Eigen::AngleAxisf prior_aa(prior_R);
        prior_rotation_deg = std::abs(prior_aa.angle()) * 180.0f / M_PI;
        PCL_INFO_STREAM("📍 收到先验位姿：平移=" << prior_t.norm() << "m, 旋转="
                       << prior_rotation_deg << "°" << std::endl
                       << "   将用于验证TEASER++结果的合理性" << std::endl);
    } else {
        PCL_INFO_STREAM("ℹ️  无先验信息，使用纯全局配准" << std::endl);
    }

    // 1. 极度降采样用于快速全局配准
    CloudType::Ptr src_sparse(new CloudType);
    CloudType::Ptr tgt_sparse(new CloudType);

    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(m_config.global_voxel_size, m_config.global_voxel_size, m_config.global_voxel_size);
    vg.setInputCloud(m_rough_inp);
    vg.filter(*src_sparse);
    vg.setInputCloud(m_rough_tgt);
    vg.filter(*tgt_sparse);

    // 备份原始source点云用于可视化
    CloudType::Ptr src_sparse_original(new CloudType);
    *src_sparse_original = *src_sparse;

    PCL_INFO_STREAM("🔥 TEASER++全局配准：输入点云大小=" << src_sparse->size()
                   << ", 目标点云大小=" << tgt_sparse->size() << std::endl);

    // 检查点云大小
    if (src_sparse->size() < 10 || tgt_sparse->size() < 10) {
        std::cerr << "全局配准失败：点云太小" << std::endl;
        return false;
    }

    // 2. 计算FPFH特征
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features(
        new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr tgt_features(
        new pcl::PointCloud<pcl::FPFHSignature33>);

    try {
        computeFPFH(src_sparse, src_features);
        computeFPFH(tgt_sparse, tgt_features);
        PCL_INFO_STREAM("📊 FPFH特征计算完成" << std::endl
                       << "  - 输入点云特征数: " << src_features->size() << std::endl
                       << "  - 目标点云特征数: " << tgt_features->size() << std::endl
                       << "  - 特征维度: 33" << std::endl);
    } catch (const std::exception& e) {
        std::cerr << "❌ FPFH特征计算失败: " << e.what() << std::endl;
        return false;
    }

    // 3. TEASER++全局配准
    // 3.1 转换PCL点云为TEASER++格式
    teaser::PointCloud src_teaser, tgt_teaser;

    // 转换点云
    for (size_t i = 0; i < src_sparse->size(); ++i) {
        src_teaser.push_back({src_sparse->points[i].x,
                              src_sparse->points[i].y,
                              src_sparse->points[i].z});
    }

    for (size_t i = 0; i < tgt_sparse->size(); ++i) {
        tgt_teaser.push_back({tgt_sparse->points[i].x,
                              tgt_sparse->points[i].y,
                              tgt_sparse->points[i].z});
    }

    // 3.2 使用FPFH特征进行对应关系匹配
    teaser::Matcher matcher;

    // 参数说明：
    // - use_absolute_scale: false (不使用绝对尺度)
    // - use_crosscheck: true (启用交叉检查，过滤明显不一致的对应点)
    // - use_tuple_test: false (关闭tuple test，给TEASER++更多候选点)
    // - tuple_scale: 0.9 (如果启用，使用较宽松的阈值)
    // 策略：放宽FPFH匹配，让TEASER++的RANSAC来过滤outliers
    PCL_INFO_STREAM("🔍 开始FPFH特征匹配（交叉检查+宽松阈值）..." << std::endl);
    auto correspondences = matcher.calculateCorrespondences(
        src_teaser, tgt_teaser, *src_features, *tgt_features,
        false, true, false, 0.9f);

    PCL_INFO_STREAM("✅ FPFH匹配完成，得到 " << correspondences.size() << " 个对应点对" << std::endl);

    // 统计对应点对的几何分布
    if (correspondences.size() > 0) {
        double max_dist = 0.0;
        double avg_dist = 0.0;
        for (const auto& corr : correspondences) {
            const auto& src_p = src_teaser[corr.first];
            const auto& tgt_p = tgt_teaser[corr.second];
            Eigen::Vector3d src_pt(src_p.x, src_p.y, src_p.z);
            Eigen::Vector3d tgt_pt(tgt_p.x, tgt_p.y, tgt_p.z);
            double dist = (src_pt - tgt_pt).norm();
            max_dist = std::max(max_dist, dist);
            avg_dist += dist;
        }
        avg_dist /= correspondences.size();
        PCL_INFO_STREAM("📐 对应点对几何统计:" << std::endl
                       << "  - 平均距离: " << avg_dist << " m" << std::endl
                       << "  - 最大距离: " << max_dist << " m" << std::endl);
    }

    // TEASER++需要至少3个对应点（用于平移估计），但建议4+以获得更好的结果
    if (correspondences.size() < 3) {
        PCL_WARN_STREAM("❌ 对应点对太少(" << correspondences.size()
                       << ")，无法进行全局配准（至少需要3个）" << std::endl);
        return false;
    }

    if (correspondences.size() < 10) {
        PCL_WARN_STREAM("⚠️  对应点对数量较少(" << correspondences.size()
                       << ")，TEASER++结果可能不够稳定" << std::endl);
    }

    // 3.3 配置TEASER++参数
    teaser::RobustRegistrationSolver::Params params;
    // 噪声边界：设置为体素大小的3-5倍，给RANSAC足够的鲁棒性
    params.noise_bound = m_config.global_voxel_size * 3.0;  // 0.3 * 3 = 0.9m
    params.cbar2 = 2.0;                               // 增大自适应噪声边界系数
    params.estimate_scaling = false;                  // 不估计尺度
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_gnc_factor = 1.4;
    params.rotation_max_iterations = 200;             // 增加迭代次数
    params.rotation_cost_threshold = 1e-6;

    // 3.4 执行TEASER++配准
    PCL_INFO_STREAM("🚀 启动TEASER++求解器..." << std::endl
                   << "  - 噪声边界: " << params.noise_bound << std::endl
                   << "  - 旋转算法: GNC-TLS" << std::endl
                   << "  - 最大迭代: " << params.rotation_max_iterations << std::endl);

    teaser::RobustRegistrationSolver solver(params);

    try {
        // TEASER++直接使用点云和对应关系，不需要手动提取矩阵
        solver.solve(src_teaser, tgt_teaser, correspondences);
        auto solution = solver.getSolution();

        PCL_INFO_STREAM("📊 TEASER++求解完成" << std::endl
                       << "  - 求解有效性: " << (solution.valid ? "✅ 有效" : "❌ 无效") << std::endl);

        if (!solution.valid) {
            std::cerr << "❌ TEASER++求解失败：解无效" << std::endl;
            return false;
        }

        // 转换为4x4变换矩阵
        Eigen::Matrix4d transformation_d = Eigen::Matrix4d::Identity();
        transformation_d.block<3,3>(0,0) = solution.rotation;
        transformation_d.block<3,1>(0,3) = solution.translation;
        M4F teaser_result = transformation_d.cast<float>();

        // 详细输出TEASER++结果
        Eigen::Vector3f teaser_t = teaser_result.block<3,1>(0,3);
        Eigen::Matrix3f teaser_R = teaser_result.block<3,3>(0,0);
        Eigen::AngleAxisf teaser_aa(teaser_R);
        float teaser_angle_deg = std::abs(teaser_aa.angle()) * 180.0f / M_PI;

        PCL_INFO_STREAM("🎯 TEASER++配准完成" << std::endl
                       << "  - 尺度因子: " << solution.scale << std::endl
                       << "  - 平移向量: [" << teaser_t(0) << ", " << teaser_t(1) << ", " << teaser_t(2) << "] m" << std::endl
                       << "  - 平移距离: " << teaser_t.norm() << " m" << std::endl
                       << "  - 旋转角度: " << teaser_angle_deg << "°" << std::endl);

        guess = teaser_result;

        // 保存可视化数据（使用原始correspondences索引）
        m_teaser_transform = guess;
        *m_teaser_source_cloud = *src_sparse_original;
        *m_teaser_target_cloud = *tgt_sparse;

        // 转换对应关系格式
        m_teaser_correspondences.clear();
        m_teaser_correspondences.reserve(correspondences.size());
        for (const auto& corr : correspondences) {
            m_teaser_correspondences.push_back(std::make_pair(corr.first, corr.second));
        }

        m_has_teaser_result = true;
        PCL_INFO_STREAM("💾 已保存TEASER++匹配点云和对应关系用于可视化" << std::endl
                       << "  - Source: " << m_teaser_source_cloud->size() << " 点" << std::endl
                       << "  - Target: " << m_teaser_target_cloud->size() << " 点" << std::endl
                       << "  - 对应点对: " << m_teaser_correspondences.size() << " 对" << std::endl);

        return true;

    } catch (const std::exception& e) {
        std::cerr << "TEASER++配准失败: " << e.what() << std::endl;
        return false;
    }
}
