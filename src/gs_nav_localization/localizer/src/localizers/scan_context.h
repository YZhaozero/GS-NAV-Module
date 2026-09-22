#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

// Scan Context 参数
struct SCConfig {
    // 形状参数
    int num_sectors = 60;     // 扇区数 (Azimuth)
    int num_rings = 20;       // 环数 (Radial)
    double max_radius = 80.0; // 最大半径 (米)

    // 匹配参数
    double search_ratio = 0.1; // 搜索最近邻的比例 (kd-tree)
    double dist_thresh = 0.15; // SC距离阈值 (越小越严格)
};

class ScanContextManager {
public:
    using PointType = pcl::PointXYZI; // 或 pcl::PointXYZ
    using CloudType = pcl::PointCloud<PointType>;

    ScanContextManager(const SCConfig& config = SCConfig()) : config_(config) {
        // 计算每个环的径向间隔
        ring_gap_ = config_.max_radius / config_.num_rings;
        // 计算每个扇区的角度间隔
        sector_gap_ = 2.0 * M_PI / config_.num_sectors;
    }

    // 生成Scan Context描述子
    // 返回: Eigen::MatrixXd (Rows=Rings, Cols=Sectors)
    Eigen::MatrixXd makeScanContext(const CloudType::Ptr& scan_cloud) {
        Eigen::MatrixXd desc = Eigen::MatrixXd::Zero(config_.num_rings, config_.num_sectors);

        if (scan_cloud->empty()) {
            return desc;
        }

        // 🔥 改进：自动计算Z轴偏移，而不是硬编码
        // 对于地面机器人，大部分点在地面附近，需要将Z轴抬高以便SC能捕获高度信息
        float z_offset = 0.0f;
        if (scan_cloud->size() > 0) {
            // 计算点云Z值的统计信息
            float z_min = scan_cloud->points[0].z;
            float z_max = scan_cloud->points[0].z;
            for (const auto& pt : scan_cloud->points) {
                if (pt.z < z_min) z_min = pt.z;
                if (pt.z > z_max) z_max = pt.z;
            }
            // 如果点云高度范围较小（< 3m），可能是地面机器人，需要抬高
            // 否则使用原始Z值
            if (z_max - z_min < 3.0f) {
                z_offset = 2.0f - z_min; // 确保最低点在2m左右
            }
        }

        PointType pt;
        float azim_angle, azim_range; // Azimuth angle, Radial distance
        int ring_idx, sctor_idx;

        for (const auto& point : scan_cloud->points) {
            float x = point.x;
            float y = point.y;
            float z = point.z + z_offset; // 使用自适应偏移

            // 计算极坐标
            azim_range = std::sqrt(x*x + y*y);
            azim_angle = std::atan2(y, x);

            if (azim_range > config_.max_radius) continue;

            ring_idx = std::max(std::min(config_.num_rings - 1, int(ceil((azim_range / config_.max_radius) * config_.num_rings)) - 1), 0);
            sctor_idx = std::max(std::min(config_.num_sectors - 1, int(ceil((azim_angle + M_PI) / sector_gap_)) - 1), 0);

            // Scan Context 核心：取该网格内最高的Z值
            if (desc(ring_idx, sctor_idx) < z) {
                desc(ring_idx, sctor_idx) = z;
            }
        }

        return desc;
    }

    // 生成Ring Key (用于快速初筛)
    // 将SC矩阵按行求平均，得到一个向量
    Eigen::VectorXd makeRingkeyFromScancontext(const Eigen::MatrixXd& desc) {
        Eigen::VectorXd ringkey = Eigen::VectorXd::Zero(config_.num_rings);
        for (int r = 0; r < config_.num_rings; r++) {
            ringkey(r) = desc.row(r).mean();
        }
        return ringkey;
    }

    // 计算两个SC描述子的距离
    // 并返回最佳对齐的Yaw偏移 (索引)
    std::pair<double, int> distanceBtnScanContext(const Eigen::MatrixXd& sc1, const Eigen::MatrixXd& sc2) {
        // 1. 列偏移 (Sector Shift) 寻找最小距离
        int best_shift = 0;
        double min_dist = 1000000.0;

        for (int shift = 0; shift < config_.num_sectors; shift++) {
            Eigen::MatrixXd sc2_shifted = circshift(sc2, shift);
            double dist = distDirectSC(sc1, sc2_shifted);

            if (dist < min_dist) {
                min_dist = dist;
                best_shift = shift;
            }
        }

        return {min_dist, best_shift};
    }

    // 辅助：计算矩阵的列循环移位
    Eigen::MatrixXd circshift(const Eigen::MatrixXd& mat, int shift) {
        if (shift == 0) return mat;
        Eigen::MatrixXd shifted = Eigen::MatrixXd::Zero(mat.rows(), mat.cols());
        int cols = mat.cols();
        for (int c = 0; c < cols; c++) {
            int new_c = (c + shift) % cols;
            shifted.col(new_c) = mat.col(c);
        }
        return shifted;
    }

    // 辅助：计算两个对齐SC的余弦距离
    double distDirectSC(const Eigen::MatrixXd& sc1, const Eigen::MatrixXd& sc2) {
        int num_eff_cols = 0;
        double sum_dist = 0;

        for (int c = 0; c < sc1.cols(); c++) {
            Eigen::VectorXd col1 = sc1.col(c);
            Eigen::VectorXd col2 = sc2.col(c);

            if (col1.norm() == 0 || col2.norm() == 0) continue; // 跳过空列

            double cos_sim = col1.dot(col2) / (col1.norm() * col2.norm());
            sum_dist += (1.0 - cos_sim);
            num_eff_cols++;
        }

        if (num_eff_cols == 0) return 1.0;
        return sum_dist / num_eff_cols;
    }

private:
    SCConfig config_;
    double ring_gap_;
    double sector_gap_;
};








