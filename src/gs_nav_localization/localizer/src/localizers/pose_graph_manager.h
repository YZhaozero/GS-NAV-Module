#pragma once

#include <memory>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include "scan_context.h"

// Forward declare SCDatabaseEntry (defined in icp_localizer.h)
struct SCDatabaseEntry;

// 关键帧结构
struct KeyFrame {
    double timestamp;
    gtsam::Pose3 pose;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    Eigen::MatrixXd sc_descriptor; // Scan Context 描述子
    int id;

    KeyFrame() : cloud(new pcl::PointCloud<pcl::PointXYZI>) {}
};

class PoseGraphManager {
public:
    PoseGraphManager() {
        // 初始化 ISAM2 参数
        gtsam::ISAM2Params params;
        params.relinearizeThreshold = 0.1;
        params.relinearizeSkip = 1;
        isam_ = std::make_shared<gtsam::ISAM2>(params);

        // SC 配置
        sc_config_.max_radius = 40.0;
        sc_config_.dist_thresh = 0.15;
        sc_manager_ = std::make_shared<ScanContextManager>(sc_config_);
    }

    void addKeyFrame(double timestamp, const Eigen::Matrix4f& pose_mat, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
        std::lock_guard<std::mutex> lock(mutex_);

        int current_id = keyframes_.size();

        // 转换位姿
        gtsam::Pose3 current_pose = toGtsamPose(pose_mat);

        // 创建关键帧
        auto kf = std::make_shared<KeyFrame>();
        kf->id = current_id;
        kf->timestamp = timestamp;
        kf->pose = current_pose;
        pcl::copyPointCloud(*cloud, *kf->cloud); // 保存点云副本
        kf->sc_descriptor = sc_manager_->makeScanContext(cloud);

        // 1. 添加 Prior 因子 (如果是第一帧)
        if (keyframes_.empty()) {
            gtsam::noiseModel::Diagonal::shared_ptr prior_noise =
                gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished());
            graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, current_pose, prior_noise));
            initial_estimate_.insert(0, current_pose);
        } else {
            // 2. 添加 Odometry 因子 (连接上一帧)
            int prev_id = current_id - 1;
            gtsam::Pose3 prev_pose = keyframes_.back()->pose;
            gtsam::Pose3 relative_pose = prev_pose.between(current_pose);

            gtsam::noiseModel::Diagonal::shared_ptr odom_noise =
                gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());

            graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_id, current_id, relative_pose, odom_noise));
            initial_estimate_.insert(current_id, current_pose);

            // 3. 检测回环
            detectLoopClosure(kf);
        }

        keyframes_.push_back(kf);

        // 4. 更新 ISAM2
        isam_->update(graph_, initial_estimate_);
        isam_->update();

        // 清空 graph 和 initial estimate，为下一次做准备
        graph_.resize(0);
        initial_estimate_.clear();

        // 获取优化后的当前位姿
        gtsam::Pose3 optimized_pose = isam_->calculateEstimate<gtsam::Pose3>(current_id);

        // 更新当前关键帧位姿
        keyframes_.back()->pose = optimized_pose;
    }

    Eigen::Matrix4f getOptimizedPose() {
        if (keyframes_.empty()) return Eigen::Matrix4f::Identity();
        std::lock_guard<std::mutex> lock(mutex_);
        gtsam::Pose3 p = keyframes_.back()->pose;
        return p.matrix().cast<float>();
    }

    // Export SC database for saving
    std::vector<SCDatabaseEntry> exportSCDatabase() const;

    void clearKeyframes() {
        std::lock_guard<std::mutex> lock(mutex_);
        keyframes_.clear();
        graph_.resize(0);
        initial_estimate_.clear();
        // 重新初始化ISAM2
        gtsam::ISAM2Params params;
        params.relinearizeThreshold = 0.1;
        params.relinearizeSkip = 1;
        isam_ = std::make_shared<gtsam::ISAM2>(params);
    }

    size_t getKeyframeCount() const;

private:
    gtsam::Pose3 toGtsamPose(const Eigen::Matrix4f& mat) {
        return gtsam::Pose3(mat.cast<double>());
    }

    void detectLoopClosure(const std::shared_ptr<KeyFrame>& current_kf) {
        if (keyframes_.size() < 5) return; // 帧数太少不检测

        int loop_candidate_idx = -1;
        double min_dist = 1000.0;
        int best_yaw_idx = 0;

        // 简单遍历历史关键帧 (除了最近的几个)
        for (size_t i = 0; i < keyframes_.size() - 5; ++i) {
            // 距离筛选 (只检测距离较近的关键帧)
            double spatial_dist = (current_kf->pose.translation() - keyframes_[i]->pose.translation()).norm();
            if (spatial_dist > 15.0) continue;

            // SC 匹配
            auto [dist, yaw_idx] = sc_manager_->distanceBtnScanContext(current_kf->sc_descriptor, keyframes_[i]->sc_descriptor);

            if (dist < sc_config_.dist_thresh && dist < min_dist) {
                min_dist = dist;
                loop_candidate_idx = i;
                best_yaw_idx = yaw_idx;
            }
        }

        if (loop_candidate_idx != -1) {
            std::cout << "🔄 Loop Closure Detected! Frame " << current_kf->id << " -> " << loop_candidate_idx
                      << " (Dist: " << min_dist << ")" << std::endl;

            // 计算相对位姿 (简单使用SC的yaw修正，平移直接用当前估计差)
            // 严谨做法：应该用ICP再精配准一下
            // 这里简化处理

            gtsam::Pose3 history_pose = keyframes_[loop_candidate_idx]->pose;

            // SC yaw is detected above; the current simplified pose-graph path
            // still uses the odometry-relative transform.
            (void)best_yaw_idx;

            // 相对变换
            gtsam::Pose3 relative_pose = history_pose.between(current_kf->pose); // 粗略估计

            gtsam::noiseModel::Diagonal::shared_ptr loop_noise =
                gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.2, 0.2, 0.2).finished());

            graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(loop_candidate_idx, current_kf->id, relative_pose, loop_noise));
        }
    }

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_estimate_;
    std::shared_ptr<gtsam::ISAM2> isam_;
    std::vector<std::shared_ptr<KeyFrame>> keyframes_;
    mutable std::mutex mutex_;

    SCConfig sc_config_;
    std::shared_ptr<ScanContextManager> sc_manager_;
};

// Need to include icp_localizer.h for SCDatabaseEntry definition
#include "icp_localizer.h"

// Implement const methods outside class to avoid mutex const issues
inline std::vector<SCDatabaseEntry> PoseGraphManager::exportSCDatabase() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<SCDatabaseEntry> db;
    for (const auto& kf : keyframes_) {
        SCDatabaseEntry entry;
        entry.pose = kf->pose.translation().cast<float>();
        entry.descriptor = kf->sc_descriptor;
        entry.ringkey = sc_manager_->makeRingkeyFromScancontext(kf->sc_descriptor);
        db.push_back(entry);
    }
    return db;
}

inline size_t PoseGraphManager::getKeyframeCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframes_.size();
}

