/**
 * @file generate_sc_tool.cpp
 * @brief Standalone tool to generate SC database for PCD maps
 */

#include <iostream>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "localizers/icp_localizer.h"
#include "localizers/scan_context.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <pcd_file>" << std::endl;
        return 1;
    }

    std::string pcd_path = argv[1];
    std::string sc_path = pcd_path + ".sc";

    // Check if PCD exists
    if (!std::filesystem::exists(pcd_path)) {
        std::cerr << "❌ PCD file not found: " << pcd_path << std::endl;
        return 1;
    }

    // Check if SC already exists
    if (std::filesystem::exists(sc_path)) {
        std::cout << "⏭️  SC database already exists: " << sc_path << std::endl;
        return 0;
    }

    std::cout << "🔄 Generating SC database for: " << pcd_path << std::endl;

    try {
        // Create ICPLocalizer with default config
        ICPConfig config;
        config.use_scan_context = true;
        config.sc_max_radius = 80.0;
        config.sc_dist_thresh = 0.5;
        config.sc_grid_resolution = 2.0;

        ICPLocalizer localizer(config);

        // Load map (this will trigger SC database generation)
        if (!localizer.loadMap(pcd_path)) {
            std::cerr << "❌ Failed to load map: " << pcd_path << std::endl;
            return 1;
        }

        // Check if SC file was created
        if (std::filesystem::exists(sc_path)) {
            auto size = std::filesystem::file_size(sc_path);
            std::cout << "✅ SC database generated: " << sc_path
                     << " (" << (size / 1024.0) << " KB)" << std::endl;
            return 0;
        } else {
            std::cerr << "❌ SC database not created" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
