//
// The MIT License (MIT)
//
// Copyright (c) 2022 Livox. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include <cstdio>
#include <cstdlib>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "driver_node.h"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  {
    auto node = std::make_shared<livox_ros::DriverNode>(rclcpp::NodeOptions{});
    rclcpp::spin(node);
  }
  rclcpp::shutdown();

  // Livox-SDK2 1.2.x embeds an older spdlog whose process-global destructor
  // conflicts with the spdlog ABI loaded by ROS 2 Fast DDS. All driver, SDK,
  // ROS and worker-thread cleanup has completed above; skip only static
  // process teardown to avoid the third-party logger SIGBUS on normal exit.
  std::fflush(nullptr);
  std::_Exit(EXIT_SUCCESS);
}
