###########################################################
#                                                         #
# Copyright (c)                                           #
#                                                         #
# The Verifiable & Control-Theoretic Robotics (VECTR) Lab #
# University of California, Los Angeles                   #
#                                                         #
# Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   #
# Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         #
#                                                         #
###########################################################

cmake_minimum_required(VERSION 3.12.4)
project(direct_lidar_inertial_odometry)

find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED COMPONENTS date_time)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
# ROS 2 Humble's pcl_ros is built against PCL 1.12.  A newer PCL installed in
# /usr/local must not be mixed into the same process because their ABIs differ.
set(system_pcl_config_dir "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake/pcl")
if(EXISTS "${system_pcl_config_dir}/PCLConfig.cmake")
  find_package(PCL REQUIRED CONFIG
    PATHS "${system_pcl_config_dir}"
    NO_DEFAULT_PATH
  )
else()
  find_package(PCL REQUIRED)
endif()
find_package(Threads REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# Not all machines have <cpuid.h> available
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

rosidl_generate_interfaces(
    ${PROJECT_NAME}
        srv/SavePCD.srv
    DEPENDENCIES       
)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(include)

# NanoFLANN
add_library(nanoflann STATIC
  src/nano_gicp/nanoflann.cc
)
target_link_libraries(nanoflann ${PCL_LIBRARIES})
ament_target_dependencies(nanoflann)

# NanoGICP
add_library(nano_gicp STATIC
  src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)
target_link_libraries(nano_gicp ${PCL_LIBRARIES} OpenMP::OpenMP_CXX nanoflann)
ament_target_dependencies(nano_gicp)

# Odometry Node
add_executable(dlio_odom_node src/dlio/odom_node.cc src/dlio/odom.cc)
target_link_libraries(dlio_odom_node ${PCL_LIBRARIES} OpenMP::OpenMP_CXX Threads::Threads nano_gicp)
ament_target_dependencies(dlio_odom_node "tf2_ros" "pcl_ros" "nav_msgs" "pcl_conversions" "rclcpp" "sensor_msgs" "geometry_msgs")

# Mapping Node
add_executable (dlio_map_node src/dlio/map_node.cc src/dlio/map.cc)
target_link_libraries(dlio_map_node ${PCL_LIBRARIES}  ${cpp_typesupport_target})
ament_target_dependencies(dlio_map_node "tf2_ros" "pcl_ros" "pcl_conversions" "rclcpp")

ament_export_dependencies(
  geometry_msgs
  nav_msgs
  pcl_conversions
  pcl_ros
  rclcpp
  sensor_msgs
  std_msgs
  tf2_ros
)
ament_package()

# Binaries
install( TARGETS dlio_odom_node dlio_map_node
         DESTINATION lib/${PROJECT_NAME} 
)
install( DIRECTORY cfg launch
         DESTINATION share/${PROJECT_NAME} 
)

file(WRITE ${CMAKE_BINARY_DIR}/test_cpuid.cpp "#include <cpuid.h>")
try_compile(HAS_CPUID ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
file(REMOVE ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
if(HAS_CPUID)
  add_compile_definitions(HAS_CPUID)
endif()
