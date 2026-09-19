SET(PROJECT_NAME vikit_common)
PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED (VERSION 3.5)
SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")
SET(USE_ROS False) # Set False if you want to build this package without Catkin

# Set build flags. Set IS_ARM on odroid board as environment variable
SET(CMAKE_CXX_FLAGS "-Wall -D_LINUX -D_REENTRANT -march=native -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas")
message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
else()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
endif()

IF(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Add plain cmake packages 
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Eigen REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)
FIND_PACKAGE(fmt REQUIRED)
# ament 版 Sophus (ros-humble-sophus) 不设旧式 Sophus_INCLUDE_DIRS, 只导出
# Sophus::Sophus target; 从 target 回填 include 路径, 否则编译找不到 sophus/*.hpp
if(NOT Sophus_INCLUDE_DIRS AND TARGET Sophus::Sophus)
  get_target_property(Sophus_INCLUDE_DIRS Sophus::Sophus INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "vikit_common: Sophus_INCLUDE_DIRS from target = ${Sophus_INCLUDE_DIRS}")
endif()

# Include dirs
INCLUDE_DIRECTORIES(
  include
  ${Eigen_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  fmt
)

IF(USE_ROS)
  FIND_PACKAGE(catkin REQUIRED COMPONENTS roscpp cmake_modules)
  LIST(APPEND INCLUDE_DIRECTORIES ${catkin_INCLUDE_DIRS})
  catkin_package(
    DEPENDS Eigen OpenCV Sophus
    CATKIN_DEPENDS roscpp
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
  )
ELSE()
  FIND_PACKAGE(ament_cmake REQUIRED)
  SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
  SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
ENDIF()

# Set Sourcefiles
LIST(APPEND SOURCEFILES src/atan_camera.cpp
                        src/omni_camera.cpp
                        src/math_utils.cpp
                        src/vision.cpp
                        src/performance_monitor.cpp
                        src/robust_cost.cpp
                        src/user_input_thread.cpp
                        src/pinhole_camera.cpp
                        src/equidistant_camera.cpp
                        src/polynomial_camera.cpp
                        src/homography.cpp
                        src/img_align.cpp)

# Create vikit library
ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCEFILES})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${Sophus_LIBRARIES}
  fmt::fmt)

IF(USE_ROS)
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${catkin_LIBRARIES})
ENDIF()

# Tests
ADD_EXECUTABLE(test_vk_common_camera test/test_camera.cpp)
TARGET_LINK_LIBRARIES(test_vk_common_camera ${PROJECT_NAME} ${OpenCV_LIBS})

ADD_EXECUTABLE(test_vk_common_triangulation test/test_triangulation.cpp)
TARGET_LINK_LIBRARIES(test_vk_common_triangulation ${PROJECT_NAME} ${OpenCV_LIBS})

ADD_EXECUTABLE(test_vk_common_patch_score test/test_patch_score.cpp)
TARGET_LINK_LIBRARIES(test_vk_common_patch_score ${PROJECT_NAME} ${OpenCV_LIBS})


################################################################################
# Create the vikit_commonConfig.cmake file for other cmake projects.
IF(NOT USE_ROS)
  INSTALL(DIRECTORY include/vikit DESTINATION include FILES_MATCHING PATTERN "*.h")
  INSTALL(TARGETS ${PROJECT_NAME}
    EXPORT export_${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )
  ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
  ament_export_include_directories(include)
  ament_export_dependencies(OpenCV Sophus fmt)
  ament_package()
ENDIF()
