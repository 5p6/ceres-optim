cmake_minimum_required(VERSION 3.0)
project(first)

aux_source_directory(. DIRS)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

add_executable(first test.cc)
target_link_libraries(first ${CERES_LIBRARIES})
