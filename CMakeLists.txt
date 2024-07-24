cmake_minimum_required(VERSION 3.19)
project(XSLAM)
message("project XSLAM")
enable_language(CUDA)
add_compile_options(-w)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(Common)
add_subdirectory(DeviceArray)
add_subdirectory(Visualization)
add_subdirectory(XKinectFusion)

add_subdirectory(Experiments/test_xkinect_fusion)
add_subdirectory(Experiments/test_CSFD)
