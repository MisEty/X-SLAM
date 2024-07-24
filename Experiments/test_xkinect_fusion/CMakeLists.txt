project(test_kinect_fusion)
message("project test_xkinect_fusion")

# Find yaml-cpp
set(YAML_LIB "${CMAKE_SOURCE_DIR}/extlibs/yaml-cpp/build")
set(YAML_INCLUDE "${CMAKE_SOURCE_DIR}/extlibs/yaml-cpp/include")
link_directories(${YAML_LIB})
include_directories(${YAML_INCLUDE})

set(Sophus_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/extlibs/Sophus)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
        xkinfu_lib
        libyaml-cpp.so
        )
