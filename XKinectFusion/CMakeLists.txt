project(XKinectFusion)
message("project XKinectFusion")
# Find Eigen
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else ()
    message("Miss Eigen3!")
endif ()

# Find cuda
find_package(CUDAToolkit REQUIRED)
if (NOT CUDAToolkit_FOUND)
    message("Miss CUDAToolkit!")
endif ()

# Find opencv
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
else ()
    message("Miss OpenCV!")
endif ()

# Find yaml-cpp
set(YAML_LIB "${CMAKE_SOURCE_DIR}/extlibs/yaml-cpp/build")
set(YAML_INCLUDE "${CMAKE_SOURCE_DIR}/extlibs/yaml-cpp/include")
link_directories(${YAML_LIB})
include_directories(${YAML_INCLUDE})

# Find Sophus
set(Sophus_DIRS ${CMAKE_SOURCE_DIR}/extlibs/Sophus/build)
set(Sophus_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/extlibs/Sophus)

find_package(Sophus REQUIRED)
if(NOT Sophus_FOUND)
    message(FATAL_ERROR "Sophus not found!")
else ()
    message(STATUS "Sophus include directories: ${Sophus_INCLUDE_DIRS}")
    message(STATUS "Sophus directories: ${Sophus_DIRS}")
endif()
include_directories(${Sophus_INCLUDE_DIRS})


file(GLOB src "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.cu")
# add_executable(${PROJECT_NAME} ${src})

add_library(${PROJECT_NAME} STATIC
        ${src}
        )
add_library(xkinfu_lib ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PUBLIC
        common_lib
        device_array_lib
        vis_lib
        CUDA::cudart
        libyaml-cpp.so
        ${OpenCV_LIBS}
        ${Sophus_LIBRARYS}
        Sophus::Sophus
        )
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "86")