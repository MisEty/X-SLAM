project(test_CSFD)
message("project test_CSFD")
# Find Eigen
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
	include_directories(${EIGEN3_INCLUDE_DIR})
else ()
	message("Miss Eigen3!")
endif ()
# Find CUDA
find_package(CUDAToolkit)
if(NOT CUDAToolkit_FOUND)
	message("Miss CUDAToolkit!")
endif()

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
        device_array_lib
        CUDA::cudart
        )
