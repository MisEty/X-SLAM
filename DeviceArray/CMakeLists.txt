project(device_array)
message("project device_array")

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

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB src ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(${PROJECT_NAME} STATIC
${src}
)
add_library(device_array_lib ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} CUDA::cudart common_lib)
