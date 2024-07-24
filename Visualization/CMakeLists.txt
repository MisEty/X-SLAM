project(Visualization)
message("project Visualization")

# Find Eigen
find_package(Eigen3)
if (Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else ()
    message("Miss Eigen3!")
endif ()
# Find OpenGL
find_package(OpenGL REQUIRED)
# # Find GLAD
# find_package(glad REQUIRED)
# if (NOT glad_FOUND)
#     message("Miss glad!")
# endif ()
# # Find GLFW
# find_package(glfw3 REQUIRED)
# if (NOT glfw3_FOUND)
#     message("Miss glfw3!")
# endif ()
# # Find Assimp
# find_package(assimp REQUIRED)
# if (NOT assimp_FOUND)
#     message("Miss assimp!")
# endif ()

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB src ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(${PROJECT_NAME} STATIC
        ${src}
        )
add_library(vis_lib ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
# target_link_libraries(${PROJECT_NAME} PUBLIC glfw assimp::assimp)