#ifndef GRAD_KINECTFUSION_VISUALIZATION_H
#define GRAD_KINECTFUSION_VISUALIZATION_H

#include "CMesh.h"
#include "CModel.h"
#include "CPointCloud.h"
// #include "IO_helper.hpp"
#include "Shader.h"
#include "TransformHelper.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// #include <vector_types.h>

#include <Windows.h>
#include <commdlg.h>
#include <deque>
#include <iostream>

static void framebufferSizeCallback(GLFWwindow *window, int width, int height);
static void mouseButtonCallback(GLFWwindow *window, int button, int action,
                                int mods);
static void cursorPosCallback(GLFWwindow *window, double x, double y);
static void scrollCallback(GLFWwindow *window, double x, double y);

class Visualization {
public:
  GLFWwindow *window;
  unsigned int windowWidth, windowHeight;

  bool is_reading = true;
//  bool is_stopped = false;
//  bool is_drawing = true;
//  bool is_control = true;
  bool is_main;

  float minX, maxX, minY, maxY, minZ, maxZ; // bbox of current scene
  int frame_id;
  float fps;
  // camera setting
  Eigen::Vector3f lightDirection{};
  Eigen::Vector3f cameraPosition{};
  float fovy{};
  Eigen::Vector3f lookCenter{};

  // objects in scene
  std::vector<CPointCloud> m_pointclouds;
  std::vector<CMesh> m_meshes;

  // render parameters
  Eigen::Matrix4f modelMat, viewMat, projectionMat;
  Eigen::Vector3f scene_center;
  Shader PointShader;

  explicit Visualization(
      bool is_main = true, unsigned int windowWidth = 1920,
      unsigned int windowHeight = 1080,
      const std::string &name = "PointcloudViewer",
      const std::string &vertexShader =
          "D:/Projects/CSFD_SLAM_new/External/shader/Vertex.glsl",
      const std::string &fragShader =
          "D:/Projects/CSFD_SLAM_new/External/shader/Fragment.glsl");
  ~Visualization();
  void setCallBack();

  void AddPointcloud(CPointCloud &pointcloud,
                     Eigen::Vector3f center = Eigen::Vector3f(0, 0, 0),
                     bool use_center = false);
  void UpdatePointCloud(CPointCloud &pointcloud, int i,
                        Eigen::Vector3f center = Eigen::Vector3f(0, 0, 0),
                        bool use_center = false);
  void AddMesh(CMesh &mesh, Eigen::Vector3f center = Eigen::Vector3f(0, 0, 0),
               bool use_center = false);
  void UpdateMesh(CMesh &mesh, int i,
                  Eigen::Vector3f center = Eigen::Vector3f(0, 0, 0),
                  bool use_center = false);
  void Render();
};

#endif // GRAD_KINECTFUSION_VISUALIZATION_H
