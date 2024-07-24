//
// Created by MiseTy on 2021/10/10.
//

#ifndef KINECTFUSION_CPOINTCLOUD_H
#define KINECTFUSION_CPOINTCLOUD_H
#include "CPoint.h"

#include <vector>

#include <GLFW/glfw3.h>

class CPointCloud
{
public:
  std::vector<CPoint> m_points;

  unsigned int m_vao{};

  Eigen::Matrix4f initMat = Eigen::Matrix4f::Identity();

  bool isShow = false;

  //////////////////////////////////  functions //////////////////////////////////////
  CPointCloud() = default;

  explicit CPointCloud(const std::vector<CPoint> &points);

  ~CPointCloud() = default;

  void bindBuffer();

  void render() const;

  size_t size() const { return m_points.size(); };

  void setColor(const Eigen::Vector3f &color);

  void addPoint(const CPoint &point);

  void exportPly(const std::string &filename);

  void readPly(const std::string &filename);

  void transform(const Eigen::Matrix4f &transform);

  float pointToPlaneDist(const CPointCloud &pcd, float dist_thres, float angle_thres);
};
#endif // KINECTFUSION_CPOINTCLOUD_H
