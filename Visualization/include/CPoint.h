//
// Created by MiseTy on 2021/9/28.
//

#ifndef KINECTFUSION_CPOINT_H
#define KINECTFUSION_CPOINT_H
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

class CPoint {
public:
  Eigen::Vector3f m_position, m_normal;
  Eigen::Vector4f m_color;
  CPoint();
  explicit CPoint(Eigen::Vector3f position);
  CPoint(Eigen::Vector3f position, Eigen::Vector3f normal);
  CPoint(Eigen::Vector3f position, Eigen::Vector3f normal,
         Eigen::Vector3f color);
  CPoint(Eigen::Vector3f position, Eigen::Vector3f normal,
         Eigen::Vector4f color);

  ~CPoint();
};
#endif // KINECTFUSION_CPOINT_H
