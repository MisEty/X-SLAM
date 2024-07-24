//
// Created by MiseTy on 2021/8/3.
//

#ifndef KINECTFUSION_TRANSFORMHELPER_H
#define KINECTFUSION_TRANSFORMHELPER_H

#include <climits>
#include <ctime>
#include <random>

#include <Eigen/Dense>

static std::default_random_engine g_engine(std::time(nullptr));
static std::uniform_int_distribution<int> g_uniform(0, INT_MAX);

static Eigen::Matrix4f scale(const float factor) {
  Eigen::Matrix4f ans = Eigen::Matrix4f::Zero();

  ans(0, 0) = ans(1, 1) = ans(2, 2) = factor;
  ans(3, 3) = 1.0f;

  return ans;
}

static Eigen::Matrix4f rotate(const Eigen::Vector3f &v, const float angle) {
  Eigen::Vector3f axis = v.normalized();
  float s = std::sin(angle);
  float c = std::cos(angle);
  Eigen::Matrix4f ans = Eigen::Matrix4f::Zero();

  ans(0, 0) = (1.0f - c) * axis(0) * axis(0) + c;
  ans(0, 1) = (1.0f - c) * axis(1) * axis(0) - s * axis(2);
  ans(0, 2) = (1.0f - c) * axis(2) * axis(0) + s * axis(1);

  ans(1, 0) = (1.0f - c) * axis(0) * axis(1) + s * axis(2);
  ans(1, 1) = (1.0f - c) * axis(1) * axis(1) + c;
  ans(1, 2) = (1.0f - c) * axis(2) * axis(1) - s * axis(0);

  ans(2, 0) = (1.0f - c) * axis(0) * axis(2) - s * axis(1);
  ans(2, 1) = (1.0f - c) * axis(1) * axis(2) + s * axis(0);
  ans(2, 2) = (1.0f - c) * axis(2) * axis(2) + c;

  ans(3, 3) = 1.0f;

  return ans;
}

static Eigen::Matrix4f translate(const Eigen::Vector3f &v) {
  Eigen::Matrix4f ans = Eigen::Matrix4f::Identity();

  ans(0, 3) = v(0);
  ans(1, 3) = v(1);
  ans(2, 3) = v(2);

  return ans;
}

static Eigen::Matrix4f lookAt(const Eigen::Vector3f &position,
                              const Eigen::Vector3f &center,
                              const Eigen::Vector3f &up) {
  Eigen::Vector3f f = (center - position).normalized();
  Eigen::Vector3f s = f.cross(up).normalized();
  Eigen::Vector3f u = s.cross(f);
  Eigen::Matrix4f ans = Eigen::Matrix4f::Zero();

  ans(0, 0) = s(0);
  ans(0, 1) = s(1);
  ans(0, 2) = s(2);
  ans(0, 3) = -s.dot(position);

  ans(1, 0) = u(0);
  ans(1, 1) = u(1);
  ans(1, 2) = u(2);
  ans(1, 3) = -u.dot(position);

  ans(2, 0) = -f(0);
  ans(2, 1) = -f(1);
  ans(2, 2) = -f(2);
  ans(2, 3) = f.dot(position);

  ans(3, 3) = 1.0f;

  return ans;
}

static Eigen::Matrix4f perspective(const float fovy, const float aspect,
                                   const float zNear, const float zFar) {
  float t = std::tan(fovy * 0.5f);
  Eigen::Matrix4f ans = Eigen::Matrix4f::Zero();

  ans(0, 0) = 1.0f / (aspect * t);
  ans(1, 1) = 1.0f / t;
  ans(2, 2) = -(zNear + zFar) / (zFar - zNear);
  ans(2, 3) = -2.0f * zNear * zFar / (zFar - zNear);
  ans(3, 2) = -1.0f;

  return ans;
}

#endif // KINECTFUSION_TRANSFORMHELPER_H
