//
// Created by MiseTy on 2021/9/28.
//
#include <CPoint.h>
#include <utility>

CPoint::CPoint()
    : m_position(0.0f, 0.0f, 0.0f), m_normal(0.0f, 0.0f, 0.0f),
      m_color(0.0f, 0.0f, 0.0f, 1.0f) {}

CPoint::CPoint(Eigen::Vector3f position)
    : m_position(std::move(position)), m_normal(0.0f, 0.0f, 0.0f),
      m_color(0.0f, 0.0f, 0.0f, 1.0f) {}

CPoint::CPoint(Eigen::Vector3f position, Eigen::Vector3f normal)
    : m_position(std::move(position)), m_normal(std::move(normal)),
      m_color(0.0f, 0.0f, 0.0f, 1.0f) {}

CPoint::CPoint(Eigen::Vector3f position, Eigen::Vector3f normal,
               Eigen::Vector3f color)
    : m_position(std::move(position)), m_normal(std::move(normal)),
      m_color(color.x(), color.y(), color.z(), 1.0f) {}
CPoint::CPoint(Eigen::Vector3f position, Eigen::Vector3f normal,
               Eigen::Vector4f color)
    : m_position(std::move(position)), m_normal(std::move(normal)),
      m_color(std::move(color)) {}

CPoint::~CPoint() = default;
