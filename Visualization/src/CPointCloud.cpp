//
// Created by MiseTy on 2021/10/10.
//
#include "CPointCloud.h"
#include <queue>

CPointCloud::CPointCloud(const std::vector<CPoint> &points)
{
    m_points = points;
    initMat = Eigen::Matrix4f::Identity();
};

void CPointCloud::addPoint(const CPoint &point) { m_points.push_back(point); }

void CPointCloud::readPly(const std::string &filename)
{
    // clear current points
    std::vector<CPoint>().swap(m_points);
    std::ifstream fin{filename};
    if (!fin.is_open())
        return;
    int read_length = 0;
    std::string s;
    while (!fin.eof())
    {
        read_length++;
        if (read_length > 11)
        {
            float x, y, z, nx, ny, nz;
            fin >> x >> y >> z >> nx >> ny >> nz;
            Eigen::Vector3f position(x, y, z);
            Eigen::Vector3f normal(nx, ny, nz);
            Eigen::Vector3f color((nx + 1.0f) / 2.0f, (ny + 1.0f) / 2.0f,
                                  (nz + 1.0f) / 2.0f);
            addPoint(CPoint(position, normal, color));
        }
        getline(fin, s);
    }
    // buildTree();
}

void CPointCloud::exportPly(const std::string &filename)
{
    std::ofstream file_out{filename};
    if (!file_out.is_open())
        return;
    int valid_count = this->size();
    file_out << "ply" << std::endl;
    file_out << "format ascii 1.0" << std::endl;
    file_out << "comment Created by myself" << std::endl;
    file_out << "element vertex " << valid_count << std::endl;
    file_out << "property float x" << std::endl;
    file_out << "property float y" << std::endl;
    file_out << "property float z" << std::endl;
    file_out << "property float nx" << std::endl;
    file_out << "property float ny" << std::endl;
    file_out << "property float nz" << std::endl;
    file_out << "end_header" << std::endl;

    for (int i = 0; i < this->size(); i++)
    {
        CPoint point = m_points[i];
        file_out << point.m_position.x() << " " << point.m_position.y() << " "
                 << point.m_position.z() << " " << point.m_normal.x() << " "
                 << point.m_normal.y() << " " << point.m_normal.z() << std::endl;
    }
}

void CPointCloud::transform(const Eigen::Matrix4f &transform)
{
    Eigen::Matrix3f rotation = transform.block<3, 3>(0, 0);
    Eigen::Vector3f translation = transform.block<3, 1>(0, 3);
    for (auto &point : m_points)
    {
        point.m_position = rotation * point.m_position + translation;
        point.m_normal = rotation * point.m_normal;
    }
}

void CPointCloud::setColor(const Eigen::Vector3f &color)
{
    for (auto &point : m_points)
    {
        point.m_color.x() = color.x();
        point.m_color.y() = color.y();
        point.m_color.z() = color.z();
        point.m_color.x() = 1.0f;
    }
}
