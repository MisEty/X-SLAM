#ifndef CSFD_SLAM_NEW_IOHELPER_H
#define CSFD_SLAM_NEW_IOHELPER_H
#include "Eigen/Dense"
#include <fstream>

Eigen::MatrixXf loadTxtMatrix(const std::string& filename, int rows, int cols);

void saveTxtMatrix(const std::string& filename, const Eigen::MatrixXf& matrix);
#endif//CSFD_SLAM_NEW_IOHELPER_H
