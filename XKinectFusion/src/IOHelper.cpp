#include "IOHelper.h"
#include <iomanip>

Eigen::MatrixXf loadTxtMatrix(const std::string& filename, int rows, int cols){
    std::ifstream inFile;
    inFile.open(filename);
    Eigen::MatrixXf matrix(rows, cols);
    matrix.setIdentity();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            float x;
            inFile >> x;
            matrix(i, j) = x;
        }
    }
    inFile.close();
    return matrix;
}

void saveTxtMatrix(const std::string& filename, const Eigen::MatrixXf& matrix){
    std::ofstream outFile;
    outFile.open(filename);
    int rows = matrix.rows();
    int cols = matrix.cols();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            outFile << std::setprecision(7) << std::fixed << matrix(i, j) << " ";
        }
        outFile << "\n";
    }
    outFile.close();
}