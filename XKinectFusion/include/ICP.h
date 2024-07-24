#ifndef CSFD_SLAM_ICP_H
#define CSFD_SLAM_ICP_H

#include "Internal.h"
#include "Eigen/Eigen"
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   ICP
/** \brief Computation Ax=b for ICP iteration
 * \param[in] Rcurr Rotation of current camera pose guess
 * \param[in] tcurr translation of current camera pose guess
 * \param[in] vmap_curr current vertex map in camera coo space
 * \param[in] nmap_curr current vertex map in camera coo space
 * \param[in] Rprev_inv inverse camera rotation at previous pose
 * \param[in] tprev camera translation at previous pose
 * \param[in] intr camera intrinsics
 * \param[in] vmap_g_prev previous vertex map in global coo space
 * \param[in] nmap_g_prev previous vertex map in global coo space
 * \param[in] distThres distance filtering threshold
 * \param[in] angleThres angle filtering threshold. Represents sine of angle
 * between normals \param[out] gbuf temp buffer for GPU reduction \param[out]
 * mbuf ouput GPU buffer for matrix computed \param[out] matrixA_host A
 * \param[out] vectorB_host b
 */
void estimateCombined(const MatS33 &Rcurr, const devComplex3 &tcurr,
                      const MapArr &vmap_curr, const MapArr &nmap_curr,
                      const MatS33 &Rprev_inv, const devComplex3 &tprev,
                      const Intr &intr, const MapArr &vmap_g_prev,
                      const MapArr &nmap_g_prev, float distThres,
                      float angleThres, DeviceArray2D<devComplexICP> &gbuf,
                      DeviceArray<devComplexICP> &mbuf, hostComplexICP *matrixA_host,
                      hostComplexICP *vectorB_host) ;

//   compute jacobi
void computeOptimizeMatrix(const MapArr &vmap_curr, const MapArr &nmap_curr,
                           const MapArr &vmap_g_prev, const MapArr &nmap_g_prev,
                           const MatS33 &Rcurr, const devComplex3 &tcurr,
                           const MatS33 &Rprev_inv, const devComplex3 &tprev,
                           const Intr &intr, float distThres, float angleThres,
                           DeviceArray2D<float> &jacobi_buf, Eigen::Matrix4f &jacobi_host,
                           DeviceArray2D<float> *hessian_buf, Eigen::Matrix4f **hessian_host);

#endif // CSFD_SLAM_ICP_H
