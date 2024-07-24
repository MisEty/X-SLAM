//
// Created by MiseTy on 2023/1/31.
//

#ifndef CSFD_SLAM_EXTRACTPOINTCLOUD_H
#define CSFD_SLAM_EXTRACTPOINTCLOUD_H

#include "Internal.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Cloud extraction

/** \brief Perform point cloud extraction from tsdf volume
 * \param[in] volume tsdf volume
 * \param[in] volume_size size of the volume
 * \param[out] output buffer large enought to store point cloud
 * \return number of point stored to passed buffer
 */
size_t extractPoints(const PtrStep<float> &value_volume, const PtrStep<int> &weight_volume, const PtrStep<float> &grad_volume, const int3 &volume_resolution,
                    float voxel_size, PtrSz<float3> output);

void extractNormals(const PtrStep<float> &value_volume, const PtrStep<int> &weight_volume, const PtrStep<float> &grad_volume, const int3 &volume_resolution,
                    float voxel_size, PtrSz<float3> points, PtrSz<float3> normal);

void extractMesh(const PtrStep<float> &value_volume, const int3 &volume_resolution,
                 float voxel_size, PtrSz<float3> points, thrustDvec<int> &indices);

////////////////////////////////////////////////////////////////////////////////////////
// Prefix Scan utility

enum ScanKind { exclusive,
                inclusive };

template<ScanKind Kind, class T>
__device__ __forceinline__ T
scan_warp(volatile T *ptr, const unsigned int idx = threadIdx.x) {
  const unsigned int lane = idx & 31;// index of thread in warp (0..31)

  if (lane >= 1) ptr[idx] = ptr[idx - 1] + ptr[idx];
  if (lane >= 2) ptr[idx] = ptr[idx - 2] + ptr[idx];
  if (lane >= 4) ptr[idx] = ptr[idx - 4] + ptr[idx];
  if (lane >= 8) ptr[idx] = ptr[idx - 8] + ptr[idx];
  if (lane >= 16) ptr[idx] = ptr[idx - 16] + ptr[idx];

  if (Kind == inclusive)
    return ptr[idx];
  else
    return (lane > 0) ? ptr[idx - 1] : 0;
}

#endif // CSFD_SLAM_EXTRACTPOINTCLOUD_H
