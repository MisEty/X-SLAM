#ifndef CSFD_SLAM_TSDFFUSION_H
#define CSFD_SLAM_TSDFFUSION_H
#include "TsdfVolume.h"
#include "cx.h"

// TSDF volume functions
__device__ __forceinline__ void
pack_tsdf(devComplex tsdf, int weight, float &save_value, int &save_weight, float &save_grad) {
    save_value = tsdf.real();
    save_weight = weight;
    save_grad = tsdf.imag();
}

__device__ __forceinline__ void
unpack_tsdf(float save_value, int save_weight, float save_grad,
            devComplex &tsdf, int &weight) {
    weight = save_weight;
    tsdf = devComplex(save_value, save_grad);
}

__device__ __forceinline__ devComplex
unpack_tsdf(float save_value, float save_grad) {
    devComplex res(save_value, save_grad);
    return res;
}

/** \brief Function that integrates volume if volume element contains: 2 bytes
 * for round(tsdf*SHORT_MAX) and 2 bytes for integer weight.
 * \param[in] depth_raw Kinect depth image \param[in] intr camera intrinsics
 * \param[in] resolution of the volume
 * \param[in] voxel size in mm
 * \param[in] Rv2c volume to camera rotation
 * \param[in] tc2v camera to volume translation, CAUTION!!!,
 * Rv2c and tc2v doesn't form the transform matrix
 * \param[in] tranc_dist tsdf truncation distance
 * \param[in] volume tsdf volume to be updated
 * \param[in] gradient of volume tsdf volume to be updated
 * \param[out] depthScaled Temp Buffer for scaled depth along ray, auto release
 */
void integrateTsdfVolume(const PtrStepSz<ushort> &depth, const Intr &intr,
                         int max_weight, const int3 &volume_resolution,
                         float voxel_size, const MatS33 &Rv2c, const devComplex3 &tv2c, const devComplex3 &tc2v,
                         float tranc_dist, PtrStep<float> value_volume,
                         PtrStep<int> weight_volume, PtrStep<float> grad_volume,
                         DeviceArray2D<float> &depthScaled, int frame_id, float threshold = 0.0f, float k = 0.0f);

// compute the tsdf volume from pose matrix and depth image
float2 ComputeLocalTsdf_loss(const PtrStepSz<ushort> &depth, const Intr &intr, DeviceArray2D<float> &depthScaled,
                             const int3 &volume_resolution, float voxel_size,
                             const Mat33 &Rv2c, const float3 &tv2c,
                             float tranc_dist, float threshold, float k,
                             thrustDvec<float> &gt_vec, thrustDvec<float> &real_vec, thrustDvec<int> &count_vec);

// compute the tsdf volume from pose matrix and depth image
float4 ComputeLocalTsdf_hessian(const PtrStepSz<ushort> &depth, const Intr &intr, DeviceArray2D<float> &depthScaled,
                                const int3 &volume_resolution, float voxel_size,
                                const MatD33 &Rv2c, const devDComplex3 &tv2c,
                                float tranc_dist, float threshold, float k,
                                thrustDvec<float> &gt_vec, thrustDvec<float> &real_vec, thrustDvec<float> &grad_vec,
                                thrustDvec<float> &hessian_vec, thrustDvec<int> &count_vec);
#endif// CSFD_SLAM_TSDFFUSION_H
