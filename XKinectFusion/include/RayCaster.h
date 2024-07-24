#ifndef CSFD_SLAM_RAYCASTER_H
#define CSFD_SLAM_RAYCASTER_H

#include "Internal.h"
#include "cx.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Raycast and view generation
/** \brief Generation vertex and normal maps from volume for current camera pose
 * \param[in] intr camera intrinsices
 * \param[in] Rc2v camera to volume rotation
 * \param[in] tc2v camera to volume translation
 * \param[in] Rv2w volume to world rotation
 * \param[in] tv2w volume to world translation
 * \param[in] tranc_dist volume truncation distance
 * \param[in] volume_size volume size in mm
 * \param[in] volume tsdf volume
 * \param[out] vmap output vertex map
 * \param[out] nmap output normals map
 */
void raycast(const Intr &intr, const MatS33 &Rc2v, const devComplex3 &tc2v,
             const MatS33 &Rv2w, const devComplex3 &tv2w, float tranc_dist,
             const int3 &volume_resolution, float voxel_size,
             const PtrStep<float> &value_volume, const PtrStep<float> &grad_volume,
             MapArr &vmap, MapArr &nmap);

#endif // CSFD_SLAM_RAYCASTER_H
