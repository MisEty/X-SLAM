//
// Created by MiseTy on 2023/1/31.
//

#ifndef CSFD_SLAM_MAP_H
#define CSFD_SLAM_MAP_H

#include "Internal.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Maps
/** \brief Performs bilateral filtering of disparity map and save as complex map
 * \param[in] src source ushort map
 * \param[out] dst output complex map
 */
void bilateralFilter(const DeviceArray2D<ushort> &src, MapArr &dst);

/** \brief Computes depth pyramid
 * \param[in] src source
 * \param[out] dst destination
 */
void pyrDown(const MapArr &src, MapArr &dst);

/** \brief Computes vertex map
 * \param[in] intr depth camera intrinsics
 * \param[in] depth depth
 * \param[out] vmap vertex map
 */
void createVMap(const Intr &intr, const MapArr &depth, MapArr &vmap);

/** \brief Computes normal map using cross product
 * \param[in] vmap vertex map
 * \param[out] nmap normal map
 */
void createNMap(const MapArr &vmap, MapArr &nmap);

/** \brief Computes normal map using Eigen/PCA approach
 * \param[in] vmap vertex map
 * \param[out] nmap normal map
 */

/** \brief Performs resize of vertex map to next pyramid level by averaging each
 * four points
 * \param[in] input vertext map
 * \param[out] output resized vertex map
 */
void resizeVMap(const MapArr &input, MapArr &output);

/** \brief Performs resize of vertex map to next pyramid level by averaging each
 * four normals
 * \param[in] input normal map
 * \param[out] output vertex map
 */
void resizeNMap(const MapArr &input, MapArr &output);

///** \brief Project vertex maps into occupy volume
// * \param[in] vmap_src source vertex map
// * \param[in] Rmat Rotation mat
// * \param[in] tvec translation
// * \param[out] occupy_volume occup flag volume
// */
//void ProjMaps2Volume(const MapArr &vmap_src, const Mat33 &Rmat,
//                     const float3 &tvec, PtrStep<bool> occupy_volume,
//                     int3 volume_resolution, float voxel_size);

#endif // CSFD_SLAM_MAP_H
