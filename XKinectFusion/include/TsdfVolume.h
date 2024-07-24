//
// Created by MiseTy on 2021/10/28.
//

#ifndef GRAD_KINECTFUSION_TSDF_VOLUME_H
#define GRAD_KINECTFUSION_TSDF_VOLUME_H
#include "Internal.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

/** \brief Perform tsdf volume initialization
 *  \param[out] array volume to be initialized
 */
void initVolume(PtrStep<short2> volume, PtrStep<float> value_volume, PtrStep<int> weight_volume, PtrStep<float> grad_volume, const int3 &volume_resolution);

class TsdfVolume {
private:
    /** \brief tsdf volume size in meters */
    float voxel_size_{};

    /** \brief tsdf volume resolution */
    Eigen::Vector3i resolution_;

    /** \brief tsdf volume data container */
    DeviceArray2D<int> volume_;

    /** \brief tsdf volume value container */
    DeviceArray2D<float> value_volume_;

    /** \brief tsdf volume weight container */
    DeviceArray2D<int> weight_volume_;

    /** \brief tsdf volume grad container */
    DeviceArray2D<float> grad_volume_;

    /** \brief tsdf truncation distance */
    float tranc_dist_{};

public:
    TsdfVolume(Eigen::Vector3i resolution, float voxel_size, float thres_range);

    void setVoxelSize(float voxel_size);

    void setTsdfTruncDist(float distance);

    DeviceArray2D<int> data() const;
    DeviceArray2D<float> value() const;
    DeviceArray2D<int> weight() const;
    DeviceArray2D<float> grad() const;

    float getTsdfTruncDist() const;

    void reset();

    void downloadTSDFWithGrad(std::vector<float> &tsdf, std::vector<float> &grad) const;
    void downloadTSDFWithoutGrad(std::vector<float> &tsdf) const;
    void downloadWeight(std::vector<int> &weight) const;
};

#endif//GRAD_KINECTFUSION_TSDF_VOLUME_H
