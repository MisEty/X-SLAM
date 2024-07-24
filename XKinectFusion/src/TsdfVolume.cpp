//
// Created by MiseTy on 2021/10/28.
//
#include "TsdfVolume.h"
#include "Internal.h"
#include <algorithm>
#include <utility>

using namespace Eigen;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TsdfVolume::TsdfVolume(Vector3i resolution, float voxel_size, float thres_range)
    : resolution_(std::move(resolution)) {
    int volume_x = resolution_(0);
    int volume_y = resolution_(1);
    int volume_z = resolution_(2);

    volume_.create(volume_y * volume_z, volume_x);
    value_volume_.create(volume_y * volume_z, volume_x);
    weight_volume_.create(volume_y * volume_z, volume_x);
    grad_volume_.create(volume_y * volume_z, volume_x);

    setVoxelSize(voxel_size);

    // const Vector3f default_volume_size = Vector3f::Constant (3.f1); //meters
    const float default_tranc_dist = voxel_size * thres_range;// meters
    setTsdfTruncDist(default_tranc_dist);

    reset();
}
void TsdfVolume::setVoxelSize(float voxel_size) {
    voxel_size_ = voxel_size;
    setTsdfTruncDist(tranc_dist_);
}

void TsdfVolume::setTsdfTruncDist(float distance) {
    // Tsdf truncation distance can't be less than 2 * voxel_size
    tranc_dist_ = std::max(distance, 2.1f * voxel_size_);
}

float TsdfVolume::getTsdfTruncDist() const {
    return tranc_dist_;
}

void TsdfVolume::reset() {
    int3 volume_resolution;
    volume_resolution.x = resolution_(0);
    volume_resolution.y = resolution_(1);
    volume_resolution.z = resolution_(2);

    initVolume(volume_, value_volume_, weight_volume_, grad_volume_, volume_resolution);
}
DeviceArray2D<int> TsdfVolume::data() const {
    return volume_;
}
DeviceArray2D<float> TsdfVolume::value() const {
    return value_volume_;
}
DeviceArray2D<int> TsdfVolume::weight() const {
    return weight_volume_;
}
DeviceArray2D<float> TsdfVolume::grad() const {
    return grad_volume_;
}
void TsdfVolume::downloadTSDFWithGrad(std::vector<float> &tsdf, std::vector<float> &grad) const {
    tsdf.resize(volume_.cols() * volume_.rows());
    grad.resize(volume_.cols() * volume_.rows());
    value_volume_.download(&tsdf[0], volume_.cols() * sizeof(float));
    grad_volume_.download(&grad[0], volume_.cols() * sizeof(float));
}
void TsdfVolume::downloadTSDFWithoutGrad(std::vector<float> &tsdf) const {
    tsdf.resize(volume_.cols() * volume_.rows());
    value_volume_.download(&tsdf[0], volume_.cols() * sizeof(float));
}
void TsdfVolume::downloadWeight(std::vector<int> &weight) const {
    weight.resize(volume_.cols() * volume_.rows());
    weight_volume_.download(&weight[0], volume_.cols() * sizeof(int));
}
