#include "TsdfFusion.h"

///////////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void initializeVolume(PtrStep<T> volume,
                                 PtrStep<float> value_volume,
                                 PtrStep<int> weight_volume,
                                 PtrStep<float> grad_volume,
                                 const int3 volume_resolution) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < volume_resolution.x && y < volume_resolution.y) {
        T *pos = volume.ptr(y) + x;
        float *value_pos = value_volume.ptr(y) + x;
        int *weight_pos = weight_volume.ptr(y) + x;
        float *grad_pos = grad_volume.ptr(y) + x;

        int z_step = volume_resolution.y * volume.step / sizeof(*pos);
        int value_z_step = volume_resolution.y * volume.step / sizeof(*value_pos);
        int weight_z_step = volume_resolution.y * volume.step / sizeof(*weight_pos);
        int grad_z_step = volume_resolution.y * volume.step / sizeof(*grad_pos);

#pragma unroll
        for (int z = 0; z < volume_resolution.z; ++z, pos += z_step,
                 value_pos += value_z_step, weight_pos += weight_z_step, grad_pos += grad_z_step) {
            pack_tsdf(devComplex(0.f, 0.f), 0, *value_pos, *weight_pos, *grad_pos);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////
void initVolume(PtrStep<short2> volume, PtrStep<float> value_volume, PtrStep<int> weight_volume, PtrStep<float> grad_volume, const int3 &volume_resolution) {
    dim3 block(32, 16);
    dim3 grid(1, 1, 1);
    grid.x = divUp(volume_resolution.x, block.x);
    grid.y = divUp(volume_resolution.y, block.y);

    initializeVolume<<<grid, block>>>(volume, value_volume, weight_volume, grad_volume, volume_resolution);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}


///////////////////////////////////////////////////////////////////////////////////
__global__ void scaleDepthToRayKernal(const PtrStepSz<ushort> depth, PtrStep<float> scaled,
                                      const Intr intr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.cols || y >= depth.rows)
        return;

    int Dp = depth.ptr(y)[x];
    if (Dp > 5000 || Dp < 200) {
        scaled.ptr(y)[x] = 0;
        return;
    }

    float xl = (float(x) - intr.cx) / intr.fx;
    float yl = (float(y) - intr.cy) / intr.fy;
    float lambda = sqrtf(xl * xl + yl * yl + 1);
    scaled.ptr(y)[x] = float(Dp) * lambda / 1000.f;// meters
}

///////////////////////////////////////////////////////////////////////////////////
__global__ void scaleDepthKernal(const PtrStepSz<ushort> depth, PtrStep<float> scaled,
                                 const Intr intr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.cols || y >= depth.rows)
        return;

    int Dp = depth.ptr(y)[x];
    if (Dp > 5000 || Dp < 200) {
        scaled.ptr(y)[x] = 0;
        return;
    }
    scaled.ptr(y)[x] = float(Dp) / 1000.f;// meters
}

///////////////////////////////////////////////////////////////////////////////////
__global__ void
tsdfFusionKernal(PtrStepSz<float> depthScaled,
                 PtrStep<float> value_volume, PtrStep<int> weight_volume,
                 PtrStep<float> grad_volume, int3 volume_resolution,
                 float tranc_dist,
                 int max_weight, MatS33 Rv2c, devComplex3 tv2c,
                 Intr intr, float voxel_size, float threshold, float k) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= volume_resolution.x || y >= volume_resolution.y)
        return;
    devComplex Rv2c_0_z_scaled = Rv2c.data[0].z * voxel_size * intr.fx;
    devComplex Rv2c_1_z_scaled = Rv2c.data[1].z * voxel_size * intr.fy;
    float tranc_dist_inv = 1.0f / tranc_dist;
    float *pos = value_volume.ptr(y) + x;
    int elem_step = value_volume.step * volume_resolution.y / sizeof(float);
    int *weight_pos = weight_volume.ptr(y) + x;
    int weight_elem_step = weight_volume.step * volume_resolution.y / sizeof(int);
    float *grad_pos = grad_volume.ptr(y) + x;
    int grad_elem_step = grad_volume.step * volume_resolution.y / sizeof(float);
    for (int z = 0; z < volume_resolution.z; ++z,
             pos += elem_step,
             weight_pos += weight_elem_step,
             grad_pos += grad_elem_step) {
        devComplex v_g_x = (x + 0.5f) * voxel_size;
        devComplex v_g_y = (y + 0.5f) * voxel_size;
        devComplex v_g_z = (z + 0.5f) * voxel_size;
        devComplex3 v_g = make_mcomplex3(v_g_x, v_g_y, v_g_z);
        devComplex3 v_c = Rv2c * v_g + tv2c;
        devComplex inv_z = 1.0f / (v_c.z);
        if (inv_z.real() < 0)
            continue;
        devComplex image_x = v_c.x * intr.fx * inv_z + intr.cx;
        devComplex image_y = v_c.y * intr.fy * inv_z + intr.cy;
        int2 coo = {__float2int_rd(image_x.real() - 0.5f),
                    __float2int_rd(image_y.real() - 0.5f)};

        if (coo.x > 1 && coo.y > 1 && coo.x < depthScaled.cols - 1 &&
            coo.y < depthScaled.rows - 1) {
            int2 coo_near = {__float2int_rn(image_x.real()),
                             __float2int_rn(image_y.real())};
            devComplex Dp_near(depthScaled.ptr(coo_near.y)[coo_near.x], 0.0f);
            devComplex Dp;
            float d00 = depthScaled.ptr(coo.y)[coo.x];
            float d10 = depthScaled.ptr(coo.y)[coo.x + 1];
            float d01 = depthScaled.ptr(coo.y + 1)[coo.x];
            float d11 = depthScaled.ptr(coo.y + 1)[coo.x + 1];
            float gird_max = fmax(d00, fmax(d01, fmax(d10, d11)));
            float gird_min = fmin(d00, fmin(d01, fmin(d10, d11)));
            if (gird_max - gird_min < threshold && d00 != 0.0f & d01 != 0.0f & d10 != 0.0f & d11 != 0.0f) {
                devComplex one(1.0f, 0.0f);
                devComplex a = image_x - devComplex(coo.x + 0.5f, 0.0f);
                devComplex b = image_y - devComplex(coo.y + 0.5f, 0.0f);
                devComplex Dp_inter = d00 * (one - a) * (one - b) + d10 * a * (one - b) + d01 * (one - a) * b + d11 * a * b;
                Dp = Dp_inter;
            } else {
                Dp = Dp_near;
            }
            devComplex xl = (image_x - intr.cx) / intr.fx;
            devComplex yl = (image_y - intr.cy) / intr.fy;
            devComplex3 v_c_1 = make_mcomplex3(Dp * xl,
                                               Dp * yl,
                                               Dp);
            devComplex sdf = norm(v_c_1) - norm(v_c);
            if (Dp.real() > 0 && sdf.real() >= -tranc_dist)// meters
            {
                devComplex tsdf;
                tsdf = sdf * tranc_dist_inv;
                if (sdf.real() > tranc_dist)
                    tsdf = devComplex(1.0f, 0.0f);// set constant, imag = 0
                else {
                    tsdf = sdf * tranc_dist_inv;
//                    printf("%d, %d, %d, %f\n", x, y, z, tsdf.real());
                }
                // update global tsdf value
                devComplex tsdf_prev;
                int weight_prev;
                unpack_tsdf(*pos, *weight_pos, *grad_pos, tsdf_prev, weight_prev);
                int Wrk = 1;
                devComplex tsdf_new = (tsdf_prev * __int2float_rn(weight_prev) + __int2float_rn(Wrk) * tsdf) / __int2float_rn(weight_prev + Wrk);
                int weight_new = min(weight_prev + Wrk, max_weight);
                pack_tsdf(tsdf_new, weight_new, *pos, *weight_pos, *grad_pos);
            }
        }
    }
}

void integrateTsdfVolume(const PtrStepSz<ushort> &depth, const Intr &intr,
                         int max_weight, const int3 &volume_resolution,
                         float voxel_size, const MatS33 &Rv2c, const devComplex3 &tv2c, const devComplex3 &tc2v,
                         float tranc_dist, PtrStep<float> value_volume,
                         PtrStep<int> weight_volume,
                         PtrStep<float> grad_volume,
                         DeviceArray2D<float> &depthScaled, int frame_id, float threshold, float k) {
    depthScaled.create(depth.rows, depth.cols);

    dim3 block_scale(32, 8);
    dim3 grid_scale(divUp(depth.cols, block_scale.x),
                    divUp(depth.rows, block_scale.y));
    // scales depth along ray and converts mm -> meters.
    //    scaleDepthToRayKernal<<<grid_scale, block_scale>>>(depth, depthScaled, intr);
    scaleDepthKernal<<<grid_scale, block_scale>>>(depth, depthScaled, intr);

    cudaSafeCall(cudaGetLastError());

    dim3 block(16, 16);
    dim3 grid(divUp(volume_resolution.x, block.x),
              divUp(volume_resolution.y, block.y));

    tsdfFusionKernal<<<grid, block>>>(
            depthScaled, value_volume, weight_volume, grad_volume, volume_resolution,
            tranc_dist, max_weight, Rv2c, tv2c, intr, voxel_size, threshold, k);
    depthScaled.release();
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComputeLocalTsdfHessianKernel(PtrStepSz<float> depthScaled,
                                              int3 volume_resolution, float voxel_size,
                                              MatD33 Rv2c, devDComplex3 tv2c,
                                              float tranc_dist, Intr intr, float threshold, float k,
                                              float *gt_ptr,
                                              float *real_ptr,
                                              float *grad_ptr,
                                              float *hessian_ptr,
                                              int *count_ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= volume_resolution.x || y >= volume_resolution.y)
        return;
    float tranc_dist_inv = 1.0f / tranc_dist;
    for (int z = 0; z < volume_resolution.z; z++) {
        int index = z * volume_resolution.y * volume_resolution.x + y * volume_resolution.x + x;
        devDComplex gt_tsdf(gt_ptr[index], 0, 0, 0);
        if (gt_tsdf.value() == 0 || fabs(gt_tsdf.value()) > 0.95)
            continue;
        devDComplex v_g_x((float(x) + 0.5f) * voxel_size, 0, 0, 0);
        devDComplex v_g_y((float(y) + 0.5f) * voxel_size, 0, 0, 0);
        devDComplex v_g_z((float(z) + 0.5f) * voxel_size, 0, 0, 0);
        devDComplex3 v_g = make_mDcomplex3(v_g_x, v_g_y, v_g_z);
        devDComplex3 v_c = Rv2c * v_g + tv2c;
        devDComplex inv_z = devDComplex(1.0f) / v_c.z;
        if (inv_z.value() < 0)
            continue;
        devDComplex image_x = v_c.x * inv_z * intr.fx + intr.cx;
        devDComplex image_y = v_c.y * inv_z * intr.fy + intr.cy;
        int2 coo = {__float2int_rd(image_x.value() - 0.5f),
                    __float2int_rd(image_y.value() - 0.5f)};
        if (coo.x > 1 && coo.y > 1 && coo.x < depthScaled.cols - 1 &&
            coo.y < depthScaled.rows - 1) {
            int2 coo_near = {__float2int_rn(image_x.value()),
                             __float2int_rn(image_y.value())};
            devDComplex Dp;
            devDComplex Dp_near = devDComplex(depthScaled.ptr(coo_near.y)[coo_near.x]);// meters
            devDComplex d00(depthScaled.ptr(coo.y)[coo.x]);
            devDComplex d10(depthScaled.ptr(coo.y)[coo.x + 1]);
            devDComplex d01(depthScaled.ptr(coo.y + 1)[coo.x]);
            devDComplex d11(depthScaled.ptr(coo.y + 1)[coo.x + 1]);
            float gird_max = fmax(d00.value(), fmax(d01.value(), fmax(d10.value(), d11.value())));
            float gird_min = fmin(d00.value(), fmin(d01.value(), fmin(d10.value(), d11.value())));
            if (d00.value() != 0.0f &&
                d01.value() != 0.0f &&
                d10.value() != 0.0f &&
                d11.value() != 0.0f) {
                devDComplex one(1.0f, 0.0f, 0.0f, 0.0f);
                devDComplex a = image_x - devDComplex(float(coo.x) + 0.5f, 0, 0, 0);
                devDComplex b = image_y - devDComplex(float(coo.y) + 0.5f, 0, 0, 0);
                devDComplex Dp_inter = d00 * (one - a) * (one - b) + d10 * a * (one - b) + d01 * (one - a) * b + d11 * a * b;
                Dp = Dp_inter;
            } else {
                Dp = Dp_near;// meters
            }
            if (Dp.value() > 5 || Dp.value() < 0.2)
                continue;
            devDComplex xl = (image_x - intr.cx) / intr.fx;
            devDComplex yl = (image_y - intr.cy) / intr.fy;
            devDComplex3 v_c_1 = make_mDcomplex3(Dp * xl,
                                                 Dp * yl,
                                                 Dp);
            devDComplex distance = norm(v_c_1) - norm(v_c);
            devDComplex gt_distance = gt_tsdf * tranc_dist;
            devDComplex error = (distance - gt_distance) * tranc_dist_inv;
            // Todo: check the distance threshold
            if (fabs(error.value()) > 1)
                continue;
//            if(x == 147 && y == 86 && z == 191)
//                printf("(%d, %d, %d, %f)\n", x, y, z, error.value());
            devDComplex loss = error * error;
            // update global tsdf value
            real_ptr[index] = loss.value();
            grad_ptr[index] = loss.grad();
            hessian_ptr[index] = loss.hessian();
            count_ptr[index] = 1;
        }
    }
}

// compute the tsdf volume from pose matrix and depth image
float4 ComputeLocalTsdf_hessian(const PtrStepSz<ushort> &depth, const Intr &intr, DeviceArray2D<float> &depthScaled,
                                const int3 &volume_resolution, float voxel_size, const MatD33 &Rv2c, const devDComplex3 &tv2c,
                                float tranc_dist, float threshold, float k,
                                thrustDvec<float> &gt_vec, thrustDvec<float> &real_vec, thrustDvec<float> &grad_vec,
                                thrustDvec<float> &hessian_vec, thrustDvec<int> &count_vec) {
    depthScaled.create(depth.rows, depth.cols);
    float *gt_vec_ptr = trDptr(gt_vec);
    int voxel_num = volume_resolution.x * volume_resolution.y * volume_resolution.z;
    real_vec.resize(voxel_num, 0);
    float *real_vec_ptr = trDptr(real_vec);
    grad_vec.resize(voxel_num, 0);
    float *grad_vec_ptr = trDptr(grad_vec);
    hessian_vec.resize(voxel_num, 0);
    float *hessian_vec_ptr = trDptr(hessian_vec);
    count_vec.resize(voxel_num, 0);
    int *count_vec_ptr = trDptr(count_vec);

    dim3 block_scale(32, 8);
    dim3 grid_scale(divUp(depth.cols, block_scale.x),
                    divUp(depth.rows, block_scale.y));
    // scales depth along ray and converts mm -> meters.
    scaleDepthKernal<<<grid_scale, block_scale>>>(depth, depthScaled, intr);
    cudaSafeCall(cudaGetLastError());
    dim3 block(16, 16);
    dim3 grid(divUp(volume_resolution.x, block.x),
              divUp(volume_resolution.y, block.y));
    ComputeLocalTsdfHessianKernel<<<grid, block>>>(depthScaled, volume_resolution, voxel_size, Rv2c, tv2c,
                                                   tranc_dist, intr, threshold, k, gt_vec_ptr, real_vec_ptr, grad_vec_ptr, hessian_vec_ptr, count_vec_ptr);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
    // reduce vector
    thrust::device_ptr<float> dev_real_ptr(&real_vec[0]);
    float loss_sum = thrust::reduce(dev_real_ptr, dev_real_ptr + voxel_num);
    thrust::device_ptr<float> dev_grad_ptr(&grad_vec[0]);
    float grad_sum = thrust::reduce(dev_grad_ptr, dev_grad_ptr + voxel_num);
    thrust::device_ptr<float> dev_hessian_ptr(&hessian_vec[0]);
    float hessian_sum = thrust::reduce(dev_hessian_ptr, dev_hessian_ptr + voxel_num);
    thrust::device_ptr<int> dev_count_ptr(&count_vec[0]);
    int count_sum = thrust::reduce(dev_count_ptr, dev_count_ptr + voxel_num);
    float4 res;
    res.x = loss_sum;
    res.y = grad_sum;
    res.z = hessian_sum;
    res.w = float(count_sum);
    return res;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComputeLocalTsdfLossKernel(PtrStepSz<float> depthScaled,
                                              int3 volume_resolution, float voxel_size,
                                              Mat33 Rv2c, float3 tv2c,
                                              float tranc_dist, Intr intr, float threshold, float k,
                                              float *gt_ptr,
                                              float *real_ptr,
                                              int *count_ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= volume_resolution.x || y >= volume_resolution.y)
        return;
    float tranc_dist_inv = 1.0f / tranc_dist;
    for (int z = 0; z < volume_resolution.z; z++) {
        int index = z * volume_resolution.y * volume_resolution.x + y * volume_resolution.x + x;
        float gt_tsdf = gt_ptr[index];
        if (gt_tsdf == 0 || fabs(gt_tsdf) > 0.95)
            continue;
        float v_g_x = (float(x) + 0.5f) * voxel_size;
        float v_g_y = (float(y) + 0.5f) * voxel_size;
        float v_g_z = (float(z) + 0.5f) * voxel_size;
        float3 v_g = make_float3(v_g_x, v_g_y, v_g_z);
        float3 v_c = Rv2c * v_g + tv2c;
        float inv_z = 1.0f / v_c.z;
        if (inv_z < 0)
            continue;
        float image_x = v_c.x * inv_z * intr.fx + intr.cx;
        float image_y = v_c.y * inv_z * intr.fy + intr.cy;
        int2 coo = {__float2int_rd(image_x - 0.5f),
                    __float2int_rd(image_y - 0.5f)};
        if (coo.x > 1 && coo.y > 1 && coo.x < depthScaled.cols - 1 &&
            coo.y < depthScaled.rows - 1) {
            int2 coo_near = {__float2int_rn(image_x),
                             __float2int_rn(image_y)};
            float Dp;
            float Dp_near = depthScaled.ptr(coo_near.y)[coo_near.x];// meters
            float d00 = depthScaled.ptr(coo.y)[coo.x];
            float d10 = depthScaled.ptr(coo.y)[coo.x + 1];
            float d01 = depthScaled.ptr(coo.y + 1)[coo.x];
            float d11 = depthScaled.ptr(coo.y + 1)[coo.x + 1];
            float gird_max = fmax(d00, fmax(d01, fmax(d10, d11)));
            float gird_min = fmin(d00, fmin(d01, fmin(d10, d11)));
            if (d00 != 0.0f &&
                d01 != 0.0f &&
                d10 != 0.0f &&
                d11 != 0.0f) {
                float one = 1.0f;
                float a = image_x - (float(coo.x) + 0.5f);
                float b = image_y - (float(coo.y) + 0.5f);
                float Dp_inter = d00 * (one - a) * (one - b) + d10 * a * (one - b) + d01 * (one - a) * b + d11 * a * b;
                Dp = Dp_inter;
            } else {
                Dp = Dp_near;// meters
            }
            if (Dp > 5 || Dp < 0.2)
                continue;
            float xl = (image_x - intr.cx) / intr.fx;
            float yl = (image_y - intr.cy) / intr.fy;
            float3 v_c_1 = make_float3(Dp * xl,
                                                 Dp * yl,
                                                 Dp);
            float distance = norm(v_c_1) - norm(v_c);
            float gt_distance = gt_tsdf * tranc_dist;
            float error = (distance - gt_distance) * tranc_dist_inv;
            // Todo: check the distance threshold
            if (fabs(error) > 1)
                continue;
//            if(x == 147 && y == 86 && z == 191)
//                printf("(%d, %d, %d, %f)\n", x, y, z, error);
            float loss = error * error;
            // update global tsdf value
            real_ptr[index] = loss;
            count_ptr[index] = 1;
        }
    }
}

float2 ComputeLocalTsdf_loss(const PtrStepSz<ushort> &depth, const Intr &intr, DeviceArray2D<float> &depthScaled,
                             const int3 &volume_resolution, float voxel_size,
                             const Mat33 &Rv2c, const float3 &tv2c,
                             float tranc_dist, float threshold, float k,
                             thrustDvec<float> &gt_vec, thrustDvec<float> &real_vec, thrustDvec<int> &count_vec) {
    depthScaled.create(depth.rows, depth.cols);
    float *gt_vec_ptr = trDptr(gt_vec);
    int voxel_num = volume_resolution.x * volume_resolution.y * volume_resolution.z;
    real_vec.resize(voxel_num, 0);
    float *real_vec_ptr = trDptr(real_vec);
    count_vec.resize(voxel_num, 0);
    int *count_vec_ptr = trDptr(count_vec);

    dim3 block_scale(32, 8);
    dim3 grid_scale(divUp(depth.cols, block_scale.x),
                    divUp(depth.rows, block_scale.y));
    // scales depth along ray and converts mm -> meters.
    scaleDepthKernal<<<grid_scale, block_scale>>>(depth, depthScaled, intr);
    cudaSafeCall(cudaGetLastError());
    dim3 block(16, 16);
    dim3 grid(divUp(volume_resolution.x, block.x),
              divUp(volume_resolution.y, block.y));
    ComputeLocalTsdfLossKernel<<<grid, block>>>(depthScaled, volume_resolution, voxel_size, Rv2c, tv2c,
                                                tranc_dist, intr, threshold, k, gt_vec_ptr, real_vec_ptr, count_vec_ptr);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
    // reduce vector
    thrust::device_ptr<float> dev_real_ptr(&real_vec[0]);
    float loss_sum = thrust::reduce(dev_real_ptr, dev_real_ptr + voxel_num);
    thrust::device_ptr<int> dev_count_ptr(&count_vec[0]);
    int count_sum = thrust::reduce(dev_count_ptr, dev_count_ptr + voxel_num);
    float2 res;
    res.x = loss_sum;
    res.y = float(count_sum);
    return res;
}