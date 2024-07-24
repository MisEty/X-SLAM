#include "Internal.h"
#include "cx.h"

const float sigma_color = 30; // in mm
const float sigma_space = 4.5;// in pixels

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void computeVmapKernel(const PtrStepSz<devComplex> depth,
                                  PtrStepSz<devComplex> vmap, float fx_inv,
                                  float fy_inv, float cx, float cy) {
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if (u < depth.cols && v < depth.rows) {
        devComplex z = depth.ptr(v)[u];
        z /= 1000.f;

        if (z.real() != 0) {
            devComplex vx = z * (float(u) - cx) * fx_inv;
            devComplex vy = z * (float(v) - cy) * fy_inv;
            devComplex vz = z;

            vmap.ptr(v)[u] = devComplex(vx.real(), vx.imag());
            vmap.ptr(v + depth.rows)[u] = devComplex(vy.real(), vy.imag());
            vmap.ptr(v + depth.rows * 2)[u] = devComplex(vz.real(), vz.imag());
        } else
            vmap.ptr(v)[u] = devComplex(cx::numeric_limits<float>::quiet_NaN(), 0);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void computeNmapKernel(int rows, int cols,
                                  const PtrStep<devComplex> vmap,
                                  PtrStep<devComplex> nmap) {
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if (u >= cols || v >= rows)
        return;

    if (u == cols - 1 || v == rows - 1) {
        nmap.ptr(v)[u] = cx::numeric_limits<float>::quiet_NaN();
        return;
    }

    devComplex3 v00, v01, v10;
    v00.x = vmap.ptr(v)[u];
    v01.x = vmap.ptr(v)[u + 1];
    v10.x = vmap.ptr(v + 1)[u];

    if (!isnan(v00.x.real()) && !isnan(v01.x.real()) && !isnan(v10.x.real())) {
        v00.y = vmap.ptr(v + rows)[u];
        v01.y = vmap.ptr(v + rows)[u + 1];
        v10.y = vmap.ptr(v + 1 + rows)[u];

        v00.z = vmap.ptr(v + 2 * rows)[u];
        v01.z = vmap.ptr(v + 2 * rows)[u + 1];
        v10.z = vmap.ptr(v + 1 + 2 * rows)[u];

        devComplex3 r = normalized(cross(v01 - v00, v10 - v00));
        //        if(v == 5 && u == 608)
        //            printf("(%f, %f, %f) (%f, %f, %f1)\n", v01.x.real(), v01.y.real(), v01.z.real(),
        //                   r.x.real(), r.y.real(), r.z.real());

        nmap.ptr(v)[u] = r.x;
        nmap.ptr(v + rows)[u] = r.y;
        nmap.ptr(v + 2 * rows)[u] = r.z;
    } else
        nmap.ptr(v)[u] = cx::numeric_limits<float>::quiet_NaN();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void createVMap(const Intr &intr, const MapArr &depth, MapArr &vmap) {
    vmap.create(depth.rows() * 3, depth.cols());

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = divUp(depth.cols(), block.x);
    grid.y = divUp(depth.rows(), block.y);

    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    computeVmapKernel<<<grid, block>>>(depth, vmap, 1.f / fx, 1.f / fy, cx, cy);
    cudaSafeCall(cudaGetLastError());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void createNMap(const MapArr &vmap, MapArr &nmap) {
    nmap.create(vmap.rows(), vmap.cols());

    int rows = vmap.rows() / 3;
    int cols = vmap.cols();

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = divUp(cols, block.x);
    grid.y = divUp(rows, block.y);

    computeNmapKernel<<<grid, block>>>(rows, cols, vmap, nmap);
    cudaSafeCall(cudaGetLastError());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<bool normalize>
__global__ void resizeMapKernel(int drows, int dcols, int srows,
                                const PtrStep<devComplex> input,
                                PtrStep<devComplex> output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dcols || y >= drows)
        return;

    const float qnan = cx::numeric_limits<float>::quiet_NaN();

    int xs = x * 2;
    int ys = y * 2;

    devComplex x00 = input.ptr(ys + 0)[xs + 0];
    devComplex x01 = input.ptr(ys + 0)[xs + 1];
    devComplex x10 = input.ptr(ys + 1)[xs + 0];
    devComplex x11 = input.ptr(ys + 1)[xs + 1];
    if (isnan(x00.real()) || isnan(x01.real()) || isnan(x10.real()) ||
        isnan(x11.real())) {
        output.ptr(y)[x] = qnan;
        return;
    } else {
        devComplex3 n;

        n.x = (x00 + x01 + x10 + x11) / 4.0f;

        devComplex y00 = input.ptr(ys + srows + 0)[xs + 0];
        devComplex y01 = input.ptr(ys + srows + 0)[xs + 1];
        devComplex y10 = input.ptr(ys + srows + 1)[xs + 0];
        devComplex y11 = input.ptr(ys + srows + 1)[xs + 1];

        n.y = (y00 + y01 + y10 + y11) / 4.0f;

        devComplex z00 = input.ptr(ys + 2 * srows + 0)[xs + 0];
        devComplex z01 = input.ptr(ys + 2 * srows + 0)[xs + 1];
        devComplex z10 = input.ptr(ys + 2 * srows + 1)[xs + 0];
        devComplex z11 = input.ptr(ys + 2 * srows + 1)[xs + 1];

        n.z = (z00 + z01 + z10 + z11) / 4.0f;
        if (normalize)
            n = normalized(n);
        output.ptr(y)[x] = n.x;
        output.ptr(y + drows)[x] = n.y;
        output.ptr(y + 2 * drows)[x] = n.z;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void bilateralKernel(const PtrStepSz<ushort> src,
                                PtrStep<devComplex> dst,
                                float sigma_space2_inv_half,
                                float sigma_color2_inv_half) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= src.cols || y >= src.rows)
        return;
    int value = src.ptr(y)[x];

    //    devComplex res(__int2float_rd(value), 0);
    //
    //    dst.ptr(y)[x] = res;
    const int R = 6;// static_cast<int>(sigma_space * 1.5);
    const int D = R * 2 + 1;

    int tx = min(x - D / 2 + D, src.cols - 1);
    int ty = min(y - D / 2 + D, src.rows - 1);

    float sum1 = 0;
    float sum2 = 0;

    for (int cy = max(y - D / 2, 0); cy < ty; ++cy) {
        for (int cx = max(x - D / 2, 0); cx < tx; ++cx) {
            int tmp = src.ptr(cy)[cx];

            float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            float color2 = (value - tmp) * (value - tmp);

            float weight = __expf(
                    -(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

            sum1 += tmp * weight;
            sum2 += weight;
        }
    }
    int round = __float2int_rn(sum1 / sum2);
    if (round > 5000 || round < 200)
        round = 0;
    round = max(0, min(round, cx::numeric_limits<short>::max()));
    devComplex res(__int2float_rd(round), 0);

    dst.ptr(y)[x] = res;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void pyrDownKernel(const PtrStepSz<devComplex> src,
                              PtrStepSz<devComplex> dst, float sigma_color) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;
    int center = __float2int_rn(src.ptr(2 * y)[2 * x].real());

    int tx = min(2 * x - D / 2 + D, src.cols - 1);
    int ty = min(2 * y - D / 2 + D, src.rows - 1);
    int cy = max(0, 2 * y - D / 2);

    int sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
        for (int cx = max(0, 2 * x - D / 2); cx < tx; ++cx) {
            int val = __float2int_rn(src.ptr(cy)[cx].real());
            if (abs(val - center) < 3 * sigma_color) {
                sum += val;
                ++count;
            }
        }
    float res = __int2float_rd(sum / count);
    dst.ptr(y)[x] = devComplex(res, 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<bool normalize>
void resizeMap(const MapArr &input, MapArr &output) {
    int in_cols = input.cols();
    int in_rows = input.rows() / 3;

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    output.create(out_rows * 3, out_cols);

    dim3 block(32, 8);
    dim3 grid(divUp(out_cols, block.x), divUp(out_rows, block.y));
    resizeMapKernel<normalize>
            <<<grid, block>>>(out_rows, out_cols, in_rows, input, output);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void resizeVMap(const MapArr &input, MapArr &output) {
    resizeMap<false>(input, output);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void resizeNMap(const MapArr &input, MapArr &output) {
    resizeMap<true>(input, output);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void bilateralFilter(const DeviceArray2D<ushort> &src, MapArr &dst) {
    dim3 block(32, 8);
    dim3 grid(divUp(src.cols(), block.x), divUp(src.rows(), block.y));

    cudaFuncSetCacheConfig(bilateralKernel, cudaFuncCachePreferL1);
    bilateralKernel<<<grid, block>>>(src, dst, 0.5f / (sigma_space * sigma_space),
                                     0.5f / (sigma_color * sigma_color));

    cudaSafeCall(cudaGetLastError());
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void pyrDown(const MapArr &src, MapArr &dst) {
    dst.create(src.rows() / 2, src.cols() / 2);

    dim3 block(32, 8);
    dim3 grid(divUp(dst.cols(), block.x), divUp(dst.rows(), block.y));

    // pyrDownGaussKernel<<<grid, block>>>(src, dst, sigma_color);
    pyrDownKernel<<<grid, block>>>(src, dst, sigma_color);
    cudaSafeCall(cudaGetLastError());
};