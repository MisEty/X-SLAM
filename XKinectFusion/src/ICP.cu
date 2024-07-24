#include "ICP.h"
#include "cx.h"


template<int CTA_SIZE_, typename T>
static __device__ __forceinline__ void reduce_c(volatile T *buffer, volatile T *grad_buffer) {
    int tid = cx::Block::flattenedThreadId();
    T val = buffer[tid];
    T grad_val = grad_buffer[tid];

    if (CTA_SIZE_ >= 1024) {
        if (tid < 512) {
            buffer[tid] = val = val + buffer[tid + 512];
            grad_buffer[tid] = grad_val = grad_val + grad_buffer[tid + 512];
        }
        __syncthreads();
    }
    if (CTA_SIZE_ >= 512) {
        if (tid < 256) {
            buffer[tid] = val = val + buffer[tid + 256];
            grad_buffer[tid] = grad_val = grad_val + grad_buffer[tid + 256];
        }
        __syncthreads();
    }
    if (CTA_SIZE_ >= 256) {
        if (tid < 128) {
            buffer[tid] = val = val + buffer[tid + 128];
            grad_buffer[tid] = grad_val = grad_val + grad_buffer[tid + 128];
        }
        __syncthreads();
    }
    if (CTA_SIZE_ >= 128) {
        if (tid < 64) {
            buffer[tid] = val = val + buffer[tid + 64];
            grad_buffer[tid] = grad_val = grad_val + grad_buffer[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        if (CTA_SIZE_ >= 64) {
            buffer[tid] = val = val + buffer[tid + 32];
            grad_buffer[tid] = grad_val = grad_val + grad_buffer[tid + 32];
        }
        if (CTA_SIZE_ >= 32) {
            buffer[tid] = val = val + buffer[tid + 16];
            grad_buffer[tid] = grad_val = grad_val + grad_buffer[tid + 16];
        }
        if (CTA_SIZE_ >= 16) {
            buffer[tid] = val = val + buffer[tid + 8];
            grad_buffer[tid] = grad_val = grad_val + grad_buffer[tid + 8];
        }
        if (CTA_SIZE_ >= 8) {
            buffer[tid] = val = val + buffer[tid + 4];
            grad_buffer[tid] = grad_val = grad_val + grad_buffer[tid + 4];
        }
        if (CTA_SIZE_ >= 4) {
            buffer[tid] = val = val + buffer[tid + 2];
            grad_buffer[tid] = grad_val = grad_val + grad_buffer[tid + 2];
        }
        if (CTA_SIZE_ >= 2) {
            buffer[tid] = val = val + buffer[tid + 1];
            grad_buffer[tid] = grad_val = grad_val + grad_buffer[tid + 1];
        }
    }
}

template<int CTA_SIZE_, typename T>
static __device__ __forceinline__ void reduce(volatile T *buffer) {
    int tid = cx::Block::flattenedThreadId();
    T val = buffer[tid];

    if (CTA_SIZE_ >= 1024) {
        if (tid < 512) {
            buffer[tid] = val = val + buffer[tid + 512];
        }
        __syncthreads();
    }
    if (CTA_SIZE_ >= 512) {
        if (tid < 256) {
            buffer[tid] = val = val + buffer[tid + 256];
        }
        __syncthreads();
    }
    if (CTA_SIZE_ >= 256) {
        if (tid < 128) {
            buffer[tid] = val = val + buffer[tid + 128];
        }
        __syncthreads();
    }
    if (CTA_SIZE_ >= 128) {
        if (tid < 64) {
            buffer[tid] = val = val + buffer[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        if (CTA_SIZE_ >= 64) {
            buffer[tid] = val = val + buffer[tid + 32];
        }
        if (CTA_SIZE_ >= 32) {
            buffer[tid] = val = val + buffer[tid + 16];
        }
        if (CTA_SIZE_ >= 16) {
            buffer[tid] = val = val + buffer[tid + 8];
        }
        if (CTA_SIZE_ >= 8) {
            buffer[tid] = val = val + buffer[tid + 4];
        }
        if (CTA_SIZE_ >= 4) {
            buffer[tid] = val = val + buffer[tid + 2];
        }
        if (CTA_SIZE_ >= 2) {
            buffer[tid] = val = val + buffer[tid + 1];
        }
    }
}

struct TranformReduction {
    enum {
        CTA_SIZE = 512,
        STRIDE = CTA_SIZE,

        B = 6,
        COLS = 6,
        ROWS = 6,
        DIAG = 6,
        UPPER_DIAG_MAT = (COLS * ROWS - DIAG) / 2 + DIAG,
        TOTAL = UPPER_DIAG_MAT + B,

        GRID_X = TOTAL
    };

    PtrStep<devComplexICP> gbuf;
    int length;
    mutable devComplexICP *output;
    __device__ __forceinline__ void operator()() const {
        const devComplexICP *beg = gbuf.ptr(blockIdx.x);
        const devComplexICP *end = beg + length;

        int tid = threadIdx.x;

        devComplexICP sum = devComplexICP(0, 0);
        for (const devComplexICP *t = beg + tid; t < end; t += STRIDE)
            sum += *t;

        __shared__ floatTypeICP smem[CTA_SIZE];
        __shared__ floatTypeICP smem_grad[CTA_SIZE];

        smem[tid] = sum.real();
        smem_grad[tid] = sum.imag();

        __syncthreads();

        reduce_c<CTA_SIZE>(smem, smem_grad);

        if (tid == 0)
            output[blockIdx.x] = devComplexICP(smem[0], smem_grad[0]);
    }
};


__global__ void TransformEstimatorKernel(const TranformReduction tr) { tr(); }

struct Combined {
    enum { JACOBI_SIZE = 12,
           CTA_SIZE_X = 32,
           CTA_SIZE_Y = 8,
           CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y };

    MatS33 Rcurr;
    devComplex3 tcurr;

    PtrStep<devComplex> vmap_curr;
    PtrStep<devComplex> nmap_curr;

    MatS33 Rprev_inv;
    devComplex3 tprev;

    Intr intr;

    PtrStep<devComplex> vmap_g_prev;
    PtrStep<devComplex> nmap_g_prev;

    float distThres;
    float angleThres;

    int cols;
    int rows;

    mutable PtrStep<devComplexICP> gbuf;
    mutable PtrStep<float> jacobi_buf;
    mutable PtrStep<float> hessian_buf[JACOBI_SIZE];

    __device__ __forceinline__ bool search_newton(int x, int y, devComplex3 &n_prev_g, devComplex3 &p_prev_g,
                                                  devComplex3 &p_curr_g, devComplex3 &p_curr_l) const {
        //        if(isnan(vmap_g_prev.ptr(y)[x].real()) || isnan(nmap_g_prev.ptr(y)[x].real()))
        //            return false;
        //        return true;
        devComplex3 ncurr;
        ncurr.x = nmap_curr.ptr(y)[x];
        if (isnan(ncurr.x.real()))
            return false;
        ncurr.y = nmap_curr.ptr(y + rows)[x];
        ncurr.z = nmap_curr.ptr(y + 2 * rows)[x];

        devComplex3 vcurr;
        vcurr.x = vmap_curr.ptr(y)[x];
        vcurr.y = vmap_curr.ptr(y + rows)[x];
        vcurr.z = vmap_curr.ptr(y + 2 * rows)[x];
        devComplex3 vcurr_g = Rcurr * vcurr + tcurr;
        devComplex3 vcurr_cp_complex = Rprev_inv * (vcurr_g - tprev);// prev camera imag space
        float3 vcurr_cp = real(vcurr_cp_complex);
        int2 ukr;                                                           // projection
        ukr.x = __float2int_rn(vcurr_cp.x * intr.fx / vcurr_cp.z + intr.cx);// 4
        ukr.y = __float2int_rn(vcurr_cp.y * intr.fy / vcurr_cp.z + intr.cy);// 4
        if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows ||
            vcurr_cp.z < 0)
            return false;
        devComplex3 nprev_g;
        nprev_g.x = nmap_g_prev.ptr(ukr.y)[ukr.x];
        if (isnan(nprev_g.x.real()))
            return false;
        nprev_g.y = nmap_g_prev.ptr(ukr.y + rows)[ukr.x];
        nprev_g.z = nmap_g_prev.ptr(ukr.y + 2 * rows)[ukr.x];

        devComplex3 vprev_g;
        vprev_g.x = vmap_g_prev.ptr(ukr.y)[ukr.x];
        vprev_g.y = vmap_g_prev.ptr(ukr.y + rows)[ukr.x];
        vprev_g.z = vmap_g_prev.ptr(ukr.y + 2 * rows)[ukr.x];
        devComplex dist = norm(vprev_g - vcurr_g);
        if (dist.real() > distThres)
            return false;
        devComplex3 ncurr_g = Rcurr * ncurr;
        devComplex sine = norm(cross(ncurr_g, nprev_g));
        if (sine.real() >= angleThres)
            return false;
        n_prev_g = nprev_g;
        p_prev_g = vprev_g;
        p_curr_g = vcurr_g;
        p_curr_l = vcurr;
        return true;
    }

    __device__ __forceinline__ void operator()() const {
        int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

        devComplex3 n, d, s, temp;
        bool found_coresp = false;
        if (x < cols && y < rows)
            found_coresp = search_newton(x, y, n, d, s, temp);
        devComplex row[7];

        if (found_coresp) {
            *(devComplex3 *) &row[0] = cross(s, n);
            *(devComplex3 *) &row[3] = n;
            row[6] = dot(n, d - s);
        } else
            row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = devComplex(0.0f, 0.0f);
        __shared__ floatTypeICP smem[CTA_SIZE];
        __shared__ floatTypeICP smem_grad[CTA_SIZE];
        int tid = cx::Block::flattenedThreadId();

        int shift = 0;
        for (int i = 0; i < 6; ++i)// rows
        {
#pragma unroll
            for (int j = i; j < 7; ++j)// cols + b
            {
                __syncthreads();
                smem[tid] = (row[i] * row[j]).real();
                smem_grad[tid] = (row[i] * row[j]).imag();
                __syncthreads();
                reduce_c<CTA_SIZE>(smem, smem_grad);
                if (tid == 0)
                    gbuf.ptr(shift++)[blockIdx.x + gridDim.x * blockIdx.y] = devComplexICP(smem[0], smem_grad[0]);
            }
        }
    }

    __device__ __forceinline__ void computeOptimizeMatrix() const {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        devComplex3 n_prev_g, p_prev_g, p_curr_g, p_curr_l;
        Eigen::Vector4f n1, p1, p0_trans, p0;
        bool found_coresp = false;
        if (x < cols && y < rows) {
            found_coresp = search_newton(x, y, n_prev_g, p_prev_g, p_curr_g, p_curr_l);
        }
        if (found_coresp) {
            n1 = Eigen::Vector4f(n_prev_g.x.real(), n_prev_g.y.real(), n_prev_g.z.real(), 1);
            p1 = Eigen::Vector4f(p_prev_g.x.real(), p_prev_g.y.real(), p_prev_g.z.real(), 1);
            p0_trans = Eigen::Vector4f(p_curr_g.x.real(), p_curr_g.y.real(), p_curr_g.z.real(), 1);
            p0 = Eigen::Vector4f(p_curr_l.x.real(), p_curr_l.y.real(), p_curr_l.z.real(), 1);
        } else {
            n1.setZero();
            p1.setZero();
            p0_trans.setZero();
            p0.setZero();
        }
        __shared__ float smem_jacobi[CTA_SIZE];
        int tid = cx::Block::flattenedThreadId();

        int shift = 0;
        for (int i = 0; i < 3; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                __syncthreads();
                float proj_norm = (p0_trans - p1).x() * n1.x() +
                                  (p0_trans - p1).y() * n1.y() +
                                  (p0_trans - p1).z() * n1.z();
                smem_jacobi[tid] = 2 * n1[i] * proj_norm * p0[j];
                __syncthreads();
                reduce<CTA_SIZE>(smem_jacobi);
                if (tid == 0) {
//                    if (i == 0 && j == 0)
//                        printf("(%d, %d, %f)\n", x, y, smem_jacobi[0]);
                    jacobi_buf.ptr(shift)[blockIdx.x + gridDim.x * blockIdx.y] = smem_jacobi[0];
                    shift++;
                }
            }
        }

        __shared__ float smem_hessian[CTA_SIZE];
        for (int i = 0; i < JACOBI_SIZE; i++) {
#pragma unroll
            for (int j = i; j < JACOBI_SIZE; j++) {
                __syncthreads();
                int i1 = i / 4;
                int j1 = i % 4;
                int i2 = j / 4;
                int j2 = j % 4;
                float jacobi_x_pose_i1j1 = (i1 == 0) ? p0[j1] : 0;
                float jacobi_x_pose_i2j2 = n1[0] * n1[i2] * p0[j2];
                float jacobi_y_pose_i1j1 = (i1 == 1) ? p0[j1] : 0;
                float jacobi_y_pose_i2j2 = n1[1] * n1[i2] * p0[j2];
                float jacobi_z_pose_i1j1 = (i1 == 2) ? p0[j1] : 0;
                float jacobi_z_pose_i2j2 = n1[2] * n1[i2] * p0[j2];
                float value = 2 * (jacobi_x_pose_i1j1 * jacobi_x_pose_i2j2 +
                                   jacobi_y_pose_i1j1 * jacobi_y_pose_i2j2 +
                                   jacobi_z_pose_i1j1 * jacobi_z_pose_i2j2);
                smem_hessian[tid] = value;
                __syncthreads();
                reduce<CTA_SIZE>(smem_hessian);
                if (tid == 0) {
                    hessian_buf[i].ptr(j)[blockIdx.x + gridDim.x * blockIdx.y] = smem_hessian[0];
                    hessian_buf[j].ptr(i)[blockIdx.x + gridDim.x * blockIdx.y] = smem_hessian[0];
                }
            }
        }
    }
};

__global__ void combinedKernel(Combined cs) { cs(); }

__global__ void computeOptimizeMatrixKernel(Combined cs) {
    cs.computeOptimizeMatrix();
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void estimateCombined(const MatS33 &Rcurr, const devComplex3 &tcurr,
                      const MapArr &vmap_curr, const MapArr &nmap_curr,
                      const MatS33 &Rprev_inv, const devComplex3 &tprev,
                      const Intr &intr, const MapArr &vmap_g_prev,
                      const MapArr &nmap_g_prev, float distThres,
                      float angleThres, DeviceArray2D<devComplexICP> &gbuf,
                      DeviceArray<devComplexICP> &mbuf, hostComplexICP *matrixA_host,
                      hostComplexICP *vectorB_host) {
    int cols = vmap_curr.cols();
    int rows = vmap_curr.rows() / 3;

    Combined cs;

    cs.Rcurr = Rcurr;
    cs.tcurr = tcurr;
    cs.vmap_curr = vmap_curr;
    cs.nmap_curr = nmap_curr;
    cs.Rprev_inv = Rprev_inv;
    cs.tprev = tprev;
    cs.intr = intr;
    cs.vmap_g_prev = vmap_g_prev;
    cs.nmap_g_prev = nmap_g_prev;
    cs.distThres = distThres;
    cs.angleThres = angleThres;

    cs.cols = cols;
    cs.rows = rows;

    //////////////////////////////

    dim3 block(Combined::CTA_SIZE_X, Combined::CTA_SIZE_Y);
    dim3 grid(1, 1, 1);
    grid.x = divUp(cols, block.x);
    grid.y = divUp(rows, block.y);

    mbuf.create(TranformReduction::TOTAL);
    if (gbuf.rows() != TranformReduction::TOTAL || gbuf.cols() < (int) (grid.x * grid.y))
        gbuf.create(TranformReduction::TOTAL, grid.x * grid.y);
    cs.gbuf = gbuf;
    combinedKernel<<<grid, block>>>(cs);
    cudaSafeCall(cudaGetLastError());
    TranformReduction tr;
    tr.gbuf = gbuf;
    tr.length = grid.x * grid.y;
    tr.output = mbuf;

    TransformEstimatorKernel<<<TranformReduction::TOTAL,
                               TranformReduction::CTA_SIZE>>>(tr);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    devComplexICP host_data[TranformReduction::TOTAL];
    mbuf.download(host_data);

    int shift = 0;
    for (int i = 0; i < 6; ++i)    // rows
        for (int j = i; j < 7; ++j)// cols + b
        {
            devComplexICP value = host_data[shift++];
            if (j == 6)// vector b
                vectorB_host[i] = hostComplexICP(value.real(), value.imag());
            else
                matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = hostComplexICP(value.real(), value.imag());
        }
}

void computeOptimizeMatrix(const MapArr &vmap_curr, const MapArr &nmap_curr,
                           const MapArr &vmap_g_prev, const MapArr &nmap_g_prev,
                           const MatS33 &Rcurr, const devComplex3 &tcurr,
                           const MatS33 &Rprev_inv, const devComplex3 &tprev,
                           const Intr &intr, float distThres, float angleThres,
                           DeviceArray2D<float> &jacobi_buf, Eigen::Matrix4f &jacobi_host,
                           DeviceArray2D<float> *hessian_buf, Eigen::Matrix4f **hessian_host) {
    int cols = vmap_curr.cols();
    int rows = vmap_curr.rows() / 3;
    Combined cs;
    cs.Rcurr = Rcurr;
    cs.tcurr = tcurr;
    cs.vmap_curr = vmap_curr;
    cs.nmap_curr = nmap_curr;
    cs.Rprev_inv = Rprev_inv;
    cs.tprev = tprev;
    cs.intr = intr;
    cs.vmap_g_prev = vmap_g_prev;
    cs.nmap_g_prev = nmap_g_prev;
    cs.distThres = distThres;
    cs.angleThres = angleThres;
    cs.cols = cols;
    cs.rows = rows;

    //////////////////////////////
    dim3 block(Combined::CTA_SIZE_X, Combined::CTA_SIZE_Y);
    dim3 grid(1, 1, 1);
    grid.x = divUp(cols, block.x);
    grid.y = divUp(rows, block.y);
    if (jacobi_buf.rows() != Combined::JACOBI_SIZE || jacobi_buf.cols() < (int) (grid.x * grid.y))
        jacobi_buf.create(Combined::JACOBI_SIZE, grid.x * grid.y);
    cs.jacobi_buf = jacobi_buf;

    for (int i = 0; i < 12; i++) {
        if (hessian_buf[i].rows() != Combined::JACOBI_SIZE || hessian_buf[i].cols() < (int) (grid.x * grid.y))
            hessian_buf[i].create(Combined::JACOBI_SIZE, grid.x * grid.y);
        cs.hessian_buf[i] = hessian_buf[i];
    }

    computeOptimizeMatrixKernel<<<grid, block>>>(cs);
    cudaSafeCall(cudaGetLastError());
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            int idx = i * 4 + j;
            thrust::device_ptr<float> dev_ptr(cs.jacobi_buf.ptr(idx));
            jacobi_host(i, j) = thrust::reduce(dev_ptr, dev_ptr + grid.x * grid.y);
        }
    }
    for (int i = 0; i < 12; i++) {
        for (int j = i; j < 12; j++) {
            int i1 = i / 4;
            int j1 = i % 4;
            int i2 = j / 4;
            int j2 = j % 4;
            thrust::device_ptr<float> dev_ptr(cs.hessian_buf[i].ptr(j));
            hessian_host[i1][j1](i2, j2) = thrust::reduce(dev_ptr, dev_ptr + grid.x * grid.y);
            hessian_host[i2][j2](i1, j1) = hessian_host[i1][j1](i2, j2);
        }
    }
}