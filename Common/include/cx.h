// file cx.h
#pragma once
// Copyright Richard Ansorge 2020.
// Part of cx utility suite devevoped for CUP Publication:
//         Programming in Parallel with CUDA
//
// cx stands for CUDA Examples, these are bits of code used by the examples
// Comments are welcome please feel free to send emails to me.
// Richard Ansorge, rea1@cam.ac.uk
// version 2.00 June 2020

// these for visual studio
#pragma warning(disable : 4244)// vebose thrust warnings
#pragma warning(disable : 4267)
#pragma warning(disable : 4996)// warnings: unsafe functions e.g fread
#pragma warning(disable : 4838)// warnings: size_t to int

// macro #defines for min & max are usually bad news,
// the native CUDA versions compile to single instuctions hence very fast.
#undef min
#undef max

// cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/system/cuda/experimental/pinned_allocator.h"

// C++ includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _USE_MATH_DEFINES

#include <algorithm>
#include <float.h>// for _controlfp_s(nullptr,_DN_FLUSH,_MCW_DN);
#include <math.h>

// for opencv etc (ASCII escape)
#define ESC 27

// these mimic the cuda analogs for vectors, e.g uchar4
using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
using ullong = unsigned long long;
using llong = long long;

// const versions of above
using cuchar = const unsigned char;
using cushort = const unsigned short;
using cuint = const unsigned int;
using culong = const unsigned long;
using cullong = const unsigned long long;

// const versions of native types
using cchar = const char;
using cshort = const short;
using cint = const int;
using cfloat = const float;
using clong = const long;
using cdouble = const double;
using cllong = const long long;

// for CUDA (incomplete add as necessary)
using cint3 = const int3;
using cfloat3 = const float3;


// These to reduce verbosity in pointer function arguments.
template<typename T>
using r_Ptr = T *__restrict__;// pointer variable data variable
template<typename T>
using cr_Ptr = const T *__restrict__;// pointer variable data constant
template<typename T>
using cvr_Ptr = T *const __restrict__;// pointer constant data variable
template<typename T>
using ccr_Ptr = const T *const __restrict__;// pointer constant data constant
// thrust vectors
template<typename T>
using thrustHvecPin =
        thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>;
template<typename T>
using thrustHvec = thrust::host_vector<T>;
template<typename T>
using thrustDvec = thrust::device_vector<T>;

// get pointer to thrust device array
template<typename T>
T *trDptr(thrustDvec<T> &a) { return a.data().get(); }

// the rest of the header lives in cx namespace
// std::atexit([]{cudaDeviceReset();})
namespace cx {
    // fancy definition of pi, anticipate float or double for type T !
    template<typename T = float>
    constexpr T pi = (T) (3.1415926535897932385L);
    template<typename T = float>
    constexpr T pi2 = (T) (2.0L * pi<T>);
    template<typename T = float>
    constexpr T piby2 = (T) (0.5L * pi<T>);

    inline const char *tail(cchar *s, char c)// strip path from file name
    {
        const char *pch = strrchr(s, c);
        return (pch != nullptr) ? pch + 1 : s;
    }

    // Based on  NVIDIA checkCudaErrors from helper_cuda.h but optional return not exit
    inline int codeCheck(cudaError_t code, cchar *file, cint line, cchar *call) {
        if (code != cudaSuccess) {
            fprintf(stderr, "cx::ok error: %s at %s:%d %s \n", cudaGetErrorString(code), tail(file, '/'), line, call);
            exit(1);// NB this to quit on error (no tidy ups)
            //return 1; // or this to continue on error (user check on return code advised)
        }
        return 0;
    }

// Common functions for cuda programming
#define cudaSafeCall cx::___cudaSafeCall
    static inline void ___cudaSafeCall(cudaError_t err, char *msg = nullptr) {
        if (cudaSuccess != err) {
            printf("CUDA error(%s): %s\n", msg, cudaGetErrorString(err));
            exit(-1);
        }
    }
    static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }

#ifdef __CUDACC__
    // device only function smallest power of 2 >= n
    __inline__ __device__ int pow2ceil(int n) {
        int pow2 = 1 << (31 - __clz(n));
        if (n > pow2) pow2 = (pow2 << 1);
        return pow2;
    }
    // device only function greatest power of 2 <= n
    __inline__ __device__ int pow2floor(int n) {
        int pow2 = 1 << (31 - __clz(n));
        return pow2;
    }
    // CUDA does not provide a swap intrinsic ?
    template<typename T>
    __inline__ __host__ __device__ void swap(T &a, T &b) {
        T temp = a;
        a = b;
        b = temp;
    }

    // numeric_limits on cuda
    template<typename T>
    struct numeric_limits;
    template<>
    struct numeric_limits<float> {
        __device__ __forceinline__ static float
        quiet_NaN() { return __int_as_float(0x7fffffff); /*CUDART_NAN_F*/ };
        __device__ __forceinline__ static float
        epsilon() { return 1.192092896e-07f /*FLT_EPSILON*/; };

        __device__ __forceinline__ static float
        min() { return 1.175494351e-38f /*FLT_MIN*/; };
        __device__ __forceinline__ static float
        max() { return 3.402823466e+38f /*FLT_MAX*/; };
    };

    template<>
    struct numeric_limits<short> {
        __device__ __forceinline__ static short
        max() { return SHRT_MAX; };
    };

    // cuda data struct
    struct Block {
        static __device__ __forceinline__ unsigned int stride() {
            return blockDim.x * blockDim.y * blockDim.z;
        }

        static __device__ __forceinline__ int
        flattenedThreadId() {
            return threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        }

        template<int CTA_SIZE, typename T, class BinOp>
        static __device__ __forceinline__ void reduce(volatile T *buffer, BinOp op) {
            int tid = flattenedThreadId();
            T val = buffer[tid];

            if (CTA_SIZE >= 1024) {
                if (tid < 512) buffer[tid] = val = op(val, buffer[tid + 512]);
                __syncthreads();
            }
            if (CTA_SIZE >= 512) {
                if (tid < 256) buffer[tid] = val = op(val, buffer[tid + 256]);
                __syncthreads();
            }
            if (CTA_SIZE >= 256) {
                if (tid < 128) buffer[tid] = val = op(val, buffer[tid + 128]);
                __syncthreads();
            }
            if (CTA_SIZE >= 128) {
                if (tid < 64) buffer[tid] = val = op(val, buffer[tid + 64]);
                __syncthreads();
            }

            if (tid < 32) {
                if (CTA_SIZE >= 64) { buffer[tid] = val = op(val, buffer[tid + 32]); }
                if (CTA_SIZE >= 32) { buffer[tid] = val = op(val, buffer[tid + 16]); }
                if (CTA_SIZE >= 16) { buffer[tid] = val = op(val, buffer[tid + 8]); }
                if (CTA_SIZE >= 8) { buffer[tid] = val = op(val, buffer[tid + 4]); }
                if (CTA_SIZE >= 4) { buffer[tid] = val = op(val, buffer[tid + 2]); }
                if (CTA_SIZE >= 2) { buffer[tid] = val = op(val, buffer[tid + 1]); }
            }
        }

        template<int CTA_SIZE, typename T, class BinOp>
        static __device__ __forceinline__ T reduce(volatile T *buffer, T init, BinOp op) {
            int tid = flattenedThreadId();
            T val = buffer[tid] = init;
            __syncthreads();

            if (CTA_SIZE >= 1024) {
                if (tid < 512) buffer[tid] = val = op(val, buffer[tid + 512]);
                __syncthreads();
            }
            if (CTA_SIZE >= 512) {
                if (tid < 256) buffer[tid] = val = op(val, buffer[tid + 256]);
                __syncthreads();
            }
            if (CTA_SIZE >= 256) {
                if (tid < 128) buffer[tid] = val = op(val, buffer[tid + 128]);
                __syncthreads();
            }
            if (CTA_SIZE >= 128) {
                if (tid < 64) buffer[tid] = val = op(val, buffer[tid + 64]);
                __syncthreads();
            }

            if (tid < 32) {
                if (CTA_SIZE >= 64) { buffer[tid] = val = op(val, buffer[tid + 32]); }
                if (CTA_SIZE >= 32) { buffer[tid] = val = op(val, buffer[tid + 16]); }
                if (CTA_SIZE >= 16) { buffer[tid] = val = op(val, buffer[tid + 8]); }
                if (CTA_SIZE >= 8) { buffer[tid] = val = op(val, buffer[tid + 4]); }
                if (CTA_SIZE >= 4) { buffer[tid] = val = op(val, buffer[tid + 2]); }
                if (CTA_SIZE >= 2) { buffer[tid] = val = op(val, buffer[tid + 1]); }
            }
            __syncthreads();
            return buffer[0];
        }
    };

    struct Warp {
        enum {
            LOG_WARP_SIZE = 5,
            WARP_SIZE = 1 << LOG_WARP_SIZE,
            STRIDE = WARP_SIZE
        };

        /** \brief Returns the warp lane ID of the calling thread. */
        static __device__ __forceinline__ unsigned int
        laneId() {
            unsigned int ret;
            asm("mov.u32 %0, %laneid;"
                : "=r"(ret));
            return ret;
        }

        static __device__ __forceinline__ unsigned int id() {
            int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
            return tid >> LOG_WARP_SIZE;
        }

        static __device__ __forceinline__ int laneMaskLt() {
#if (__CUDA_ARCH__ >= 200)
            unsigned int ret;
            asm("mov.u32 %0, %lanemask_lt;"
                : "=r"(ret));
            return ret;
#else
            return 0xFFFFFFFF >> (32 - laneId());
#endif
        }

        static __device__ __forceinline__ int binaryExclScan(int ballot_mask) {
            return __popc(Warp::laneMaskLt() & ballot_mask);
        }
    };
#endif

}// end namespace cx
