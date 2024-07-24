//
// Created by MiseTy on 2023/1/31.
//

#ifndef CSFD_SLAM_INTERNAL_H
#define CSFD_SLAM_INTERNAL_H
#include "EigenSupport.h"
#include "cuda_double_complex.hpp"
#include "cx.h"
#include "device_array.hpp"

#include "cuda/std/complex"
#include "cuda_complex.hpp"
#include "vector_functions.hpp"

#include <complex>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
using ushort = unsigned short;
using floatType = float;
using floatTypeICP = double;
using hostComplex = std::complex<floatType>;
using hostComplexICP = std::complex<floatTypeICP>;
using devComplex = cuda::std::complex<floatType>;
using devComplexICP = cuda::std::complex<floatTypeICP>;
using devDComplex = d_complex<floatType>;

//using devComplex = complex<floatType>;
//using devComplexICP = complex<floatTypeICP>;

using MapArr = DeviceArray2D<devComplex>;
using cx::divUp;
#define H_ 1e-7
#define invH_ 1e7

// TSDF fixed point divisor (if old format is enabled)
constexpr int DIVISOR = std::numeric_limits<short>::max();
/** \brief Default buffer size for fetching cloud. It limits max number of
 * points that can be extracted */
enum { DEFAULT_CLOUD_BUFFER_SIZE = 10 * 1000 * 1000 };

template<class D, class Matx>
D &device_cast(Matx &matx) {
    return (*reinterpret_cast<D *>(matx.data()));
}

/** \brief Camera intrinsics structure
 */
struct Intr {
    float fx, fy, cx, cy;
    Intr() {}
    Intr(float fx_, float fy_, float cx_, float cy_)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    Intr operator()(int level_index) const {
        int div = 1 << level_index;
        return (Intr(fx / div, fy / div, cx / div, cy / div));
    }
};

/** \brief 3 element complex vector for device code
 */
struct devComplex3 {
    devComplex x, y, z;
};

inline __host__ __device__ devComplex3 make_mcomplex3(devComplex x, devComplex y, devComplex z) {
    devComplex3 t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}

__device__ __forceinline__ devComplex
dot(const devComplex3 &v1, const devComplex3 &v2) {

    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ __forceinline__ devComplex3 &
operator+=(devComplex3 &vec, const float &v) {
    vec.x += v;
    vec.y += v;
    vec.z += v;
    return vec;
}

__device__ __forceinline__ devComplex3
operator+(const devComplex3 &v1, const devComplex3 &v2) {
    return make_mcomplex3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ __forceinline__ devComplex3
operator-(const devComplex3 &v1, const devComplex3 &v2) {
    return make_mcomplex3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__device__ __forceinline__ devComplex3
operator*(const devComplex3 &v1, const float &v) {
    return make_mcomplex3(v1.x * v, v1.y * v, v1.z * v);
}

__device__ __forceinline__ devComplex3
operator*(const devComplex3 &v1, const devComplex &v) {
    return make_mcomplex3(v1.x * v, v1.y * v, v1.z * v);
}

__device__ __forceinline__ float3
real(const devComplex3 &v) {
    return make_float3(v.x.real(), v.y.real(), v.z.real());
}

__device__ __forceinline__ float3
imag(const devComplex3 &v) {
    return make_float3(v.x.imag(), v.y.imag(), v.z.imag());
}

__device__ __forceinline__ devComplex3
conj(const devComplex3 &v) {
    return make_mcomplex3(conj(v.x), conj(v.y), conj(v.z));
}

__device__ __forceinline__ devComplex
norm(const devComplex3 &v) {
    return sqrt(dot(v, v));
}

__device__ __forceinline__ devComplex
squarednorm(const devComplex3 &v) {
    return dot(v, v);
}

__device__ __forceinline__ devComplex3
normalized(const devComplex3 &v) {
    return make_mcomplex3(v.x / norm(v), v.y / norm(v), v.z / norm(v));
}

__device__ __host__ __forceinline__ devComplex3
cross(const devComplex3 &v1, const devComplex3 &v2) {
    return make_mcomplex3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

/** \brief 3x3 complex Matrix for device code
 */
struct MatS33 {
    devComplex3 data[3];
};

__device__ __forceinline__ devComplex3 operator*(const MatS33 &m,
                                                 const devComplex3 &vec) {
    return make_mcomplex3(dot(m.data[0], vec), dot(m.data[1], vec),
                          dot(m.data[2], vec));
}


/** \brief 3 element double complex vector for device code
 */
struct devDComplex3 {
    devDComplex x, y, z;
};

inline __host__ __device__ devDComplex3 make_mDcomplex3(devDComplex x, devDComplex y, devDComplex z) {
    devDComplex3 t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}

__device__ __forceinline__ devDComplex
dot(const devDComplex3 &v1, const devDComplex3 &v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ __forceinline__ devDComplex3
operator+(const devDComplex3 &v1, const devDComplex3 &v2) {
    return make_mDcomplex3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ __forceinline__ devDComplex3
operator-(const devDComplex3 &v1, const devDComplex3 &v2) {
    return make_mDcomplex3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__device__ __forceinline__ devDComplex
norm(const devDComplex3 &v) {
    return sqrt(dot(v, v));
}

/** \brief 3x3 double complex Matrix for device code
 */
struct MatD33 {
    devDComplex3 data[3];
};

__device__ __forceinline__ devDComplex3 operator*(const MatD33 &m,
                                                  const devDComplex3 &vec) {
    return make_mDcomplex3(dot(m.data[0], vec), dot(m.data[1], vec),
                           dot(m.data[2], vec));
}

///////////////////////////////////////////////////////////
// 3 element float vector for device code
__device__ __forceinline__ float3
operator+(const float3 &v1, const float3 &v2) {
    return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ __forceinline__ float3
operator-(const float3 &v1, const float3 &v2) {
    return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__device__ __forceinline__ float
dot(const float3 &v1, const float3 &v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ __forceinline__ float
norm(const float3 &v) {
    return sqrt(dot(v, v));
}

/** \brief 3x3 float Matrix for device code
 */


struct Mat33 {
    float3 data[3];
};

__device__ __forceinline__ float3 operator*(const Mat33 &m,
                                            const float3 &vec) {
    return make_float3(dot(m.data[0], vec), dot(m.data[1], vec),
                       dot(m.data[2], vec));
}

#endif// CSFD_SLAM_INTERNAL_H
