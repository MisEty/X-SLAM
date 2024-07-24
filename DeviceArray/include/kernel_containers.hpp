//
// Created by pzx on 2021/10/6.
//

#ifndef KERNEL_CONTAINERS_H
#define KERNEL_CONTAINERS_H

#if defined(__CUDACC__)
#define __PCL_GPU_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
#define __PCL_GPU_HOST_DEVICE__
#endif

#include <cstddef>

template <typename T> struct DevPtr {
  typedef T elem_type;
  const static size_t elem_size = sizeof(elem_type);

  T *data;

  __PCL_GPU_HOST_DEVICE__
  DevPtr() : data(nullptr) {}

  __PCL_GPU_HOST_DEVICE__
  DevPtr(T *data_arg) : data(data_arg) {}

  __PCL_GPU_HOST_DEVICE__ std::size_t elemSize() const { return elem_size; }

  __PCL_GPU_HOST_DEVICE__
  operator T *() { return data; }

  __PCL_GPU_HOST_DEVICE__ operator const T *() const { return data; }
};

template <typename T> struct PtrSz : public DevPtr<T> {
  __PCL_GPU_HOST_DEVICE__
  PtrSz() : size(0) {}

  __PCL_GPU_HOST_DEVICE__
  PtrSz(T *data_arg, std::size_t size_arg)
      : DevPtr<T>(data_arg), size(size_arg) {}

  std::size_t size;
};

template <typename T> struct PtrStep : public DevPtr<T> {
  __PCL_GPU_HOST_DEVICE__ PtrStep() : step(0) {}

  __PCL_GPU_HOST_DEVICE__ PtrStep(T *data_arg, size_t step_arg)
      : DevPtr<T>(data_arg), step(step_arg) {}

  /** \brief stride between two consecutive rows in bytes. Step is stored always
   * and everywhere in bytes!!! */
  size_t step;

  __PCL_GPU_HOST_DEVICE__ T *ptr(int y = 0) {
    return (T *)((char *)DevPtr<T>::data + y * step);
  }

  __PCL_GPU_HOST_DEVICE__ const T *ptr(int y = 0) const {
    return (const T *)((const char *)DevPtr<T>::data + y * step);
  }

  __PCL_GPU_HOST_DEVICE__ T &operator()(int y, int x) { return ptr(y)[x]; }

  __PCL_GPU_HOST_DEVICE__ const T &operator()(int y, int x) const {
    return ptr(y)[x];
  }
};

template <typename T> struct PtrStepSz : public PtrStep<T> {
  __PCL_GPU_HOST_DEVICE__ PtrStepSz() : cols(0), rows(0) {}

  __PCL_GPU_HOST_DEVICE__ PtrStepSz(int rows_arg, int cols_arg, T *data_arg,
                                    size_t step_arg)
      : PtrStep<T>(data_arg, step_arg), cols(cols_arg), rows(rows_arg) {}

  int cols;
  int rows;
};

#undef __PCL_GPU_HOST_DEVICE__
#endif // KERNEL_CONTAINERS_H
