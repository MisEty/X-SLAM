//
// Created by pzx on 2021/10/8.
//
#include "device_memory.h"
#include "cx.h"

#include "assert.h"
#include <algorithm>
#define HAVE_CUDA
#if !defined(HAVE_CUDA)

void throw_nogpu() { throw "PCL 2.0 exception"; }

DeviceMemory::DeviceMemory() { throw_nogpu(); }
DeviceMemory::DeviceMemory(void *, size_t) { throw_nogpu(); }
DeviceMemory::DeviceMemory(size_t) { throw_nogpu(); }
DeviceMemory::~DeviceMemory() { throw_nogpu(); }
DeviceMemory::DeviceMemory(const DeviceMemory& ) { throw_nogpu(); }
DeviceMemory& DeviceMemory::operator=(const DeviceMemory&) { throw_nogpu(); return *this;}
void DeviceMemory::create(size_t) { throw_nogpu(); }
void DeviceMemory::release() { throw_nogpu(); }
void DeviceMemory::copyTo(DeviceMemory&) const { throw_nogpu(); }
void DeviceMemory::upload(const void*, size_t) { throw_nogpu(); }
void DeviceMemory::download(void*) const { throw_nogpu(); }
bool DeviceMemory::empty() const { throw_nogpu(); }
DeviceMemory2D::DeviceMemory2D() { throw_nogpu(); }
DeviceMemory2D::DeviceMemory2D(int, int)  { throw_nogpu(); }
DeviceMemory2D::DeviceMemory2D(int, int, void*, size_t)  { throw_nogpu(); }
DeviceMemory2D::~DeviceMemory2D() { throw_nogpu(); }
DeviceMemory2D::DeviceMemory2D(const DeviceMemory2D&)  { throw_nogpu(); }
DeviceMemory2D& DeviceMemory2D::operator=(const DeviceMemory2D&) { throw_nogpu(); return *this;}
void DeviceMemory2D::create(int, int )  { throw_nogpu(); }
void DeviceMemory2D::release()  { throw_nogpu(); }
void DeviceMemory2D::copyTo(DeviceMemory2D&) const  { throw_nogpu(); }
void DeviceMemory2D::upload(const void *, size_t, int, int )  { throw_nogpu(); }
void DeviceMemory2D::download(void *, size_t ) const  { throw_nogpu(); }
bool DeviceMemory2D::empty() const { throw_nogpu(); }

#else

//////////////////////////    XADD    ///////////////////////////////

#ifdef __GNUC__

#if __GNUC__*10 + __GNUC_MINOR__ >= 42

#if !defined WIN32 && (defined __i486__ || defined __i586__ || defined __i686__ || defined __MMX__ || defined __SSE__  || defined __ppc__)
#define CV_XADD __sync_fetch_and_add
#else
#include <ext/atomicity.h>
#define CV_XADD __gnu_cxx::__exchange_and_add
#endif
#else
#include <bits/atomicity.h>
#if __GNUC__*10 + __GNUC_MINOR__ >= 34
#define CV_XADD __gnu_cxx::__exchange_and_add
#else
#define CV_XADD __exchange_and_add
#endif
#endif

#elif defined WIN32 || defined _WIN32
#include <intrin.h>
#define CV_XADD(addr,delta) _InterlockedExchangeAdd((long volatile*)(addr), (delta))
#else

template<typename _Tp> static inline _Tp CV_XADD(_Tp* addr, _Tp delta)
{ int tmp = *addr; *addr += delta; return tmp; }

#endif

////////////////////////    DeviceMemory    /////////////////////////////

DeviceMemory::DeviceMemory() : data_(0), sizeBytes_(0), refcount_(0) {}
DeviceMemory::DeviceMemory(void *ptr_arg, size_t sizeBytes_arg) : data_(ptr_arg), sizeBytes_(sizeBytes_arg), refcount_(0){}
DeviceMemory::DeviceMemory(size_t sizeBtes_arg)  : data_(0), sizeBytes_(0), refcount_(0) { create(sizeBtes_arg); }
DeviceMemory::~DeviceMemory() { release(); }

DeviceMemory::DeviceMemory(const DeviceMemory& other_arg)
    : data_(other_arg.data_), sizeBytes_(other_arg.sizeBytes_), refcount_(other_arg.refcount_)
{
  if( refcount_ )
    CV_XADD(refcount_, 1);
}

DeviceMemory& DeviceMemory::operator = (const DeviceMemory& other_arg)
{
  if( this != &other_arg )
  {
    if( other_arg.refcount_ )
      CV_XADD(other_arg.refcount_, 1);
    release();

    data_      = other_arg.data_;
    sizeBytes_ = other_arg.sizeBytes_;
    refcount_  = other_arg.refcount_;
  }
  return *this;
}

void DeviceMemory::create(size_t sizeBytes_arg)
{
  if (sizeBytes_arg == sizeBytes_)
    return;

  if( sizeBytes_arg > 0)
  {
    if( data_ )
      release();

    sizeBytes_ = sizeBytes_arg;

    cudaSafeCall( cudaMalloc(&data_, sizeBytes_), "DeviceMemory::create, alloc" );

    //refcount_ = (int*)cv::fastMalloc(sizeof(*refcount_));
    refcount_ = new int;
    *refcount_ = 1;
  }
}

void DeviceMemory::copyTo(DeviceMemory& other) const
{
  if (empty())
    other.release();
  else
  {
    other.create(sizeBytes_);
    cudaSafeCall( cudaMemcpy(other.data_, data_, sizeBytes_, cudaMemcpyDeviceToDevice), "DeviceMemory::copyTo, cpy" );
    cudaSafeCall( cudaDeviceSynchronize(), "DeviceMemory::copyTo, sync" );
  }
}

void DeviceMemory::release()
{
  if( refcount_ && CV_XADD(refcount_, -1) == 1 )
  {
    //cv::fastFree(refcount);
    delete refcount_;
    cudaSafeCall( cudaFree(data_), "DeviceMemory::release, free" );
  }
  data_ = 0;
  sizeBytes_ = 0;
  refcount_ = 0;
}

void DeviceMemory::upload(const void *host_ptr_arg, size_t sizeBytes_arg)
{
  create(sizeBytes_arg);
  cudaSafeCall( cudaMemcpy(data_, host_ptr_arg, sizeBytes_, cudaMemcpyHostToDevice), "DeviceMemory::upload, cpy" );
  cudaSafeCall( cudaDeviceSynchronize(), "DeviceMemory::upload, sync" );
}

void DeviceMemory::download(void *host_ptr_arg) const
{
  cudaSafeCall( cudaMemcpy(host_ptr_arg, data_, sizeBytes_, cudaMemcpyDeviceToHost), "DeviceMemory::download, cpy" );
  cudaSafeCall( cudaDeviceSynchronize(), "DeviceMemory::download, sync" );
}

void DeviceMemory::swap(DeviceMemory& other_arg)
{
  std::swap(data_, other_arg.data_);
  std::swap(sizeBytes_, other_arg.sizeBytes_);
  std::swap(refcount_, other_arg.refcount_);
}

bool DeviceMemory::empty() const { return !data_; }
size_t DeviceMemory::sizeBytes() const { return sizeBytes_; }


////////////////////////    DeviceMemory2D    /////////////////////////////

DeviceMemory2D::DeviceMemory2D() : data_(0), step_(0), colsBytes_(0), rows_(0), refcount_(0) {}

DeviceMemory2D::DeviceMemory2D(int rows_arg, int colsBytes_arg)
    : data_(0), step_(0), colsBytes_(0), rows_(0), refcount_(0)
{
  create(rows_arg, colsBytes_arg);
}

DeviceMemory2D::DeviceMemory2D(int rows_arg, int colsBytes_arg, void *data_arg, size_t step_arg)
    :  data_(data_arg), step_(step_arg), colsBytes_(colsBytes_arg), rows_(rows_arg), refcount_(0) {}

DeviceMemory2D::~DeviceMemory2D() { release(); }


DeviceMemory2D::DeviceMemory2D(const DeviceMemory2D& other_arg) :
                                                                  data_(other_arg.data_), step_(other_arg.step_), colsBytes_(other_arg.colsBytes_), rows_(other_arg.rows_), refcount_(other_arg.refcount_)
{
  if( refcount_ )
    CV_XADD(refcount_, 1);
}

DeviceMemory2D& DeviceMemory2D::operator = (const DeviceMemory2D& other_arg)
{
  if( this != &other_arg )
  {
    if( other_arg.refcount_ )
      CV_XADD(other_arg.refcount_, 1);
    release();

    colsBytes_ = other_arg.colsBytes_;
    rows_ = other_arg.rows_;
    data_ = other_arg.data_;
    step_ = other_arg.step_;

    refcount_ = other_arg.refcount_;
  }
  return *this;
}

void DeviceMemory2D::create(int rows_arg, int colsBytes_arg)
{
  if (colsBytes_ == colsBytes_arg && rows_ == rows_arg)
    return;

  if( rows_arg > 0 && colsBytes_arg > 0)
  {
    if( data_ )
      release();

    colsBytes_ = colsBytes_arg;
    rows_ = rows_arg;

    cudaSafeCall( cudaMallocPitch( (void**)&data_, &step_, colsBytes_, rows_), "DeviceMemory2D::create, alloc pitch" );

    //refcount = (int*)cv::fastMalloc(sizeof(*refcount));
    refcount_ = new int;
    *refcount_ = 1;
  }
}

void DeviceMemory2D::release()
{
  if( refcount_ && CV_XADD(refcount_, -1) == 1 )
  {
    //cv::fastFree(refcount);
    delete refcount_;
    cudaSafeCall( cudaFree(data_), "DeviceMemory2D::release, free" );
  }

  colsBytes_ = 0;
  rows_ = 0;
  data_ = nullptr;
  step_ = 0;
  refcount_ = nullptr;
}

void DeviceMemory2D::copyTo(DeviceMemory2D& other) const
{
  if (empty())
    other.release();
  else
  {
    other.create(rows_, colsBytes_);
    cudaSafeCall( cudaMemcpy2D(other.data_, other.step_, data_, step_, colsBytes_, rows_, cudaMemcpyDeviceToDevice), "DeviceMemory2D::copyTo cpy" );
    cudaSafeCall( cudaDeviceSynchronize(), "DeviceMemory2D::copyTo, sync" );
  }
}

void DeviceMemory2D::upload(const void *host_ptr_arg, size_t host_step_arg, int rows_arg, int colsBytes_arg)
{
  create(rows_arg, colsBytes_arg);
  cudaSafeCall( cudaMemcpy2D(data_, step_, host_ptr_arg, host_step_arg, colsBytes_, rows_, cudaMemcpyHostToDevice), "DeviceMemory2D::upload, cpy" );
  cudaSafeCall( cudaDeviceSynchronize(), "DeviceMemory2D::upload, sync" );
}

void DeviceMemory2D::download(void *host_ptr_arg, size_t host_step_arg) const
{
  cudaSafeCall( cudaMemcpy2D(host_ptr_arg, host_step_arg, data_, step_, colsBytes_, rows_, cudaMemcpyDeviceToHost), "DeviceMemory2D::download, cpy" );
  cudaSafeCall( cudaDeviceSynchronize(), "DeviceMemory2D::download, sync" );
}

void DeviceMemory2D::swap(DeviceMemory2D& other_arg)
{
  std::swap(data_, other_arg.data_);
  std::swap(step_, other_arg.step_);

  std::swap(colsBytes_, other_arg.colsBytes_);
  std::swap(rows_, other_arg.rows_);
  std::swap(refcount_, other_arg.refcount_);
}

bool DeviceMemory2D::empty() const { return !data_; }
int DeviceMemory2D::colsBytes() const { return colsBytes_; }
int DeviceMemory2D::rows() const { return rows_; }
size_t DeviceMemory2D::step() const { return step_; }

#endif
