//
// Created by pzx on 2021/10/6.
//

#ifndef DEVICE_ARRAY_H
#define DEVICE_ARRAY_H

#include "device_memory.h"
#include <vector>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/** \brief @b DeviceArray class
 *
 * \note Typed container for GPU memory with reference counting.
 *
 * \author Anatoly Baksheev
 */
//////////////////////////////////////////////////////////////////////////////
/** \brief @b DeviceArray class
 *
 * \note Typed container for GPU memory with reference counting.
 *
 * \author Anatoly Baksheev
 */
template <class T>
class DeviceArray

    : public DeviceMemory {
public:
  /** \brief Element type. */
  using type = T;

  /** \brief Element size. */
  enum { elem_size = sizeof(T) };

  /** \brief Empty constructor. */
  DeviceArray();

  /** \brief Allocates internal buffer in GPU memory
   * \param size number of elements to allocate
   * */
  DeviceArray(std::size_t size);

  /** \brief Initializes with user allocated buffer. Reference counting is
   * disabled in this case. \param ptr pointer to buffer \param size elements
   * number
   * */
  DeviceArray(T *ptr, std::size_t size);

  /** \brief Copy constructor. Just increments reference counter. */
  DeviceArray(const DeviceArray &other);

  /** \brief Assignment operator. Just increments reference counter. */
  DeviceArray &operator=(const DeviceArray &other);

  /** \brief Allocates internal buffer in GPU memory. If internal buffer was
   * created before the function recreates it with new size. If new and old
   * sizes are equal it does nothing. \param size elements number
   * */
  void create(std::size_t size);

  /** \brief Decrements reference counter and releases internal buffer if
   * needed. */
  void release();

  /** \brief Performs data copying. If destination size differs it will be
   * reallocated. \param other destination container
   * */
  void copyTo(DeviceArray &other) const;

  /** \brief Uploads data to internal buffer in GPU memory. It calls create()
   * inside to ensure that intenal buffer size is enough. \param host_ptr
   * pointer to buffer to upload \param size elements number
   * */
  void upload(const T *host_ptr, std::size_t size);

  /** \brief Uploads data from CPU memory to internal buffer.
   * \return true if upload successful
   * \note In contrast to the other upload function, this function
   * never allocates memory.
   * \param host_ptr pointer to buffer to upload
   * \param device_begin_offset begin upload
   * \param num_elements number of elements from device_bein_offset
   * */
  bool upload(const T *host_ptr, std::size_t device_begin_offset,
              std::size_t num_elements);

  /** \brief Downloads data from internal buffer to CPU memory
   * \param host_ptr pointer to buffer to download
   * */
  void download(T *host_ptr) const;

  /** \brief Downloads data from internal buffer to CPU memory.
   * \return true if download successful
   * \param host_ptr pointer to buffer to download
   * \param device_begin_offset begin download location
   * \param num_elements number of elements from device_begin_offset
   * */
  bool download(T *host_ptr, std::size_t device_begin_offset,
                std::size_t num_elements) const;

  /** \brief Uploads data to internal buffer in GPU memory. It calls create()
   * inside to ensure that intenal buffer size is enough. \param data host
   * vector to upload from
   * */
  template <class A> void upload(const std::vector<T, A> &data);

  /** \brief Downloads data from internal buffer to CPU memory
   * \param data  host vector to download to
   * */
  template <typename A> void download(std::vector<T, A> &data) const;

  /** \brief Performs swap of data pointed with another device array.
   * \param other_arg device array to swap with
   * */
  void swap(DeviceArray &other_arg);

  /** \brief Returns pointer for internal buffer in GPU memory. */
  T *ptr();

  /** \brief Returns const pointer for internal buffer in GPU memory. */
  const T *ptr() const;

  // using DeviceMemory::ptr;

  /** \brief Returns pointer for internal buffer in GPU memory. */
  operator T *();

  /** \brief Returns const pointer for internal buffer in GPU memory. */
  operator const T *() const;

  /** \brief Returns size in elements. */
  std::size_t size() const;
};

///////////////////////////////////////////////////////////////////////////////
/** \brief @b DeviceArray2D class
 *
 * \note Typed container for pitched GPU memory with reference counting.
 *
 * \author Anatoly Baksheev
 */
template <class T> class DeviceArray2D : public DeviceMemory2D {
public:
  /** \brief Element type. */
  using type = T;

  /** \brief Element size. */
  enum { elem_size = sizeof(T) };

  /** \brief Empty constructor. */
  DeviceArray2D();

  /** \brief Allocates internal buffer in GPU memory
   * \param rows number of rows to allocate
   * \param cols number of elements in each row
   * */
  DeviceArray2D(int rows, int cols);

  /** \brief Initializes with user allocated buffer. Reference counting is
   * disabled in this case. \param rows number of rows \param cols number of
   * elements in each row \param data pointer to buffer \param stepBytes stride
   * between two consecutive rows in bytes
   * */
  DeviceArray2D(int rows, int cols, void *data, std::size_t stepBytes);

  /** \brief Copy constructor. Just increments reference counter. */
  DeviceArray2D(const DeviceArray2D &other);

  /** \brief Assignment operator. Just increments reference counter. */
  DeviceArray2D &operator=(const DeviceArray2D &other);

  /** \brief Allocates internal buffer in GPU memory. If internal buffer was
   * created before the function recreates it with new size. If new and old
   * sizes are equal it does nothing. \param rows number of rows to allocate
   * \param cols number of elements in each row
   * */
  void create(int rows, int cols);

  /** \brief Decrements reference counter and releases internal buffer if
   * needed. */
  void release();

  /** \brief Performs data copying. If destination size differs it will be
   * reallocated. \param other destination container
   * */
  void copyTo(DeviceArray2D &other) const;

  /** \brief Uploads data to internal buffer in GPU memory. It calls create()
   * inside to ensure that intenal buffer size is enough. \param host_ptr
   * pointer to host buffer to upload \param host_step stride between two
   * consecutive rows in bytes for host buffer \param rows number of rows to
   * upload \param cols number of elements in each row
   * */
  void upload(const void *host_ptr, std::size_t host_step, int rows, int cols);

  /** \brief Downloads data from internal buffer to CPU memory. User is
   * responsible for correct host buffer size. \param host_ptr pointer to host
   * buffer to download \param host_step stride between two consecutive rows in
   * bytes for host buffer
   * */
  void download(void *host_ptr, std::size_t host_step) const;

  /** \brief Performs swap of data pointed with another device array.
   * \param other_arg device array to swap with
   * */
  void swap(DeviceArray2D &other_arg);

  /** \brief Uploads data to internal buffer in GPU memory. It calls create()
   * inside to ensure that intenal buffer size is enough. \param data host
   * vector to upload from \param cols stride in elements between two
   * consecutive rows for host buffer
   * */
  template <class A> void upload(const std::vector<T, A> &data, int cols);

  /** \brief Downloads data from internal buffer to CPU memory
   * \param data host vector to download to
   * \param cols Output stride in elements between two consecutive rows for host
   * vector.
   * */
  template <class A> void download(std::vector<T, A> &data, int &cols) const;

  /** \brief Returns pointer to given row in internal buffer.
   * \param y row index
   * */
  T *ptr(int y = 0);

  /** \brief Returns const pointer to given row in internal buffer.
   * \param y row index
   * */
  const T *ptr(int y = 0) const;

  // using DeviceMemory2D::ptr;

  /** \brief Returns pointer for internal buffer in GPU memory. */
  operator T *();

  /** \brief Returns const pointer for internal buffer in GPU memory. */
  operator const T *() const;

  /** \brief Returns number of elements in each row. */
  int cols() const;

  /** \brief Returns number of rows. */
  int rows() const;

  /** \brief Returns frame_step in elements. */
  std::size_t elem_step() const;
};

////////////////////  Inline implementations of DeviceArray //////////////////
template <class T> inline DeviceArray<T>::DeviceArray() {}

template <class T>
inline DeviceArray<T>::DeviceArray(std::size_t size)
    : DeviceMemory(size * elem_size) {}

template <class T>
inline DeviceArray<T>::DeviceArray(T *ptr, std::size_t size)
    : DeviceMemory(ptr, size * elem_size) {}

template <class T>
inline DeviceArray<T>::DeviceArray(const DeviceArray &other)
    : DeviceMemory(other) {}

template <class T>
inline DeviceArray<T> &DeviceArray<T>::operator=(const DeviceArray &other) {
  DeviceMemory::operator=(other);
  return *this;
}

template <class T> inline void DeviceArray<T>::create(std::size_t size) {
  DeviceMemory::create(size * elem_size);
}

template <class T> inline void DeviceArray<T>::release() {
  DeviceMemory::release();
}

template <class T>
inline void DeviceArray<T>::copyTo(DeviceArray &other) const {
  DeviceMemory::copyTo(other);
}

template <class T>
inline void DeviceArray<T>::upload(const T *host_ptr, std::size_t size) {
  DeviceMemory::upload(host_ptr, size * elem_size);
}

template <class T>
inline bool DeviceArray<T>::upload(const T *host_ptr,
                                   std::size_t device_begin_offset,
                                   std::size_t num_elements) {
  std::size_t begin_byte_offset = device_begin_offset * sizeof(T);
  std::size_t num_bytes = num_elements * sizeof(T);
  return DeviceMemory::upload(host_ptr, begin_byte_offset, num_bytes);
}

template <class T> inline void DeviceArray<T>::download(T *host_ptr) const {
  DeviceMemory::download(host_ptr);
}

template <class T>
inline bool DeviceArray<T>::download(T *host_ptr,
                                     std::size_t device_begin_offset,
                                     std::size_t num_elements) const {
  std::size_t begin_byte_offset = device_begin_offset * sizeof(T);
  std::size_t num_bytes = num_elements * sizeof(T);
  return DeviceMemory::download(host_ptr, begin_byte_offset, num_bytes);
}

template <class T> void DeviceArray<T>::swap(DeviceArray &other_arg) {
  DeviceMemory::swap(other_arg);
}

template <class T> inline DeviceArray<T>::operator T *() { return ptr(); }

template <class T> inline DeviceArray<T>::operator const T *() const {
  return ptr();
}

template <class T> inline std::size_t DeviceArray<T>::size() const {
  return sizeBytes() / elem_size;
}

template <class T> inline T *DeviceArray<T>::ptr() {
  return DeviceMemory::ptr<T>();
}

template <class T> inline const T *DeviceArray<T>::ptr() const {
  return DeviceMemory::ptr<T>();
}

template <class T>
template <class A>
inline void DeviceArray<T>::upload(const std::vector<T, A> &data) {
  upload(&data[0], data.size());
}

template <class T>
template <class A>
inline void DeviceArray<T>::download(std::vector<T, A> &data) const {
  data.resize(size());
  if (!data.empty())
    download(&data[0]);
}

///////////////////  Inline implementations of DeviceArray2D //////////////////

template <class T> inline DeviceArray2D<T>::DeviceArray2D() {}

template <class T>
inline DeviceArray2D<T>::DeviceArray2D(int rows, int cols)
    : DeviceMemory2D(rows, cols * elem_size) {}

template <class T>
inline DeviceArray2D<T>::DeviceArray2D(int rows, int cols, void *data,
                                       std::size_t stepBytes)
    : DeviceMemory2D(rows, cols * elem_size, data, stepBytes) {}

template <class T>
inline DeviceArray2D<T>::DeviceArray2D(const DeviceArray2D &other)
    : DeviceMemory2D(other) {}

template <class T>
inline DeviceArray2D<T> &
DeviceArray2D<T>::operator=(const DeviceArray2D &other) {
  DeviceMemory2D::operator=(other);
  return *this;
}

template <class T> inline void DeviceArray2D<T>::create(int rows, int cols) {
  DeviceMemory2D::create(rows, cols * elem_size);
}

template <class T> inline void DeviceArray2D<T>::release() {
  DeviceMemory2D::release();
}

template <class T>
inline void DeviceArray2D<T>::copyTo(DeviceArray2D &other) const {
  DeviceMemory2D::copyTo(other);
}

template <class T>
inline void DeviceArray2D<T>::upload(const void *host_ptr,
                                     std::size_t host_step, int rows,
                                     int cols) {
  DeviceMemory2D::upload(host_ptr, host_step, rows, cols * elem_size);
}

template <class T>
inline void DeviceArray2D<T>::download(void *host_ptr,
                                       std::size_t host_step) const {
  DeviceMemory2D::download(host_ptr, host_step);
}

template <class T>
template <class A>
inline void DeviceArray2D<T>::upload(const std::vector<T, A> &data, int cols) {
  upload(&data[0], cols * elem_size, data.size() / cols, cols);
}

template <class T>
template <class A>
inline void DeviceArray2D<T>::download(std::vector<T, A> &data,
                                       int &elem_step) const {
  elem_step = cols();
  data.resize(cols() * rows());
  if (!data.empty())
    download(&data[0], colsBytes());
}

template <class T> void DeviceArray2D<T>::swap(DeviceArray2D &other_arg) {
  DeviceMemory2D::swap(other_arg);
}

template <class T> inline T *DeviceArray2D<T>::ptr(int y) {
  return DeviceMemory2D::ptr<T>(y);
}

template <class T> inline const T *DeviceArray2D<T>::ptr(int y) const {
  return DeviceMemory2D::ptr<T>(y);
}

template <class T> inline DeviceArray2D<T>::operator T *() { return ptr(); }

template <class T> inline DeviceArray2D<T>::operator const T *() const {
  return ptr();
}

template <class T> inline int DeviceArray2D<T>::cols() const {
  return DeviceMemory2D::colsBytes() / elem_size;
}

template <class T> inline int DeviceArray2D<T>::rows() const {
  return DeviceMemory2D::rows();
}

template <class T> inline std::size_t DeviceArray2D<T>::elem_step() const {
  return DeviceMemory2D::step() / elem_size;
}

#endif // DEVICE_ARRAY_H
