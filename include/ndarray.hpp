/*!
 *  Copyright (c) 2016 by Contributors
 * \file ndarray.hpp
 * \brief implementation of the ndarray
 * \author Zhang Chen, Chuntao Hong
 */

#ifndef MXNETCPP_NDARRAY_HPP
#define MXNETCPP_NDARRAY_HPP

#include <map>
#include <string>
#include <vector>
#include "logging.h"
#include "ndarray.h"

namespace mxnet {
namespace cpp {

NDArray::NDArray() {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreateNone(&handle), 0);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
NDArray::NDArray(const NDArrayHandle &handle) {
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
NDArray::NDArray(const std::vector<mx_uint> &shape, const Context &context,
                 bool delay_alloc) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreate(shape.data(), shape.size(), context.GetDeviceType(),
                           context.GetDeviceId(), delay_alloc, &handle),
           0);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
NDArray::NDArray(const Shape &shape, const Context &context, bool delay_alloc) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreate(shape.data(), shape.ndim(), context.GetDeviceType(),
                           context.GetDeviceId(), delay_alloc, &handle),
           0);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
NDArray::NDArray(const mx_float *data, size_t size) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreateNone(&handle), 0);
  MXNDArraySyncCopyFromCPU(handle, data, size);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
NDArray::NDArray(const std::vector<mx_float> &data) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreateNone(&handle), 0);
  MXNDArraySyncCopyFromCPU(handle, data.data(), data.size());
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}

NDArray NDArray::operator+(mx_float scalar) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_plus_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &ret.blob_ptr_->handle_),
           0);
  return ret;
}
NDArray NDArray::operator-(mx_float scalar) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_minus_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &ret.blob_ptr_->handle_),
           0);
  return ret;
}
NDArray NDArray::operator*(mx_float scalar) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_mul_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &ret.blob_ptr_->handle_),
           0);
  return ret;
}
NDArray NDArray::operator/(mx_float scalar) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_div_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &ret.blob_ptr_->handle_),
           0);
  return ret;
}
NDArray NDArray::operator+(const NDArray &rhs) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_plus", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &ret.blob_ptr_->handle_),
      0);
  return ret;
}
NDArray NDArray::operator-(const NDArray &rhs) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_minus", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &ret.blob_ptr_->handle_),
      0);
  return ret;
}
NDArray NDArray::operator*(const NDArray &rhs) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_mul", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &ret.blob_ptr_->handle_),
      0);
  return ret;
}
NDArray NDArray::operator/(const NDArray &rhs) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_div", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &ret.blob_ptr_->handle_),
      0);
  return ret;
}
NDArray &NDArray::operator=(mx_float scalar) {
  FunctionHandle func_handle;
  MXGetFunction("_set_value", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, nullptr, &scalar, &blob_ptr_->handle_), 0);
  return *this;
}
NDArray &NDArray::operator+=(mx_float scalar) {
  FunctionHandle func_handle;
  MXGetFunction("_plus_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &blob_ptr_->handle_),
           0);
  return *this;
}
NDArray &NDArray::operator-=(mx_float scalar) {
  FunctionHandle func_handle;
  MXGetFunction("_minus_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &blob_ptr_->handle_),
           0);
  return *this;
}
NDArray &NDArray::operator*=(mx_float scalar) {
  FunctionHandle func_handle;
  MXGetFunction("_mul_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &blob_ptr_->handle_),
           0);
  return *this;
}
NDArray &NDArray::operator/=(mx_float scalar) {
  FunctionHandle func_handle;
  MXGetFunction("_div_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &blob_ptr_->handle_),
           0);
  return *this;
}
NDArray &NDArray::operator+=(const NDArray &rhs) {
  FunctionHandle func_handle;
  MXGetFunction("_plus", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &blob_ptr_->handle_), 0);
  return *this;
}
NDArray &NDArray::operator-=(const NDArray &rhs) {
  FunctionHandle func_handle;
  MXGetFunction("_minus", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &blob_ptr_->handle_), 0);
  return *this;
}
NDArray &NDArray::operator*=(const NDArray &rhs) {
  FunctionHandle func_handle;
  MXGetFunction("_mul", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &blob_ptr_->handle_), 0);
  return *this;
}
NDArray &NDArray::operator/=(const NDArray &rhs) {
  FunctionHandle func_handle;
  MXGetFunction("_div", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &blob_ptr_->handle_), 0);
  return *this;
}

void NDArray::SyncCopyFromCPU(const mx_float *data, size_t size) {
  MXNDArraySyncCopyFromCPU(blob_ptr_->handle_, data, size);
}
void NDArray::SyncCopyFromCPU(const std::vector<mx_float> &data) {
  MXNDArraySyncCopyFromCPU(blob_ptr_->handle_, data.data(), data.size());
}
void NDArray::SyncCopyToCPU(mx_float *data, size_t size) {
  MXNDArraySyncCopyToCPU(blob_ptr_->handle_, data, size);
}
NDArray NDArray::Copy(const Context &ctx) const {
  NDArray ret(GetShape(), ctx);
  FunctionHandle func_handle;
  MXGetFunction("_copyto", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, nullptr,
                        &ret.blob_ptr_->handle_),
           0);
  return ret;
}

NDArray NDArray::Slice(mx_uint begin, mx_uint end) const {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArraySlice(GetHandle(), begin, end, &handle), 0);
  return NDArray(handle);
}

void NDArray::WaitToRead() const {
  CHECK_EQ(MXNDArrayWaitToRead(blob_ptr_->handle_), 0);
}
void NDArray::WaitToWrite() {
  CHECK_EQ(MXNDArrayWaitToWrite(blob_ptr_->handle_), 0);
}
void NDArray::WaitAll() { CHECK_EQ(MXNDArrayWaitAll(), 0); }
void NDArray::SampleGaussian(mx_float mu, mx_float sigma, NDArray *out) {
  FunctionHandle func_handle;
  MXGetFunction("_random_gaussian", &func_handle);
  mx_float scalar[2] = {mu, sigma};
  CHECK_EQ(MXFuncInvoke(func_handle, nullptr, scalar, &out->blob_ptr_->handle_),
           0);
}
void NDArray::SampleUniform(mx_float begin, mx_float end, NDArray *out) {
  FunctionHandle func_handle;
  MXGetFunction("_random_uniform", &func_handle);
  mx_float scalar[2] = {begin, end};
  CHECK_EQ(MXFuncInvoke(func_handle, nullptr, scalar, &out->blob_ptr_->handle_),
           0);
}
void NDArray::Load(const std::string &file_name,
                   std::vector<NDArray> *array_list,
                   std::map<std::string, NDArray> *array_map) {
  mx_uint out_size, out_name_size;
  NDArrayHandle *out_arr;
  const char **out_names;
  CHECK_EQ(MXNDArrayLoad(file_name.c_str(), &out_size, &out_arr, &out_name_size,
                         &out_names),
           0);
  if (array_list != nullptr) {
    for (mx_uint i = 0; i < out_size; ++i) {
      array_list->push_back(NDArray(out_arr[i]));
    }
  }
  if (array_map != nullptr && out_name_size > 0) {
    CHECK_EQ(out_name_size, out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      (*array_map)[out_names[i]] = NDArray(out_arr[i]);
    }
  }
}
std::map<std::string, NDArray> NDArray::LoadToMap(
    const std::string &file_name) {
  std::map<std::string, NDArray> array_map;
  mx_uint out_size, out_name_size;
  NDArrayHandle *out_arr;
  const char **out_names;
  CHECK_EQ(MXNDArrayLoad(file_name.c_str(), &out_size, &out_arr, &out_name_size,
                         &out_names),
           0);
  if (out_name_size > 0) {
    CHECK_EQ(out_name_size, out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      array_map[out_names[i]] = NDArray(out_arr[i]);
    }
  }
  return array_map;
}
std::vector<NDArray> NDArray::LoadToList(const std::string &file_name) {
  std::vector<NDArray> array_list;
  mx_uint out_size, out_name_size;
  NDArrayHandle *out_arr;
  const char **out_names;
  CHECK_EQ(MXNDArrayLoad(file_name.c_str(), &out_size, &out_arr, &out_name_size,
                         &out_names),
           0);
  for (mx_uint i = 0; i < out_size; ++i) {
    array_list.push_back(NDArray(out_arr[i]));
  }
  return array_list;
}
void NDArray::Save(const std::string &file_name,
                   const std::map<std::string, NDArray> &array_map) {
  std::vector<NDArrayHandle> args;
  std::vector<const char *> keys;
  for (const auto &t : array_map) {
    args.push_back(t.second.GetHandle());
    keys.push_back(t.first.c_str());
  }
  CHECK_EQ(
      MXNDArraySave(file_name.c_str(), args.size(), args.data(), keys.data()),
      0);
}
void NDArray::Save(const std::string &file_name,
                   const std::vector<NDArray> &array_list) {
  std::vector<NDArrayHandle> args;
  std::vector<const char *> keys;
  CHECK_EQ(MXNDArraySave(file_name.c_str(), args.size(), args.data(), nullptr),
           0);
}

size_t NDArray::Offset(size_t h, size_t w) const {
  return (h * GetShape()[1]) + w;
}

size_t NDArray::Offset(size_t c, size_t h, size_t w) const {
  auto const shape = GetShape();
  return h * shape[0] * shape[2] + w * shape[0] + c;
}

mx_float NDArray::At(size_t h, size_t w) const {
  return GetData()[Offset(h, w)];
}

mx_float NDArray::At(size_t c, size_t h, size_t w) const {
  return GetData()[Offset(c, h, w)];
}

std::vector<mx_uint> NDArray::GetShape() const {
  const mx_uint *out_pdata;
  mx_uint out_dim;
  MXNDArrayGetShape(blob_ptr_->handle_, &out_dim, &out_pdata);
  std::vector<mx_uint> ret;
  for (mx_uint i = 0; i < out_dim; ++i) {
    ret.push_back(out_pdata[i]);
  }
  return ret;
}
const mx_float *NDArray::GetData() const {
  mx_float *ret;
  MXNDArrayGetData(blob_ptr_->handle_, &ret);
  return ret;
}
Context NDArray::GetContext() const {
  int out_dev_type;
  int out_dev_id;
  MXNDArrayGetContext(blob_ptr_->handle_, &out_dev_type, &out_dev_id);
  return Context((DeviceType)out_dev_type, out_dev_id);
}
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNETCPP_NDARRAY_HPP
