/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file hbmpim_device_api.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include "hbmpim_common.h"

#ifdef OPENCL_ENABLE_HOST_PTR
#define CL_MEM_CREATE_FLAGS CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR
#else
#define CL_MEM_CREATE_FLAGS CL_MEM_READ_WRITE
#endif

namespace tvm {
namespace runtime {
namespace cl {

OpenCLThreadEntry* HBMPIMWorkspace::GetThreadEntry() { return HBMPIMThreadEntry::ThreadLocal(); }

OpenCLWorkspace* HBMPIMWorkspace::Global() {
  static OpenCLWorkspace* inst = new HBMPIMWorkspace();
  return inst;
}

void HBMPIMWorkspace::Init() { OpenCLWorkspace::Init("hbmpim", "gpu"); }

bool HBMPIMWorkspace::IsOpenCLDevice(Device dev) {
  return dev.device_type == static_cast<DLDeviceType>(kDLHBMPIM);
}

void* HBMPIMWorkspace::AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                                      Optional<String> mem_scope) {
  if (!mem_scope.defined() || mem_scope.value() == "global") {
    return DeviceAPI::AllocDataSpace(dev, ndim, shape, dtype, mem_scope);
  }
  ICHECK(IsTextureStorage(std::string(mem_scope.value())))
      << "Device does not support allocate data space with "
      << "specified memory scope: " << mem_scope.value();

  ICHECK(ndim > 2) << "Shape for texture allocation must be at least rank 3; "
                   << "provided shape is rank " << ndim;

  cl::BufferDescriptor* desc = new cl::BufferDescriptor(mem_scope);
  size_t axis = DefaultTextureLayoutSeparator(ndim, mem_scope.value());
  auto texture = ApplyTexture2DFlattening<int64_t>(shape, ndim, axis);
  desc->buffer = AllocTexture(dev, texture.width, texture.height, dtype);
  return desc;
}

void* HBMPIMWorkspace::AllocDataSpace(Device dev, size_t size, size_t alignment,
                                      DLDataType type_hint, Optional<String> mem_scope) {
  this->Init();
  cl_device_id device_id = GetCLDeviceID(dev.device_id);
  auto platform = device_to_platform[device_id];
  cl_int err_code;
  cl::BufferDescriptor* desc = new cl::BufferDescriptor;
  // CL_INVALID_BUFFER_SIZE if size is 0.
  if (size == 0) {
    size = 1;
  }
  desc->buffer =
      clCreateBuffer(this->contexts[platform], CL_MEM_CREATE_FLAGS, size, nullptr, &err_code);
  desc->layout = cl::BufferDescriptor::MemoryLayout::kBuffer1D;
  OPENCL_CHECK_ERROR(err_code);
  return CreateHostPtrIfEnabled(desc, dev, size);
}

typedef dmlc::ThreadLocalStore<HBMPIMThreadEntry> HBMPIMThreadStore;

HBMPIMThreadEntry* HBMPIMThreadEntry::ThreadLocal() { return HBMPIMThreadStore::Get(); }

TVM_REGISTER_GLOBAL("device_api.hbmpim").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = HBMPIMWorkspace::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace cl
}  // namespace runtime
}  // namespace tvm
