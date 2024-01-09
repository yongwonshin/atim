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
#include "pim_library/block_allocator.h"
#include "pim_library/simple_heap.h"

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

HBMPIMWorkspace::HBMPIMWorkspace() {
  crf_generator_ = std::make_shared<pim_library::PimCrfBinGen>(this);
}

void* HBMPIMWorkspace::GetBaseMemobj() { return fragment_allocator_[0]->get_pim_base(); }

// TODO[ywshin]: refactor
int HBMPIMWorkspace::copyDataFromTo(void* from, void* to, size_t size,
                                    pim_library::PimMemCpyType mem_copy_type) {
  VLOG(2) << "[START] " << __FUNCTION__ << " called";
  int ret = 0;
  cl_command_queue queue = this->GetQueue(this->GetThreadEntry()->device);
  cl_mem to_buf = reinterpret_cast<cl::BufferDescriptor*>(to)->buffer;

  switch (mem_copy_type) {
    // case pim_library::PimMemCpyType::HOST_TO_PIM:
    //   uint64_t src_addr = (uint64_t)from;
    //   uint64_t dst_addr = (uint64_t)to;
    //   memcpy((void*)dst_addr, (void*)src_addr, size);
    //   break;
    case pim_library::PimMemCpyType::HOST_TO_DEVICE:
      OPENCL_CALL(
          clEnqueueWriteBuffer(queue, to_buf, CL_FALSE, 0, size, from, 0, nullptr, nullptr));
      break;
    // case pim_library::PimMemCpyType::PIM_TO_HOST:
    //   uint64_t src_addr = (uint64_t)from;
    //   uint64_t dst_addr = (uint64_t)to;
    //   memcpy((void*)dst_addr, (void*)src_addr, size);
    //   break;
    // case pim_library::PimMemCpyType::DEVICE_TO_HOST:
    //   err_code = clEnqueueReadBuffer(queue, src_buff, CL_TRUE, 0, size, to, 0, nullptr, nullptr);
    //   OPENCL_CHECK_ERROR(err_code);
    //   break;
    // case pim_library::PimMemCpyType::DEVICE_TO_PIM:
    //   err_code = clEnqueueReadBuffer(queue, src_buff, CL_TRUE, 0, size, (void*)dst_buff, 0,
    //   nullptr,
    //                                  nullptr);
    //   OPENCL_CHECK_ERROR(err_code);
    //   break;
    // case pim_library::PimMemCpyType::PIM_TO_DEVICE:
    //   err_code = clEnqueueWriteBuffer(queue, dst_buff, CL_TRUE, 0, size, (void*)src_buff, 0,
    //                                   nullptr, nullptr);
    //   OPENCL_CHECK_ERROR(err_code);
    //   break;
    // case pim_library::PimMemCpyType::DEVICE_TO_DEVICE:
    //   err_code = clEnqueueCopyBuffer(queue, src_buff, dst_buff, 0, 0, size, 0, nullptr, nullptr);
    //   OPENCL_CHECK_ERROR(err_code);
    //   err_code = clFinish(queue);
    //   OPENCL_CHECK_ERROR(err_code);
    //   break;
    // case pim_library::PimMemCpyType::HOST_TO_HOST:
    //   memcpy(to, from, size);
    //   break;
    default:
      LOG(FATAL) << "Invalid copy type";
      break;
  }
  OPENCL_CALL(clFinish(queue));
  DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
  return ret;
}

void* HBMPIMWorkspace::GetCrfBin(pim_library::PimOpType op_type, int output_size) {
  void* crf_bin = crf_generator_->FindCrf(op_type, output_size);
  if (crf_bin == nullptr) {
    crf_bin = makeCrfBin(op_type, output_size);
  }
  return crf_bin;
}

void* HBMPIMWorkspace::makeCrfBin(pim_library::PimOpType op_type, int data_size) {
  VLOG(2) << "[START] " << __FUNCTION__ << " called";
  int max_crf_size = crf_generator_->GetMaxCrfSize();
  uint8_t h_crf[max_crf_size];
  void* d_crf = nullptr;
  int crf_size;
  DLDataType dtype;
  dtype.code = kDLUInt;
  dtype.bits = 8;
  dtype.lanes = 1;
  d_crf = AllocDataSpace(GetThreadEntry()->device, max_crf_size, kTempAllocaAlignment, dtype);

  int lc = crf_generator_->GetLoopCounter(op_type, data_size);
  crf_generator_->GenBinaryWithLoop(op_type, lc, h_crf, &crf_size);
  copyDataFromTo((void*)h_crf, d_crf, max_crf_size, pim_library::PimMemCpyType::HOST_TO_DEVICE);
  crf_generator_->InsertToCrfLUT(op_type, data_size, d_crf);

  VLOG(2) << "[END] " << __FUNCTION__ << " called";
  return d_crf;
}

void HBMPIMWorkspace::Init() { OpenCLWorkspace::Init("hbmpim", "gpu"); }

bool HBMPIMWorkspace::IsOpenCLDevice(Device dev) {
  return dev.device_type == static_cast<DLDeviceType>(kDLHBMPIM);
}

void* HBMPIMWorkspace::AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                                      Optional<String> mem_scope) {
  if (!mem_scope.defined() || mem_scope.value() == "global" || mem_scope.value() == "internal") {
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
  if (mem_scope.defined() && mem_scope == "internal") {
    if (fragment_allocator_.size() <= dev.device_id) {
      fragment_allocator_.resize(dev.device_id + 1);
      fragment_allocator_[dev.device_id] =
          std::make_shared<pim_library::SimpleHeap<pim_library::BlockAllocator>>();
    }
    // refer to PIMLibrary/runtime/source/manager/ocl/OclMemoryManager.cpp
    // it uses fragment allocator to :
    // 1. either allocates a pim block of block size if pim_alloc_done is false and returns the base
    // address as the requested buffer address.
    // 2. else return a virtual address of the buffer in the above allocated region which has to be
    // then allocted as Subbuffer of above buffer in opencl. Note: get_cl_mem_obj return a base
    // buffer in device address space and fragment allocator returns host mapped address.
    fragment_allocator_[dev.device_id]->set_context(this->contexts[platform]);
    void* local_buffer = fragment_allocator_[dev.device_id]->alloc(size, dev);
    // create sub buffer object in host mapped device addres space.
    cl_mem base_buffer = (cl_mem)fragment_allocator_[dev.device_id]->get_pim_base();
    cl_buffer_region sub_buffer_region = {
        reinterpret_cast<uint64_t>(
            local_buffer - fragment_allocator_[dev.device_id]->get_g_pim_base_addr(dev.device_id)),
        size};
    desc->buffer = clCreateSubBuffer(base_buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                     (void*)&sub_buffer_region, &err_code);
    desc->layout = cl::BufferDescriptor::MemoryLayout::kBuffer1D;
    OPENCL_CHECK_ERROR(err_code);
  } else {
    desc->buffer =
        clCreateBuffer(this->contexts[platform], CL_MEM_CREATE_FLAGS, size, nullptr, &err_code);
    desc->layout = cl::BufferDescriptor::MemoryLayout::kBuffer1D;
    OPENCL_CHECK_ERROR(err_code);
  }
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
