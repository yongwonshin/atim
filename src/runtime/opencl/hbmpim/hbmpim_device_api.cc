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
#include "pim_library/pim_util.h"
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

void* HBMPIMWorkspace::GetBaseMemobj() { return fragment_allocator_[0]->get_pim_base(); }

// TODO[ywshin]: refactor
int HBMPIMWorkspace::CopyDataFromTo(void* from, void* to, size_t size,
                                    pim_library::PimMemCpyType mem_copy_type) {
  int ret = 0;
  cl_command_queue queue = this->GetQueue(this->GetThreadEntry()->device);

  switch (mem_copy_type) {
    // case pim_library::PimMemCpyType::HOST_TO_PIM:
    //   uint64_t src_addr = (uint64_t)from;
    //   uint64_t dst_addr = (uint64_t)to;
    //   memcpy((void*)dst_addr, (void*)src_addr, size);
    //   break;
    case pim_library::PimMemCpyType::HOST_TO_DEVICE:
      OPENCL_CALL(
          clEnqueueWriteBuffer(queue, (cl_mem)to, CL_FALSE, 0, size, from, 0, nullptr, nullptr));
      break;
    // case pim_library::PimMemCpyType::PIM_TO_HOST:
    //   uint64_t src_addr = (uint64_t)from;
    //   uint64_t dst_addr = (uint64_t)to;
    //   memcpy((void*)dst_addr, (void*)src_addr, size);
    //   break;
    case pim_library::PimMemCpyType::DEVICE_TO_HOST:
      OPENCL_CALL(
          clEnqueueReadBuffer(queue, (cl_mem)from, CL_TRUE, 0, size, to, 0, nullptr, nullptr));
      break;
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
  return ret;
}

void* HBMPIMWorkspace::GetCrfBin(pim_library::PimOpType op_type, int output_size) {
  void* crf_bin = crf_generator_->FindCrf(op_type, output_size);
  if (crf_bin == nullptr) {
    crf_bin = MakeCrfBin(op_type, output_size);
  }
  return crf_bin;
}

void* HBMPIMWorkspace::MakeCrfBin(pim_library::PimOpType op_type, int data_size) {
  int max_crf_size = crf_generator_->GetMaxCrfSize();
  uint8_t h_crf[max_crf_size];
  void* d_crf = nullptr;
  int crf_size;
  DLDataType dtype;
  dtype.code = kDLUInt;
  dtype.bits = 8;
  dtype.lanes = 1;
  d_crf = AllocDataSpace(GetThreadEntry()->device, max_crf_size, kTempAllocaAlignment, dtype);
  cl_mem d_crf_buffer = reinterpret_cast<cl::BufferDescriptor*>(d_crf)->buffer;

  int lc = crf_generator_->GetLoopCounter(op_type, data_size);
  crf_generator_->GenBinaryWithLoop(op_type, lc, h_crf, &crf_size);
  CopyDataFromTo((void*)h_crf, (void*)d_crf_buffer, max_crf_size,
                 pim_library::PimMemCpyType::HOST_TO_DEVICE);
  crf_generator_->InsertToCrfLUT(op_type, data_size, (void*)d_crf);

  return d_crf;
}

void HBMPIMWorkspace::EmulatorTraceGen(unsigned int block_size, pim_library::PimOpType op_type) {
  h_fmtd16_size_[0] = 0;
  CopyDataFromTo((void*)cl_d_fmtd16_size_, (void*)h_fmtd16_size_, sizeof(size_t),
                 pim_library::PimMemCpyType::DEVICE_TO_HOST);
  CopyDataFromTo((void*)cl_d_fmtd16_, (void*)h_fmtd16_,
                 sizeof(pim_library::PimMemTraceData) * max_fmtd_size_,
                 pim_library::PimMemCpyType::DEVICE_TO_HOST);

  for (size_t i = 1; i < block_size; i++) {
    memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
           h_fmtd16_size_[0] * sizeof(pim_library::PimMemTraceData));
  }
  h_fmtd16_size_[0] *= block_size;
  ConvertMemTraceFrom16BTo32B(h_fmtd32_, (int*)h_fmtd32_size_, h_fmtd16_, (int)h_fmtd16_size_[0],
                              op_type);
}

int HBMPIMWorkspace::ConvertMemTraceFrom16BTo32B(pim_library::PimMemTraceData* fmtd32,
                                                 int* fmtd32_size,
                                                 pim_library::PimMemTraceData* fmtd16,
                                                 int fmtd16_size, pim_library::PimOpType op_type) {
  VLOG(2) << "fmtd16_size : " << fmtd16_size;
  int ret = 0;
  pim_library::TraceParser trace_converter;
  trace_converter.coalesce_trace(fmtd32, fmtd32_size, fmtd16, fmtd16_size);
#ifdef DEBUG_PIM
  const char* op_str = get_pim_op_string(op_type);
  std::string dump_data = TEST_VECTORS_DATA;
  dump_data.append("dump/");
  dump_data.append(op_str);
  std::string dump_fmtd16 = dump_data + "/fmtd16.dat";
  std::string dump_fmtd32 = dump_data + "/fmtd32.dat";
  dump_fmtd<16>(dump_fmtd16.c_str(), fmtd16, fmtd16_size);
  dump_fmtd<32>(dump_fmtd32.c_str(), fmtd32, fmtd32_size[0]);
#endif

  return ret;
}

int HBMPIMWorkspace::ExecuteGemmBiasAct(void* output, void* pim_data,
                                        pim_library::PimMemTraceData* fmtd32, int fmtd32_size,
                                        pim_library::PimOpType op_type, uint64_t pim_base_addr,
                                        void* temp_buf, pim_library::PimBo* bias,
                                        pim_library::PimActFunc act_func) {
  cl_command_queue queue = this->GetQueue(this->GetThreadEntry()->device);

  int ret = 0;
  int offset = 0;
  int offset_r = 0;

  int is_bias = (bias != nullptr) ? 1 : 0;
  int is_relu = (act_func == pim_library::PimActFunc::ACT_RELU) ? 1 : 0;

  auto output_data = static_cast<cl::BufferDescriptor*>(output)->host_ptr;
  int output_size = buffer_size_map_[output_data];
  int out_dim = output_size / pim_library::vega20_pbi.num_out_per_grf / sizeof(half_float::half);
  uint16_t* sim_output = new uint16_t[out_dim];
  uint16_t* h_bias = new uint16_t[out_dim];

  uint64_t tmp_data_addr =
      reinterpret_cast<uint64_t>(static_cast<cl::BufferDescriptor*>(temp_buf)->host_ptr);
  uint64_t pim_data_addr =
      reinterpret_cast<uint64_t>(static_cast<cl::BufferDescriptor*>(pim_data)->host_ptr);
  void* input_data = static_cast<cl::BufferDescriptor*>(pim_data)->host_ptr;

  // // TODO[ywshin]: for test
  // {
  //   for (int i = buffer_size_map_[input_data] / 2 - 10; i < buffer_size_map_[input_data] / 2 +
  //   10;
  //        i++) {
  //     std::cerr << ((half_float::half*)input_data)[i] << " ";
  //   }
  //   std::cerr << std::endl;
  // }

  void* input_data_ = nullptr;
  {
    // TODO[ywshin]: it must NOT fixed!!!!
    input_data_ = internal_buffer_map_["A"].data();
  }

  pim_sim_.preload_data_with_addr(pim_data_addr - pim_base_addr, input_data_,
                                  buffer_size_map_[input_data]);
  pim_sim_.execute_kernel((void*)fmtd32, fmtd32_size);
  pim_sim_.read_result_gemv(sim_output, tmp_data_addr - pim_base_addr, out_dim);

  if (is_bias) {
    clEnqueueReadBuffer(queue, (cl_mem)bias->data, CL_TRUE, 0, bias->size, (void*)h_bias, 0,
                        nullptr, nullptr);
    for (int i = 0; i < out_dim; i++) {
      ((half_float::half*)sim_output)[i] += ((half_float::half*)h_bias)[i];
    }
  }

  if (is_relu) {
    for (int i = 0; i < out_dim; i++) {
      if (((half_float::half*)sim_output)[i] < 0) ((half_float::half*)sim_output)[i] = 0;
    }
  }

  // // TODO[ywshin]: for test
  // {
  //   std::cerr << "pim_sim_ result: " << std::endl;
  //   for (int i = 0; i < out_dim; i++) {
  //     std::cerr << ((half_float::half*)sim_output)[i] << " ";
  //   }
  //   std::cerr << std::endl;
  // }

  // TODO[ywshin]: 이 부분 코딩해야 함!!!
  // cl_int err_code;
  // void* host_addr = clEnqueueMapBuffer(queue, output_data, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
  // 0,
  //                                      output_size, 0, nullptr, nullptr, &err_code);
  // OPENCL_CHECK_ERROR(err_code);
  // memcpy((half_float::half*)host_addr, (half_float::half*)sim_output, output_size);
  // OPENCL_CALL(clEnqueueUnmapMemObject(queue, output_data, host_addr, 0, nullptr, nullptr));
  // OPENCL_CALL(clFinish(queue));

  if (sim_output_ != nullptr) free(sim_output_);
  sim_output_ = sim_output;
  // delete[] sim_output;
  delete[] h_bias;

  return ret;
}

void HBMPIMWorkspace::Init() {
  if (initialized_) return;
  OpenCLWorkspace::Init("hbmpim", "gpu");

  crf_generator_ = std::make_shared<pim_library::PimCrfBinGen>(this);
  pim_library::PimGemvType pim_gemv_type = pim_library::PimGemvType::TILE_ACCUM;
  bool is_gemv_tile_tree = pim_gemv_type == pim_library::PimGemvType::TILE_TREE ? true : false;
  crf_generator_->SetGemvTileTree(is_gemv_tile_tree);

  int zero = 0;
  // TODO[ywshin]: for real hardware
  // int max_srf_size = 2048;
  // pim_manager_->alloc_memory((void**)&d_srf_bin_buffer_, max_srf_size * 2, MEM_TYPE_DEVICE);
  // pim_manager_->alloc_memory((void**)&zero_buffer_, 32 * 2, MEM_TYPE_DEVICE);
  // clEnqueueFillBuffer(queue, (cl_mem)zero_buffer_, (void*)&zero, sizeof(int), 0, 32 * 2, 0, NULL,
  //                     NULL);

  // #ifdef EMULATOR
  pim_sim_.initialize("/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                      "/include/dramsim2/ini/system_hbm_vega20.ini", 256 * 64 * 2, 64, 1);

  fmtd_size_per_ch_ = 100000;
  max_block_size_ = pim_library::vega20_pbi.num_pim_chan;
  max_fmtd_size_ = fmtd_size_per_ch_ * max_block_size_;
  // #endif

  // #ifdef EMULATOR
  size_t reserved_fmtd_size = max_fmtd_size_ * sizeof(pim_library::PimMemTraceData);

  d_emulator_trace_ = (pim_library::PimMemTracer*)malloc(sizeof(pim_library::PimMemTracer));
  h_fmtd16_ = (pim_library::PimMemTraceData*)malloc(reserved_fmtd_size);
  h_fmtd32_ = (pim_library::PimMemTraceData*)malloc(reserved_fmtd_size);
  h_fmtd16_size_ = (size_t*)malloc(sizeof(size_t));
  h_fmtd32_size_ = (size_t*)malloc(sizeof(size_t));

  cl_device_id device_id = GetCLDeviceID(this->GetThreadEntry()->device.device_id);
  auto platform = device_to_platform[device_id];
  auto context = this->contexts[platform];
  auto queue = this->GetQueue(this->GetThreadEntry()->device);
  cl_int err_code;
  cl_d_fmtd16_ = clCreateBuffer(context, CL_MEM_READ_WRITE, reserved_fmtd_size, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  OPENCL_CALL(clEnqueueFillBuffer(queue, cl_d_fmtd16_, (void*)&zero, sizeof(int), 0,
                                  reserved_fmtd_size, 0, nullptr, nullptr));

  cl_d_fmtd16_size_ =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(size_t), nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  OPENCL_CALL(clEnqueueFillBuffer(queue, cl_d_fmtd16_size_, (void*)&zero, sizeof(int), 0,
                                  sizeof(size_t), 0, nullptr, nullptr));

  cl_d_emulator_trace_ = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        sizeof(pim_library::PimMemTracer), nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  OPENCL_CALL(clEnqueueFillBuffer(queue, cl_d_emulator_trace_, (void*)&zero, sizeof(int), 0,
                                  sizeof(pim_library::PimMemTracer), 0, nullptr, nullptr));
  // #endif
}

bool HBMPIMWorkspace::IsOpenCLDevice(Device dev) {
  return dev.device_type == static_cast<DLDeviceType>(kDLHBMPIM);
}

void HBMPIMWorkspace::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  size_t nbytes = GetDataSize(*from);
  ICHECK_EQ(nbytes, GetDataSize(*to));
  ICHECK(IsContiguous(*from) && IsContiguous(*to))
      << "CopyDataFromTo only support contiguous array for now";

  if (IsOpenCLDevice(from->device) && to->device.device_type == kDLCPU) {
    // TODO[ywshin]: pim data to host
    auto from_desc = static_cast<cl::BufferDescriptor*>(from->data);
    std::memcpy(to->data, sim_output_, nbytes);
    // OpenCLWorkspace::CopyDataFromTo(from, to, stream);
  } else if (from->device.device_type == kDLCPU && IsOpenCLDevice(to->device)) {
    auto to_desc = static_cast<cl::BufferDescriptor*>(to->data);
    if (to_desc->layout == cl::BufferDescriptor::MemoryLayout::kBuffer1DInternal) {
      memcpy(to_desc->host_ptr, static_cast<const char*>(from->data) + from->byte_offset, nbytes);
      // TODO[ywshin]: unmap buffer
    } else {
      OpenCLWorkspace::CopyDataFromTo(from, to, stream);
    }
  } else {
    OpenCLWorkspace::CopyDataFromTo(from, to, stream);
  }
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
    uint64_t offset = reinterpret_cast<uint64_t>(local_buffer) -
                      fragment_allocator_[dev.device_id]->get_g_pim_base_addr(dev.device_id);
    cl_buffer_region sub_buffer_region = {offset, size};
    desc->buffer = clCreateSubBuffer(base_buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                     (void*)&sub_buffer_region, &err_code);
    desc->host_ptr = reinterpret_cast<cl_uchar*>(local_buffer);
    buffer_size_map_[(void*)desc->host_ptr] = size;
    desc->layout = cl::BufferDescriptor::MemoryLayout::kBuffer1DInternal;
    OPENCL_CHECK_ERROR(err_code);
    return desc;
  } else {
    desc->buffer =
        clCreateBuffer(this->contexts[platform], CL_MEM_CREATE_FLAGS, size, nullptr, &err_code);
    desc->layout = cl::BufferDescriptor::MemoryLayout::kBuffer1D;
    OPENCL_CHECK_ERROR(err_code);
    return CreateHostPtrIfEnabled(desc, dev, size);
  }
}

void* HBMPIMWorkspace::CreateHostPtrIfEnabled(cl::BufferDescriptor* desc, Device dev, size_t size) {
  auto ret = OpenCLWorkspace::CreateHostPtrIfEnabled(desc, dev, size);
#if defined(OPENCL_ENABLE_HOST_PTR)
  buffer_size_map_[(void*)desc->host_ptr] = size;
#endif  // OPENCL_ENABLE_HOST_PTR
  return ret;
}

int HBMPIMWorkspace::TransferHostToDevice(void* handle, uint64_t host_addr, uint64_t in_bank_addr,
                                          int bank_index, int size) {
  static size_t n = 0;
  size_t offset = static_cast<size_t>(in_bank_addr) * GetBytes(handle);
  int chan = bank_index / pim_library::vega20_pbi.num_banks;
  int bg = bank_index % pim_library::vega20_pbi.num_banks / pim_library::vega20_pbi.num_bank_groups;
  int bk = bank_index % pim_library::vega20_pbi.num_banks % pim_library::vega20_pbi.num_bank_groups;
  uint64_t pim_address = pim_library::addr_gen_s(chan, 0, bg, bk, 0, 0, offset);
  void* host_address = reinterpret_cast<void*>(HostOffset(handle, host_addr));

  std::string var_name = GetSymbolName(handle);
  auto& v = internal_buffer_map_[var_name];
  if (v.size() <= pim_address / 2) v.resize(pim_address / 2 + size);
  std::memcpy((char*)v.data() + pim_address, host_address, size * GetBytes(handle));
  return 0;
}
int HBMPIMWorkspace::TransferDeviceToHost(void* handle, uint64_t host_addr, uint64_t in_bank_addr,
                                          int bank_idx, int size) {
  // TODO[ywshin]
  return 0;
}

typedef dmlc::ThreadLocalStore<HBMPIMThreadEntry> HBMPIMThreadStore;

HBMPIMThreadEntry* HBMPIMThreadEntry::ThreadLocal() { return HBMPIMThreadStore::Get(); }

TVM_REGISTER_GLOBAL("device_api.hbmpim").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = HBMPIMWorkspace::Global();
  *rv = static_cast<void*>(ptr);
});

TVM_REGISTER_GLOBAL("device_api.hbmpim.acquire_resources")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      int32_t bank_num = args[0];
      HBMPIMWorkspace::Global()->AcquireResources(bank_num);
    });

TVM_REGISTER_GLOBAL("device_api.hbmpim.release_resources")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = static_cast<int>(HBMPIMWorkspace::Global()->ReleaseResources());
    });

TVM_REGISTER_GLOBAL("device_api.hbmpim.pim_allocate_memory")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      void* buffer_handle = args[0];
      std::string var_name = args[1];
      std::string type = args[2];
      DataType dtype = DataType(String2DLDataType(type));
      int32_t size = args[3];
      int32_t bank_index = args[4];
      HBMPIMWorkspace::Global()->SetPimMemoryEntry(buffer_handle, var_name, dtype, size,
                                                   bank_index);
    });

TVM_REGISTER_GLOBAL("device_api.hbmpim.pim_free_memory")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      void* handle = args[0];
      HBMPIMWorkspace::Global()->ErasePimMemoryEntry(handle);
    });

TVM_REGISTER_GLOBAL("device_api.hbmpim.pim_transfer_host_to_device")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      void* handle = args[0];
      uint64_t host_address = args[1];
      uint64_t in_bank_addr = args[2];
      int bank_index = args[3];
      int size = args[4];
      *rv = static_cast<int>(HBMPIMWorkspace::Global()->TransferHostToDevice(
          handle, host_address, in_bank_addr, bank_index, size));
    });

TVM_REGISTER_GLOBAL("device_api.hbmpim.pim_transfer_device_to_host")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      void* handle = args[0];
      uint64_t host_address = args[1];
      uint64_t in_bank_addr = args[2];
      int bank_index = args[3];
      int size = args[4];
      *rv = static_cast<int>(HBMPIMWorkspace::Global()->TransferDeviceToHost(
          handle, host_address, in_bank_addr, bank_index, size));
    });

}  // namespace cl

TVM_REGISTER_OBJECT_TYPE(HBMPIMTimerNode);
TVM_REGISTER_GLOBAL("profiling.timer.hbmpim").set_body_typed([](Device dev) {
  return Timer(make_object<HBMPIMTimerNode>(dev));
});

}  // namespace runtime
}  // namespace tvm
