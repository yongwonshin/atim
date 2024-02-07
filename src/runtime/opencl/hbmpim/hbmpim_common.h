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
 * \file hbmpim_common.h
 * \brief HBMPIM common header
 */
#ifndef TVM_RUNTIME_OPENCL_HBMPIM_HBMPIM_COMMON_H_
#define TVM_RUNTIME_OPENCL_HBMPIM_HBMPIM_COMMON_H_

#include <memory>

#include "../opencl_common.h"
#include "pim_library/block_allocator.h"
#include "pim_library/pim_command.h"
#include "pim_library/pim_data_types.h"
#include "pim_library/pim_info.h"
#include "pim_library/pim_trace_coalescer.h"
#include "pim_library/simple_heap.h"
#include "tools/emulator_api/PimSimulator2.h"

namespace tvm {
namespace runtime {

namespace pim_library {
class PimCrfBinGen;
}

class HBMPIMModuleNode : public OpenCLModuleNode {
 public:
  explicit HBMPIMModuleNode(std::string data, std::string fmt,
                            std::unordered_map<std::string, FunctionInfo> fmap, std::string source)
      : OpenCLModuleNode(data, fmt, fmap, source) {}
  cl::OpenCLWorkspace* GetGlobalWorkspace() final;
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final;
  void Init() final;
};

namespace cl {

/*!
 * \brief Process global HBMPIM workspace.
 */
class HBMPIMWorkspace final : public OpenCLWorkspace {
 public:
  // override OpenCL device API
  void Init() final;
  bool IsOpenCLDevice(Device dev) final;
  bool IsHBMPIMDevice(Device dev);
  OpenCLThreadEntry* GetThreadEntry() final;
  // get the global workspace
  static OpenCLWorkspace* Global();
  void* AllocDataSpace(Device dev, size_t size, size_t alignment, DLDataType type_hint,
                       Optional<String> mem_scope = NullOpt) final;
  void* AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                       Optional<String> mem_scope = NullOpt) final;
  std::vector<std::shared_ptr<pim_library::SimpleHeap<pim_library::BlockAllocator>>>
      fragment_allocator_;
  std::shared_ptr<pim_library::PimCrfBinGen> crf_generator_;
  void* GetCrfBin(pim_library::PimOpType op_type, int output_size);
  void* GetBaseMemobj();
  void EmulatorTraceGen(unsigned int block_size, pim_library::PimOpType op_type);
  void* MakeCrfBin(pim_library::PimOpType op_type, int data_size);
  int CopyDataFromTo(void* from, void* to, size_t size, pim_library::PimMemCpyType mem_copy_type);
  void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) final;
  int ExecuteGemmBiasAct(void* output, void* pim_data, pim_library::PimMemTraceData* fmtd32,
                         int fmtd32_size, pim_library::PimOpType op_type, uint64_t pim_base_addr,
                         void* temp_buf, pim_library::PimBo* bias,
                         pim_library::PimActFunc act_func);
  int ConvertMemTraceFrom16BTo32B(pim_library::PimMemTraceData* fmtd32, int* fmtd32_size,
                                  pim_library::PimMemTraceData* fmtd16, int fmtd16_size,
                                  pim_library::PimOpType op_type);
  void* CreateHostPtrIfEnabled(cl::BufferDescriptor* desc, Device dev, size_t size) final;

  // #ifdef EMULATOR
  pim_library::PimMemTraceData* h_fmtd16_;
  pim_library::PimMemTraceData* h_fmtd32_;
  pim_library::PimMemTracer* d_emulator_trace_;
  cl_mem cl_d_fmtd16_;
  cl_mem cl_d_fmtd16_size_;
  cl_mem cl_d_emulator_trace_;

  size_t* h_fmtd16_size_;
  size_t* h_fmtd32_size_;
  int fmtd_size_per_ch_;
  int max_block_size_;
  int max_fmtd_size_;
  PimSimulator2 pim_sim_;
  std::unordered_map<const void*, size_t> buffer_size_map_;
  // #endif
};

/*! \brief Thread local workspace for HBMPIM*/
class HBMPIMThreadEntry : public OpenCLThreadEntry {
 public:
  // constructor
  HBMPIMThreadEntry()
      : OpenCLThreadEntry(static_cast<DLDeviceType>(kDLHBMPIM), HBMPIMWorkspace::Global()) {}

  // get the global workspace
  static HBMPIMThreadEntry* ThreadLocal();
};
}  // namespace cl

namespace pim_library {

class PimCrfBinGen {
 public:
  PimCrfBinGen(cl::HBMPIMWorkspace* w);
  virtual ~PimCrfBinGen();

  int initialize();
  int deinitialize();
  void createPimCmd(PimOpType op_type, int lc);
  void SetGemvTileTree(bool is_gemv_tile_tree);
  int GetLoopCounter(PimOpType op_type, int input_size);
  void* make_crf_bin(PimOpType op_type, int data_size);
  void* FindCrf(PimOpType op_type, int data_size);
  void GenBinaryWithLoop(PimOpType op_type, int lc, uint8_t* bin_buf, int* crf_sz);
  void InsertToCrfLUT(PimOpType op_type, int data_size, void* data);
  int GetMaxCrfSize();

 private:
  void changeToBinary(uint8_t* crf_binary, int* crf_size);

  cl::HBMPIMWorkspace* w_;
  std::vector<PimCommand> cmds_;
  std::map<std::pair<PimOpType, int>, void*> crf_lut_;
  const PimBlockInfo* pbi_;
  bool is_gemv_tile_tree_;
  int max_crf_size_;
};

}  // namespace pim_library
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENCL_HBMPIM_HBMPIM_COMMON_H_
