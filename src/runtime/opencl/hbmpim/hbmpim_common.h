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
#include "pim_library/simple_heap.h"

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
};

namespace cl {

/*!
 * \brief Process global HBMPIM workspace.
 */
class HBMPIMWorkspace final : public OpenCLWorkspace {
 public:
  HBMPIMWorkspace();
  // override OpenCL device API
  void Init() final;
  bool IsOpenCLDevice(Device dev) final;
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

 private:
  void* makeCrfBin(pim_library::PimOpType op_type, int data_size);
  int copyDataFromTo(void* from, void* to, size_t size, pim_library::PimMemCpyType mem_copy_type);
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
