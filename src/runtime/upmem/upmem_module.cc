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
 * \file upmem_module.cc
 */
#include "upmem_module.h"
#include "upmem_common.h"

#include <tvm/runtime/registry.h>

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "upmem_common.h"

namespace tvm {
namespace runtime {

// Module to support thread-safe multi-GPU execution.
// cuModule is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class UPMEMModuleNode : public runtime::ModuleNode {
 public:
  explicit UPMEMModuleNode(std::string binary_file, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string upmem_source)
      : binary_file_(binary_file), fmt_(fmt), fmap_(fmap), upmem_source_(upmem_source) {
    // std::fill(module_.begin(), module_.end(), nullptr);
  }
  // destructor
  ~UPMEMModuleNode() {
  }

  const char* type_key() const final { return "upmem"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    auto it = fmap_.find(name);
    if (it == fmap_.end()) {
      return PackedFunc();
    } else {
      return PackedFunc([](TVMArgs args, TVMRetValue* rv) {
        UPMEMDeviceAPI* api = UPMEMDeviceAPI::Global();
        UPMEM_CALL(dpu_launch(api->dpu_set, DPU_SYNCHRONOUS));
      });
    }
  }

  void SaveToFile(const String& file_name, const String& format) final {
    LOG(FATAL) << "UpmemModuleNode::SaveToFile is not implemented";
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    LOG(FATAL) << "UpmemModuleNode::SaveToBinary is not implemented";
  }

  String GetSource(const String& format) final {
    return upmem_source_;
  }

 private:
  // the binary data
  std::string binary_file_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The upmem source.
  std::string upmem_source_;
  // the internal modules per GPU, to be lazily initialized.
  // std::array<CUmodule, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class UPMEMWrappedFunc {
 public:
  // initialize the UPMEM function.
  void Init(UPMEMModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            size_t num_void_args, const std::vector<std::string>& launch_param_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    // install kernel
    // launch kernel
  }

 private:
  // internal module
  UPMEMModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // launch parameters configuration
  LaunchParamConfig launch_param_config_;
};

Module UPMEMModuleCreate(std::string binary_file, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string upmem_source) {
  // TODO: build upmem module with dpurte-upmem-clang
  auto n = make_object<UPMEMModuleNode>(binary_file, fmt, fmap, upmem_source);
  return Module(n);
}

// Load module from module.
Module UPMEMModuleLoadFile(const std::string& file_name, const String& format) {
  LOG(FATAL) << "UpmemModuleLoadFile is not implemented";
  return Module();
}

Module UPMEMModuleLoadBinary(void* strm) {
  LOG(FATAL) << "UpmemModuleLoadBinary is not implemented";
  return Module();
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_dpukernel").set_body_typed(UPMEMModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dpukernel").set_body_typed(UPMEMModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm
