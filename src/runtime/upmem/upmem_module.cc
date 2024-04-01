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

#include <tvm/runtime/registry.h>

#include <array>
#include <chrono>
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
                           std::string upmem_source,
                           std::unordered_map<std::string, size_t> padded_buffer_size)
      : binary_file_(binary_file),
        fmt_(fmt),
        fmap_(fmap),
        upmem_source_(upmem_source),
        padded_buffer_size(padded_buffer_size) {
    // std::fill(module_.begin(), module_.end(), nullptr);
  }
  // destructor
  ~UPMEMModuleNode() {}

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

        api->kernel_start = std::chrono::high_resolution_clock::now();
        dpu_launch(api->dpu_set, DPU_SYNCHRONOUS);
        api->kernel_end = std::chrono::high_resolution_clock::now();

        // struct dpu_set_t dpu;

        // FILE* original_stderr = stderr;
        // stderr = freopen("/dev/null", "w", stderr);
        // DPU_FOREACH(api->dpu_set, dpu) {
        //   FILE* tempFile = tmpfile();
        //   dpu_error_t error = dpu_log_read(dpu, tempFile);
        //   if (error == DPU_OK) {
        //     rewind(tempFile);
        //     char* buffer = NULL;
        //     size_t size = 0;
        //     while (getline(&buffer, &size, tempFile) != -1) {
        //       printf("%s", buffer);
        //     }
        //     free(buffer);
        //   }
        //   fclose(tempFile);
        // }
        // stderr = original_stderr;
      });
    }
  }

  void SaveToFile(const String& file_name, const String& format) final {
    LOG(FATAL) << "UpmemModuleNode::SaveToFile is not implemented";
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(upmem_source_);
  }

  String GetSource(const String& format) final { return upmem_source_; }

  int32_t GetPaddedSize(std::string name_hint) {
    if (padded_buffer_size.find(name_hint) == padded_buffer_size.end()) {
      return 0;
    }
    return padded_buffer_size[name_hint];
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

  std::unordered_map<std::string, size_t> padded_buffer_size;
  // the internal modules per GPU, to be lazily initialized.
  // std::array<CUmodule, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

class UPMEMModule : public Module {
 public:
  UPMEMModule() {}
  explicit UPMEMModule(ObjectPtr<Object> n) : Module(n) {}
  inline UPMEMModuleNode* operator->();
  inline const UPMEMModuleNode* operator->() const;
};

inline UPMEMModuleNode* UPMEMModule::operator->() {
  return static_cast<UPMEMModuleNode*>(get_mutable());
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
                         std::string upmem_source,
                         std::unordered_map<std::string, size_t> padded_buffer_size) {
  // TODO: build upmem module with dpurte-upmem-clang
  auto n = make_object<UPMEMModuleNode>(binary_file, fmt, fmap, upmem_source, padded_buffer_size);
  return Module(n);
}

// Load module from module.
Module UPMEMModuleLoadFile(const std::string& file_name, const String& format) {
  LOG(FATAL) << "UpmemModuleLoadFile is not implemented";
  return Module();
}

Module UPMEMModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return UPMEMModuleCreate(data, fmt, fmap, data, {});
}

// TVM_REGISTER_GLOBAL("runtime.module.loadfile_upmem").set_body_typed(UPMEMModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_upmem").set_body_typed(UPMEMModuleLoadBinary);

TVM_REGISTER_GLOBAL("runtime.module.upmem.padded_size")
    .set_body_typed([](UPMEMModule mod, std::string name_hint) {
      return mod->GetPaddedSize(name_hint);
    });

}  // namespace runtime
}  // namespace tvm
