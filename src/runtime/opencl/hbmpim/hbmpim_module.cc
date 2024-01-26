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
 * \file hbmpim_module.cc
 */
#include "hbmpim_module.h"

#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "hbmpim_common.h"

namespace tvm {
namespace runtime {

class HBMPIMWrappedFunc {
 public:
  // initialize the OpenCL function.
  void Init(OpenCLModuleNode* m, ObjectPtr<Object> sptr, HBMPIMModuleNode::KTRefEntry entry,
            std::string func_name, std::vector<size_t> arg_size,
            const std::vector<std::string>& launch_param_tags) {
    w_ = m->GetGlobalWorkspace();
    m_ = m;
    sptr_ = sptr;
    entry_ = entry;
    func_name_ = func_name;
    arg_size_ = arg_size;
    launch_param_config_.Init(arg_size.size() - 2, launch_param_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    ICHECK(w_->devices.size() > 0) << "No OpenCL device";
    cl::HBMPIMWorkspace* w = dynamic_cast<cl::HBMPIMWorkspace*>(w_);
    cl::OpenCLThreadEntry* t = w_->GetThreadEntry();
    // get the kernel from thread local kernel table.
    if (entry_.kernel_id >= t->kernel_table.size()) {
      t->kernel_table.resize(entry_.kernel_id + 1);
    }
    const auto& e = t->kernel_table[entry_.kernel_id];
    cl_kernel kernel = e.kernel;
    if (kernel == nullptr || e.version != entry_.version) {
      kernel = m_->InstallKernel(w_, t, func_name_, entry_);
    }

    // setup arguments.
    for (cl_uint i = 0; i < arg_size_.size() - 2; ++i) {
      void* arg = nullptr;
      if (args.type_codes[i] == DLDataTypeCode::kDLOpaqueHandle) {
        arg = static_cast<cl::BufferDescriptor*>(void_args[i])->buffer;
      } else {
        arg = void_args[i];
      }
      OPENCL_CALL(clSetKernelArg(kernel, i, arg_size_[i], arg));
    }
    {
      // TODO[ywshin]: need check for real hardware
      int i = arg_size_.size() - 2;
      void* arg = w->GetBaseMemobj();
      OPENCL_CALL(clSetKernelArg(kernel, i, arg_size_[i], (void*)&arg));
    }
    {
      int i = arg_size_.size() - 1;
      void* arg = nullptr;
      if (func_name_ == "main_kernel") {
        ICHECK_EQ(args.type_codes[1], DLDataTypeCode::kDLOpaqueHandle);
        // void* buffer = (*static_cast<cl::BufferDescriptor**>(void_args[1]))->host_ptr;
        // arg = w->GetCrfBin(pim_library::PimOpType::OP_GEMV, w->buffer_size_map_[buffer]);
        arg = w->GetCrfBin(pim_library::PimOpType::OP_GEMV,
                           pim_library::vega20_pbi.num_grf_A *
                               pim_library::vega20_pbi.num_elem_per_grf *
                               pim_library::vega20_pbi.num_grf_B * sizeof(half_float::half));
        arg = (void*)&(static_cast<cl::BufferDescriptor*>(arg)->buffer);
      }
      OPENCL_CALL(clSetKernelArg(kernel, i, arg_size_[i], arg));
    }
    {
      int i = arg_size_.size();
      // #ifdef EMULATOR
      OPENCL_CALL(clSetKernelArg(kernel, i, sizeof(cl_mem), (void*)&w->cl_d_fmtd16_));
      OPENCL_CALL(clSetKernelArg(kernel, i + 1, sizeof(cl_mem), (void*)&w->cl_d_fmtd16_size_));
      OPENCL_CALL(clSetKernelArg(kernel, i + 2, sizeof(cl_int), (void*)&w->fmtd_size_per_ch_));
      OPENCL_CALL(clSetKernelArg(kernel, i + 3, sizeof(cl_mem), (void*)&w->cl_d_emulator_trace_));
      // #endif
    }

    cl_command_queue queue = w_->GetQueue(t->device);
    ThreadWorkLoad wl = launch_param_config_.Extract(args);
    cl_uint work_dim = static_cast<cl_uint>(launch_param_config_.work_dim());
    ICHECK_EQ(work_dim, 1);
    for (cl_uint i = 0; i < work_dim; ++i) {
      // NOTE: HBM-PIM always have 64 threads
      wl.work_size[i + 3] = 64;
      wl.work_size[i] *= wl.work_size[i + 3];
    }
    // launch kernel
    if (w_->IsProfiling(t->device)) {
      w_->GetEventQueue(t->device).resize(w_->GetEventQueue(t->device).size() + 1);
      OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, work_dim, nullptr, wl.work_size,
                                         wl.work_size + 3, 0, nullptr,
                                         &(w_->GetEventQueue(t->device).back())));
    } else {
      OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, work_dim, nullptr, wl.work_size,
                                         wl.work_size + 3, 0, nullptr, nullptr));
    }
    // #ifdef EMULATOR
    if (func_name_ == "main_kernel") {
      w->EmulatorTraceGen(pim_library::vega20_pbi.num_pim_chan, pim_library::PimOpType::OP_GEMV);
      auto pim_gemv_tmp_buffer = *static_cast<cl::BufferDescriptor**>(void_args[2]);
      auto weight = *static_cast<cl::BufferDescriptor**>(void_args[0]);
      auto output = *static_cast<cl::BufferDescriptor**>(void_args[2]);
      w->ExecuteGemmBiasAct(
          output, weight, w->h_fmtd32_, w->h_fmtd32_size_[0], pim_library::PimOpType::OP_GEMV,
          w->fragment_allocator_[t->device.device_id]->get_g_pim_base_addr(t->device.device_id),
          pim_gemv_tmp_buffer, nullptr, pim_library::PimActFunc::NONE);
    }
    // #endif
  }

 private:
  // global workspace.
  cl::OpenCLWorkspace* w_;
  // The module
  OpenCLModuleNode* m_;
  // resource handle
  ObjectPtr<Object> sptr_;
  // global kernel id in the kernel table.
  OpenCLModuleNode::KTRefEntry entry_;
  // The name of the function.
  std::string func_name_;
  // convert code for void argument
  std::vector<size_t> arg_size_;
  // launch parameters config
  LaunchParamConfig launch_param_config_;
};

void HBMPIMModuleNode::Init() {
  compile_options_ = "-DEMULATOR=1";
  OpenCLModuleNode::Init();
}

cl::OpenCLWorkspace* HBMPIMModuleNode::GetGlobalWorkspace() {
  return cl::HBMPIMWorkspace::Global();
}

PackedFunc HBMPIMModuleNode::GetFunction(const String& name,
                                         const ObjectPtr<Object>& sptr_to_self) {
  ICHECK_EQ(sptr_to_self.get(), this);
  if (name == "opencl.GetPreCompiledPrograms") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->GetPreCompiledPrograms();
    });
  } else if (name == "opencl.SetPreCompiledPrograms") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      this->SetPreCompiledPrograms(args[0]);
    });
  }

  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  FunctionInfo& info = it->second;
  HBMPIMWrappedFunc f;
  std::vector<size_t> arg_size(info.arg_types.size());
  for (size_t i = 0; i < info.arg_types.size(); ++i) {
    DLDataType t = info.arg_types[i];
    ICHECK_EQ(t.lanes, 1U);
    if (t.code == kTVMOpaqueHandle) {
      // specially store pointer type size in OpenCL driver
      arg_size[i] = sizeof(void*);
    } else {
      uint32_t bits = t.bits;
      ICHECK_EQ(bits % 8, 0U);
      arg_size[i] = bits / 8;
    }
  }
  // if (std::string(name.c_str()).compare(0, 11, "main_kernel") == 0) {
  for (int i = 0; i < 2; i++) {
    arg_size.push_back(sizeof(void*));
    info.arg_types.push_back(DataType::Handle());
  }
  // }

  // initialize the wrapped func.
  f.Init(this, sptr_to_self, kid_map_.at(name), name, arg_size, info.launch_param_tags);
  return PackFuncVoidAddr(f, info.arg_types);
}

Module HBMPIMModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source) {
  auto n = make_object<HBMPIMModuleNode>(data, fmt, fmap, source);
  n->Init();
  return Module(n);
}

}  // namespace runtime
}  // namespace tvm
