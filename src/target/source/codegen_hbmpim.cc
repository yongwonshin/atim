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
 * \file codegen_hbmpim.cc
 */
#include "codegen_hbmpim.h"

#include <cmath>
#include <string>
#include <vector>

#include "../../runtime/opencl/opencl_module.h"
#include "../../runtime/texture.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"
#include "../spirv/spirv_utils.h"

namespace tvm {
namespace codegen {

std::ostringstream& CodeGenHBMPIM::Stream() {
  PrintIndent();
  return stream;
}

void CodeGenHBMPIM::PrintPIMPrologue() {
  // park in
  Stream() << "if (get_local_id(0) < 32) {\n";
  Stream() << "  // park_in(pim_ctr, gidx, num_ba, offset)\n";
  Stream() << "}\n";

  // change SB mode to HAB mode
  Stream() << "if (get_local_id(0) < 2) {\n";
  Stream() << "  // change_sb_hab(pim_ctr, offset);\n";
  Stream() << "}\n";
  Stream() << "barrier(CLK_GLOBAL_MEM_FENCE);\n";

  // program CRF
  Stream() << "if (get_local_id(0) < (" << crf_size_ << " >> 4)) {\n";
  Stream() << "  // program_crf_mod(pim_ctr, gidx, crf_binary, offset);\n";
  Stream() << "}\n";
  Stream() << "barrier(CLK_GLOBAL_MEM_FENCE);\n";

  // change HAB mode to HAB_PIM mode
  Stream() << "// change_hab_habpim(pim_ctr, offset);\n";

  // Limit threads
  Stream() << "if (get_local_id(0) < 16) {\n";
  pim_scope_ = this->BeginScope();
}

void CodeGenHBMPIM::PrintPIMEpilogue() {
  // change HAB_PIM mode to HAB_PIM mode
  Stream() << "// change_habpim_hab(pim_ctr, offset);\n";
  this->EndScope(pim_scope_);
  Stream() << "}\n";
}

void CodeGenHBMPIM::VisitStmt_(const BufferStoreNode* op) {
  // std::cerr << "[codegen] " << op->buffer->name << "(Store): " << op->global_indices <<
  // std::endl;
  CodeGenC::VisitStmt_(op);
}

void CodeGenHBMPIM::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {
  // std::cerr << "[codegen] " << op->buffer->name << "(Load): " << op->global_indices << std::endl;
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenHBMPIM::VisitStmt_(const ForNode* op) {
  std::string extent = PrintExpr(op->extent);
  if (op->annotations.Get("pim").as<Bool>()) {
    PrintPIMPrologue();
  }
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  ICHECK(is_zero(op->min));
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = 0; " << vid << " < " << extent << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
  if (op->annotations.Get("pim").as<Bool>()) {
    PrintPIMEpilogue();
  }
}

void CodeGenHBMPIM::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
  std::ostringstream os;
  if (ts.rank == 1) {
    os << "get_local_id(" << ts.dim_index << ")";
  } else if (ts.rank == 2) {
    os << 0;
  } else {
    os << "get_group_id(" << ts.dim_index << ")";
  }
  var_idmap_[iv->var.get()] = CastFromTo(os.str(), DataType::UInt(64), iv->var.dtype());
}

runtime::Module BuildHBMPIM(IRModule mod, Target target) {
#if TVM_ENABLE_SPIRV
  Optional<String> device = target->GetAttr<String>("device");
  if (device && device.value() == "spirv") {
    auto [smap, spirv_text] = LowerToSPIRV(mod, target);
    return runtime::OpenCLModuleCreate(smap, spirv_text, ExtractFuncInfo(mod));
  }
#endif

  using tvm::runtime::Registry;
  bool output_ssa = false;

  std::stringstream code;
  const auto* fpostproc = Registry::Get("tvm_callback_opencl_postproc");
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenHBMPIM: Can only take PrimFunc";
    code << "// Function: " << kv.first->name_hint << std::endl;
    CodeGenHBMPIM cg;
    cg.Init(output_ssa);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenHBMPIM: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
    std::string fsource = cg.Finish();
    if (fpostproc) {
      fsource = (*fpostproc)(fsource, target).operator std::string();
    }
    code << fsource;
  }

  return OpenCLModuleCreate(code.str(), "cl", ExtractFuncInfo(mod), code.str());
}

TVM_REGISTER_GLOBAL("target.build.hbmpim").set_body_typed(BuildHBMPIM);
}  // namespace codegen
}  // namespace tvm
