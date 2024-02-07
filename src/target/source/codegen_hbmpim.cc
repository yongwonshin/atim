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

#include "../../runtime/opencl/hbmpim/hbmpim_module.h"
#include "../../runtime/opencl/opencl_module.h"
#include "../../runtime/texture.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"
#include "../spirv/spirv_utils.h"
#include "literal/hbmpim_utils.h"

namespace tvm {
namespace codegen {

CodeGenHBMPIM::CodeGenHBMPIM() { thread_vars_ = {IterVar(), IterVar(), IterVar()}; }

std::ostringstream& CodeGenHBMPIM::Stream() {
  PrintIndent();
  return stream;
}

class ThreadReindexer : public StmtExprMutator {
  Stmt VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag.length() != 0) {
        runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
        if (ts.rank == 1) {
          if (thread_vars_.size() <= ts.dim_index) {
            thread_vars_.resize(ts.dim_index + 1);
          }
          thread_vars_.Set(ts.dim_index, iv);
        } else if (ts.rank == 2) {
          active_ = false;
        } else if (ts.rank == 3) {
          active_ = true;
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  PrimExpr VisitExpr_(const BufferLoadNode* _op) {
    if (_op->buffer->name == "C_rf_internal") {  // TODO[ywshin]
      active2_ = true;
    }
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
    if (_op->buffer->name == "C_rf_internal") {  // TODO[ywshin]
      active2_ = false;
    }
    return std::move(load);
  }
  PrimExpr VisitExpr_(const VarNode* op) {
    Var v = GetRef<Var>(op);
    if (active_ && active2_) {
      for (auto tv : thread_vars_) {
        if (v.get() == tv->var.get()) {
          // std::cerr << tvm::truncmod(v, tv->dom->extent) << std::endl;
          return tvm::truncmod(v, tv->dom->extent);
        }
      }
    }
    return std::move(v);
  }
  Array<IterVar> thread_vars_;
  bool active_ = false;
  bool active2_ = false;
};

class BankIndexInspector : public ExprVisitor {
 public:
  explicit BankIndexInspector(Map<Var, IntImm> bank_ordering_map, Map<Var, IntImm> bank_extent_map,
                              Array<IterVar> thread_vars)
      : bank_ordering_map_(bank_ordering_map),
        bank_extent_map_(bank_extent_map),
        thread_vars_(thread_vars) {}
  PrimExpr Inspect(PrimExpr index) {
    bank_index_ = IntImm(DataType::Int(32), 0);
    ExprVisitor::VisitExpr(index);
    return bank_index_;
  }

 private:
  PrimExpr BankOrderingToBankIndex(PrimExpr bank_var, IntImm ordering) {
    if (ordering->value == 1) {
      return (thread_vars_[0] / thread_vars_[0]->dom->extent);
    } else if (ordering->value == 2) {
      return (thread_vars_[0] / thread_vars_[0]->dom->extent) * 2;
    } else if (ordering->value == 3) {
      return (thread_vars_[0] / thread_vars_[0]->dom->extent) * 2 + 1;
    } else {
      LOG(FATAL) << "Unknown bank ordering " << ordering->value << "!";
    }
  }
  void VisitExpr_(const VarNode* op) {
    Var v = GetRef<Var>(op);
    if (bank_ordering_map_.find(v) != bank_ordering_map_.end()) {
      bank_index_ = BankOrderingToBankIndex(v, bank_ordering_map_[v]);
    }
    // if (v.get() == thread_vars_[0]->var.get()) {
    //   std::cerr << "MATCH1!!!" << std::endl;
    // }
  }
  PrimExpr bank_index_ = IntImm(DataType::Int(32), 0);
  Map<Var, IntImm> bank_ordering_map_;
  Map<Var, IntImm> bank_extent_map_;
  Array<IterVar> thread_vars_;
};

void CodeGenHBMPIM::PreFunctionBody(const PrimFunc& f) {
  CodeGenOpenCL::PreFunctionBody(f);
  pim_scope_ = this->BeginScope();
  // stream << "#ifdef EMPTY_BODY\n";
  stream << "#ifdef EMULATOR\n";
  Stream() << "emulator_trace->g_fba = (ulong)pim_ctr;\n";
  Stream() << "emulator_trace->g_fmtd16 = fmtd16;\n";
  Stream() << "emulator_trace->g_ridx[get_group_id(0)] = 0;\n";
  Stream() << "emulator_trace->m_width = mt_width;\n";
  Stream() << "barrier(CLK_LOCAL_MEM_FENCE);\n";
  stream << "#endif\n";

  // stream << "#ifdef PREPARE_KERNEL\n";
  Stream() << "int num_ba = 4;\n";
  Stream() << "int w_idx = get_local_id(0) % 2;\n";
  Stream() << "int gidx = get_local_id(0) >> 1;\n";
  Stream() << "ulong offset = w_idx << 4;\n";
  // temp
  Stream() << "int grf_shift = 3;\n";
  Stream() << "int ba_shift = 2;\n";
  Stream() << "int num_col = 32;\n";
  Stream() << "int col_shift = 5;\n";
  Stream() << "int trans_shift = 5;\n";
  Stream() << "int even_row, odd_row, row, col, loc;\n";
  Stream() << "int ch = get_group_id(0);\n";
  Stream() << "int teidx = get_local_id(0) * 16;\n";
  Stream() << "ulong addr;\n";
  // stream << "#endif\n";
  this->EndScope(pim_scope_);
}

void CodeGenHBMPIM::PostFunctionBody(const PrimFunc& f) {
  pim_scope_ = this->BeginScope();
  stream << "#ifdef EMULATOR\n";
  Stream() << "if (get_group_id(0) == 0 && get_local_id(0) == 0) {\n";
  PrintIndent();
  Stream() << "frd_size[0] = emulator_trace->g_ridx[0];\n";
  Stream() << "}\n";
  stream << "#endif\n";
  // stream << "#endif\n";
  this->EndScope(pim_scope_);
}

void CodeGenHBMPIM::PrintChangeGemvHabHabPim() {
  Stream() << "change_gemv_hab_habpim(pim_ctr, offset);\n";
}

void CodeGenHBMPIM::PrintChangeGemvHabPimHab() {
  Stream() << "change_habpim_hab(pim_ctr, offset);\n";
}

void CodeGenHBMPIM::PrintPIMPrologue() {
  // park in
  // stream << "#if PARK_IN\n";
  Stream() << "if (get_local_id(0) < 32) {\n";
  PrintIndent();
  Stream() << "park_in(pim_ctr, gidx, num_ba, offset);\n";
  Stream() << "}\n";
  // stream << "#endif\n";

  // change SB mode to HAB mode
  // stream << "#if CHANGE_SB_HAB\n";
  Stream() << "if (get_local_id(0) < 2) {\n";
  PrintIndent();
  Stream() << "change_sb_hab(pim_ctr, offset);\n";
  Stream() << "}\n";
  Stream() << "barrier(CLK_GLOBAL_MEM_FENCE);\n";
  // stream << "#endif\n";

  // program CRF
  // stream << "#if PROGRAM_CRF\n";
  Stream() << "if (get_local_id(0) < (" << crf_size_ << " >> 4)) {\n";
  PrintIndent();
  Stream() << "program_crf_mod(pim_ctr, gidx, crf_binary, offset);\n";
  Stream() << "}\n";
  Stream() << "barrier(CLK_GLOBAL_MEM_FENCE);\n";
  // stream << "#endif\n";

  // Limit threads
  // stream << "#if COMPUTE_GEMM\n";
  Stream() << "if (get_local_id(0) < 16) {\n";
  pim_scope_ = this->BeginScope();
}

void CodeGenHBMPIM::PrintExtraFuncParams(const PrimFunc& f) {
  stream << ", __global uchar* __restrict__ pim_ctr";
  stream << ", __global uchar* crf_binary";
  stream << "\n#ifdef EMULATOR\n";
  stream << ", __global PimMemTraceData* fmtd16"
         << ", __global size_t* frd_size"
         << ", int mt_width"
         << ", __global PimMemTracer* emulator_trace";
  stream << "\n#endif\n";
}

void CodeGenHBMPIM::PrintPIMEpilogue() {
  // change HAB_PIM mode to HAB_PIM mode
  this->EndScope(pim_scope_);
  Stream() << "}\n";
  // stream << "#endif\n";

  // stream << "#if CHANGE_HAB_SB\n";
  Stream() << "if (get_local_id(0) < 4) {\n";
  PrintIndent();
  Stream() << "change_hab_sb(pim_ctr, gidx, offset);\n";
  Stream() << "}\n";
  Stream() << "barrier(CLK_GLOBAL_MEM_FENCE);\n";
  // stream << "#endif\n";

  // stream << "#if PARK_OUT\n";
  Stream() << "if (get_local_id(0) < 32) {\n";
  PrintIndent();
  Stream() << "park_out(pim_ctr, gidx, num_ba, offset);\n";
  Stream() << "}\n";
  // stream << "#endif\n";
}

void CodeGenHBMPIM::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::bank) {
    Var v = Downcast<Var>(op->node);
    IntImm bank_ordering = Downcast<IntImm>(op->value);
    bank_ordering_map_.Set(v, bank_ordering);
    this->VisitStmt(op->body);
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenHBMPIM::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (op->op.same_as(builtin::R_CMD())) {
    std::string ptr = Downcast<StringImm>(op->args[1])->value;
    if (ptr.empty()) {
      os << "R_CMD(";
      this->PrintExpr(op->args[0], os);
      os << ")";
    } else {
      os << "R_CMD(&" << ptr << "[";
      this->PrintExpr(op->args[0], os);
      os << "])";
    }
  } else if (op->op.same_as(builtin::W_CMD())) {
    std::string ptr = Downcast<StringImm>(op->args[1])->value;
    if (ptr.empty()) {
      os << "W_CMD(";
      this->PrintExpr(op->args[0], os);
      os << ")";
    } else {
      os << "W_CMD(&" << ptr << "[";
      this->PrintExpr(op->args[0], os);
      os << "])";
    }
  } else if (op->op.same_as(builtin::W_CMD_R())) {
    std::string ptr = Downcast<StringImm>(op->args[2])->value;
    if (ptr.empty()) {
      os << "W_CMD_R(";
      this->PrintExpr(op->args[0], os);
      os << ", ";
      this->PrintExpr(op->args[1], os);
      os << ")";
    } else {
      os << "W_CMD_R(&" << ptr << "[";
      this->PrintExpr(op->args[0], os);
      os << "], ";
      this->PrintExpr(op->args[1], os);
      os << ")";
    }
  } else if (op->op.same_as(builtin::W_CMD_R_C())) {
    std::string ptr = Downcast<StringImm>(op->args[2])->value;
    if (ptr.empty()) {
      os << "W_CMD_R_C(";
      this->PrintExpr(op->args[0], os);
      os << ", ";
      this->PrintExpr(op->args[1], os);
      os << ")";
    } else {
      os << "W_CMD_R(&" << ptr << "[";
      this->PrintExpr(op->args[0], os);
      os << "], ";
      this->PrintExpr(op->args[1], os);
      os << ")";
    }
  } else if (op->op.same_as(builtin::B_CMD())) {
    int64_t type = Downcast<IntImm>(op->args[0])->value;
    os << "B_CMD(" << type << ")";
  } else if (op->op.same_as(builtin::vloadn())) {
    int64_t offset = Downcast<IntImm>(op->args[0])->value;
    std::string ptr = Downcast<StringImm>(op->args[2])->value;
    ICHECK(op->args.size() == 3U);
    ICHECK(op->dtype == DataType::Int(32).with_lanes(4));
    if (ptr.empty()) {
      os << "vload4(" << offset << ", (__global float*)";
      this->PrintExpr(op->args[1], os);
      os << ")";
    } else {
      os << "vload4(" << offset << ", &pim_ctr[";
      this->PrintExpr(op->args[1], os);
      os << "])";
    }
  } else if (op->op.same_as(builtin::vstoren())) {
    int64_t offset = Downcast<IntImm>(op->args[1])->value;
    std::string ptr = Downcast<StringImm>(op->args[3])->value;
    ICHECK(op->args.size() == 4U);
    ICHECK(op->dtype == DataType::Int(32).with_lanes(4));
    if (ptr.empty()) {
      os << "vstore4(";
      this->PrintExpr(op->args[0], os);
      os << ", " << offset << ", (__global float*)";
      this->PrintExpr(op->args[2], os);
      os << ")";
    } else {
      os << "vstore4(";
      this->PrintExpr(op->args[0], os);
      os << ", " << offset << ", &pim_ctr[";
      this->PrintExpr(op->args[2], os);
      os << "])";
    }
  } else if (op->op.same_as(builtin::barrier())) {
    os << "barrier(CLK_LOCAL_MEM_FENCE)";
  } else if (op->op.same_as(builtin::mem_fence())) {
    os << "mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE)";
  } else if (op->op.same_as(builtin::addr_gen())) {
    os << "addr_gen_s(";
    this->PrintExpr(op->args[0], os);
    os << ", ";
    this->PrintExpr(op->args[1], os);
    os << ", ";
    this->PrintExpr(op->args[2], os);
    os << ", ";
    this->PrintExpr(op->args[3], os);
    os << ", ";
    this->PrintExpr(op->args[4], os);
    os << ", ";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[6], os);
    os << ")";
  } else {
    CodeGenOpenCL::VisitExpr_(op, os);
  }
}

void CodeGenHBMPIM::VisitStmt_(const BufferStoreNode* op) {
  if (op->buffer->name == "C_rf_internal") {  // TODO[ywshin]
    LOG(FATAL) << "Internal buffer store is NOT implemented: " << op->buffer << "!";
  }
  CodeGenC::VisitStmt_(op);
}

// Print a reference expression to a buffer.
std::string CodeGenHBMPIM::GetBufferRef(DataType t, const BufferNode* buffer, std::string index) {
  const VarNode* buffer_var = buffer->data.get();
  std::ostringstream os;
  std::string vid = GetVarID(buffer_var);
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  bool is_vol = IsVolatile(buffer_var);

  auto ptr_cast = [this, is_vol, scope](DataType pointed_to) {
    std::ostringstream ptr_os;
    ptr_os << "(";
    if (is_vol) {
      ptr_os << "volatile ";
    }
    if (!scope.empty() && IsScopePartOfType()) {
      PrintStorageScope(scope, ptr_os);
    }
    PrintType(pointed_to, ptr_os);
    ptr_os << "*)";
    return ptr_os.str();
  };

  DataType buffer_element_dtype = buffer->dtype;

  std::string buffer_str = vid;
  if (!HandleTypeMatch(buffer_var, buffer_element_dtype) || is_vol) {
    std::stringstream temp;
    temp << "(" << ptr_cast(buffer_element_dtype) << vid << ")";
    buffer_str = temp.str();
  }

  std::string index_str = index;
  if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
    // This is a special case, because CodegenCUDA::PrintType()
    // returns "int" for bool and for 4-bit integers. In most cases,
    // we divide by the number of lanes to determine the index.
    // However, the backing type for scalar int4 and scalar bool is
    // int32.  Therefore, we need to divide by the ratio of their
    // sizes in that case.
    int div_factor = (t.lanes() == 1) ? (32 / t.bits()) : t.lanes();

    os << "*("
       << "(" << ptr_cast(t) << vid << ")"
       << " + " << index_str << " / " << div_factor << ")";
  } else if (t == buffer_element_dtype) {
    os << buffer_str << "[" << index_str << "]";
  } else {
    os << "*" << ptr_cast(t) << "(" << buffer_str << " + " << index_str << ")";
  }

  return os.str();
}

void CodeGenHBMPIM::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {
  if (op->buffer->name == "C_rf_internal") {  // TODO[ywshin]
    BankIndexInspector inspector(bank_ordering_map_, bank_extent_map_, thread_vars_);
    PrimExpr bank_index = inspector.Inspect(op->indices[0]);
    PrimExpr offset = op->global_indices[0];
    std::stringstream ss;
    ss << "addr_gen_s(get_group_id(0), 0, ";
    this->PrintExpr(tvm::truncdiv(bank_index, 4), ss);
    ss << ", ";
    this->PrintExpr(tvm::truncmod(bank_index, 4), ss);
    ss << ", 0, 0, ";
    this->PrintExpr(offset * 2, ss);
    ss << ") >> 1";

    std::string ref = GetBufferRef(op->dtype, op->buffer.get(), ss.str());
    HandleVolatileLoads(ref, op, os);
  } else {
    // std::cerr << "[codegen] " << op->buffer->name << "(Load): " << op->global_indices <<
    // std::endl;
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenHBMPIM::VisitStmt_(const ForNode* op) {
  // if (op->annotations.find("bank") != op->annotations.end()) {
  //   std::cerr << op->loop_var << std::endl;
  // }

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
  if (op->annotations.Get("change_pim_mode").as<Bool>()) {
    PrintChangeGemvHabHabPim();
  }
  PrintStmt(op->body);
  if (op->annotations.Get("change_pim_mode").as<Bool>()) {
    PrintChangeGemvHabPimHab();
  }
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
  if (op->annotations.Get("barrier").as<Bool>()) {
    PrintIndent();
    stream << "B_CMD(1);\n";
  }
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
    thread_vars_.Set(ts.dim_index, iv);
  } else if (ts.rank == 2) {
    os << 0;
  } else if (ts.rank == 3) {
    bank_extent_map_.Set(iv->var, Downcast<IntImm>(iv->dom->extent));
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
    return runtime::HBMPIMModuleCreate(smap, spirv_text, ExtractFuncInfo(mod));
  }
#endif

  using tvm::runtime::Registry;
  bool output_ssa = false;

  std::stringstream code;
  const auto* fpostproc = Registry::Get("tvm_callback_opencl_postproc");
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenHBMPIM: Can only take PrimFunc";
    code << "// Function: " << kv.first->name_hint << std::endl;
    code << _hbmpim_info_def;
    code << _hbmpim_kernel_utils_def;
    CodeGenHBMPIM cg;
    cg.Init(output_ssa);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenHBMPIM: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    PrimFuncNode* fptr = f.CopyOnWrite();
    ThreadReindexer reindexer;
    fptr->body = std::move(reindexer(f->body));
    cg.AddFunction(f);
    std::string fsource = cg.Finish();
    if (fpostproc) {
      fsource = (*fpostproc)(fsource, target).operator std::string();
    }
    code << fsource;
  }

  return HBMPIMModuleCreate(code.str(), "pclbin", ExtractFuncInfo(mod), code.str());
}

TVM_REGISTER_GLOBAL("target.build.hbmpim").set_body_typed(BuildHBMPIM);
}  // namespace codegen
}  // namespace tvm
