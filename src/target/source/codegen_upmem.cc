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
 * \file codegen_upmem.cc
 */
#include "codegen_upmem.h"

#include <tvm/tir/stmt_functor.h>

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

// #include "../../runtime/upmem/upmem_module.h"

#include "../../runtime/thread_storage_scope.h"
#include "../../runtime/upmem/upmem_module.h"
#include "../build_common.h"
#include "../spirv/spirv_utils.h"

namespace tvm {
namespace codegen {

class TaskletNumFinder : public StmtExprVisitor {
 public:
  int tasklet_num = 1;
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tvm::tir::attr::thread_extent) {
      const IterVarNode* iv = op->node.as<IterVarNode>();
      tvm::runtime::ThreadScope ts = tvm::runtime::ThreadScope::Create(iv->thread_tag);
      if (ts.rank == 1) {
        tasklet_num *= op->value.as<IntImmNode>()->value;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

CodeGenUpmem::CodeGenUpmem(std::string uuid) : uuid(uuid) {
  decl_stream << "#include <stdint.h>\n"
              << "#include <stdio.h>\n"
              << "#include <defs.h>\n"
              << "#include <mram.h>\n"
              << "#include <alloc.h>\n"
              << "#include <barrier.h>\n"
              << "#include <seqread.h>\n"
              << "#include <handshake.h>\n"
              << "\ntypedef struct { int32_t x, y, z; } BlockInfo;"
              << "\nBARRIER_INIT(barrier, NR_TASKLETS);\n\n"
              << "__host BlockInfo blockIdx;\n\n"
              << "inline int min(int x, int y) { return x < y ? x : y; }\n"
              << "inline int max(int x, int y) { return x > y ? x : y; }\n";
}

void CodeGenUpmem::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);

  Map<String, Array<PrimExpr>> sym_map =
      f->GetAttr<Map<String, Array<PrimExpr>>>("upmem_symbol_map").value_or({});

  for (Var arg : f->params) {
    if (sym_map.count(arg->name_hint)) {
      auto arr = sym_map[arg->name_hint];
      std::string alias = Downcast<StringImm>(arr[0])->value;
      DataType dtype = DataType(String2DLDataType(Downcast<StringImm>(arr[1])->value));
      int size = Downcast<IntImm>(arr[2])->value;
      padded_size_[arg->name_hint] = Downcast<IntImm>(arr[3])->value;

      this->stream << Downcast<StringImm>(arr[4])->value << " ";  // __mram_noinit or __mram
      PrintType(dtype, this->stream);
      this->stream << " " << alias << "[" << size << "];\n";
      var_idmap_[arg.get()] = alias;
      RegisterHandleType(arg.get(), dtype);
    } else {
      this->stream << "__host ";
      PrintType(arg.dtype(), this->stream);
      std::string vid = AllocVarID(arg.get());
      this->stream << " " << vid << ";\n";
      RegisterHandleType(arg.get(), arg->dtype);
    }
  }

  // reserve keywords
  ReserveKeywordsAsUnique();

  stream << "int main() {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenUpmem::PreFunctionBody(const PrimFunc& f) {
  int scope = this->BeginScope();
  stream << "  const int blockIdx_x = blockIdx.x;\n"
         << "  const int blockIdx_y = blockIdx.y;\n"
         << "  const int blockIdx_z = blockIdx.z;\n\n"
         << "  unsigned int tasklet_id = me();\n"
         << "  if (tasklet_id == 0) mem_reset();\n"
         << "  barrier_wait(&barrier);\n";
  this->EndScope(scope);
}

std::string CodeGenUpmem::Finish() { return CodeGenC::Finish(); }

void CodeGenUpmem::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
  std::ostringstream os;
  if (ts.rank == 1) {
    var_idmap_[iv->var.get()] = "tasklet_id";
  } else if (ts.rank == 0) {
    if (ts.dim_index == 0) var_idmap_[iv->var.get()] = "blockIdx_x";
    if (ts.dim_index == 1) var_idmap_[iv->var.get()] = "blockIdx_y";
    if (ts.dim_index == 2) var_idmap_[iv->var.get()] = "blockIdx_z";
  }
}

void CodeGenUpmem::VisitStmt_(const AllocateNode* op) {
  allocation_size_.insert({op->buffer_var.get(), op->ConstantAllocationSize() * op->dtype.lanes()});

  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

  auto scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;

  if (scope == "local") {
    PrintStorageScope(scope, stream);

    PrintType(op->dtype, stream);
    stream << "* " << vid << " = (";
    PrintType(op->dtype, stream);
    stream << "*) mem_alloc(" << constant_size << " * sizeof(";
    PrintType(op->dtype, stream);
    stream << "));\n";
  } else if (scope == "shared") {
    PrintType(op->dtype, decl_stream);
    decl_stream << " " << vid << "[" << constant_size << "];\n";
  }

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

const VarNode* GetIndexFlatVar(const PrimExpr& expr) {
  if (const VarNode* v = expr.as<VarNode>())
    return v;
  else if (const AddNode* v = expr.as<AddNode>()) {
    if (const VarNode* n = v->a.as<VarNode>()) return n;
    if (const VarNode* n = v->b.as<VarNode>()) return n;
    if (const VarNode* n = GetIndexFlatVar(v->a)) return n;
    if (const VarNode* n = GetIndexFlatVar(v->b)) return n;
  }
  return nullptr;
}

const PrimExpr GetIndexStrided(const PrimExpr& expr) {
  if (expr.as<VarNode>() || expr.as<IntImmNode>())
    return PrimExpr(0);
  else if (const AddNode* v = expr.as<AddNode>()) {
    const PrimExpr a = GetIndexStrided(v->a);
    if (is_zero(a)) return v->b;
    const PrimExpr b = GetIndexStrided(v->b);
    if (is_zero(b)) return v->a;
    return v->a + v->b;
  }
  return expr;
}

const bool isFlatEqual(const PrimExpr& a, const PrimExpr& b, std::string lvar) {
  if (const VarNode* va = GetIndexFlatVar(a)) {
    if (const VarNode* vb = GetIndexFlatVar(b)) {
      return va->name_hint == lvar && vb->name_hint == lvar;
    }
  }
  return false;
}

void CodeGenUpmem::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "shared") {
    this->PrintIndent();
    this->stream << "barrier_wait(&barrier);\n";
  } else {
    LOG(FATAL) << "Not suppored";
  }
}

void CodeGenUpmem::VisitStmt_(const ForNode* op) {
  if (op->kind == tir::ForKind::kUnrolled) {
    auto auto_max_step = Downcast<IntImm>(op->annotations.Get("auto_max_step").value_or(make_const(DataType::Int(32), 0)))->value;
    PrintIndent();
    if (auto_max_step > 0) {
      stream << "#pragma unroll(" << auto_max_step << ")\n";
    } else {
      stream << "#pragma unroll\n";
    }
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenUpmem::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (op->op.same_as(builtin::dpu_mram_read())) {
    ICHECK(is_const_int(op->args[4])) << "mram transfer size must be constant";
    int size = Downcast<IntImm>(op->args[4])->value;
    ICHECK(size % 8 == 0) << "mram transfer size must be a multiple of 8, to be aligned";
    // Current version of driver doesn't support mram_read_unaligned so just constrained it
    // Update driver will solve the problem
    this->PrintIndent();
    stream << "mram_read"
           << "((__mram_ptr void*)(" << PrintExpr(op->args[0]) << " + " << PrintExpr(op->args[1])
           << "), " << PrintExpr(op->args[2]) << " + " << PrintExpr(op->args[3]) << ", "
           << PrintExpr(op->args[4]) << ");\n";
  } else if (op->op.same_as(builtin::dpu_mram_write())) {
    ICHECK(is_const_int(op->args[4])) << "mram transfer size must be constant";
    int size = Downcast<IntImm>(op->args[4])->value;
    ICHECK(size % 8 == 0) << "mram transfer size must be a multiple of 8, to be aligned";
    this->PrintIndent();
    stream << "mram_write"
           << "(" << PrintExpr(op->args[0]) << " + " << PrintExpr(op->args[1])
           << ", (__mram_ptr void*)(" << PrintExpr(op->args[2]) << " + " << PrintExpr(op->args[3])
           << "), " << PrintExpr(op->args[4]) << ");\n";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenUpmem::PrintStorageScope(const std::string& scope, std::ostream& os) {}

std::string DPUClangCompile(const std::string& code, int tasklet_num, std::string uuid,
                            bool use_dummy = false) {  // hack
  std::string exec = "dpu-upmem-dpurte-clang";
  int valid = std::system(("command -v " + exec + " >/dev/null 2>&1").c_str());
  if (valid != 0) {
    LOG(FATAL) << "dpu-upmem-dpurte-clang not found in PATH.";
  }
  std::string output_binary = "temp-" + uuid;
  std::string flags = "-DNR_TASKLETS=" + std::to_string(tasklet_num) + " -O2 -x c";
  std::string command = exec + " " + flags + " -o " + output_binary;
  if (use_dummy) {
    command += " dummy_kernel.c";
    int result = std::system(command.c_str());
    if (result == 0) {
      return "temp";
    } else if (result == -1) {
      LOG(FATAL) << "Failed to execute pclose command.";
    } else {
      LOG(FATAL) << "Failed to compile code for upmem.";
    }
  }
  command += " -";
  FILE* pipe = popen(command.c_str(), "w");

  if (pipe) {
    fwrite(code.c_str(), 1, code.size(), pipe);
    int result = pclose(pipe);
    if (result == 0) {
      return output_binary;
    } else if (result == -1) {
      LOG(FATAL) << "Failed to execute pclose command.";
    } else {
      LOG(FATAL) << "Failed to compile code for upmem.";
    }
  } else {
    LOG(FATAL) << "Failed to pipe code into dpu-upmem-dpurte-clang";
    return "";
  }
  return "";
}

runtime::Module BuildUpmem(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;

  std::stringstream code;

  std::unordered_map<std::string, size_t> padded_buffer_size;

  auto fmap = ExtractFuncInfo(mod);

  ICHECK(mod->functions.size() == 1) << "Only one function supported for now.";
  int tasklet_num = 1;
  bool use_dummy_kernel = false;
  for (auto kv : mod->functions) {
    auto f = Downcast<PrimFunc>(kv.second);
    TaskletNumFinder tnf;
    tnf(f->body);
    tasklet_num = tnf.tasklet_num;

    code << "// Function: " << kv.first->name_hint << std::endl;
    CodeGenUpmem cg(mod->uuid);
    cg.Init(output_ssa);

    cg.AddFunction(f);
    std::string fsource = cg.Finish();
    code << fsource;

    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    use_dummy_kernel = use_dummy_kernel || f->GetAttr<Bool>("upmem_use_dummy_kernel", Bool(false)).value();
    padded_buffer_size = cg.padded_size();
  }
  VLOG(2) << code.str();

  // return runtime::Module();

  ICHECK(!mod->uuid.empty());
  DPUClangCompile(code.str(), tasklet_num, mod->uuid, use_dummy_kernel);

  return UPMEMModuleCreate(code.str(), "upmem", ExtractFuncInfo(mod), code.str(),
                           padded_buffer_size);
}

TVM_REGISTER_GLOBAL("target.build.upmem").set_body_typed(BuildUpmem);
}  // namespace codegen
}  // namespace tvm
