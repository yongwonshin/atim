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

#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <cstdio>

#include <tvm/tir/stmt_functor.h>

// #include "../../runtime/upmem/upmem_module.h"
#include "../../runtime/upmem/upmem_module.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"
#include "../spirv/spirv_utils.h"

namespace tvm {
namespace codegen {

class TaskletNumFinder: public StmtExprVisitor {
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

CodeGenUpmem::CodeGenUpmem() {}

void CodeGenUpmem::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);

  Map<String, Array<PrimExpr>> sym_map = f->GetAttr<Map<String, Array<PrimExpr>>>("upmem_symbol_map").value_or({});

  for (Var arg: f->params) {
    if (sym_map.count(arg->name_hint)) {
      auto arr = sym_map[arg->name_hint];
      std::string alias = Downcast<StringImm>(arr[0])->value;
      DataType dtype = DataType(String2DLDataType(Downcast<StringImm>(arr[1])->value));
      int size = Downcast<IntImm>(arr[2])->value;

      this->stream << "__mram_noinit ";
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
  stream << "  unsigned int tasklet_id = me();\n"
         << "  if (tasklet_id == 0) mem_reset();\n"
         << "  barrier_wait(&barrier);\n";
}

std::string CodeGenUpmem::Finish() {
  decl_stream << "#include <stdint.h>\n"
              << "#include <stdio.h>\n"
              << "#include <defs.h>\n"
              << "#include <mram.h>\n"
              << "#include <alloc.h>\n"
              << "#include <barrier.h>\n"
              << "#include <seqread.h>\n"
              << "\nBARRIER_INIT(barrier, NR_TASKLETS);\n";
  return CodeGenC::Finish();
}

void CodeGenUpmem::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
  std::ostringstream os;
  if (ts.rank == 1) {
    var_idmap_[iv->var.get()] = "tasklet_id";
  } else {
    var_idmap_[iv->var.get()] = "0"; // TODO
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
  PrintStorageScope(scope, stream);

  PrintType(op->dtype, stream);
  stream << "* " << vid << " = (";
  PrintType(op->dtype, stream);
  stream << "*) mem_alloc(" << constant_size << " * sizeof(";
  PrintType(op->dtype, stream);
  stream << "));\n";

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

const VarNode* GetIndexFlatVar(const PrimExpr &expr) {
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

const PrimExpr GetIndexStrided(const PrimExpr &expr) {
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

const bool isFlatEqual(const PrimExpr &a, const PrimExpr &b, std::string lvar) {
  if (const VarNode* va = GetIndexFlatVar(a)) {
    if (const VarNode* vb = GetIndexFlatVar(b)) {
      return va->name_hint == lvar && vb->name_hint == lvar;
    }
  }
  return false;
}

void CodeGenUpmem::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {
  ICHECK_EQ(op->indices.size(), 1) << "Load from non-flat memory not supported.";

  DataType value_dtype = op->dtype;
  DataType element_dtype = op->buffer->dtype;
  ICHECK_EQ(value_dtype.lanes(), element_dtype.lanes()) << "Vectorization not supported.";
  std::string scope = GetPtrStorageScope(op->buffer->data);

  PrimExpr index;
  if (alloc_global_index.defined()) {
    ICHECK(scope == "" || scope == "global") << "In local<-global pattern, BufferLoad scope should be global.";
    index = alloc_global_index;
  } else {
    ICHECK(scope == "local") << "BufferLoad scope should be local, except in local<-global pattern.";
    index = op->indices[0];
  }

  Var buffer_var = op->buffer->data;
  std::string ref = GetBufferRef(op->dtype, op->buffer.get(), index);
  HandleVolatileLoads(ref, op, os);
}

void CodeGenUpmem::VisitStmt_(const BufferStoreNode* op) {
  ICHECK_EQ(op->indices.size(), 1) << "Store to non-flat memory not supported.";

  DataType value_dtype = op->value.dtype();
  DataType element_dtype = op->buffer->dtype;
  ICHECK_EQ(value_dtype.lanes(), element_dtype.lanes()) << "Vectorization not supported.";

  PrimExpr index_expr = op->indices[0];

  // Most store<-load pattern should be intercepted in ForNode visit, using mram_read or mram_write.
  // TODO: move mram_read/mram_write into LowerIntrinsic

  if (const BufferLoadNode* load = op->value.as<BufferLoadNode>()) {
    std::string lscope = GetPtrStorageScope(load->buffer->data);
    std::string sscope = GetPtrStorageScope(op->buffer->data);
    ICHECK(sscope == "local" || lscope == "local") << "Either source or destination must be local.";
    if (sscope == "local" && (lscope == "" || lscope == "global")) { // local <- global
      ICHECK(op->global_indices.size() == 1) << "In local->global pattern, BufferStore global_indices should be size 1.";
      alloc_global_index = op->global_indices[0];
    }
    if (lscope == "local" && (sscope == "" || sscope == "global")) { // global <- local
      ICHECK(load->global_indices.size() == 1) << "In global->local pattern, BufferLoad global_indices should be size 1.";
      index_expr = load->global_indices[0];
    }
  }
  std::string value = PrintExpr(op->value);
  alloc_global_index = PrimExpr();

  Var buffer_var = op->buffer->data;
  // stream << "// " << value_dtype;
  std::string ref = this->GetBufferRef(value_dtype, op->buffer.get(), index_expr);
  this->PrintIndent();
  stream << ref << " = " << value << ";\n";
}

void CodeGenUpmem::VisitStmt_(const ForNode* op) {
    PrintIndent();
    std::string lvar = op->loop_var->name_hint;
    ICHECK(is_zero(op->min));

    if (const BufferStoreNode* store = op->body.as<BufferStoreNode>()) {
      if (const BufferLoadNode* load = store->value.as<BufferLoadNode>()) {
        if (isFlatEqual(store->indices[0], load->indices[0], lvar)) {
          // mram_ pattern
          std::string lscope = GetPtrStorageScope(load->buffer->data);
          std::string sscope = GetPtrStorageScope(store->buffer->data);
          std::string sid = GetVarID(store->buffer->data.get());
          std::string lid = GetVarID(load->buffer->data.get());

          std::stringstream size_stream;
          size_stream << op->extent << " * " << "sizeof(";
          PrintType(load->buffer->dtype, size_stream);
          size_stream << ")";
          std::string size = size_stream.str();

          if (sscope == "local" && (lscope  == ""|| lscope == "global")) {
            ICHECK(store->global_indices.size() == 1) << "In local->global pattern, BufferStore global_indices should be size 1.";
            std::string l_ptr = lid + " + " + PrintExpr(GetIndexStrided(store->global_indices[0]));
            std::string s_ptr = sid + " + " + PrintExpr(GetIndexStrided(store->indices[0])); 
            stream << "mram_read((__mram_ptr void*)(" << l_ptr << "), " << s_ptr << ", " << size << ");\n";
            return;
          }
          else if ((sscope == "" || sscope == "global") && lscope == "local") {
            std::string l_ptr = lid + " + " + PrintExpr(GetIndexStrided(load->indices[0]));
            std::string s_ptr = sid + " + " + PrintExpr(GetIndexStrided(load->global_indices[0]));
            ICHECK(load->global_indices.size() == 1) << "In global->local pattern, BufferLoad global_indices should be size 1.";
            stream << "mram_write(" << l_ptr << ", (__mram_ptr void*)(" << s_ptr << "), " << size << ");\n";
            return;
          }
        }
      }
    }

    std::string extent = PrintExpr(op->extent);
    std::string vid = AllocVarID(op->loop_var.get());
    stream << "for (";
    PrintType(op->loop_var.dtype(), stream);
    stream << ' ' << vid << " = 0; " << vid << " < " << extent << "; ++" << vid << ") {\n";
    int for_scope = BeginScope();
    for_tags.push(vid);
    PrintStmt(op->body);
    // check if body is mram_read or mram_write
    for_tags.pop();
    this->EndScope(for_scope);
    PrintIndent();
    stream << "}\n";
}

void CodeGenUpmem::PrintStorageScope(const std::string& scope, std::ostream& os) {
}

std::string DPUClangCompile(const std::string& code, int tasklet_num, bool use_dummy = false) { // hack
  std::string exec = "dpu-upmem-dpurte-clang";
  int valid = std::system(("command -v " + exec + " >/dev/null 2>&1").c_str());
  if (valid != 0) {
    LOG(FATAL) << "dpu-upmem-dpurte-clang not found in PATH.";
  }
  std::string output_binary = "temp";
  std::string flags = "-DNR_TASKLETS=" + std::to_string(tasklet_num) + " -O3 -x c";
  std::string command = exec + " " + flags + " -o " + output_binary;
  if (use_dummy) {
    command += " dummy_kernel.c";
    std::system(command.c_str());
    return "temp";
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
  // const auto* fpostproc = Registry::Get("tvm_callback_upmem_postproc");
  ICHECK(mod->functions.size() == 1) << "Only one function supported for now.";
  int tasklet_num = 1;
  for (auto kv : mod->functions) {
    auto f = Downcast<PrimFunc>(kv.second);
    TaskletNumFinder tnf;
    tnf(f->body);
    tasklet_num = tnf.tasklet_num;

    code << "// Function: " << kv.first->name_hint << std::endl;
    CodeGenUpmem cg;
    cg.Init(output_ssa);
    
    cg.AddFunction(f);
    std::string fsource = cg.Finish();
    code << fsource;
  }
  VLOG(2) << code.str();

  // return runtime::Module();

  DPUClangCompile(code.str(), tasklet_num);

  return UPMEMModuleCreate("kernel", "upmem", ExtractFuncInfo(mod), code.str());
}

TVM_REGISTER_GLOBAL("target.build.upmem").set_body_typed(BuildUpmem);
}  // namespace codegen
}  // namespace tvm
