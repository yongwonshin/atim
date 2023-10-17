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

// #include "../../runtime/upmem/upmem_module.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"
#include "../spirv/spirv_utils.h"

namespace tvm {
namespace codegen {

CodeGenUpmem::CodeGenUpmem() {}

void CodeGenUpmem::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);

  for (Var arg : f->params) {
    this->stream << "__host ";
    PrintType(arg.dtype(), this->stream);
    std::string vid = AllocVarID(arg.get());
    this->stream << " " << vid << ";\n";
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
    return PrimExpr();
  else if (const AddNode* v = expr.as<AddNode>()) {
    const PrimExpr a = GetIndexStrided(v->a);
    if (!a.defined()) return v->b;
    const PrimExpr b = GetIndexStrided(v->b);
    if (!b.defined()) return v->a;
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

void CodeGenUpmem::VisitStmt_(const ForNode* op) {
    PrintIndent();
    std::string lvar = op->loop_var->name_hint;
    ICHECK(is_zero(op->min));
    if (const BufferStoreNode* store = op->body.as<BufferStoreNode>()) {
      if (const BufferLoadNode* load = store->value.as<BufferLoadNode>()) {
        if (isFlatEqual(store->indices[0], load->indices[0], lvar)) {
          std::string lscope = GetPtrStorageScope(load->buffer->data);
          std::string sscope = GetPtrStorageScope(store->buffer->data);
          std::string sid = GetVarID(store->buffer->data.get());
          std::string lid = GetVarID(load->buffer->data.get());

          std::stringstream size_stream;
          size_stream << "sizeof(";
          PrintType(load->buffer->dtype, size_stream);
          size_stream << ")";
          std::string size = size_stream.str();

          if (sscope == "local" && (lscope  == ""|| lscope == "global")) {
            stream << "mram_read((__mram_ptr void const*)(" << lid << " + (" << PrintExpr(GetIndexStrided(load->indices[0]));
            stream << ") * " <<  size << "), " << sid << ", " << op->extent << " * " << size << ");\n";
            return;
          }
          else if ((sscope == "" || sscope == "global") && lscope == "local") {
            stream << "mram_write(" << lid << ", (__mram_ptr void*)(" << sid << " + (" << PrintExpr(GetIndexStrided(store->indices[0]));
            stream << ") * " << size << "), " << op->extent << " * " << size << ");\n";
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

void CodeGenUpmem::VisitStmt_(const BufferStoreNode* op) {
  CodeGenC::VisitStmt_(op);
}

void CodeGenUpmem::VisitExpr_(const MulNode* op, std::ostream& os) {
  std::string pa = PrintExpr(op->a);
  std::string pb = PrintExpr(op->b);
  if (pa == "0" || pb == "0") os << "0";
  else {
    os << "(" << pa << " * " << pb << ")";
  }
}

void CodeGenUpmem::VisitExpr_(const AddNode* op, std::ostream& os) {
  std::string pa = PrintExpr(op->a);
  std::string pb = PrintExpr(op->b);
  if (pa == "0") os << pb;
  else if (pb == "0") os << pa;
  else {
    os << "(" << pa << " + " << pb << ")";
  }
}

runtime::Module BuildUpmem(IRModule mod, Target target) {

  using tvm::runtime::Registry;
  bool output_ssa = false;

  std::stringstream code;
  // const auto* fpostproc = Registry::Get("tvm_callback_upmem_postproc");
  for (auto kv : mod->functions) {
    code << "// Function: " << kv.first->name_hint << std::endl;
    CodeGenUpmem cg;
    cg.Init(output_ssa);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(f);
    std::string fsource = cg.Finish();

    VLOG(2) << fsource;
    // if (fpostproc) {
    //   fsource = (*fpostproc)(fsource, target).operator std::string();
    // }
    code << fsource;
  }

  return runtime::Module();
}

TVM_REGISTER_GLOBAL("target.build.upmem").set_body_typed(BuildUpmem);
}  // namespace codegen
}  // namespace tvm
