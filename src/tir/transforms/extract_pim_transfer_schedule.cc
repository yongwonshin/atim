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
 * \file transfer_schedule.cc
 * \brief Extract Transfer Schedule for PIM API
*/

#include <tvm/runtime/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/builtin.h>
#include <tvm/arith/analyzer.h>
#include <tvm/arith/bound.h>
#include <tvm/tir/transform.h>
#include <tvm/arith/analyzer.h>

#include "ir_utils.h"
#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"
#include "remove_no_op.h"

namespace tvm {
namespace tir {

class FindCopyCandidates: public StmtExprVisitor {
public:
  std::vector<Buffer> h2d, d2h;
  bool inside_kernel = false;

  explicit FindCopyCandidates() { }

  void VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == tvm::tir::attr::thread_extent) {
      const IterVarNode* iv = op->node.as<IterVarNode>();
      ICHECK(iv);
      Var var = iv->var;
      runtime::ThreadScope scope = runtime::ThreadScope::Create(iv->thread_tag);
      bool prev_inside_kernel = false;
      if (scope.rank == 0) { ;
        prev_inside_kernel = inside_kernel;
        inside_kernel = true;
      }
      StmtExprVisitor::VisitStmt_(op);
      if (scope.rank == 0) {
        inside_kernel = prev_inside_kernel;
      }
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const AllocateNode* op) {
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) {
    PrimExpr host_index, in_bank_index;
    Buffer buffer;
    if (inside_kernel) {
      if (const BufferLoadNode* load = op->value.as<BufferLoadNode>()) {
        std::string lscope = GetPtrStorageScope(load->buffer->data);
        std::string sscope = GetPtrStorageScope(op->buffer->data);
        ICHECK(sscope == "local" || lscope == "local") << "Either source or destination must be local.";
        if (sscope == "local" && (lscope == "" || lscope == "global")) { // local <- global: h->d pattern
          ICHECK(op->global_indices.size() == 1) << "In local->global pattern, BufferStore global_indices should be size 1.";
          h2d.push_back(load->buffer);
        }
        if (lscope == "local" && (sscope == "" || sscope == "global")) { // global <- local: d->h pattern
          ICHECK(load->global_indices.size() == 1) << "In global->local pattern, BufferLoad global_indices should be size 1.";
          d2h.push_back(op->buffer);
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

class ScheduleExtractor: public StmtExprMutator {
public:
  const Buffer& target_buffer_;
  bool found = false;
  bool inside_kernel = false;
  PrimExpr bank_index = 0;

  ScheduleExtractor(const Buffer& buffer) : target_buffer_(buffer) { }

  Stmt VisitStmt_(const ForNode* op) {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    if (found) {
      return stmt;
    }
    return Evaluate(0);
  }

  Stmt VisitStmt_(const AllocateNode* op) {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    if (found) {
      return Downcast<Allocate>(stmt)->body;
    }
    return Evaluate(0);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == tvm::tir::attr::thread_extent) {
      const IterVarNode* iv = op->node.as<IterVarNode>();
      ICHECK(iv);
      Var var = iv->var;
      runtime::ThreadScope scope = runtime::ThreadScope::Create(iv->thread_tag);
      runtime::Map<runtime::String, runtime::ObjectRef> ann;

      PrimExpr prev_bank_index;
      bool prev_inside_kernel = false;
      if (scope.rank == 0) { 
        prev_bank_index = bank_index;
        bank_index = bank_index * op->value + var;
        prev_inside_kernel = inside_kernel;
        inside_kernel = true;
        ann.Set("bank", IntImm(runtime::DataType::Bool(), true));
      }
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      if (scope.rank == 0) {
        bank_index = prev_bank_index;
        inside_kernel = prev_inside_kernel;
      }
      if (!found) {
        return Evaluate(0);
      }
      const AttrStmtNode* attr = stmt.as<AttrStmtNode>();
      return For(Downcast<IterVar>(attr->node)->var, PrimExpr(0), attr->value, ForKind::kSerial, attr->body, NullOpt, ann);
    }
    else {
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      if (!found) {
        return Evaluate(0);
      }
      const AttrStmtNode* attr = stmt.as<AttrStmtNode>();
      return attr->body;
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) {
    PrimExpr host_index, in_bank_index;
    Buffer buffer;
    if (!inside_kernel) return Evaluate(0);
    if (const BufferLoadNode* load = op->value.as<BufferLoadNode>()) {
      std::string lscope = GetPtrStorageScope(load->buffer->data);
      std::string sscope = GetPtrStorageScope(op->buffer->data);
      ICHECK(sscope == "local" || lscope == "local") << "Either source or destination must be local.";
      if (sscope == "local" && (lscope == "" || lscope == "global")) { // local <- global: h->d pattern
         ICHECK(op->global_indices.size() == 1) << "In local->global pattern, BufferStore global_indices should be size 1.";
        if (load->buffer->data.get() == target_buffer_->data.get()) {
          found = true;
          host_index = load->indices[0];
          in_bank_index = op->global_indices[0];
          buffer = load->buffer;
          return Evaluate(Call(DataType::Int(32), builtin::pim_transfer(), { buffer->data, host_index, in_bank_index, bank_index }));
        }
      }
      if (lscope == "local" && (sscope == "" || sscope == "global")) { // global <- local: d->h pattern
        ICHECK(load->global_indices.size() == 1) << "In global->local pattern, BufferLoad global_indices should be size 1.";
        if (op->buffer->data.get() == target_buffer_->data.get()) {
          found = true;
          host_index = op->indices[0];
          in_bank_index = load->global_indices[0];
          buffer = op->buffer;
          return Evaluate(Call(DataType::Int(32), builtin::pim_transfer(), { buffer->data, host_index, in_bank_index, bank_index }));
        }
      }
    }
    return Evaluate(0);
  }
};

class AllocateFreeExtractor: public StmtExprVisitor {
public:
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> alloca_candidates;
  bool inside_kernel = false;
  PrimExpr bank_index = 0;
  std::vector<Var> bank_vars;
  std::vector<PrimExpr> bank_extents;
  bool traversed = false;
  Map<Var, PrimExpr> vmap;

  Stmt extract(Stmt stmt, bool is_allocate_mode) {
    if (!traversed) {
      VisitStmt(stmt);
      traversed = true;
    }
    if (alloca_candidates.size() == 0) return Evaluate(0);
    Array<Stmt> allocates;
    for (auto& kv: alloca_candidates) {
      if (is_allocate_mode)
        allocates.push_back(Evaluate(Call(DataType::Int(32), builtin::pim_allocate(), { kv.first, kv.second, bank_index })));
      else
        allocates.push_back(Evaluate(Call(DataType::Int(32), builtin::pim_free(), { kv.first, bank_index })));
    }
    Stmt seq;
    if (allocates.size() == 1) seq = allocates[0];
    else seq = SeqStmt(allocates);
    for (int i = bank_vars.size() - 1; i >= 0; i--) {
      auto& var = bank_vars[i];
      auto& extent = bank_extents[i];
      runtime::Map<runtime::String, runtime::ObjectRef> ann;
      ann.Set("bank", IntImm(runtime::DataType::Bool(), true));
      seq = For(var, PrimExpr(0), extent, ForKind::kSerial, seq, NullOpt, ann);
    }
    return seq;
  }

  void VisitStmt_(const AllocateNode* op) {
    if (!inside_kernel) {
      alloca_candidates[op->buffer_var] = op->extents[0];
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode* op) {
    vmap.Set(op->loop_var, op->min + op->extent - 1);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == tvm::tir::attr::thread_extent) {
      const IterVarNode* iv = op->node.as<IterVarNode>();
      ICHECK(iv);
      Var var = iv->var;
      runtime::ThreadScope scope = runtime::ThreadScope::Create(iv->thread_tag);
      bool prev_inside_kernel = false;
       PrimExpr prev_bank_index;
      runtime::Map<runtime::String, runtime::ObjectRef> ann;

      if (scope.rank == 0) { ;
        prev_inside_kernel = inside_kernel;
        prev_bank_index = bank_index;
        bank_index = bank_index * op->value + var;
        // bankIdx.x가 커널 통틀어 딱 한 번 있어야 한다는 조건이 필요...
        inside_kernel = true;
        bank_vars.push_back(var);
        bank_extents.push_back(op->value);
      }
      vmap.Set(var, op->value - 1);
      StmtExprVisitor::VisitStmt_(op);
      if (scope.rank == 0) {
        inside_kernel = prev_inside_kernel;
      }
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const BufferStoreNode* op) {
    if (inside_kernel) {
      if (const BufferLoadNode* load = op->value.as<BufferLoadNode>()) {
        std::string lscope = GetPtrStorageScope(load->buffer->data);
        std::string sscope = GetPtrStorageScope(op->buffer->data);
        PrimExpr target_expr;
        ICHECK(sscope == "local" || lscope == "local") << "Either source or destination must be local.";
        if (sscope == "local" && (lscope == "" || lscope == "global")) { // local <- global: h->d pattern
          if ((alloca_candidates.find(load->buffer->data) != alloca_candidates.end())) {
            target_expr = op->global_indices[0];
          }
        }
        if (lscope == "local" && (sscope == "" || sscope == "global")) { // global <- local: d->h pattern
          if ((alloca_candidates.find(op->buffer->data) != alloca_candidates.end())) {
            target_expr = load->global_indices[0];
          }
        }
        if (!target_expr.defined()) return;
        target_expr = Substitute(target_expr + 1, vmap);
        arith::Analyzer ana;
        target_expr = ana.Simplify(target_expr);
        if (is_const_number(target_expr)) {
          alloca_candidates[op->buffer->data] = target_expr;
        }
      } // TODO (stonerdk): UNSTABLE!!!!
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

Stmt ConstructTransferStmt(Stmt stmt) {
  FindCopyCandidates candidates;
  candidates(stmt);

  Array<Stmt> seq;
  AllocateFreeExtractor v;

  auto allocates = v.extract(stmt, true);
  auto frees = v.extract(stmt, false);
  seq.push_back(allocates);

  for (auto h2d_bf : candidates.h2d) {
    arith::Analyzer ana;
    auto new_stmt = ScheduleExtractor(h2d_bf)(stmt);
    new_stmt = tir::RemoveNoOp(new_stmt, &ana);
    seq.push_back(new_stmt);
  }

  seq.push_back(Evaluate(Call(DataType::Int(32), builtin::pim_kernel_marker(), {})));

  for (auto h2d_bf : candidates.d2h) {
    arith::Analyzer ana;
    auto new_stmt = ScheduleExtractor(h2d_bf)(stmt);
    new_stmt = tir::RemoveNoOp(new_stmt, &ana);
    seq.push_back(new_stmt);
  }

  seq.push_back(frees);
  return SeqStmt(seq);
}

namespace transform {

Pass ExtractPimTransferSchedule() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ConstructTransferStmt(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ExtractPimTransferSchedule", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ExtractPimTransferSchedule")
    .set_body_typed(ExtractPimTransferSchedule);

}
}
}
