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

#include <tvm/arith/analyzer.h>
#include <tvm/arith/bound.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"
#include "../../support/utils.h"
#include "../analysis/var_use_def_analysis.h"
#include "ir_utils.h"
#include "pim_transfer_schedule.h"
#include "remove_no_op.h"

namespace tvm {
namespace tir {

class AllocateFreeOnce : public StmtExprMutator {
 public:
  // bool to_parallelize = false;
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->annotations.find("bank") != op->annotations.end()) {
      Stmt stmt = this->VisitStmt(op->body);
      if (const EvaluateNode* eval = stmt.as<EvaluateNode>()) {
        if (const CallNode* call = eval->value.as<CallNode>()) {
          if (call->op.same_as(builtin::pim_allocate_memory())) {
            return Evaluate(Call(DataType::Int(32), builtin::pim_allocate_memory(),
                                 {call->args[0], call->args[1], call->args[2], call->args[3]}));
          } else if (call->op.same_as(builtin::pim_free_memory())) {
            return Evaluate(Call(DataType::Int(32), builtin::pim_free_memory(), {call->args[0]}));
          }
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

class EliminateTransferBranch : public StmtExprMutator {
 public:
  Map<Var, Range> dom_map;
  Array<Var> loop_vars;
  bool inside_copy_loop = false;

  Stmt VisitStmt_(const ForNode* op) final {
    dom_map.Set(op->loop_var, Range::FromMinExtent(0, op->extent));
    loop_vars.push_back(op->loop_var);
    bool prev_inside_copy_loop = inside_copy_loop;
    inside_copy_loop |= (op->annotations.find("bank") != op->annotations.end());

    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    dom_map.erase(op->loop_var);

    if (const IfThenElseNode* branch = op->body.as<IfThenElseNode>()) {
      if (inside_copy_loop) {
        VarUseDefAnalyzer use_def({}, false);
        use_def(branch->condition);

        auto constraint = arith::SolveLinearInequalities(
            arith::IntConstraints({op->loop_var}, dom_map, {branch->condition}));
        auto bounds = constraint.first.at(op->loop_var);

        auto extent = op->extent;
        if (bounds->upper.size() >= 1) {
          auto boundary_value = bounds->upper[0] + 1;
          for (size_t i = 1; i < bounds->upper.size(); i++) {
            boundary_value = Min(extent, bounds->upper[i] + 1);
          }

          extent = Max(0, Min(extent, boundary_value));
          arith::Analyzer ana;
          extent = ana.Simplify(extent);
        }

        stmt = For(op->loop_var, op->min, extent, op->kind, branch->then_case, NullOpt,
                   {{"bulk", op->extent}});
      }
    }
    inside_copy_loop = prev_inside_copy_loop;
    loop_vars.pop_back();
    return stmt;
  }
};

class BulkPimCopy : public StmtExprMutator {
 public:
  std::vector<Var> loop_vars;
  Map<String, Array<PrimExpr>> symbol_map;
  Map<Var, PrimExpr> vmap;

  class FactorMap : public ExprVisitor {
   public:
    std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> m;
    bool is_affine = true;
    PrimExpr factor = 1;

    FactorMap(PrimExpr expr) { VisitExpr(expr); }

    void VisitExpr_(const VarNode* op) final {
      if (m.count(GetRef<Var>(op)) == 0) {
        m[GetRef<Var>(op)] = factor;
      } else {
        m[GetRef<Var>(op)] = 0;
      }
      ExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MulNode* op) final {
      if (op->a.as<VarNode>() && op->b.as<IntImmNode>()) {
        factor = Downcast<IntImm>(op->b);
      } else if (op->b.as<VarNode>() && op->a.as<IntImmNode>()) {
        factor = Downcast<IntImm>(op->a);
      }
      ExprVisitor::VisitExpr_(op);
      factor = 1;
    }

    void VisitExpr_(const FloorDivNode* op) final {
      if (op->a.as<VarNode>()) {
        m[Downcast<Var>(op->a)] = 0;
      }
      ExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const FloorModNode* op) final {
      if (op->a.as<VarNode>()) {
        m[Downcast<Var>(op->a)] = 0;
      }
      ExprVisitor::VisitExpr_(op);
    }

    int get_factor(Var v) {
      if (m.count(v) > 0) {
        auto imm = Downcast<IntImm>(m[v]);
        if (imm.defined()) {
          return imm->value;
        }
      }
      return 0;
    }
  };

  PrimExpr GetBulkOffset(FactorMap fmap, PrimExpr expr, Var target, IntImm factor) {
    if (fmap.get_factor(target) == factor->value) {
      arith::Analyzer analyzer;
      return analyzer.Simplify(expr - target * factor);
    }
    return PrimExpr();
  }

  Stmt rewrite(Stmt stmt) {
    class ReplaceAttr : public StmtMutator {
     public:
      Map<String, Array<PrimExpr>> symbol_map;
      ReplaceAttr(Map<String, Array<PrimExpr>>& symbol_map) : symbol_map(symbol_map) {}
      Stmt VisitStmt_(const AttrStmtNode* op) final {
        if (op->attr_key == "upmem_symbol_map") {
          return AttrStmt(symbol_map, op->attr_key, op->value, op->body);
        }
        return StmtMutator::VisitStmt_(op);
      }
    };
    PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (const AttrStmtNode* attr = node.as<AttrStmtNode>()) {
        if (attr->attr_key == "upmem_symbol_map") {
          this->symbol_map = Downcast<Map<String, Array<PrimExpr>>>(attr->node);
        }
      }
    });
    stmt = this->VisitStmt(stmt);
    ReplaceAttr replacer(symbol_map);
    return replacer(stmt);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tvm::tir::attr::thread_extent) {
      const IterVarNode* iv = op->node.as<IterVarNode>();
      vmap.Set(iv->var, op->value - 1);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const LetStmtNode* op) {
    vmap.Set(op->var, op->value);
    return StmtExprMutator::VisitStmt_(op);
    vmap.erase(op->var);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    vmap.Set(op->loop_var, op->extent - 1);
    loop_vars.push_back(op->loop_var);
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    loop_vars.pop_back();
    vmap.erase(op->loop_var);

    auto body = Downcast<For>(stmt)->body;
    if (const EvaluateNode* eval = body.as<EvaluateNode>()) {
      if (const CallNode* call = eval->value.as<CallNode>()) {
        if (call->op.same_as(builtin::pim_transfer_device_to_host()) ||
            call->op.same_as(builtin::pim_transfer_host_to_device())) {
          ICHECK(call->args.size() == 6);
          PrimExpr clamp_value;
          if (const MaxNode* max_ = call->args[4].as<MaxNode>()) {
            if (const MinNode* min_ = max_->b.as<MinNode>()) {
              if (is_zero(max_->a)) clamp_value = min_->b;
            }
          }
          if (!is_const_number(call->args[4]) && !clamp_value.defined()) {
            return stmt;
          }
          IntImm size = Downcast<IntImm>(call->args[5]);

          FactorMap host_fac(call->args[1]), pim_fac(call->args[2]);
          if (!host_fac.is_affine || !pim_fac.is_affine) return stmt;

          PrimExpr host = GetBulkOffset(host_fac, call->args[1], op->loop_var, size);
          PrimExpr pim = GetBulkOffset(pim_fac, call->args[2], op->loop_var, size);

          if (host.defined() && pim.defined()) {
            PrimExpr bulk_size, new_extent;
            if (is_const_int(op->extent)) {
              bulk_size = op->extent * size;
              if (loop_vars.size() >= 1) {
                PrimExpr prev_factor = host_fac.get_factor(loop_vars.back());
                if (!is_zero(prev_factor)) {
                  bulk_size = Min(prev_factor, bulk_size);
                }
              }
              bulk_size = arith::Analyzer().Simplify(bulk_size);
              new_extent = bulk_size;
            } else {
              ICHECK(!clamp_value.defined());
              ICHECK(is_one(size));
              ICHECK(op->annotations.find("bulk") != op->annotations.end());
              bulk_size = Downcast<IntImm>(op->annotations["bulk"]);
              new_extent = op->extent;
            }
            if (clamp_value.defined()) {
              clamp_value = arith::Analyzer().Simplify(clamp_value + op->loop_var * size);
              VarUseDefAnalyzer use_def({}, false);
              use_def(clamp_value);
              if (use_def.use_count_.count(op->loop_var.get()) > 0) {
                return stmt;
              }
              new_extent = Max(0, Min(bulk_size, clamp_value));
            }
            std::string vname = Downcast<Var>(call->args[0])->name_hint->data;
            String var_name = vname;
            if (symbol_map.find(var_name) != symbol_map.end()) {
              arith::Analyzer ana;
              PrimExpr max_global_index = host;
              PrimExpr max_host_index = pim;
              for (int i = 0; i < 10 && !max_global_index.as<IntImmNode>(); i++) {
                max_global_index = ana.Simplify(Substitute(max_global_index, vmap));
              }
              for (int i = 0; i < 10 && !max_host_index.as<IntImmNode>(); i++) {
                max_host_index = ana.Simplify(Substitute(max_host_index, vmap));
              }
              max_global_index = ana.Simplify(max_global_index + 1 + bulk_size);
              max_host_index = ana.Simplify(max_host_index + 1 + bulk_size);

              Array<PrimExpr> symbol_arr = symbol_map[var_name];
              ICHECK(max_global_index.as<IntImmNode>())
                  << "max_global_index must be a constant " << max_global_index;

              symbol_arr.Set(2, Max(symbol_arr[2], max_global_index));

              ICHECK(max_host_index.as<IntImmNode>())
                  << "max_host_index must be a constant " << max_host_index;
              symbol_arr.Set(3, Max(symbol_arr[3], max_host_index));
            }

            return Evaluate(Call(DataType::Int(32), call->op,
                                 {call->args[0], host, pim, call->args[3], new_extent, bulk_size}));
          }
        }
      }
    }
    return stmt;
  }
};

class UpmemParallelTransfer : public StmtExprMutator {
 public:
  std::vector<const ForNode*> loop_stack;
  bool inside_upmem = false;
  arith::Analyzer ana;
  Map<Buffer, PrimExpr> alloca_size_map;

  UpmemParallelTransfer(Map<Buffer, PrimExpr> alloca_size_map) : alloca_size_map(alloca_size_map) {}

  Stmt VisitStmt_(const ForNode* op) final {
    if (op->annotations.find("bank") != op->annotations.end()) {
      loop_stack.push_back(op);
      Stmt stmt = this->VisitStmt(op->body);
      loop_stack.pop_back();
      return stmt;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  std::pair<Buffer, Stmt> bindSingle(Var buffer_var, PrimExpr bank_index, PrimExpr base,
                                     PrimExpr extent, PrimExpr bulk_size, bool d2h) {
    // get number of banks
    int n_banks = 1;
    for (auto it = loop_stack.rbegin(); it != loop_stack.rend(); ++it) {
      n_banks *= Downcast<IntImm>((*it)->extent)->value;
    }
    // Var buf_ptr(buffer_var->name_hint + "_ptr", DataType::UInt(64));

    Buffer buf;
    for (const auto& kv : alloca_size_map) {
      if (kv.first->data.get() == buffer_var.get()) {
        buf = kv.first;
        break;
      }
    }
    ICHECK(buf.defined()) << "Cannot find buffer for " << buffer_var;

    Buffer bind_buffer =
        decl_buffer({IntImm(DataType::UInt(64), n_banks)}, DataType::Handle(), "bind_buffer");
    int dtype_size = buffer_var->dtype.bytes();
    Stmt stmt = BufferStore(
        bind_buffer, Call(DataType::Handle(), builtin::address_of(), {BufferLoad(buf, {base})}),
        {bank_index});
    if (!ana.CanProve(extent == bulk_size) && d2h) {
      BufferStore bounded_bind =
          BufferStore(bind_buffer,
                      Call(DataType::Handle(), builtin::dpu_parallel_transfer_bind_bounded(),
                           {bank_index, base, extent}),
                      {bank_index});
      stmt = IfThenElse(extent == bulk_size, stmt, bounded_bind);
    }

    for (auto it = loop_stack.rbegin(); it != loop_stack.rend(); ++it) {
      const ForNode* loop = *it;
      stmt = For(loop->loop_var, 0, loop->extent, ForKind::kSerial, stmt, NullOpt);
    }
    // stmt = LetStmt(buf_ptr, Cast(DataType::UInt(64), buf.access_ptr(1)), stmt);
    stmt = Allocate(bind_buffer->data, DataType::Handle(), {n_banks}, const_true(), stmt);
    return {bind_buffer, stmt};
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (const CallNode* call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::pim_transfer_device_to_host()) ||
          call->op.same_as(builtin::pim_transfer_host_to_device())) {
        bool direction = call->op.same_as(builtin::pim_transfer_device_to_host());
        Stmt initialize_transfer = Evaluate(
            Call(DataType::Int(32), builtin::dpu_parallel_transfer_init(),
                 {call->args[0], call->args[2], call->args[5], static_cast<int>(!direction)}));

        auto [bind_buffer, bind_transfer] =
            bindSingle(Downcast<Var>(call->args[0]), call->args[3], call->args[1], call->args[4],
                       call->args[5], direction);
        Stmt bind_all = Evaluate(Call(DataType::Int(32), builtin::dpu_parallel_transfer_bind_all(),
                                      {bind_buffer->data}));
        Stmt commit_transfer =
            Evaluate(Call(DataType::Int(32), builtin::dpu_parallel_transfer_commit(), {}));
        return SeqStmt({initialize_transfer, bind_transfer, bind_all, commit_transfer});
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

Stmt OptimizePimTransferSchedule(Stmt stmt, Target target, Map<Buffer, PrimExpr> alloca_size_map) {
  Stmt res = AllocateFreeOnce()(std::move(stmt));
  res = EliminateTransferBranch()(std::move(res));
  res = BulkPimCopy()(std::move(res));

  if (target->HasKey("upmem")) res = UpmemParallelTransfer(alloca_size_map)(std::move(res));
  return res;
}

}  // namespace tir
}  // namespace tvm