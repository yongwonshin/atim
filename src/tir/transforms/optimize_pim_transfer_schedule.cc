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
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"
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

class EliminateBranch : public StmtExprMutator {
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
  class FindBulkConstant : public StmtExprVisitor {
   public:
    Var target_var;
    IntImm factor;
    bool found = false, is_single_target = false;
    bool mulnode_traversing = false, mulnode_found_var = false, mulnode_found_imm = false;

    FindBulkConstant(Var target_var, IntImm factor) : target_var(target_var), factor(factor) {
      if (!factor.defined() || is_one(factor)) {
        is_single_target = true;
      }
    }

    void VisitExpr_(const VarNode* op) final {
      if (is_single_target && op->name_hint == target_var->name_hint) {
        found = true;
      }
      if (!is_single_target && mulnode_traversing && op->name_hint == target_var->name_hint) {
        mulnode_found_var = true;
      }
    }

    void VisitExpr_(const MulNode* op) final {
      if (!is_single_target) {
        mulnode_traversing = true;
        VisitExpr(op->a);
        VisitExpr(op->b);
        mulnode_traversing = false;
        if (mulnode_found_var && mulnode_found_imm) {
          found = true;
        }
        mulnode_found_var = false;
        mulnode_found_imm = false;
      }
    }

    void VisitExpr_(const IntImmNode* op) final {
      if (!is_single_target && mulnode_traversing && op->value == factor->value) {
        mulnode_found_imm = true;
      }
    }
  };

  PrimExpr GetBulkOffset(PrimExpr expr, Var target, IntImm factor) {
    FindBulkConstant visitor(target, factor);
    visitor(expr);
    if (visitor.found) {
      arith::Analyzer analyzer;
      return analyzer.Simplify(expr - target * factor);
    }
    return PrimExpr();
  }

  Stmt VisitStmt_(const ForNode* op) final {
    Stmt stmt = this->VisitStmt(op->body);
    if (const EvaluateNode* eval = stmt.as<EvaluateNode>()) {
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
            return StmtExprMutator::VisitStmt_(op);
          }
          IntImm size = Downcast<IntImm>(call->args[5]);
          PrimExpr host = GetBulkOffset(call->args[1], op->loop_var, size);
          PrimExpr pim = GetBulkOffset(call->args[2], op->loop_var, size);
          if (host.defined() && pim.defined()) {
            PrimExpr bulk_size, new_extent;
            if (is_const_int(op->extent)) {
              bulk_size = arith::Analyzer().Simplify(op->extent * size);
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
                return StmtExprMutator::VisitStmt_(op);
              }
              new_extent = Max(0, Min(bulk_size, clamp_value));
            }
            return Evaluate(Call(DataType::Int(32), call->op,
                                 {call->args[0], host, pim, call->args[3], new_extent, bulk_size}));
          }
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

class UpmemParallelTransfer : public StmtExprMutator {
 public:
  std::vector<const ForNode*> loop_stack;
  bool inside_upmem = false;

  Stmt VisitStmt_(const ForNode* op) final {
    if (op->annotations.find("bank") != op->annotations.end()) {
      loop_stack.push_back(op);
      Stmt stmt = this->VisitStmt(op->body);
      loop_stack.pop_back();
      return stmt;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (const CallNode* call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::pim_transfer_device_to_host()) ||
          call->op.same_as(builtin::pim_transfer_host_to_device())) {
        PrimExpr direction = call->op.same_as(builtin::pim_transfer_device_to_host())
                                 ? make_const(DataType::Int(32), 0)
                                 : make_const(DataType::Int(32), 1);
        Stmt initialize_transfer =
            Evaluate(Call(DataType::Int(32), builtin::dpu_parallel_transfer_init(),
                          {call->args[0], call->args[2], call->args[5], direction}));
        Stmt nested_func_call =
            Evaluate(Call(DataType::Int(32), builtin::dpu_parallel_transfer_bind(),
                          {call->args[3], call->args[1], call->args[4]}));
        for (auto it = loop_stack.rbegin(); it != loop_stack.rend(); ++it) {
          const ForNode* loop = *it;
          nested_func_call =
              For(loop->loop_var, 0, loop->extent, ForKind::kSerial, nested_func_call, NullOpt);
        }
        Stmt commit_transfer =
            Evaluate(Call(DataType::Int(32), builtin::dpu_parallel_transfer_commit(), {}));
        return SeqStmt({initialize_transfer, nested_func_call, commit_transfer});
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

Stmt OptimizePimTransferSchedule(Stmt stmt, Target target) {
  Stmt res = AllocateFreeOnce()(std::move(stmt));
  res = EliminateBranch()(std::move(res));
  res = BulkPimCopy()(std::move(res));

  if (target->HasKey("upmem")) res = UpmemParallelTransfer()(std::move(res));
  return res;
}

}  // namespace tir
}  // namespace tvm