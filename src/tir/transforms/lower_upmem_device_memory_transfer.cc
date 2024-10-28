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
 *  Lower TVM related builtin intrinsics such as packed call.
 * \file tir/transforms/lower_upmem_device_memory_transfer.cc
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <queue>
#include <string>
#include <unordered_set>

#include "../../arith/conjunctive_normal_form.h"
#include "../../meta_schedule/utils.h"
#include "../analysis/var_use_def_analysis.h"
#include "ir_utils.h"
#include "remove_no_op.h"
#include "simplify.h"

namespace tvm {
namespace tir {

class EliminateBranch : public StmtExprMutator {
 public:
  Map<Var, Range> dom_map;
  Array<Var> loop_vars;
  arith::Analyzer* ana;

  EliminateBranch(arith::Analyzer* ana) : ana(ana) {}

  Stmt VisitStmt_(const ForNode* op) final {
    dom_map.Set(op->loop_var, Range::FromMinExtent(0, op->extent));
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    dom_map.erase(op->loop_var);

    if (const IfThenElseNode* branch = op->body.as<IfThenElseNode>()) {
      if (branch->else_case.defined()) {
        return stmt;
      }
      std::queue<PrimExpr> que;
      que.push(branch->condition);
      Array<PrimExpr> conditions;

      while (!que.empty()) {
        PrimExpr cond = que.front();
        que.pop();
        if (const AndNode* an = cond.as<AndNode>()) {
          que.push(an->a);
          que.push(an->b);
        } else if (const LetNode* l = cond.as<LetNode>()) {
          que.push(Substitute(l->body, {{l->var, l->value}}));
        } else {
          conditions.push_back(cond);
        }
      }

      auto constraint = arith::SolveLinearInequalities(
          arith::IntConstraints({op->loop_var}, dom_map, conditions));

      stmt = branch->then_case;
      if (!constraint.second.empty()) {
        PrimExpr aggregated_cond = std::accumulate(
            constraint.second.begin(), constraint.second.end(), make_const(DataType::Bool(), true),
            [](PrimExpr a, PrimExpr b) { return a && b; });
        stmt = IfThenElse(aggregated_cond, stmt);
      }

      auto bounds = constraint.first.at(op->loop_var);
      auto extent = ana->Simplify(
          Max(0, std::accumulate(bounds->upper.begin(), bounds->upper.end(), op->extent,
                                 [](PrimExpr a, PrimExpr b) { return Min(a, b + 1); })));
      stmt = For(op->loop_var, op->min, extent, op->kind, stmt);
    }
    return stmt;
  }
};

class Hoist : public StmtExprMutator {
 public:
  arith::Analyzer* ana;
  bool ignore_mram_read = false;

  Hoist(arith::Analyzer* ana, bool ignore_mram_read = false)
      : ana(ana), ignore_mram_read(ignore_mram_read) {}

  const IfThenElseNode* CheckInvariantIf(Stmt stmt, Var ref_var) {
    if (const IfThenElseNode* branch = stmt.as<IfThenElseNode>()) {
      if (!UsesVar(branch->condition,
                   [ref_var](const VarNode* var) { return var == ref_var.get(); }) &&
          !branch->else_case.defined()) {
        return branch;
      }
    }
    return nullptr;
  }

  bool isDPUCopyNode(Stmt stmt) {
    if (const EvaluateNode* eval = stmt.as<EvaluateNode>()) {
      if (const CallNode* call = eval->value.as<CallNode>()) {
        if (call->op.same_as(builtin::dpu_mram_read())) {
          return true;
        }
      }
    }
    return false;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    Stmt body = this->VisitStmt(op->body);
    if (const IfThenElseNode* branch = CheckInvariantIf(body, op->loop_var)) {
      return IfThenElse(branch->condition,
                        For(op->loop_var, op->min, op->extent, op->kind, branch->then_case));
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    Stmt body = this->VisitStmt(op->body);
    if (const IfThenElseNode* branch = CheckInvariantIf(body, op->var)) {
      return IfThenElse(branch->condition, LetStmt(op->var, op->value, branch->then_case));
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Array<Stmt> InternalVisitSubsequence(Array<Stmt> seq) {
    Stmt first_stmt = this->VisitStmt(seq[0]);
    if (seq.size() == 1) {
      return {first_stmt};
    }

    Array<Stmt> rest = InternalVisitSubsequence(Array<Stmt>(seq.begin() + 1, seq.end()));
    if (rest.size() == 1) {
      if (const IfThenElseNode* branch = rest[0].as<IfThenElseNode>()) {
        if (ignore_mram_read && isDPUCopyNode(first_stmt)) {
          auto new_seq = SeqStmt::Flatten(std::vector<Stmt>({first_stmt, branch->then_case}));
          return {IfThenElse(branch->condition, new_seq)};
        }
        if (const IfThenElseNode* candidate_branch = first_stmt.as<IfThenElseNode>()) {
          auto new_seq =
              SeqStmt::Flatten(std::vector<Stmt>({candidate_branch->then_case, branch->then_case}));
          return {IfThenElse(branch->condition, new_seq)};
        }
      }
    }
    rest.insert(rest.begin(), first_stmt);
    return rest;
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Array<Stmt> seq = InternalVisitSubsequence(op->seq);
    if (seq.size() == 1) {
      return seq[0];
    }
    return SeqStmt(seq);
  }
};

class MoveGlobalIndices : public StmtExprMutator {
  PrimExpr alloc_global_index = PrimExpr();

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    ICHECK_EQ(op->indices.size(), 1) << "Load from non-flat memory not supported.";

    DataType value_dtype = op->dtype;
    DataType element_dtype = op->buffer->dtype;
    ICHECK_EQ(value_dtype.lanes(), element_dtype.lanes()) << "Vectorization not supported.";
    std::string scope = GetPtrStorageScope(op->buffer->data);

    auto ret = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));

    if (alloc_global_index.defined() && (scope == "global" || scope == "")) {
      ret.CopyOnWrite()->indices.Set(0, alloc_global_index);
    }
    return ret;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    ICHECK_EQ(op->indices.size(), 1) << "Store to non-flat memory not supported.";

    DataType value_dtype = op->value.dtype();
    DataType element_dtype = op->buffer->dtype;
    ICHECK_EQ(value_dtype.lanes(), element_dtype.lanes()) << "Vectorization not supported.";

    if (const BufferLoadNode* load = op->value.as<BufferLoadNode>()) {
      std::string lscope = GetPtrStorageScope(load->buffer->data);
      std::string sscope = GetPtrStorageScope(op->buffer->data);
      if ((sscope == "local" || sscope == "shared") &&
          (lscope == "" || lscope == "global")) {  // local <- global
        alloc_global_index = op->global_indices[0];
        auto ret = StmtExprMutator::VisitStmt_(op);
        alloc_global_index = PrimExpr();
        return ret;
      }
      if ((lscope == "local" || lscope == "shared") &&
          (sscope == "" || sscope == "global")) {  // global <- local
        auto ret = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
        ret.CopyOnWrite()->indices.Set(0, load->global_indices[0]);
        return ret;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

class LowerMemoryTransfer : public StmtExprMutator {
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

  Stmt VisitStmt_(const ForNode* op) {
    std::string lvar = op->loop_var->name_hint;
    ICHECK(is_zero(op->min));

    Stmt body = op->body;
    PrimExpr new_cond;

    const AttrStmtNode* attr = body.as<AttrStmtNode>();
    const IfThenElseNode* branch = body.as<IfThenElseNode>();
    if (branch) {
      body = branch->then_case;  // bypass boundary check
      new_cond = Substitute(branch->condition, {{op->loop_var, 0}});
    }

    if (const BufferStoreNode* store = body.as<BufferStoreNode>()) {
      if (const BufferLoadNode* load = store->value.as<BufferLoadNode>()) {
        if (isFlatEqual(store->indices[0], load->indices[0], lvar)) {
          std::string lscope = GetPtrStorageScope(load->buffer->data);
          std::string sscope = GetPtrStorageScope(store->buffer->data);

          PrimExpr size = op->extent * load->buffer->dtype.bytes();

          // size should be 8-byte aligned constant
          const IntImmNode* size_imm = size.as<IntImmNode>();
          if (size_imm && size_imm->value % 8 == 0) {
            Op new_op;
            if ((sscope == "local" || sscope == "shared") && (lscope == "" || lscope == "global")) {
              new_op = builtin::dpu_mram_read();
            } else if ((lscope == "local" || lscope == "shared") &&
                       (sscope == "" || sscope == "global")) {
              new_op = builtin::dpu_mram_write();
            }
            if (new_op.defined()) {
              int div = size_imm->value / 2048;
              int mod = size_imm->value % 2048;

              Array<PrimExpr> div_args({load->buffer->data, GetIndexStrided(load->indices[0]),
                                        store->buffer->data, GetIndexStrided(store->indices[0]),
                                        2048});
              Array<PrimExpr> mod_args({load->buffer->data, GetIndexStrided(load->indices[0]),
                                        store->buffer->data, GetIndexStrided(store->indices[0]),
                                        mod});
              Stmt ret = Evaluate(Call(DataType::Int(32), new_op, mod_args));
              if (div >= 1) {
                Stmt bulk_stmt = Evaluate(Call(DataType::Int(32), new_op, div_args));
                if (div > 1) {
                  bulk_stmt = For(op->loop_var, 0, div, ForKind::kUnrolled, bulk_stmt);
                }
                return SeqStmt({bulk_stmt, ret});
              }
              return ret;
            }
          }
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

namespace transform {

TVM_REGISTER_PASS_CONFIG_OPTION("tir.UpmemKernelOptimize", Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.UpmemUseDummyKernel", Bool);

Pass LowerUpmemDeviceMemoryTransfer() {
  auto pre_pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (target.value()->kind->name == "upmem") {
      bool use_dummy_kernel = ctx->GetConfig<Bool>("tir.UpmemUseDummyKernel", Bool(false)).value();
      if (use_dummy_kernel) {
        f = WithAttr(std::move(f), "upmem_use_dummy_kernel", Bool(1));
      }
      auto m = f->body;

      arith::Analyzer ana;
      ana.rewrite_simplify.SetEnabledExtensions(static_cast<arith::RewriteSimplifier::Extension>(
          arith::RewriteSimplifier::kTransitivelyProveInequalities |
          arith::RewriteSimplifier::kApplyConstraintsToBooleanBranches));

      m = MoveGlobalIndices()(std::move(m));  // required

      // int64_t opt_level =
      //     ctx->GetConfig<Integer>("tir.UpmemKernelOptimize", Integer(4)).value().IntValue();
      ICHECK(f->GetAttr<Integer>("optimization_level").defined());
      int64_t opt_level = f->GetAttr<Integer>("optimization_level").value().IntValue();
      // std::cerr << "OPT_LEVEL: " << opt_level << std::endl;
      if (opt_level >= 1) {
        m = LowerMemoryTransfer()(std::move(m));
      }
      // 0: NO OPT
      // 1: MRAM_READ
      // 2: CLIP
      // 3: CLIP -> HOIST (WEAK)
      // 4: CLIP -> HOIST (STRONG)
      // 5: CLIP -> HOIST (WEAK) -> CLIP
      // 6: CLIP -> HOIST (STRONG) -> CLIP
      // 7: HOIST (STRONG) -> CLIP
      switch (opt_level) {
        case 2:
          m = EliminateBranch(&ana)(std::move(m));
          break;
        case 3:
          m = EliminateBranch(&ana)(std::move(m));
          m = Hoist(&ana, false)(std::move(m));
          break;
        case 4:
          m = EliminateBranch(&ana)(std::move(m));
          m = Hoist(&ana, true)(std::move(m));
          break;
        case 5:
          m = EliminateBranch(&ana)(std::move(m));
          m = Hoist(&ana, false)(std::move(m));
          m = EliminateBranch(&ana)(std::move(m));
          break;
        case 6:
          m = EliminateBranch(&ana)(std::move(m));
          m = Hoist(&ana, true)(std::move(m));
          m = EliminateBranch(&ana)(std::move(m));
          break;
        // case 6:
        //   m = Hoist(&ana, true)(std::move(m));
        //   break;
        // case 7:
        //   m = Hoist(&ana, true)(std::move(m));
        //   m = EliminateBranch(&ana)(std::move(m));
        //   break;
        default:
          break;
      }

      // m = tir::HoistExpressionBlock(std::move(m));
      // m = tir::Simplify(std::move(m), &ana);
      // m = tir::RemoveNoOp(std::move(m), &ana, std::nullopt, nullptr);

      // m = MergeBranch(&ana)(std::move(m));
      // m = EliminateBranch(&ana)(std::move(m));
      n->body = std::move(m);
    }
    return f;
  };

  return CreatePrimFuncPass(pre_pass_func, 0, "tir.LowerUpmemDeviceMemoryTransfer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerUpmemDeviceMemoryTransfer")
    .set_body_typed(LowerUpmemDeviceMemoryTransfer);

}  // namespace transform

}  // namespace tir
}  // namespace tvm