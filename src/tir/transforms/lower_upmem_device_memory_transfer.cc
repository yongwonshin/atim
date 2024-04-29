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

#include <queue>
#include <unordered_set>

#include "../../arith/conjunctive_normal_form.h"
#include "../analysis/var_use_def_analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

class AggressiveHoistBranch : public StmtExprMutator {
 public:
  Array<Var> loop_vars;
  Map<Var, Array<Var>> let_seq;
  Map<Var, PrimExpr> let_bindings;
  size_t max_loop_vars = 0;
  Var nearest_loop_var;
  bool hoist_target_flag = false;
  PrimExpr new_cond = make_const(DataType::Bool(), false);

  void fallback() { new_cond = make_const(DataType::Bool(), true); }

  class IsSubNodeExists : public ExprVisitor {
   public:
    bool flag = false;
    void VisitExpr_(const SubNode* op) { flag = true; }
    void VisitExpr_(const GTNode* op) { flag = true; }
    void VisitExpr_(const GENode* op) { flag = true; }
    void VisitExpr_(const NENode* op) { flag = true; }
    void VisitExpr_(const EQNode* op) { flag = true; }
    void VisitExpr_(const CallNode* op) { flag = true; }
  };

  PrimExpr AggOr(PrimExpr a, PrimExpr b) {
    arith::Analyzer ana;
    if (ana.CanProve(Not(a) || b))
      return b;
    else if (ana.CanProve(Not(b) || a))
      return a;
    return a || b;
  }

  PrimExpr VisitExpr(const PrimExpr& expr) final {
    if (auto op = expr.as<CallNode>()) {
      if (op->op.same_as(builtin::dpu_mram_read()) || op->op.same_as(builtin::dpu_mram_write())) {
        return StmtExprMutator::VisitExpr(expr);
      }
    }
    if (SideEffect(expr) >= CallEffectKind::kReadState) {
      fallback();
    }
    return StmtExprMutator::VisitExpr(expr);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) {
    fallback();
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) {
    fallback();
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const WhileNode* op) {
    fallback();
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const LetStmtNode* op) {
    let_bindings.Set(op->var, op->value);
    Stmt ret;
    if (nearest_loop_var.defined()) {
      Array<Var> seq = let_seq.Get(nearest_loop_var).value_or({});
      seq.push_back(op->var);
      let_seq.Set(nearest_loop_var, seq);
      ret = StmtExprMutator::VisitStmt_(op);
      seq.pop_back();
      let_seq.Set(nearest_loop_var, seq);
    } else {
      ret = StmtExprMutator::VisitStmt_(op);
    }
    let_bindings.erase(op->var);
    return ret;
  }

  Stmt VisitStmt_(const IfThenElseNode* op) {
    IsSubNodeExists c;
    c(op->condition);
    if (c.flag) {
      fallback();
    } else {
      new_cond = AggOr(new_cond, op->condition);
    }

    return GetRef<Stmt>(op);  // do not traverse
  }

  Stmt VisitStmt_(const ForNode* op) {
    Var prev_nearest_loop_var = nearest_loop_var;
    nearest_loop_var = op->loop_var;
    PrimExpr prev_cond = new_cond;
    new_cond = make_const(DataType::Bool(), false);
    loop_vars.push_back(op->loop_var);
    max_loop_vars = std::max(max_loop_vars, loop_vars.size());
    auto ret = StmtExprMutator::VisitStmt_(op);
    loop_vars.pop_back();
    arith::Analyzer ana;
    if (let_seq.find(nearest_loop_var) != let_seq.end()) {
      Array<Var> seq = let_seq[nearest_loop_var];
      for (auto it = seq.rbegin(); it != seq.rend(); ++it) {
        auto var = *it;
        auto value = let_bindings[var];
        new_cond = Substitute(new_cond, {{var, value}});
      }
    }
    new_cond = ana.Simplify(AggOr(prev_cond, Substitute(new_cond, {{op->loop_var, op->min}})), 3);
    if (!is_one(new_cond) && !(op->body.as<ForNode>()) && max_loop_vars - loop_vars.size() >= 3) {
      ret = IfThenElse(new_cond, ret);
      fallback();
    }
    nearest_loop_var = prev_nearest_loop_var;
    return ret;
  }
};

// class AggressiveHoistBranch : public StmtExprMutator {
//  public:
//   Array<Var> loop_vars;
//   PrimExpr new_cond = make_const(DataType::Bool(), false);

//   void fallback() { new_cond = make_const(DataType::Bool(), true); }

//   class IsSubNodeExists : public ExprVisitor {
//    public:
//     bool flag = false;
//     void VisitExpr_(const SubNode* op) { flag = true; }
//     void VisitExpr_(const GTNode* op) { flag = true; }
//     void VisitExpr_(const GENode* op) { flag = true; }
//     void VisitExpr_(const NENode* op) { flag = true; }
//     void VisitExpr_(const EQNode* op) { flag = true; }
//     void VisitExpr_(const CallNode* op) { flag = true; }
//   };

//   PrimExpr AggOr(PrimExpr a, PrimExpr b) {
//     arith::Analyzer ana;
//     if (ana.CanProve(Not(a) || b))
//       return b;
//     else if (ana.CanProve(Not(b) || a))
//       return a;
//     return a || b;
//   }

//   PrimExpr VisitExpr_(const CallNode* op) {
//     if (!(op->op.same_as(builtin::dpu_mram_read()) &&
//           !(op->op.same_as(builtin::dpu_mram_write())))) {
//       static auto eff_map = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
//       auto eff = static_cast<CallEffectKind>(eff_map[op->op.as<Op>().value()]->value);
//       if (eff >= CallEffectKind::kUpdateState) {
//         fallback();
//       }
//     }
//     return StmtExprMutator::VisitExpr_(op);
//   }

//   Stmt VisitStmt_(const BufferStoreNode* op) {
//     fallback();
//     return StmtExprMutator::VisitStmt_(op);
//   }

//   Stmt VisitStmt_(const AllocateNode* op) {
//     fallback();
//     return StmtExprMutator::VisitStmt_(op);
//   }

//   Stmt VisitStmt_(const IfThenElseNode* op) {
//     // it only deals with LT/LENode without Sub.
//     IsSubNodeExists c;
//     c(op->condition);
//     if (c.flag) {
//       fallback();
//     } else {
//       new_cond = AggOr(new_cond, op->condition);
//     }

//     return GetRef<Stmt>(op);  // do not traverse
//   }

//   Stmt VisitStmt_(const ForNode* op) {
//     PrimExpr prev_cond = new_cond;
//     new_cond = make_const(DataType::Bool(), false);
//     loop_vars.push_back(op->loop_var);
//     auto ret = StmtExprMutator::VisitStmt_(op);
//     loop_vars.pop_back();
//     if (is_one(new_cond)) {
//       return ret;
//     }

//     arith::Analyzer ana;
//     new_cond = ana.Simplify(Substitute(new_cond, {{op->loop_var, op->min}}), 3);
//     new_cond = AggOr(prev_cond, new_cond);
//     if (!is_one(new_cond) && loop_vars.size() < 2) {
//       return IfThenElse(new_cond, ret);
//     }
//     return ret;
//   }
// };

class EliminateBranch : public StmtExprMutator {
 public:
  Map<Var, Range> dom_map;
  Array<Var> loop_vars;

  Stmt VisitStmt_(const ForNode* op) final {
    dom_map.Set(op->loop_var, Range::FromMinExtent(0, op->extent));
    loop_vars.push_back(op->loop_var);

    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    dom_map.erase(op->loop_var);

    if (const IfThenElseNode* branch = op->body.as<IfThenElseNode>()) {
      Array<PrimExpr> conditions;

      std::queue<PrimExpr> que;
      que.push(branch->condition);
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

      if (constraint.second.size() > 0) {
        return stmt;
      }
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

        Var hoisted_var = Var(op->loop_var->name_hint + "_ext");

        stmt = For(op->loop_var, op->min, hoisted_var, op->kind, branch->then_case);
        stmt = LetStmt(hoisted_var, extent, stmt);
      } else {
        stmt = For(op->loop_var, op->min, op->extent, op->kind, branch->then_case);
      }
    }
    loop_vars.pop_back();
    return stmt;
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    // if (const BufferStoreNode* store = op->then_case.as<BufferStoreNode>()) {
    //   if (GetPtrStorageScope(store->buffer->data) == "local" && is_zero(store->value)) {
    //     return StmtExprMutator::VisitStmt(op->then_case);
    //   }
    // }
    return StmtExprMutator::VisitStmt_(op);
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

Pass LowerUpmemDeviceMemoryTransfer() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    VLOG(1) << "LowerUpmemDeviceMemoryTransfer: \n" << f;
    if (target.value()->kind->name == "upmem") {
      auto m = f->body;

      m = MoveGlobalIndices()(std::move(m));
      m = LowerMemoryTransfer()(std::move(m));
      m = AggressiveHoistBranch()(std::move(m));
      m = EliminateBranch()(std::move(m));
      n->body = std::move(m);
    }
    VLOG(1) << "LowerUpmemDeviceMemoryTransfer: \n" << f;
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerUpmemDeviceMemoryTransfer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerUpmemDeviceMemoryTransfer")
    .set_body_typed(LowerUpmemDeviceMemoryTransfer);

}  // namespace transform

}  // namespace tir
}  // namespace tvm