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

#include "../analysis/var_use_def_analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

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
      }
    }
    loop_vars.pop_back();
    return stmt;
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

    auto ret = StmtExprMutator::VisitStmt_(op);

    Stmt body = op->body;
    PrimExpr new_cond;
    const IfThenElseNode* branch = op->body.as<IfThenElseNode>();

    if (branch) {
      body = branch->then_case;  // bypass boundary check
      new_cond = Substitute(branch->condition, {{op->loop_var, 0}});
    }

    if (const BufferStoreNode* store = body.as<BufferStoreNode>()) {
      if (const BufferLoadNode* load = store->value.as<BufferLoadNode>()) {
        if (isFlatEqual(store->indices[0], load->indices[0], lvar)) {
          std::string lscope = GetPtrStorageScope(load->buffer->data);
          std::string sscope = GetPtrStorageScope(store->buffer->data);

          Array<PrimExpr> args({load->buffer->data, GetIndexStrided(load->indices[0]),
                                store->buffer->data, GetIndexStrided(store->indices[0]),
                                op->extent * load->buffer->dtype.bytes()});

          if ((sscope == "local" || sscope == "shared") && (lscope == "" || lscope == "global")) {
            ret = Evaluate(Call(DataType::Int(32), builtin::dpu_mram_read(), args));
          } else if ((lscope == "local" || lscope == "shared") &&
                     (sscope == "" || sscope == "global")) {
            ret = Evaluate(Call(DataType::Int(32), builtin::dpu_mram_write(), args));
          } else {
            return ret;
          }
        }
      }
    }
    return ret;
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
      MoveGlobalIndices moveGlobalIndices;
      m = moveGlobalIndices(m);
      LowerMemoryTransfer lowerMemoryTransfer;
      m = lowerMemoryTransfer(m);
      EliminateBranch eliminateBranch;
      n->body = eliminateBranch(m);
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