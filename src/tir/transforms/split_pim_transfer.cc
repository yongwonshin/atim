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
 * \file split_pim_transfer.cc
 * \brief Split PIM transfer function into multiple functions.
 */
#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "../../runtime/thread_storage_scope.h"
#include "../../support/utils.h"
#include "../analysis/var_use_def_analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

class PimTransferSplitter : public StmtMutator {
 public:
  explicit PimTransferSplitter(IRModule* transfer_mod, GlobalVarSupply global_var_supply,
                               const Target& target)
      : transfer_mod_(transfer_mod), global_var_supply_(global_var_supply), target(target) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == "pim_explicit_transfer") {
      std::string var_name = Downcast<StringImm>(op->value)->value;
      VarUseDefAnalyzer use_def({}, false);
      use_def(op->body);
      Map<Var, PrimExpr> vmap;
      // TODO[ywshin]: temporary fix
      for (auto undefined_var : use_def.undefined_) {
        if (support::StartsWith(undefined_var->name_hint, "puIdx")) {
          vmap.Set(undefined_var, Integer(0));
        }
      }
      auto new_body = Substitute(op->body, vmap);
      Buffer buf = Downcast<Buffer>(op->node);
      std::vector<Var> params{buf->data};
      Map<Var, Buffer> buffer_map = {{buf->data, buf}};
      std::string symbol_name = "copy_" + var_name;
      GlobalVar symbol = global_var_supply_->FreshGlobal(symbol_name, false);
      PrimFunc new_func = WithAttrs(PrimFunc(params, new_body, VoidType(), buffer_map),
                                    {{tvm::attr::kTarget, target},
                                     {tvm::attr::kGlobalSymbol, String(symbol_name)},
                                     {tir::attr::kNoAlias, Bool(true)},
                                     {tir::attr::kIsGlobalFunc, Bool(true)}});
      (*transfer_mod_)->Add(symbol, new_func);
      return Evaluate(0);
    }
    return StmtMutator::VisitStmt_(op);
  }

 private:
  IRModule* transfer_mod_;
  GlobalVarSupply global_var_supply_;
  const Target& target;
};

namespace transform {

Pass SplitPimTransfer() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    GlobalVarSupply global_var_supply(mod);

    IRModule transfer_mod = IRModule(Map<GlobalVar, BaseFunc>({}));
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        PrimFunc func = opt.value();

        auto target = func->GetAttr<Target>(tvm::attr::kTarget).value();
        if (func->GetAttr<tvm::Integer>(tvm::attr::kCallingConv, Integer(CallingConv::kDefault)) !=
            CallingConv::kDeviceKernelLaunch) {
          PimTransferSplitter splitter(&transfer_mod, global_var_supply, target);
          auto body = splitter(func->body);
          if (!body.same_as(func->body)) {
            func.CopyOnWrite()->body = body;
            updates->Add(gvar, func);
          }
        }
      }
    }

    mod->Update(updates);
    mod->Update(transfer_mod);
    return ConvertSSA()(mod);
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.SplitPimTransfer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SplitPimTransfer").set_body_typed(SplitPimTransfer);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
