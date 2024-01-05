/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file infer_buffer_transitivity.cc
 * \brief
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

class HBMParamsInjector : public StmtExprMutator {
 public:
  static Stmt Substitute(const PrimFunc& f) {
    HBMParamsInjector reindexer;
    return reindexer.VisitStmt(f->body);
  }

 private:
  HBMParamsInjector() = default;
};

PrimFunc InjectHBMPIMParams(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    // Var pim_ctr("pim_ctr", DataType::Handle(8));
    // Var crf_binary("crf_binary", DataType::Handle(8));
    // Array<Var> new_params;
    // for (size_t i = 0; i < f->params.size(); ++i) {
    //   new_params.push_back(f->params[i]);
    // }
    // new_params.push_back(pim_ctr);
    // new_params.push_back(crf_binary);
    // fptr->params = std::move(new_params);
    fptr->body = HBMParamsInjector::Substitute(f);
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass InjectHBMPIMParams() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return InjectHBMPIMParams(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectHBMPIMParams", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectHBMPIMParams").set_body_typed(InjectHBMPIMParams);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
