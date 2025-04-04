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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

class AddUPMEMRFactorNode : public ScheduleRuleNode {
 public:
  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {}

  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv);

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<AddUPMEMRFactorNode> n = make_object<AddUPMEMRFactorNode>(*this);
    return ScheduleRule(n);
  }

 public:
  int min_n_splits;
  int max_n_splits;
  String mem_scope;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("min_n_splits", &min_n_splits);
    v->Visit("max_n_splits", &max_n_splits);
    v->Visit("mem_scope", &mem_scope);
  }

  static constexpr const char* _type_key = "meta_schedule.AddUPMEMRFactor";
  TVM_DECLARE_FINAL_OBJECT_INFO(AddUPMEMRFactorNode, ScheduleRuleNode);
};

ScheduleRule ScheduleRule::AddUPMEMRFactor(int min_n_splits, int max_n_splits,
                                           const String& mem_scope) {
  ObjectPtr<AddUPMEMRFactorNode> n = make_object<AddUPMEMRFactorNode>();
  n->min_n_splits = min_n_splits;
  n->max_n_splits = max_n_splits;
  n->mem_scope = mem_scope;
  return ScheduleRule(n);
}

Array<tir::Schedule> AddUPMEMRFactorNode::Apply(const tir::Schedule& sch,
                                                const tir::BlockRV& block_rv) {
  tir::StmtSRef block_sref = sch->GetSRef(block_rv);
  Array<tir::LoopRV> loops = sch->GetLoops(block_rv);
  if (loops.empty()) {
    // std::cerr << "NOT RFACTORED (empty): "
    //           << sch->GetSRef(block_rv)->StmtAs<tir::BlockNode>()->name_hint << std::endl;
    return {sch};
  }

  const tir::ScheduleState& self = sch->state();
  const tir::StmtSRef& scope_sref = GetScopeRoot(self, block_sref,
                                                 /*require_stage_pipeline=*/false);
  if (!IsReductionBlock(self, block_sref, scope_sref) || HasBeenMultiLevelTiled(block_sref) ||
      IsSpatial(block_sref)) {
    // std::cerr << "NOT RFACTORED (no reduction): "
    //           << sch->GetSRef(block_rv)->StmtAs<tir::BlockNode>()->name_hint << std::endl;
    return {sch};
  }

  // Make a copy of the original schedule.
  tir::Schedule ori_sch = sch->Copy();
  ori_sch->Seed(sch->ForkSeed());

  // Reorder the loop axes if reduction loops are not innermost.
  // After the reordering, fuse all the reduction loops.
  size_t num_spatial_loops;
  tir::LoopRV fused_reduce_loop;
  ReorderAndFuseReductionLoops(sch, block_rv, &fused_reduce_loop, &num_spatial_loops);

  // Split the fused reduction loop.
  if (num_spatial_loops == 0) max_n_splits = 2048;
  Array<tir::ExprRV> factors =
      sch->SamplePerfectTile2(fused_reduce_loop, 2, min_n_splits, max_n_splits);
  Array<tir::LoopRV> split_loops = sch->Split(fused_reduce_loop, {factors.begin(), factors.end()});

  Array<tir::Schedule> res;
  for (const tir::LoopRV& split_loop : split_loops) {
    tir::Schedule sch_tmp = sch->Copy();
    sch_tmp->Seed(sch->ForkSeed());
    try {
      // rfactor at 0-axis
      int axis = num_spatial_loops - 1;
      if (num_spatial_loops == 0) axis = 0;
      const tir::BlockRV& block_rf = sch_tmp->RFactor(split_loop, axis, mem_scope);
      Array<tir::LoopRV> axes = sch_tmp->GetLoops(block_rf);
      ICHECK_GT(axes.size(), num_spatial_loops);

      // Annotate that the rfactor block, which is now the producer of the original block, needs to
      // be considered by the rule Random-Compute-Location.
      sch_tmp->Annotate(block_rv, tir::attr::meta_schedule_random_compute_producer, Integer(1));
      sch_tmp->Annotate(block_rf, tir::attr::meta_schedule_rfactor_producer_block, Integer(1));
      sch_tmp->Annotate(block_rv, tir::attr::meta_schedule_rfactor_consumer_block, Integer(1));
      res.push_back(sch_tmp);
      // std::cerr << "AFTER PIM RFACTOR: "
      //           << sch_tmp->GetSRef(block_rv)->StmtAs<tir::BlockNode>()->name_hint << std::endl;
      // std::cerr << sch_tmp->mod() << std::endl;
    } catch (const tvm::runtime::Error& e) {
      // std::cerr << "ERROR WHILE RFACTORING" << std::endl;
    }
    // only first loop must be rfactored
    // TODO[ywshin]: option?
    break;
  }

  res.push_back(ori_sch);
  return res;
}

bool ScheduleRule::IsAddUPMEMRFactor(const ScheduleRule& rule) {
  return rule->IsInstance<AddUPMEMRFactorNode>();
}

TVM_REGISTER_NODE_TYPE(AddUPMEMRFactorNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleAddUPMEMRFactor")
    .set_body_typed(ScheduleRule::AddUPMEMRFactor);

}  // namespace meta_schedule
}  // namespace tvm
