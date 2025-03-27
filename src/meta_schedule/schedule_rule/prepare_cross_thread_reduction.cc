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

class PrepareCrossThreadReductionNode : public ScheduleRuleNode {
 public:
  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {}

  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv);

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<PrepareCrossThreadReductionNode> n =
        make_object<PrepareCrossThreadReductionNode>(*this);
    return ScheduleRule(n);
  }

  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "meta_schedule.PrepareCrossThreadReduction";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrepareCrossThreadReductionNode, ScheduleRuleNode);
};

ScheduleRule ScheduleRule::PrepareCrossThreadReduction() {
  ObjectPtr<PrepareCrossThreadReductionNode> n = make_object<PrepareCrossThreadReductionNode>();
  return ScheduleRule(n);
}

Array<tir::Schedule> PrepareCrossThreadReductionNode::Apply(const tir::Schedule& sch,
                                                            const tir::BlockRV& block_rv) {
  if (NeedsMultiLevelTilingReduction(sch->state(), sch->GetSRef(block_rv)) &&
      tir::GetAnn<Integer>(sch->GetSRef(block_rv), tir::attr::meta_schedule_rfactor_producer_block)
          .defined()) {
    // Reorder the loop axes if reduction loops are not innermost.
    // After the reordering, fuse all the reduction loops.
    size_t num_spatial_loops;
    tir::LoopRV fused_reduce_loop;
    ReorderAndFuseReductionLoops(sch, block_rv, &fused_reduce_loop, &num_spatial_loops);

    // Split the fused reduction loop.
    Array<tir::ExprRV> factors = sch->SamplePerfectTile2(fused_reduce_loop, 2, 2, 24);
    Array<tir::LoopRV> split_loops =
        sch->Split(fused_reduce_loop, {factors.begin(), factors.end()});
    const tir::BlockRV& block_rf = sch->RFactor(split_loops[0], 0, "shared");
    // Array<tir::LoopRV> axes = sch->GetLoops(block_rf);
    // sch->ReverseComputeAt(block_rv, axes[1], false);
    Array<tir::LoopRV> axes = sch->GetLoops(block_rf);
    sch->ReverseComputeAt(block_rv, axes[1], false);

    sch->Unannotate(block_rv, tir::attr::meta_schedule_rfactor_producer_block);
    sch->Annotate(block_rv, tir::attr::meta_schedule_cross_thread_reduction_block, Integer(1));
    sch->Annotate(block_rf, tir::attr::meta_schedule_rfactor_producer_block, Integer(1));
  }
  return {sch};
}

bool ScheduleRule::IsPrepareCrossThreadReduction(const ScheduleRule& rule) {
  return rule->IsInstance<PrepareCrossThreadReductionNode>();
}

TVM_REGISTER_NODE_TYPE(PrepareCrossThreadReductionNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRulePrepareCrossThreadReduction")
    .set_body_typed(ScheduleRule::PrepareCrossThreadReduction);

}  // namespace meta_schedule
}  // namespace tvm
