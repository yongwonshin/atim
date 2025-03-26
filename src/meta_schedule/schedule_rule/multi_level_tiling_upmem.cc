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
#include <tvm/meta_schedule/schedule_rule.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <thread>
#include <utility>
#include <vector>

#include "../utils.h"
#include "multi_level_tiling.h"

namespace tvm {
namespace meta_schedule {

using tir::BlockRV;
using tir::IterVarType;
using tir::LoopRV;
using tir::Schedule;

/*!
 * \brief Extension of MultiLevelTiling for auto-tensorizing with a single group of tensor core
 * intrinsics.
 */
class MultiLevelTilingUPMEMNode : public MultiLevelTilingNode {
 private:
  // Subrule: Add tensorized store
  inline std::vector<State> HandleReductionBlockUPMEM(State state) const;

  // Override ApplySubRules to apply tensorization-specific sub-rules
  virtual std::vector<State> ApplySubRules(std::vector<State> states) final;
  virtual std::vector<State> ApplyExtraSubRules(std::vector<State> states) final;

  // Override Apply to apply tensorization-specific analysis before applying sub-rules
  Array<Schedule> Apply(const Schedule& sch, const BlockRV& block_rv) final;

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<MultiLevelTilingUPMEMNode> n = make_object<MultiLevelTilingUPMEMNode>(*this);
    return ScheduleRule(n);
  }

 public:
  /*! \brief Whether to use software pipeline */
  bool use_software_pipeline = false;
  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingUPMEM";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingUPMEMNode, MultiLevelTilingNode);

 private:
};

// Entry of the mega rule; Inherited from ScheduleRuleNode
Array<Schedule> MultiLevelTilingUPMEMNode::Apply(const Schedule& sch, const BlockRV& block_rv) {
  auto res = MultiLevelTilingNode::Apply(sch->Copy(), block_rv);

  if (res.empty()) {
    TVM_PY_LOG(INFO, logger) << "The workload cannot be tensorized.";
    return {sch};
  }
  return res;
}

std::vector<State> MultiLevelTilingUPMEMNode::ApplySubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) {
    auto new_states = TileLoopNest(std::move(state));
    return new_states;
  });
  states = SubRule(std::move(states), [&](State state) {
    auto new_states = AddWriteReuse(state);
    return new_states;
  });
  states = SubRule(std::move(states), [&](State state) {
    auto new_states = AddReadReuse(state);
    return new_states;
  });
  return states;
}

std::vector<State> MultiLevelTilingUPMEMNode::ApplyExtraSubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) {
    auto new_states = HandleReductionBlockUPMEM(state);
    return new_states;
  });
  return states;
}

std::vector<State> MultiLevelTilingUPMEMNode::HandleReductionBlockUPMEM(State state) const {
  Schedule& sch = state->sch;
  const BlockRV& block_rv = state->block_rv;
  if (tir::GetAnn<Integer>(sch->GetSRef(block_rv), tir::attr::meta_schedule_rfactor_consumer_block)
          .defined()) {
    Array<LoopRV> loops = sch->GetLoops(block_rv);
    std::vector<IterVarType> iter_types = GetBlockVarTypes(sch->GetSRef(state->block_rv));

    Array<LoopRV> fused;
    bool all_reduction_iter_vars = true;
    for (int i = 0; i < loops.size(); i++) {
      if (iter_types[i] == IterVarType::kDataPar) {
        fused.push_back(loops[i]);
        all_reduction_iter_vars = false;
      }
    }
    if (fused.size() > 1) {
      fused = {sch->Fuse(fused)};
    }
    ICHECK(!fused.empty());

    if (!all_reduction_iter_vars) {
      Array<tir::ExprRV> factors =
          sch->SamplePerfectTile2(fused[0], 2, 1, std::thread::hardware_concurrency());
      Array<tir::LoopRV> splits = sch->Split(/*loop=*/fused[0], {factors.begin(), factors.end()});
      sch->Parallel(splits[0]);

      // Prevent multi-level tiling
      sch->Annotate(block_rv, tir::attr::meta_schedule_tiling_structure, String(""));
      sch->Unannotate(block_rv, tir::attr::meta_schedule_rfactor_consumer_block);
    }
  }

  return {state};
}

ScheduleRule ScheduleRule::MultiLevelTilingUPMEM(
    Optional<Array<Map<String, String>>> intrin_groups, String structure,
    Optional<Array<String>> tile_binds, Optional<Integer> max_innermost_factor,
    Optional<Array<Integer>> vector_load_lens, Optional<Map<String, ObjectRef>> reuse_read,
    Optional<Map<String, ObjectRef>> reuse_write, Optional<Integer> min_innermost_factor,
    Optional<Array<Integer>> reordering, Optional<Array<Array<Integer>>> s_split_factors,
    Optional<Array<Array<Integer>>> r_split_factors, Optional<Bool> hoist_rfactor_loop,
    Optional<Array<Map<String, ObjectRef>>> annotations,
    Optional<Array<String>> reduction_tile_binds,
    Optional<Array<Map<String, ObjectRef>>> reduction_annotations) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingUPMEMNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write,
      min_innermost_factor, reordering, s_split_factors, r_split_factors, hoist_rfactor_loop,
      annotations, reduction_tile_binds, reduction_annotations);
  return ScheduleRule(node);
}

bool ScheduleRule::IsMultiLevelTilingUPMEM(const ScheduleRule& rule) {
  return rule->IsInstance<MultiLevelTilingUPMEMNode>();
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingUPMEMNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTilingUPMEM")
    .set_body_typed(ScheduleRule::MultiLevelTilingUPMEM);

}  // namespace meta_schedule
}  // namespace tvm
