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
class MultiLevelTilingSpatialUPMEMNode : public MultiLevelTilingSpatialNode {
 private:
  // Override ApplySubRules to apply tensorization-specific sub-rules
  virtual std::vector<State> ApplySubRules(std::vector<State> states) final;
  virtual std::vector<State> ApplyExtraSubRules(std::vector<State> states) final;

  // Override Apply to apply tensorization-specific analysis before applying sub-rules
  Array<Schedule> Apply(const Schedule& sch, const BlockRV& block_rv) final;

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<MultiLevelTilingSpatialUPMEMNode> n =
        make_object<MultiLevelTilingSpatialUPMEMNode>(*this);
    return ScheduleRule(n);
  }

 public:
  /*! \brief Whether to use software pipeline */
  bool use_software_pipeline = false;
  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingSpatialUPMEM";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingSpatialUPMEMNode, MultiLevelTilingNode);

 private:
};

// Entry of the mega rule; Inherited from ScheduleRuleNode
Array<Schedule> MultiLevelTilingSpatialUPMEMNode::Apply(const Schedule& sch,
                                                        const BlockRV& block_rv) {
  auto res = MultiLevelTilingSpatialNode::Apply(sch->Copy(), block_rv);

  if (res.empty()) {
    TVM_PY_LOG(INFO, logger) << "The workload cannot be tensorized.";
    return {sch};
  }
  return res;
}

std::vector<State> MultiLevelTilingSpatialUPMEMNode::ApplySubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) {
    auto new_states = TileLoopNest(std::move(state));
    for (auto new_state : new_states) {
      // std::cerr << "AFTER TILING: " << std::endl;
      // std::cerr << new_state->sch->mod() << std::endl;
    }
    return new_states;
  });
  states = SubRule(std::move(states), [&](State state) {
    auto new_states = AddWriteReuse(state);
    for (auto new_state : new_states) {
      // std::cerr << "AFTER AddWriteReuse: " << std::endl;
      // std::cerr << new_state->sch->mod() << std::endl;
    }
    return new_states;
  });
  states = SubRule(std::move(states), [&](State state) {
    auto new_states = AddReadReuse(state);
    for (auto new_state : new_states) {
      // std::cerr << "AFTER AddReadReuse: " << std::endl;
      // std::cerr << new_state->sch->mod() << std::endl;
    }
    return new_states;
  });
  return states;
}

std::vector<State> MultiLevelTilingSpatialUPMEMNode::ApplyExtraSubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) {
    // std::cerr << "AFTER MultiLevelTilingSpatial: " << std::endl;
    // std::cerr << state->sch->mod() << std::endl;
    return std::vector<State>{state};
  });
  return states;
}

ScheduleRule ScheduleRule::MultiLevelTilingSpatialUPMEM(
    Optional<Array<Map<String, String>>> intrin_groups, String structure,
    Optional<Array<String>> tile_binds, Optional<Integer> max_innermost_factor,
    Optional<Array<Integer>> vector_load_lens, Optional<Map<String, ObjectRef>> reuse_read,
    Optional<Map<String, ObjectRef>> reuse_write, Optional<Integer> min_innermost_factor,
    Optional<Array<Integer>> reordering, Optional<Array<Array<Integer>>> s_split_factors,
    Optional<Array<Array<Integer>>> r_split_factors, Optional<Bool> hoist_rfactor_loop,
    Optional<Array<Map<String, ObjectRef>>> annotations,
    Optional<Array<String>> reduction_tile_binds,
    Optional<Array<Map<String, ObjectRef>>> reduction_annotations) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingSpatialUPMEMNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write,
      min_innermost_factor, reordering, s_split_factors, r_split_factors, hoist_rfactor_loop,
      annotations, reduction_tile_binds, reduction_annotations);
  return ScheduleRule(node);
}

bool ScheduleRule::IsMultiLevelTilingSpatialUPMEM(const ScheduleRule& rule) {
  return rule->IsInstance<MultiLevelTilingSpatialUPMEMNode>();
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingSpatialUPMEMNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTilingSpatialUPMEM")
    .set_body_typed(ScheduleRule::MultiLevelTilingSpatialUPMEM);

}  // namespace meta_schedule
}  // namespace tvm
