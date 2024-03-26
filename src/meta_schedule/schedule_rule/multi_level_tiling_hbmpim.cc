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
#include <utility>
#include <vector>

#include "../utils.h"
#include "multi_level_tiling.h"

namespace tvm {
namespace meta_schedule {

Optional<tir::BlockRV> TileForIntrin_(tir::Schedule sch, tir::BlockRV block,
                                      const std::string& intrin_name) {
  Optional<tir::LoopRV> tiled_loop_rv = TileWithTensorIntrin(sch, block, intrin_name);
  if (!tiled_loop_rv) {
    // std::cerr << "NOT TileForIntrin_: " <<
    // sch->GetSRef(block)->StmtAs<tir::BlockNode>()->name_hint
    //           << std::endl;
    // std::cerr << sch->mod() << std::endl;
    return NullOpt;
  }
  ICHECK(tiled_loop_rv.defined());
  tir::BlockRV outer_block = sch->Blockize(tiled_loop_rv.value());
  sch->Annotate(outer_block, tir::attr::meta_schedule_auto_tensorize, String(intrin_name));
  return outer_block;
}

using tir::BlockRV;
using tir::IterVarType;
using tir::LoopRV;
using tir::Schedule;

struct HBMPIMIntrinGroup {
  String load_a_intrin;
  String load_b_intrin;
  String compute_intrin;
  String partial_reduction;

  /*! \brief Create HBMPIMIntrinGroup from config in a map. The map should contains the
   * following keys:
   *  - load_a
   *  - load_b
   *  - compute
   *  - partial_reduction;
   * The values of the keys should be the names of the corresponding intrinsics and should be
   * registered via TensorIntrin.Register beforehand.
   */
  static HBMPIMIntrinGroup FromConfig(const Map<String, String>& config);
};

HBMPIMIntrinGroup HBMPIMIntrinGroup::FromConfig(const Map<String, String>& config) {
  auto f_initialize_intrin = [&config](String key_name, String* intrin_name) {
    CHECK(config.count(key_name)) << "ValueError: " << key_name << " is not set.";
    *intrin_name = config.at(key_name);
    // Check the existence of the intrin
    tir::TensorIntrin::Get(*intrin_name);
  };
  HBMPIMIntrinGroup intrin_group;
  f_initialize_intrin("load_a", &intrin_group.load_a_intrin);
  f_initialize_intrin("load_b", &intrin_group.load_b_intrin);
  f_initialize_intrin("compute", &intrin_group.compute_intrin);
  f_initialize_intrin("partial_reduction", &intrin_group.partial_reduction);
  return intrin_group;
}

class HBMPIMStateNode : public StateNode {
 public:
  /*! \brief The tensor core intrinsic group. */
  HBMPIMIntrinGroup intrin_group;
  /*! \brief The auto tensorization maping info. */
  tir::TensorizeInfo mapping_info{nullptr};

  State Copy() const final;

  static constexpr const char* _type_key = "meta_schedule.HBMPIMState";
  TVM_DECLARE_FINAL_OBJECT_INFO(HBMPIMStateNode, StateNode);
};

class HBMPIMState : public State {
 public:
  explicit HBMPIMState(HBMPIMIntrinGroup intrin_group, tir::TensorizeInfo mapping_info,
                       Schedule sch, BlockRV block_rv, Array<Array<tir::LoopRV>> tiles = {});

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HBMPIMState, State, HBMPIMStateNode);
};

TVM_REGISTER_OBJECT_TYPE(HBMPIMStateNode);

HBMPIMState::HBMPIMState(HBMPIMIntrinGroup intrin_group, tir::TensorizeInfo mapping_info,
                         Schedule sch, BlockRV block_rv, Array<Array<LoopRV>> tiles) {
  ObjectPtr<HBMPIMStateNode> node = make_object<HBMPIMStateNode>();
  node->intrin_group = intrin_group;
  node->mapping_info = mapping_info;
  node->sch = std::move(sch);
  node->block_rv = std::move(block_rv);
  node->tiles = std::move(tiles);
  data_ = std::move(node);
}

State HBMPIMStateNode::Copy() const {
  ObjectPtr<HBMPIMStateNode> node = make_object<HBMPIMStateNode>(*this);
  node->sch = sch->Copy();
  return State(node);
}

/*!
 * \brief Extension of MultiLevelTiling for auto-tensorizing with a single group of tensor core
 * intrinsics.
 */
class MultiLevelTilingHBMPIMNode : public MultiLevelTilingNode {
 private:
  // Subrule: Add tensorized load
  inline std::vector<State> TensorizeReadReuseHBMPIM(State state) const;
  // Subrule: Add tensorized store
  inline std::vector<State> TensorizeWriteReuseHBMPIM(State state) const;
  inline std::vector<State> HandleReductionBlockHBMPIM(State state) const;

  // Override ApplySubRules to apply tensorization-specific sub-rules
  virtual std::vector<State> ApplySubRules(std::vector<State> states) final;
  virtual std::vector<State> ApplyExtraSubRules(std::vector<State> states) final;

  // Override Apply to apply tensorization-specific analysis before applying sub-rules
  Array<Schedule> Apply(const Schedule& sch, const BlockRV& block_rv) final;

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<MultiLevelTilingHBMPIMNode> n = make_object<MultiLevelTilingHBMPIMNode>(*this);
    return ScheduleRule(n);
  }

  /*!
   * \brief Tile, blockize and annotate for tensorization with the given intrin
   * \param block_rv The block to be tensorized
   * \param intrin_name The name of the tensor intrin
   */
  void TileAndAnnotateTensorize(Schedule* sch, const BlockRV& block_rv,
                                const String& intrin_name) const;

 public:
  /*! \brief The candidate tensor core intrin groups to apply */
  std::vector<HBMPIMIntrinGroup> intrin_groups;
  /*! \brief Whether to use software pipeline */
  bool use_software_pipeline = false;
  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingHBMPIM";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingHBMPIMNode, MultiLevelTilingNode);

 private:
};

// Entry of the mega rule; Inherited from ScheduleRuleNode
Array<Schedule> MultiLevelTilingHBMPIMNode::Apply(const Schedule& sch, const BlockRV& block_rv) {
  // auto desc_func = tir::TensorIntrin::Get(intrin_groups[0].compute_intrin).value()->desc;
  // if (!CheckAutoTensorizeApplicable(sch, block_rv, desc_func)) {
  //   std::cerr << "NOT CheckAutoTensorizeApplicable: "
  //             << sch->GetSRef(block_rv)->StmtAs<tir::BlockNode>()->name_hint << std::endl;
  //   std::cerr << sch->mod() << std::endl;
  //   TVM_PY_LOG(INFO, logger) << "The workload cannot be tensorized.";
  //   return {sch};
  // }
  // std::cerr << "OK CheckAutoTensorizeApplicable: "
  //           << sch->GetSRef(block_rv)->StmtAs<tir::BlockNode>()->name_hint << std::endl;

  auto res = MultiLevelTilingNode::Apply(sch->Copy(), block_rv);

  if (res.empty()) {
    TVM_PY_LOG(INFO, logger) << "The workload cannot be tensorized.";
    return {sch};
  }
  TVM_PY_LOG(INFO, logger) << "Tensorizing with " << intrin_groups[0].compute_intrin;
  return res;
}

std::vector<State> MultiLevelTilingHBMPIMNode::ApplySubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) {
    if (auto block_rv =
            TileForIntrin_(state->sch, state->block_rv, intrin_groups[0].compute_intrin)) {
      state->block_rv = block_rv.value();
      // std::cerr << "Tensorization Annotated: "
      //           << state->sch->GetSRef(state->block_rv)->StmtAs<tir::BlockNode>()->name_hint
      //           << std::endl;
      // std::cerr << state->sch->mod() << std::endl;
      return std::vector<State>(1, state);
    }
    return std::vector<State>();
  });
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
    auto new_states = TensorizeWriteReuseHBMPIM(state);
    for (auto new_state : new_states) {
      // std::cerr << "AFTER TensorizeWriteReuseHBMPIM: " << std::endl;
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
  states = SubRule(std::move(states), [&](State state) {
    auto new_states = TensorizeReadReuseHBMPIM(state);
    for (auto new_state : new_states) {
      // std::cerr << "AFTER TensorizeReadReuseHBMPIM: " << std::endl;
      // std::cerr << new_state->sch->mod() << std::endl;
    }
    return new_states;
  });
  return states;
}

std::vector<State> MultiLevelTilingHBMPIMNode::ApplyExtraSubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) {
    auto new_states = HandleReductionBlockHBMPIM(state);
    for (auto new_state : new_states) {
      // std::cerr << "AFTER HandleReductionBlockHBMPIM: " << std::endl;
      // std::cerr << new_state->sch->mod() << std::endl;
    }
    return new_states;
  });
  return states;
}

void MultiLevelTilingHBMPIMNode::TileAndAnnotateTensorize(Schedule* sch, const BlockRV& block_rv,
                                                          const String& intrin_name) const {
  Optional<LoopRV> loop = TileWithTensorIntrin(*sch, block_rv, intrin_name).value();
  ICHECK(loop.defined());
  BlockRV blockized_outer = (*sch)->Blockize(loop.value());
  (*sch)->Annotate(blockized_outer, tir::attr::meta_schedule_auto_tensorize, intrin_name);
}

std::vector<State> MultiLevelTilingHBMPIMNode::TensorizeReadReuseHBMPIM(State state) const {
  Schedule& sch = state->sch;
  const ReuseConfig& config = this->reuse_read_;
  auto extract_intrin = [](String key_name, HBMPIMIntrinGroup intrin_group) {
    if (key_name == "load_a") {
      return intrin_group.load_a_intrin;
    } else if (key_name == "load_b") {
      return intrin_group.load_b_intrin;
    } else {
      LOG(FATAL) << "Unknown intrinsic key: " << key_name;
    }
  };

  for (auto [i, block_rv] : state->read_reuse) {
    auto loops = sch->GetLoops(block_rv);
    String intrin_name = std::get<0>(config.intrin.at(i));
    int offset = std::get<1>(config.intrin.at(i));
    auto blockized_load = sch->Blockize(loops[loops.size() + offset]);
    sch->Annotate(blockized_load, tir::attr::meta_schedule_auto_tensorize,
                  extract_intrin(intrin_name, this->intrin_groups[0]));
  }
  return {state};
}

std::vector<State> MultiLevelTilingHBMPIMNode::HandleReductionBlockHBMPIM(State state) const {
  Schedule& sch = state->sch;
  const BlockRV& block_rv = state->block_rv;
  Array<LoopRV> s_loops;
  if (tir::GetAnn<Integer>(sch->GetSRef(block_rv), tir::attr::meta_schedule_rfactor_consumer_block)
          .defined()) {
    Array<LoopRV> loops = sch->GetLoops(block_rv);
    std::vector<IterVarType> iter_types = GetBlockVarTypes(sch->GetSRef(state->block_rv));
    for (int i = 0, n = loops.size(); i < n; ++i) {
      LoopRV loop = loops[i];
      if (iter_types[i] == IterVarType::kDataPar) {
        auto [factors, splits] =
            SplitLoop(sch, block_rv, loop, s_split_factors[0].size(), s_split_factors[0]);
        s_loops = splits;
      } else {
        continue;
      }
    }
    Array<LoopRV> reordered_loops;
    for (auto idx : rfactor_reordering) {
      reordered_loops.push_back(s_loops[idx]);
    }

    sch->Reorder(reordered_loops);

    int n_binds = std::min(reduction_tile_binds.size(), reordered_loops.size());
    for (int i = 0; i < n_binds; ++i) {
      if (!reduction_tile_binds[i].empty()) {
        sch->Bind(reordered_loops[i], reduction_tile_binds[i]);
      }
    }

    int n_annotations = std::min(reduction_annotations.size(), reordered_loops.size());
    for (int i = 0; i < n_annotations; ++i) {
      auto annotation_for_loop = reduction_annotations[i];
      for (auto annotation : annotation_for_loop) {
        if (!annotation.first.empty()) {
          sch->Annotate(reordered_loops[i], annotation.first, annotation.second);
        }
      }
    }
    // std::cerr << "CHECK ORDERING: ";
    // for (auto idx : rfactor_reordering) {
    //   std::cerr << idx << ", ";
    // }
    // std::cerr << std::endl;
    // std::cerr << "CHECK FACTORS: ";
    // for (auto idx : rfactor_reordering) {
    //   std::cerr << s_split_factors[0][idx] << ", ";
    // }
    // std::cerr << std::endl;
  }

  return {state};
}

std::vector<State> MultiLevelTilingHBMPIMNode::TensorizeWriteReuseHBMPIM(State state) const {
  Schedule& sch = state->sch;
  const ReuseConfig& config = this->reuse_write_;
  auto extract_intrin = [](String key_name, HBMPIMIntrinGroup intrin_group) {
    if (key_name == "partial_reduction") {
      return intrin_group.partial_reduction;
    } else {
      LOG(FATAL) << "Unknown intrinsic key: " << key_name;
    }
  };

  for (auto [i, block_rv] : state->write_reuse) {
    auto loops = sch->GetLoops(block_rv);
    String intrin_name = std::get<0>(config.intrin.at(i));
    int offset = std::get<1>(config.intrin.at(i));
    auto blockized_store = sch->Blockize(loops[loops.size() + offset]);
    sch->Annotate(blockized_store, tir::attr::meta_schedule_auto_tensorize,
                  extract_intrin(intrin_name, this->intrin_groups[0]));
  }
  return {state};
}

ScheduleRule ScheduleRule::MultiLevelTilingHBMPIM(
    Array<Map<String, String>> intrin_groups, String structure, Optional<Array<String>> tile_binds,
    Optional<Integer> max_innermost_factor, Optional<Array<Integer>> vector_load_lens,
    Optional<Map<String, ObjectRef>> reuse_read, Optional<Map<String, ObjectRef>> reuse_write,
    Optional<Integer> min_innermost_factor, Optional<Array<Integer>> reordering,
    Optional<Array<Array<Integer>>> s_split_factors,
    Optional<Array<Array<Integer>>> r_split_factors, Optional<Array<Integer>> hoisted_loops,
    Optional<Array<Map<String, ObjectRef>>> annotations,
    Optional<Array<String>> reduction_tile_binds,
    Optional<Array<Map<String, ObjectRef>>> reduction_annotations) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingHBMPIMNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write,
      min_innermost_factor, reordering, s_split_factors, r_split_factors, hoisted_loops,
      annotations, reduction_tile_binds, reduction_annotations);
  node->intrin_groups.reserve(intrin_groups.size());
  for (const auto& intrin_group_config : intrin_groups) {
    HBMPIMIntrinGroup group = HBMPIMIntrinGroup::FromConfig(intrin_group_config);
    node->intrin_groups.emplace_back(group);
  }

  return ScheduleRule(node);
}

bool ScheduleRule::IsMultiLevelTilingHBMPIM(const ScheduleRule& rule) {
  return rule->IsInstance<MultiLevelTilingHBMPIMNode>();
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingHBMPIMNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTilingHBMPIM")
    .set_body_typed(ScheduleRule::MultiLevelTilingHBMPIM);

}  // namespace meta_schedule
}  // namespace tvm
