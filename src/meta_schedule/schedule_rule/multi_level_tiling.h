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
#ifndef TVM_META_SCHEDULE_SCHEDULE_RULE_MULTI_LEVEL_TILING_H_
#define TVM_META_SCHEDULE_SCHEDULE_RULE_MULTI_LEVEL_TILING_H_

#include <tvm/meta_schedule/schedule_rule.h>
#include <tvm/tir/schedule/schedule.h>

#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../support/array.h"

namespace tvm {
namespace tir {
/*!
 * \brief Get the buffer dimensions for all the read buffers of a block, but marks the reduction
 * buffers' dimensions as -1
 * \param block_sref The block to be processed
 * \return The buffer dimensions for all the read buffers of a block, except for reduction buffers
 * \note The method is not designed for generic analysis and relies on assumptions in the scenario
 * of multi-level tiling, so it's intentionally kept inside this file not in the analysis header
 */
std::vector<int> GetReadBufferNDims(const StmtSRef& block_sref);

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Configuration of data reuse type:
 * 0) kNoReuse: no reuse is allowed, then no cache_read/write is performed.
 * 1) kMayReuse: reuse is allowed, but no reuse is explored.
 * 2) kMustReuse: reuse is allowed and no reuse is not explored.
 */
enum class ReuseType : int32_t {
  kNoReuse = 0,
  kMayReuse = 1,
  kMustReuse = 2,
};

/*!
 * \brief Converts a string to ReuseType.
 * \param str The string to be converted.
 * \return The converted ReuseType.
 */
inline ReuseType Str2ReuseType(const String& str) {
  if (str == "no") {
    return ReuseType::kNoReuse;
  } else if (str == "may") {
    return ReuseType::kMayReuse;
  } else if (str == "must") {
    return ReuseType::kMustReuse;
  } else {
    LOG(FATAL) << "ValueError: Unknown ReuseType: " << str;
    throw;
  }
}

/*! \brief Configuration of data reuse patterns */
struct ReuseConfig {
  /*! \brief Type of data reuse: no-reuse, may-reuse or must-reuse */
  ReuseType req;
  /*! \brief Which levels are caching stage inserted at */
  std::vector<int> levels;
  /*! \brief The storage scope */
  String scope;
  bool sep;
  std::vector<std::tuple<String, int>> intrin;

  /*! \brief Default constructor: no data reuse */
  ReuseConfig() : req(ReuseType::kNoReuse) {}

  /*! \brief Construct from a configuration dictionary */
  explicit ReuseConfig(const Map<String, ObjectRef>& config)
      : req(Str2ReuseType(Downcast<String>(config.at("req")))),
        levels(support::AsVector<Integer, int>(Downcast<Array<Integer>>(config.at("levels")))),
        scope(Downcast<String>(config.at("scope"))),
        sep(Downcast<Bool>(config.count("sep") > 0 ? config.at("sep") : Bool(false))) {
    ICHECK_GE(config.size(), 3);
    ICHECK_LE(config.size(), 5);
    if (config.count("intrin") > 0) {
      auto intrins = Downcast<Array<Array<ObjectRef>>>(config.at("intrin"));
      for (auto intrin_tuple : intrins) {
        intrin.push_back(
            {Downcast<String>(intrin_tuple[0]), Downcast<Integer>(intrin_tuple[1]).IntValue()});
      }
    }
  }
};

// Forware declaration
class State;

/*! \brief The state of auto scheduling for the multi-level tiling rule */
class StateNode : public Object {
 public:
  /*! \brief The schedule to date */
  tir::Schedule sch;
  /*! \brief The block to be tiled */
  tir::BlockRV block_rv;
  /*! \brief The loop tiles */
  Array<Array<tir::LoopRV>> tiles;
  /*! \brief The factors of the loop tiles. */
  Array<Array<tir::ExprRV>> tile_factors;
  /*! \brief The mapping from buffer index to read cache block. */
  std::unordered_map<int, tir::BlockRV> read_reuse;
  /*! \brief The mapping from buffer index to write cache block. */
  std::unordered_map<int, tir::BlockRV> write_reuse;
  std::vector<int> reordering;
  Array<Array<Optional<PrimExpr>>> s_split_factors;
  Array<Array<Optional<PrimExpr>>> r_split_factors;

  /*!
   * \brief Create a copy of the state. The underlying schedule is copied. Schedule rules that
   * produce multiple states should use this method to create new states.
   */
  virtual State Copy() const;

  static constexpr const char* _type_key = "meta_schedule.State";
  TVM_DECLARE_BASE_OBJECT_INFO(StateNode, Object);
};

/*! \brief Managed reference to StateNode */
class State : public ObjectRef {
 public:
  /*! \brief Default constructor */
  explicit State(tir::Schedule sch, tir::BlockRV block_rv, Array<Array<tir::LoopRV>> tiles = {});
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(State, ObjectRef, StateNode);
};

/*!
 * \brief Helper to apply a sub-rule to a list of auto scheduling states
 * \tparam FLambda The type of the sub-rule functor
 * \param states The list of states to be applied
 * \return The list of states after applying the sub-rule
 */
template <class FLambda>
std::vector<State> SubRule(std::vector<State> states, FLambda sub_rule) {
  std::vector<State> results;
  for (auto&& state : states) {
    std::vector<State> next = sub_rule(std::move(state));
    results.insert(results.end(),                          //
                   std::make_move_iterator(next.begin()),  //
                   std::make_move_iterator(next.end()));
  }
  return results;
}

/*!
 * \brief The mega rule: multi-level tiling with data reuse
 */
class MultiLevelTilingNode : public ScheduleRuleNode {
 public:
  virtual ~MultiLevelTilingNode() = default;

  // SubRule 1. add write cache
  std::vector<State> AddWriteReuse(State state) const;
  // SubRule 2. tile the loop nest
  std::vector<State> TileLoopNest(State state) const;
  // SubRule 3. add read cache
  std::vector<State> AddReadReuse(State state) const;
  // SubRule 4. add async pipeline
  std::vector<State> AddAsyncPipeline(State state) const;

  // Do nothing; Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final;

  // Entry of the mega rule; Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) override;

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const override;

 protected:
  virtual std::vector<State> ApplySubRules(std::vector<State> states);
  virtual std::vector<State> ApplyExtraSubRules(std::vector<State> states);

  virtual std::pair<Array<tir::ExprRV>, Array<tir::LoopRV>> SplitLoop(
      const tir::Schedule& sch, tir::BlockRV block, tir::LoopRV loop, int n_tiles,
      Array<Optional<PrimExpr>> split_factors = {}) const;

  // Annotate a block to use cooperative fetching
  void AnnotateCooperativeFetching(tir::Schedule* sch, const tir::BlockRV& block) const;

 public:
  /*!
   * \brief The tiling structure. Recommended:
   * - 'SSRSRS' on CPU
   * - 'SSSRRSRS' on GPU
   */
  String structure;
  /*! \brief For each level of tiles, which thread axis it is bound to */
  Array<String> tile_binds;
  /*! \brief The maximum size of the innermost factor */
  int max_innermost_factor;
  int min_innermost_factor;
  /*! \brief The length of vector lane in vectorized cooperative fetching */
  std::vector<int> vector_load_lens;
  /*! \brief Data reuse configuration for reading */
  ReuseConfig reuse_read_;
  /*! \brief Data reuse configuration for writing */
  ReuseConfig reuse_write_;
  std::vector<int> reordering;
  Array<Array<Optional<PrimExpr>>> s_split_factors;
  Array<Array<Optional<PrimExpr>>> r_split_factors;
  Array<Map<String, ObjectRef>> annotations;
  Array<String> reduction_tile_binds;
  Array<Map<String, ObjectRef>> reduction_annotations;
  std::vector<int> rfactor_reordering;
  /*! \brief The indices of spatial tiles in `structure` */
  std::vector<int> s_indices_;
  /*! \brief The indices of reduction tiles in `structure` */
  std::vector<int> r_indices_;
  std::set<int> hoisted_loops;
  /*! \brief The size of the thread warp */
  int thread_warp_size_;
  /*! \brief The maximum number of threads to be used size of a thread warp */
  int max_threads_per_block_;
  /*! \brief All available async pipeline stages. */
  std::vector<int> stages;
  /*! \brief The logging function */
  PackedFunc logger;
  /*! \brief The function to overwrite the default condition for applying MultiLevelTiling. */
  Optional<PackedFunc> filter_fn_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("structure", &structure);
    v->Visit("tile_binds", &tile_binds);
    v->Visit("annotations", &annotations);
    v->Visit("max_innermost_factor", &max_innermost_factor);
    v->Visit("min_innermost_factor", &min_innermost_factor);
    // `vector_load_lens` is not visited
    // `reuse_read_` is not visited
    // `reuse_write_` is not visited
    // `s_indices_` is not visited
    // `r_indices_` is not visited
    // `thread_warp_size_` is not visited
    // `max_threads_per_block` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.MultiLevelTiling";
  TVM_DECLARE_BASE_OBJECT_INFO(MultiLevelTilingNode, ScheduleRuleNode);
};

template <typename NodeType>
ObjectPtr<NodeType> MultiLevelTilingInitCommon(
    String structure, Optional<Array<String>> tile_binds, Optional<Integer> max_innermost_factor,
    Optional<Array<Integer>> vector_load_lens, Optional<Map<String, ObjectRef>> reuse_read,
    Optional<Map<String, ObjectRef>> reuse_write, Optional<Integer> min_innermost_factor = NullOpt,
    Optional<Array<Integer>> reordering = NullOpt,
    Optional<Array<Array<Integer>>> s_split_factors = NullOpt,
    Optional<Array<Array<Integer>>> r_split_factors = NullOpt,
    Optional<Array<Integer>> hoisted_loops = NullOpt,
    Optional<Array<Map<String, ObjectRef>>> annotations = NullOpt,
    Optional<Array<String>> reduction_tile_binds = NullOpt,
    Optional<Array<Map<String, ObjectRef>>> reduction_annotations = NullOpt) {
  ObjectPtr<NodeType> n = make_object<NodeType>();
  n->structure = structure;
  n->tile_binds = tile_binds.value_or({});
  n->annotations = annotations.value_or({});
  n->reduction_tile_binds = reduction_tile_binds.value_or({});
  n->reduction_annotations = reduction_annotations.value_or({});
  n->max_innermost_factor = max_innermost_factor.value_or(Integer(-1))->value;
  n->min_innermost_factor = min_innermost_factor.value_or(Integer(-1))->value;
  n->vector_load_lens = vector_load_lens.defined()
                            ? support::AsVector<Integer, int>(vector_load_lens.value())
                            : std::vector<int>();
  n->reuse_read_ = reuse_read.defined() ? ReuseConfig(reuse_read.value()) : ReuseConfig();
  n->reuse_write_ = reuse_write.defined() ? ReuseConfig(reuse_write.value()) : ReuseConfig();
  if (reordering.defined()) {
    ICHECK_EQ(structure.size(), reordering.value().size());
    auto reorder_value = reordering.value();
    n->reordering.reserve(reorder_value.size());
    for (int i = 0; i < reorder_value.size(); i++) {
      n->reordering.push_back(reorder_value[i].IntValue());
    }
  }
  if (s_split_factors.defined()) {
    for (auto factors : s_split_factors.value()) {
      Array<Optional<PrimExpr>> split_factors;
      for (auto factor : factors) {
        if (factor == 0) {
          split_factors.push_back(NullOpt);
        } else {
          split_factors.push_back(factor);
        }
      }
      n->s_split_factors.push_back(split_factors);
    }
  }
  if (r_split_factors.defined()) {
    for (auto factors : r_split_factors.value()) {
      Array<Optional<PrimExpr>> split_factors;
      for (auto factor : factors) {
        if (factor == 0) {
          split_factors.push_back(NullOpt);
        } else {
          split_factors.push_back(factor);
        }
      }
      n->r_split_factors.push_back(split_factors);
    }
  }
  if (hoisted_loops.defined()) {
    for (auto index : hoisted_loops.value()) {
      n->hoisted_loops.insert(index.IntValue());
    }
  }
  for (int i = 0, len = structure.size(); i < len; ++i) {
    char c = structure.data()[i];
    if (c == 'S') {
      if (!reordering.defined()) {
        n->s_indices_.push_back(i);
      } else {
        n->s_indices_.push_back(n->reordering[i]);
      }
    } else if (c == 'R') {
      if (!reordering.defined()) {
        n->r_indices_.push_back(i);
      } else {
        n->r_indices_.push_back(n->reordering[i]);
      }
    } else {
      LOG(FATAL) << "ValueError: Invalid tiling structure: " << structure;
    }
  }
  if (reordering.defined()) {
    int n_s_indices = n->s_indices_.size();
    n->rfactor_reordering.reserve(n_s_indices);
    for (int i = 0; i < n_s_indices; i++) {
      n->rfactor_reordering.push_back(-1);
    }

    int checked_min = INT32_MIN;
    int cur_idx = -1;
    for (int i = 0; i < n_s_indices; i++) {
      int cur_min = INT32_MAX;
      for (int j = 0; j < n_s_indices; j++) {
        int elem = n->s_indices_[j];
        if (elem <= checked_min) {
          continue;
        }
        if (cur_min > elem) {
          cur_min = elem;
          cur_idx = j;
        }
      }
      checked_min = cur_min;
      n->rfactor_reordering[cur_idx] = i;
    }
  }

  n->thread_warp_size_ = -1;
  n->max_threads_per_block_ = -1;
  return n;
}

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_SCHEDULE_RULE_MULTI_LEVEL_TILING_H_
