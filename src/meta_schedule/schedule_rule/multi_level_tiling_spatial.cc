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

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include "../utils.h"
#include "./multi_level_tiling.h"

namespace tvm {
namespace tir {
namespace {

std::vector<int> GetReadBufferNDims(const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  const BufferNode* write_buffer = block->writes[0]->buffer.get();
  int n = block->reads.size();
  std::vector<int> results(n, -1);
  for (int i = 0; i < n; ++i) {
    const BufferNode* read_buffer = block->reads[i]->buffer.get();
    if (read_buffer != write_buffer) {
      results[i] = read_buffer->shape.size();
    }
  }
  return results;
}

}  // namespace
}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

using tir::BlockRV;
using tir::IterVarType;
using tir::LoopRV;
using tir::Schedule;

// Do nothing; Inherited from ScheduleRuleNode
void MultiLevelTilingSpatialNode::InitializeWithTuneContext(const TuneContext& context) {
  if (Optional<Integer> v = context->target.value()->GetAttr<Integer>("max_threads_per_block")) {
    this->max_threads_per_block_ = v.value()->value;
    if (Optional<Integer> v = context->target.value()->GetAttr<Integer>("thread_warp_size")) {
      this->thread_warp_size_ = v.value()->value;
    } else {
      TVM_PY_LOG(INFO, context->logger) << "'thread_warp_size' is not defined in the target";
    }
  }
  if (Optional<String> opt_sm = context->target.value()->GetAttr<String>("arch")) {
    std::string sm = opt_sm.value();
    if (support::StartsWith(sm, "sm_")) {
      sm = sm.substr(3);
      try {
        // only sm_80 or higher supports async memcopy
        if (std::stoi(sm) >= 80) {
          // only stage = 4 & 5 is tested. all integer that is bigger than 2
          // is theoretically feasible, but no guarantee for great performance.
          this->stages = {4, 5};
        }
      } catch (const std::invalid_argument& e) {
        LOG(WARNING) << "ValueError: Unable to parse `target.arch`: " << sm
                     << ". Details: " << e.what();
      }
    }
  }
  logger = context->logger;
}

// Entry of the mega rule; Inherited from ScheduleRuleNode
Array<Schedule> MultiLevelTilingSpatialNode::Apply(const Schedule& sch, const BlockRV& block_rv) {
  // std::cerr << "MultiLevelTilingSpatialNode::Apply: "
  //           << sch->GetSRef(block_rv)->StmtAs<tir::BlockNode>()->name_hint << std::endl;
  bool try_reorder = false;
  if ((filter_fn_ && filter_fn_.value()(sch, sch->GetSRef(block_rv))) ||
      NeedsMultiLevelTilingSpatial(sch->state(), sch->GetSRef(block_rv), &try_reorder)) {
    sch->Annotate(block_rv, tir::attr::meta_schedule_tiling_structure, structure);

    Array<Schedule> results;
    for (auto&& state : ApplySubRules({State(sch, block_rv)})) {
      results.push_back(std::move(state->sch));
    }
    return results;
  }
  if (try_reorder && IsTrivialBindingOrTryReorder(sch, block_rv)) {
    // std::cerr << "RECOVER SUCCESSFUL: " << std::endl;
    // std::cerr << sch->mod() << std::endl;
  }
  if ((filter_fn_ && filter_fn_.value()(sch, sch->GetSRef(block_rv))) ||
      NeedsMultiLevelTilingSpatial(sch->state(), sch->GetSRef(block_rv))) {
    sch->Annotate(block_rv, tir::attr::meta_schedule_tiling_structure, structure);

    Array<Schedule> results;
    for (auto&& state : ApplySubRules({State(sch, block_rv)})) {
      results.push_back(std::move(state->sch));
    }
    return results;
  }
  {
    Array<Schedule> results;
    for (auto&& state : ApplyExtraSubRules({State(sch, block_rv)})) {
      results.push_back(std::move(state->sch));
    }
    return results;
  }
  return {sch};
}

// Inherited from ScheduleRuleNode
ScheduleRule MultiLevelTilingSpatialNode::Clone() const {
  ObjectPtr<MultiLevelTilingSpatialNode> n = make_object<MultiLevelTilingSpatialNode>(*this);
  return ScheduleRule(n);
}

std::vector<State> MultiLevelTilingSpatialNode::ApplySubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) { return TileLoopNest(std::move(state)); });
  states = SubRule(std::move(states), [&](State state) { return AddWriteReuse(std::move(state)); });
  states = SubRule(std::move(states), [&](State state) { return AddReadReuse(std::move(state)); });
  return states;
}

std::vector<State> MultiLevelTilingSpatialNode::ApplyExtraSubRules(std::vector<State> states) {
  return states;
}

std::vector<State> MultiLevelTilingSpatialNode::AddWriteReuse(State state) const {
  const ReuseConfig& config = this->reuse_write_;
  if (config.req == ReuseType::kNoReuse) {
    return {std::move(state)};
  }
  std::vector<int> levels = config.levels;
  ReuseType req = config.req;
  if (Optional<Array<Integer>> ann = tir::GetAnn<Array<Integer>>(
          state->sch->GetSRef(state->block_rv), "meta_schedule.write_cache_level")) {
    req = ReuseType::kMustReuse;
    levels.clear();
    std::transform(ann.value().begin(), ann.value().end(), std::back_inserter(levels),
                   [](auto&& v) { return v.IntValue(); });
  }
  std::vector<State> results;
  if (req == ReuseType::kMayReuse) {
    // Case 1. If the write cache is already there, we don't need to add another.
    Array<BlockRV> consumer_rvs = state->sch->GetConsumers(state->block_rv);
    if (consumer_rvs.size() == 1 && IsWriteCache(state->sch->GetSRef(consumer_rvs[0]))) {
      for (int level : levels) {
        State new_state = state->Copy();
        const LoopRV& loop_rv = new_state->tiles[level - 1].back();
        new_state->sch->ReverseComputeAt(consumer_rvs[0], loop_rv, true);
        results.push_back(std::move(new_state));
      }
      state->write_reuse.emplace(0, consumer_rvs[0]);
      results.push_back(state);
      return results;
    } else {
      // Case 2. No write cache is added
      State new_state = state->Copy();
      results.emplace_back(std::move(new_state));
    }
  }

  // Case 3. Add one write cache
  BlockRV write_cache =
      state->sch->CacheWrite(/*block_rv=*/state->block_rv, /*read_buffer_index=*/0,
                             /*storage_scope=*/config.scope);
  state->write_reuse.emplace(0, write_cache);
  for (int level : levels) {
    State new_state = state->Copy();
    const LoopRV& loop_rv = new_state->tiles[level - 1].back();
    new_state->sch->ReverseComputeAt(write_cache, loop_rv, true);
    results.push_back(std::move(new_state));
  }
  return results;
}

std::pair<Array<tir::ExprRV>, Array<tir::LoopRV>> MultiLevelTilingSpatialNode::SplitLoop(
    const Schedule& sch, BlockRV block, LoopRV loop, int n_tiles,
    Array<Optional<PrimExpr>> split_factors) const {
  if (split_factors.empty()) {
    Array<tir::ExprRV> factors = sch->SamplePerfectTile(
        /*loop=*/loop,
        /*n=*/n_tiles,
        /*max_innermost_factor=*/max_innermost_factor,
        /*min_innermost_factor=*/min_innermost_factor);
    Array<tir::LoopRV> splits = sch->Split(/*loop=*/loop,
                                           /*factors=*/{factors.begin(), factors.end()});
    return {factors, splits};
  } else {
    Array<tir::ExprRV> factors = sch->SamplePerfectTile(
        /*loop=*/loop,
        /*n=*/n_tiles,
        /*max_innermost_factor=*/max_innermost_factor,
        /*min_innermost_factor=*/min_innermost_factor);
    Array<tir::LoopRV> splits = sch->Split(/*loop=*/loop, split_factors);
    return {factors, splits};  // TODO[ywshin]: we should modify "factors" later
  }
}

std::vector<State> MultiLevelTilingSpatialNode::TileLoopNest(State state) const {
  Schedule& sch = state->sch;
  const BlockRV& block_rv = state->block_rv;
  // Step 1. Assuming trivial binding, pair the loops and their iter-var-types
  Array<LoopRV> loops = sch->GetLoops(block_rv);
  std::vector<IterVarType> iter_types = GetBlockVarTypes(sch->GetSRef(state->block_rv));
  ICHECK_EQ(loops.size(), iter_types.size());
  // Step 2. For each loop axis, tile it
  int64_t spatial_loop_product = 1;
  std::vector<Array<LoopRV>> tiles(s_indices_.size());
  state->tile_factors.resize(tiles.size());
  std::vector<Array<tir::ExprRV>> tile_factors;
  tile_factors.resize(tiles.size());
  int s = 0, r = 0;
  int num_spatial_loops = 0;
  for (int i = 0, n = loops.size(); i < n; ++i) {
    if (iter_types[i] == IterVarType::kDataPar) {
      num_spatial_loops++;
    }
  }
  for (int i = 0, n = loops.size(); i < n; ++i) {
    LoopRV loop = loops[i];
    const std::vector<int>* idx = nullptr;

    Array<Optional<PrimExpr>> split_factors;
    if (iter_types[i] == IterVarType::kDataPar) {
      idx = &s_indices_;
      if (spatial_loop_product != -1) {
        if (const int64_t* extent = tir::GetLoopIntExtent(sch->Get(loop).get())) {
          spatial_loop_product *= *extent;
        } else {
          spatial_loop_product = -1;
        }
      }
      if (!this->s_split_factors.empty()) {
        split_factors = this->s_split_factors[s];
        s++;
        if (split_factors.empty()) {
          // Skip splitting
          continue;
        }
      }
    } else {
      continue;
    }

    const int n_tiles = idx->size();

    if (n_tiles == 1) {
      tiles[idx->at(0)].push_back(loop);
    } else {
      auto [factors, splits] = SplitLoop(sch, block_rv, loop, n_tiles, split_factors);

      // Put every tile to its slot
      for (int j = 0; j < n_tiles; ++j) {
        tiles[idx->at(j)].push_back(splits[j]);
        tile_factors[idx->at(j)].push_back(factors[j]);
      }
    }
  }
  state->tile_factors = std::move(tile_factors);
  // Step 3. Reorder to organize the tiles
  sch->Reorder(support::ConcatArrayList<LoopRV>(tiles.begin(), tiles.end()));
  // Step 4. Bind the tiles to threads
  int n_binds = std::min(tile_binds.size(), tiles.size());
  for (int i = 0; i < n_binds; ++i) {
    if (!tiles[i].empty()) {
      LoopRV fused = tiles[i][0];
      if (tiles[i].size() > 1) {
        fused = sch->Fuse(tiles[i]);
      }
      sch->Bind(fused, tile_binds[i]);
      tiles[i] = {fused};
    }
  }
  int n_annotations = std::min(annotations.size(), tiles.size());
  for (int i = 0; i < n_annotations; ++i) {
    if (!tiles[i].empty()) {
      LoopRV fused = tiles[i][0];
      if (tiles[i].size() > 1) {
        fused = sch->Fuse(tiles[i]);
      }
      auto annotation_for_fused = annotations[i];
      for (auto annotation : annotation_for_fused) {
        if (!annotation.first.empty()) {
          sch->Annotate(fused, annotation.first, annotation.second);
        }
      }
      tiles[i] = {fused};
    }
  }
  state->tiles = Array<Array<LoopRV>>{tiles.begin(), tiles.end()};
  if (this->thread_warp_size_ != -1) {
    int64_t low_inclusive = 1;
    int64_t high_inclusive = this->max_threads_per_block_;
    if (spatial_loop_product > 2 * this->thread_warp_size_) {
      low_inclusive = this->thread_warp_size_;
    }
    sch->Annotate(block_rv, tir::attr::meta_schedule_thread_extent_low_inclusive,
                  Integer(low_inclusive));
    sch->Annotate(block_rv, tir::attr::meta_schedule_thread_extent_high_inclusive,
                  Integer(high_inclusive));
  }
  return {state};
}

std::vector<State> MultiLevelTilingSpatialNode::AddReadReuse(State state) const {
  const ReuseConfig& config = this->reuse_read_;
  if (config.req == ReuseType::kNoReuse) {
    return {std::move(state)};
  }
  ICHECK(config.req != ReuseType::kMayReuse);
  const BlockRV& block_rv = state->block_rv;
  std::vector<State> results;
  results.reserve(config.levels.size());
  for (int i = 0; i < config.levels.size(); i++) {
    int level = config.levels[i];
    if (level < 0) {
      auto n_tile = state->tiles.size();
      level = n_tile + level;
    }
    State new_state = state->Copy();
    Schedule& sch = new_state->sch;
    // if (!config.sep) {
    //   sch = new_state->sch;
    // }
    const LoopRV& loop_rv = state->tiles[level - 1].back();
    // Enumerate all buffers that are read but not written
    std::vector<int> read_buffer_ndims = tir::GetReadBufferNDims(sch->GetSRef(block_rv));
    for (int j = 0, n_reads = read_buffer_ndims.size(); j < n_reads; ++j) {
      // if (config.sep && i != j) {
      //   continue;
      // }
      int buffer_ndim = read_buffer_ndims[j];
      if (buffer_ndim == -1) {
        continue;
      }
      // Do cache_read
      BlockRV cache_read_block = sch->CacheRead(block_rv, j, config.scope, {block_rv});
      // Insert cache_read block to the proper place
      sch->ComputeAt(cache_read_block, loop_rv, true);
      // Fuse the iterators of the cache_read
      Array<LoopRV> buffer_loops = sch->GetLoops(cache_read_block);
      sch->Fuse(Array<LoopRV>{buffer_loops.end() - buffer_ndim,  //
                              buffer_loops.end()});
      AnnotateCooperativeFetching(&sch, cache_read_block);
      // if (!config.sep) {
      new_state->read_reuse.emplace(j, cache_read_block);
      // } else {
      //   state->read_reuse.emplace(j, cache_read_block);
      // }
    }
    // if (!config.sep) {
    results.push_back(std::move(new_state));
    // }
  }
  // if (config.sep) {
  //   results.push_back(std::move(state));
  // }
  return results;
}

void MultiLevelTilingSpatialNode::AnnotateCooperativeFetching(Schedule* sch,
                                                              const tir::BlockRV& block) const {
  // Filter out invalid vector lanes according to the data type.
  const tir::BlockNode* block_node = (*sch)->GetSRef(block)->StmtAs<tir::BlockNode>();
  ICHECK_EQ(block_node->writes.size(), 1);
  const runtime::DataType dtype = block_node->writes[0]->buffer->dtype;
  std::function<bool(int)> f_filter = nullptr;
  if (dtype == runtime::DataType::Float(32)) {
    f_filter = [&](int vector_len) { return vector_len <= 4; };
  } else if (dtype == runtime::DataType::Float(16)) {
    f_filter = [&](int vector_len) {
      return (vector_len == 1 || vector_len % 2 == 0) && vector_len <= 8;
    };
  } else if (dtype == runtime::DataType::Int(8)) {
    f_filter = [&](int vector_len) { return vector_len <= 16; };
  }
  std::vector<int> valid_vector_lens;
  valid_vector_lens.reserve(vector_load_lens.size());
  if (f_filter != nullptr) {
    std::copy_if(vector_load_lens.begin(), vector_load_lens.end(),
                 std::back_inserter(valid_vector_lens), f_filter);
  } else {
    valid_vector_lens = vector_load_lens;
  }

  if (!valid_vector_lens.empty()) {
    int n = valid_vector_lens.size();
    double prob = 1.0 / n;
    tir::ExprRV vector_load_len =
        (*sch)->SampleCategorical(support::AsArray<int, Integer>(valid_vector_lens),
                                  Array<FloatImm>(n, FloatImm(DataType::Float(64), prob)));
    (*sch)->Annotate(block, tir::attr::meta_schedule_cooperative_fetch, vector_load_len);
  }
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingSpatialNode);

}  // namespace meta_schedule
}  // namespace tvm
