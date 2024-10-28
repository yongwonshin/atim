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
#include <algorithm>
#include <unordered_map>

#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Check if the instruction is annotation with `meta_schedule_optimization_level`
 * \param inst The instruction to be checked
 * \return Whether the instruction is annotation with `meta_schedule_optimization_level`
 */
bool IsAnnotateWithOptimizationLevel(const Instruction& inst) {
  static const InstructionKind& inst_annotate = InstructionKind::Get("Annotate");
  if (!inst->kind.same_as(inst_annotate)) {
    return false;
  }
  ICHECK_EQ(inst->attrs.size(), 1);
  String ann_key = Downcast<String>(inst->attrs[0]);
  return ann_key == attr::meta_schedule_optimization_level;
}

namespace {
/*!
 * \brief Replace the annotation value
 * \param inst The instruction to be replaced
 * \param ann_val The new annotation value
 * \return The replaced instruction
 */
Instruction ReplaceAnnValue(Instruction inst, int64_t ann_val) {
  ICHECK_EQ(inst->inputs.size(), 2);
  return Instruction(/*kind=*/inst->kind,                             //
                     /*inputs=*/{inst->inputs[0], Integer(ann_val)},  //
                     /*attrs=*/inst->attrs,
                     /*outputs=*/inst->outputs);
}
}  // namespace

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

using tir::Instruction;
using tir::Trace;

/*! \brief Create a Mutator that mutates the parallel extent */
class MutateOptimizationLevelNode : public MutatorNode {
 public:
  /*!
   * \brief The maximum number of jobs to be launched per CPU core.
   * It sets the uplimit of CPU parallelism, i.e. `num_cores * max_level`.
   * Use -1 to disable parallelism.
   */
  int64_t max_level;
  /*! \brief The number of cores in CPU. */
  int max_optimization_level_;
  /*! \brief JSON representation of the workload */
  std::string json_mod_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("max_level", &max_level);
    // `max_optimization_level_` is not visited.
    // `json_mod` is not visited.
  }

  static constexpr const char* _type_key = "meta_schedule.MutateOptimizationLevel";
  TVM_DECLARE_FINAL_OBJECT_INFO(MutateOptimizationLevelNode, MutatorNode);

 public:
  struct Candidate;
  // Inherit from `MutatorNode`
  void InitializeWithTuneContext(const TuneContext& context) final {
    Target target = context->target.value();
    this->max_optimization_level_ = GetTargetNumCores(target) * this->max_level;
    this->json_mod_ = SaveJSON(context->mod.value());
  }
  // Inherit from `MutatorNode`
  Optional<Trace> Apply(const Trace& trace, TRandState* rand_state) final;
  // Inherit from `MutatorNode`
  Mutator Clone() const final {
    ObjectPtr<MutateOptimizationLevelNode> n = make_object<MutateOptimizationLevelNode>(*this);
    return Mutator(n);
  }
};

/*! \brief The candidate to be mutated */
struct MutateOptimizationLevelNode::Candidate {
  /*! \brief The annotation instruction */
  Instruction inst;
  /*! \brief The current parallel extent */
  int64_t optimization_level;
  /*! \brief The name of the root block */
  String block_name;
  /*! \brief The name of the PrimFunc */
  String func_name;
};

/*!
 * \brief Get an instruction that annotates the maximum parallel extent
 * \param trace The trace to be mutated
 * \param rand_state The random state
 * \param candidate The candidate to be mutated
 * \return Whether a decision is found
 */
bool FindOptimizationLevelDecision(const Trace& trace, TRandState* rand_state,
                                   MutateOptimizationLevelNode::Candidate* candidate) {
  using tir::BlockRVNode;
  using tir::InstructionNode;
  std::vector<const InstructionNode*> ann_insts;
  ann_insts.reserve(trace->insts.size());
  for (const Instruction& inst : trace->insts) {
    if (tir::IsAnnotateWithOptimizationLevel(inst)) {
      ann_insts.push_back(inst.get());
    }
  }
  int n_ann_insts = ann_insts.size();
  // [ywshin]: 항상 있다고 가정한다.
  ICHECK_EQ(n_ann_insts, 1);
  const InstructionNode* ann_inst = ann_insts[tir::SampleInt(rand_state, 0, n_ann_insts)];
  ICHECK_EQ(ann_inst->inputs.size(), 2);
  candidate->inst = GetRef<Instruction>(ann_inst);
  candidate->optimization_level = Downcast<IntImm>(ann_inst->inputs[1])->value;
  return true;
}

Optional<Trace> MutateOptimizationLevelNode::Apply(const Trace& trace, TRandState* rand_state) {
  // Step 1. Find a parallel decision.
  Candidate candidate;
  FindOptimizationLevelDecision(trace, rand_state, &candidate);
  // Step 2. Replay the instructions to recover loop extents
  tir::Schedule sch = tir::Schedule::Traced(                  //
      /*mod=*/Downcast<IRModule>(LoadJSON(this->json_mod_)),  //
      /*rand_state=*/ForkSeed(rand_state),                    //
      /*debug_mode=*/0,
      /*error_render_level=*/tir::ScheduleErrorRenderLevel::kNone);
  trace->ApplyToSchedule(sch, /*remove_postproc=*/true);

  int64_t new_level = tir::SampleInt(rand_state, 1, this->max_level + 1);
  if (new_level >= candidate.optimization_level) {
    new_level++;
  }

  Array<Instruction> insts;
  insts.reserve(trace->insts.size());
  for (const Instruction& inst : trace->insts) {
    if (inst.same_as(candidate.inst)) {
      insts.push_back(tir::ReplaceAnnValue(candidate.inst, new_level));
    } else if (inst->kind->IsPostproc()) {
      break;
    } else {
      insts.push_back(inst);
    }
  }
  return Trace(insts, trace->decisions);
}

Mutator Mutator::MutateOptimizationLevel(int64_t max_level) {
  ObjectPtr<MutateOptimizationLevelNode> n = make_object<MutateOptimizationLevelNode>();
  n->max_level = max_level;
  return Mutator(n);
}

TVM_REGISTER_NODE_TYPE(MutateOptimizationLevelNode);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorMutateOptimizationLevel")
    .set_body_typed(Mutator::MutateOptimizationLevel);

}  // namespace meta_schedule
}  // namespace tvm
