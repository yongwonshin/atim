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

namespace {
/*! \brief Collecting all the blocks */
class BlockCollector : public tir::StmtVisitor {
 public:
  static Array<tir::BlockRV> Collect(const tir::Schedule& sch,
                                     const runtime::PackedFunc f_block_filter = nullptr) {  //
    return BlockCollector(sch, f_block_filter).Run();
  }

 private:
  /*! \brief Entry point */
  Array<tir::BlockRV> Run() {
    std::vector<tir::BlockRV> results;
    for (const auto& kv : sch_->mod()->functions) {
      const GlobalVar& gv = kv.first;         // `gv->name_hint` is the name of the function
      const BaseFunc& base_func = kv.second;  // this can be PrimFunc or relay::Function
      if (const auto* func = base_func.as<tir::PrimFuncNode>()) {
        func_name_ = gv->name_hint;
        block_names_.clear();
        blocks_to_collect_.clear();
        VisitStmt(func->body);
        for (const String& name : blocks_to_collect_) {
          results.push_back(sch_->GetBlock(name, func_name_));
        }
      }
    }
    return results;
  }
  /*! \brief Constructor */
  explicit BlockCollector(const tir::Schedule& sch,
                          const runtime::PackedFunc f_block_filter = nullptr)
      : sch_(sch), f_block_filter_(f_block_filter) {}
  /*! \brief Override the Stmt visiting behaviour */
  void VisitStmt_(const tir::BlockNode* block) override {
    tir::StmtVisitor::VisitStmt_(block);
    CHECK(block_names_.count(block->name_hint) == 0)
        << "Duplicated block name " << block->name_hint << " in function " << func_name_
        << " not supported!";
    block_names_.insert(block->name_hint);

    // If filter function is provided, use it to selectively collect blocks.
    // Otherwise collect all blocks.
    Bool collect_block = Bool(true);
    if (f_block_filter_ != nullptr) {
      collect_block = f_block_filter_(GetRef<tir::Block>(block));
    }
    if (collect_block) {
      blocks_to_collect_.push_back(block->name_hint);
    }
  }

  /*! \brief The schedule to be collected */
  const tir::Schedule& sch_;
  /*! \brief An optional packed func that allows only certain blocks to be collected. */
  const runtime::PackedFunc f_block_filter_;
  /*! \brief The set of func name and block name pair */
  std::unordered_set<String> block_names_;
  /* \brief The list of blocks to collect in order */
  Array<String> blocks_to_collect_;
  /*! \brief Name of the current PrimFunc */
  String func_name_;
};
}  // namespace

class SetOptimizationLevelNode : public ScheduleRuleNode {
 public:
  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {}

  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv);

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<SetOptimizationLevelNode> n = make_object<SetOptimizationLevelNode>(*this);
    return ScheduleRule(n);
  }

 public:
  int level;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("level", &level); }

  static constexpr const char* _type_key = "meta_schedule.SetOptimizationLevel";
  TVM_DECLARE_FINAL_OBJECT_INFO(SetOptimizationLevelNode, ScheduleRuleNode);
};

ScheduleRule ScheduleRule::SetOptimizationLevel(int level) {
  ObjectPtr<SetOptimizationLevelNode> n = make_object<SetOptimizationLevelNode>();
  n->level = level;
  return ScheduleRule(n);
}

Array<tir::Schedule> SetOptimizationLevelNode::Apply(const tir::Schedule& sch,
                                                     const tir::BlockRV& block_rv) {
  Array<tir::BlockRV> blocks = BlockCollector::Collect(sch, nullptr);
  bool found = false;
  while (!blocks.empty()) {
    tir::BlockRV block = blocks.front();
    blocks.erase(blocks.begin());
    Optional<Integer> level =
        tir::GetAnn<Integer>(sch->GetSRef(block), tir::attr::meta_schedule_optimization_level);
    if (level.defined()) {
      found = true;
      break;
    }
  }
  if (!found) {
    // Attach to the "root" block for easy search in the future
    tir::BlockRV root_block = sch->GetBlock("root");
    sch->Annotate(root_block, tir::attr::meta_schedule_optimization_level, Integer(this->level));
  }

  return {sch};
}

TVM_REGISTER_NODE_TYPE(SetOptimizationLevelNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleSetOptimizationLevel")
    .set_body_typed(ScheduleRule::SetOptimizationLevel);

}  // namespace meta_schedule
}  // namespace tvm
