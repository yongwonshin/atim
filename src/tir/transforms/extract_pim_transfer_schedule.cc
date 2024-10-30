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

/*!
 * \file transfer_schedule.cc
 * \brief Extract Transfer Schedule for PIM API
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/bound.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <chrono>
#include <set>
#include <string>

#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"
#include "../../support/utils.h"
#include "../analysis/var_use_def_analysis.h"
#include "ir_utils.h"
#include "pim_transfer_schedule.h"
#include "remove_no_op.h"

namespace tvm {
namespace tir {

bool IsUPMEMDevice(Target& target) {
  int dev_type = target->GetTargetDeviceType();
  return kDLUPMEM == dev_type;
}

class PimKernelFinder : public StmtExprVisitor {
 public:
  std::vector<Buffer> h2d_explicit, h2d_implicit, d2h;
  std::vector<const VarNode*> consumed_before_buffer, allocated_before_buffer,
      consumed_after_buffer;
  Map<Buffer, PrimExpr> alloca_size_map;
  Map<Buffer, PrimExpr> padded_size_map;
  Map<String, Array<PrimExpr>> symbol_map;
  Map<Var, PrimExpr> vmap;
  bool before_kernel = true, inside_kernel = false, after_kernel = false;
  bool is_single_kernel = false;
  std::string uuid;
  Array<String> h2d_explicit_attr;
  Stmt kernel_body;
  Array<Stmt> kernel_bodies;
  int32_t bank_count = 1;
  std::vector<PrimExpr> bank_array;
  std::vector<const ForNode*> loops;
  std::vector<const IfThenElseNode*> ifs;

  explicit PimKernelFinder(std::string uuid, Array<String> h2d_explicit_attr)
      : uuid(uuid), h2d_explicit_attr(h2d_explicit_attr) {}

  void VisitStmt_(const ForNode* op) {
    loops.push_back(op);
    vmap.Set(op->loop_var, op->min + op->extent - 1);
    StmtExprVisitor::VisitStmt_(op);
    loops.pop_back();
  }

  void VisitStmt_(const IfThenElseNode* op) {
    ifs.push_back(op);
    StmtExprVisitor::VisitStmt_(op);
    ifs.pop_back();
  }

  Stmt GetAllocateStmt(Var var) {
    auto symbol_arr = symbol_map[var->name_hint];
    StringImm var_name = Downcast<StringImm>(symbol_arr[0]);
    StringImm type_str = Downcast<StringImm>(symbol_arr[1]);
    PrimExpr size = symbol_arr[2];
    return Evaluate(Call(DataType::Int(32), builtin::pim_allocate_memory(),
                         {var, var_name, type_str, size, -1}));
  }

  Stmt GetAcquireStmt() {
    if (bank_array.empty()) {
      bank_array.push_back(1);
    }
    Array<PrimExpr> args = Array<PrimExpr>(bank_array.begin(), bank_array.end());
    args.push_back(std::stoi(uuid));
    return Evaluate(Call(DataType::Int(32), builtin::pim_acquire_resources(), args));
  }

  Stmt GetFreeStmt(Var var) {
    return Evaluate(Call(DataType::Int(32), builtin::pim_free_memory(), {var, -1}));
  }

  void postFilter() {
    std::vector<int> indices_to_remove;
    for (auto it = h2d_explicit.begin(); it != h2d_explicit.end(); it++) {
      if (std::find(consumed_before_buffer.begin(), consumed_before_buffer.end(),
                    (*it)->data.get()) != consumed_before_buffer.end()) {
        h2d_implicit.push_back(*it);
        indices_to_remove.push_back(it - h2d_explicit.begin());
      }
    }
    for (int i : indices_to_remove) {
      h2d_explicit.erase(h2d_explicit.begin() + i);
    }

    indices_to_remove.clear();
    for (auto it = d2h.begin(); it != d2h.end(); it++) {
      if (std::find(allocated_before_buffer.begin(), allocated_before_buffer.end(),
                    (*it)->data.get()) != allocated_before_buffer.end()) {
        if (std::find(consumed_after_buffer.begin(), consumed_after_buffer.end(),
                      (*it)->data.get()) == consumed_after_buffer.end()) {
          indices_to_remove.push_back(it - d2h.begin());
        }
      }
    }
    for (int i : indices_to_remove) {
      d2h.erase(d2h.begin() + i);
    }
  }

  void VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == tvm::attr::kTarget) {
      ICHECK(inside_kernel == false || !kernel_body.defined()) << "Only one kernel is supported";
      kernel_body = GetRef<Stmt>(op);

      VarUseDefAnalyzer use_def({}, false);
      use_def(kernel_body);

      for (auto& buf : use_def.undefined_buffers_) {
        for (auto& v : use_def.undefined_) {
          if (buf->data.get() == v.get()) {
            alloca_size_map.Set(buf, 1);
            padded_size_map.Set(buf, buf->shape[0]);
          }
        }
      }
      inside_kernel = true;
      before_kernel = false;
      StmtExprVisitor::VisitStmt_(op);
      inside_kernel = false;
      after_kernel = true;

      for (auto& kv : alloca_size_map) {
        StringImm symbol("__mram");
        std::string var_name = kv.first->data->name_hint;
        std::replace(var_name.begin(), var_name.end(), '.', '_');
        std::replace(var_name.begin(), var_name.end(), '-', '_');
        std::replace(var_name.begin(), var_name.end(), ':', '_');

        if (std::find(h2d_explicit.begin(), h2d_explicit.end(), kv.first) != h2d_explicit.end()) {
          symbol = StringImm("__mram_noinit");
        }
        symbol_map.Set(kv.first->data->name_hint,
                       {StringImm(var_name), StringImm(DLDataType2String(kv.first->dtype)),
                        kv.second, padded_size_map[kv.first], symbol});
      }
      kernel_body = AttrStmt(symbol_map, "upmem_symbol_map", 0, kernel_body);
      kernel_bodies.push_back(kernel_body);
    } else if (op->attr_key == tvm::tir::attr::thread_extent) {
      const IterVarNode* iv = op->node.as<IterVarNode>();
      ICHECK(iv);
      Var var = iv->var;
      runtime::ThreadScope scope = runtime::ThreadScope::Create(iv->thread_tag);
      if (scope.rank == 0) {
        auto imm = op->value.as<IntImmNode>();
        ICHECK(imm) << "Bank index must be constant.";
        int32_t value = imm->value;
        bank_count *= value;
        bank_array.push_back(value);
      }
      vmap.Set(iv->var, op->value - 1);
      if (ifs.empty() && loops.empty()) {
        is_single_kernel = true;
      }
      StmtExprVisitor::VisitStmt_(op);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const BufferStoreNode* op) {
    PrimExpr host_index, in_bank_index;
    Buffer buffer;
    if (before_kernel) {
      consumed_before_buffer.push_back(op->buffer->data.get());
    }
    if (inside_kernel) {
      if (const BufferLoadNode* load = op->value.as<BufferLoadNode>()) {
        std::string lscope = GetPtrStorageScope(load->buffer->data);
        std::string sscope = GetPtrStorageScope(op->buffer->data);
        PrimExpr global_index, host_index;
        Buffer target_buffer;
        if ((sscope == "local" || sscope == "shared") &&
            (lscope == "" || lscope == "global")) {  // local <- global: h->d pattern
          ICHECK(op->global_indices.size() == 1)
              << "In global->local pattern, BufferStore global_indices should be size 1."
              << load->buffer << " -> " << op->buffer << ", " << op->global_indices;
          if (is_single_kernel && std::find(h2d_explicit_attr.begin(), h2d_explicit_attr.end(),
                                            load->buffer->name) != h2d_explicit_attr.end()) {
            h2d_explicit.push_back(load->buffer);
          } else
            h2d_implicit.push_back(load->buffer);
          if (alloca_size_map.find(load->buffer) != alloca_size_map.end()) {
            global_index = op->global_indices[0];
            host_index = load->indices[0];
            target_buffer = load->buffer;
          }
        }
        if ((lscope == "local" || lscope == "shared") &&
            (sscope == "" || sscope == "global")) {  // global <- local: d->h pattern
          ICHECK(load->global_indices.size() == 1)
              << "In local->global pattern, BufferLoad global_indices should be size 1.";
          d2h.push_back(op->buffer);
          if (alloca_size_map.find(op->buffer) != alloca_size_map.end()) {
            global_index = load->global_indices[0];
            host_index = op->indices[0];
            target_buffer = op->buffer;
          }
        }

        if (target_buffer.defined()) {
          global_index = Substitute(global_index + 1, vmap);
          host_index = Substitute(host_index + 1, vmap);
          arith::Analyzer ana;
          global_index = ana.Simplify(global_index);
          host_index = ana.Simplify(host_index);
          if (is_const_number(global_index) && is_const_number(host_index)) {
            const IntImmNode* gIdx = global_index.as<IntImmNode>();
            ICHECK(gIdx) << "allocate_size_map must be inferred into constant.";
            alloca_size_map.Set(target_buffer, global_index);
            padded_size_map.Set(target_buffer, host_index);
          }
        }
      }
    }
    if (after_kernel) {
      consumed_after_buffer.push_back(op->buffer->data.get());
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode* op) {
    vmap.Set(op->var, op->value);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) {
    if (before_kernel) consumed_before_buffer.push_back(op->buffer->data.get());
    if (after_kernel) consumed_after_buffer.push_back(op->buffer->data.get());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AllocateNode* op) {
    if (before_kernel) {
      consumed_before_buffer.push_back(op->buffer_var.get());
      allocated_before_buffer.push_back(op->buffer_var.get());
    }
    if (after_kernel) {
      consumed_after_buffer.push_back(op->buffer_var.get());
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

class ScheduleExtractor : public StmtExprVisitor {
 public:
  const Buffer& target_buffer_;
  bool inside_kernel = false;
  Stmt res_stmt;
  PrimExpr bank_index = 0, host_index = 0;
  std::vector<Var> loops;
  std::vector<PrimExpr> conds;
  Map<Var, PrimExpr> vmap;
  Map<Var, Range> rmap;
  PrimExpr bank_index_;
  const std::set<std::string>& bank_vars_;

  ScheduleExtractor(const Buffer& buffer, PrimExpr bank_index,
                    const std::set<std::string>& bank_vars)
      : target_buffer_(buffer), bank_index_(bank_index), bank_vars_(bank_vars) {}

  static Stmt extract(const Buffer& buffer, Stmt stmt, PrimExpr bank_index,
                      const std::set<std::string>& bank_vars) {
    arith::Analyzer ana;
    ScheduleExtractor v(buffer, bank_index, bank_vars);
    v(stmt);
    return tir::RemoveNoOp(v.res_stmt, &ana);
  }

  class IsVarUsed : public ExprVisitor {
   public:
    Var target_var;
    bool found = false;
    IsVarUsed(Var var) : target_var(var) {}
    void VisitExpr_(const VarNode* op) final {
      if (op == target_var.get()) {
        found = true;
      }
    }
  };

  void VisitStmt_(const ForNode* op) {
    auto new_loop_var = op->loop_var.copy_with_suffix("_");
    // bank_index_ = Substitute(bank_index_, {{op->loop_var, new_loop_var}});
    vmap.Set(op->loop_var, new_loop_var);
    rmap.Set(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    loops.push_back(op->loop_var);
    StmtExprVisitor::VisitStmt_(op);
    loops.pop_back();
    rmap.erase(op->loop_var);
    vmap.erase(op->loop_var);
  }

  void VisitStmt_(const LetStmtNode* op) {
    vmap.Set(op->var, op->value);
    StmtExprVisitor::VisitStmt_(op);
    vmap.erase(op->var);
  }

  void VisitStmt_(const IfThenElseNode* op) {
    conds.push_back(Substitute(op->condition, vmap));
    StmtExprVisitor::VisitStmt_(op);
    conds.pop_back();
  }

  void VisitStmt_(const AttrStmtNode* op) {
    // if (found) return;
    if (op->attr_key == tvm::attr::kTarget) {
      inside_kernel = true;
      StmtExprVisitor::VisitStmt_(op);
      inside_kernel = false;
    }
    if (op->attr_key == tvm::tir::attr::thread_extent && inside_kernel) {
      const IterVarNode* iv = op->node.as<IterVarNode>();
      ICHECK(iv);
      Var var = iv->var;
      Var new_var = iv->var.copy_with_suffix("_");
      // bank_index_ = Substitute(bank_index_, {{var, new_var}});
      runtime::ThreadScope scope = runtime::ThreadScope::Create(iv->thread_tag);
      PrimExpr prev_bank_index;
      if (scope.rank == 0) {
        prev_bank_index = bank_index;
        bank_index = bank_index * op->value + new_var;
      }
      vmap.Set(var, new_var);
      rmap.Set(var, Range::FromMinExtent(PrimExpr(0), op->value));
      loops.push_back(var);
      StmtExprVisitor::VisitStmt_(op);
      loops.pop_back();
      rmap.erase(var);
      vmap.erase(var);

      if (scope.rank == 0) {
        bank_index = prev_bank_index;
      }
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const BufferStoreNode* op) {
    bool found = false;
    PrimExpr in_bank_index;
    Buffer buffer;
    if (!inside_kernel) return;
    if (const BufferLoadNode* load = op->value.as<BufferLoadNode>()) {
      std::string lscope = GetPtrStorageScope(load->buffer->data);
      std::string sscope = GetPtrStorageScope(op->buffer->data);
      if ((sscope == "local" || sscope == "shared") &&
          (lscope == "" || lscope == "global")) {  // local <- global: h->d pattern
        ICHECK(op->global_indices.size() == 1)
            << "In local->global pattern, BufferStore global_indices should be size 1.";
        if (load->buffer->data.get() == target_buffer_->data.get()) {
          found = true;
          host_index = load->indices[0];
          in_bank_index = op->global_indices[0];
          for (int i = 0; i < 10 && !host_index.as<IntImmNode>(); i++) {
            host_index = Substitute(host_index, vmap);
          }
          for (int i = 0; i < 10 && !in_bank_index.as<IntImmNode>(); i++) {
            in_bank_index = Substitute(in_bank_index, vmap);
          }
          in_bank_index = Substitute(op->global_indices[0], vmap);
          buffer = load->buffer;
          auto replaced_bank_index = Substitute(bank_index_, vmap);
          res_stmt =
              Evaluate(Call(DataType::Int(32), builtin::pim_transfer_host_to_device(),
                            {buffer->data, host_index, in_bank_index, replaced_bank_index, 1, 1}));
        }
      }
      if ((lscope == "local" || lscope == "shared") &&
          (sscope == "" || sscope == "global")) {  // global <- local: d->h pattern
        ICHECK(load->global_indices.size() == 1)
            << "In global->local pattern, BufferLoad global_indices should be size 1.";
        if (op->buffer->data.get() == target_buffer_->data.get()) {
          found = true;
          host_index = op->indices[0];
          in_bank_index = load->global_indices[0];
          for (int i = 0; i < 10 && !host_index.as<IntImmNode>(); i++) {
            host_index = Substitute(host_index, vmap);
          }
          for (int i = 0; i < 10 && !in_bank_index.as<IntImmNode>(); i++) {
            in_bank_index = Substitute(in_bank_index, vmap);
          }
          buffer = op->buffer;
          auto replaced_bank_index = Substitute(bank_index_, vmap);
          res_stmt =
              Evaluate(Call(DataType::Int(32), builtin::pim_transfer_device_to_host(),
                            {buffer->data, host_index, in_bank_index, replaced_bank_index, 1, 1}));
        }
      }
      if (found) {
        for (auto it = conds.rbegin(); it != conds.rend(); it++) {
          // exception (hack): if the condition is "threadIdx.x == 0", just bypass it
          if (auto eq = (*it).as<EQNode>()) {
            if (auto var = eq->a.as<VarNode>()) {
              if (support::StartsWith(var->name_hint, "threadIdx") && is_zero(eq->b)) {
                continue;
              }
            }
          }
          res_stmt = IfThenElse(*it, std::move(res_stmt), NullOpt);
        }
        for (auto it = loops.rbegin(); it != loops.rend(); it++) {
          auto v = *it;
          Var new_var = Downcast<Var>(vmap[v]);
          bool is_bank = support::StartsWith(new_var->name_hint,
                                             "blockIdx");  // todo-stonerdk: a little bit hack
          IsVarUsed visitor(new_var);
          visitor(host_index);
          if (visitor.found || is_bank) {
            Map<runtime::String, runtime::ObjectRef> ann;
            if (is_bank) ann.Set("bank", IntImm(runtime::DataType::Bool(), true));
            res_stmt = For(new_var, rmap[v]->min, rmap[v]->extent, ForKind::kSerial,
                           std::move(res_stmt), NullOpt, ann);
          }
        }
      }
    }
    return;
  }
};

class BankIndex : public StmtVisitor {
 public:
  void VisitStmt_(const ForNode* op) {
    if (op->annotations.find("bank") != op->annotations.end()) {
      bank_index_ = bank_index_ * op->extent + op->loop_var;
    }
    StmtVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == tvm::tir::attr::thread_extent && n_target < 2) {
      const IterVarNode* iv = op->node.as<IterVarNode>();
      ICHECK(iv);
      Var var = iv->var;
      runtime::ThreadScope scope = runtime::ThreadScope::Create(iv->thread_tag);
      if (bank_vars_.count(iv->var->name_hint) == 0 && (scope.rank == 0 || scope.rank == 2)) {
        bank_index_ = bank_index_ * op->value + var;
        bank_vars_.insert(iv->var->name_hint);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
  PrimExpr bank_index_ = Integer(0);
  std::set<std::string> bank_vars_;
  int n_target = 0;
};

class KernelReplacer : public StmtExprMutator {
 public:
  Array<Stmt>& replace_target;
  Stmt& early_prologue;
  Stmt& prologue;
  Stmt& epilogue;
  Stmt kernel_body;
  bool inside_kernel = false;
  size_t n_visit = 0;
  KernelReplacer(Array<Stmt> replace, Stmt early_prologue, Stmt prologue, Stmt epilogue)
      : replace_target(replace),
        early_prologue(early_prologue),
        prologue(prologue),
        epilogue(epilogue) {}

  Stmt operator()(Stmt stmt) { return SeqStmt({early_prologue, this->VisitStmt(stmt)}); }

  Stmt VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == tvm::attr::kTarget) {
      if (inside_kernel == false) {
        ICHECK(!kernel_body.defined()) << "Only one kernel is supported";
        return SeqStmt::Flatten(Array<Stmt>{prologue, replace_target[n_visit++], epilogue});
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

class EliminateAttr : public StmtMutator {
 public:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::pragma_explicit_h2d) {
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }
};

class HostBufferPadder: public StmtExprMutator {
public:
  PimKernelFinder& finder;
  HostBufferPadder(PimKernelFinder& finder) : finder(finder) {}

  Stmt VisitStmt_(const AllocateNode* op) final {
    if (std::find(finder.allocated_before_buffer.begin(), finder.allocated_before_buffer.end(),
      op->buffer_var.get()) != finder.allocated_before_buffer.end()) {
      auto symbol_arr = finder.symbol_map[op->buffer_var->name_hint];
      PrimExpr size = symbol_arr[3];
      return Allocate(op->buffer_var, op->dtype, {size}, op->condition, op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

Stmt ConstructTransferStmt(Stmt stmt, Target target, Map<Var, Buffer> buffer_map,
                           PrimExpr bank_index, const std::set<std::string>& bank_vars,
                           std::string uuid, Array<String> h2d_explicit_attr) {
  PimKernelFinder finder(uuid, h2d_explicit_attr);
  finder(stmt);
  finder.postFilter();
  Stmt kernel_body = finder.kernel_body;
  Array<Stmt> kernel_bodies = finder.kernel_bodies;

  Array<Stmt> early_prologue;
  Array<Stmt> prologue;
  Array<Stmt> epilogue;
  for (auto bf : finder.h2d_explicit) {
    StringImm var_name = Downcast<StringImm>(finder.symbol_map[bf->data->name_hint][0]);
    Stmt new_stmt = SeqStmt({finder.GetAcquireStmt(), finder.GetAllocateStmt(bf->data),
                             ScheduleExtractor::extract(bf, stmt, bank_index, bank_vars)});
    Buffer unflattened_buffer;
    for (auto& kv : buffer_map) {
      Buffer buf = kv.second;
      if (buf->data.get() == bf->data.get()) {
        unflattened_buffer =
            Buffer(buf->data, buf->dtype, buf->shape, buf->strides, PrimExpr(), buf->name,
                   buf->data_alignment, 0, kDefault, buf->axis_separators, buf->span);
      }
    }
    ICHECK(unflattened_buffer.defined()) << "Cannot find unflattened buffer for " << bf->data;
    new_stmt = AttrStmt(unflattened_buffer, "pim_explicit_transfer", var_name, new_stmt);
    early_prologue.push_back(new_stmt);
  }
  prologue.push_back(finder.GetAcquireStmt());
  for (auto h2d_bf : finder.h2d_implicit) {
    prologue.push_back(finder.GetAllocateStmt(h2d_bf->data));
    prologue.push_back(ScheduleExtractor::extract(h2d_bf, stmt, bank_index, bank_vars));
  }
  for (auto d2h_bf : finder.d2h) {
    prologue.push_back(finder.GetAllocateStmt(d2h_bf->data));
  }
  for (auto d2h_bf : finder.d2h) {
    epilogue.push_back(ScheduleExtractor::extract(d2h_bf, stmt, bank_index, bank_vars));
    epilogue.push_back(finder.GetFreeStmt(d2h_bf->data));
  }
  for (auto h2d_bf : finder.h2d_implicit) {
    epilogue.push_back(finder.GetFreeStmt(h2d_bf->data));
  }
  for (auto bf : finder.h2d_explicit) {
    epilogue.push_back(finder.GetFreeStmt(bf->data));
  }
  Stmt res = KernelReplacer(kernel_bodies, SeqStmt::Flatten(early_prologue),
                            SeqStmt::Flatten(prologue), SeqStmt::Flatten(epilogue))(stmt);

  res = EliminateAttr()(res);
  res = HostBufferPadder(finder)(res);
  // TODO[ywshin]: maybe trans_size = 32 for HBMPIM
  if (IsUPMEMDevice(target)) {
    res = OptimizePimTransferSchedule(res, target);
  }

  // Array<Stmt> wrapped = {res,
  //                        Evaluate(Call(DataType::Int(32), builtin::pim_release_resources(),
  //                        {}))};

  // res = SeqStmt::Flatten(wrapped);
  return res;
}

bool IsPIMDevice(Target& target) {
  int dev_type = target->GetTargetDeviceType();
  return kDLUPMEM == dev_type || kDLHBMPIM == dev_type;
}
namespace transform {

std::string UUID() {
  // uuid_t uuid;
  // char uuid_str[37];
  // uuid_generate_random(uuid);
  // uuid_unparse_lower(uuid, uuid_str);
  // return std::string(uuid_str);

  auto now = std::chrono::system_clock::now();
  auto microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  microseconds %= INT32_MAX;
  return std::to_string(microseconds);
}

Pass ExtractPimTransferSchedule() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    BankIndex b;
    b(f->body);
    auto* n = f.CopyOnWrite();
    Target target = [&]() {
      auto opt = f->GetAttr<Target>(tvm::attr::kTarget);
      ICHECK(opt) << "MakePackedAPI required the function to be annotated with tvm::attr::kTarget ("
                  << tvm::attr::kTarget << "), but the function only has attributes " << f->attrs;
      return opt.value();
    }();

    if (IsPIMDevice(target)) {
      ICHECK(m->uuid.empty());
      m->uuid = UUID();
      Array<String> h2d_explicit_attr =
          f->GetAttr<Array<String>>(tir::attr::pragma_explicit_h2d).value_or({});
      n->body = ConstructTransferStmt(n->body, target, n->buffer_map, b.bank_index_, b.bank_vars_,
                                      m->uuid, h2d_explicit_attr);
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ExtractPimTransferSchedule", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ExtractPimTransferSchedule")
    .set_body_typed(ExtractPimTransferSchedule);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
