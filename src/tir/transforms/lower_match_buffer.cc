/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file lower_match_buffer.cc
 * \brief The pass for lowering match_buffer.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../ir/functor_common.h"
#include "ir_utils.h"
#include "../schedule/analysis.h"

namespace tvm {
namespace tir {
class MatchBufferLower : public StmtExprMutator {
 public:
  explicit MatchBufferLower(const PrimFunc& func) {
    for (const Var& param : func->params) {
      // Mark input var as const variable.
      if (!param.dtype().is_handle()) var_map_.Set(param, param);
    }
  }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    for (const MatchBufferRegion& match_buffer : op->match_buffers) {
      CheckAndUpdateVarMap(match_buffer);
    }

    Stmt stmt = StmtExprMutator ::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op != nullptr);
    Array<BufferRegion> reads =
        op->reads.Map(std::bind(&MatchBufferLower::VisitBufferRegion, this, std::placeholders::_1));
    Array<BufferRegion> writes = op->writes.Map(
        std::bind(&MatchBufferLower::VisitBufferRegion, this, std::placeholders::_1));

    if (reads.same_as(op->reads) && writes.same_as(op->writes) && op->match_buffers.empty()) {
      return stmt;
    } else {
      auto n = CopyOnWrite(op);
      n->match_buffers = {};
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    auto it = var_map_.find(v);
    if (it != var_map_.end()) {
      return (*it).second;
    } else {
      return std::move(v);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    ICHECK(op != nullptr);

    auto it = match_buffers_.find(op->buffer);
    if (it == match_buffers_.end()) {
      return stmt;
    } else {
      const Buffer& buffer = (*it).first;
      const BufferRegion& source = (*it).second;

      auto n = CopyOnWrite(op);
      n->indices = ConvertIndices(MatchBufferRegion(buffer, source), op->indices);
      if (op->buffer.scope() == "local") {
        auto it = match_buffers_global_.find(op->buffer);
        const Buffer& buffer = (*it).first;
        const BufferRegion& source = (*it).second;
        n->global_indices = ConvertIndices(MatchBufferRegion(buffer, source), op->global_indices);
      }
      n->buffer = source->buffer;
      return Stmt(n);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    ICHECK(op != nullptr);

    auto it = match_buffers_.find(op->buffer);
    if (it == match_buffers_.end()) {
      return expr;
    } else {
      const Buffer& buffer = (*it).first;
      const BufferRegion& source = (*it).second;
      Array<PrimExpr> indices = ConvertIndices(MatchBufferRegion(buffer, source), op->indices);
      if (op->buffer.scope() == "local") {
        auto it = match_buffers_global_.find(op->buffer);
        const Buffer& buffer = (*it).first;
        const BufferRegion& source = (*it).second;
        Array<PrimExpr> global_indices =
            ConvertIndices(MatchBufferRegion(buffer, source), op->global_indices);
        return BufferLoad(source->buffer, indices, global_indices);
      }
      return BufferLoad(source->buffer, indices);
    }
  }

  BufferRegion VisitBufferRegion(const BufferRegion& buffer_region) {
    const Buffer& buffer = buffer_region->buffer;
    auto it = match_buffers_.find(buffer);
    if (it == match_buffers_.end()) {
      return buffer_region;
    } else {
      const BufferRegion& source = (*it).second;
      Region region = ConvertRegion(MatchBufferRegion(buffer, source), buffer_region->region);
      return BufferRegion(source->buffer, std::move(region));
    }
  }

  BufferRegion VisitBufferGlobalRegion(const BufferRegion& buffer_region) {
    const Buffer& buffer = buffer_region->buffer;
    auto it = match_buffers_global_.find(buffer);
    if (it == match_buffers_global_.end()) {
      return buffer_region;
    } else {
      const BufferRegion& source = (*it).second;
      Region region = ConvertRegion(MatchBufferRegion(buffer, source), buffer_region->region);
      return BufferRegion(source->buffer, std::move(region));
    }
  }

 private:
  void CheckAndUpdateVarMap(const MatchBufferRegion& match_buffer) {
    // Step.1. Check
    const Buffer& buffer = match_buffer->buffer;
    const BufferRegion& source = VisitBufferRegion(match_buffer->source);
    const BufferRegion& global_source = VisitBufferGlobalRegion(match_buffer->global_source);
    const PrimExpr& bank_index = match_buffer->bank_index;
    const Buffer& source_buffer = source->buffer;
    const Buffer& global_source_buffer = global_source->buffer;

    // Step.1.1. Check scope & dtype
    ICHECK_EQ(buffer.scope(), source_buffer.scope())
        << "MatchBuffer " << buffer << " scope mismatch:" << buffer.scope() << "vs."
        << source_buffer.scope();
    ICHECK_EQ(buffer->dtype, source_buffer->dtype)
        << "MatchBuffer " << buffer << " data type mismatch:" << buffer->dtype << "vs."
        << source_buffer->dtype;

    // Step.1.2. Check data alignment
    if (source_buffer->data_alignment % buffer->data_alignment != 0) {
      LOG(WARNING) << "Trying to bind buffer to another one with lower alignment requirement "
                   << " required_alignment=" << buffer->data_alignment
                   << ", provided_alignment=" << source_buffer->data_alignment;
    }
    if (is_zero(buffer->elem_offset)) {
      ICHECK(is_zero(source_buffer->elem_offset))
          << "Trying to bind a Buffer with offset into one without offset "
          << " required elem_offset=" << buffer->elem_offset
          << ", provided elem_offset=" << source_buffer->elem_offset;
    }

    // Step.2. Update
    match_buffers_.Set(buffer, source);
    match_buffers_global_.Set(buffer, global_source);
    // Step.2.1. Update buffer data
    Bind(buffer->data, source_buffer->data, buffer->name + ".data");

    // Step.2.2. Update element offset
    // We use the ElemOffset method to avoid duplicating the index calculation.
    {
      Array<PrimExpr> indices;
      indices.reserve(source->region.size());
      for (const Range& range : source->region) {
        indices.push_back(range->min);
      }

      Array<PrimExpr> buffer_start_indices = source_buffer->ElemOffset(indices);
      if (buffer_start_indices.size() == 1) {
        Bind(buffer->elem_offset, buffer_start_indices[0], buffer->name + ".elem_offset");
        CHECK(analyzer_.CanProve(truncmod(buffer->elem_offset, buffer->offset_factor) == 0))
            << "The source elem_offset " << buffer_start_indices[0]
            << " does not satisfy the offset_factor " << buffer->offset_factor << ".";
      } else {
        // Non-zero elem_offset is ill-defined for non-flat memory.
        // If needed in the future, will require `Array<PrimExpr>
        // elem_offsets`, with one offset for each flattened index.
        Bind(buffer->elem_offset, make_const(buffer->elem_offset.dtype(), 0));
      }
    }

    // Step.2.2. Update element offset
    // We use the ElemOffset method to avoid duplicating the index calculation.
    {
      Array<PrimExpr> indices;
      indices.reserve(global_source->region.size());
      for (const Range& range : global_source->region) {
        indices.push_back(range->min);
      }

      Array<PrimExpr> buffer_start_indices = source_buffer->InBankElemOffset(indices);
      tvm::tir::Var var("test");
      tvm::tir::Let b(var, buffer_start_indices[0], var);
      in_bank_elem_offset_map_.Set(var, buffer->name);
      if (buffer_start_indices.size() == 1) {
        Bind(buffer->in_bank_elem_offset, b, buffer->name + ".in_bank_elem_offset");
        // CHECK(analyzer_.CanProve(truncmod(buffer->elem_offset, buffer->offset_factor) == 0))
        //     << "The source elem_offset " << buffer_start_indices[0]
        //     << " does not satisfy the offset_factor " << buffer->offset_factor << ".";
      } else {
        // Non-zero elem_offset is ill-defined for non-flat memory.
        // If needed in the future, will require `Array<PrimExpr>
        // elem_offsets`, with one offset for each flattened index.
        Bind(buffer->in_bank_elem_offset, make_const(buffer->in_bank_elem_offset.dtype(), 0));
      }
    }

    {
      tvm::tir::Var var("test");
      tvm::tir::Let b(var, bank_index, var);
      bank_index_map_.Set(var, buffer->name);
      Bind(buffer->bank_index, b, buffer->name + ".bank_index");
    }

    // Step 2.3. Check and update strides
    // Check if target buffer strides are defined
    ICHECK(source->region.size() >= buffer->shape.size());
    int offset = source->region.size() - buffer->shape.size();
    if (!buffer->strides.empty()) {
      ICHECK_EQ(buffer->strides.size(), buffer->shape.size());
      if (source_buffer->strides.empty()) {
        PrimExpr stride = make_const(buffer->strides.back().dtype(), 1);
        for (size_t i = buffer->shape.size(); i > 0; --i) {
          const PrimExpr& shape = source_buffer->shape[i - 1 + offset];
          Bind(buffer->strides[i - 1], stride, buffer->name + ".strides_" + std::to_string(i - 1));
          stride *= shape;
        }
      } else {
        ICHECK_EQ(buffer->shape.size() + offset, source_buffer->strides.size());
        for (size_t i = buffer->shape.size(); i > 0; --i) {
          const PrimExpr& stride = source_buffer->strides[i - 1 + offset];
          Bind(buffer->strides[i - 1], stride, buffer->name + ".strides_" + std::to_string(i - 1));
        }
      }
    }

    // Step 2.4. Check and update shape
    for (size_t i = 0; i < buffer->shape.size(); ++i) {
      const Range& range = source->region[i + offset];
      Bind(buffer->shape[i], range->extent, buffer->name + ".shape_" + std::to_string(i));
    }
  }

  void Bind(const PrimExpr& arg, PrimExpr value, const std::string& arg_name = "argument") {
    CHECK_EQ(arg.dtype(), value.dtype())
        << "The data type mismatched: " << arg->dtype << " vs. " << value->dtype;
    // Handle recursive case
    value = Substitute(std::move(value), var_map_);
    if (arg->IsInstance<VarNode>()) {
      Var v = Downcast<Var>(arg);
      auto it = var_map_.find(v);
      if (it == var_map_.end()) {
        var_map_.Set(v, value);
        analyzer_.Bind(v, value);
      } else {
        AssertBinding((*it).second, value, arg_name);
      }
    } else {
      AssertBinding(arg, value, arg_name);
    }
  }

  void AssertBinding(const PrimExpr& lhs, const PrimExpr& rhs,
                     const std::string& arg_name = "argument") {
    CHECK(analyzer_.CanProve(lhs == rhs)) << "The buffer match constraint for " << arg_name
                                          << " unmet: " << lhs << "==" << rhs << ".";
  }

 public:
  Map<Var, String> in_bank_elem_offset_map_;
  Map<Var, String> bank_index_map_;

 private:
  /*! \brief Buffer region mapping. */
  Map<Buffer, BufferRegion> match_buffers_;
  Map<Buffer, BufferRegion> match_buffers_global_;
  /*! \brief Var mapping for buffer signature (data, strides, element_offset, etc.) */
  Map<Var, PrimExpr> var_map_;
  /*! \brief The analyzer */
  arith::Analyzer analyzer_;
};

class TermSeparator : public ExprVisitor {
 public:
  PrimExpr Assembler() {
    PrimExpr e;
    for (auto t : terms) {
      if (!e.defined())
        e = t.first * t.second;
      else
        e += t.first * t.second;
    }
    return e;
  }
  void VisitExpr_(const AddNode* op) final {
    // Extract and store the terms
    this->VisitExpr(op->a);
    this->VisitExpr(op->b);
  }

  void VisitExpr_(const MulNode* op) final {
    // apply distribution law if possible
    if (op->a.as<AddNode>()) {
      auto l = Downcast<Add>(op->a)->a * op->b;
      auto r = Downcast<Add>(op->a)->b * op->b;
      this->VisitExpr(l + r);
    } else if (op->b.as<AddNode>()) {
      auto l = op->a * Downcast<Add>(op->b)->a;
      auto r = op->a * Downcast<Add>(op->b)->b;
      this->VisitExpr(l + r);
    } else if (op->a.as<MulNode>()) {
      // evaluate constants on the right
      auto e = Downcast<Mul>(op->a)->a;
      auto c1 = Downcast<IntImm>(Downcast<Mul>(op->a)->b);
      auto c2 = Downcast<IntImm>(op->b);
      auto c = IntImm(c1.dtype(), c1->value * c2->value);
      this->VisitExpr(e * c);
    } else {
      terms.Set(Downcast<Var>(op->a), op->b.as<IntImmNode>()->value);
    }
  }

  void VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    terms.Set(v, 1);
  }

 public:
  Map<Var, Integer> terms;
};

class Test : public StmtExprMutator {
 public:
  explicit Test(Map<Var, String>& in_bank_elem_offset_map, Map<Var, String>& bank_index_map)
      : StmtExprMutator() {
    this->in_bank_elem_offset_map_ = std::move(in_bank_elem_offset_map);
    this->bank_index_map_ = std::move(bank_index_map);
  }
  PrimExpr VisitExpr_(const LetNode* op) final {
    Let l = GetRef<Let>(op);
    {
      auto it = in_bank_elem_offset_map_.find(op->var);
      if (it != in_bank_elem_offset_map_.end()) {
        TermSeparator sep;
        sep(op->value);

        String buffer_name = (*it).second;
        auto it2 = alloc_loop_level_.find(buffer_name);
        if (it2 != alloc_loop_level_.end()) {
          int loop_level = (*it2).second->value;
          ICHECK(buffer_size_.count(buffer_name));
          int scale = buffer_size_[buffer_name].IntValue();
          for (int i = loop_level - 1; i >= 0; i--) {
            const ForNode* op = loop_order_[i];
            if (op->annotations.find("bank") != op->annotations.end()) {
              if (sep.terms.count(op->loop_var) > 0) {
                sep.terms.Set(op->loop_var, 0);
              }
            } else {
              if (sep.terms.find(op->loop_var) != sep.terms.end()) {
                sep.terms.Set(op->loop_var, scale);
                scale *= Downcast<IntImm>(loop_range_[i].max() + 1)->value;
              }
            }
          }
        }
        return sep.Assembler();
      }
    }
    {
      auto it = bank_index_map_.find(op->var);
      if (it != in_bank_elem_offset_map_.end()) {
        String buffer_name = (*it).second;
        auto it2 = alloc_loop_level_.find(buffer_name);
        if (it2 != alloc_loop_level_.end()) {
          int loop_level = (*it2).second->value;
          ICHECK(buffer_size_.count(buffer_name));
          PrimExpr index = make_const(DataType::Int(32), 0);
          for (int i = 0; i < loop_level; i++) {
            const ForNode* op = loop_order_[i];
            if (op->annotations.find("bank") != op->annotations.end()) {
              index = op->extent * index + op->loop_var;
            }
          }
          return index;
          // return op->value;
        }
      }
    }
    return std::move(l);
  }

  bool IsBankBinded(const ForNode* op) {
    return op->annotations.find("bank") != op->annotations.end() || 
      (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() && 
        runtime::ThreadScope::Create(op->thread_binding.value()->thread_tag).rank == 0);
  }

  void RewriteBufferAccess(String buffer_name, int64_t ndim, Array<PrimExpr>* global_indices,
                           const PrimExpr& flattened) {
    Array<PrimExpr> new_global_indices;

    TermSeparator sep;
    sep(flattened);
    auto it2 = alloc_loop_level_.find(buffer_name);
    if (it2 != alloc_loop_level_.end()) {
      int loop_level = (*it2).second->value;
      ICHECK(buffer_size_.count(buffer_name));
      int scale = buffer_size_[buffer_name].IntValue();
      for (int i = loop_level - 1; i >= 0; i--) {
        const ForNode* op = loop_order_[i];
        if (IsBankBinded(op)) {
          if (sep.terms.count(op->loop_var) > 0) {
            // VLOG(2) << "Banking " << op->loop_var << " for " << buffer_name;
            sep.terms.Set(op->loop_var, 0);
          }
        } else {
          if (sep.terms.find(op->loop_var) != sep.terms.end()) {
            sep.terms.Set(op->loop_var, scale);
            scale *= Downcast<IntImm>(loop_range_[i].max() + 1)->value;
            // VLOG(2) << "Rewrite " << op->loop_var << " for " << buffer_name << " to " << scale;
          }
        }
      }
    }
    for (int i = 0; i < ndim - 1; i++) {
      new_global_indices.push_back(0);
    }
    new_global_indices.push_back(sep.Assembler());
    // VLOG(2) << "RewriteBufferAccess called for \n\t" << buffer_name << " with " << *global_indices << " to \n\t" << new_global_indices;
    *(global_indices) = std::move(new_global_indices);
  }

  Stmt VisitStmt_(const BufferStoreNode* _op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
    BufferStoreNode* op = store.CopyOnWrite();
    if (op->buffer.scope() != "local") {
      return std::move(store);
    }
    const Buffer& buffer = op->buffer;
    auto ndim = buffer->shape.size();
    PrimExpr flattened = op->global_indices[0];

    for (int i = 1; i < ndim; i++) {
      flattened = flattened * buffer->shape[i] + op->global_indices[i];
    }
    RewriteBufferAccess(buffer->name, ndim, &op->global_indices, flattened);

    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
    BufferLoadNode* op = load.CopyOnWrite();
    if (op->buffer.scope() != "local") {
      return std::move(load);
    }
    const Buffer& buffer = op->buffer;
    auto ndim = buffer->shape.size();
    PrimExpr flattened = op->global_indices[0];

    for (int i = 1; i < ndim; i++) {
      flattened = flattened * buffer->shape[i] + op->global_indices[i];
    }
    RewriteBufferAccess(buffer->name, ndim, &op->global_indices, flattened);

    return std::move(load);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    for (auto buffer : op->alloc_buffers) {
      alloc_loop_level_.Set(buffer->name, loop_order_.size());
      size_t size = 1;
      for (PrimExpr dim : buffer->shape) {
        int64_t pval = Downcast<IntImm>(dim)->value;
        size *= pval;
      }
      buffer_size_.Set(buffer->name, size);
    }
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    return stmt;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    PrimExpr extent = op->extent;
    if (op->annotations.find("bank") != op->annotations.end()) {
      extent = IntImm(DataType::Int(32), 1);
    }
    if (op->kind == ForKind::kThreadBinding) {
      extent = IntImm(DataType::Int(32), 1);
    }
    Range loop_range = Range::FromMinExtent(op->min, extent);
    loop_range_.push_back(arith::IntSet::FromRange(loop_range));
    loop_order_.push_back(op);
    Stmt res = StmtExprMutator::VisitStmt_(op);
    loop_range_.pop_back();
    loop_order_.pop_back();
    return res;
  }

 private:
  Map<Var, String> in_bank_elem_offset_map_;
  Map<Var, String> bank_index_map_;
  std::vector<arith::IntSet> loop_range_;
  std::vector<const ForNode*> loop_order_;
  Map<String, Integer> alloc_loop_level_;
  Map<String, Integer> buffer_size_;
  Map<Buffer, PrimExpr> global_indices_map_;
};

PrimFunc LowerMatchBuffer(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  auto pass = MatchBufferLower(func);
  fptr->body = pass(std::move(fptr->body));
  fptr->body = Test(pass.in_bank_elem_offset_map_, pass.bank_index_map_)(std::move(func->body));
  return func;
}

namespace transform {

Pass LowerMatchBuffer() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerMatchBuffer(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerMatchBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerMatchBuffer").set_body_typed(LowerMatchBuffer);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
