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
 * \file codegen_upmem.h
 * \brief Generate Upmem device code.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_UPMEM_H_
#define TVM_TARGET_SOURCE_CODEGEN_UPMEM_H_

#include <tvm/target/codegen.h>

#include <stack>
#include <string>
#include <unordered_map>

#include "codegen_c.h"
#include "codegen_source_base.h"

namespace tvm {
namespace codegen {

class CodeGenUpmem final : public CodeGenC {
 public:
  CodeGenUpmem();
  std::string Finish();

  // override print thread tag.                      // NOLINT(*)
  void AddFunction(const PrimFunc& f);
  void PreFunctionBody(const PrimFunc& f) final;
  // NOLINT(*)
  void BindThreadIndex(const IterVar& iv) final;  // NOLINT(*)
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;
  void PrintStorageSync(const CallNode* op) final;  // NOLINT(*)
  // the address of load/store

  // overload visitor
  void VisitStmt_(const AllocateNode* op) final;  // NOLINT(*)
  void VisitStmt_(const ForNode* op) final;

  void VisitExpr_(const BufferLoadNode* op, std::ostream& os) final;
  void VisitStmt_(const BufferStoreNode* op) final;

 private:
  // whether enable fp16 and fp64 extension
  bool enable_fp16_{false};
  bool enable_fp64_{false};
  // Whether to enable atomics extension.
  bool enable_atomics_{false};
  // Whether to enable sampler or sampler-less texture reads,
  // where the choice depends on the Upmem version used.
  bool enable_compliant_texture_reads_{false};

  PrimExpr alloc_global_index;

  std::stack<std::string> for_tags;
  // Mapping from buffer to allocation size.
  // Useful to track when a scalar store of a vectorized texture load is required.
  std::unordered_map<const Object*, size_t> allocation_size_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_UPMEM_H_
