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
 * \file codegen_hbmpim.h
 * \brief Generate HBMPIM device code.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_HBMPIM_H_
#define TVM_TARGET_SOURCE_CODEGEN_HBMPIM_H_

#include <tvm/target/codegen.h>

#include <string>
#include <unordered_map>

#include "codegen_opencl.h"

namespace tvm {
namespace codegen {

class CodeGenHBMPIM final : public CodeGenOpenCL {
 public:
  CodeGenHBMPIM();
  void PreFunctionBody(const PrimFunc& f) final;                                      // NOLINT(*)
  void PostFunctionBody(const PrimFunc& f) final;                                     // NOLINT(*)
  void PrintExtraFuncParams(const PrimFunc& f) final;                                 // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) final;                        // NOLINT(*)
  void VisitStmt_(const ForNode* op) final;                                           // NOLINT(*)
  void VisitStmt_(const BufferStoreNode* op) final;                                   // NOLINT(*)
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os);                        // NOLINT(*)
  void VisitStmt_(const AttrStmtNode* op) final;                                      // NOLINT(*)
  void BindThreadIndex(const IterVar& iv) final;                                      // NOLINT(*)
  std::ostringstream& Stream();                                                       // NOLINT(*)
  void PrintPIMPrologue();                                                            // NOLINT(*)
  void PrintPIMEpilogue();                                                            // NOLINT(*)
  void PrintChangeGemvHabHabPim();                                                    // NOLINT(*)
  void PrintChangeGemvHabPimHab();                                                    // NOLINT(*)
  std::string GetBufferRef(DataType t, const BufferNode* buffer, std::string index);  // NOLINT(*)
  std::string Finish() final;                                                         // NOLINT(*)

 private:
  int pim_scope_;
  bool skip_scope_ = false;
  const int crf_size_ = 32;
  Map<Var, IntImm> bank_ordering_map_;
  Map<Var, IntImm> bank_extent_map_;
  Array<IterVar> thread_vars_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_HBMPIM_H_
