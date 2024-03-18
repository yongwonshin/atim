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
 * \file upmem_common.h
 * \brief Common utilities for UPMEM
 */
#ifndef TVM_RUNTIME_UPMEM_UPMEM_COMMON_H_
#define TVM_RUNTIME_UPMEM_UPMEM_COMMON_H_

extern "C" {
#include <dpu.h>
}

#include <tvm/runtime/packed_func.h>

#include <chrono>
#include <string>

#include "../workspace_pool.h"

namespace tvm {
namespace runtime {

#define UPMEM_DRIVER_CALL(x)                                                               \
  {                                                                                        \
    dpu_error_t err = x;                                                                   \
    if (x != DPU_OK) {                                                                     \
      LOG(FATAL) << "UPMEM Error: " #x " failed with error: " << dpu_error_to_string(err); \
    }                                                                                      \
  }

#define UPMEM_CALL(x)                                               \
  {                                                                 \
    dpu_error_t err = x;                                            \
    ICHECK(err == DPU_OK) << "UPMEM: " << dpu_error_to_string(err); \
  }

class UPMEMDeviceAPI final : public DeviceAPI {
 public:
  struct DpuVarInfo {
    int32_t bytes;
    std::string var_name;
  };
  static UPMEMDeviceAPI* Global();

  void SetDevice(Device dev) final {}

  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final;

  void StreamSync(Device dev, TVMStreamHandle stream) final {}

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;

  void FreeWorkspace(Device dev, void* data) final;

  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint,
                       Optional<String> mem_scope = NullOpt) final;

  void FreeDataSpace(Device dev, void* ptr);

  int AcquireResources(TVMArgs banks);

  int ReleaseResources();

  void SetPimMemoryEntry(void* handle, std::string var_name, DataType dtype, int size,
                         int bank_index);

  void ErasePimMemoryEntry(void* handle);

  int TransferHostToDevice(void* handle, uint64_t host_addr, uint64_t in_bank_addr, int bank_idx,
                           int size);

  int TransferDeviceToHost(void* handle, uint64_t host_addr, uint64_t in_bank_addr, int bank_idx,
                           int size);

  int Broadcast(void* handle, uint64_t host_addr, int size);

  int InitXfer(void* handle, uint64_t in_bank_addr, uint64_t size, int direction);

  int BindXfer(int bank_index, uint64_t host_addr, uint64_t size);

  int PushXfer();

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final;

  int GetBytes(void* handle) { return dpu_addr_ptr[handle].bytes; }

  std::string GetSymbolName(void* handle) { return dpu_addr_ptr[handle].var_name; }

  void* HostOffset(void* handle, uint64_t offset) {
    return (char*)handle + (size_t)offset * GetBytes(handle);
  }

 public:
  dpu_set_t dpu_set;

  std::unordered_map<int, dpu_set_t> dpu_entry;
  std::unordered_map<void*, DpuVarInfo> dpu_addr_ptr;

  void* xfer_handle = nullptr;
  uint64_t xfer_offset = 0;
  _dpu_xfer_t xfer_direction = DPU_XFER_TO_DPU;
  uint64_t xfer_bulk_size = 0;

  struct TemporaryMapInfo {
    void* temp_ptr;
    void* dest_ptr;
    uint64_t size;
  };

  std::unordered_map<int, TemporaryMapInfo> d2h_temp;

  std::chrono::high_resolution_clock::time_point acquire_start;
  std::chrono::high_resolution_clock::time_point kernel_start;
  std::chrono::high_resolution_clock::time_point kernel_end;
  std::chrono::high_resolution_clock::time_point release_end;
  void* recent_host_address;
};

class UPMEMThreadEntry {  // is it necessary?
 public:
  /*! \brief The cuda stream */
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  UPMEMThreadEntry();
  // get the threadlocal workspace
  static UPMEMThreadEntry* ThreadLocal();
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_UPMEM_UPMEM_COMMON_H_
