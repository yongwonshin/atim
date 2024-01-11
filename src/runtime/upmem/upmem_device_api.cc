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
 * \file upmem_device_api.cc
 * \brief UPMEM specific API
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <cstring>

#include "upmem_common.h"

namespace tvm {
namespace runtime {

void UPMEMDeviceAPI::GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) {
  // todo-stonerdk: not that significant so far
  int value = 0;
  switch(kind) {
    case kExist: { *rv = 1; }
    case kMaxThreadsPerBlock: { value = 24; break; } // hw.dpu.nr_of_threads
    // case kComputeVersion: dpu_get_description()->hw
    // case kDeviceName: hw.signature.chip_id
    // case kMaxClockRate: hw.timings.fck_frequency_in_mhz * 1000000
    // case kMultiProcessorCount: hw.topology.nr_of_control_interfaces
    case kMaxThreadDimensions: { value = 1; break; }
    default: break;
  }
  *rv = value;
}

// NOTE. These function in UPMEMDeviceAPI is not used for in-bank environment
// In-bank environment transfer is accomplished by custom intrinsics registered below the code
// This functions the borrow those of cpu_device_api. Thus, UPMEM function is called with ndarray in CPU,
// and TVMBackendAllocWorkspace, TVMBackendFreeWorkspace in host llvm code calls CPU's
// ignore MSC, Android first. Reuse memalloc in cpu_device_api first.

void* UPMEMDeviceAPI::AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint, Optional<String> mem_scope) {
  void *ptr;
  int ret = posix_memalign(&ptr, alignment, nbytes);
  if (ret != 0) throw std::bad_alloc();
  return ptr;
}

void UPMEMDeviceAPI::FreeDataSpace(Device dev, void* ptr) {
  free(ptr);
}

void UPMEMDeviceAPI::CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                    Device dev_from, Device dev_to, DLDataType type_hint,
                    TVMStreamHandle stream) {
  memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
}

struct UPMEMWorkspacePool : public WorkspacePool {
  UPMEMWorkspacePool() : WorkspacePool(static_cast<DLDeviceType>(kDLUPMEM), UPMEMDeviceAPI::Global()) {}
};

void* UPMEMDeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  return dmlc::ThreadLocalStore<UPMEMWorkspacePool>::Get()->AllocWorkspace(dev, size);
}

void UPMEMDeviceAPI::FreeWorkspace(Device dev, void* data) {
  dmlc::ThreadLocalStore<UPMEMWorkspacePool>::Get()->FreeWorkspace(dev, data);
}

UPMEMDeviceAPI* UPMEMDeviceAPI::Global() 
{
  static auto* inst = new UPMEMDeviceAPI();
  return inst;
}

int UPMEMDeviceAPI::AcquireResources(int32_t bank_num) {
  if (!dpu_entry.empty()) {
    LOG(FATAL) << "DPU resources already acquired. Release them first.";
    return 1;
  }
  VLOG(3) << "dpu_alloc(" << bank_num << ", NULL, &dpu_set)";
  UPMEM_CALL(dpu_alloc(bank_num, NULL, &(dpu_set)));

  uint32_t nr_dpus;
  UPMEM_CALL(dpu_get_nr_dpus(dpu_set, &nr_dpus));
  if (nr_dpus != static_cast<uint32_t>(bank_num)) {
    LOG(FATAL) << "DPU resource allocation failed. Requested " << bank_num << " but got " << nr_dpus;
    return 1;
  }

  UPMEM_CALL(dpu_load(dpu_set, "./temp", NULL)); // todo-stonerdk: hack
  dpu_set_t dpu; int32_t i;
  DPU_FOREACH(dpu_set, dpu, i)
    dpu_entry[i] = dpu;
  return 0;
}

int UPMEMDeviceAPI::ReleaseResources() {
  if (dpu_entry.empty()) {
    LOG(FATAL) << "DPU resource is already empty.";
    return 1;
  } else {
    VLOG(3) << "dpu_free(dpu_set)";
    UPMEM_CALL(dpu_free(dpu_set));
  }
  dpu_entry.clear();
  dpu_addr_ptr.clear();
  return 0;
}

void UPMEMDeviceAPI::SetPimMemoryEntry(void* handle, std::string var_name, DataType dtype, int size, int bank_index) {
  dpu_addr_ptr[handle] = { dtype.bytes(), var_name };
}

void UPMEMDeviceAPI::ErasePimMemoryEntry(void* handle) {
  dpu_addr_ptr.erase(handle);
}

int UPMEMDeviceAPI::TransferHostToDevice(void* handle, uint64_t host_addr, uint64_t in_bank_addr, int bank_index, int size) {
  VLOG(3) << "dpu_copy_to(" << bank_index << ", "
    << std::string(GetSymbolName(handle)) << ", " << in_bank_addr << ", " << host_addr
    << ", " << size << " * " << GetBytes(handle) << ")";
  UPMEM_CALL(dpu_copy_to(dpu_entry[bank_index], GetSymbolName(handle).c_str(), 
    static_cast<uint32_t>(in_bank_addr) * GetBytes(handle), HostOffset(handle, host_addr), size * GetBytes(handle)));
  return 0;
}

int UPMEMDeviceAPI::TransferDeviceToHost(void* handle, uint64_t host_addr, uint64_t in_bank_addr, int bank_idx, int size) {
  // not implemented yet
  return 0;
}

int UPMEMDeviceAPI::Broadcast(void* handle, uint64_t host_addr, int size) {
  VLOG(3) << "dpu_broadcast_to(dpu_set, " << GetSymbolName(handle) << ", " << host_addr << ", " 
    << size << " * " << GetBytes(handle) << ", DPU_XFER_DEFAULT)";
  UPMEM_CALL(dpu_broadcast_to(dpu_set, GetSymbolName(handle).c_str(), 
    0, HostOffset(handle, host_addr), size * GetBytes(handle), DPU_XFER_DEFAULT));
  return 0;
}

int UPMEMDeviceAPI::PrepareXfer(void* handle, uint64_t host_addr, int bank_index) {
  VLOG(3) << "dpu_prepare_xfer(" << bank_index << ", " << handle << " + (" << host_addr << ") * " 
     << GetBytes(handle) << ")"; 
  UPMEM_CALL(dpu_prepare_xfer(dpu_entry[bank_index], HostOffset(handle, host_addr)));
  return 0;
}

int UPMEMDeviceAPI::PushXfer(void* handle, uint64_t in_bank_addr, int size, int direction) {
  _dpu_xfer_t xfer = direction == 1 ? DPU_XFER_TO_DPU : DPU_XFER_FROM_DPU;
  uint32_t in_bank_addr_cast = static_cast<uint32_t>(in_bank_addr);
  VLOG(3) << "dpu_push_xfer(dpu_set, " 
    << (xfer == DPU_XFER_TO_DPU ? "DPU_XFER_TO_DPU" : "DPU_XFER_FROM_DPU") << ", "
    << std::string(GetSymbolName(handle)) << ", "
    << in_bank_addr_cast << ", "
    << size << " * " << GetBytes(handle) << ", DPU_XFER_DEFAULT);";
  UPMEM_CALL(dpu_push_xfer(dpu_set, xfer, GetSymbolName(handle).c_str(), 
    in_bank_addr_cast * GetBytes(handle), size * GetBytes(handle), DPU_XFER_DEFAULT));
  return 0;
}


TVM_REGISTER_GLOBAL("device_api.upmem").set_body([](TVMArgs args, TVMRetValue* rv) {
  UPMEMDeviceAPI* api = UPMEMDeviceAPI::Global();
  *rv = static_cast<void*>(api);
});

TVM_REGISTER_GLOBAL("device_api.upmem.acquire_resources").set_body([](TVMArgs args, TVMRetValue* rv) {
  int32_t bank_num = args[0];
  *rv = static_cast<int>(UPMEMDeviceAPI::Global()->AcquireResources(bank_num));
});

TVM_REGISTER_GLOBAL("device_api.upmem.release_resources").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = static_cast<int>(UPMEMDeviceAPI::Global()->ReleaseResources());
});

TVM_REGISTER_GLOBAL("device_api.upmem.pim_allocate_memory").set_body([](TVMArgs args, TVMRetValue* rv) {
  void* buffer_handle = args[0];
  std::string var_name = args[1];
  std::string type = args[2];
  DataType dtype = DataType(String2DLDataType(type));
  int32_t size = args[3];
  int32_t bank_index = args[4];
  UPMEMDeviceAPI::Global()->SetPimMemoryEntry(buffer_handle, var_name, dtype, size, bank_index);
});

TVM_REGISTER_GLOBAL("device_api.upmem.pim_free_memory").set_body([](TVMArgs args, TVMRetValue* rv) {
  void* handle = args[0];
  UPMEMDeviceAPI::Global()->ErasePimMemoryEntry(handle);
});

TVM_REGISTER_GLOBAL("device_api.upmem.pim_transfer_host_to_device").set_body([](TVMArgs args, TVMRetValue* rv) {
  void* handle = args[0];
  uint64_t host_address = args[1];
  uint64_t in_bank_addr = args[2];
  int bank_index = args[3];
  int size = args[4];
  *rv = static_cast<int>(UPMEMDeviceAPI::Global()
    ->TransferHostToDevice(handle, host_address, in_bank_addr, bank_index, size));
});

TVM_REGISTER_GLOBAL("device_api.upmem.pim_transfer_device_to_host").set_body([](TVMArgs args, TVMRetValue* rv) {
  void* handle = args[0];
  uint64_t host_address = args[1];
  uint64_t in_bank_addr = args[2];
  int bank_index = args[3];
  int size = args[4];
  *rv = static_cast<int>(UPMEMDeviceAPI::Global()
    ->TransferDeviceToHost(handle, host_address, in_bank_addr, bank_index, size));
});

TVM_REGISTER_GLOBAL("device_api.upmem.pim_broadcast").set_body([](TVMArgs args, TVMRetValue* rv) {
  void* handle = args[0];
  uint64_t host_address = args[1];
  int size = args[2];
  *rv = static_cast<int>(UPMEMDeviceAPI::Global()->Broadcast(handle, host_address, size));
});

TVM_REGISTER_GLOBAL("device_api.upmem.dpu_prepare_parallel_transfer").set_body([](TVMArgs args, TVMRetValue* rv) {
  void* handle = args[0];
  uint64_t host_address = args[1];
  int bank_index = args[2];
  *rv = static_cast<int>(UPMEMDeviceAPI::Global()->PrepareXfer(handle, host_address, bank_index));
});

TVM_REGISTER_GLOBAL("device_api.upmem.dpu_parallel_transfer").set_body([](TVMArgs args, TVMRetValue* rv) {
  void* handle = args[0];
  uint64_t in_bank_addr = args[1];
  int size = args[2];
  int direction = args[3];
  *rv = static_cast<int>(UPMEMDeviceAPI::Global()->PushXfer(handle, in_bank_addr, size, direction));
});

class UPMEMTimerNode : public TimerNode {
 public:
  virtual void Start() { start = 0; }
  virtual void Stop() { end = 0; }
  virtual int64_t SyncAndGetElapsedNanos() { return 0; }
  virtual ~UPMEMTimerNode() {}

  static constexpr const char* _type_key = "UPMEMTimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(UPMEMTimerNode, TimerNode);

 private:
  uint64_t start, end;
};

TVM_REGISTER_OBJECT_TYPE(UPMEMTimerNode);

TVM_REGISTER_GLOBAL("profiling.timer.upmem").set_body_typed([](Device dev) {
  return Timer(make_object<UPMEMTimerNode>());
});

}  // namespace runtime
}  // namespace tvm
