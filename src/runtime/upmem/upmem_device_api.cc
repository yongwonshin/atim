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

#include <chrono>
#include <cstring>

#include "upmem_common.h"

namespace tvm {
namespace runtime {

void UPMEMDeviceAPI::GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) {
  // todo-stonerdk: not that significant so far
  int value = 0;
  switch (kind) {
    case kExist: {
      *rv = 1;
    }
    case kMaxThreadsPerBlock: {
      value = 24;
      break;
    }  // hw.dpu.nr_of_threads
    // case kComputeVersion: dpu_get_description()->hw
    // case kDeviceName: hw.signature.chip_id
    // case kMaxClockRate: hw.timings.fck_frequency_in_mhz * 1000000
    // case kMultiProcessorCount: hw.topology.nr_of_control_interfaces
    case kMaxThreadDimensions: {
      value = 1;
      break;
    }
    default:
      break;
  }
  *rv = value;
}

// NOTE. These function in UPMEMDeviceAPI is not used for in-bank environment
// In-bank environment transfer is accomplished by custom intrinsics registered below the code
// This functions the borrow those of cpu_device_api. Thus, UPMEM function is called with ndarray in
// CPU, and TVMBackendAllocWorkspace, TVMBackendFreeWorkspace in host llvm code calls CPU's ignore
// MSC, Android first. Reuse memalloc in cpu_device_api first.

void* UPMEMDeviceAPI::AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                     DLDataType type_hint, Optional<String> mem_scope) {
  void* ptr;
  VLOG(3) << "Attempt to alloc dataspace " << nbytes << " bytes with alignment " << alignment;
  int ret = posix_memalign(&ptr, 32, nbytes);
  if (ret != 0) throw std::bad_alloc();
  return ptr;
}

void UPMEMDeviceAPI::FreeDataSpace(Device dev, void* ptr) {
  VLOG(3) << "Attempt to free dataspace " << ptr;
  free(ptr);
}

void UPMEMDeviceAPI::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                    size_t to_offset, size_t size, Device dev_from, Device dev_to,
                                    DLDataType type_hint, TVMStreamHandle stream) {
  memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
}

struct UPMEMWorkspacePool : public WorkspacePool {
  UPMEMWorkspacePool()
      : WorkspacePool(static_cast<DLDeviceType>(kDLUPMEM), UPMEMDeviceAPI::Global()) {}
};

void* UPMEMDeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  return dmlc::ThreadLocalStore<UPMEMWorkspacePool>::Get()->AllocWorkspace(dev, size);
}

void UPMEMDeviceAPI::FreeWorkspace(Device dev, void* data) {
  dmlc::ThreadLocalStore<UPMEMWorkspacePool>::Get()->FreeWorkspace(dev, data);
}

UPMEMDeviceAPI* UPMEMDeviceAPI::Global() {
  static auto* inst = new UPMEMDeviceAPI();
  return inst;
}

int UPMEMDeviceAPI::AcquireResources(TVMArgs args) {
  int32_t bank_num = 1;
  std::vector<int32_t> bank_vec;
  int32_t n_bank_args = args.num_args - 1;
  for (int32_t i = 0; i < n_bank_args; i++) {
    int32_t n = args[i];
    bank_vec.push_back(n);
    bank_num *= n;
  }
  std::string uuid = std::to_string(static_cast<int>(args[n_bank_args]));  // last argument

  if (!dpu_entry.empty()) {
    return 0;
  }

  VLOG(3) << "dpu_alloc(" << bank_num << ", disableSafeChecks=1, NULL, &dpu_set)";
  int attempt = 0;
  for (; attempt< 10; attempt++) {
    if (dpu_alloc(bank_num, "disableSafeChecks=1", &dpu_set) == DPU_OK) {
      break;
    }
    if (attempt == 9) {
      LOG(FATAL) << "Error: Unable to allocate DPUs";
    }
  }
  printf("Attempt in %d\n", attempt);

  uint32_t nr_dpus;
  UPMEM_CALL(dpu_get_nr_dpus(dpu_set, &nr_dpus));
  if (nr_dpus != static_cast<uint32_t>(bank_num)) {
    LOG(FATAL) << "DPU resource allocation failed. Requested " << bank_num << " but got "
               << nr_dpus;
    dpu_free(dpu_set);
    return 1;
  }

  UPMEM_CALL(
      dpu_load(dpu_set, ("./temp-" + uuid).c_str(), NULL));  // TODO[ywshin]: who cleans the binary?
  dpu_set_t dpu;
  int32_t i;

  DPU_FOREACH(dpu_set, dpu, i) { dpu_entry[i] = dpu; }
  int** dpu_indices = new int*[nr_dpus];
  for (int j = 0; j < nr_dpus; j++) {
    dpu_indices[j] = new int[n_bank_args];
    if (n_bank_args == 1) {
      dpu_indices[j] = new int[1]{j};
    } else if (n_bank_args == 2) {
      dpu_indices[j] = new int[2]{j / bank_vec[1], j % bank_vec[1]};
    } else if (n_bank_args == 3) {
      dpu_indices[j] = new int[3]{(j / bank_vec[2]) / bank_vec[1], (j / bank_vec[2]) % bank_vec[1],
                                  j % bank_vec[2]};
    } else {
      LOG(FATAL) << "bank argument number should be equal or less than 3. " << n_bank_args;
    }
  }

  for (int j = 0; j < nr_dpus; j++) {
    UPMEM_CALL(dpu_prepare_xfer(dpu_entry[j], &dpu_indices[j][0]));
  }
  UPMEM_CALL(
      dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "blockIdx", 0, 4 * n_bank_args, DPU_XFER_DEFAULT));

  return 0;
}

UPMEMDeviceAPI::~UPMEMDeviceAPI() { ReleaseResources(); }

int UPMEMDeviceAPI::ReleaseResources() {
  if (dpu_entry.empty()) {
    return 1;
  } else {
    VLOG(3) << "dpu_free(dpu_set)";
    UPMEM_CALL(dpu_free(dpu_set));
  }
  dpu_entry.clear();
  dpu_addr_ptr.clear();
  return 0;
}

void UPMEMDeviceAPI::SetPimMemoryEntry(void* handle, std::string var_name, DataType dtype, int size,
                                       int bank_index) {
  dpu_addr_ptr[handle] = {dtype.bytes(), var_name};
}

void UPMEMDeviceAPI::ErasePimMemoryEntry(void* handle) {
  VLOG(3) << "Attempt to erase PIM memory entry for " << handle;
  if (dpu_addr_ptr.find(handle) != dpu_addr_ptr.end()) {
    dpu_addr_ptr.erase(handle);
  }
}

int UPMEMDeviceAPI::TransferHostToDevice(void* handle, uint64_t host_addr, uint64_t in_bank_addr,
                                         int bank_index, int size) {
  VLOG(3) << "dpu_copy_to(" << bank_index << ", " << std::string(GetSymbolName(handle)) << ", "
          << in_bank_addr << ", " << host_addr << ", " << size << " * " << GetBytes(handle) << ")";
  UPMEM_CALL(dpu_copy_to(dpu_entry[bank_index], GetSymbolName(handle).c_str(),
                         static_cast<uint32_t>(in_bank_addr) * GetBytes(handle),
                         HostOffset(handle, host_addr), size * GetBytes(handle)));
  return 0;
}

int UPMEMDeviceAPI::TransferDeviceToHost(void* handle, uint64_t host_addr, uint64_t in_bank_addr,
                                         int bank_idx, int size) {
  // not implemented yet
  return 0;
}

void UPMEMDeviceAPI::Timestamp(std::string type) {
  // std::cerr << "One or more DPUs are in fault" << std::endl;
  if (type == "start") {
    this->entire_start = std::chrono::high_resolution_clock::now();
  } else if (type == "end") {
    this->entire_end = std::chrono::high_resolution_clock::now();
  } else {
    LOG(FATAL) << "Invalid type: " << type;
  }
}

size_t UPMEMDeviceAPI::ElapsedTime(std::string type) {
  std::chrono::high_resolution_clock::time_point start, end;
  if (type == "entire") {
    end = this->entire_end;
    start = this->entire_start;
  } else if (type == "kernel") {
    end = this->kernel_end;
    start = this->kernel_start;
  } else if (type == "after_kernel") {
    end = this->entire_end;
    start = this->kernel_end;
  } else if (type == "before_kernel") {
    end = this->kernel_start;
    start = this->entire_start;
  } else if (type == "d2h") {
    return static_cast<double>(this->d2h_time);
  } else if (type == "after_d2h") {
    end = this->entire_end;
    start = this->last_d2h_end;
  } else {
    LOG(FATAL) << "Invalid type: " << type;
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int UPMEMDeviceAPI::Broadcast(void* handle, uint64_t host_addr, int size) {
  VLOG(3) << "dpu_broadcast_to(dpu_set, " << GetSymbolName(handle) << ", " << host_addr << ", "
          << size << " * " << GetBytes(handle) << ", DPU_XFER_DEFAULT)";
  UPMEM_CALL(dpu_broadcast_to(dpu_set, GetSymbolName(handle).c_str(), 0,
                              HostOffset(handle, host_addr), size * GetBytes(handle),
                              DPU_XFER_DEFAULT));
  return 0;
}

int UPMEMDeviceAPI::InitXfer(void* handle, uint64_t in_bank_addr, uint64_t size, int direction) {
  xfer_handle = handle;
  xfer_offset = in_bank_addr;
  xfer_direction = direction == 1 ? DPU_XFER_TO_DPU : DPU_XFER_FROM_DPU;
  int padded_size = ((size * GetBytes(handle) + 7) / 8) * 8 / GetBytes(handle);  // TODO-stonerdk
  xfer_bulk_size = padded_size;
  VLOG(3) << "InitXfer(" << handle << ", " << in_bank_addr << ", " << size << "(" << padded_size
          << "), " << direction << ")";
  return 0;
}

int UPMEMDeviceAPI::BindXfer(int bank_index, uint64_t host_addr, uint64_t size) {
  if (xfer_handle == nullptr)
    LOG(FATAL) << "No xfer handle is set. InitXfer should be invoked first.";
  if (size > xfer_bulk_size) {
    LOG(FATAL) << "Bulk size " << size << " exceeds the maximum size " << xfer_bulk_size;
  }
  if (xfer_direction == DPU_XFER_FROM_DPU && size < xfer_bulk_size) {
    void* new_ptr;
    posix_memalign(&new_ptr, 64, xfer_bulk_size * GetBytes(xfer_handle));
    d2h_temp[bank_index] = {new_ptr, HostOffset(xfer_handle, host_addr), size};
    UPMEM_CALL(dpu_prepare_xfer(dpu_entry[bank_index], new_ptr));
    VLOG(3) << "dpu_prepare_xfer(" << bank_index << ", " << xfer_handle << " + (" << host_addr
            << ") * " << GetBytes(xfer_handle) << ") // tmp(" << size << ")";
  } else {
    UPMEM_CALL(dpu_prepare_xfer(dpu_entry[bank_index], HostOffset(xfer_handle, host_addr)));
    VLOG(3) << "dpu_prepare_xfer(" << bank_index << ", " << xfer_handle << " + (" << host_addr
            << ") * " << GetBytes(xfer_handle) << ") // normal(" << size << ")";
  }
  return 0;
}

void* UPMEMDeviceAPI::BindBounded(void* bind_buffer_ptr, int bank_index, uint64_t host_addr,
                                  uint64_t size) {
  void* new_ptr;
  posix_memalign(&new_ptr, 64, xfer_bulk_size * GetBytes(xfer_handle));
  d2h_temp[bank_index] = {new_ptr, HostOffset(xfer_handle, host_addr), size};
  // std::cerr << "BindBounded " << new_ptr << " " << xfer_bulk_size << std::endl;
  *reinterpret_cast<void**>(static_cast<uint64_t*>(bind_buffer_ptr) + bank_index) = new_ptr;
  return new_ptr;
}

int UPMEMDeviceAPI::BindAll(void* bind_buffer) {
  for (int i = 0; i < dpu_entry.size(); i++) {
    void* handle = (*((uint64_t**)bind_buffer + i));
    UPMEM_CALL(dpu_prepare_xfer(dpu_entry[i], handle));
  }
  VLOG(3) << "for (int i = 0; i < " << dpu_entry.size() << "; i++) {\n"
          << "  dpu_prepare_xfer(i, ...);\n}";
  return 0;
}

int UPMEMDeviceAPI::PushXfer() {
  auto api = UPMEMDeviceAPI::Global();
  if (xfer_direction == DPU_XFER_FROM_DPU) {
    api->last_d2h_start = std::chrono::high_resolution_clock::now();
    auto t = api->last_d2h_start - kernel_end;
    before_d2h_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
    UPMEM_CALL(dpu_push_xfer(dpu_set, xfer_direction, GetSymbolName(xfer_handle).c_str(),
                             static_cast<uint32_t>(xfer_offset) * GetBytes(xfer_handle),
                             xfer_bulk_size * GetBytes(xfer_handle), DPU_XFER_DEFAULT));
    api->last_d2h_end = std::chrono::high_resolution_clock::now();
    auto nt = api->last_d2h_end - api->last_d2h_start;
    api->d2h_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
  } else {
    UPMEM_CALL(dpu_push_xfer(dpu_set, xfer_direction, GetSymbolName(xfer_handle).c_str(),
                             static_cast<uint32_t>(xfer_offset) * GetBytes(xfer_handle),
                             xfer_bulk_size * GetBytes(xfer_handle), DPU_XFER_DEFAULT));
  }

  VLOG(3) << "dpu_push_xfer(" << xfer_direction << ", " << GetSymbolName(xfer_handle) << ", "
          << xfer_offset << ", " << xfer_bulk_size << " * " << GetBytes(xfer_handle) << ")";

  if (xfer_direction == DPU_XFER_FROM_DPU) {
    for (auto& kv : d2h_temp) {
      if (kv.second.size > 0)
        memcpy(kv.second.dest_ptr, kv.second.temp_ptr, kv.second.size * GetBytes(xfer_handle));
    }
  }

  for (auto& kv : d2h_temp) {
    free(kv.second.temp_ptr);
  }

  d2h_temp.clear();
  xfer_handle = nullptr;
  xfer_offset = 0;
  xfer_bulk_size = 0;
  return 0;
}

TVM_REGISTER_GLOBAL("device_api.upmem").set_body([](TVMArgs args, TVMRetValue* rv) {
  UPMEMDeviceAPI* api = UPMEMDeviceAPI::Global();
  *rv = static_cast<void*>(api);
});

TVM_REGISTER_GLOBAL("device_api.upmem.timestamp").set_body([](TVMArgs args, TVMRetValue* rv) {
  auto api = UPMEMDeviceAPI::Global();
  std::string type = args[0];
  if (type == "start") {
    api->entire_start = std::chrono::high_resolution_clock::now();
  } else if (type == "end") {
    api->entire_end = std::chrono::high_resolution_clock::now();
  }
});

TVM_REGISTER_GLOBAL("device_api.upmem.elapsed_time").set_body([](TVMArgs args, TVMRetValue* rv) {
  auto api = UPMEMDeviceAPI::Global();
  std::string type = args[0];
  std::chrono::high_resolution_clock::time_point start, end;
  if (type == "entire") {
    end = api->entire_end;
    start = api->entire_start;
  } else if (type == "kernel") {
    end = api->kernel_end;
    start = api->kernel_start;
  } else if (type == "after_kernel") {
    end = api->entire_end;
    start = api->kernel_end;
  } else if (type == "before_kernel") {
    end = api->kernel_start;
    start = api->entire_start;
  } else if (type == "d2h") {
    *rv = static_cast<double>(api->d2h_time);
    return;
  } else if (type == "after_d2h") {
    end = api->entire_end;
    start = api->last_d2h_end;
  } else {
    LOG(FATAL) << "Invalid type: " << type;
  }
  *rv = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
});

TVM_REGISTER_GLOBAL("device_api.upmem.d2h_time").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = static_cast<double>(UPMEMDeviceAPI::Global()->d2h_time);
});

TVM_REGISTER_GLOBAL("device_api.upmem.acquire_resources")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      auto api = UPMEMDeviceAPI::Global();
      api->entire_start = std::chrono::high_resolution_clock::now();
      if (api->dpu_entry.empty()) {
        UPMEMDeviceAPI::Global()->AcquireResources(args);
      }
    });

TVM_REGISTER_GLOBAL("device_api.upmem.release_resources")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = static_cast<int>(UPMEMDeviceAPI::Global()->ReleaseResources());
    });

TVM_REGISTER_GLOBAL("device_api.upmem.pim_allocate_memory")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
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

TVM_REGISTER_GLOBAL("device_api.upmem.pim_transfer_host_to_device")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      void* handle = args[0];
      uint64_t host_address = args[1];
      uint64_t in_bank_addr = args[2];
      int bank_index = args[3];
      int size = args[4];
      *rv = static_cast<int>(UPMEMDeviceAPI::Global()->TransferHostToDevice(
          handle, host_address, in_bank_addr, bank_index, size));
    });

TVM_REGISTER_GLOBAL("device_api.upmem.pim_transfer_device_to_host")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      void* handle = args[0];
      uint64_t host_address = args[1];
      uint64_t in_bank_addr = args[2];
      int bank_index = args[3];
      int size = args[4];
      *rv = static_cast<int>(UPMEMDeviceAPI::Global()->TransferDeviceToHost(
          handle, host_address, in_bank_addr, bank_index, size));
    });

TVM_REGISTER_GLOBAL("device_api.upmem.pim_broadcast").set_body([](TVMArgs args, TVMRetValue* rv) {
  void* handle = args[0];
  uint64_t host_address = args[1];
  int size = args[2];
  *rv = static_cast<int>(UPMEMDeviceAPI::Global()->Broadcast(handle, host_address, size));
});

TVM_REGISTER_GLOBAL("device_api.upmem.dpu_parallel_transfer_init")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      void* handle = args[0];
      uint64_t in_bank_addr = args[1];
      uint64_t size = args[2];
      int direction = args[3];
      *rv = static_cast<int>(
          UPMEMDeviceAPI::Global()->InitXfer(handle, in_bank_addr, size, direction));
    });

TVM_REGISTER_GLOBAL("device_api.upmem.dpu_parallel_transfer_bind")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      int bank_index = args[0];
      uint64_t host_address = args[1];
      uint64_t size = args[2];
      *rv = static_cast<int>(UPMEMDeviceAPI::Global()->BindXfer(bank_index, host_address, size));
    });

TVM_REGISTER_GLOBAL("device_api.upmem.dpu_parallel_transfer_bind_bounded")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      void* bind_buffer_ptr = args[0];
      int bank_index = args[1];
      uint64_t host_address = args[2];
      uint64_t size = args[3];
      *rv = static_cast<void*>(
          UPMEMDeviceAPI::Global()->BindBounded(bind_buffer_ptr, bank_index, host_address, size));
    });

TVM_REGISTER_GLOBAL("device_api.upmem.dpu_parallel_transfer_bind_all")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      void* bind_buffer = args[0];
      *rv = static_cast<int>(UPMEMDeviceAPI::Global()->BindAll(bind_buffer));
    });

TVM_REGISTER_GLOBAL("device_api.upmem.dpu_parallel_transfer_commit")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = static_cast<int>(UPMEMDeviceAPI::Global()->PushXfer());
    });

class UPMEMTimerNode : public TimerNode {
 public:
  virtual void Start() { start_ = std::chrono::high_resolution_clock::now(); }
  virtual void Stop() { duration_ = std::chrono::high_resolution_clock::now() - start_; }
  virtual int64_t SyncAndGetElapsedNanos() { return duration_.count(); }
  virtual ~UPMEMTimerNode() {}

  static constexpr const char* _type_key = "UPMEMTimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(UPMEMTimerNode, TimerNode);

 private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::duration<int64_t, std::nano> duration_;
};

TVM_REGISTER_OBJECT_TYPE(UPMEMTimerNode);

TVM_REGISTER_GLOBAL("profiling.timer.upmem").set_body_typed([](Device dev) {
  return Timer(make_object<UPMEMTimerNode>());
});

}  // namespace runtime
}  // namespace tvm
