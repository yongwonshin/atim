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
#include <dpu.h>
#include <dpu_log.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <cstring>

#include "upmem_common.h"

namespace tvm {
namespace runtime {

class UPMEMDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final {
    // device = dev;
  }

  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final {
    int value = 0;
    switch(kind) {
      case kExist: {
        break;
      }
      case kMaxThreadsPerBlock: {
        // hw.dpu.nr_of_threads
        value = 24;
        break;
      }
      case kWarpSize: {
        value = 1;
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        // hw.memories.wram_size
        value = 65536;
        break;
      }
      case kComputeVersion: {
        // dpu_get_description()->hw
        break;
      }
      case kDeviceName: {
        // hw.signature.chip_id
        break;
      }
      case kMaxClockRate: {
        // hw.timings.fck_frequency_in_mhz * 1000000
        break;
      }
      case kMultiProcessorCount: {
        // hw.topology.nr_of_control_interfaces
        break;
      }
      case kMaxThreadDimensions: {
        // 1
        break;
      }
      case kMaxRegistersPerBlock: {
        break;
      }
      case kGcnArch: {
        break;
      }
      case kApiVersion: {
        break;
      }
      case kDriverVersion: {
        break;
      }
    }
    *rv = value;
  }

  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    // malloc is not needed...:(
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    // free dataspace 
  }

  void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) final {
    size_t nbytes = GetDataSize(*from);
    ICHECK_EQ(nbytes, GetDataSize(*to));
    ICHECK(IsContiguous(*from) && IsContiguous(*to))
      << "CopyDataFromTo only support contiguous array for now";

    auto from_device_type = from->device.device_type;
    auto to_device_type = to->device.device_type;

    if (from_device_type == kDLUPMEM && to_device_type == kDLCPU) {
      // copy_from
    } else if (from_device_type == kDLCPU && to_device_type == kDLUPMEM) {
      // copy_to
    } else if (from_device_type == kDLUPMEM && to_device_type == kDLUPMEM) {
      LOG(FATAL) << "Transfer between UPMEM is not supported"
    } else {
      LOG(FATAL) << "expect copy from/to UPMEM or between UPMEM";
    }
  }

 public:
  static UPMEMDeviceAPI* Global() {
    static auto* inst = new UPMEMDeviceAPI();
    return inst;
  }
};

typedef dmlc::ThreadLocalStore<UPMEMThreadEntry> UPMEMThreadStore;

UPMEMThreadEntry::UPMEMThreadEntry() : pool(kDLUPMEM, UPMEMDeviceAPI::Global()) {}

UPMEMThreadEntry* UPMEMThreadEntry::ThreadLocal() { return UPMEMThreadStore::Get(); }

TVM_REGISTER_GLOBAL("device_api.upmem").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = UPMEMDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

TVM_REGISTER_GLOBAL("device_api.upmem_host").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = UPMEMDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

class UPMEMTimerNode : public TimerNode {
 public:
  virtual void Start() {
  }
  virtual void Stop() {
  }
  
  virtual int64_t SyncAndGetElapsedNanos() {
  }

  virtual ~UPMEMTimerNode() {
  }

  UPMEMTimerNode() {
  }

  static constexpr const char* _type_key = "UPMEMTimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(UPMEMTimerNode, TimerNode);

 private:
  upmemEvent_t start_;
  upmemEvent_t stop_;
};

TVM_REGISTER_OBJECT_TYPE(UPMEMTimerNode);

TVM_REGISTER_GLOBAL("profiling.timer.upmem").set_body_typed([](Device dev) {
  return Timer(make_object<UPMEMTimerNode>());
});

}  // namespace runtime
}  // namespace tvm
