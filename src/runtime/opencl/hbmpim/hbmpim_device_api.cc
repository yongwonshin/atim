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
 * \file hbmpim_device_api.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include "hbmpim_common.h"

namespace tvm {
namespace runtime {
namespace cl {

OpenCLThreadEntry* HBMPIMWorkspace::GetThreadEntry() { return HBMPIMThreadEntry::ThreadLocal(); }

OpenCLWorkspace* HBMPIMWorkspace::Global() {
  static OpenCLWorkspace* inst = new HBMPIMWorkspace();
  return inst;
}

void HBMPIMWorkspace::Init() { OpenCLWorkspace::Init("hbmpim", "gpu"); }

bool HBMPIMWorkspace::IsOpenCLDevice(Device dev) {
  return dev.device_type == static_cast<DLDeviceType>(kDLHBMPIM);
}

typedef dmlc::ThreadLocalStore<HBMPIMThreadEntry> HBMPIMThreadStore;

HBMPIMThreadEntry* HBMPIMThreadEntry::ThreadLocal() { return HBMPIMThreadStore::Get(); }

TVM_REGISTER_GLOBAL("device_api.hbmpim").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = HBMPIMWorkspace::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace cl
}  // namespace runtime
}  // namespace tvm
