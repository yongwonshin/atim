/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system or translated into any human or computer
 * language in any form by any means, electronic, mechanical, manual or otherwise, or disclosed to
 * third parties without the express written permission of Samsung Electronics. (Use of the Software
 * is restricted to non-commercial, personal or academic, research purpose only)
 */

#include "block_allocator.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <list>
#include <map>

#include "../../opencl_common.h"
#include "../hbmpim_common.h"

namespace tvm {
namespace runtime {
namespace pim_library {
void* BlockAllocator::alloc(cl_context context, size_t request_size, size_t& allocated_size,
                            Device dev) {
  VLOG(2) << "[START] " << __FUNCTION__ << " called";
  assert(request_size <= block_size() && "BlockAllocator alloc request exceeds block size.");
  void* ret = nullptr;
  size_t bsize = block_size();

  ret = allocate_pim_block(context, bsize, dev);

  if (ret == 0) return nullptr;

  allocated_size = block_size();

  VLOG(2) << "[END] " << __FUNCTION__ << " called";
  return ret;
}

void BlockAllocator::free(void* ptr, size_t length) {
  OPENCL_CALL(clEnqueueUnmapMemObject(m_[ptr], base_address_memobject_, base_host_address_, 0,
                                      nullptr, nullptr));
  OPENCL_CALL(clReleaseMemObject(base_address_memobject_));
}

void* BlockAllocator::allocate_pim_block(cl_context context, size_t bsize, Device dev) {
  auto queue = cl::HBMPIMWorkspace::Global()->GetQueue(dev);
  void* ret = nullptr;
  cl_int err_code;
  VLOG(2) << "Device ID : " << dev.device_id;
  if (pim_alloc_done[dev.device_id] == true) {
    std::cerr << "alloc done!" << std::endl;
    return 0;
  }

  base_address_memobject_ = clCreateBuffer(context, CL_MEM_READ_WRITE, bsize, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  base_host_address_ =
      clEnqueueMapBuffer(queue, base_address_memobject_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                         bsize, 0, nullptr, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  ret = base_host_address_;

  if (ret != nullptr) {
    pim_alloc_done[dev.device_id] = true;
    g_pim_base_addr[dev.device_id] = reinterpret_cast<uint64_t>(ret);
  } else {
    LOG(FATAL) << "fmm_map_pim failed! " << ret;
  }

  m_[ret] = queue;

  return ret;
}
}  // namespace pim_library
}  // namespace runtime
}  // namespace tvm
