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

#ifndef TVM_RUNTIME_OPENCL_HBMPIM_PIM_LIBRARY_BLOCK_ALLOCATOR_H_
#define TVM_RUNTIME_OPENCL_HBMPIM_PIM_LIBRARY_BLOCK_ALLOCATOR_H_

#include <CL/cl.h>

#include <map>

#include "../../opencl_common.h"

namespace tvm {
namespace runtime {
namespace pim_library {

#define MAX_NUM_GPUS 10

class BlockAllocator {
  /**
   * @brief Block allocator of size 2MB
   *
   * TODO: This is a simple block allocator where it uses malloc for allocation and free
   *       It has to be modified to use PIM memory region for alloc and free.
   */

 public:
  explicit BlockAllocator(void) {
    for (int i = 0; i < MAX_NUM_GPUS; ++i) {
      pim_alloc_done[i] = false;
      g_pim_base_addr[i] = reinterpret_cast<uint64_t>(nullptr);
    }
  }
  void* alloc(cl_context context, size_t request_size, size_t& allocated_size, Device dev);
  void free(void* ptr, size_t length);
  void* allocate_pim_block(cl_context context, size_t request_size, Device dev);
  size_t block_size(void) const { return block_size_; }
  void* get_pim_base() { return (void*)base_address_memobject_; };

  bool pim_alloc_done[MAX_NUM_GPUS] = {false};
  uint64_t g_pim_base_addr[MAX_NUM_GPUS];

 private:
  cl_mem base_address_memobject_;
  void* base_host_address_;
  // static const size_t block_size_ = 134217728;  // 128M Pim area

  // target mode space.
  static const size_t block_size_ = 1073741824;  // 1GB PIM area
  std::unordered_map<void*, cl_command_queue> m_;
};

}  // namespace pim_library
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_OPENCL_HBMPIM_PIM_LIBRARY_BLOCK_ALLOCATOR_H_
