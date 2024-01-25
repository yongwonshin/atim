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

#ifndef TVM_RUNTIME_OPENCL_HBMPIM_PIM_TRACE_COALESCER_H_
#define TVM_RUNTIME_OPENCL_HBMPIM_PIM_TRACE_COALESCER_H_

#include "pim_data_types.h"
#include "pim_info.h"

namespace tvm {
namespace runtime {
namespace pim_library {
// TODO: Define Stride for each PIM configuration
#define TRANS_SIZE 0x10
#define MASK (~((TRANS_SIZE << 1) - 1))

class TraceParser {
 public:
  void coalesce_trace(PimMemTraceData* fmtd32, int* fmtd32_size, PimMemTraceData* fmtd16,
                      int fmtd16_size);

 private:
  void append_data(uint8_t* dst, uint8_t* src, int size);
  void move_data(uint8_t* dst, uint8_t* src, int size);
};

}  // namespace pim_library
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_OPENCL_HBMPIM_PIM_TRACE_COALESCER_H_
