# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_UPMEM)
  message(STATUS "Build with UPMEM support")
  find_path(UPMEM_INCLUDE_DIR dpu.h PATH_SUFFIXES dpu)
  find_library(UPMEM_LIB dpu)
  tvm_file_glob(GLOB RUNTIME_UPMEM_SRCS src/runtime/upmem/*.cc)
  include_directories(SYSTEM ${UPMEM_INCLUDE_DIR})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${UPMEM_LIB})
  list(APPEND RUNTIME_SRCS ${RUNTIME_UPMEM_SRCS})
else(USE_UPMEM)
  list(APPEND COMPILER_SRCS src/target/opt/build_upmem_off.cc)
endif(USE_UPMEM)
