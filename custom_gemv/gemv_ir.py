# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((2048, 2048), "float32"), B: T.Buffer((2048,), "float32"), C: T.Buffer((2048,), "float32")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "global_symbol": "main", "tir.noalias": T.bool(True)})
        threadIdx_x = T.launch_thread("threadIdx.x", 4)
        C_local = T.allocate([2], "float32", "local")
        A_local = T.allocate([64], "float32", "local")
        B_local = T.allocate([64], "float32", "local")
        for y_inner_outer in range(256):
            C_local_1 = T.Buffer((2,), data=C_local, scope="local", align=8)
            for y_c in range(2):
                C_local_1[y_c] = T.float32(0)
                for k_outer in range(32):
                    A_local_1 = T.Buffer((64,), data=A_local, scope="local")
                    for ax1 in range(64):
                        A_1 = T.Buffer((4194304,), data=A.data)
                        A_local_1[ax1] = A_1[threadIdx_x * 1048576 + y_inner_outer * 4096 + y_c * 2048 + k_outer * 64 + ax1]
                    B_local_1 = T.Buffer((64,), data=B_local, scope="local")
                    for ax0 in range(64):
                        B_local_1[ax0] = B[k_outer * 64 + ax0]
                    for k_inner in range(64):
                        C_local_1[y_c] = C_local_1[y_c] + A_local_1[k_inner] * B_local_1[k_inner]
            for y_inner_inner in range(2):
                C[threadIdx_x * 512 + y_inner_outer * 2 + y_inner_inner] = C_local_1[y_inner_inner]

# IR
@I.ir_module
class Module:
    I.module_attrs({"runtime": None})
    @T.prim_func
    def mmult_kernel(A: T.handle("float32"), B: T.handle("float32"), C: T.handle("float32")):
        T.func_attr({"calling_conv": 2, "global_symbol": "mmult_kernel", "target": T.target({"host": {"keys": ["cpu"], "kind": "llvm", "tag": ""}, "keys": ["upmem", "pim"], "kind": "upmem", "tag": ""}), "tir.is_global_func": T.bool(True), "tir.kernel_launch_params": ["threadIdx.x"], "tir.noalias": T.bool(True)})
        threadIdx_x = T.launch_thread("threadIdx.x", 4)
        C_local = T.allocate([2], "float32", "local")
        A_local = T.allocate([64], "float32", "local")
        B_local = T.allocate([64], "float32", "local")
        for y_inner_outer in range(256):
            C_local_1 = T.Buffer((2,), data=C_local, scope="local", align=8)
            for y_c in range(2):
                C_local_1[y_c] = T.float32(0)
                for k_outer in range(32):
                    A_local_1 = T.Buffer((64,), data=A_local, scope="local")
                    for ax1 in range(64):
                        A_1 = T.Buffer((4194304,), data=A)
                        A_local_1[ax1] = A_1[threadIdx_x * 1048576 + y_inner_outer * 4096 + y_c * 2048 + k_outer * 64 + ax1]
                    B_local_1 = T.Buffer((64,), data=B_local, scope="local")
                    for ax0 in range(64):
                        B_1 = T.Buffer((2048,), data=B)
                        B_local_1[ax0] = B_1[k_outer * 64 + ax0]
                    for k_inner in range(64):
                        C_local_1[y_c] = C_local_1[y_c] + A_local_1[k_inner] * B_local_1[k_inner]
            for y_inner_inner in range(2):
                C_1 = T.Buffer((2048,), data=C)
                C_1[threadIdx_x * 512 + y_inner_outer * 2 + y_inner_inner] = C_local_1[y_inner_inner]

#ㅜㅊ
__kernel void mmult_kernel(float* A, float* B, float* C) {
  float C_local[2];
  float A_local[64];
  float B_local[64];
  for (int y_inner_outer = 0; y_inner_outer < 256; ++y_inner_outer) {
    for (int y_c = 0; y_c < 2; ++y_c) {
      C_local[y_c] = 0.000000e+00f;
      for (int k_outer = 0; k_outer < 32; ++k_outer) {
        for (int ax1 = 0; ax1 < 64; ++ax1) {
          A_local[ax1] = A[((((((convert_int(get_local_id(0))) * 1048576) + (y_inner_outer * 4096)) + (y_c * 2048)) + (k_outer * 64)) + ax1)];
        }
        for (int ax0 = 0; ax0 < 64; ++ax0) {
          B_local[ax0] = B[((k_outer * 64) + ax0)];
        }
        for (int k_inner = 0; k_inner < 64; ++k_inner) {
          C_local[y_c] = (C_local[y_c] + (A_local[k_inner] * B_local[k_inner]));
        }
      }
    }
    for (int y_inner_inner = 0; y_inner_inner < 2; ++y_inner_inner) {
      C[((((convert_int(get_local_id(0))) * 512) + (y_inner_outer * 2)) + y_inner_inner)] = C_local[y_inner_inner];
    }
  }
}

# UPMEM

int main() {
  float* C_local = (float*) mem_alloc(2 * sizeof(float)); // 동적
  float* A_local = (float*) mem_alloc(64 * sizeof(float));
  float* B_local = (float*) mem_alloc(64 * sizeof(float));
  for (int y_inner_outer = 0; y_inner_outer < 256; ++y_inner_outer) { // 그대로
    for (int y_c = 0; y_c < 2; ++y_c) {
      C_local[y_c] = 0;
      for (k_outer = 0; k_outer < 32; ++k_outer) {
        mram_read((__mram_ptr void const*)(A + (me() * 1048576 + y_inner_outer * 4096 + y_c * 2048 + 64 * k_outer) * sizeof(float)), A_local, 64 * sizeof(float));
        mram_read((__mram_ptr void const*)(B + (64 * k_outer) * sizeof(float)), B_local, 64 * sizeof(float));
        for (int k_inner = 0; k_inner < 64; ++k_inner) {
          C_local[y_c] = (C_local[y_c] + (A_local[k_inner] * B_local[k_inner]));
        }
      }
      mram_temp_addr = mram_base_addr_A + 32 * 
    }
    mram_write(C_local, (__mram_ptr void*)(C + y_inner_outer * 2 * sizeof(float)), sizeof(float) * 2);   
  }                                         
}


__kernel void mmult_kernel(float* A, float* B, float* C) {
  float C_local[2];
  float A_local[64];
  float B_local[64];
  for (int y_inner_inner_outer = 0; y_inner_inner_outer < 16; ++y_inner_inner_outer) {
    for (int y_c = 0; y_c < 2; ++y_c) {
      C_local[y_c] = 0.000000e+00f;
      for (int k_outer = 0; k_outer < 32; ++k_outer) {
        for (int ax1 = 0; ax1 < 64; ++ax1) {
          A_local[ax1] = A[(((((((convert_int(get_group_id(0))) * 1048576) + ((convert_int(get_local_id(0))) * 65536)) + (y_inner_inner_outer * 4096)) + (y_c * 2048)) + (k_outer * 64)) + ax1)];
        }
        for (int ax0 = 0; ax0 < 64; ++ax0) {
          B_local[ax0] = B[((k_outer * 64) + ax0)];
        }
        for (int k_inner = 0; k_inner < 64; ++k_inner) {
          C_local[y_c] = (C_local[y_c] + (A_local[k_inner] * B_local[k_inner]));
        }
      }
    }
    for (int y_inner_inner_inner = 0; y_inner_inner_inner < 2; ++y_inner_inner_inner) {
      C[(((((convert_int(get_group_id(0))) * 512) + ((convert_int(get_local_id(0))) * 32)) + (y_inner_inner_outer * 2)) + y_inner_inner_inner)] = C_local[y_inner_inner_inner];
    }
  }
}

#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>

BARRIER_INIT(my_barrier, NR_TASKLETS);
__host uint32_t A, B, C;

int main() {
  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&my_barrier);

  float* C_local = (float*) mem_alloc(2 * sizeof(float)); // 동적
  float* A_local = (float*) mem_alloc(64 * sizeof(float));
  float* B_local = (float*) mem_alloc(64 * sizeof(float));
  for (int y_inner_outer = 0; y_inner_outer < 16; ++y_inner_outer) { // 그대로
    for (int y_c = 0; y_c < 2; ++y_c) {
      C_local[y_c] = 0;
      for (int k_outer = 0; k_outer < 32; ++k_outer) {
        mram_read((__mram_ptr void const*)(A + (tasklet_id * 65536 + y_inner_outer * 4096 + y_c * 2048 + k_outer * 64) * sizeof(float)), A_local, 64 * sizeof(float));
        mram_read((__mram_ptr void const*)(B + (64 * k_outer) * sizeof(float)), B_local, 64 * sizeof(float));
        for (int k_inner = 0; k_inner < 64; ++k_inner) {
          C_local[y_c] = (C_local[y_c] + (A_local[k_inner] * B_local[k_inner]));
        }
      }
    }
    mram_write(C_local, (__mram_ptr void*)(C + (tasklet_id * 32 + y_inner_outer * 2) * sizeof(float)), sizeof(float) * 2);   
  }                                         
}

#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>

BARRIER_INIT(barrier, NR_TASKLETS);
__host void*A;
__host void*B;
__host void*C;

int main() {
  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  float* C_local = (float*) mem_alloc(2* sizeof(float));
  float* A_local = (float*) mem_alloc(64* sizeof(float));
  float* B_local = (float*) mem_alloc(64* sizeof(float));
  for (int32_t y_inner_inner_outer = 0; y_inner_inner_outer < 16; ++y_inner_inner_outer) {
    for (int32_t y_c = 0; y_c < 2; ++y_c) {
      C_local[y_c] = 0.000000e+00f;
      for (int32_t k_outer = 0; k_outer < 32; ++k_outer) {
        for (int32_t ax1 = 0; ax1 < 64; ++ax1) {
          A_local[ax1] = ((float*)A)[((((((0 * 1048576) + (tasklet_id * 65536)) + (y_inner_inner_outer * 4096)) + (y_c * 2048)) + (k_outer * 64)) + ax1)];
        }
        for (int32_t ax0 = 0; ax0 < 64; ++ax0) {
          B_local[ax0] = ((float*)B)[((k_outer * 64) + ax0)];
        }
        for (int32_t k_inner = 0; k_inner < 64; ++k_inner) {
          C_local[y_c] = (C_local[y_c] + (A_local[k_inner] * B_local[k_inner]));
        }
      }
    }
    for (int32_t y_inner_inner_inner = 0; y_inner_inner_inner < 2; ++y_inner_inner_inner) {
      ((float*)C)[((((0 * 512) + (tasklet_id * 32)) + (y_inner_inner_outer * 2)) + y_inner_inner_inner)] = C_local[y_inner_inner_inner];
    }
  }
}
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>

BARRIER_INIT(barrier, NR_TASKLETS);
__host void*A;
__host void*B;
__host void*C;

int main() {
  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  int32_t* C_local = (int32_t *) mem_alloc(2 * sizeof(int32_t));
  int32_t* A_local = (int32_t *) mem_alloc(64 * sizeof(int32_t));
  int32_t* B_local = (int32_t *) mem_alloc(64 * sizeof(int32_t));
  for (int32_t y_inner_inner_outer = 0; y_inner_inner_outer < 16; ++y_inner_inner_outer) {
    for (int32_t y_c = 0; y_c < 2; ++y_c) {
      C_local[y_c] = 0;
      for (int32_t k_outer = 0; k_outer < 32; ++k_outer) {
        mram_read((__mram_ptr void const*)(A + (((((tasklet_id * 65536) + (y_inner_inner_outer * 4096)) + (y_c * 2048)) + (k_outer * 64))) * sizeof(int32_t)), A_local, 64 * sizeof(int32_t));
        mram_read((__mram_ptr void const*)(B + ((k_outer * 64)) * sizeof(int32_t)), B_local, 64 * sizeof(int32_t));
        for (int32_t k_inner = 0; k_inner < 64; ++k_inner) {
          C_local[y_c] = (C_local[y_c] + (A_local[k_inner] * B_local[k_inner]));
        }
      }
    }
    mram_write(C_local, (__mram_ptr void*)(C + (((tasklet_id * 32) + (y_inner_inner_outer * 2))) * sizeof(int32_t)), 2 * sizeof(int32_t));
  }
}
