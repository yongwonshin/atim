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