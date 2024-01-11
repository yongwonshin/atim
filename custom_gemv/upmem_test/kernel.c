#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>

BARRIER_INIT(barrier, NR_TASKLETS);
__mram_noinit float  A[262144];
__mram_noinit float  B[512];
__mram_noinit float  C_rf[512];
int main() {
  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  float* C_rf_local = (float*) mem_alloc(2 * sizeof(float));
  float* A_local = (float*) mem_alloc(64 * sizeof(float));
  float* B_local = (float*) mem_alloc(64 * sizeof(float));
  for (int32_t i_2 = 0; i_2 < 16; ++i_2) {
    for (int32_t i_3 = 0; i_3 < 2; ++i_3) {
      C_rf_local[i_3] = 0.000000e+00f;
      for (int32_t k_1_0 = 0; k_1_0 < 8; ++k_1_0) {
        mram_read((__mram_ptr void const*)(A + (((((tasklet_id * 16384) + (i_2 * 1024)) + (i_3 * 512)) + (k_1_0 * 64))) * sizeof(float)), A_local, 64 * sizeof(float));
        mram_read((__mram_ptr void const*)(B + ((k_1_0 * 64)) * sizeof(float)), B_local, 64 * sizeof(float));
        for (int32_t k_1_1 = 0; k_1_1 < 64; ++k_1_1) {
          C_rf_local[i_3] = (C_rf_local[i_3] + (A_local[k_1_1] * B_local[k_1_1]));
        }
      }
    }
    mram_write(C_rf_local, (__mram_ptr void*)(C_rf + (((tasklet_id * 32) + (i_2 * 2))) * sizeof(float)), 2 * sizeof(float));
  }
}
