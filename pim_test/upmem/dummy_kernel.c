// Function: main_kernel
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>
#include <handshake.h>

typedef struct { int32_t x, y, z; } BlockInfo;
BARRIER_INIT(barrier, NR_TASKLETS);

__host BlockInfo blockIdx;

inline int min(int x, int y) { return x < y ? x : y; }
inline int max(int x, int y) { return x > y ? x : y; }
__mram_noinit int32_t A[169336];
__mram int32_t B[4912];
__mram int32_t C_rf_global[32];
int main() {
  const int blockIdx_x = blockIdx.x;
  const int blockIdx_y = blockIdx.y;
  const int blockIdx_z = blockIdx.z;

  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  int32_t* C_rf_global_local = (int32_t*) mem_alloc(2 * sizeof(int32_t));
  int32_t* A_local = (int32_t*) mem_alloc(376 * sizeof(int32_t));
  int32_t* B_local = (int32_t*) mem_alloc(376 * sizeof(int32_t));
  for (int32_t i_1_2 = 0; i_1_2 < 2; ++i_1_2) {
    C_rf_global_local[i_1_2] = 0;
    for (int32_t k_1_0 = 0; k_1_0 < 13; ++k_1_0) {
      mram_read((__mram_ptr void*)(A + (((tasklet_id * 10608) + (i_1_2 * 5304)) + (k_1_0 * 408))), A_local + 0, 64);
      mram_read((__mram_ptr void*)(B + (k_1_0 * 408)), B_local + 0, 64);
      for (int32_t k_1_1 = 0; k_1_1 < max(0, min(16, (200 - (k_1_0 * 16)))); ++k_1_1) {
        C_rf_global_local[i_1_2] = (C_rf_global_local[i_1_2] + (A_local[k_1_1] * B_local[k_1_1]));
      }
    }
  }
  mram_write(C_rf_global_local + 0, (__mram_ptr void*)(C_rf_global + (tasklet_id * 2)), 8);
}


