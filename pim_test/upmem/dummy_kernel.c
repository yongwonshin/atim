// Function: main_kernel
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <handshake.h>
#include <mram.h>
#include <seqread.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  int32_t x, y, z;
} BlockInfo;
BARRIER_INIT(barrier, NR_TASKLETS);

__host BlockInfo blockIdx;

inline int min(int x, int y) { return x < y ? x : y; }
inline int max(int x, int y) { return x > y ? x : y; }
__mram_noinit int32_t A[393216];
__mram int32_t B[4096];
__mram int32_t C[96];
int main() {
  const int blockIdx_x = blockIdx.x;
  const int blockIdx_y = blockIdx.y;
  const int blockIdx_z = blockIdx.z;

  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  int32_t* C_local = (int32_t*)mem_alloc(2 * sizeof(int32_t));
  int32_t* A_local = (int32_t*)mem_alloc(256 * sizeof(int32_t));
  int32_t* B_local = (int32_t*)mem_alloc(256 * sizeof(int32_t));
  for (int32_t i_1_1 = 0; i_1_1 < 3; ++i_1_1) {
    // if (tasklet_id * 6 + i_1_1 * 2 < 80) {
    for (int32_t i_1_2 = 0; i_1_2 < 2; ++i_1_2) {
      if ((((tasklet_id * 6) + (i_1_1 * 2)) + i_1_2) < 80) {
        C_local[i_1_2] = 0;
      }
      for (int32_t k_0 = 0; k_0 < 16; ++k_0) {
        mram_read(
            (__mram_ptr void*)(A + ((((tasklet_id * 24576) + (i_1_1 * 8192)) + (i_1_2 * 4096)) +
                                    (k_0 * 256))),
            A_local + 0, 1024);
        mram_read((__mram_ptr void*)(B + (k_0 * 256)), B_local + 0, 1024);
        for (int32_t k_1 = 0; k_1 < 256; ++k_1) {
          C_local[i_1_2] = (C_local[i_1_2] + (A_local[k_1] * B_local[k_1]));
        }
      }
    }
    // }
    mram_write(C_local + 0, (__mram_ptr void*)(C + ((tasklet_id * 6) + (i_1_1 * 2))), 8);
  }
}
