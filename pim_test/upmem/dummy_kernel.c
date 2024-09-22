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
__mram_noinit int32_t A[16384];
__mram int32_t B[256];
__mram int32_t C[64];
int main() {
  const int blockIdx_x = blockIdx.x;
  const int blockIdx_y = blockIdx.y;
  const int blockIdx_z = blockIdx.z;

  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  int32_t* C_local = (int32_t*) mem_alloc(2 * sizeof(int32_t));
  int32_t* A_local = (int32_t*) mem_alloc(64 * sizeof(int32_t));
  int32_t* B_local = (int32_t*) mem_alloc(64 * sizeof(int32_t));
  for (int32_t i_1_1 = 0; i_1_1 < 4; ++i_1_1) {
    for (int32_t i_1_2 = 0; i_1_2 < 2; ++i_1_2) {
      C_local[i_1_2] = 0;
      for (int32_t k_0 = 0; k_0 < 4; ++k_0) {
        mram_read((__mram_ptr void*)(A + ((((tasklet_id * 2048) + (i_1_1 * 512)) + (i_1_2 * 256)) + (k_0 * 64))), A_local + 0, 256);
        mram_read((__mram_ptr void*)(B + (k_0 * 64)), B_local + 0, 256);
        //int32_t k_1_ext = ;
        for (int32_t k_1 = 0; k_1 < max(0, min(64, (200 - (k_0 * 64)))); ++k_1) {
          C_local[i_1_2] = (C_local[i_1_2] + (A_local[k_1] * B_local[k_1]));
        }
      }
    }
    mram_write(C_local + 0, (__mram_ptr void*)(C + ((tasklet_id * 8) + (i_1_1 * 2))), 8);
  }
}