#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <seqread.h>
#include <stdint.h>
#include <stdio.h>

BARRIER_INIT(barrier, NR_TASKLETS);

__host int32_t blockIdx_x;
__host int32_t blockIdx_y;
__host int32_t blockIdx_z;

__mram_noinit int32_t A[2048];
__mram_noinit int32_t B[2048];
__mram int32_t C[2048];
int main() {
  unsigned int tasklet_id = me();
  int bx = blockIdx_x;
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  int32_t* A_local = (int32_t*)mem_alloc(256 * sizeof(int32_t));
  int32_t* B_local = (int32_t*)mem_alloc(256 * sizeof(int32_t));
  int t = tasklet_id * 256;
  int licm = (blockIdx_x * 2048) + t;
  if (licm < 63172) {
    mram_read((__mram_ptr void*)(A + t), A_local + 0, 1024);
    mram_read((__mram_ptr void*)(B + t), B_local + 0, 1024);
    int size = 256 < 63172 - licm ? 256 : 63172 - licm;
    for (int32_t i_3 = 0; i_3 < size; ++i_3) {
      // if (licm + i_3 < 63172) {
      A_local[i_3] = (A_local[i_3] + B_local[i_3]);
      //}
    }
    mram_write(A_local + 0, (__mram_ptr void*)(C + (tasklet_id * 256)), 1024);
  }
}