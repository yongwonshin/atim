#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>

BARRIER_INIT(barrier, NR_TASKLETS);
int64_t red_buf0[16];
__mram_noinit int64_t A[16384];
__mram_noinit int64_t B_rf_global[1];
__mram_noinit int64_t B_rf_global_rf_global[16];
int main() {
  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  int64_t* B_rf_global_rf_global_local = (int64_t*) mem_alloc(1 * sizeof(int64_t));
  int64_t* A_local = (int64_t*) mem_alloc(64 * sizeof(int64_t));
    B_rf_global_rf_global_local[0] = (int64_t)0;
  for (int32_t i_2_0 = 0; i_2_0 < 16; ++i_2_0) {
    mram_read((__mram_ptr void*)(A + ((tasklet_id * 1024) + (i_2_0 * 64))), A_local + 0, 64 * sizeof(int64_t));
    for (int32_t i_2_1 = 0; i_2_1 < 64; ++i_2_1) {
      B_rf_global_rf_global_local[0] = (B_rf_global_rf_global_local[0] + A_local[i_2_1]);
    }
    B_rf_global_rf_global[tasklet_id] = B_rf_global_rf_global_local[0];
  }
  barrier_wait(&barrier);
  red_buf0[tasklet_id] = B_rf_global_rf_global[tasklet_id];
  barrier_wait(&barrier);
  if (tasklet_id < 8) {
    red_buf0[tasklet_id] = (red_buf0[tasklet_id] + red_buf0[(tasklet_id + 8)]);
  }
  barrier_wait(&barrier);
  if (tasklet_id < 4) {
    red_buf0[tasklet_id] = (red_buf0[tasklet_id] + red_buf0[(tasklet_id + 4)]);
  }
  barrier_wait(&barrier);
  if (tasklet_id < 2) {
    red_buf0[tasklet_id] = (red_buf0[tasklet_id] + red_buf0[(tasklet_id + 2)]);
  }
  barrier_wait(&barrier);
  if (tasklet_id < 1) {
    red_buf0[tasklet_id] = (red_buf0[tasklet_id] + red_buf0[(tasklet_id + 1)]);
  }
  barrier_wait(&barrier);
  if (tasklet_id == 0) {
    B_rf_global[0] = red_buf0[0];
  }
}