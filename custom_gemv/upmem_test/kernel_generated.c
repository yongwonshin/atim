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
  int32_t* C_local = (int32_t*) mem_alloc(2* sizeof(int32_t));
  int32_t* A_local = (int32_t*) mem_alloc(64* sizeof(int32_t));
  int32_t* B_local = (int32_t*) mem_alloc(64* sizeof(int32_t));
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

/*
NO optimization
	lw r0, r22, -56
	lw r1, zero, C
	lw r2, r22, -60
	lsl r2, r2, 5
	lw r3, r22, -44
	lsl_add r2, r2, r3, 1
	lsl_add r1, r1, r2, 2
	move r2, 8
	call r23, mram_write
	jump .LBB0_17

W/ optimization
	lw r0, zero, C
	lw r1, r22, -12
	lw r2, r22, -24
	lsl_add r1, r1, r2, 1
	lsl_add r0, r0, r1, 2
*/