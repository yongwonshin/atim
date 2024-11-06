/*
 * Matrix vector multiplication with multiple tasklet
 *
 */
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>

#include "../support/common.h"

#define roundup(n, m) ((n / m) * m + m)

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

// GEMV
static void gemv(T *bufferC, T *bufferA, T *bufferB, int block_size, int pos)
{
  for (unsigned int i = 0; i < block_size / sizeof(T); i++)
  {
    bufferC[pos] += bufferA[i] * bufferB[i];
  }
  return;
}

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// main
int main()
{
  unsigned int tasklet_id = me();
#if PRINT
  // printf("tasklet_id = %u\n", tasklet_id);
#endif
  if (tasklet_id == 0)
  {              // Initialize once the cycle counter
    mem_reset(); // Reset the heap
  }
  // Barrier
  barrier_wait(&my_barrier);

  int32_t x_size = DPU_INPUT_ARGUMENTS.x_size / DPU_INPUT_ARGUMENTS.pipeline;
  int32_t y_size = DPU_INPUT_ARGUMENTS.y_size;

  unsigned int y_per_tasklet = y_size / NR_TASKLETS;
  unsigned int start_y = tasklet_id * y_per_tasklet;

  unsigned int x_size_in_bytes = x_size * sizeof(T);

  // Address of the current row in MRAM
  uint32_t mram_base_addr_A = (uint32_t)(DPU_MRAM_HEAP_POINTER + start_y * x_size * sizeof(T));
  uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + y_size * x_size * sizeof(T));
  uint32_t mram_base_addr_C = (uint32_t)(DPU_MRAM_HEAP_POINTER + y_size * x_size * sizeof(T) + x_size_in_bytes);

  unsigned int block_size = x_size_in_bytes;
  unsigned int n_iter = 1;
  if (block_size > 1024)
  {
    block_size = 1024;
    n_iter = x_size_in_bytes / 1024;
  }

  // Inititalize a local cache to store the MRAM block
  T *cache_A = (T *)mem_alloc(block_size);
  T *cache_B = (T *)mem_alloc(block_size);
  T *cache_C = (T *)mem_alloc(y_per_tasklet * sizeof(T));

#if PRINT
  printf("id: %d, rows_per_tasklet = %d\n", tasklet_id, rows_per_tasklet);
  printf("id: %d, start_row = %d\n", tasklet_id, start_row);
#endif

  // clear the cache
  for (unsigned int c = 0; c < y_per_tasklet; c++)
  {
    cache_C[c] = 0;
  }

  for (int n = 0; n < n_iter; n++)
  {
    mram_read((__mram_ptr void const *)(mram_base_addr_B + n * block_size), cache_B, block_size);

    // Iterate over rows
    for (unsigned int i = 0; i < y_per_tasklet; i++)
    {
      mram_read((__mram_ptr void const *)(mram_base_addr_A + (i * n_iter + n) * block_size), cache_A, block_size);

      // Compute GEMV
      gemv(cache_C, cache_A, cache_B, block_size, i);
    }
  }

  // Write cache to current MRAM block
  mram_write(cache_C, (__mram_ptr void *)(mram_base_addr_C + start_y * sizeof(T)), y_per_tasklet * sizeof(T));

  return 0;
}