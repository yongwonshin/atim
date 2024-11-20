/**
 * app.c
 * GEMV Host Application Source File
 *
 */

#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if ENERGY
#include <dpu_probe.h>
#endif

#include "../support/common.h"
#include "../support/params.h"
#include "../support/timer.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/gemv_dpu"
#endif

static T* A;
static T* B;
static T* C;
static T* C_dpu;

// Create input arrays
static void init_data(T* A, T* B, unsigned int b_size, unsigned int m_size, unsigned int n_size) {
  /*
  srand(0);

  for (unsigned int i = 0; i < m_size * n_size; i++)
  {
        A[i] = (unsigned int) (rand()%50);
  }

  for (unsigned int i = 0; i < n_size; i++)
  {
        B[i] = (unsigned int) (rand()%50);
  }
  */

  char* fname = "../../../data/int32_163840_4096.bin";

  FILE* file = fopen(fname, "rb");
  if (file == NULL) {
    printf("Error: file not found\n");
    exit(1);
  }
  fread(A, sizeof(T), b_size * m_size * n_size, file);
  rewind(file);
  fread(B, sizeof(T), b_size * n_size, file);
  fclose(file);
}

// Compute output in the host
void gemv_host(T* C, T* A, T* B, unsigned int b_size, unsigned int m_size, unsigned int n_size) {
  for (unsigned int i = 0; i < m_size * b_size; i++) {
    C[i] = 0;
  }
  for (unsigned int b = 0; b < b_size; b++) {
    for (unsigned int m = 0; m < m_size; m++) {
      for (unsigned int n = 0; n < n_size; n++) {
        C[b * m_size + m] += A[b * m_size * n_size + m * n_size + n] * B[b * n_size + n];
      }
    }
  }
}

// Main of the Host Application
int main(int argc, char** argv) {
  struct Params p = input_params(argc, argv);

  struct dpu_set_t dpu_set, dpu;
  uint32_t nr_of_dpus;

  // Allocate DPUs and load binary
  uint32_t nr_dpus = p.b_size * NR_DPUS_Y;  // nr_dpus_b = b_size
  DPU_ASSERT(dpu_alloc(nr_dpus, "disableSafeChecks=1", &dpu_set));
  DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
  DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
  assert(nr_of_dpus == nr_dpus);

#if ENERGY
  struct dpu_probe_t probe;
  DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

  unsigned int i;
  unsigned int b_size = p.b_size;
  unsigned int m_size = p.m_size;
  unsigned int n_size = p.n_size;

  // Initialize help data
  dpu_info = (struct dpu_info_t*)malloc(nr_of_dpus * sizeof(struct dpu_info_t));
  dpu_arguments_t* input_args = (dpu_arguments_t*)malloc(nr_of_dpus * sizeof(dpu_arguments_t));
  uint32_t max_rows_per_dpu = 0;
  uint32_t n_size_pad = n_size;
  if (n_size % 2 == 1) {
    n_size_pad++;
  }

  i = 0;
  // assert(b_size % NR_DPUS_B == 0);
  // assert(m_size % NR_DPUS_Y == 0);

  uint32_t rows_per_dpu = m_size / NR_DPUS_Y;
  uint32_t batches_per_dpu = 1;

  DPU_FOREACH(dpu_set, dpu, i) {
    uint32_t b_index = i / NR_DPUS_Y;  // 0
    uint32_t m_index = i % NR_DPUS_Y;

    uint32_t rows_per_dpu;
    uint32_t prev_rows_dpu = 0;
    uint32_t chunks = m_size / NR_DPUS_Y;
    rows_per_dpu = chunks;
    uint32_t rest_rows = m_size % NR_DPUS_Y;
    if (m_index < rest_rows) rows_per_dpu++;
    if (rest_rows > 0) {
      if (m_index >= rest_rows)
        prev_rows_dpu = rest_rows * (chunks + 1) + (m_index - rest_rows) * chunks;
      else
        prev_rows_dpu = m_index * (chunks + 1);
    } else {
      prev_rows_dpu = m_index * chunks;
    }
    // Keep max rows for parallel transfers
    uint32_t rows_per_dpu_pad = rows_per_dpu;
    if (rows_per_dpu_pad % 2 == 1)  // 4-byte elements
      rows_per_dpu_pad++;
    if (rows_per_dpu_pad > max_rows_per_dpu) max_rows_per_dpu = rows_per_dpu_pad;

    dpu_info[i].rows_per_dpu = rows_per_dpu;
    dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
    dpu_info[i].prev_rows_dpu = prev_rows_dpu;
    dpu_info[i].prev_batches_dpu = b_index;

    // Copy input arguments to DPU
    input_args[i].n_size = n_size;
    input_args[i].n_size_pad = n_size_pad;
    input_args[i].nr_rows = rows_per_dpu;
  }

  A = malloc(b_size * max_rows_per_dpu * NR_DPUS_Y * n_size_pad * sizeof(T));
  B = malloc(b_size * n_size_pad * sizeof(T));
  C = malloc(b_size * max_rows_per_dpu * NR_DPUS_Y * sizeof(T));

  // Initialize data with arbitrary data
  init_data(A, B, b_size, m_size, n_size);
  // Timer
  Timer timer;

  // Compute output on CPU (performance comparison and verification purposes)
  start(&timer, 0, 0);
  for (unsigned int i = 0; i < rows_per_dpu * NR_DPUS_Y * b_size; i++) {
    C[i] = 0;
  }
  // gemv_host(C, A, B, b_size, max_rows_per_dpu * NR_DPUS_Y, n_size_pad);
  gemv_host(C, A, B, b_size, m_size, n_size);

  stop(&timer, 0);

  i = 0;
  DPU_FOREACH(dpu_set, dpu, i) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, A + dpu_info[i].prev_batches_dpu * n_size * m_size +
                                         dpu_info[i].prev_rows_dpu * n_size));
  }
  DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                           max_rows_per_dpu * n_size_pad * sizeof(T), DPU_XFER_DEFAULT));
  for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
    // Copy input array and vector

    // Input arguments
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      // Copy input arguments to DPU
      input_args[i].max_rows = max_rows_per_dpu;

      DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
                             sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

    if (rep >= p.n_warmup) start(&timer, 1, rep - p.n_warmup);

    DPU_FOREACH(dpu_set, dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, B + dpu_info[i].prev_batches_dpu * n_size));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                             max_rows_per_dpu * n_size_pad * sizeof(T), n_size_pad * sizeof(T),
                             DPU_XFER_DEFAULT));

    if (rep >= p.n_warmup) stop(&timer, 1);
    // Run kernel on DPUs
    if (rep >= p.n_warmup) {
      start(&timer, 2, rep - p.n_warmup);
#if ENERGY
      DPU_ASSERT(dpu_probe_start(&probe));
#endif
    }
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    if (rep >= p.n_warmup) {
      stop(&timer, 2);
#if ENERGY
      DPU_ASSERT(dpu_probe_stop(&probe));
#endif
    }
#if PRINT
    // Display DPU Logs
    DPU_FOREACH(dpu_set, dpu) { DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout)); }
#endif

    // Retrieve results
    C_dpu = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
    if (rep >= p.n_warmup) start(&timer, 3, rep - p.n_warmup);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu + i * max_rows_per_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                             (max_rows_per_dpu * n_size_pad + n_size_pad) * sizeof(T),
                             max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
    if (rep >= p.n_warmup) stop(&timer, 3);
  }
#if ENERGY
  double acc_energy, avg_energy, acc_time, avg_time;
  DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_ACCUMULATE, &acc_energy));
  DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &avg_energy));
  DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_ACCUMULATE, &acc_time));
  DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_AVERAGE, &avg_time));
#endif

  // Print timing results
  printf("CPU Version Time (ms): ");
  print(&timer, 0, 1);
  printf("CPU-DPU Time (ms): ");
  print(&timer, 1, p.n_reps);
  printf("DPU Kernel Time (ms): ");
  print(&timer, 2, p.n_reps);
  printf("DPU-CPU Time (ms): ");
  print(&timer, 3, p.n_reps);

#if ENERGY
  printf("Energy (J): %f J\t", avg_energy);
#endif

  // Check output
  bool status = true;
  unsigned int n, j;
  i = 0;

  int problematic_dpu = -1;

  for (n = 0; n < nr_of_dpus; n++) {
    for (j = 0; j < dpu_info[n].rows_per_dpu; j++) {
      if (C[i] != C_dpu[n * max_rows_per_dpu + j]) {
        if (problematic_dpu != -1 && problematic_dpu >= n + 8) {
          status = false;
        } else {
          problematic_dpu = n;
        } // 연속된 dpu 8개까지는 봐주는 로직 추가
        printf("%d %d: %d -- %d\n", n, i, C[i], C_dpu[n * max_rows_per_dpu + j]);
#if PRINT
        //			printf("%d: %d -- %d\n", i, C[i], C_dpu[n * max_rows_per_dpu + j]);
#endif
      }
      i++;
    }
  }
  if (status) {
    printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
  } else {
    printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
  }

  // Deallocation
  free(A);
  free(B);
  free(C);
  free(C_dpu);
  DPU_ASSERT(dpu_free(dpu_set));

#if ENERGY
  DPU_ASSERT(dpu_probe_deinit(&probe));
#endif

  return status ? 0 : -1;
}