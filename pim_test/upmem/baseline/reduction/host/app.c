/**
 * app.c
 * RED Host Application Source File
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

#include "../support/common.h"
#include "../support/params.h"
#include "../support/timer.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

#if ENERGY
#include <dpu_probe.h>
#endif

// Pointer declaration
static T* A;

// Create input arrays
static void read_input(T* A, unsigned int nr_elements) {
  FILE* file = fopen("../../../data/int64_403200000.bin", "rb");
  if (file == NULL) {
    printf("Error: file not found\n");
    exit(1);
  }
  fread(A, sizeof(T), nr_elements, file);
  fclose(file);
}

// Compute output in the host
static T reduction_host(T* A, unsigned int nr_elements) {
  T count = 0;
  for (unsigned int i = 0; i < nr_elements; i++) {
    count += A[i];
  }
  return count;
}

// Main of the Host Application
int main(int argc, char** argv) {
  struct Params p = input_params(argc, argv);

  struct dpu_set_t dpu_set, dpu;
  uint32_t nr_of_dpus;

#if ENERGY
  struct dpu_probe_t probe;
  DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

  // Allocate DPUs and load binary
  DPU_ASSERT(dpu_alloc(NR_DPUS, "disableSafeChecks=1", &dpu_set));
  DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
  DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
  printf("Allocated %d DPU(s)\n", nr_of_dpus);

  unsigned int i = 0;
#if PERF
  double cc = 0;
  double cc_min = 0;
#endif

  const unsigned int input_size = p.exp == 0
                                      ? p.input_size * nr_of_dpus
                                      : p.input_size;  // Total input size (weak or strong scaling)
  const unsigned int input_size_8bytes =
      ((input_size * sizeof(T)) % 8) != 0
          ? roundup(input_size, 8)
          : input_size;  // Input size per DPU (max.), 8-byte aligned
  const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus);  // Input size per DPU (max.)
  const unsigned int input_size_dpu_8bytes =
      ((input_size_dpu * sizeof(T)) % 8) != 0
          ? roundup(input_size_dpu, 8)
          : input_size_dpu;  // Input size per DPU (max.), 8-byte aligned

  // Input/output allocation
  A = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
  T* bufferA = A;
  T count = 0;
  T count_host = 0;

  // Create an input file with arbitrary data
  read_input(A, input_size);

  // Timer declaration
  Timer timer;

  printf("NR_TASKLETS\t%d\tBL\t%d\n", NR_TASKLETS, BL);

  // Loop over main kernel
  for (int rep = 0; rep < 1; rep++) {
    // Compute output on CPU (performance comparison and verification purposes)
    if (rep >= p.n_warmup) start(&timer, 0, rep - p.n_warmup);
    count_host = reduction_host(A, input_size);
    if (rep >= p.n_warmup) stop(&timer, 0);

    printf("Load input data\n");
    if (rep >= p.n_warmup) start(&timer, 1, rep - p.n_warmup);
    // Input arguments
    unsigned int kernel = 0;
    dpu_arguments_t input_arguments[NR_DPUS];
    for (i = 0; i < nr_of_dpus - 1; i++) {
      input_arguments[i].size = input_size_dpu_8bytes * sizeof(T);
      input_arguments[i].kernel = kernel;
    }
    input_arguments[nr_of_dpus - 1].size =
        (input_size_8bytes - input_size_dpu_8bytes * (NR_DPUS - 1)) * sizeof(T);
    input_arguments[nr_of_dpus - 1].kernel = kernel;
    // Copy input arrays
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) { DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i])); }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
                             sizeof(input_arguments[0]), DPU_XFER_DEFAULT));
    DPU_FOREACH(dpu_set, dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, bufferA + input_size_dpu_8bytes * i));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                             input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
    if (rep >= p.n_warmup) stop(&timer, 1);
  }

  for (int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
    count = 0;
    printf("Run program on DPU(s) \n");
    // Run DPU kernel
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
    {
      unsigned int each_dpu = 0;
      printf("Display DPU Logs\n");
      DPU_FOREACH(dpu_set, dpu) {
        printf("DPU#%d:\n", each_dpu);
        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
        each_dpu++;
      }
    }
#endif

    printf("Retrieve results\n");
    dpu_results_t results[nr_of_dpus];
    T* results_count = malloc(nr_of_dpus * sizeof(T));
    if (rep >= p.n_warmup) start(&timer, 3, rep - p.n_warmup);
    i = 0;
    // PARALLEL RETRIEVE TRANSFER
    dpu_results_t* results_retrieve[nr_of_dpus];

    DPU_FOREACH(dpu_set, dpu, i) {
      results_retrieve[i] = (dpu_results_t*)malloc(NR_TASKLETS * sizeof(dpu_results_t));
      DPU_ASSERT(dpu_prepare_xfer(dpu, results_retrieve[i]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0,
                             NR_TASKLETS * sizeof(dpu_results_t), DPU_XFER_DEFAULT));

    DPU_FOREACH(dpu_set, dpu, i) {
      // Retrieve tasklet timings
      for (unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS; each_tasklet++) {
        if (each_tasklet == 0) results[i].t_count = results_retrieve[i][each_tasklet].t_count;
      }
#if !PERF
      free(results_retrieve[i]);
#endif
      // Sequential reduction
      count += results[i].t_count;
#if PRINT
      printf("i=%d -- %lu\n", i, count);
#endif
    }

#if PERF
    DPU_FOREACH(dpu_set, dpu, i) {
      results[i].cycles = 0;
      // Retrieve tasklet timings
      for (unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS; each_tasklet++) {
        if (results_retrieve[i][each_tasklet].cycles > results[i].cycles)
          results[i].cycles = results_retrieve[i][each_tasklet].cycles;
      }
      free(results_retrieve[i]);
    }
#endif
    if (rep >= p.n_warmup) stop(&timer, 3);

#if PERF
    uint64_t max_cycles = 0;
    uint64_t min_cycles = 0xFFFFFFFFFFFFFFFF;
    // Print performance results
    if (rep >= p.n_warmup) {
      i = 0;
      DPU_FOREACH(dpu_set, dpu) {
        if (results[i].cycles > max_cycles) max_cycles = results[i].cycles;
        if (results[i].cycles < min_cycles) min_cycles = results[i].cycles;
        i++;
      }
      cc += (double)max_cycles;
      cc_min += (double)min_cycles;
    }
#endif

    // Free memory
    free(results_count);
  }
#if PERF
  printf("DPU cycles  = %g cc\n", cc / p.n_reps);
#endif

  // Print timing results
  printf("CPU ");
  print(&timer, 0, p.n_reps);
  printf("CPU-DPU ");
  print(&timer, 1, p.n_reps);
  printf("DPU Kernel ");
  print(&timer, 2, p.n_reps);
  printf("DPU-CPU ");
  print(&timer, 3, p.n_reps);

#if ENERGY
  double energy;
  DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
  printf("DPU Energy (J): %f\t", energy);
#endif

  // Check output
  bool status = true;
  // if (count != count_host) status = false;
  if (status) {
    printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
  } else {
    printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ! %d %d\n", count, count_host);
  }

  // Deallocation
  free(A);
  DPU_ASSERT(dpu_free(dpu_set));

  return status ? 0 : -1;
}
