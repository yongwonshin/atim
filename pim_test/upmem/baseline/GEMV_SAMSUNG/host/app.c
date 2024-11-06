/**
 * app.c
 * GEMV Host Application Source File
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#if ENERGY
#include <dpu_probe.h>
#endif

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/gemv_dpu"
#endif

static T *A;
static T *B;
static T *C;
static T *C_dpu;

// Create input arrays
static void init_data(T *A, T *B, unsigned int m_size, unsigned int n_size)
{
  srand(0);

  for (unsigned int i = 0; i < m_size * n_size; i++)
  {
    A[i] = (unsigned int)(rand() % 50);
  }

  for (unsigned int i = 0; i < n_size; i++)
  {
    B[i] = (unsigned int)(rand() % 50);
  }
}

// Compute output in the host
static void gemv_host(T *C, T *A, T *B, unsigned int m_size, unsigned int n_size)
{
  for (unsigned int i = 0; i < m_size; i++)
  {
    C[i] = 0;
  }

  for (unsigned int m = 0; m < m_size; m++)
  {
    for (unsigned int n = 0; n < n_size; n++)
    {
      C[m] += A[m * n_size + n] * B[n];
    }
  }
}

// Main of the Host Application
int main(int argc, char **argv)
{

  struct Params p = input_params(argc, argv);

  struct dpu_set_t dpu_set, dpu;
  uint32_t nr_of_dpus;

  // Allocate DPUs and load binary
  DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
  DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
  DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));

#if ENERGY
  struct dpu_probe_t probe;
  DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

  unsigned int i;
  unsigned int m_size = p.m_size;
  unsigned int n_size = p.n_size;

  unsigned int N = p.N;
  unsigned int P = 1;

  // Initialize help data
  dpu_info = (struct dpu_info_t *)malloc(nr_of_dpus * sizeof(struct dpu_info_t));
  dpu_arguments_t *input_args = (dpu_arguments_t *)malloc(nr_of_dpus * sizeof(dpu_arguments_t));

  uint32_t x_per_dpu = n_size / N;
  uint32_t y_per_dpu = m_size / (nr_of_dpus / N);

  i = 0;
  DPU_FOREACH(dpu_set, dpu, i)
  {
    // printf("NR_DPUS: %d, n_size: %d, nr_of_dpus: %d, N: %d, m_size: %d\n", NR_DPUS, n_size, nr_of_dpus, N, m_size);
    assert(n_size % N == 0);
    assert(nr_of_dpus % N == 0);
    assert(m_size % (nr_of_dpus / N) == 0);

    uint32_t prev_x_dpu = i % N;
    uint32_t prev_y_dpu = i / N;

    dpu_info[i].x_per_dpu = x_per_dpu;
    dpu_info[i].y_per_dpu = y_per_dpu;
    dpu_info[i].prev_x_dpu = prev_x_dpu;
    dpu_info[i].prev_y_dpu = prev_y_dpu;
    dpu_info[i].prev_w_dpu = prev_y_dpu * y_per_dpu * n_size + prev_x_dpu * x_per_dpu;
    dpu_info[i].prev_i_dpu = prev_x_dpu * x_per_dpu;

    // Copy input arguments to DPU
    input_args[i].x_size = x_per_dpu;
    input_args[i].y_size = y_per_dpu;
    input_args[i].pipeline = P;
  }

  A = malloc(m_size * n_size * sizeof(T));
  B = malloc(n_size * sizeof(T));
  C = malloc(m_size * sizeof(T));

  // Initialize data with arbitrary data
  init_data(A, B, m_size, n_size);

  // Timer
  Timer timer;

  // Compute output on CPU (performance comparison and verification purposes)
  start(&timer, 0, 0);
  gemv_host(C, A, B, m_size, n_size);
  stop(&timer, 0);

  // Copy weight matrix
  for (int c = 0; c < y_per_dpu; c++)
  {
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i)
    {
      DPU_ASSERT(dpu_prepare_xfer(dpu, A + dpu_info[i].prev_w_dpu + c * n_size));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, x_per_dpu * sizeof(T) * c, x_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
  }

  for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++)
  {
    C_dpu = malloc(m_size * N * sizeof(T));



    // Input arguments
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i)
    {
      // Copy input arguments to DPU
      DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

    {
      if (rep >= p.n_warmup)
        start(&timer, 1, rep - p.n_warmup);

      // Copy input vector (P = 0)
      DPU_FOREACH(dpu_set, dpu, i)
      {
        if (i < NR_DPUS / P)
        {
          DPU_ASSERT(dpu_prepare_xfer(dpu, B + dpu_info[i].prev_i_dpu));
        }
      }
      DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, x_per_dpu * y_per_dpu * sizeof(T), x_per_dpu * sizeof(T) / P, DPU_XFER_DEFAULT));

      if (rep >= p.n_warmup)
        stop(&timer, 1);

      // Run kernel on DPUs (P = 0)
      if (rep >= p.n_warmup)
      {
        start(&timer, 2, rep - p.n_warmup);
#if ENERGY
        DPU_ASSERT(dpu_probe_start(&probe));
#endif
      }

      DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

      if (rep >= p.n_warmup)
      {
        stop(&timer, 2);
#if ENERGY
        DPU_ASSERT(dpu_probe_stop(&probe));
#endif
      }
#if PRINT
      // Display DPU Logs
      DPU_FOREACH(dpu_set, dpu)
      {
        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
      }
#endif

      // Retrieve results (P = 0)
      if (rep >= p.n_warmup)
        start(&timer, 3, rep - p.n_warmup);
      i = 0;
      DPU_FOREACH(dpu_set, dpu, i)
      {
        // if (i < NR_DPUS / P)
        DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu + m_size * dpu_info[i].prev_x_dpu + dpu_info[i].prev_y_dpu * y_per_dpu));
      }
      DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, x_per_dpu * y_per_dpu * sizeof(T) + x_per_dpu * sizeof(T), y_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
      if (rep >= p.n_warmup)
        stop(&timer, 3);
    }

    {
      if (rep >= p.n_warmup)
        start(&timer, 4, rep - p.n_warmup);

      // Copy input vector (P = 1)
      DPU_FOREACH(dpu_set, dpu, i)
      {
        if (i >= NR_DPUS / P)
          DPU_ASSERT(dpu_prepare_xfer(dpu, B + dpu_info[i].prev_i_dpu));
      }
      DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, x_per_dpu * y_per_dpu * sizeof(T) + x_per_dpu * sizeof(T) / P, x_per_dpu * sizeof(T) / P, DPU_XFER_DEFAULT));

      if (rep >= p.n_warmup)
        stop(&timer, 4);

      // Run kernel on DPUs (P = 1)
      if (rep >= p.n_warmup)
      {
        start(&timer, 5, rep - p.n_warmup);
#if ENERGY
        DPU_ASSERT(dpu_probe_start(&probe));
#endif
      }

      DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

      if (rep >= p.n_warmup)
      {
        stop(&timer, 5);
#if ENERGY
        DPU_ASSERT(dpu_probe_stop(&probe));
#endif
      }
#if PRINT
      // Display DPU Logs
      DPU_FOREACH(dpu_set, dpu)
      {
        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
      }
#endif

      // Retrieve results (P = 1)
      if (rep >= p.n_warmup)
        start(&timer, 6, rep - p.n_warmup);
      i = 0;
      DPU_FOREACH(dpu_set, dpu, i)
      {
        // if (i >= NR_DPUS / 2)
        DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu + m_size * dpu_info[i].prev_x_dpu + dpu_info[i].prev_y_dpu * y_per_dpu));
      }
      DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, x_per_dpu * y_per_dpu * sizeof(T) + x_per_dpu * sizeof(T), y_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
      if (rep >= p.n_warmup)
        stop(&timer, 6);
    }

    // Final reduction
    if (rep >= p.n_warmup)
      start(&timer, 7, rep - p.n_warmup);

    for (int i = 0; i < m_size; i++)
    {
      for (int j = 1; j < N; j++)
      {
        C_dpu[i] += C_dpu[j * m_size + i];
      }
    }

    if (rep >= p.n_warmup)
      stop(&timer, 7);
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
  //printf("CPU-DPU Time (ms): ");
  print(&timer, 1, p.n_reps);
  //printf("DPU Kernel Time (ms): ");
  print(&timer, 2, p.n_reps);
  // printf("DPU-CPU Time (ms): ");
  print(&timer, 3, p.n_reps);
  printf("CPU-DPU Time (ms): ");
  print(&timer, 4, p.n_reps);
  printf("DPU Kernel Time (ms): ");
  print(&timer, 5, p.n_reps);
  printf("DPU-CPU Time (ms): ");
  print(&timer, 6, p.n_reps);
  printf("Reduction Time (ms): ");
  print(&timer, 7, p.n_reps);
  printf("\n");

#if ENERGY
  printf("Energy (J): %f J\t", avg_energy);
#endif

  // Check output
  bool status = true;
  unsigned int n, j;

  for (int i = 0; i < m_size; i++)
  {
    if (C[i] != C_dpu[i])
    {
      status = false;
      printf("%d, %d, %d\n", i, C[i], C_dpu[i]);
    }
  }
  if (status)
  {
    printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
  }
  else
  {
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
