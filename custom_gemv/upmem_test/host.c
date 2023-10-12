#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

// 8192 * 8192

int main() {
    uint32_t M = 2048, N = 2048, DPUS = 4;
    uint32_t T = M / DPUS;
    uint32_t S = sizeof(float);

    struct dpu_set_t dpu_set, dpu;
    DPU_ASSERT(dpu_alloc(DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, "./kernel", NULL));

    uint32_t A_dpu = 0;
    float *A = (float *)malloc(M * N * S);
    uint32_t B_dpu = A_dpu + T * M * S;
    float *B = (float *)malloc(M * S);
    uint32_t C_dpu = B_dpu + M * S;
    float *C = (float *)malloc(N * S);

    float *C_host = (float *)malloc(N * S);
    
    srand(0);

    for (unsigned int i = 0; i < M * N; i++)
        A[i] = rand() % 20;
    for (unsigned int i = 0; i < M; i++)
        B[i] = rand() % 20;

    unsigned int i = 0;

    printf("DPU calculating \n");

    DPU_ASSERT(dpu_broadcast_to(dpu_set, "A", 0, &A_dpu, sizeof(uint32_t), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(dpu_set, "B", 0, &B_dpu, sizeof(uint32_t), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(dpu_set, "C", 0, &C_dpu, sizeof(uint32_t), DPU_XFER_DEFAULT));

    DPU_FOREACH(dpu_set, dpu, i)
        DPU_ASSERT(dpu_prepare_xfer(dpu, A + T * M * i)); 
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, A_dpu, T * M * S, DPU_XFER_DEFAULT));

    DPU_FOREACH(dpu_set, dpu, i)
        DPU_ASSERT(dpu_prepare_xfer(dpu, B));
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, B_dpu, M * S, DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    DPU_FOREACH(dpu_set, dpu, i)
        DPU_ASSERT(dpu_prepare_xfer(dpu, C + T * i));
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, C_dpu, T * S, DPU_XFER_DEFAULT));

    printf("HOST calculating \n");
    for (unsigned int i = 0; i < N; i++) {
        unsigned int c = 0;
        for (unsigned int j = 0; j < M; j++) {
            c += A[i * M + j] * B[j];
        }
        C_host[i] = c;
    }

    printf("CORRECTNESS CHECK\n");

    unsigned int wrong = 0;
    for (unsigned int i = 0; i < 128; i++) {
        for (unsigned int j = 0; j < 16; j++) {
            if (C[i * 16 + j] != C_host[i * 16 + j]) {
                wrong++;
                printf("%u %u wrong %f %f\n", i, j, C[i * 16 + j], C_host[i * 16 + j]);
            }
        }
    }
    printf("%u\n", wrong);


    free(A);
    free(B);
    free(C);
    free(C_host);
    DPU_ASSERT(dpu_free(dpu_set));
}

// gcc host.c `dpu-pkg-config --cflags --libs dpu` -g -lm -Wall -Wextra