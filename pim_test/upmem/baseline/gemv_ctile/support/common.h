#ifndef _COMMON_H_
#define _COMMON_H_

// Structures used by both the host and the dpu to communicate information
typedef struct
{
    uint32_t x_size;
    uint32_t y_size;
    uint32_t pipeline;
} dpu_arguments_t;

// Specific information for each DPU
struct dpu_info_t
{
    uint32_t x_per_dpu;
    uint32_t y_per_dpu;
    uint32_t prev_x_dpu;
    uint32_t prev_y_dpu;
    uint32_t prev_w_dpu;
    uint32_t prev_i_dpu;
};
struct dpu_info_t *dpu_info;

// Transfer size between MRAM and WRAM
#ifdef BL
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#else
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#endif

// Data type
#define T uint32_t

#ifndef ENERGY
#define ENERGY 0
#endif
#define PRINT 0

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_RESET "\x1b[0m"
#endif