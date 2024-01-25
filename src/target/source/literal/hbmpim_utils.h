#ifndef TVM_TARGET_SOURCE_LITERAL_HBMPIM_UTILS_H_
#define TVM_TARGET_SOURCE_LITERAL_HBMPIM_UTILS_H_

static constexpr const char* _hbmpim_info_def = R"(
#define DIM_OUT_PIM (3200)
#define PIM_GEMV_IN_ALIGN (256)
#define PIM_GEMV_OUT_ALIGN (4096)
#define PIM_ELTWISE_ALIGN (256 * 1024)

typedef enum __PimAddrMap {
    AMDGPU_VEGA20,
} PimAddrMap;

typedef struct __PimBlockInfo {
    PimAddrMap pim_addr_map;
    int num_banks;
    int num_bank_groups;
    int num_rank_bit;
    int num_row_bit;
    int num_col_high_bit;
    int num_bank_high_bit;
    int num_bankgroup_bit;
    int num_bank_low_bit;
    int num_chan_bit;
    int num_col_low_bit;
    int num_offset_bit;
    int num_grf;
    int num_grf_A;
    int num_grf_B;
    int num_srf;
    int num_col;
    int num_row;
    int bl;
    int num_pim_blocks;
    int num_pim_rank;
    int num_pim_chan;
    int trans_size;
    int num_out_per_grf;
} PimBlockInfo;

static const PimBlockInfo vega20_pbi = {
    .pim_addr_map = AMDGPU_VEGA20,
    .num_banks = 16,
    .num_bank_groups = 4,
    .num_rank_bit = 1,
    .num_row_bit = 14,
    .num_col_high_bit = 3,
    .num_bank_high_bit = 1,
    .num_bankgroup_bit = 2,
    .num_bank_low_bit = 1,
    .num_chan_bit = 6,
    .num_col_low_bit = 2,
    .num_offset_bit = 5,
    .num_grf = 8,
    .num_grf_A = 8,
    .num_grf_B = 8,
    .num_srf = 4,
    .num_col = 128,
    .num_row = 16384,
    .bl = 4,
    .num_pim_blocks = 8,
    .num_pim_rank = 1,
    .num_pim_chan = 64,
    .trans_size = 32,
    .num_out_per_grf = 16,
};

typedef struct __PimMemTraceData {
    uchar data[32];
    ulong addr;
    int block_id;
    int thread_id;
    char cmd;
} PimMemTraceData;

typedef enum __PimBankType {
    EVEN_BANK,
    ODD_BANK,
    ALL_BANK,
} PimBankType;

typedef enum __PimMode {
    SB_MODE,
    HAB_MODE,
    HAB_PIM_MODE,
} PimMode;

typedef enum __PimGemvType {
    TILE_ACCUM,
    TILE_TREE,
    NEXT_PIM,
} PimGemvType;

typedef enum __PimKrnlType {
    OPTIMAL,
    PIM,
    CUSTOM_GPU,
} PimKrnlType;

#ifdef EMULATOR
typedef struct __PimMemTracer {
    ulong g_fba;
    PimMemTraceData* g_fmtd16;
    int g_ridx[64];
    int g_idx[64];
    int m_width;
} PimMemTracer;
#endif

)";

static constexpr const char* _hbmpim_kernel_utils_def = R"(
#define NVIDIA_GPU 1

static __constant uchar hab_to_hab_pim[32] = {
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

static __constant uchar hab_pim_to_hab[32] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

static __constant uchar gemv_hab_to_hab_pim[32] = {
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

uint mask_by_bit_(uint value, uint start, uint end)
{
    int length = start - end + 1;
    value >>= end;
    return value & ((1 << length) - 1);
}

ulong addr_gen_(uint chan, uint rank, uint bankgroup, uint bank, uint row, uint col)
{
    /* vega20 memory map info */
    int num_row_bit = 14;
    int num_col_high_bit = 3;
    int num_bank_high_bit = 1;
    int num_bankgroup_bit = 2;
    int num_bank_low_bit = 1;
    int num_chan_bit = 6;
    int num_offset_bit = 5;

    ulong addr = 0;

    addr = rank;

    addr <<= num_row_bit;
    addr |= row;

    addr <<= num_col_high_bit;
    addr |= mask_by_bit_(col, 4, 2);

    addr <<= num_bank_high_bit;
    addr |= mask_by_bit_(bank, 1, 1);

    addr <<= num_bankgroup_bit;
    addr |= bankgroup;

    addr <<= num_bank_low_bit;
    addr |= mask_by_bit_(bank, 0, 0);

    addr <<= num_chan_bit - 1;
    addr |= mask_by_bit_(chan, num_chan_bit - 1, 1);

    addr <<= 1;
    addr |= mask_by_bit_(col, 1, 1);

    addr <<= 1;
    addr |= mask_by_bit_(chan, 0, 0);

    addr <<= 1;
    addr |= mask_by_bit_(col, 0, 0);

    addr <<= num_offset_bit;

#if TARGET && RADEON7
    /* we assume pim kernel run on vega20(32GB) system */
    /* but SAIT server is vega20(16GB) system */
    /* so upper 2bit should be set as 0 for normal work */
    ulong mask = 0x1FFFFFFFF;
    addr &= mask;
#endif

#ifndef EMULATOR
    // only for TARGET.
    // currently for opencl, memory is not mapped for control registers, so this
    //  address reduction is done for performance check on MI50 hardware.(32GB - > 1GB)
    ulong mask = 0x3FFFFFFF;
    addr &= mask;
#endif

    return addr;
}

#ifdef EMULATOR
void _R_CMD(__global volatile uchar* __restrict__ addr, __global PimMemTracer* __restrict__ emulator_trace)
{
    int bid = get_group_id(0);
    int tid = get_local_id(0);
    int row = bid * emulator_trace->m_width;
    int ridx = row + atomic_add(&emulator_trace->g_ridx[bid], 1);

    emulator_trace->g_fmtd16[ridx].block_id = bid;
    emulator_trace->g_fmtd16[ridx].thread_id = tid;
    emulator_trace->g_fmtd16[ridx].addr = (ulong)addr - emulator_trace->g_fba;
    emulator_trace->g_fmtd16[ridx].cmd = 'R';
}

void _W_CMD(__global volatile uchar* __restrict__ addr, __global PimMemTracer* __restrict__ emulator_trace)
{
    int bid = get_group_id(0);
    int tid = get_local_id(0);
    int row = bid * emulator_trace->m_width;
    int ridx = row + atomic_add(&emulator_trace->g_ridx[bid], 1);

    emulator_trace->g_fmtd16[ridx].block_id = bid;
    emulator_trace->g_fmtd16[ridx].thread_id = tid;
    emulator_trace->g_fmtd16[ridx].addr = (ulong)addr - emulator_trace->g_fba;
    emulator_trace->g_fmtd16[ridx].cmd = 'W';
}

void _W_CMD_R(__global volatile uchar* __restrict__ addr, __global volatile uchar* __restrict__ src,
              __global PimMemTracer* __restrict__ emulator_trace)
{
    int bid = get_group_id(0);
    int tid = get_local_id(0);
    int row = bid * emulator_trace->m_width;
    int ridx = row + atomic_add(&emulator_trace->g_ridx[bid], 1);

    // memcpy(emulator_trace->g_fmtd16[ridx].data, (uchar*)src, 16);
    for (int i = 0; i < 16; i++) {
        emulator_trace->g_fmtd16[ridx].data[i] = src[i];
    }
    emulator_trace->g_fmtd16[ridx].block_id = bid;
    emulator_trace->g_fmtd16[ridx].thread_id = tid;
    emulator_trace->g_fmtd16[ridx].addr = (ulong)addr - emulator_trace->g_fba;
    emulator_trace->g_fmtd16[ridx].cmd = 'W';
}

void _W_CMD_R_C(__global volatile uchar* __restrict__ addr, __constant volatile uchar* __restrict__ src,
                __global PimMemTracer* __restrict__ emulator_trace)
{
    int bid = get_group_id(0);
    int tid = get_local_id(0);
    int row = bid * emulator_trace->m_width;
    int ridx = row + atomic_add(&emulator_trace->g_ridx[bid], 1);

    // memcpy(emulator_trace->g_fmtd16[ridx].data, (uchar*)src, 16);
    for (int i = 0; i < 16; i++) {
        emulator_trace->g_fmtd16[ridx].data[i] = src[i];
    }
    emulator_trace->g_fmtd16[ridx].block_id = bid;
    emulator_trace->g_fmtd16[ridx].thread_id = tid;
    emulator_trace->g_fmtd16[ridx].addr = (ulong)addr - emulator_trace->g_fba;
    emulator_trace->g_fmtd16[ridx].cmd = 'W';
}

void _B_CMD(int type, __global PimMemTracer* __restrict__ emulator_trace)
{
    int row = get_group_id(0) * emulator_trace->m_width;
    int midx = row + atomic_add(&emulator_trace->g_ridx[get_group_id(0)], 1);

    // memset(emulator_trace->g_fmtd16[midx].data, 0, 16);
    for (int i = 0; i < 16; i++) {
        emulator_trace->g_fmtd16[midx].data[i] = 0;
    }
    emulator_trace->g_fmtd16[midx].block_id = get_group_id(0);
    emulator_trace->g_fmtd16[midx].thread_id = get_local_id(0);
    emulator_trace->g_fmtd16[midx].addr = 0;
    emulator_trace->g_fmtd16[midx].cmd = 'B';

    (type == 0) ? barrier(CLK_LOCAL_MEM_FENCE) : mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

#else  /* TARGET */

// dummy command to send address via load/ read call in opencl.
void _R_CMD(__global volatile uchar* __restrict__ addr)
{
    float4 read_data;
    read_data = vload4(0, (__global float*)addr);
}

// dummy command to send address via store/ write call in opencl.
void _W_CMD(__global volatile uchar* __restrict__ addr)
{
    float4 write_src;
    vstore4(write_src, 0, (__global float*)addr);
}

// reading the data from src and writing it to addr using load and store calls.
void _W_CMD_R(__global volatile uchar* __restrict__ addr, __global volatile uchar* __restrict__ src)
{
    int4 write_data = vload4(0, (__global int*)src);
    vstore4(write_data, 0, (__global int*)addr);
}

void _W_CMD_R_C(__global volatile uchar* __restrict__ addr, __constant volatile uchar* __restrict__ src)
{
    int4 write_data = vload4(0, (__constant int*)src);
    vstore4(write_data, 0, (__global int*)addr);
}

void _B_CMD(int type)
{
    (type == 0) ? barrier(CLK_LOCAL_MEM_FENCE) : mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}
#endif /* EMULATOR */

#ifdef EMULATOR
#define R_CMD(x) _R_CMD(x, emulator_trace)
#define W_CMD(x) _W_CMD(x, emulator_trace)
#define W_CMD_R(x, y) _W_CMD_R(x, y, emulator_trace)
#define W_CMD_R_C(x, y) _W_CMD_R_C(x, y, emulator_trace)
#define B_CMD(x) _B_CMD(x, emulator_trace)
#else
#define R_CMD(x) _R_CMD(x)
#define W_CMD(x) _W_CMD(x)
#define W_CMD_R(x, y) _W_CMD_R(x, y)
#define B_CMD(x) _B_CMD(x)
#define W_CMD_R_C(x, y) _W_CMD_R_C(x, y)
#endif

/*
Sugars to represent each common step in executing a particular kernel on PIM.
* park_in
* change sb to hab mode
* program_crf / program srf
* change hab to hab pim mode
* change hab pim to hab mode.
* change hab to sb mode.
* park_out
*/
void _park_in(__global uchar* __restrict__ pim_ctr, int gidx, int num_ba, ulong offset
#ifdef EMULATOR
              ,
              __global PimMemTracer* emulator_trace
#endif
              )
{
    ulong addr;
    addr = addr_gen_(get_group_id(0), 0, gidx / num_ba, gidx % num_ba, (1 << 13), 0);
    W_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
}

void _change_sb_hab(__global uchar* __restrict__ pim_ctr, ulong offset
#ifdef EMULATOR
                    ,
                    __global PimMemTracer* emulator_trace
#endif
                    )
{
    ulong addr;
    addr = addr_gen_(get_group_id(0), 0, 2, 0, 0x27ff, 0x1f);
    W_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
    addr = addr_gen_(get_group_id(0), 0, 2, 1, 0x27ff, 0x1f);
    W_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
    addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x27ff, 0x1f);
    W_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
    addr = addr_gen_(get_group_id(0), 0, 0, 1, 0x27ff, 0x1f);
    W_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
}

void _program_crf(__global uchar* __restrict__ pim_ctr, int gidx, __global uchar* crf_binary, ulong offset
#ifdef EMULATOR
                  ,
                  __global PimMemTracer* emulator_trace
#endif
                  )
{
    ulong addr;
    addr = addr_gen_(get_group_id(0), 0, 0, 1, 0x3fff, 0x4 + gidx);
    W_CMD_R(&pim_ctr[addr + offset], crf_binary + offset);
}

// specific to gemv and batch norm,
void _program_crf_mod(__global uchar* __restrict__ pim_ctr, int gidx, __global uchar* crf_binary, ulong offset
#ifdef EMULATOR
                     ,
                     __global PimMemTracer* emulator_trace
#endif
                     )
{
    ulong addr;
    addr = addr_gen_(get_group_id(0), 0, 0, 1, 0x3fff, 0x4 + gidx);
    W_CMD_R(&pim_ctr[addr + offset], crf_binary + (get_local_id(0) << 4));
    R_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
}

void _program_srf(__global uchar* __restrict__ pim_ctr, __global uchar* srf_binary, ulong offset
#ifdef EMULATOR
                  ,
                  __global PimMemTracer* emulator_trace
#endif
                  )
{
    ulong addr;
    addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x3fff, 0x0);
    W_CMD_R(&pim_ctr[addr + 32 + offset], srf_binary + offset);
}

void _change_hab_habpim(__global uchar* __restrict__ pim_ctr, ulong offset
#ifdef EMULATOR
                        ,
                        __global PimMemTracer* emulator_trace
#endif
                        )
{
    ulong addr;
    addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x3fff, 0x0);
    W_CMD_R_C(&pim_ctr[addr + offset], hab_to_hab_pim + offset);
    R_CMD(&pim_ctr[addr + offset]);
}

void _change_gemv_hab_habpim(__global uchar* __restrict__ pim_ctr, ulong offset
#ifdef EMULATOR
                        ,
                        __global PimMemTracer* emulator_trace
#endif
                        )
{
    ulong addr;
    addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x3fff, 0x0);
    W_CMD_R_C(&pim_ctr[addr + offset], gemv_hab_to_hab_pim + offset);
    R_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
}

void _change_habpim_hab(__global uchar* __restrict__ pim_ctr, ulong offset
#ifdef EMULATOR
                        ,
                        __global PimMemTracer* emulator_trace
#endif
                        )
{
    ulong addr;
    addr = addr_gen_(get_group_id(0), 0, 0, 0, 0x3fff, 0x0);
    W_CMD_R_C(&pim_ctr[addr + offset], hab_pim_to_hab + offset);
    R_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
}

void _change_hab_sb(__global uchar* __restrict__ pim_ctr, int gidx, ulong offset
#ifdef EMULATOR
                    ,
                    __global PimMemTracer* emulator_trace
#endif
                    )
{
    ulong addr;
    addr = addr_gen_(get_group_id(0), 0, 0, gidx, 0x2fff, 0x1f);
    W_CMD(&pim_ctr[addr + offset]);
    R_CMD(&pim_ctr[addr + offset]);
    B_CMD(1);
}

void _park_out(__global uchar* __restrict__ pim_ctr, int gidx, int num_ba, ulong offset
#ifdef EMULATOR
               ,
               __global PimMemTracer* emulator_trace
#endif
               )
{
    ulong addr;
    addr = addr_gen_(get_group_id(0), 0, gidx / num_ba, gidx % num_ba, (1 << 13), 0);
    W_CMD(&pim_ctr[addr + offset]);
}

ulong addr_gen_s(uint chan, uint rank, uint bankgroup, uint bank, uint row, uint col, uint offset)
{
    uint offset_size = 1 << vega20_pbi.num_offset_bit;
    uint col_size = vega20_pbi.num_col / vega20_pbi.bl;

    uint offset_s = offset % offset_size;
    uint new_col = col + offset / offset_size;
    uint col_s = new_col % col_size;
    uint row_s = row + new_col / col_size;

    return addr_gen_(chan, rank, bankgroup, bank, row_s, col_s) + offset_s;
}

#ifdef EMULATOR
#define park_in(a, b, c, d) _park_in(a, b, c, d, emulator_trace)
#define change_sb_hab(a, b) _change_sb_hab(a, b, emulator_trace)
#define program_crf(a, b, c, d) _program_crf(a, b, c, d, emulator_trace)
#define program_crf_mod(a, b, c, d) _program_crf_mod(a, b, c, d, emulator_trace)
#define program_srf(a, b, c) _program_srf(a, b, c, emulator_trace)
#define change_hab_habpim(a, b) _change_hab_habpim(a, b, emulator_trace)
#define change_gemv_hab_habpim(a, b) _change_gemv_hab_habpim(a, b, emulator_trace)
#define change_habpim_hab(a, b) _change_habpim_hab(a, b, emulator_trace)
#define change_hab_sb(a, b, c) _change_hab_sb(a, b, c, emulator_trace)
#define park_out(a, b, c, d) _park_out(a, b, c, d, emulator_trace)

#else
#define park_in(a, b, c, d) _park_in(a, b, c, d)
#define change_sb_hab(a, b) _change_sb_hab(a, b)
#define program_crf(a, b, c, d) _program_crf(a, b, c, d)
#define program_crf_mod(a, b, c, d) _program_crf_mod(a, b, c, d)
#define program_srf(a, b, c) _program_srf(a, b, c)
#define change_hab_habpim(a, b) _change_hab_habpim(a, b)
#define change_gemv_hab_habpim(a, b) _change_gemv_hab_habpim(a, b)
#define change_habpim_hab(a, b) _change_habpim_hab(a, b)
#define change_hab_sb(a, b, c) _change_hab_sb(a, b, c)
#define park_out(a, b, c, d) _park_out(a, b, c, d)
#endif

)";

#endif  // TVM_TARGET_SOURCE_LITERAL_HBMPIM_UTILS_H_
