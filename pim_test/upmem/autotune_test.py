import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from gemv import GEMV
from tensor import host_array
import numpy as np
import math
import argparse
from tvm.script import ir as I
from tvm.script import tir as T



# from tvm.script import ir as I
# from tvm.script import tir as T

def gemv_10000_factory(M, K, dtype):
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((10000, 10000), "int32"), B: T.Buffer((10000,), "int32"), C: T.Buffer((10000,), "int32")):
            T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
            with T.block("root"):
                T.reads()
                T.writes()
                T.block_attr({"meta_schedule.optimization_level": 5})
                C_rf_global = T.alloc_buffer((40, 10000), "int32")
                C_rf_global_local = T.alloc_buffer((40, 10000), "int32", scope="local")
                A_local = T.alloc_buffer((10000, 10000), "int32", scope="local")
                B_local = T.alloc_buffer((10000,), "int32", scope="local")
                for k_0 in T.thread_binding(40, thread="blockIdx.x", annotations={"bank": 1}):
                    for i_0 in T.thread_binding(40, thread="blockIdx.y", annotations={"bank": 1}):
                        for i_1 in T.thread_binding(2, thread="threadIdx.x"):
                            for i_2, i_3 in T.grid(1, 2):
                                for i_4_init, i_5_init in T.grid(4, 16):
                                    with T.block("C_rf_init"):
                                        v_i = T.axis.spatial(10000, (((i_0 * 2 + i_1 + i_2) * 2 + i_3) * 4 + i_4_init) * 16 + i_5_init)
                                        vk_0 = T.axis.spatial(40, k_0)
                                        T.where((((i_0 * 2 + i_1 + i_2) * 2 + i_3) * 4 + i_4_init) * 16 + i_5_init < 10000)
                                        T.reads()
                                        T.writes(C_rf_global_local[vk_0, v_i])
                                        T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": 1, "meta_schedule.tiling_structure": "SSSSRSRSR"})
                                        C_rf_global_local[vk_0, v_i] = 0
                                for k_1_0, i_4 in T.grid(1, 4):
                                    for k_1_1, i_5 in T.grid(1, 16):
                                        for ax0_ax1_fused in range(256):
                                            with T.block("A_local"):
                                                v0 = T.axis.spatial(10000, i_0 * 256 + i_1 * 128 + i_3 * 64 + i_4 * 16 + i_5)
                                                v1 = T.axis.spatial(10000, k_0 * 256 + ax0_ax1_fused)
                                                T.where(i_0 * 256 + i_1 * 128 + i_3 * 64 + i_4 * 16 + i_5 < 10000 and k_0 * 256 + ax0_ax1_fused % 256 < 10000)
                                                T.reads(A[v0, v1])
                                                T.writes(A_local[v0, v1])
                                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                                A_local[v0, v1] = A[v0, v1]
                                        for ax0_fused in range(256):
                                            with T.block("B_local"):
                                                v0 = T.axis.spatial(10000, k_0 * 256 + ax0_fused)
                                                T.where(k_0 * 256 + ax0_fused < 10000)
                                                T.reads(B[v0])
                                                T.writes(B_local[v0])
                                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                                B_local[v0] = B[v0]
                                        for k_1_2 in range(256):
                                            with T.block("C_rf_update"):
                                                v_i = T.axis.spatial(10000, (((i_0 * 2 + i_1 + i_2) * 2 + i_3) * 4 + i_4) * 16 + i_5)
                                                vk_0 = T.axis.spatial(40, k_0)
                                                vk_1 = T.axis.reduce(256, (k_1_0 + k_1_1) * 256 + k_1_2)
                                                T.where(k_0 * 256 + ((k_1_0 + k_1_1) * 256 + k_1_2) < 10000 and (((i_0 * 2 + i_1 + i_2) * 2 + i_3) * 4 + i_4) * 16 + i_5 < 10000)
                                                T.reads(C_rf_global_local[vk_0, v_i], A_local[v_i, vk_0 * 256 + vk_1], B_local[vk_0 * 256 + vk_1])
                                                T.writes(C_rf_global_local[vk_0, v_i])
                                                T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": 1, "meta_schedule.tiling_structure": "SSSSRSRSR"})
                                                C_rf_global_local[vk_0, v_i] = C_rf_global_local[vk_0, v_i] + A_local[v_i, vk_0 * 256 + vk_1] * B_local[vk_0 * 256 + vk_1]
                                    for ax0, ax1 in T.grid(1, 16):
                                        with T.block("C_rf_global_local"):
                                            v0 = T.axis.spatial(40, k_0 + ax0)
                                            v1 = T.axis.spatial(10000, i_0 * 256 + i_1 * 128 + i_3 * 64 + i_4 * 16 + ax1)
                                            T.where(i_0 * 256 + i_1 * 128 + i_3 * 64 + i_4 * 16 + ax1 < 10000)
                                            T.reads(C_rf_global_local[v0, v1])
                                            T.writes(C_rf_global[v0, v1])
                                            C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
                for i_0 in T.parallel(1):
                    for i_1 in range(10240):
                        with T.block("C_init"):
                            v_i = T.axis.spatial(10000, i_0 * 10240 + i_1)
                            T.where(i_0 * 10240 + i_1 < 10000)
                            T.reads()
                            T.writes(C[v_i])
                            T.block_attr({"meta_schedule.random_compute_producer": 1, "meta_schedule.tiling_structure": ""})
                            C[v_i] = 0
                        for k_0 in range(40):
                            with T.block("C_update"):
                                v_i = T.axis.spatial(10000, i_0 * 10240 + i_1)
                                vk_0 = T.axis.reduce(40, k_0)
                                T.where(i_0 * 10240 + i_1 < 10000)
                                T.reads(C[v_i], C_rf_global[vk_0, v_i])
                                T.writes(C[v_i])
                                T.block_attr({"meta_schedule.random_compute_producer": 1, "meta_schedule.tiling_structure": ""})
                                C[v_i] = C[v_i] + C_rf_global[vk_0, v_i]
    return Module

def upmem_gemv_factory(M, K, dtype):
    @I.ir_module
    class Module:
        @T.prim_func
        def main(
            A: T.Buffer((12288, 4096), "int32"),
            B: T.Buffer((4096,), "int32"),
            C: T.Buffer((12288,), "int32"),
        ):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "pragma_explicit_h2d": ["A"],
                    "tir.noalias": T.bool(True),
                }
            )
            with T.block("root"):
                T.reads()
                T.writes()
                T.block_attr({"meta_schedule.optimization_level": 4})
                C_rf_global = T.alloc_buffer((8, 12288), "int32")
                C_rf_global_local = T.alloc_buffer((8, 12288), "int32", scope="local")
                A_local = T.alloc_buffer((12288, 4096), "int32", scope="local")
                B_local = T.alloc_buffer((4096,), "int32", scope="local")
                for k_0 in T.thread_binding(
                    8, thread="blockIdx.x", annotations={"bank": 1}
                ):
                    for i_0 in T.thread_binding(
                        256, thread="blockIdx.y", annotations={"bank": 1}
                    ):
                        for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                            for i_2, i_3 in T.grid(1, 1):
                                for i_4_init, i_5_init in T.grid(1, 3):
                                    with T.block("C_rf_init"):
                                        v_i = T.axis.spatial(
                                            12288,
                                            i_0 * 48
                                            + i_1 * 3
                                            + i_2 * 3
                                            + i_3 * 3
                                            + i_4_init * 3
                                            + i_5_init,
                                        )
                                        vk_0 = T.axis.spatial(8, k_0)
                                        T.reads()
                                        T.writes(C_rf_global_local[vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRSRSR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, v_i] = 0
                                for k_1_0, i_4 in T.grid(4, 1):
                                    for k_1_1, i_5 in T.grid(16, 3):
                                        for ax0_ax1_fused in range(8):
                                            with T.block("A_local"):
                                                v0 = T.axis.spatial(
                                                    12288, i_0 * 48 + i_1 * 3 + i_5
                                                )
                                                v1 = T.axis.spatial(
                                                    4096,
                                                    k_0 * 512
                                                    + k_1_0 * 128
                                                    + k_1_1 * 8
                                                    + ax0_ax1_fused,
                                                )
                                                T.reads(A[v0, v1])
                                                T.writes(A_local[v0, v1])
                                                T.block_attr(
                                                    {"meta_schedule.cooperative_fetch": 1}
                                                )
                                                A_local[v0, v1] = A[v0, v1]
                                        for ax0_fused in range(8):
                                            with T.block("B_local"):
                                                v0 = T.axis.spatial(
                                                    4096,
                                                    k_0 * 512
                                                    + k_1_0 * 128
                                                    + k_1_1 * 8
                                                    + ax0_fused,
                                                )
                                                T.reads(B[v0])
                                                T.writes(B_local[v0])
                                                T.block_attr(
                                                    {"meta_schedule.cooperative_fetch": 1}
                                                )
                                                B_local[v0] = B[v0]
                                        for k_1_2 in range(8):
                                            with T.block("C_rf_update"):
                                                v_i = T.axis.spatial(
                                                    12288,
                                                    i_0 * 48
                                                    + i_1 * 3
                                                    + i_2 * 3
                                                    + i_3 * 3
                                                    + i_4 * 3
                                                    + i_5,
                                                )
                                                vk_0 = T.axis.spatial(8, k_0)
                                                vk_1 = T.axis.reduce(
                                                    512, k_1_0 * 128 + k_1_1 * 8 + k_1_2
                                                )
                                                T.reads(
                                                    C_rf_global_local[vk_0, v_i],
                                                    A_local[v_i, vk_0 * 512 + vk_1],
                                                    B_local[vk_0 * 512 + vk_1],
                                                )
                                                T.writes(C_rf_global_local[vk_0, v_i])
                                                T.block_attr(
                                                    {
                                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                        "meta_schedule.tiling_structure": "SSSSRSRSR",
                                                    }
                                                )
                                                C_rf_global_local[vk_0, v_i] = (
                                                    C_rf_global_local[vk_0, v_i]
                                                    + A_local[v_i, vk_0 * 512 + vk_1]
                                                    * B_local[vk_0 * 512 + vk_1]
                                                )
                                    for ax0, ax1 in T.grid(1, 3):
                                        with T.block("C_rf_global_local"):
                                            v0 = T.axis.spatial(8, k_0 + ax0)
                                            v1 = T.axis.spatial(
                                                12288, i_0 * 48 + i_1 * 3 + ax1
                                            )
                                            T.reads(C_rf_global_local[v0, v1])
                                            T.writes(C_rf_global[v0, v1])
                                            C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
                for i in range(12288):
                    with T.block("C_init"):
                        v_i = T.axis.spatial(12288, i)
                        T.reads()
                        T.writes(C[v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[v_i] = 0
                    for k_0 in range(8):
                        with T.block("C_update"):
                            vk_0, v_i = T.axis.remap("RS", [k_0, i])
                            T.reads(C[v_i], C_rf_global[vk_0, v_i])
                            T.writes(C[v_i])
                            T.block_attr(
                                {
                                    "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                    "meta_schedule.random_compute_producer": 1,
                                }
                            )
                            C[v_i] = C[v_i] + C_rf_global[vk_0, v_i]
    return Module


def autotune(M, K, n_xb=1, n_yb=1, n_yt=16, n_cache=64, n_rt=64, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(gemv_10000_factory(M, K, dtype))
    return sch

gemv=GEMV(repeat=1)
gemv.test(autotune, M=10000, K=10000)