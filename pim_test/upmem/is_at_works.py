import numpy as np
import sys

sys.path.append("/root/dev/tvm/python")

import math
import tvm
import time
import math
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.transform import *
from tensor import host_array
from tvm.script import ir as I
from tvm.script import tir as T


@I.ir_module
class Module6BFC:
    @T.prim_func
    def main(
        A: T.Buffer((16384, 4096), "int32"),
        B: T.Buffer((4096,), "int32"),
        C: T.Buffer((16384,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((16, 16384), "int32")
        C_rf_global_local = T.alloc_buffer((16, 16384), "int32", scope="local")
        A_local = T.alloc_buffer((16384, 4096), "int32", scope="local")
        B_local = T.alloc_buffer((4096,), "int32", scope="local")
        for k_0 in T.thread_binding(16, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(128, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(2):
                        for i_3 in range(4):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(16384, i_0 * 128 + i_1 * 8 + i_2 * 4 + i_3)
                                vk_0 = T.axis.spatial(16, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(32):
                                for ax0_ax1_fused in range(8):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            16384, i_0 * 128 + i_1 * 8 + i_2 * 4 + i_3
                                        )
                                        v1 = T.axis.spatial(
                                            4096, k_0 * 256 + k_1_0 * 8 + ax0_ax1_fused
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(8):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(4096, k_0 * 256 + k_1_0 * 8 + ax0_fused)
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(8):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            16384, i_0 * 128 + i_1 * 8 + i_2 * 4 + i_3
                                        )
                                        vk_0 = T.axis.spatial(16, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 8 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 256 + vk_1],
                                            B_local[vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 256 + vk_1]
                                            * B_local[vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 4):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(16, k_0 + ax0)
                                v1 = T.axis.spatial(16384, i_0 * 128 + i_1 * 8 + i_2 * 4 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(1):
            for i_1 in range(16384):
                with T.block("update_init"):
                    vi = T.axis.spatial(16384, i_0 * 16384 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(16):
                    with T.block("update_update"):
                        vi = T.axis.spatial(16384, i_0 * 16384 + i_1)
                        vk_0 = T.axis.reduce(16, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module6BFCProj:
    @T.prim_func
    def main(
        A: T.Buffer((4096, 16384), "int32"),
        B: T.Buffer((16384,), "int32"),
        C: T.Buffer((4096,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((32, 4096), "int32")
        C_rf_global_local = T.alloc_buffer((32, 4096), "int32", scope="local")
        A_local = T.alloc_buffer((4096, 16384), "int32", scope="local")
        B_local = T.alloc_buffer((16384,), "int32", scope="local")
        for k_0 in T.thread_binding(32, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(64, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(1):
                        for i_3 in range(4):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(4096, i_0 * 64 + i_1 * 4 + i_2 * 4 + i_3)
                                vk_0 = T.axis.spatial(32, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(64):
                                for ax0_ax1_fused in range(8):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(4096, i_0 * 64 + i_1 * 4 + i_3)
                                        v1 = T.axis.spatial(
                                            16384, k_0 * 512 + k_1_0 * 8 + ax0_ax1_fused
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(8):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            16384, k_0 * 512 + k_1_0 * 8 + ax0_fused
                                        )
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(8):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            4096, i_0 * 64 + i_1 * 4 + i_2 * 4 + i_3
                                        )
                                        vk_0 = T.axis.spatial(32, k_0)
                                        vk_1 = T.axis.reduce(512, k_1_0 * 8 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 512 + vk_1],
                                            B_local[vk_0 * 512 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 512 + vk_1]
                                            * B_local[vk_0 * 512 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 4):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(32, k_0 + ax0)
                                v1 = T.axis.spatial(4096, i_0 * 64 + i_1 * 4 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(1):
            for i_1 in range(4096):
                with T.block("update_init"):
                    vi = T.axis.spatial(4096, i_0 * 4096 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(32):
                    with T.block("update_update"):
                        vi = T.axis.spatial(4096, i_0 * 4096 + i_1)
                        vk_0 = T.axis.reduce(32, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module6BQKVGen:
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
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((8, 12288), "int32")
        C_rf_global_local = T.alloc_buffer((8, 12288), "int32", scope="local")
        A_local = T.alloc_buffer((12288, 4096), "int32", scope="local")
        B_local = T.alloc_buffer((4096,), "int32", scope="local")
        for k_0 in T.thread_binding(8, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(256, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(24, thread="threadIdx.x"):
                    for i_2 in range(1):
                        for i_3 in range(2):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(12288, i_0 * 48 + i_1 * 2 + i_2 * 2 + i_3)
                                vk_0 = T.axis.spatial(8, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(32):
                                for ax0_ax1_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(12288, i_0 * 48 + i_1 * 2 + i_3)
                                        v1 = T.axis.spatial(
                                            4096, k_0 * 512 + k_1_0 * 16 + ax0_ax1_fused
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            4096, k_0 * 512 + k_1_0 * 16 + ax0_fused
                                        )
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(16):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            12288, i_0 * 48 + i_1 * 2 + i_2 * 2 + i_3
                                        )
                                        vk_0 = T.axis.spatial(8, k_0)
                                        vk_1 = T.axis.reduce(512, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 512 + vk_1],
                                            B_local[vk_0 * 512 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 512 + vk_1]
                                            * B_local[vk_0 * 512 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 2):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(8, k_0 + ax0)
                                v1 = T.axis.spatial(12288, i_0 * 48 + i_1 * 2 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(16):
            for i_1 in range(768):
                with T.block("update_init"):
                    vi = T.axis.spatial(12288, i_0 * 768 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(8):
                    with T.block("update_update"):
                        vi = T.axis.spatial(12288, i_0 * 768 + i_1)
                        vk_0 = T.axis.reduce(8, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module6BQKVProj:
    @T.prim_func
    def main(
        A: T.Buffer((4096, 4096), "int32"),
        B: T.Buffer((4096,), "int32"),
        C: T.Buffer((4096,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((16, 4096), "int32")
        C_rf_global_local = T.alloc_buffer((16, 4096), "int32", scope="local")
        A_local = T.alloc_buffer((4096, 4096), "int32", scope="local")
        B_local = T.alloc_buffer((4096,), "int32", scope="local")
        for k_0 in T.thread_binding(16, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(128, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(2):
                        for i_3 in range(1):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(4096, i_0 * 32 + i_1 * 2 + i_2 + i_3)
                                vk_0 = T.axis.spatial(16, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(32):
                                for ax0_ax1_fused in range(8):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(4096, i_0 * 32 + i_1 * 2 + i_2)
                                        v1 = T.axis.spatial(
                                            4096, k_0 * 256 + k_1_0 * 8 + ax0_ax1_fused
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(8):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(4096, k_0 * 256 + k_1_0 * 8 + ax0_fused)
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(8):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(4096, i_0 * 32 + i_1 * 2 + i_2 + i_3)
                                        vk_0 = T.axis.spatial(16, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 8 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 256 + vk_1],
                                            B_local[vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 256 + vk_1]
                                            * B_local[vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 1):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(16, k_0 + ax0)
                                v1 = T.axis.spatial(4096, i_0 * 32 + i_1 * 2 + i_2 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(1):
            for i_1 in range(4096):
                with T.block("update_init"):
                    vi = T.axis.spatial(4096, i_0 * 4096 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(16):
                    with T.block("update_update"):
                        vi = T.axis.spatial(4096, i_0 * 4096 + i_1)
                        vk_0 = T.axis.reduce(16, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module13BQKVGen:
    @T.prim_func
    def main(
        A: T.Buffer((15360, 5120), "int32"),
        B: T.Buffer((5120,), "int32"),
        C: T.Buffer((15360,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((8, 15360), "int32")
        C_rf_global_local = T.alloc_buffer((8, 15360), "int32", scope="local")
        A_local = T.alloc_buffer((15360, 5120), "int32", scope="local")
        B_local = T.alloc_buffer((5120,), "int32", scope="local")
        for k_0 in T.thread_binding(8, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for i_0 in T.thread_binding(256, thread="blockIdx.y", annotations={"bank": T.int64(1)}):
                for i_1 in T.thread_binding(15, thread="threadIdx.x"):
                    for i_2 in range(2):
                        for i_3 in range(2):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(15360, i_0 * 60 + i_1 * 4 + i_2 * 2 + i_3)
                                vk_0 = T.axis.spatial(8, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(80):
                                for ax0_ax1_fused in range(8):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            15360, i_0 * 60 + i_1 * 4 + i_2 * 2 + i_3
                                        )
                                        v1 = T.axis.spatial(
                                            5120, k_0 * 640 + k_1_0 * 8 + ax0_ax1_fused
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(8):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(5120, k_0 * 640 + k_1_0 * 8 + ax0_fused)
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(8):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            15360, i_0 * 60 + i_1 * 4 + i_2 * 2 + i_3
                                        )
                                        vk_0 = T.axis.spatial(8, k_0)
                                        vk_1 = T.axis.reduce(640, k_1_0 * 8 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 640 + vk_1],
                                            B_local[vk_0 * 640 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 640 + vk_1]
                                            * B_local[vk_0 * 640 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 2):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(8, k_0 + ax0)
                                v1 = T.axis.spatial(15360, i_0 * 60 + i_1 * 4 + i_2 * 2 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(8):
            for i_1 in range(1920):
                with T.block("update_init"):
                    vi = T.axis.spatial(15360, i_0 * 1920 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[vi] = 0
                for k_0 in range(8):
                    with T.block("update_update"):
                        vi = T.axis.spatial(15360, i_0 * 1920 + i_1)
                        vk_0 = T.axis.reduce(8, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module13BQKVProj:
    @T.prim_func
    def main(
        A: T.Buffer((5120, 5120), "int32"),
        B: T.Buffer((5120,), "int32"),
        C: T.Buffer((5120,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((32, 5120), "int32")
        C_rf_global_local = T.alloc_buffer((32, 5120), "int32", scope="local")
        A_local = T.alloc_buffer((5120, 5120), "int32", scope="local")
        B_local = T.alloc_buffer((5120,), "int32", scope="local")
        for k_0 in T.thread_binding(32, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(64, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(10, thread="threadIdx.x"):
                    for i_2 in range(4):
                        for i_3 in range(2):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(5120, i_0 * 80 + i_1 * 8 + i_2 * 2 + i_3)
                                vk_0 = T.axis.spatial(32, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(20):
                                for ax0_ax1_fused in range(8):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            5120, i_0 * 80 + i_1 * 8 + i_2 * 2 + i_3
                                        )
                                        v1 = T.axis.spatial(
                                            5120, k_0 * 160 + k_1_0 * 8 + ax0_ax1_fused
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(8):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(5120, k_0 * 160 + k_1_0 * 8 + ax0_fused)
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(8):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            5120, i_0 * 80 + i_1 * 8 + i_2 * 2 + i_3
                                        )
                                        vk_0 = T.axis.spatial(32, k_0)
                                        vk_1 = T.axis.reduce(160, k_1_0 * 8 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 160 + vk_1],
                                            B_local[vk_0 * 160 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 160 + vk_1]
                                            * B_local[vk_0 * 160 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 2):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(32, k_0 + ax0)
                                v1 = T.axis.spatial(5120, i_0 * 80 + i_1 * 8 + i_2 * 2 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(1):
            for i_1 in range(5120):
                with T.block("update_init"):
                    vi = T.axis.spatial(5120, i_0 * 5120 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(32):
                    with T.block("update_update"):
                        vi = T.axis.spatial(5120, i_0 * 5120 + i_1)
                        vk_0 = T.axis.reduce(32, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module13BFC:
    @T.prim_func
    def main(
        A: T.Buffer((20480, 5120), "int32"),
        B: T.Buffer((5120,), "int32"),
        C: T.Buffer((20480,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((64, 20480), "int32")
        C_rf_global_local = T.alloc_buffer((64, 20480), "int32", scope="local")
        A_local = T.alloc_buffer((20480, 5120), "int32", scope="local")
        B_local = T.alloc_buffer((5120,), "int32", scope="local")
        for k_0 in T.thread_binding(64, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(32, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(4):
                        for i_3 in range(10):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(20480, i_0 * 640 + i_1 * 40 + i_2 * 10 + i_3)
                                vk_0 = T.axis.spatial(64, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(4):
                                for ax0_ax1_fused in range(20):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            20480, i_0 * 640 + i_1 * 40 + i_2 * 10 + i_3
                                        )
                                        v1 = T.axis.spatial(
                                            5120, k_0 * 80 + k_1_0 * 20 + ax0_ax1_fused
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(20):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(5120, k_0 * 80 + k_1_0 * 20 + ax0_fused)
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(20):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            20480, i_0 * 640 + i_1 * 40 + i_2 * 10 + i_3
                                        )
                                        vk_0 = T.axis.spatial(64, k_0)
                                        vk_1 = T.axis.reduce(80, k_1_0 * 20 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 80 + vk_1],
                                            B_local[vk_0 * 80 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 80 + vk_1]
                                            * B_local[vk_0 * 80 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 10):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(64, k_0 + ax0)
                                v1 = T.axis.spatial(20480, i_0 * 640 + i_1 * 40 + i_2 * 10 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(40):
            for i_1 in range(512):
                with T.block("update_init"):
                    vi = T.axis.spatial(20480, i_0 * 512 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(64):
                    with T.block("update_update"):
                        vi = T.axis.spatial(20480, i_0 * 512 + i_1)
                        vk_0 = T.axis.reduce(64, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module13BFCProj:
    @T.prim_func
    def main(
        A: T.Buffer((5120, 20480), "int32"),
        B: T.Buffer((20480,), "int32"),
        C: T.Buffer((5120,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((64, 5120), "int32")
        C_rf_global_local = T.alloc_buffer((64, 5120), "int32", scope="local")
        A_local = T.alloc_buffer((5120, 20480), "int32", scope="local")
        B_local = T.alloc_buffer((20480,), "int32", scope="local")
        for k_0 in T.thread_binding(64, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(32, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(10):
                        for i_3 in range(1):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(5120, i_0 * 160 + i_1 * 10 + i_2 + i_3)
                                vk_0 = T.axis.spatial(64, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_fused in range(20):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(5120, i_0 * 160 + i_1 * 10 + i_2)
                                        v1 = T.axis.spatial(
                                            20480,
                                            k_0 * 320 + k_1_0 * 20 + ax0_ax1_fused,
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(20):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            20480, k_0 * 320 + k_1_0 * 20 + ax0_fused
                                        )
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(20):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(5120, i_0 * 160 + i_1 * 10 + i_2 + i_3)
                                        vk_0 = T.axis.spatial(64, k_0)
                                        vk_1 = T.axis.reduce(320, k_1_0 * 20 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 320 + vk_1],
                                            B_local[vk_0 * 320 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 320 + vk_1]
                                            * B_local[vk_0 * 320 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 1):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(64, k_0 + ax0)
                                v1 = T.axis.spatial(5120, i_0 * 160 + i_1 * 10 + i_2 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(32):
            for i_1 in range(160):
                with T.block("update_init"):
                    vi = T.axis.spatial(5120, i_0 * 160 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(64):
                    with T.block("update_update"):
                        vi = T.axis.spatial(5120, i_0 * 160 + i_1)
                        vk_0 = T.axis.reduce(64, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module30BQKVGen:
    @T.prim_func
    def main(
        A: T.Buffer((21504, 7168), "int32"),
        B: T.Buffer((7168,), "int32"),
        C: T.Buffer((21504,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((64, 21504), "int32")
        C_rf_global_local = T.alloc_buffer((64, 21504), "int32", scope="local")
        A_local = T.alloc_buffer((21504, 7168), "int32", scope="local")
        B_local = T.alloc_buffer((7168,), "int32", scope="local")
        for k_0 in T.thread_binding(64, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(32, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(24, thread="threadIdx.x"):
                    for i_2 in range(1):
                        for i_3 in range(28):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(21504, i_0 * 672 + i_1 * 28 + i_2 * 28 + i_3)
                                vk_0 = T.axis.spatial(64, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(7):
                                for ax0_ax1_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(21504, i_0 * 672 + i_1 * 28 + i_3)
                                        v1 = T.axis.spatial(
                                            7168, k_0 * 112 + k_1_0 * 16 + ax0_ax1_fused
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            7168, k_0 * 112 + k_1_0 * 16 + ax0_fused
                                        )
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(16):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            21504, i_0 * 672 + i_1 * 28 + i_2 * 28 + i_3
                                        )
                                        vk_0 = T.axis.spatial(64, k_0)
                                        vk_1 = T.axis.reduce(112, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 112 + vk_1],
                                            B_local[vk_0 * 112 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 112 + vk_1]
                                            * B_local[vk_0 * 112 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 28):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(64, k_0 + ax0)
                                v1 = T.axis.spatial(21504, i_0 * 672 + i_1 * 28 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(32):
            for i_1 in range(672):
                with T.block("update_init"):
                    vi = T.axis.spatial(21504, i_0 * 672 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(64):
                    with T.block("update_update"):
                        vi = T.axis.spatial(21504, i_0 * 672 + i_1)
                        vk_0 = T.axis.reduce(64, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module30BFC:
    @T.prim_func
    def main(
        A: T.Buffer((28672, 7168), "int32"),
        B: T.Buffer((7168,), "int32"),
        C: T.Buffer((28672,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((64, 28672), "int32")
        C_rf_global_local = T.alloc_buffer((64, 28672), "int32", scope="local")
        A_local = T.alloc_buffer((28672, 7168), "int32", scope="local")
        B_local = T.alloc_buffer((7168,), "int32", scope="local")
        for k_0 in T.thread_binding(64, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(32, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(1):
                        for i_3 in range(56):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(28672, i_0 * 896 + i_1 * 56 + i_2 * 56 + i_3)
                                vk_0 = T.axis.spatial(64, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(7):
                                for ax0_ax1_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(28672, i_0 * 896 + i_1 * 56 + i_3)
                                        v1 = T.axis.spatial(
                                            7168, k_0 * 112 + k_1_0 * 16 + ax0_ax1_fused
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            7168, k_0 * 112 + k_1_0 * 16 + ax0_fused
                                        )
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(16):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            28672, i_0 * 896 + i_1 * 56 + i_2 * 56 + i_3
                                        )
                                        vk_0 = T.axis.spatial(64, k_0)
                                        vk_1 = T.axis.reduce(112, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 112 + vk_1],
                                            B_local[vk_0 * 112 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 112 + vk_1]
                                            * B_local[vk_0 * 112 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 56):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(64, k_0 + ax0)
                                v1 = T.axis.spatial(28672, i_0 * 896 + i_1 * 56 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(32):
            for i_1 in range(896):
                with T.block("update_init"):
                    vi = T.axis.spatial(28672, i_0 * 896 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(64):
                    with T.block("update_update"):
                        vi = T.axis.spatial(28672, i_0 * 896 + i_1)
                        vk_0 = T.axis.reduce(64, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module30BQKVProj:
    @T.prim_func
    def main(
        A: T.Buffer((7168, 7168), "int32"),
        B: T.Buffer((7168,), "int32"),
        C: T.Buffer((7168,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((32, 7168), "int32")
        C_rf_global_local = T.alloc_buffer((32, 7168), "int32", scope="local")
        A_local = T.alloc_buffer((7168, 7168), "int32", scope="local")
        B_local = T.alloc_buffer((7168,), "int32", scope="local")
        for k_0 in T.thread_binding(32, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(56, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(1):
                        for i_3 in range(8):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(7168, i_0 * 128 + i_1 * 8 + i_2 * 8 + i_3)
                                vk_0 = T.axis.spatial(32, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_fused in range(14):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(7168, i_0 * 128 + i_1 * 8 + i_3)
                                        v1 = T.axis.spatial(
                                            7168, k_0 * 224 + k_1_0 * 14 + ax0_ax1_fused
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(14):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            7168, k_0 * 224 + k_1_0 * 14 + ax0_fused
                                        )
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(14):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            7168, i_0 * 128 + i_1 * 8 + i_2 * 8 + i_3
                                        )
                                        vk_0 = T.axis.spatial(32, k_0)
                                        vk_1 = T.axis.reduce(224, k_1_0 * 14 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 224 + vk_1],
                                            B_local[vk_0 * 224 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 224 + vk_1]
                                            * B_local[vk_0 * 224 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 8):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(32, k_0 + ax0)
                                v1 = T.axis.spatial(7168, i_0 * 128 + i_1 * 8 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(1):
            for i_1 in range(7168):
                with T.block("update_init"):
                    vi = T.axis.spatial(7168, i_0 * 7168 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(32):
                    with T.block("update_update"):
                        vi = T.axis.spatial(7168, i_0 * 7168 + i_1)
                        vk_0 = T.axis.reduce(32, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module30BFCProj:
    @T.prim_func
    def main(
        A: T.Buffer((7168, 28672), "int32"),
        B: T.Buffer((28672,), "int32"),
        C: T.Buffer((7168,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((16, 7168), "int32")
        C_rf_global_local = T.alloc_buffer((16, 7168), "int32", scope="local")
        A_local = T.alloc_buffer((7168, 28672), "int32", scope="local")
        B_local = T.alloc_buffer((28672,), "int32", scope="local")
        for k_0 in T.thread_binding(16, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(128, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(14, thread="threadIdx.x"):
                    for i_2 in range(2):
                        for i_3 in range(2):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(7168, i_0 * 56 + i_1 * 4 + i_2 * 2 + i_3)
                                vk_0 = T.axis.spatial(16, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(128):
                                for ax0_ax1_fused in range(14):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            7168, i_0 * 56 + i_1 * 4 + i_2 * 2 + i_3
                                        )
                                        v1 = T.axis.spatial(
                                            28672,
                                            k_0 * 1792 + k_1_0 * 14 + ax0_ax1_fused,
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(14):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            28672, k_0 * 1792 + k_1_0 * 14 + ax0_fused
                                        )
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(14):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            7168, i_0 * 56 + i_1 * 4 + i_2 * 2 + i_3
                                        )
                                        vk_0 = T.axis.spatial(16, k_0)
                                        vk_1 = T.axis.reduce(1792, k_1_0 * 14 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 1792 + vk_1],
                                            B_local[vk_0 * 1792 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 1792 + vk_1]
                                            * B_local[vk_0 * 1792 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 2):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(16, k_0 + ax0)
                                v1 = T.axis.spatial(7168, i_0 * 56 + i_1 * 4 + i_2 * 2 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(1):
            for i_1 in range(7168):
                with T.block("update_init"):
                    vi = T.axis.spatial(7168, i_0 * 7168 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(16):
                    with T.block("update_update"):
                        vi = T.axis.spatial(7168, i_0 * 7168 + i_1)
                        vk_0 = T.axis.reduce(16, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module175BQKVProj:
    @T.prim_func
    def main(
        A: T.Buffer((12288, 12288), "int32"),
        B: T.Buffer((12288,), "int32"),
        C: T.Buffer((12288,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((16, 12288), "int32")
        C_rf_global_local = T.alloc_buffer((16, 12288), "int32", scope="local")
        A_local = T.alloc_buffer((12288, 12288), "int32", scope="local")
        B_local = T.alloc_buffer((12288,), "int32", scope="local")
        for k_0 in T.thread_binding(16, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(128, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(2):
                        for i_3 in range(3):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(12288, i_0 * 96 + i_1 * 6 + i_2 * 3 + i_3)
                                vk_0 = T.axis.spatial(16, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(32):
                                for ax0_ax1_fused in range(24):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            12288, i_0 * 96 + i_1 * 6 + i_2 * 3 + i_3
                                        )
                                        v1 = T.axis.spatial(
                                            12288,
                                            k_0 * 768 + k_1_0 * 24 + ax0_ax1_fused,
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(24):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            12288, k_0 * 768 + k_1_0 * 24 + ax0_fused
                                        )
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(24):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            12288, i_0 * 96 + i_1 * 6 + i_2 * 3 + i_3
                                        )
                                        vk_0 = T.axis.spatial(16, k_0)
                                        vk_1 = T.axis.reduce(768, k_1_0 * 24 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 768 + vk_1],
                                            B_local[vk_0 * 768 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 768 + vk_1]
                                            * B_local[vk_0 * 768 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 3):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(16, k_0 + ax0)
                                v1 = T.axis.spatial(12288, i_0 * 96 + i_1 * 6 + i_2 * 3 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(4):
            for i_1 in range(3072):
                with T.block("update_init"):
                    vi = T.axis.spatial(12288, i_0 * 3072 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(16):
                    with T.block("update_update"):
                        vi = T.axis.spatial(12288, i_0 * 3072 + i_1)
                        vk_0 = T.axis.reduce(16, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module175BFC:
    @T.prim_func
    def main(
        A: T.Buffer((49152, 12288), "int32"),
        B: T.Buffer((12288,), "int32"),
        C: T.Buffer((49152,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((32, 49152), "int32")
        C_rf_global_local = T.alloc_buffer((32, 49152), "int32", scope="local")
        A_local = T.alloc_buffer((49152, 12288), "int32", scope="local")
        B_local = T.alloc_buffer((12288,), "int32", scope="local")
        for k_0 in T.thread_binding(32, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for i_0 in T.thread_binding(64, thread="blockIdx.y", annotations={"bank": T.int64(1)}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(6):
                        for i_3 in range(8):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(49152, i_0 * 768 + i_1 * 48 + i_2 * 8 + i_3)
                                vk_0 = T.axis.spatial(32, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_fused in range(24):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            49152, i_0 * 768 + i_1 * 48 + i_2 * 8 + i_3
                                        )
                                        v1 = T.axis.spatial(
                                            12288,
                                            k_0 * 384 + k_1_0 * 24 + ax0_ax1_fused,
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(24):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            12288, k_0 * 384 + k_1_0 * 24 + ax0_fused
                                        )
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(24):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            49152, i_0 * 768 + i_1 * 48 + i_2 * 8 + i_3
                                        )
                                        vk_0 = T.axis.spatial(32, k_0)
                                        vk_1 = T.axis.reduce(384, k_1_0 * 24 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 384 + vk_1],
                                            B_local[vk_0 * 384 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 384 + vk_1]
                                            * B_local[vk_0 * 384 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 8):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(32, k_0 + ax0)
                                v1 = T.axis.spatial(49152, i_0 * 768 + i_1 * 48 + i_2 * 8 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(8):
            for i_1 in range(6144):
                with T.block("update_init"):
                    vi = T.axis.spatial(49152, i_0 * 6144 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[vi] = 0
                for k_0 in range(32):
                    with T.block("update_update"):
                        vi = T.axis.spatial(49152, i_0 * 6144 + i_1)
                        vk_0 = T.axis.reduce(32, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class Module175BFCProj:
    @T.prim_func
    def main(
        A: T.Buffer((12288, 49152), "int32"),
        B: T.Buffer((49152,), "int32"),
        C: T.Buffer((12288,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((64, 12288), "int32")
        C_rf_global_local = T.alloc_buffer((64, 12288), "int32", scope="local")
        A_local = T.alloc_buffer((12288, 49152), "int32", scope="local")
        B_local = T.alloc_buffer((49152,), "int32", scope="local")
        for k_0 in T.thread_binding(64, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(32, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(24, thread="threadIdx.x"):
                    for i_2 in range(1):
                        for i_3 in range(16):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(12288, i_0 * 384 + i_1 * 16 + i_2 * 16 + i_3)
                                vk_0 = T.axis.spatial(64, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(32):
                                for ax0_ax1_fused in range(24):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(12288, i_0 * 384 + i_1 * 16 + i_3)
                                        v1 = T.axis.spatial(
                                            49152,
                                            k_0 * 768 + k_1_0 * 24 + ax0_ax1_fused,
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(24):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            49152, k_0 * 768 + k_1_0 * 24 + ax0_fused
                                        )
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(24):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            12288, i_0 * 384 + i_1 * 16 + i_2 * 16 + i_3
                                        )
                                        vk_0 = T.axis.spatial(64, k_0)
                                        vk_1 = T.axis.reduce(768, k_1_0 * 24 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 768 + vk_1],
                                            B_local[vk_0 * 768 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 768 + vk_1]
                                            * B_local[vk_0 * 768 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 16):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(64, k_0 + ax0)
                                v1 = T.axis.spatial(12288, i_0 * 384 + i_1 * 16 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(12):
            for i_1 in range(1024):
                with T.block("update_init"):
                    vi = T.axis.spatial(12288, i_0 * 1024 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[vi] = 0
                for k_0 in range(64):
                    with T.block("update_update"):
                        vi = T.axis.spatial(12288, i_0 * 1024 + i_1)
                        vk_0 = T.axis.reduce(64, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


@I.ir_module
class MHAModule_128_480_256:
    @T.prim_func
    def main(
        A: T.Buffer((128, 480, 256), "int32"),
        B: T.Buffer((128, 256), "int32"),
        C: T.Buffer((128, 480), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((128, 2, 480), "int32")
        C_rf_global_local = T.alloc_buffer((128, 2, 480), "int32", scope="local")
        A_local = T.alloc_buffer((128, 480, 256), "int32", scope="local")
        B_local = T.alloc_buffer((128, 256), "int32", scope="local")
        for k_0 in T.thread_binding(2, thread="blockIdx.x", annotations={"bank": 1}):
            for n_0_i_0_fused in T.thread_binding(16, thread="blockIdx.y", annotations={"bank": 1}):
                for n_1_i_1_fused in T.thread_binding(
                    8, thread="blockIdx.z", annotations={"bank": 1}
                ):
                    for n_2_i_2_fused in T.thread_binding(8, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 60):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(128, n_0_i_0_fused * 8 + n_1_i_1_fused + n_3)
                                v_i = T.axis.spatial(480, n_2_i_2_fused * 60 + i_3)
                                vk_0 = T.axis.spatial(2, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(4):
                                for ax0_ax1_ax2_fused in range(32):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(128, n_0_i_0_fused * 8 + n_1_i_1_fused)
                                        v1 = T.axis.spatial(480, n_2_i_2_fused * 60 + i_3)
                                        v2 = T.axis.spatial(
                                            256,
                                            k_0 * 128 + k_1_0 * 32 + ax0_ax1_ax2_fused,
                                        )
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(32):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(128, n_0_i_0_fused * 8 + n_1_i_1_fused)
                                        v1 = T.axis.spatial(
                                            256, k_0 * 128 + k_1_0 * 32 + ax0_ax1_fused
                                        )
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(32):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            128, n_0_i_0_fused * 8 + n_1_i_1_fused + n_3
                                        )
                                        v_i = T.axis.spatial(480, n_2_i_2_fused * 60 + i_3)
                                        vk_0 = T.axis.spatial(2, k_0)
                                        vk_1 = T.axis.reduce(128, k_1_0 * 32 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 128 + vk_1],
                                            B_local[v_n, vk_0 * 128 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 128 + vk_1]
                                            * B_local[v_n, vk_0 * 128 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 60):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(128, n_0_i_0_fused * 8 + n_1_i_1_fused + ax0)
                                v1 = T.axis.spatial(2, k_0 + ax1)
                                v2 = T.axis.spatial(480, n_2_i_2_fused * 60 + ax2)
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_0 in T.parallel(2):
            for n_1, i in T.grid(64, 480):
                with T.block("C_init"):
                    v_n = T.axis.spatial(128, n_0 * 64 + n_1)
                    v_i = T.axis.spatial(480, i)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(2):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(128, n_0 * 64 + n_1)
                        v_i, vk_0 = T.axis.remap("SR", [i, k_0])
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class MHAModule_256_64_256:
    @T.prim_func
    def main(
        A: T.Buffer((256, 64, 256), "int32"),
        B: T.Buffer((256, 256), "int32"),
        C: T.Buffer((256, 64), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((256, 4, 64), "int32")
        C_rf_global_local = T.alloc_buffer((256, 4, 64), "int32", scope="local")
        A_local = T.alloc_buffer((256, 64, 256), "int32", scope="local")
        B_local = T.alloc_buffer((256, 256), "int32", scope="local")
        for k_0 in T.thread_binding(4, thread="blockIdx.x", annotations={"bank": 1}):
            for n_0_i_0_fused in T.thread_binding(4, thread="blockIdx.y", annotations={"bank": 1}):
                for n_1_i_1_fused in T.thread_binding(
                    64, thread="blockIdx.z", annotations={"bank": 1}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 4):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(256, n_0_i_0_fused * 64 + n_1_i_1_fused + n_3)
                                v_i = T.axis.spatial(64, n_2_i_2_fused * 4 + i_3)
                                vk_0 = T.axis.spatial(4, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(8):
                                for ax0_ax1_ax2_fused in range(8):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(256, n_0_i_0_fused * 64 + n_1_i_1_fused)
                                        v1 = T.axis.spatial(64, n_2_i_2_fused * 4 + i_3)
                                        v2 = T.axis.spatial(
                                            256,
                                            k_0 * 64 + k_1_0 * 8 + ax0_ax1_ax2_fused,
                                        )
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(8):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(256, n_0_i_0_fused * 64 + n_1_i_1_fused)
                                        v1 = T.axis.spatial(
                                            256, k_0 * 64 + k_1_0 * 8 + ax0_ax1_fused
                                        )
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(8):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            256,
                                            n_0_i_0_fused * 64 + n_1_i_1_fused + n_3,
                                        )
                                        v_i = T.axis.spatial(64, n_2_i_2_fused * 4 + i_3)
                                        vk_0 = T.axis.spatial(4, k_0)
                                        vk_1 = T.axis.reduce(64, k_1_0 * 8 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 64 + vk_1],
                                            B_local[v_n, vk_0 * 64 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 64 + vk_1]
                                            * B_local[v_n, vk_0 * 64 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 4):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(256, n_0_i_0_fused * 64 + n_1_i_1_fused + ax0)
                                v1 = T.axis.spatial(4, k_0 + ax1)
                                v2 = T.axis.spatial(64, n_2_i_2_fused * 4 + ax2)
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_0 in T.parallel(1):
            for n_1, i in T.grid(256, 64):
                with T.block("C_init"):
                    v_n = T.axis.spatial(256, n_0 * 256 + n_1)
                    v_i = T.axis.spatial(64, i)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(4):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(256, n_0 * 256 + n_1)
                        v_i, vk_0 = T.axis.remap("SR", [i, k_0])
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                                "meta_schedule.random_compute_producer": 1,
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class Module175BQKVGen:
    @T.prim_func
    def main(
        A: T.Buffer((36864, 12288), "int32"),
        B: T.Buffer((12288,), "int32"),
        C: T.Buffer((36864,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((32, 36864), "int32")
        C_rf_global_local = T.alloc_buffer((32, 36864), "int32", scope="local")
        A_local = T.alloc_buffer((36864, 12288), "int32", scope="local")
        B_local = T.alloc_buffer((12288,), "int32", scope="local")
        for k_0 in T.thread_binding(32, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for i_0 in T.thread_binding(64, thread="blockIdx.y", annotations={"bank": T.int64(1)}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(1):
                        for i_3 in range(36):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(36864, i_0 * 576 + i_1 * 36 + i_2 * 36 + i_3)
                                vk_0 = T.axis.spatial(32, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[vk_0, vi])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[vk_0, vi] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_fused in range(24):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(36864, i_0 * 576 + i_1 * 36 + i_3)
                                        v1 = T.axis.spatial(
                                            12288,
                                            k_0 * 384 + k_1_0 * 24 + ax0_ax1_fused,
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(24):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            12288, k_0 * 384 + k_1_0 * 24 + ax0_fused
                                        )
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(24):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            36864, i_0 * 576 + i_1 * 36 + i_2 * 36 + i_3
                                        )
                                        vk_0 = T.axis.spatial(32, k_0)
                                        vk_1 = T.axis.reduce(384, k_1_0 * 24 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[vk_0, vi],
                                            A_local[vi, vk_0 * 384 + vk_1],
                                            B_local[vk_0 * 384 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[vk_0, vi])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[vk_0, vi] = (
                                            C_rf_global_local[vk_0, vi]
                                            + A_local[vi, vk_0 * 384 + vk_1]
                                            * B_local[vk_0 * 384 + vk_1]
                                        )
                        for ax0, ax1 in T.grid(1, 36):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(32, k_0 + ax0)
                                v1 = T.axis.spatial(36864, i_0 * 576 + i_1 * 36 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(64):
            for i_1 in range(576):
                with T.block("update_init"):
                    vi = T.axis.spatial(36864, i_0 * 576 + i_1)
                    T.reads()
                    T.writes(C[vi])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[vi] = 0
                for k_0 in range(32):
                    with T.block("update_update"):
                        vi = T.axis.spatial(36864, i_0 * 576 + i_1)
                        vk_0 = T.axis.reduce(32, k_0)
                        T.reads(C[vi], C_rf_global[vk_0, vi])
                        T.writes(C[vi])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[vi] = C[vi] + C_rf_global[vk_0, vi]


m = 28672
k = 7168

tuple_mv = {
    # "6B_qkvgen": (Module6BQKVGen, 12288, 4096),
    # "6B_qkvproj": (Module6BQKVProj, 4096, 4096),
    # "6B_fc": (Module6BFC, 16384, 4096),
    # "6B_fcproj": (Module6BFCProj, 4096, 16384),
    # "13B_qkvgen": (Module13BQKVGen, 15360, 5120),
    # "13B_qkvproj": (Module13BQKVProj, 5120, 5120),
    # "13B_fc": (Module13BFC, 20480, 5120),
    # "13B_fcproj": (Module13BFCProj, 5120, 20480),
    # "30B_qkvgen": (Module30BQKVGen, 21504, 7168),
    # "30B_qkvproj": (Module30BQKVProj, 7168, 7168),
    # "30B_fc": (Module30BFC, 28672, 7168),
    # "30B_fcproj": (Module30BFCProj, 7168, 28672),
    "175B_qkvgen": (Module175BQKVGen, 36864, 12288),  # 21775 -> 아놔 진짜 다시 돌려야함
    # "175B_qkvproj": (Module175BQKVProj, 12288, 12288),  # 7278
    # "175B_fc": (Module175BFC, 49152, 12288),  # 27918
    # "175B_fcproj": (Module175BFCProj, 12288, 49152),  # 27847
}

tuple_mha = {
    # "sample": (MHAModule_128_480_256, 128, 480, 256),
    "sample2": (MHAModule_256_64_256, 256, 64, 256)
}

dev = tvm.device("upmem", 0)


def lower_to_div(mod):
    l = tvm.lower(mod)
    target = tvm.target.Target(target="upmem", host="llvm")
    m = Target.canon_target_map_and_host({target: l}, "llvm")[0][target]
    m = BindTarget(target)(m)
    m = VerifyMemory()(m)
    m = AnnotateEntryFunc()(m)
    m = ThreadSync("global")(m)
    m = ThreadSync("shared")(m)
    m = ThreadSync("shared.dyn")(m)
    m = MergeDynamicSharedMemoryAllocations()(m)
    m = ThreadSync("warp")(m)
    m = InferFragment()(m)
    m = LowerThreadAllreduce()(m)
    m = AnnotateDeviceRegions()(m)
    m = ExtractPimTransferSchedule()(m)
    m = SplitHostDevice()(m)
    m = SplitPimTransfer()(m)
    return m


for conf, (cl, m, k) in tuple_mv.items():
    func = tvm.build(cl, target="upmem", name="gemv")

    dev.load_function(func)
    ha = host_array((m, k), "int32")
    hb = host_array((k), "int32")
    # print("load", conf)
    a = tvm.nd.array(ha, device=dev, symbol="A")
    # print("INJECTED A")
    b = tvm.nd.array(hb, device=dev, symbol="B")
    # print("INJECTED B")
    c = tvm.nd.array(
        np.zeros((m,)).astype("int32"),
        device=dev,
        symbol="C",
    )
    # print("LOADED B&C")
    timestamp = tvm._ffi.get_global_func("device_api.upmem.timestamp")
    elapsed_time = tvm._ffi.get_global_func("device_api.upmem.elapsed_time")
    func(a, b, c)
    # print("LAUNCHED")

    bs, ks, total = [], [], []
    for i in range(100):
        s = time.time()
        timestamp("start")
        func(a, b, c)
        timestamp("end")
        e = time.time()

        before_kernel_time = elapsed_time("before_kernel") / 1e6
        kernel_time = elapsed_time("kernel") / 1e6
        after_kernel_time = elapsed_time("d2h") / 1e6

        bs.append(before_kernel_time)
        ks.append(kernel_time)
        total.append((e - s) * 1000)

    print(f"{conf}\t{np.mean(ks)}\t{np.mean(total)}")
    dev.free()
