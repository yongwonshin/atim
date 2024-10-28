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
class params175B_batch16_token128:
    @T.prim_func
    def main(
        A: T.Buffer((768, 128, 256), "int32"),
        B: T.Buffer((768, 256), "int32"),
        C: T.Buffer((768, 128), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((768, 1, 128), "int32")
        C_rf_global_local = T.alloc_buffer((768, 1, 128), "int32", scope="local")
        A_local = T.alloc_buffer((768, 128, 256), "int32", scope="local")
        B_local = T.alloc_buffer((768, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                96, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    16, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 4):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    768, n_0_i_0_fused * 8 + n_1_i_1_fused // 2 + n_3
                                )
                                v_i = T.axis.spatial(
                                    128, n_1_i_1_fused % 2 * 64 + n_2_i_2_fused * 4 + i_3
                                )
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_ax2_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            768, n_0_i_0_fused * 8 + n_1_i_1_fused // 2
                                        )
                                        v1 = T.axis.spatial(
                                            128, n_1_i_1_fused % 2 * 64 + n_2_i_2_fused * 4 + i_3
                                        )
                                        v2 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            768, n_0_i_0_fused * 8 + n_1_i_1_fused // 2
                                        )
                                        v1 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(16):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            768, n_0_i_0_fused * 8 + n_1_i_1_fused // 2 + n_3
                                        )
                                        v_i = T.axis.spatial(
                                            128, n_1_i_1_fused % 2 * 64 + n_2_i_2_fused * 4 + i_3
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 4):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    768, n_0_i_0_fused * 8 + n_1_i_1_fused // 2 + ax0
                                )
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    128, n_1_i_1_fused % 2 * 64 + n_2_i_2_fused * 4 + ax2
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(48):
            for n_i_fused_1 in range(2048):
                with T.block("C_init"):
                    v_n = T.axis.spatial(768, (n_i_fused_0 * 2048 + n_i_fused_1) // 128)
                    v_i = T.axis.spatial(128, (n_i_fused_0 * 2048 + n_i_fused_1) % 128)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(768, (n_i_fused_0 * 2048 + n_i_fused_1) // 128)
                        v_i = T.axis.spatial(128, (n_i_fused_0 * 2048 + n_i_fused_1) % 128)
                        vk_0 = T.axis.reduce(1, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params175B_batch16_token256:
    @T.prim_func
    def main(
        A: T.Buffer((768, 256, 256), "int32"),
        B: T.Buffer((768, 256), "int32"),
        C: T.Buffer((768, 256), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((768, 1, 256), "int32")
        C_rf_global_local = T.alloc_buffer((768, 1, 256), "int32", scope="local")
        A_local = T.alloc_buffer((768, 256, 256), "int32", scope="local")
        B_local = T.alloc_buffer((768, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                3, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    512, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 8):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    768, n_0_i_0_fused * 256 + n_1_i_1_fused // 2 + n_3
                                )
                                v_i = T.axis.spatial(
                                    256, n_1_i_1_fused % 2 * 128 + n_2_i_2_fused * 8 + i_3
                                )
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_ax2_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            768, n_0_i_0_fused * 256 + n_1_i_1_fused // 2
                                        )
                                        v1 = T.axis.spatial(
                                            256, n_1_i_1_fused % 2 * 128 + n_2_i_2_fused * 8 + i_3
                                        )
                                        v2 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            768, n_0_i_0_fused * 256 + n_1_i_1_fused // 2
                                        )
                                        v1 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(16):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            768, n_0_i_0_fused * 256 + n_1_i_1_fused // 2 + n_3
                                        )
                                        v_i = T.axis.spatial(
                                            256, n_1_i_1_fused % 2 * 128 + n_2_i_2_fused * 8 + i_3
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 8):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    768, n_0_i_0_fused * 256 + n_1_i_1_fused // 2 + ax0
                                )
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    256, n_1_i_1_fused % 2 * 128 + n_2_i_2_fused * 8 + ax2
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(3):
            for n_i_fused_1 in range(65536):
                with T.block("C_init"):
                    v_n = T.axis.spatial(768, (n_i_fused_0 * 65536 + n_i_fused_1) // 256)
                    v_i = T.axis.spatial(256, (n_i_fused_0 * 65536 + n_i_fused_1) % 256)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(768, (n_i_fused_0 * 65536 + n_i_fused_1) // 256)
                        v_i = T.axis.spatial(256, (n_i_fused_0 * 65536 + n_i_fused_1) % 256)
                        vk_0 = T.axis.reduce(1, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params175B_batch16_token512:
    @T.prim_func
    def main(
        A: T.Buffer((768, 512, 256), "int32"),
        B: T.Buffer((768, 256), "int32"),
        C: T.Buffer((768, 512), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((768, 1, 512), "int32")
        C_rf_global_local = T.alloc_buffer((768, 1, 512), "int32", scope="local")
        A_local = T.alloc_buffer((768, 512, 256), "int32", scope="local")
        B_local = T.alloc_buffer((768, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                2, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    1024, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(6, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 32):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    768,
                                    n_0_i_0_fused * 384
                                    + n_1_i_1_fused // 16 * 6
                                    + n_2_i_2_fused
                                    + n_3,
                                )
                                v_i = T.axis.spatial(512, n_1_i_1_fused % 16 * 32 + i_3)
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_ax2_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            768,
                                            n_0_i_0_fused * 384
                                            + n_1_i_1_fused // 16 * 6
                                            + n_2_i_2_fused,
                                        )
                                        v1 = T.axis.spatial(512, n_1_i_1_fused % 16 * 32 + i_3)
                                        v2 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            768,
                                            n_0_i_0_fused * 384
                                            + n_1_i_1_fused // 16 * 6
                                            + n_2_i_2_fused,
                                        )
                                        v1 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(16):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            768,
                                            n_0_i_0_fused * 384
                                            + n_1_i_1_fused // 16 * 6
                                            + n_2_i_2_fused
                                            + n_3,
                                        )
                                        v_i = T.axis.spatial(512, n_1_i_1_fused % 16 * 32 + i_3)
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 32):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    768,
                                    n_0_i_0_fused * 384
                                    + n_1_i_1_fused // 16 * 6
                                    + n_2_i_2_fused
                                    + ax0,
                                )
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(512, n_1_i_1_fused % 16 * 32 + ax2)
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(192):
            for n_i_fused_1 in range(2048):
                with T.block("C_init"):
                    v_n = T.axis.spatial(768, (n_i_fused_0 * 2048 + n_i_fused_1) // 512)
                    v_i = T.axis.spatial(512, (n_i_fused_0 * 2048 + n_i_fused_1) % 512)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(768, (n_i_fused_0 * 2048 + n_i_fused_1) // 512)
                        v_i = T.axis.spatial(512, (n_i_fused_0 * 2048 + n_i_fused_1) % 512)
                        vk_0 = T.axis.reduce(1, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params175B_batch16_token64:
    @T.prim_func
    def main(
        A: T.Buffer((768, 64, 256), "int32"),
        B: T.Buffer((768, 256), "int32"),
        C: T.Buffer((768, 64), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((768, 1, 64), "int32")
        C_rf_global_local = T.alloc_buffer((768, 1, 64), "int32", scope="local")
        A_local = T.alloc_buffer((768, 64, 256), "int32", scope="local")
        B_local = T.alloc_buffer((768, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                1, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    1536, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 2):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(768, n_1_i_1_fused // 2 + n_3)
                                v_i = T.axis.spatial(
                                    64, n_1_i_1_fused % 2 * 32 + n_2_i_2_fused * 2 + i_3
                                )
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_ax2_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(768, n_1_i_1_fused // 2)
                                        v1 = T.axis.spatial(
                                            64, n_1_i_1_fused % 2 * 32 + n_2_i_2_fused * 2 + i_3
                                        )
                                        v2 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(768, n_1_i_1_fused // 2)
                                        v1 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(16):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(768, n_1_i_1_fused // 2 + n_3)
                                        v_i = T.axis.spatial(
                                            64, n_1_i_1_fused % 2 * 32 + n_2_i_2_fused * 2 + i_3
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 2):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(768, n_1_i_1_fused // 2 + ax0)
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    64, n_1_i_1_fused % 2 * 32 + n_2_i_2_fused * 2 + ax2
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(8):
            for n_i_fused_1 in range(6144):
                with T.block("C_init"):
                    v_n = T.axis.spatial(768, (n_i_fused_0 * 6144 + n_i_fused_1) // 64)
                    v_i = T.axis.spatial(64, (n_i_fused_0 * 6144 + n_i_fused_1) % 64)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(768, (n_i_fused_0 * 6144 + n_i_fused_1) // 64)
                        v_i = T.axis.spatial(64, (n_i_fused_0 * 6144 + n_i_fused_1) % 64)
                        vk_0 = T.axis.reduce(1, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params175B_batch1_token128:
    @T.prim_func
    def main(
        A: T.Buffer((48, 128, 256), "int32"),
        B: T.Buffer((48, 256), "int32"),
        C: T.Buffer((48, 128), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((48, 2, 128), "int32")
        C_rf_global_local = T.alloc_buffer((48, 2, 128), "int32", scope="local")
        A_local = T.alloc_buffer((48, 128, 256), "int32", scope="local")
        B_local = T.alloc_buffer((48, 256), "int32", scope="local")
        for k_0 in T.thread_binding(2, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                48, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    4, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 2):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    48, n_0_i_0_fused // 2 * 2 + n_1_i_1_fused // 2 + n_3
                                )
                                v_i = T.axis.spatial(
                                    128,
                                    n_0_i_0_fused % 2 * 64
                                    + n_1_i_1_fused % 2 * 32
                                    + n_2_i_2_fused * 2
                                    + i_3,
                                )
                                vk_0 = T.axis.spatial(2, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(2):
                                for ax0_ax1_ax2_fused in range(64):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            48, n_0_i_0_fused // 2 * 2 + n_1_i_1_fused // 2
                                        )
                                        v1 = T.axis.spatial(
                                            128,
                                            n_0_i_0_fused % 2 * 64
                                            + n_1_i_1_fused % 2 * 32
                                            + n_2_i_2_fused * 2
                                            + i_3,
                                        )
                                        v2 = T.axis.spatial(
                                            256, k_0 * 128 + k_1_0 * 64 + ax0_ax1_ax2_fused
                                        )
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(64):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            48, n_0_i_0_fused // 2 * 2 + n_1_i_1_fused // 2
                                        )
                                        v1 = T.axis.spatial(
                                            256, k_0 * 128 + k_1_0 * 64 + ax0_ax1_fused
                                        )
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(64):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            48, n_0_i_0_fused // 2 * 2 + n_1_i_1_fused // 2 + n_3
                                        )
                                        v_i = T.axis.spatial(
                                            128,
                                            n_0_i_0_fused % 2 * 64
                                            + n_1_i_1_fused % 2 * 32
                                            + n_2_i_2_fused * 2
                                            + i_3,
                                        )
                                        vk_0 = T.axis.spatial(2, k_0)
                                        vk_1 = T.axis.reduce(128, k_1_0 * 64 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 128 + vk_1],
                                            B_local[v_n, vk_0 * 128 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 128 + vk_1]
                                            * B_local[v_n, vk_0 * 128 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 2):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    48, n_0_i_0_fused // 2 * 2 + n_1_i_1_fused // 2 + ax0
                                )
                                v1 = T.axis.spatial(2, k_0 + ax1)
                                v2 = T.axis.spatial(
                                    128,
                                    n_0_i_0_fused % 2 * 64
                                    + n_1_i_1_fused % 2 * 32
                                    + n_2_i_2_fused * 2
                                    + ax2,
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(24):
            for n_i_fused_1 in range(256):
                with T.block("C_init"):
                    v_n = T.axis.spatial(48, (n_i_fused_0 * 256 + n_i_fused_1) // 128)
                    v_i = T.axis.spatial(128, (n_i_fused_0 * 256 + n_i_fused_1) % 128)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(2):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(48, (n_i_fused_0 * 256 + n_i_fused_1) // 128)
                        v_i = T.axis.spatial(128, (n_i_fused_0 * 256 + n_i_fused_1) % 128)
                        vk_0 = T.axis.reduce(2, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params175B_batch1_token256:
    @T.prim_func
    def main(
        A: T.Buffer((48, 256, 256), "int32"),
        B: T.Buffer((48, 256), "int32"),
        C: T.Buffer((48, 256), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((48, 1, 256), "int32")
        C_rf_global_local = T.alloc_buffer((48, 1, 256), "int32", scope="local")
        A_local = T.alloc_buffer((48, 256, 256), "int32", scope="local")
        B_local = T.alloc_buffer((48, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                16, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    24, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 2):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    48, n_0_i_0_fused * 3 + n_1_i_1_fused // 8 + n_3
                                )
                                v_i = T.axis.spatial(
                                    256, n_1_i_1_fused % 8 * 32 + n_2_i_2_fused * 2 + i_3
                                )
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_ax2_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            48, n_0_i_0_fused * 3 + n_1_i_1_fused // 8
                                        )
                                        v1 = T.axis.spatial(
                                            256, n_1_i_1_fused % 8 * 32 + n_2_i_2_fused * 2 + i_3
                                        )
                                        v2 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            48, n_0_i_0_fused * 3 + n_1_i_1_fused // 8
                                        )
                                        v1 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(16):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            48, n_0_i_0_fused * 3 + n_1_i_1_fused // 8 + n_3
                                        )
                                        v_i = T.axis.spatial(
                                            256, n_1_i_1_fused % 8 * 32 + n_2_i_2_fused * 2 + i_3
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 2):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    48, n_0_i_0_fused * 3 + n_1_i_1_fused // 8 + ax0
                                )
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    256, n_1_i_1_fused % 8 * 32 + n_2_i_2_fused * 2 + ax2
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(1):
            for n_i_fused_1 in range(12288):
                with T.block("C_init"):
                    v_n = T.axis.spatial(48, (n_i_fused_0 * 12288 + n_i_fused_1) // 256)
                    v_i = T.axis.spatial(256, (n_i_fused_0 * 12288 + n_i_fused_1) % 256)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(48, (n_i_fused_0 * 12288 + n_i_fused_1) // 256)
                        v_i = T.axis.spatial(256, (n_i_fused_0 * 12288 + n_i_fused_1) % 256)
                        vk_0 = T.axis.reduce(1, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params175B_batch1_token512:
    @T.prim_func
    def main(
        A: T.Buffer((48, 512, 256), "int32"),
        B: T.Buffer((48, 256), "int32"),
        C: T.Buffer((48, 512), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((48, 1, 512), "int32")
        C_rf_global_local = T.alloc_buffer((48, 1, 512), "int32", scope="local")
        A_local = T.alloc_buffer((48, 512, 256), "int32", scope="local")
        B_local = T.alloc_buffer((48, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": 1}):
            for n_0_i_0_fused in T.thread_binding(4, thread="blockIdx.y", annotations={"bank": 1}):
                for n_1_i_1_fused in T.thread_binding(
                    96, thread="blockIdx.z", annotations={"bank": 1}
                ):
                    for n_2_i_2_fused in T.thread_binding(8, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 8):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    48, n_0_i_0_fused // 2 * 24 + n_1_i_1_fused // 4 + n_3
                                )
                                v_i = T.axis.spatial(
                                    512,
                                    n_0_i_0_fused % 2 * 256
                                    + n_1_i_1_fused % 4 * 64
                                    + n_2_i_2_fused * 8
                                    + i_3,
                                )
                                vk_0 = T.axis.spatial(1, k_0)
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
                                for ax0_ax1_ax2_fused in range(64):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            48, n_0_i_0_fused // 2 * 24 + n_1_i_1_fused // 4
                                        )
                                        v1 = T.axis.spatial(
                                            512,
                                            n_0_i_0_fused % 2 * 256
                                            + n_1_i_1_fused % 4 * 64
                                            + n_2_i_2_fused * 8
                                            + i_3,
                                        )
                                        v2 = T.axis.spatial(256, k_1_0 * 64 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(64):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            48, n_0_i_0_fused // 2 * 24 + n_1_i_1_fused // 4
                                        )
                                        v1 = T.axis.spatial(256, k_1_0 * 64 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(64):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            48, n_0_i_0_fused // 2 * 24 + n_1_i_1_fused // 4 + n_3
                                        )
                                        v_i = T.axis.spatial(
                                            512,
                                            n_0_i_0_fused % 2 * 256
                                            + n_1_i_1_fused % 4 * 64
                                            + n_2_i_2_fused * 8
                                            + i_3,
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 64 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
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
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 8):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    48, n_0_i_0_fused // 2 * 24 + n_1_i_1_fused // 4 + ax0
                                )
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    512,
                                    n_0_i_0_fused % 2 * 256
                                    + n_1_i_1_fused % 4 * 64
                                    + n_2_i_2_fused * 8
                                    + ax2,
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(12):
            for n_i_fused_1 in range(2048):
                with T.block("C_init"):
                    v_n = T.axis.spatial(48, (n_i_fused_0 * 2048 + n_i_fused_1) // 512)
                    v_i = T.axis.spatial(512, (n_i_fused_0 * 2048 + n_i_fused_1) % 512)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(48, (n_i_fused_0 * 2048 + n_i_fused_1) // 512)
                        v_i = T.axis.spatial(512, (n_i_fused_0 * 2048 + n_i_fused_1) % 512)
                        vk_0 = T.axis.reduce(1, k_0)
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
class params175B_batch1_token64:
    @T.prim_func
    def main(
        A: T.Buffer((48, 64, 256), "int32"),
        B: T.Buffer((48, 256), "int32"),
        C: T.Buffer((48, 64), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((48, 1, 64), "int32")
        C_rf_global_local = T.alloc_buffer((48, 1, 64), "int32", scope="local")
        A_local = T.alloc_buffer((48, 64, 256), "int32", scope="local")
        B_local = T.alloc_buffer((48, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                3, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    64, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 1):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    48, n_0_i_0_fused * 16 + n_1_i_1_fused // 4 + n_3
                                )
                                v_i = T.axis.spatial(
                                    64, n_1_i_1_fused % 4 * 16 + n_2_i_2_fused + i_3
                                )
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_ax2_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            48, n_0_i_0_fused * 16 + n_1_i_1_fused // 4
                                        )
                                        v1 = T.axis.spatial(
                                            64, n_1_i_1_fused % 4 * 16 + n_2_i_2_fused
                                        )
                                        v2 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            48, n_0_i_0_fused * 16 + n_1_i_1_fused // 4
                                        )
                                        v1 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(16):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            48, n_0_i_0_fused * 16 + n_1_i_1_fused // 4 + n_3
                                        )
                                        v_i = T.axis.spatial(
                                            64, n_1_i_1_fused % 4 * 16 + n_2_i_2_fused + i_3
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 1):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    48, n_0_i_0_fused * 16 + n_1_i_1_fused // 4 + ax0
                                )
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    64, n_1_i_1_fused % 4 * 16 + n_2_i_2_fused + ax2
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(4):
            for n_i_fused_1 in range(768):
                with T.block("C_init"):
                    v_n = T.axis.spatial(48, (n_i_fused_0 * 768 + n_i_fused_1) // 64)
                    v_i = T.axis.spatial(64, (n_i_fused_0 * 768 + n_i_fused_1) % 64)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(48, (n_i_fused_0 * 768 + n_i_fused_1) // 64)
                        v_i = T.axis.spatial(64, (n_i_fused_0 * 768 + n_i_fused_1) % 64)
                        vk_0 = T.axis.reduce(1, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params6B_batch16_token128:
    @T.prim_func
    def main(
        A: T.Buffer((256, 128, 256), "int32"),
        B: T.Buffer((256, 256), "int32"),
        C: T.Buffer((256, 128), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((256, 1, 128), "int32")
        C_rf_global_local = T.alloc_buffer((256, 1, 128), "int32", scope="local")
        A_local = T.alloc_buffer((256, 128, 256), "int32", scope="local")
        B_local = T.alloc_buffer((256, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                64, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    16, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 2):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    256, n_0_i_0_fused // 2 * 8 + n_1_i_1_fused // 2 + n_3
                                )
                                v_i = T.axis.spatial(
                                    128,
                                    n_0_i_0_fused % 2 * 64
                                    + n_1_i_1_fused % 2 * 32
                                    + n_2_i_2_fused * 2
                                    + i_3,
                                )
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_ax2_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            256, n_0_i_0_fused // 2 * 8 + n_1_i_1_fused // 2
                                        )
                                        v1 = T.axis.spatial(
                                            128,
                                            n_0_i_0_fused % 2 * 64
                                            + n_1_i_1_fused % 2 * 32
                                            + n_2_i_2_fused * 2
                                            + i_3,
                                        )
                                        v2 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            256, n_0_i_0_fused // 2 * 8 + n_1_i_1_fused // 2
                                        )
                                        v1 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(16):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            256, n_0_i_0_fused // 2 * 8 + n_1_i_1_fused // 2 + n_3
                                        )
                                        v_i = T.axis.spatial(
                                            128,
                                            n_0_i_0_fused % 2 * 64
                                            + n_1_i_1_fused % 2 * 32
                                            + n_2_i_2_fused * 2
                                            + i_3,
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 2):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    256, n_0_i_0_fused // 2 * 8 + n_1_i_1_fused // 2 + ax0
                                )
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    128,
                                    n_0_i_0_fused % 2 * 64
                                    + n_1_i_1_fused % 2 * 32
                                    + n_2_i_2_fused * 2
                                    + ax2,
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(4):
            for n_i_fused_1 in range(8192):
                with T.block("C_init"):
                    v_n = T.axis.spatial(256, (n_i_fused_0 * 8192 + n_i_fused_1) // 128)
                    v_i = T.axis.spatial(128, (n_i_fused_0 * 8192 + n_i_fused_1) % 128)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(256, (n_i_fused_0 * 8192 + n_i_fused_1) // 128)
                        v_i = T.axis.spatial(128, (n_i_fused_0 * 8192 + n_i_fused_1) % 128)
                        vk_0 = T.axis.reduce(1, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params6B_batch16_token256:
    @T.prim_func
    def main(
        A: T.Buffer((256, 256, 256), "int32"),
        B: T.Buffer((256, 256), "int32"),
        C: T.Buffer((256, 256), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((256, 1, 256), "int32")
        C_rf_global_local = T.alloc_buffer((256, 1, 256), "int32", scope="local")
        A_local = T.alloc_buffer((256, 256, 256), "int32", scope="local")
        B_local = T.alloc_buffer((256, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                1, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    1024, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 4):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(256, n_1_i_1_fused // 4 + n_3)
                                v_i = T.axis.spatial(
                                    256, n_1_i_1_fused % 4 * 64 + n_2_i_2_fused * 4 + i_3
                                )
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(32):
                                for ax0_ax1_ax2_fused in range(8):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(256, n_1_i_1_fused // 4)
                                        v1 = T.axis.spatial(
                                            256, n_1_i_1_fused % 4 * 64 + n_2_i_2_fused * 4 + i_3
                                        )
                                        v2 = T.axis.spatial(256, k_1_0 * 8 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(8):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(256, n_1_i_1_fused // 4)
                                        v1 = T.axis.spatial(256, k_1_0 * 8 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(8):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(256, n_1_i_1_fused // 4 + n_3)
                                        v_i = T.axis.spatial(
                                            256, n_1_i_1_fused % 4 * 64 + n_2_i_2_fused * 4 + i_3
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 8 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 4):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(256, n_1_i_1_fused // 4 + ax0)
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    256, n_1_i_1_fused % 4 * 64 + n_2_i_2_fused * 4 + ax2
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(8):
            for n_i_fused_1 in range(8192):
                with T.block("C_init"):
                    v_n = T.axis.spatial(256, (n_i_fused_0 * 8192 + n_i_fused_1) // 256)
                    v_i = T.axis.spatial(256, (n_i_fused_0 * 8192 + n_i_fused_1) % 256)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(256, (n_i_fused_0 * 8192 + n_i_fused_1) // 256)
                        v_i = T.axis.spatial(256, (n_i_fused_0 * 8192 + n_i_fused_1) % 256)
                        vk_0 = T.axis.reduce(1, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params6B_batch16_token512:
    @T.prim_func
    def main(
        A: T.Buffer((256, 512, 256), "int32"),
        B: T.Buffer((256, 256), "int32"),
        C: T.Buffer((256, 512), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((256, 1, 512), "int32")
        C_rf_global_local = T.alloc_buffer((256, 1, 512), "int32", scope="local")
        A_local = T.alloc_buffer((256, 512, 256), "int32", scope="local")
        B_local = T.alloc_buffer((256, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                2, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    1024, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 4):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(256, n_1_i_1_fused // 4 + n_3)
                                v_i = T.axis.spatial(
                                    512,
                                    n_0_i_0_fused * 256
                                    + n_1_i_1_fused % 4 * 64
                                    + n_2_i_2_fused * 4
                                    + i_3,
                                )
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_ax2_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(256, n_1_i_1_fused // 4)
                                        v1 = T.axis.spatial(
                                            512,
                                            n_0_i_0_fused * 256
                                            + n_1_i_1_fused % 4 * 64
                                            + n_2_i_2_fused * 4
                                            + i_3,
                                        )
                                        v2 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(256, n_1_i_1_fused // 4)
                                        v1 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(16):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(256, n_1_i_1_fused // 4 + n_3)
                                        v_i = T.axis.spatial(
                                            512,
                                            n_0_i_0_fused * 256
                                            + n_1_i_1_fused % 4 * 64
                                            + n_2_i_2_fused * 4
                                            + i_3,
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 4):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(256, n_1_i_1_fused // 4 + ax0)
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    512,
                                    n_0_i_0_fused * 256
                                    + n_1_i_1_fused % 4 * 64
                                    + n_2_i_2_fused * 4
                                    + ax2,
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(64):
            for n_i_fused_1 in range(2048):
                with T.block("C_init"):
                    v_n = T.axis.spatial(256, (n_i_fused_0 * 2048 + n_i_fused_1) // 512)
                    v_i = T.axis.spatial(512, (n_i_fused_0 * 2048 + n_i_fused_1) % 512)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(256, (n_i_fused_0 * 2048 + n_i_fused_1) // 512)
                        v_i = T.axis.spatial(512, (n_i_fused_0 * 2048 + n_i_fused_1) % 512)
                        vk_0 = T.axis.reduce(1, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params6B_batch16_token64:
    @T.prim_func
    def main(
        A: T.Buffer((256, 64, 256), "int32"),
        B: T.Buffer((256, 256), "int32"),
        C: T.Buffer((256, 64), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((256, 1, 64), "int32")
        C_rf_global_local = T.alloc_buffer((256, 1, 64), "int32", scope="local")
        A_local = T.alloc_buffer((256, 64, 256), "int32", scope="local")
        B_local = T.alloc_buffer((256, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                128, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    4, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 2):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    256, n_0_i_0_fused * 2 + n_1_i_1_fused // 2 + n_3
                                )
                                v_i = T.axis.spatial(
                                    64, n_1_i_1_fused % 2 * 32 + n_2_i_2_fused * 2 + i_3
                                )
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_ax2_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            256, n_0_i_0_fused * 2 + n_1_i_1_fused // 2
                                        )
                                        v1 = T.axis.spatial(
                                            64, n_1_i_1_fused % 2 * 32 + n_2_i_2_fused * 2 + i_3
                                        )
                                        v2 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            256, n_0_i_0_fused * 2 + n_1_i_1_fused // 2
                                        )
                                        v1 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(16):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            256, n_0_i_0_fused * 2 + n_1_i_1_fused // 2 + n_3
                                        )
                                        v_i = T.axis.spatial(
                                            64, n_1_i_1_fused % 2 * 32 + n_2_i_2_fused * 2 + i_3
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 2):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    256, n_0_i_0_fused * 2 + n_1_i_1_fused // 2 + ax0
                                )
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    64, n_1_i_1_fused % 2 * 32 + n_2_i_2_fused * 2 + ax2
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(32):
            for n_i_fused_1 in range(512):
                with T.block("C_init"):
                    v_n = T.axis.spatial(256, (n_i_fused_0 * 512 + n_i_fused_1) // 64)
                    v_i = T.axis.spatial(64, (n_i_fused_0 * 512 + n_i_fused_1) % 64)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(256, (n_i_fused_0 * 512 + n_i_fused_1) // 64)
                        v_i = T.axis.spatial(64, (n_i_fused_0 * 512 + n_i_fused_1) % 64)
                        vk_0 = T.axis.reduce(1, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params6B_batch1_token64:
    @T.prim_func
    def main(
        A: T.Buffer((16, 64, 256), "int32"),
        B: T.Buffer((16, 256), "int32"),
        C: T.Buffer((16, 64), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((16, 2, 64), "int32")
        C_rf_global_local = T.alloc_buffer((16, 2, 64), "int32", scope="local")
        A_local = T.alloc_buffer((16, 64, 256), "int32", scope="local")
        B_local = T.alloc_buffer((16, 256), "int32", scope="local")
        for k_0 in T.thread_binding(2, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                8, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    8, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 1):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    16, n_0_i_0_fused * 2 + n_1_i_1_fused // 4 + n_3
                                )
                                v_i = T.axis.spatial(
                                    64, n_1_i_1_fused % 4 * 16 + n_2_i_2_fused + i_3
                                )
                                vk_0 = T.axis.spatial(2, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_ax2_fused in range(8):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            16, n_0_i_0_fused * 2 + n_1_i_1_fused // 4
                                        )
                                        v1 = T.axis.spatial(
                                            64, n_1_i_1_fused % 4 * 16 + n_2_i_2_fused
                                        )
                                        v2 = T.axis.spatial(
                                            256,
                                            k_0 * 128 + k_1_0 * 8 + ax0_ax1_ax2_fused,
                                        )
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(8):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            16, n_0_i_0_fused * 2 + n_1_i_1_fused // 4
                                        )
                                        v1 = T.axis.spatial(
                                            256, k_0 * 128 + k_1_0 * 8 + ax0_ax1_fused
                                        )
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(8):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            16,
                                            n_0_i_0_fused * 2 + n_1_i_1_fused // 4 + n_3,
                                        )
                                        v_i = T.axis.spatial(
                                            64,
                                            n_1_i_1_fused % 4 * 16 + n_2_i_2_fused + i_3,
                                        )
                                        vk_0 = T.axis.spatial(2, k_0)
                                        vk_1 = T.axis.reduce(128, k_1_0 * 8 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 128 + vk_1],
                                            B_local[v_n, vk_0 * 128 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 128 + vk_1]
                                            * B_local[v_n, vk_0 * 128 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 1):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    16, n_0_i_0_fused * 2 + n_1_i_1_fused // 4 + ax0
                                )
                                v1 = T.axis.spatial(2, k_0 + ax1)
                                v2 = T.axis.spatial(
                                    64, n_1_i_1_fused % 4 * 16 + n_2_i_2_fused + ax2
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(1):
            for n_i_fused_1 in range(1024):
                with T.block("C_init"):
                    v_n = T.axis.spatial(16, (n_i_fused_0 * 1024 + n_i_fused_1) // 64)
                    v_i = T.axis.spatial(64, (n_i_fused_0 * 1024 + n_i_fused_1) % 64)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(2):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(16, (n_i_fused_0 * 1024 + n_i_fused_1) // 64)
                        v_i = T.axis.spatial(64, (n_i_fused_0 * 1024 + n_i_fused_1) % 64)
                        vk_0 = T.axis.reduce(2, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params6B_batch1_token128:
    @T.prim_func
    def main(
        A: T.Buffer((16, 128, 256), "int32"),
        B: T.Buffer((16, 256), "int32"),
        C: T.Buffer((16, 128), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((16, 1, 128), "int32")
        C_rf_global_local = T.alloc_buffer((16, 1, 128), "int32", scope="local")
        A_local = T.alloc_buffer((16, 128, 256), "int32", scope="local")
        B_local = T.alloc_buffer((16, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                4, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    32, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 1):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    16, n_0_i_0_fused // 2 * 8 + n_1_i_1_fused // 4 + n_3
                                )
                                v_i = T.axis.spatial(
                                    128,
                                    n_0_i_0_fused % 2 * 64
                                    + n_1_i_1_fused % 4 * 16
                                    + n_2_i_2_fused
                                    + i_3,
                                )
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(32):
                                for ax0_ax1_ax2_fused in range(8):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            16, n_0_i_0_fused // 2 * 8 + n_1_i_1_fused // 4
                                        )
                                        v1 = T.axis.spatial(
                                            128,
                                            n_0_i_0_fused % 2 * 64
                                            + n_1_i_1_fused % 4 * 16
                                            + n_2_i_2_fused,
                                        )
                                        v2 = T.axis.spatial(256, k_1_0 * 8 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(8):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            16, n_0_i_0_fused // 2 * 8 + n_1_i_1_fused // 4
                                        )
                                        v1 = T.axis.spatial(256, k_1_0 * 8 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(8):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            16, n_0_i_0_fused // 2 * 8 + n_1_i_1_fused // 4 + n_3
                                        )
                                        v_i = T.axis.spatial(
                                            128,
                                            n_0_i_0_fused % 2 * 64
                                            + n_1_i_1_fused % 4 * 16
                                            + n_2_i_2_fused
                                            + i_3,
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 8 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 1):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    16, n_0_i_0_fused // 2 * 8 + n_1_i_1_fused // 4 + ax0
                                )
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    128,
                                    n_0_i_0_fused % 2 * 64
                                    + n_1_i_1_fused % 4 * 16
                                    + n_2_i_2_fused
                                    + ax2,
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(8):
            for n_i_fused_1 in range(256):
                with T.block("C_init"):
                    v_n = T.axis.spatial(16, (n_i_fused_0 * 256 + n_i_fused_1) // 128)
                    v_i = T.axis.spatial(128, (n_i_fused_0 * 256 + n_i_fused_1) % 128)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(16, (n_i_fused_0 * 256 + n_i_fused_1) // 128)
                        v_i = T.axis.spatial(128, (n_i_fused_0 * 256 + n_i_fused_1) % 128)
                        vk_0 = T.axis.reduce(1, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


@I.ir_module
class params6B_batch1_token256:
    @T.prim_func
    def main(
        A: T.Buffer((16, 256, 256), "int32"),
        B: T.Buffer((16, 256), "int32"),
        C: T.Buffer((16, 256), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((16, 1, 256), "int32")
        C_rf_global_local = T.alloc_buffer((16, 1, 256), "int32", scope="local")
        A_local = T.alloc_buffer((16, 256, 256), "int32", scope="local")
        B_local = T.alloc_buffer((16, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": 1}):
            for n_0_i_0_fused in T.thread_binding(8, thread="blockIdx.y", annotations={"bank": 1}):
                for n_1_i_1_fused in T.thread_binding(
                    64, thread="blockIdx.z", annotations={"bank": 1}
                ):
                    for n_2_i_2_fused in T.thread_binding(4, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 2):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    16, n_0_i_0_fused // 4 * 8 + n_1_i_1_fused // 8 + n_3
                                )
                                v_i = T.axis.spatial(
                                    256,
                                    n_0_i_0_fused % 4 * 64
                                    + n_1_i_1_fused % 8 * 8
                                    + n_2_i_2_fused * 2
                                    + i_3,
                                )
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": 1,
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(1):
                                for ax0_ax1_ax2_fused in range(256):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            16, n_0_i_0_fused // 4 * 8 + n_1_i_1_fused // 8
                                        )
                                        v1 = T.axis.spatial(
                                            256,
                                            n_0_i_0_fused % 4 * 64
                                            + n_1_i_1_fused % 8 * 8
                                            + n_2_i_2_fused * 2
                                            + i_3,
                                        )
                                        v2 = T.axis.spatial(256, ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(256):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            16, n_0_i_0_fused // 4 * 8 + n_1_i_1_fused // 8
                                        )
                                        v1 = T.axis.spatial(256, ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(256):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            16, n_0_i_0_fused // 4 * 8 + n_1_i_1_fused // 8 + n_3
                                        )
                                        v_i = T.axis.spatial(
                                            256,
                                            n_0_i_0_fused % 4 * 64
                                            + n_1_i_1_fused % 8 * 8
                                            + n_2_i_2_fused * 2
                                            + i_3,
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 256 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
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
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 2):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    16, n_0_i_0_fused // 4 * 8 + n_1_i_1_fused // 8 + ax0
                                )
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    256,
                                    n_0_i_0_fused % 4 * 64
                                    + n_1_i_1_fused % 8 * 8
                                    + n_2_i_2_fused * 2
                                    + ax2,
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(1):
            for n_i_fused_1 in range(4096):
                with T.block("C_init"):
                    v_n = T.axis.spatial(16, (n_i_fused_0 * 4096 + n_i_fused_1) // 256)
                    v_i = T.axis.spatial(256, (n_i_fused_0 * 4096 + n_i_fused_1) % 256)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": 1,
                            "meta_schedule.random_compute_producer": 1,
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(16, (n_i_fused_0 * 4096 + n_i_fused_1) // 256)
                        v_i = T.axis.spatial(256, (n_i_fused_0 * 4096 + n_i_fused_1) % 256)
                        vk_0 = T.axis.reduce(1, k_0)
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
class params6B_batch1_token512:
    @T.prim_func
    def main(
        A: T.Buffer((16, 512, 256), "int32"),
        B: T.Buffer((16, 256), "int32"),
        C: T.Buffer((16, 512), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((16, 1, 512), "int32")
        C_rf_global_local = T.alloc_buffer((16, 1, 512), "int32", scope="local")
        A_local = T.alloc_buffer((16, 512, 256), "int32", scope="local")
        B_local = T.alloc_buffer((16, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                4, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    64, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(1, 2):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    16, n_0_i_0_fused * 4 + n_1_i_1_fused // 16 + n_3
                                )
                                v_i = T.axis.spatial(
                                    512, n_1_i_1_fused % 16 * 32 + n_2_i_2_fused * 2 + i_3
                                )
                                vk_0 = T.axis.spatial(1, k_0)
                                T.reads()
                                T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                T.block_attr(
                                    {
                                        "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                            1
                                        ),
                                        "meta_schedule.tiling_structure": "SSSSRR",
                                    }
                                )
                                C_rf_global_local[v_n, vk_0, v_i] = 0
                            for k_1_0 in range(16):
                                for ax0_ax1_ax2_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            16, n_0_i_0_fused * 4 + n_1_i_1_fused // 16
                                        )
                                        v1 = T.axis.spatial(
                                            512, n_1_i_1_fused % 16 * 32 + n_2_i_2_fused * 2 + i_3
                                        )
                                        v2 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            16, n_0_i_0_fused * 4 + n_1_i_1_fused // 16
                                        )
                                        v1 = T.axis.spatial(256, k_1_0 * 16 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(16):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            16, n_0_i_0_fused * 4 + n_1_i_1_fused // 16 + n_3
                                        )
                                        v_i = T.axis.spatial(
                                            512, n_1_i_1_fused % 16 * 32 + n_2_i_2_fused * 2 + i_3
                                        )
                                        vk_0 = T.axis.spatial(1, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 16 + k_1_1)
                                        T.reads(
                                            C_rf_global_local[v_n, vk_0, v_i],
                                            A_local[v_n, v_i, vk_0 * 256 + vk_1],
                                            B_local[v_n, vk_0 * 256 + vk_1],
                                        )
                                        T.writes(C_rf_global_local[v_n, vk_0, v_i])
                                        T.block_attr(
                                            {
                                                "meta_schedule.meta_schedule_rfactor_producer_block": T.int64(
                                                    1
                                                ),
                                                "meta_schedule.tiling_structure": "SSSSRR",
                                            }
                                        )
                                        C_rf_global_local[v_n, vk_0, v_i] = (
                                            C_rf_global_local[v_n, vk_0, v_i]
                                            + A_local[v_n, v_i, vk_0 * 256 + vk_1]
                                            * B_local[v_n, vk_0 * 256 + vk_1]
                                        )
                        for ax0, ax1, ax2 in T.grid(1, 1, 2):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(
                                    16, n_0_i_0_fused * 4 + n_1_i_1_fused // 16 + ax0
                                )
                                v1 = T.axis.spatial(1, ax1)
                                v2 = T.axis.spatial(
                                    512, n_1_i_1_fused % 16 * 32 + n_2_i_2_fused * 2 + ax2
                                )
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(1):
            for n_i_fused_1 in range(8192):
                with T.block("C_init"):
                    v_n = T.axis.spatial(16, (n_i_fused_0 * 8192 + n_i_fused_1) // 512)
                    v_i = T.axis.spatial(512, (n_i_fused_0 * 8192 + n_i_fused_1) % 512)
                    T.reads()
                    T.writes(C[v_n, v_i])
                    T.block_attr(
                        {
                            "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                            "meta_schedule.random_compute_producer": T.int64(1),
                        }
                    )
                    C[v_n, v_i] = 0
                for k_0 in range(1):
                    with T.block("C_update"):
                        v_n = T.axis.spatial(16, (n_i_fused_0 * 8192 + n_i_fused_1) // 512)
                        v_i = T.axis.spatial(512, (n_i_fused_0 * 8192 + n_i_fused_1) % 512)
                        vk_0 = T.axis.reduce(1, k_0)
                        T.reads(C[v_n, v_i], C_rf_global[v_n, vk_0, v_i])
                        T.writes(C[v_n, v_i])
                        T.block_attr(
                            {
                                "meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1),
                                "meta_schedule.random_compute_producer": T.int64(1),
                            }
                        )
                        C[v_n, v_i] = C[v_n, v_i] + C_rf_global[v_n, vk_0, v_i]


tuples = {
    "params6B_batch1_token64": (params6B_batch1_token64, 16, 64, 256),
    "params6B_batch1_token128": (params6B_batch1_token128, 16, 128, 256),
    "params6B_batch1_token256": (params6B_batch1_token256, 16, 256, 256),
    "params6B_batch1_token512": (params6B_batch1_token512, 16, 512, 256),
    "params6B_batch16_token64": (params6B_batch16_token64, 256, 64, 256),
    "params6B_batch16_token128": (params6B_batch16_token128, 256, 128, 256),
    "params6B_batch16_token256": (params6B_batch16_token256, 256, 256, 256),
    "params6B_batch16_token512": (params6B_batch16_token512, 256, 512, 256),
    "params175B_batch1_token64": (params175B_batch1_token64, 48, 64, 256),
    "params175B_batch1_token128": (params175B_batch1_token128, 48, 128, 256),
    "params175B_batch1_token256": (params175B_batch1_token256, 48, 256, 256),
    "params175B_batch1_token512": (params175B_batch1_token512, 48, 512, 256),
    "params175B_batch16_token64": (params175B_batch16_token64, 768, 64, 256),
    "params175B_batch16_token128": (params175B_batch16_token128, 768, 128, 256),
    "params175B_batch16_token256": (params175B_batch16_token256, 768, 256, 256),
    "params175B_batch16_token512": (params175B_batch16_token512, 768, 512, 256),
}

# "params13B_batch1_token64": (20, 64, 256),
# "params13B_batch1_token128": (20, 128, 256),
# "params13B_batch1_token256": (20, 256, 256),
# "params13B_batch1_token512": (20, 512, 256),
# "params13B_batch16_token64": (320, 64, 256),
# "params13B_batch16_token128": (320, 128, 256),
# "params13B_batch16_token256": (320, 256, 256),
# "params13B_batch16_token512": (320, 512, 256),
# "params30B_batch1_token64": (28, 64, 256),
# "params30B_batch1_token128": (28, 128, 256),
# "params30B_batch1_token256": (28, 256, 256),
# "params30B_batch1_token512": (28, 512, 256),
# "params30B_batch16_token64": (448, 64, 256),
# "params30B_batch16_token128": (448, 128, 256),
# "params30B_batch16_token256": (448, 256, 256),
# "params30B_batch16_token512": (448, 512, 256),

target = Target("upmem --num-cores=96")

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


import numpy as np

print()

for conf, (cl, n, m, k) in tuples.items():
    print("##########################", cl, n, m, k)
    func = tvm.build(cl, target="upmem", name="gemv")
    dev.load_function(func)
    ha = np.ones((n, m, k)).astype("int32")
    hb = np.ones((n, k)).astype("int32")

    hc = np.einsum("nmk,nk->nm", ha, hb)
    # print("load", conf)
    a = tvm.nd.array(ha, device=dev, symbol="A")
    # print("INJECTED A")
    b = tvm.nd.array(hb, device=dev, symbol="B")
    # print("INJECTED B")


    c = tvm.nd.array(
        np.zeros(
            (
                n,
                m,
            )
        ).astype("int32"),
        device=dev,
        symbol="C",
    )
    # print("LOADED B&C")
    timestamp = tvm._ffi.get_global_func("device_api.upmem.timestamp")
    elapsed_time = tvm._ffi.get_global_func("device_api.upmem.elapsed_time")
    func(a, b, c)
    # print("LAUNCHED")

    # Compare hc and c, and print differences
    correct_count = 0
    incorrect_count = 0
    differences = []
    for i in range(n):
        for j in range(m):
            if hc[i, j] != c.asnumpy()[i, j]:
                incorrect_count += 1
                differences.append((i, j, hc[i, j], c.asnumpy()[i, j]))
            else:
                correct_count += 1

    print(f"Correct count: {correct_count}")
    print(f"Incorrect count: {incorrect_count}")

    if differences:
        print("Differences found (up to 1000):")
        for diff in differences:
            print(f"Index {diff[0], diff[1]}: Expected {diff[2]}, Got {diff[3]}")
    else:
        print("No differences found.")

    bs, ks, ass, ds, total = [], [], [], [], []
    for i in range(100):
        s = time.time()
        timestamp("start")
        func(a, b, c)
        timestamp("end")
        e = time.time()

        before_kernel_time = elapsed_time("before_kernel") / 1e6
        kernel_time = elapsed_time("kernel") / 1e6
        d2h_time = elapsed_time("after_d2h") / 1e6
        after_kernel_time = elapsed_time("after_kernel") / 1e6

        bs.append(before_kernel_time)
        ks.append(kernel_time)
        ass.append(after_kernel_time)
        ds.append(d2h_time)
        total.append((e - s) * 1000)

    print(f"{np.mean(bs)} {np.mean(ks)} {np.mean(ds)} {np.mean(ass)} {np.mean(total)}")
    dev.free()
