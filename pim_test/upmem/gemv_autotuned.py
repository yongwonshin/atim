import tvm
from tvm.script import tir as T
import tvm.script as I
from base import UPMEMWorkload, cleanup
from tensor import host_array
import numpy as np
import time
import math
import argparse
from tvm.tir.transform import *
from tvm.target import Target


@I.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer((8192, 8192), "int32"),
        B: T.Buffer((8192,), "int32"),
        C: T.Buffer((8192,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((16, 8192), "int32")
        C_rf_global_local = T.alloc_buffer((16, 8192), "int32", scope="local")
        A_local = T.alloc_buffer((8192, 8192), "int32", scope="local")
        B_local = T.alloc_buffer((8192,), "int32", scope="local")
        for k_0 in T.thread_binding(16, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(128, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(1):
                        for i_3 in range(4):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(8192, i_0 * 64 + i_1 * 4 + i_2 * 4 + i_3)
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
                            for k_1_0 in range(64):
                                for ax0_ax1_fused in range(8):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(8192, i_0 * 64 + i_1 * 4 + i_3)
                                        v1 = T.axis.spatial(
                                            8192, k_0 * 512 + k_1_0 * 8 + ax0_ax1_fused
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(8):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(8192, k_0 * 512 + k_1_0 * 8 + ax0_fused)
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(8):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            8192, i_0 * 64 + i_1 * 4 + i_2 * 4 + i_3
                                        )
                                        vk_0 = T.axis.spatial(16, k_0)
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
                                v0 = T.axis.spatial(16, k_0 + ax0)
                                v1 = T.axis.spatial(8192, i_0 * 64 + i_1 * 4 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(32):
            for i_1 in range(256):
                with T.block("update_init"):
                    vi = T.axis.spatial(8192, i_0 * 256 + i_1)
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
                        vi = T.axis.spatial(8192, i_0 * 256 + i_1)
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
class ModuleBig:

    @T.prim_func
    def main(
        A: T.Buffer((163840, 4096), "int32"),
        B: T.Buffer((4096,), "int32"),
        C: T.Buffer((163840,), "int32"),
    ):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((16, 163840), "int32")
        C_rf_global_local = T.alloc_buffer((16, 163840), "int32", scope="local")
        A_local = T.alloc_buffer((163840, 4096), "int32", scope="local")
        B_local = T.alloc_buffer((4096,), "int32", scope="local")
        for k_0 in T.thread_binding(16, thread="blockIdx.x", annotations={"bank": 1}):
            for i_0 in T.thread_binding(128, thread="blockIdx.y", annotations={"bank": 1}):
                for i_1 in T.thread_binding(20, thread="threadIdx.x"):
                    for i_2 in range(1):
                        for i_3 in range(64):
                            with T.block("update_rf_init"):
                                vi = T.axis.spatial(163840, i_0 * 1280 + i_1 * 64 + i_2 * 64 + i_3)
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
                            for k_1_0 in range(16):
                                for ax0_ax1_fused in range(16):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(163840, i_0 * 1280 + i_1 * 64 + i_3)
                                        v1 = T.axis.spatial(
                                            4096, k_0 * 256 + k_1_0 * 16 + ax0_ax1_fused
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0_fused in range(16):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            4096, k_0 * 256 + k_1_0 * 16 + ax0_fused
                                        )
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for k_1_1 in range(16):
                                    with T.block("update_rf_update"):
                                        vi = T.axis.spatial(
                                            163840,
                                            i_0 * 1280 + i_1 * 64 + i_2 * 64 + i_3,
                                        )
                                        vk_0 = T.axis.spatial(16, k_0)
                                        vk_1 = T.axis.reduce(256, k_1_0 * 16 + k_1_1)
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
                        for ax0, ax1 in T.grid(1, 64):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(16, k_0 + ax0)
                                v1 = T.axis.spatial(163840, i_0 * 1280 + i_1 * 64 + ax1)
                                T.reads(C_rf_global_local[v0, v1])
                                T.writes(C_rf_global[v0, v1])
                                C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
        for i_0 in T.parallel(40):
            for i_1 in range(4096):
                with T.block("update_init"):
                    vi = T.axis.spatial(163840, i_0 * 4096 + i_1)
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
                        vi = T.axis.spatial(163840, i_0 * 4096 + i_1)
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


sch, M, K = Module, 8192, 8192
sch, M, K = ModuleBig, 163840, 4096

func = tvm.build(sch, target="upmem", name="gemv")

l = tvm.lower(sch)
target = tvm.target.Target(target="upmem", host="llvm")
mp, _ = Target.canon_target_map_and_host({target: l}, "llvm")
m = mp[target]
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
print("[TIR with PIM data copy]\n", m)

print("\n\n[UPMEM source]\n")
print(func.imported_modules[0].get_source())

device = tvm.upmem(func)
a = tvm.nd.array(host_array((M, K), "int32"), device, symbol="A")
b = tvm.nd.array(host_array((K,), "int32"), device, symbol="B")
c = tvm.nd.empty((M,), "int32", device)

repeat, warmup = 100, 3
timestamp = tvm._ffi.get_global_func("device_api.upmem.timestamp")
elapsed_time = tvm._ffi.get_global_func("device_api.upmem.elapsed_time")

bt, kt, at, rt, tt = 0, 0, 0, 0, 0

for j in range(repeat + warmup):
    st = time.time()
    timestamp("start")
    func(a, b, c)
    timestamp("end")
    et = time.time()

    if j >= warmup:
        bt += elapsed_time("before_kernel") / 1e6
        kt += elapsed_time("kernel") / 1e6
        at += elapsed_time("d2h") / 1e6
        rt += elapsed_time("after_d2h") / 1e6
        tt += (et - st) * 1e3

print(bt / repeat)
print(kt / repeat)
print(at / repeat)
print(rt / repeat)
print(tt / repeat)
