from tvm.script import ir as I
from tvm.script import tir as T
import tvm
from tvm.tir.transform import *
from tvm.target import Target

import numpy as np

# from tvm.script import ir as I
# from tvm.script import tir as T


@I.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer((16, 64, 256), "int32"),
        B: T.Buffer((16, 256), "int32"),
        C: T.Buffer((16, 64), "int32"),
    ):
        T.func_attr(
            {"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)}
        )
        # with T.block("root"):
        C_rf_global = T.alloc_buffer((16, 1, 64), "int32")
        C_rf_global_local = T.alloc_buffer((16, 1, 64), "int32", scope="local")
        A_local = T.alloc_buffer((16, 64, 256), "int32", scope="local")
        B_local = T.alloc_buffer((16, 256), "int32", scope="local")
        for k_0 in T.thread_binding(1, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
            for n_0_i_0_fused in T.thread_binding(
                2, thread="blockIdx.y", annotations={"bank": T.int64(1)}
            ):
                for n_1_i_1_fused in T.thread_binding(
                    4, thread="blockIdx.z", annotations={"bank": T.int64(1)}
                ):
                    for n_2_i_2_fused in T.thread_binding(1, thread="threadIdx.x"):
                        for n_3, i_3 in T.grid(2, 64):
                            with T.block("C_rf_init"):
                                v_n = T.axis.spatial(
                                    16, n_0_i_0_fused * 8 + n_1_i_1_fused * 2 + n_3
                                )
                                v_i, vk_0 = T.axis.remap("SS", [i_3, k_0])
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
                            for k_1_0 in range(1):
                                for ax0_ax1_ax2_fused in range(256):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(
                                            16, n_0_i_0_fused * 8 + n_1_i_1_fused * 2 + n_3
                                        )
                                        v1, v2 = T.axis.remap("SS", [i_3, ax0_ax1_ax2_fused])
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(256):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            16, n_0_i_0_fused * 8 + n_1_i_1_fused * 2 + n_3
                                        )
                                        v1 = T.axis.spatial(256, ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for k_1_1 in range(256):
                                    with T.block("C_rf_update"):
                                        v_n = T.axis.spatial(
                                            16, n_0_i_0_fused * 8 + n_1_i_1_fused * 2 + n_3
                                        )
                                        v_i, vk_0 = T.axis.remap("SS", [i_3, k_0])
                                        vk_1 = T.axis.reduce(256, k_1_0 * 256 + k_1_1)
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
                        for ax0, ax1, ax2 in T.grid(2, 1, 64):
                            with T.block("C_rf_global_local"):
                                v0 = T.axis.spatial(16, n_0_i_0_fused * 8 + n_1_i_1_fused * 2 + ax0)
                                v1, v2 = T.axis.remap("SS", [ax1, ax2])
                                T.reads(C_rf_global_local[v0, v1, v2])
                                T.writes(C_rf_global[v0, v1, v2])
                                C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
        for n_i_fused_0 in T.parallel(2):
            for n_i_fused_1 in range(512):
                with T.block("C_init"):
                    v_n = T.axis.spatial(16, (n_i_fused_0 * 512 + n_i_fused_1) // 64)
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
                        v_n = T.axis.spatial(16, (n_i_fused_0 * 512 + n_i_fused_1) // 64)
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


# func = tvm.build(Module, target="upmem", name="mha_crash")


# print(func.imported_modules[0].get_source())

target = tvm.target.Target(target="upmem", host="llvm")
l = tvm.lower(Module)
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
# m = ExtractPimTransferSchedule()(m)
# m = SplitHostDevice()(m)
# m = SplitPimTransfer()(m)
print(m)


# dev = tvm.device("upmem")
# dev.load_function(func)

# a = tvm.nd.array(np.random.randint(0, 100, (16, 64, 256), dtype="int32"), device=dev, symbol="A")
# b = tvm.nd.array(np.random.randint(0, 100, (16, 256), dtype="int32"), device=dev, symbol="B")
# c = tvm.nd.array(np.zeros((16, 64), dtype="int32"), device=dev)

# print(func.imported_modules[0].get_source())
# # func(a, b, c)

# dev.free()
