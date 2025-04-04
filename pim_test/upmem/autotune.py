import logging
import tempfile

import numpy as np
import pytest
import tvm
import sys
import time
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.local_rpc import LocalRPC
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule
from typing import Callable


def matvec_factory(M: int, K: int, dtype="int32") -> Callable[[T.handle, T.handle, T.handle], None]:
    @T.prim_func
    def matvec(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"pragma_explicit_h2d": ["A"]})
        A = T.match_buffer(a, (M, K), dtype=dtype)
        B = T.match_buffer(b, (K,), dtype=dtype)
        C = T.match_buffer(c, (M,), dtype=dtype)
        for i, k in T.grid(M, K):
            with T.block("update"):
                vi, vk = T.axis.remap("SR", [i, k])
                with T.init():
                    C[vi] = 0
                C[vi] = C[vi] + A[vi, vk] * B[vk]

    return matvec


def bgemv_factory(N: int, M: int, K: int, dtype="int32"):
    @T.prim_func
    def batched_gemv(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"pragma_explicit_h2d": ["A"]})
        A = T.match_buffer(a, (N, M, K), dtype=dtype)
        B = T.match_buffer(b, (N, K), dtype=dtype)
        C = T.match_buffer(c, (N, M), dtype=dtype)

        for n, i, k in T.grid(N, M, K):
            with T.block("C"):
                v_n, v_i, v_k = T.axis.remap("SSR", [n, i, k])
                with T.init():
                    C[v_n, v_i] = 0
                C[v_n, v_i] = C[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]

    return batched_gemv


def gemm_factory(M: int, N: int, L: int, dtype):
    @T.prim_func
    def gemm(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"pragma_explicit_h2d": ["A"]})
        A = T.match_buffer(a, (M, N), dtype=dtype)
        B = T.match_buffer(b, (N, L), dtype=dtype)
        C = T.match_buffer(c, (M, L), dtype=dtype)
        for i, j, k in T.grid(M, L, N):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.int32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    return gemm


# mod_mv = [matvec_factory(163840, 4096, dtype="int32")]
tuple_bmv = [(16, j, 256) for j in [16, 32, 64, 128, 256, 512]] + [
    (j, 256, 256) for j in [16, 32, 64, 128, 256, 512, 1024]
]
# 175, 30, 13, 6
tuple_mv = {
    # "175B_qkvgen": (36864, 12288),  # 21775 -> 아놔 진짜 다시 돌려야함
    # "175B_qkvproj": (12288, 12288),  # 7278
    # "175B_fc": (49152, 12288),  # 27918
    # "175B_fcproj": (12288, 49152),  # 27847
    # "13B_qkvgen": (15360, 5120),
    # "13B_qkvproj": (5120, 5120),
    # "13B_fc": (20480, 5120),
    # "13B_fcproj": (5120, 20480),
    # "6B_qkvgen": (12288, 4096),
    # "6B_qkvproj": (4096, 4096),
    # "6B_fc": (16384, 4096),
    # "6B_fcproj": (4096, 16384),
    # "30B_qkvgen": (21504, 7168),
    # "30B_qkvproj": (7168, 7168),
    # "30B_fc": (28672, 7168),
    # "30B_fcproj": (7168, 28672),
    # "175B_fcproj_1": (12288, 49152),  # 27847
    # "175B_fcproj_2": (12288, 49152),  # 27847
    # "175B_fcproj_3": (12288, 49152),  # 27847
    # "163840_4096_1": (163840, 4096),  # 27847
    # "163840_4096_2": (163840, 4096),  # 27847
    # "163840_4096_3": (163840, 4096),  # 27847
    "ablation": (163840, 4096),
}

# layers = {6: 16, 13: 20, 30: 28, 175: 48}
tuple_bmv = {
    # "params6B_batch1_token64": (16, 64, 256),
    # "params6B_batch1_token128": (16, 128, 256),
    # "params6B_batch1_token256": (16, 256, 256),
    # "params6B_batch1_token512": (16, 512, 256),
    # "params6B_batch16_token64": (256, 64, 256),
    # "params6B_batch16_token128": (256, 128, 256),
    # "params6B_batch16_token256": (256, 256, 256),
    # "params6B_batch16_token512": (256, 512, 256),
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
    # "params30B_batch16_token512_1": (448, 512, 256),
    # "params30B_batch16_token512_2": (448, 512, 256),
    # "params30B_batch16_token512_3": (448, 512, 256),
    # "params175B_batch1_token64": (48, 64, 256),
    "params175B_batch1_token128": (48, 128, 256),
    "params175B_batch1_token256": (48, 256, 256),
    "params175B_batch1_token512": (48, 512, 256),
    "params175B_batch16_token64": (768, 64, 256),
    "params175B_batch16_token128": (768, 128, 256),
    "params175B_batch16_token256": (768, 256, 256),
    "params175B_batch16_token512": (768, 512, 256),
}

target = Target("upmem --num-cores=96")
# target = Target("llvm --num-cores=96")
for name, (M, K) in tuple_mv.items():
    start = time.time()
    with open(f"./autotuner_result_mv_ablation/gemv_{name}.txt", "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        mod = matvec_factory(M, K, dtype="int32")
        database = ms.tir_integration.tune_tir(
            mod=mod,
            target=target,
            work_dir="./autotuner_result_mv_ablation",
            max_trials_global=1000,
            num_trials_per_iter=64,
            # num_tuning_cores=1,  # to prevent dpu allocation error
        )
        sch = ms.tir_integration.compile_tir(database, mod, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()
        sys.stdout = original_stdout
    end = time.time()
    print("DONE ", name, " in ", end - start, " seconds")

# for name, (N, M, K) in tuple_bmv.items():
#     start = time.time()
#     with open(f"./autotuner_result_bmv_new_resume/bgemv_{name}.txt", "w") as f:
#         original_stdout = sys.stdout
#         sys.stdout = f
#         mod = bgemv_factory(N, M, K, dtype="int32")
#         database = ms.tir_integration.tune_tir(
#             mod=mod,
#             target=target,
#             work_dir="./autotuner_result_bmv_new_resume",
#             max_trials_global=1000,
#             num_trials_per_iter=64,
#             # num_tuning_cores=1,  # to prevent dpu allocation error
#         )
#         sch = ms.tir_integration.compile_tir(database, mod, target)
#         if sch is None:
#             print("No valid schedule found!")
#         else:
#             sch.trace.show()
#             print("######################################################")
#             sch.mod.show(
#                 black_format=False,
#                 name=name,
#             )

#         sys.stdout = original_stdout
#     end = time.time()
#     print("DONE ", name, " in ", end - start, " seconds")
