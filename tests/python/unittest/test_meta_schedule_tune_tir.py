# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring,no-member,invalid-name,unused-variable
import logging
import tempfile

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.local_rpc import LocalRPC
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule
from typing import Callable

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


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


def va_factory(L, dtype):
    @tvm.script.ir_module
    class VAModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A", "B"],
                }
            )
            A = T.match_buffer(a, (L,), dtype=dtype)
            B = T.match_buffer(b, (L,), dtype=dtype)
            C = T.match_buffer(c, (L,), dtype=dtype)
            for i in T.serial(0, L):
                with T.block("C"):
                    C[i] = A[i] + B[i]

    return VAModule


@T.prim_func
def two_step(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (1024, 1024), "float32")
    B = T.alloc_buffer((1024, 1024), "float32")
    C = T.match_buffer(c, (1024, 1024), "float32")
    for i, j in T.grid(1024, 1024):
        with T.block("A"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(1024, 1024):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 3.0


@tvm.testing.requires_llvm
def test_tune_matmul_cpu():
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("llvm --num-cores=16")
        database = ms.tir_integration.tune_tir(
            mod=matmul,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
        )
        sch = ms.tir_integration.compile_tir(database, matmul, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()


def test_tune_matmul_hbmpim():
    matvec = matvec_factory(4096, 1024, dtype="int16")
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("hbmpim")
        database = ms.tir_integration.tune_tir(
            mod=matvec,
            target=target,
            work_dir=work_dir,
            max_trials_global=1,
            num_trials_per_iter=1,
            per_iter_timeout_sec=180,  # NOTE: timeout 에러가 조용하게 발생하는 버그가 있으니 유의해야 한다.
            min_repeat_ms=0,
            num_tuning_cores=1,  # running simulator
        )
        sch = ms.tir_integration.compile_tir(database, matvec, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()


def test_tune_matmul_upmem():
    matvec = va_factory(160000000, dtype="int32")
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("upmem --num-cores=96")
        database = ms.tir_integration.tune_tir(
            mod=matvec,
            target=target,
            work_dir=work_dir,
            max_trials_global=1000,
            num_trials_per_iter=64,
            num_tuning_cores=1,  # to prevent dpu allocation error
        )
        sch = ms.tir_integration.compile_tir(database, matvec, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()


# @tvm.testing.requires_cuda
def test_tune_matmul_cuda():
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/geforce-rtx-3070")
        database = ms.tir_integration.tune_tir(
            mod=matmul,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
        )
        sch = ms.tir_integration.compile_tir(database, matmul, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()


def test_tune_run_module_via_rpc():
    target = tvm.target.Target("llvm")
    rt_mod = tvm.build(matmul, target)

    # construct the input
    input_data = {}
    input_shape = (128, 128)
    input_dtype = "float32"
    a_np = np.random.uniform(size=input_shape).astype(input_dtype)
    b_np = np.random.uniform(size=input_shape).astype(input_dtype)
    c_np = np.zeros(input_shape).astype(input_dtype)
    for i in range(128):
        for j in range(128):
            for k in range(128):
                c_np[i, j] = c_np[i, j] + a_np[i, k] * b_np[j, k]
    input_data["a"] = a_np
    input_data["b"] = b_np
    input_data["c"] = np.zeros(input_shape).astype(input_dtype)

    with LocalRPC() as rpc:
        rpc_config = ms.runner.RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=100,
        )

        def f_timer(rt_mod, dev, input_data):
            rt_mod(input_data["a"], input_data["b"], input_data["c"])
            return input_data["c"]

        result = run_module_via_rpc(
            rpc_config=rpc_config,
            lib=rt_mod,
            dev_type=target.kind.name,
            args=input_data,
            continuation=f_timer,
        )
        tvm.testing.assert_allclose(result.numpy(), c_np, rtol=1e-3)


def test_tune_block_cpu():
    @ms.derived_object
    class RemoveBlock(ms.schedule_rule.PyScheduleRule):
        def _initialize_with_tune_context(self, context: ms.TuneContext) -> None:
            pass

        def apply(self, sch: Schedule, block: BlockRV):
            if sch.get(block).name_hint == "root":
                return [sch]
            sch = sch.copy()
            sch.compute_inline(block)
            return [sch]

        def clone(self) -> "RemoveBlock":
            return RemoveBlock()

    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("llvm --num-cores=16")
        database = ms.tir_integration.tune_tir(
            mod=two_step,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
            space=ms.space_generator.PostOrderApply(
                f_block_filter=lambda block: block.name_hint == "A",
                sch_rules=[RemoveBlock()],
                postprocs=[],
                mutator_probs={},
            ),
        )
        sch = ms.tir_integration.compile_tir(database, two_step, target)
        assert sch is not None
        sch.mod.show()
        sch.trace.show()


if __name__ == """__main__""":
    test_tune_matmul_upmem()
    # test_tune_matmul_hbmpim()
    # test_tune_matmul_cpu()
    # test_tune_matmul_cuda()
    # test_tune_run_module_via_rpc()
    # test_tune_block_cpu()
