# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownershiC_rf_internal.  The ASF licenses this file
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
# pylint: disable=invalid-name,missing-function-docstring
"""Intrinsics for tensorization on NVIDIA GPU."""
import re
from typing import Dict, Tuple

from typing_extensions import Literal

from tvm.script import tir as T
from tvm.tir.function import PrimFunc

from ..._ffi import register_func
from ...runtime import convert
from .. import Cast, IntImm, TensorIntrin

CLK_LOCAL_MEM_FENCE = 0
CLK_GLOBAL_MEM_FENCE = 1


def R_CMD(dtype, addr, ptr=""):
    return T.evaluate(T.R_CMD(addr, ptr=ptr, dtype=dtype))
    return T.evaluate(T.vloadn(0, addr, ptr=ptr, dtype=dtype))


def W_CMD(dtype, addr, ptr=""):
    return T.evaluate(T.W_CMD(addr, ptr=ptr, dtype=dtype))
    return T.evaluate(T.vstoren(0, 0, addr, ptr=ptr, dtype=dtype))


def W_CMD_R(dtype, addr, src, ptr=""):
    return T.evaluate(T.W_CMD_R(addr, src, ptr=ptr, dtype=dtype))
    return T.evaluate(
        T.vstoren(T.vloadn(0, src, ptr="", dtype=dtype), 0, addr, ptr="pim_ctr", dtype=dtype)
    )


def W_CMD_R_C(dtype, addr, src, ptr=""):
    return T.evaluate(T.W_CMD_R_C(addr, src, ptr=ptr, dtype=dtype))
    return T.evaluate(
        T.vstoren(T.vloadn(0, src, ptr="", dtype=dtype), 0, addr, ptr="pim_ctr", dtype=dtype)
    )


def B_CMD(type):
    return T.evaluate(T.B_CMD(type))
    if type == CLK_LOCAL_MEM_FENCE:
        return T.evaluate(T.barrier(CLK_LOCAL_MEM_FENCE))
    else:
        return T.evaluate(T.mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE))


def get_input_intrin(dtype):
    @T.prim_func
    def input_desc(a: T.handle, b: T.handle) -> None:
        B = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="global")
        B_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(B[0:128])
            T.writes(B_local[0:128])
            for i in T.serial(0, 128):
                with T.block(""):
                    v_i = T.axis.remap("S", [i])
                    B_local[v_i] = B[v_i]

        # B = T.match_buffer(a, (1024,), dtype=dtype, scope="global")
        # B_local = T.match_buffer(b, (1024,), dtype=dtype, scope="local")
        # with T.block("root"):
        #     v0_o = T.int32()
        #     T.reads(B[v0_o * 128 : v0_o * 128 + 128])
        #     T.writes(B_local[v0_o * 128 : v0_o * 128 + 128])
        #     for i in T.serial(0, 128):
        #         with T.block(""):
        #             v0_i = T.axis.spatial(128, i)
        #             T.reads(B[v0_o * 128 + v0_i])
        #             T.writes(B_local[v0_o * 128 + v0_i])
        #             B_local[v0_o * 128 + v0_i] = B[v0_o * 128 + v0_i]

    @T.prim_func
    def input_impl(a: T.handle, b: T.handle) -> None:
        B = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="global")
        B_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(B[0:128])
            T.writes(B_local[0:128])
            ch = T.env_thread("blockIdx.x")
            T.launch_thread(ch, 64)
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, 16)

            addr = T.addr_gen(ch, 0, 0, B_local.bank_index() % 2, 0x3FFF, 0x8, tx * 16)
            W_CMD_R("int32x4", addr, B.access_ptr("r", offset=tx * 8), ptr="pim_ctr")
            R_CMD(
                "int32x4",
                addr,
                ptr="pim_ctr",
            )
            B_CMD(1)

        # B = T.match_buffer(a, (1024,), dtype=dtype, scope="global")
        # B_local = T.match_buffer(b, (1024,), dtype=dtype, scope="local")
        # with T.block("root"):
        #     v0_o = T.int32()
        #     T.reads(B[v0_o * 128 : v0_o * 128 + 128])
        #     T.writes(B_local[v0_o * 128 : v0_o * 128 + 128])
        #     for i in T.serial(0, 128):
        #         with T.block(""):
        #             v0_i = T.axis.spatial(128, i)
        #             T.reads(B[v0_o * 128 + v0_i])
        #             T.writes(B_local[v0_o * 128 + v0_i])
        #             B_local[v0_o * 128 + v0_i] = B[v0_o * 128 + v0_i]

    return input_desc, input_impl


def get_input_intrin_mm(dtype, N):
    @T.prim_func
    def input_desc(a: T.handle, b: T.handle) -> None:
        B = T.match_buffer(a, (N, 128 // N), dtype=dtype, offset_factor=1, scope="global")
        B_local = T.match_buffer(b, (N, 128 // N), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(B[0:N, 0 : 128 // N])
            T.writes(B_local[0:N, 0 : 128 // N])
            for i, j in T.grid(N, 128 // N):
                with T.block(""):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    B_local[v_i, v_j] = B[v_i, v_j]

    @T.prim_func
    def input_impl(a: T.handle, b: T.handle) -> None:
        B = T.match_buffer(a, (N, 128 // N), dtype=dtype, offset_factor=1, scope="global")
        B_local = T.match_buffer(b, (N, 128 // N), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(B[0:N, 0 : 128 // N])
            T.writes(B_local[0:N, 0 : 128 // N])
            ch = T.env_thread("blockIdx.x")
            T.launch_thread(ch, 64)
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, 16)

            offset = (tx % 8) * 8 + (tx // 8) * 1024  # TODO[ywshin]
            addr = T.addr_gen(ch, 0, 0, B_local.bank_index() % 2, 0x3FFF, 0x8, tx * 16)
            W_CMD_R("int32x4", addr, B.access_ptr("r", offset=offset), ptr="pim_ctr")
            R_CMD(
                "int32x4",
                addr,
                ptr="pim_ctr",
            )
            B_CMD(1)

    return input_desc, input_impl


def get_weight_intrin(dtype):
    @T.prim_func
    def weight_desc(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="global")
        A_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(A[0:128])
            T.writes(A_local[0:128])
            for i in T.serial(0, 128):
                with T.block(""):
                    v_i = T.axis.remap("S", [i])
                    A_local[v_i] = A[v_i]

    @T.prim_func
    def weight_impl(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="global")
        A_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(A[0:128])
            T.writes(A_local[0:128])
            ch = T.env_thread("blockIdx.x")
            T.launch_thread(ch, 64)
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, 16)
            R_CMD(
                "int32x4",
                A.access_ptr(
                    "r",
                    offset=T.addr_gen(
                        ch,
                        0,
                        0,
                        A_local.bank_index() % 2,
                        0,
                        0,
                        offset=A_local.in_bank_offset_of([tx * 8])[0] * 2,
                    )
                    // 2,
                    ignore_elem_offset=True,
                ),
            )
            # B_CMD(1)  # TODO: optimize position

        # A = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="global")
        # A_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        # with T.block("root"):
        #     T.reads(A[0:128])
        #     T.writes(A_local[0:128])
        #     tx = T.env_thread("threadIdx.x")
        #     T.launch_thread(tx, 16)
        #     for i in T.serial(0, 128):
        #         with T.block(""):
        #             v_i = T.axis.remap("S", [i])
        #             A_local[v_i] = A[v_i]
        #             # A_local.in_bank_offset_of([tx * 16])[0]

    return weight_desc, weight_impl


def get_weight_intrin_mm(dtype, N):
    @T.prim_func
    def weight_desc(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, (128 // N,), dtype=dtype, offset_factor=1, scope="global")
        A_local = T.match_buffer(b, (128 // N,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(A[0 : 128 // N])
            T.writes(A_local[0 : 128 // N])
            for i in T.serial(0, 128 // N):
                with T.block(""):
                    v_i = T.axis.remap("S", [i])
                    A_local[v_i] = A[v_i]

    @T.prim_func
    def weight_impl(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, (128 // N,), dtype=dtype, offset_factor=1, scope="global")
        A_local = T.match_buffer(b, (128 // N,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(A[0 : 128 // N])
            T.writes(A_local[0 : 128 // N])
            ch = T.env_thread("blockIdx.x")
            T.launch_thread(ch, 64)
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, 16)  # TODO[ywshin]
            R_CMD(
                "int32x4",
                A.access_ptr(
                    "r",
                    offset=T.addr_gen(
                        ch,
                        0,
                        0,
                        A_local.bank_index() % 2,
                        0,
                        0,
                        offset=A_local.in_bank_offset_of([(tx // N) * 8])[0] * 2,
                    )
                    // 2,
                    ignore_elem_offset=True,
                ),
            )

    return weight_desc, weight_impl


def get_mac_intrin(dtype):
    @T.prim_func
    def mac_desc(a: T.handle, b: T.handle, p: T.handle) -> None:
        A_local = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="local")
        B_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        C_rf_internal_local = T.match_buffer(p, (16,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(C_rf_internal_local[0:16], A_local[0:128], B_local[0:128])
            T.writes(C_rf_internal_local[0:16])
            for k, r in T.grid(8, 16):
                with T.block(""):
                    # v_k, v_r = T.axis.remap("RS", [k, r])
                    v_r, v_k = T.axis.remap("SR", [r, k])
                    C_rf_internal_local[v_r] = (
                        C_rf_internal_local[v_r] + A_local[v_k * 16 + v_r] * B_local[v_k * 16 + v_r]
                    )

    @T.prim_func
    def mac_impl(a: T.handle, b: T.handle, p: T.handle) -> None:
        # A = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="local")
        # B = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        # C_rf_internal = T.match_buffer(c, (16,), dtype=dtype, offset_factor=1, scope="global")
        # with T.block("root"):
        #     T.reads(C_rf_internal[0:16], A[0:128], B[0:128])
        #     T.writes(C_rf_internal[0:16])
        #     ch = T.env_thread("blockIdx.x")
        #     T.launch_thread(ch, 64)
        #     tx = T.env_thread("threadIdx.x")
        #     T.launch_thread(tx, 16)
        #     T.evaluate(
        #         T.W_CMD(
        #             C_rf_internal.access_ptr(
        #                 "r",
        #                 offset=T.addr_gen(ch=ch, bk=1, offset=C_rf_internal.in_bank_offset_of([tx * 16])[0]),
        #             )
        #         )
        #     )

        A_local = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="local")
        B_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        C_rf_internal_local = T.match_buffer(p, (16,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(C_rf_internal_local[0:16], A_local[0:128], B_local[0:128])
            T.writes(C_rf_internal_local[0:16])
            # for k, r in T.grid(8, 16):
            #     with T.block(""):
            #         v_k, v_r = T.axis.remap("RS", [k, r])
            #         C_rf_internal_local[v_r] = C_rf_internal_local[v_r] + A_local[v_k * 16 + v_r] * B_local[v_k * 16 + v_r]

    return mac_desc, mac_impl


def get_mac_intrin_mm(dtype, N):
    @T.prim_func
    def mac_desc(a: T.handle, b: T.handle, p: T.handle) -> None:
        A_local = T.match_buffer(a, (128 // N,), dtype=dtype, offset_factor=1, scope="local")
        B_local = T.match_buffer(b, (128 // N,), dtype=dtype, offset_factor=1, scope="local")
        C_rf_internal_local = T.match_buffer(p, (16,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(C_rf_internal_local[0:16], A_local[0 : 128 // N], B_local[0 : 128 // N])
            T.writes(C_rf_internal_local[0:16])
            for k, r in T.grid(8 // N, 16):
                with T.block(""):
                    v_k, v_r = T.axis.remap("RS", [k, r])
                    C_rf_internal_local[v_r] = (
                        C_rf_internal_local[v_r] + A_local[v_k * 16 + v_r] * B_local[v_k * 16 + v_r]
                    )

    @T.prim_func
    def mac_impl(a: T.handle, b: T.handle, p: T.handle) -> None:
        A_local = T.match_buffer(a, (128 // N,), dtype=dtype, offset_factor=1, scope="local")
        B_local = T.match_buffer(b, (128 // N,), dtype=dtype, offset_factor=1, scope="local")
        C_rf_internal_local = T.match_buffer(p, (16,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(C_rf_internal_local[0:16], A_local[0 : 128 // N], B_local[0 : 128 // N])
            T.writes(C_rf_internal_local[0:16])

    return mac_desc, mac_impl


def get_partial_reduction_intrin(dtype):
    @T.prim_func
    def partial_reduction_desc(a: T.handle, b: T.handle) -> None:
        C_rf_internal = T.match_buffer(a, (8, 16), dtype=dtype, offset_factor=1, scope="internal")
        C_rf_internal_local = T.match_buffer(
            b, (8, 16), dtype=dtype, offset_factor=1, scope="local"
        )
        with T.block("root"):
            T.reads(C_rf_internal_local[0:8, 0:16])
            T.writes(C_rf_internal[0:8, 0:16])
            for k, r in T.grid(8, 16):
                with T.block(""):
                    v_k, v_r = T.axis.remap("SS", [k, r])
                    C_rf_internal[v_k, v_r] = C_rf_internal_local[v_k, v_r]

    @T.prim_func
    def partial_reduction_impl(a: T.handle, b: T.handle) -> None:
        C_rf_internal = T.match_buffer(a, (8, 16), dtype=dtype, offset_factor=1, scope="internal")
        C_rf_internal_local = T.match_buffer(
            b, (8, 16), dtype=dtype, offset_factor=1, scope="local"
        )
        with T.block("root"):
            T.reads(C_rf_internal_local[0:8, 0:16])
            T.writes(C_rf_internal[0:8, 0:16])
            ch = T.env_thread("blockIdx.x")
            T.launch_thread(ch, 64)
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, 16)
            addr = (
                T.addr_gen(
                    ch,
                    0,
                    0,
                    1,
                    0,
                    0,
                    offset=(C_rf_internal_local.in_bank_offset_of([0, 0])[0] + tx * 8) * 2,
                )
                // 2
            )
            W_CMD("int32x4", C_rf_internal.access_ptr("rw", offset=addr, ignore_elem_offset=True))
            W_CMD("int32x4", C_rf_internal.access_ptr("rw", offset=addr, ignore_elem_offset=True))
            R_CMD("int32x4", C_rf_internal.access_ptr("rw", offset=addr, ignore_elem_offset=True))
            B_CMD(1)

    return partial_reduction_desc, partial_reduction_impl


def get_partial_reduction_intrin_mm(dtype, N):
    @T.prim_func
    def partial_reduction_desc(a: T.handle, b: T.handle) -> None:
        C_rf_internal = T.match_buffer(a, (N, 8, 16), dtype=dtype, offset_factor=1, scope="global")
        C_rf_internal_local = T.match_buffer(
            b, (N, 8, 16), dtype=dtype, offset_factor=1, scope="local"
        )
        with T.block("root"):
            T.reads(C_rf_internal_local[0:N, 0:8, 0:16])
            T.writes(C_rf_internal[0:N, 0:8, 0:16])
            for j, k, r in T.grid(2, 8, 16):
                with T.block(""):
                    v_j, v_k, v_r = T.axis.remap("SSS", [j, k, r])
                    C_rf_internal[v_j, v_k, v_r] = C_rf_internal_local[v_j, v_k, v_r]

    @T.prim_func
    def partial_reduction_impl(a: T.handle, b: T.handle) -> None:
        C_rf_internal = T.match_buffer(a, (N, 8, 16), dtype=dtype, offset_factor=1, scope="global")
        C_rf_internal_local = T.match_buffer(
            b, (N, 8, 16), dtype=dtype, offset_factor=1, scope="local"
        )
        with T.block("root"):
            T.reads(C_rf_internal_local[0:N, 0:8, 0:16])
            T.writes(C_rf_internal[0:N, 0:8, 0:16])
            ch = T.env_thread("blockIdx.x")
            T.launch_thread(ch, 64)
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, 16)
            addr = (
                T.addr_gen(
                    ch,
                    0,
                    0,
                    1,
                    0,
                    0,
                    offset=(C_rf_internal_local.in_bank_offset_of([0, 0, 0])[0] + tx * 8) * 2,
                )
                // 2
            )
            W_CMD("int32x4", C_rf_internal.access_ptr("rw", offset=addr, ignore_elem_offset=True))
            W_CMD("int32x4", C_rf_internal.access_ptr("rw", offset=addr, ignore_elem_offset=True))
            R_CMD("int32x4", C_rf_internal.access_ptr("rw", offset=addr, ignore_elem_offset=True))
            B_CMD(1)

    return partial_reduction_desc, partial_reduction_impl


HBMPIM_INPUT_INTRIN = "hbmpim_input_intrin"
TensorIntrin.register(
    HBMPIM_INPUT_INTRIN,
    *get_input_intrin("int16"),
)

HBMPIM_INPUT_INTRIN_MM = "hbmpim_input_intrin_mm"
TensorIntrin.register(
    HBMPIM_INPUT_INTRIN_MM,
    *get_input_intrin_mm("int16", 2),
)

HBMPIM_WEIGHT_INTRIN = "hbmpim_weight_intrin"
TensorIntrin.register(
    HBMPIM_WEIGHT_INTRIN,
    *get_weight_intrin("int16"),
)

HBMPIM_WEIGHT_INTRIN_MM = "hbmpim_weight_intrin_mm"
TensorIntrin.register(
    HBMPIM_WEIGHT_INTRIN_MM,
    *get_weight_intrin_mm("int16", 2),
)

HBMPIM_MAC_INTRIN = "hbmpim_mac_intrin"
TensorIntrin.register(
    HBMPIM_MAC_INTRIN,
    *get_mac_intrin("int16"),
)

HBMPIM_MAC_INTRIN_MM = "hbmpim_mac_intrin_mm"
TensorIntrin.register(
    HBMPIM_MAC_INTRIN_MM,
    *get_mac_intrin_mm("int16", 2),
)

HBMPIM_PARTIAL_REDUCTION_INTRIN = "hbmpim_partial_reduction_intrin"
TensorIntrin.register(
    HBMPIM_PARTIAL_REDUCTION_INTRIN,
    *get_partial_reduction_intrin("int16"),
)
HBMPIM_PARTIAL_REDUCTION_INTRIN_MM = "hbmpim_partial_reduction_intrin_mm"
TensorIntrin.register(
    HBMPIM_PARTIAL_REDUCTION_INTRIN_MM,
    *get_partial_reduction_intrin_mm("int16", 2),
)
