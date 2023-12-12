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
        # B = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="global")
        # B_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        # with T.block("root"):
        #     T.reads(B[0:128])
        #     T.writes(B_local[0:128])
        #     ch = T.env_thread("blockIdx.x")
        #     T.launch_thread(ch, 64)
        #     tx = T.env_thread("threadIdx.x")
        #     T.launch_thread(tx, 16)
        #     T.evaluate(
        #         T.W_CMD_R(
        #             "pim_ctr",
        #             T.addr_gen(ch=ch, bk=B.bank_index() % 2, row=0x3FFF, col=0x8, offset=tx * 16),
        #             B.access_ptr("r", offset=B.offset_of([tx * 16])[0]),
        #         )
        #     )
        #     T.evaluate(
        #         T.R_CMD(
        #             T.addr_gen(ch=ch, bk=B.bank_index(), row=0x3FFF, col=0x8, offset=tx * 16),
        #             ptr="pim_ctr",
        #         )
        #     )

        B = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="global")
        B_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(B[0:128])
            T.writes(B_local[0:128])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, 16)
            for i in T.serial(0, 128):
                with T.block(""):
                    v_i = T.axis.remap("S", [i])
                    B_local[v_i] = B[v_i]  # B.offset_of([tx * 16])[0]

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
        # A = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="global")
        # A_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        # with T.block("root"):
        #     T.reads(A[0:128])
        #     T.writes(A_local[0:128])
        #     ch = T.env_thread("blockIdx.x")
        #     T.launch_thread(ch, 64)
        #     tx = T.env_thread("threadIdx.x")
        #     T.launch_thread(tx, 16)
        #     T.evaluate(
        #         T.R_CMD(
        #             A.access_ptr(
        #                 "r", offset=T.addr_gen(ch=ch, offset=A.in_bank_offset_of([tx * 16])[0])
        #             )
        #         )
        #     )

        A = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="global")
        A_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(A[0:128])
            T.writes(A_local[0:128])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, 16)
            for i in T.serial(0, 128):
                with T.block(""):
                    v_i = T.axis.remap("S", [i])
                    A_local[v_i] = A[v_i]
                    # A_local.in_bank_offset_of([tx * 16])[0]

    return weight_desc, weight_impl


def get_mac_intrin(dtype):
    @T.prim_func
    def mac_desc(a: T.handle, b: T.handle, p: T.handle) -> None:
        A_local = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="local")
        B_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        P_local = T.match_buffer(p, (16,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(P_local[0:16], A_local[0:128], B_local[0:128])
            T.writes(P_local[0:16])
            for k, r in T.grid(8, 16):
                with T.block(""):
                    v_k, v_r = T.axis.remap("RS", [k, r])
                    P_local[v_r] = P_local[v_r] + A_local[v_k * 16 + v_r] * B_local[v_k * 16 + v_r]

    @T.prim_func
    def mac_impl(a: T.handle, b: T.handle, p: T.handle) -> None:
        # A = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="local")
        # B = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        # P = T.match_buffer(c, (16,), dtype=dtype, offset_factor=1, scope="global")
        # with T.block("root"):
        #     T.reads(P[0:16], A[0:128], B[0:128])
        #     T.writes(P[0:16])
        #     ch = T.env_thread("blockIdx.x")
        #     T.launch_thread(ch, 64)
        #     tx = T.env_thread("threadIdx.x")
        #     T.launch_thread(tx, 16)
        #     T.evaluate(
        #         T.W_CMD(
        #             P.access_ptr(
        #                 "r",
        #                 offset=T.addr_gen(ch=ch, bk=1, offset=P.in_bank_offset_of([tx * 16])[0]),
        #             )
        #         )
        #     )

        A_local = T.match_buffer(a, (128,), dtype=dtype, offset_factor=1, scope="local")
        B_local = T.match_buffer(b, (128,), dtype=dtype, offset_factor=1, scope="local")
        P_local = T.match_buffer(p, (16,), dtype=dtype, offset_factor=1, scope="local")
        with T.block("root"):
            T.reads(P_local[0:16], A_local[0:128], B_local[0:128])
            T.writes(P_local[0:16])
        #     for k, r in T.grid(8, 16):
        #         with T.block(""):
        #             v_k, v_r = T.axis.remap("RS", [k, r])
        #             P_local[v_r] = P_local[v_r] + A_local[v_k * 16 + v_r] * B_local[v_k * 16 + v_r]

    return mac_desc, mac_impl


HBMPIM_INPUT_INTRIN = "hbmpim_input_intrin"
TensorIntrin.register(
    HBMPIM_INPUT_INTRIN,
    *get_input_intrin("float32"),
)

HBMPIM_WEIGHT_INTRIN = "hbmpim_weight_intrin"
TensorIntrin.register(
    HBMPIM_WEIGHT_INTRIN,
    *get_weight_intrin("float32"),
)

HBMPIM_MAC_INTRIN = "hbmpim_mac_intrin"
TensorIntrin.register(
    HBMPIM_MAC_INTRIN,
    *get_mac_intrin("float32"),
)
