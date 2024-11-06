import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from tensor import host_array
import numpy as np
import math
from tvm.script import ir as I
from tvm.script import tir as T


# basic kernels
def upmem_va_factory(M, dtype):
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
            A = T.match_buffer(a, (M,), dtype=dtype)
            B = T.match_buffer(b, (M,), dtype=dtype)
            C = T.match_buffer(c, (M,), dtype=dtype)
            for i in T.grid(M):
                with T.block("C"):
                    v_i = T.axis.remap("S", [i])
                    C[v_i] = A[v_i] + B[v_i]

    return VAModule


def upmem_dot_factory(M, dtype):
    @tvm.script.ir_module
    class DOTModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A", "B"],
                }
            )
            A = T.match_buffer(a, (M,), dtype=dtype)
            B = T.match_buffer(b, (M,), dtype=dtype)
            C = T.match_buffer(c, (1,), dtype=dtype)
            for i in T.grid(M):
                with T.block("C"):
                    with T.init():
                        C[0] = 0
                    v_i = T.axis.remap("R", [i])
                    C[0] = C[0] + A[v_i] * B[v_i]

    return DOTModule


def upmem_red_factory(M, dtype):
    @tvm.script.ir_module
    class REDModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A"],
                }
            )
            A = T.match_buffer(a, (M,), dtype=dtype)
            B = T.match_buffer(b, (1,), dtype=dtype)
            for i in T.grid(M):
                with T.block("C"):
                    with T.init():
                        B[0] = 0
                    v_i = T.axis.remap("R", [i])
                    B[0] = B[0] + A[v_i]

    return REDModule


def upmem_mtv_factory(M, K, dtype):
    @tvm.script.ir_module
    class MTVModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": T.bool(True), "pragma_explicit_h2d": ["A"]}
            )
            A = T.match_buffer(a, (M, K), dtype=dtype)
            B = T.match_buffer(b, (K,), dtype=dtype)
            C = T.match_buffer(c, (M,), dtype=dtype)
            for i, k in T.grid(M, K):
                with T.block("C"):
                    v_i, v_k = T.axis.remap("SR", [i, k])
                    with T.init():
                        C[v_i] = 0
                    C[v_i] = C[v_i] + A[v_i, v_k] * B[v_k]

    return MTVModule


# higher dimension kernels
def upmem_ta_factory(M, N, K, dtype):
    @tvm.script.ir_module
    class TAModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A", "B"],
                }
            )
            A = T.match_buffer(a, (M, N, K), dtype=dtype)
            B = T.match_buffer(b, (M, N, K), dtype=dtype)
            C = T.match_buffer(c, (M, N, K), dtype=dtype)
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
                    C[v_i, v_j, v_k] = A[v_i, v_j, v_k] + B[v_i, v_j, v_k]

    return TAModule


def upmem_innerprod_factory(M, N, K, dtype):
    @tvm.script.ir_module
    class InnerprodModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A", "B"],
                }
            )
            A = T.match_buffer(a, (M, N, K), dtype=dtype)
            B = T.match_buffer(b, (M, N, K), dtype=dtype)
            C = T.match_buffer(c, (1,), dtype=dtype)
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    with T.init():
                        C[0] = 0
                    v_i, v_j, v_k = T.axis.remap("RRR", [i, j, k])
                    C[0] = C[0] + A[v_i, v_j, v_k] * B[v_i, v_j, v_k]

    return InnerprodModule


def upmem_ttv_factory(M, N, K, dtype):
    @tvm.script.ir_module
    class TTVModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": T.bool(True), "pragma_explicit_h2d": ["A"]}
            )
            A = T.match_buffer(a, (M, N, K), dtype=dtype)
            B = T.match_buffer(b, (K,), dtype=dtype)
            C = T.match_buffer(c, (M, N), dtype=dtype)
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[v_i, v_j] = 0
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_j, v_k] * B[v_k]

    return TTVModule


# batched multi-head attention
def upmem_mmtv_factory(M, N, K, dtype):
    @tvm.script.ir_module
    class MTTVModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A"],
                }
            )
            A = T.match_buffer(a, (M, N, K), dtype=dtype)
            B = T.match_buffer(
                b,
                (
                    M,
                    K,
                ),
                dtype=dtype,
            )
            C = T.match_buffer(c, (M, N), dtype=dtype)
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[v_i, v_j] = 0
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_j, v_k] * B[v_i, v_k]

    return MTTVModule


# polybench kernels
def upmem_poly_va_factory(M, dtype):
    @tvm.script.ir_module
    class PolyVAModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle, alpha: T.handle, beta: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A", "B"],
                }
            )
            A = T.match_buffer(a, (M,), dtype=dtype)
            B = T.match_buffer(b, (M,), dtype=dtype)
            C = T.match_buffer(c, (M,), dtype=dtype)
            ALPHA = T.match_buffer(alpha, (1,), dtype=dtype)
            BETA = T.match_buffer(beta, (1,), dtype=dtype)
            alpha_val = ALPHA[0]
            beta_val = BETA[0]
            for i in T.grid(M):
                with T.block("C"):
                    v_i = T.axis.remap("S", [i])
                    C[v_i] = alpha_val * A[v_i] + beta_val * B[v_i]

    return PolyVAModule


def upmem_poly_gemv1_factory(M, K, dtype):
    @tvm.script.ir_module
    class PolyGEMV1Module:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle, alpha: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": T.bool(True), "pragma_explicit_h2d": ["A"]}
            )
            A = T.match_buffer(a, (M, K), dtype=dtype)
            B = T.match_buffer(b, (K,), dtype=dtype)
            C = T.match_buffer(c, (M), dtype=dtype)
            ALPHA = T.match_buffer(alpha, (1,), dtype=dtype)
            alpha_val = ALPHA[0]
            for i, k in T.grid(M, K):
                with T.block("C"):
                    v_i, v_k = T.axis.remap("SR", [i, k])
                    with T.init():
                        C[v_i] = 0
                    C[v_i] = C[v_i] + alpha_val * A[v_i, v_k] * B[v_k]

    return PolyGEMV1Module


def upmem_poly_mixed_factory(M, N, dtype):
    @tvm.script.ir_module
    class PolyMixedModule:
        @T.prim_func
        def main(c: T.handle, u1: T.handle, v1: T.handle, u2: T.handle, v2: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["C", "U1", "V1", "U2", "V2"],
                }
            )
            C = T.match_buffer(c, (M, N), dtype=dtype)
            U1 = T.match_buffer(u1, (M,), dtype=dtype)
            V1 = T.match_buffer(v1, (N,), dtype=dtype)
            U2 = T.match_buffer(u2, (M,), dtype=dtype)
            V2 = T.match_buffer(v2, (N,), dtype=dtype)
            for i, j in T.grid(M, N):
                with T.block("C"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    with T.init():
                        C[v_i, v_j] = 0
                    C[v_i, v_j] = C[v_i, v_j] + U1[v_i] * V1[v_j] + U2[v_i] * V2[v_j]

    return PolyMixedModule


def upmem_poly_gemv2_factory(M, K, dtype):
    @tvm.script.ir_module
    class PolyGEMV2Module:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": T.bool(True), "pragma_explicit_h2d": ["A"]}
            )
            A = T.match_buffer(a, (K, M), dtype=dtype)
            B = T.match_buffer(b, (K,), dtype=dtype)
            C = T.match_buffer(c, (M), dtype=dtype)
            for i, k in T.grid(M, K):
                with T.block("C"):
                    v_i, v_k = T.axis.remap("SR", [i, k])
                    with T.init():
                        C[v_i] = 0
                    C[v_i] = C[v_i] + A[v_k, v_i] * B[v_k]

    return PolyGEMV2Module


def mtvRCTile(M, K, n_xb, n_yb, n_yt=16, n_cache=64, n_rt=16, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, k = sch.get_loops(block_c)
    xb, xo = sch.split(k, factors=[n_xb, None])
    block_crf = sch.rfactor(xb, factor_axis=0)
    ca = sch.cache_read(block_crf, 0, "local")
    cb = sch.cache_read(block_crf, 1, "local")
    cc = sch.cache_write(block_crf, 0, "local")
    i, xb, k = sch.get_loops(block_crf)
    yb, yo, yi, yc = sch.split(i, factors=[n_yb, n_yt, None, 2])
    xo, xi = sch.split(k, factors=[None, n_cache])
    sch.reorder(xb, yb, yo, yi, yc, xo, xi)
    sch.compute_at(ca, xo)
    sch.compute_at(cb, xo)
    sch.reverse_compute_at(cc, yi)
    sch.bind(xb, "blockIdx.x")
    sch.bind(yb, "blockIdx.y")
    sch.bind(yo, "threadIdx.x")
    sch.annotate(xb, "bank", True)
    sch.annotate(yb, "bank", True)
    sch.decompose_reduction(block_crf, xo)
    i, _ = sch.get_loops(block_c)
    it, ii = sch.split(i, factors=[n_rt, None])
    sch.parallel(it)
    return sch


def customTile(**kwargs):
    sch = tvm.tir.Schedule(Module)
    return sch


def mtvRTile(M, K, n_yb, n_cache=64, n_yt=16, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, xo = sch.get_loops(block_c)
    ca = sch.cache_read(block_c, 0, "local")
    cb = sch.cache_read(block_c, 1, "local")
    cc = sch.cache_write(block_c, 0, "local")
    i, k = sch.get_loops(block_c)

    rounded = math.ceil(M / n_yb / 2) * 2
    yb, yo = sch.split(i, factors=[n_yb, rounded])
    yo, yi, yc = sch.split(yo, factors=[n_yt, None, 2])

    xo, xi = sch.split(k, factors=[None, n_cache])
    sch.reorder(yb, yo, yi, yc, xo, xi)
    sch.compute_at(ca, xo)
    sch.compute_at(cb, xo)
    sch.reverse_compute_at(cc, yi)
    sch.bind(yb, "blockIdx.x")
    sch.bind(yo, "threadIdx.x")
    sch.annotate(sch.get_block("A_local"), "pragma_explicit_h2d", True)
    sch.annotate(yb, "bank", True)
    sch.decompose_reduction(block_c, xo)
    return sch


def StridedBankTile(M, K, n_xb, n_yb, n_yt=16, n_cache=64, n_rt=64, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, k = sch.get_loops(block_c)
    xb, x2, xo = sch.split(k, factors=[n_xb, 2, None])
    block_crf = sch.rfactor(x2, factor_axis=0)
    ca = sch.cache_read(block_crf, 0, "local")
    cb = sch.cache_read(block_crf, 1, "local")
    cc = sch.cache_write(block_crf, 0, "local")
    i, xb, x2, xo = sch.get_loops(block_crf)
    yb, yo, yi, yc = sch.split(i, factors=[n_yb, n_yt, M // n_yb // n_yt // 2, None])
    xo, xi = sch.split(xo, factors=[K // n_xb // n_cache, None])
    sch.reorder(x2, xb, yb, yo, yi, yc, xo, xi)
    sch.compute_at(ca, xo)
    sch.compute_at(cb, xo)
    sch.reverse_compute_at(cc, yo)
    sch.bind(x2, "blockIdx.x")
    sch.bind(yb, "blockIdx.y")
    sch.bind(yo, "threadIdx.x")
    i, _ = sch.get_loops(block_c)
    it, _ = sch.split(i, factors=[n_rt, None])
    sch.parallel(it)
    return sch


def HBMStyleTile(dtype="int32", **kwargs):
    M, K, N_CHAN, N_BANK, N_PU, N_GRF_A, N_GRF_B = 4096, 1024, 64, 16, 8, 8, 8
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, k = sch.get_loops(block_c)
    k_1, k_0, k_2 = sch.split(k, factors=[None, N_BANK // N_PU, 128])
    block_crf = sch.rfactor(k_0, factor_axis=0)
    cache_read_a = sch.cache_read(block_crf, 0, "local")
    cache_read_b = sch.cache_read(block_crf, 1, "local")
    cache_write_p = sch.cache_write(block_crf, 0, "local")
    i, k_1, k_0, k_2 = sch.get_loops(block_crf)
    i_1, i_2, i_3 = sch.split(i, factors=[N_CHAN, N_PU, N_GRF_B])
    sch.reorder(i_1, i_2, k_0, k_1, i_3, k_2)
    sch.compute_at(cache_read_a, loop=i_3)
    sch.compute_at(cache_read_b, loop=k_1)
    sch.reverse_compute_at(cache_write_p, i_3)
    sch.blockize(k_2)
    sch.bind(i_1, "blockIdx.x")
    sch.bind(i_2, "blockIdx.y")
    sch.bind(k_0, "blockIdx.z")
    sch.bind(i_3, "threadIdx.x")
    i, r = sch.get_loops(block_c)
    i_0, i_1, i_2, i_3 = sch.split(i, factors=[None, N_CHAN, N_PU, N_GRF_B])
    sch.reorder(i_1, i_0, i_2, i_3)
    return sch


class GEMV(UPMEMWorkload):
    def __init__(self, **kwargs):
        required = dict(M=8192, K=8192, dtype="int32", n_xb=1, n_yb=1, n_cache=64, n_yt=16, n_rt=64)
        super().__init__(
            profile="gemv", required=required, symbols=["A", "B", "C"], output_symbol="C", **kwargs
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, self.K), self.dtype)
        self.host.B = host_array((self.K,), self.dtype)

    def host_version(self):
        self.host.C = np.dot(self.host.A, self.host.B)

    def benchmark_command(self, config):
        bl = int(math.log2(config["n_cache"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"make clean && NR_DPUS={config['n_xb'] * config['n_yb']} \
            NR_TASKLETS={config['n_yt']} TYPE={pbtype} BL={bl} make && \
            ./bin/gemv_host -m {config['M']} -n {config['K']} -w {self.warmup} -e {self.repeat}"
