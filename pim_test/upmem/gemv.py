import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from tensor import host_array
import numpy as np


def upmem_gemv_factory(M, K, dtype):
    @tvm.script.ir_module
    class UPMEMModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            A = T.match_buffer(a, (M, K), dtype=dtype)
            B = T.match_buffer(b, (K,), dtype=dtype)
            C = T.match_buffer(c, (M,), dtype=dtype)
            for i, k in T.grid(M, K):
                with T.block("C"):
                    v_i, v_k = T.axis.remap("SR", [i, k])
                    with T.init():
                        C[v_i] = 0
                    C[v_i] = C[v_i] + A[v_i, v_k] * B[v_k]
    return UPMEMModule

def gemvRCTile(M, K, n_xb, n_yb, n_yt = 16, n_cache = 64, n_rt = 128, dtype = "int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_gemv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, k = sch.get_loops(block_c)
    xb, xo = sch.split(k, factors = [n_xb, None])
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

def gemvRTile(M, K, n_yb, n_cache = 64, n_yt = 16, dtype = "int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_gemv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, xo = sch.get_loops(block_c)
    ca = sch.cache_read(block_c, 0, "local")
    cb = sch.cache_read(block_c, 1, "local")
    cc = sch.cache_write(block_c, 0, "local")
    i, k = sch.get_loops(block_c)
    yb, yo, yi, yc = sch.split(i, factors=[n_yb, n_yt, None, 2])
    xo, xi = sch.split(k, factors=[None, n_cache])
    sch.reorder(yb, yo, yi, yc, xo, xi)
    sch.compute_at(ca, xo)
    sch.compute_at(cb, xo)
    sch.reverse_compute_at(cc, yi)
    sch.bind(yb, "blockIdx.x")
    sch.bind(yo, "threadIdx.x")
    sch.annotate(yb, "bank", True)
    sch.decompose_reduction(block_c, xo)
    return sch

def StridedBankTile(M, K, n_xb, n_yb, n_yt = 16, n_cache = 64, n_rt = 64, dtype = "int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_gemv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, k = sch.get_loops(block_c)
    xb, x2, xo = sch.split(k, factors = [n_xb, 2, None])
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
    sch = tvm.tir.Schedule(upmem_gemv_factory(M, K, dtype))
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
    def __init__(self):
        required = dict(M=8192, K=8192, dtype="int32", n_xb=1, n_yb=1, n_cache=64, n_yt=16, n_rt=64)
        super().__init__(profile="gemv", required=required, symbols=["A", "B", "C"], output_symbol="C")

    def fetch_data(self):
        self.host.A = host_array((self.M, self.K), self.dtype)
        self.host.B = host_array((self.K,), self.dtype)

    def host_version(self):
        self.host.C = np.dot(self.host.A, self.host.B)

if __name__ == "__main__":
    cleanup()
    gemv = GEMV()
    gemv.test(gemvRTile, n_yb=32, n_cache=256)
    gemv.test(gemvRTile, n_yb=64, n_cache=256)
    gemv.test(gemvRCTile, n_yb=32, n_cache=256)
    gemv.test(gemvRCTile, n_yb=64, n_cache=256)
