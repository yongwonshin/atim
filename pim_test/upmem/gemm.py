from tvm import te as T
import tvm
from base import UPMEMWorkload, cleanup
from .tensor import host_array

def gemm_prim_schedule(M, N, L, dtype):
    @tvm.script.ir_module
    class UPMEMModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            A = T.match_buffer(a, (M, N), dtype=dtype)
            B = T.match_buffer(b, (N, L), dtype=dtype)
            C = T.match_buffer(c, (M, L), dtype=dtype)
            for i, j, k in T.grid(M, L, N):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.int32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    return UPMEMModule

def gemmTile(M, N, L, n_mb, n_nb, n_lb, n_mc, n_nc, n_lc, n_yt, n_rt, dtype):
    sch = tvm.tir.Schedule(gemm_prim_schedule(M, N, L, dtype))
    block_c = sch.get_block("C")
    _, _, k = sch.get_loops(block_c)
    kb, ko = sch.split(k, factors=[n_nb, None])
    block_crf = sch.rfactor(kb, factor_axis=0)
    ca = sch.cache_read(block_crf, 0, "local")
    cb = sch.cache_read(block_crf, 1, "local")
    cc = sch.cache_write(block_crf, 0, "local")
    i, j, kb, ko = sch.get_loops(block_crf)
    ib, io, ii, ic = sch.split(i, factors=[n_mb, n_yt, None, n_mc])
    jb, jo, jc = sch.split(j, factors=[n_lb, None, n_lc])
    ko, ki = sch.split(ko, factors=[None, n_nc])
    sch.reorder(ib, jb, kb, io, jo, ii, ko, ic, jc,ki)
    sch.compute_at(ca, ko)
    sch.compute_at(cb, ko)
    sch.reverse_compute_at(cc, ii)
    sch.bind(ib, "blockIdx.x")
    sch.bind(jb, "blockIdx.y")
    sch.bind(kb, "blockIdx.z")
    sch.bind(io, "threadIdx.x")
    i, _, _ = sch.get_loops(block_c)
    it, ii = sch.split(i, factors=[n_rt, None])
    sch.parallel(it)
    return sch

class GEMM(UPMEMWorkload):
    def __init__(self):
        required = dict(M=2048, N=2048, L=2048, dtype="int32", n_mb=1, n_nb=1, n_lb=1, 
                        n_mc=8, n_nc=16, n_lc=8, n_yt=16, n_rt=64)
        super().__init__(profile="gemm", required=required, symbols=["A", "B", "C"], output_symbol="C")
        
    def fetch_data(self):
        self.host.A = host_array(self.M, self.N, self.dtype)
        self.host.B = host_array(self.N, self.L, self.dtype)
    
    def host_version(self):
        self.host.C = self.host.A @ self.host.B
        
if __name__ == "__main__":
    cleanup()
    gemm = GEMM()
    """
    gemmTest(512, 512, 512, 4, 4, 4, 8, 16, 8, "int32", "gemm_A")
    gemmTest(512, 1024, 512, 4, 4, 4, 8, 16, 8, "int32", "gemm_B")
    gemmTest(512, 2048, 512, 4, 4, 4, 8, 16, 8, "int32", "gemm_C")
    gemmTest(512, 2048, 1024, 4, 4, 4, 8, 16, 8, "int32", "gemm_D")
    gemmTest(2048, 2048, 2048, 4, 4, 4, 8, 16, 8, "int32", "gemm_E")
    gemmTest(2048, 2048, 2048, 8, 8, 8, 8, 16, 8, "int32", "gemm_F")
    gemmTest(2048, 2048, 2048, 8, 8, 8, 8, 32, 8, "int32", "gemm_G")
    """
    gemm.test(gemmTile, n_mb=4, n_nb=4, n_lb=4, n_mc=8, n_nc=16, n_lc=8, n_yt=16, n_rt=64)
    gemm.test(gemmTile, n_mb=8, n_nb=8, n_lb=8, n_mc=8, n_nc=16, n_lc=8, n_yt=16, n_rt=64)
    gemm.test(gemmTile, n_mb=8, n_nb=8, n_lb=8, n_mc=8, n_nc=32, n_lc=8, n_yt=16, n_rt=64)
    
