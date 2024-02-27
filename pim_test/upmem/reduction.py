import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from tensor import host_array
import numpy as np

def red_prim_schedule(L, dtype):
    M = 1
    @tvm.script.ir_module
    class ReductionModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            A = T.match_buffer(a, [L, M], dtype=dtype)
            B = T.match_buffer(b, [M], dtype=dtype)
            for i, k in T.grid(L, M):
                with T.block("C"):
                    v_i, v_k = T.axis.remap("RS", [i, k])
                    with T.init():
                        B[v_k] = 0
                    B[v_k] = B[v_k] + A[v_i, v_k]
    return ReductionModule

def crossReduction(L, n_b, n_t, n_c, dtype):
    sch = tvm.tir.Schedule(red_prim_schedule(L, dtype))

    br = sch.get_block("C")
    i, _ = sch.get_loops(br)
    ib, _, _ = sch.split(i, factors=[n_b, n_t, None])
    brf = sch.rfactor(ib, factor_axis=0) #C_rf
    _, it, _, _ = sch.get_loops(brf)
    trf = sch.rfactor(it, factor_axis=0) #C_rf_rf
    ca = sch.cache_read(trf, 0, "local")
    cc = sch.cache_write(trf, 0, "local")
    tib, tit, tii, _ = sch.get_loops(trf)
    tii, _ = sch.split(tii, factors = [None, n_c])
    sch.compute_at(ca, tii)
    sch.reverse_compute_at(cc, tii)
    sch.reverse_compute_at(brf, tit)
    sch.bind(tib, "blockIdx.x")
    sch.bind(tit, "threadIdx.x")
    sch.decompose_reduction(trf, tii)
    return sch
    
class REDUCE(UPMEMWorkload):
    def __init__(self):
        super().__init__(profile="reduction",
                         required=dict(L=8388608, dtype="int64", n_b=1024, n_t=16, n_c=64),
                         symbols=["A", "B"], output_symbol="B")

    def fetch_data(self):
        self.host.A = host_array((self.L, 1), self.dtype, new=True)
        self.host.B = host_array((1, ), self.dtype, new=True)
    
    def host_version(self):
        self.host.B = np.sum(self.host.A)
        
    def h2d(self):
        self.dev.A = tvm.nd.array(self.host.A, self.target_device, symbol=self.func[f"copy_A"])
        self.dev.B = tvm.nd.empty((1, ), self.dtype, self.target_device)
    
if __name__ == "__main__":
    cleanup()
    reduce = REDUCE()
    reduce.test(crossReduction, n_b=128)
    reduce.test(crossReduction, n_b=256)
    reduce.test(crossReduction, n_b=512)
    reduce.test(crossReduction, n_b=1024)
    reduce.test(crossReduction, n_b=2048)