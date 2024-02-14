import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from tensor import host_array

def va_prim_schedule(L, dtype):
    @tvm.script.ir_module
    class VAModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            A = T.match_buffer(a, (L,), dtype=dtype)
            B = T.match_buffer(b, (L,), dtype=dtype)
            C = T.match_buffer(c, (L,), dtype=dtype)
            for i in T.serial(0, L):
                with T.block("C"):
                    C[i] = A[i] + B[i]
    return VAModule

def vaTile(L, n_b, n_t, n_c, dtype):
    sch = tvm.tir.Schedule(va_prim_schedule(L, dtype))
    block_c = sch.get_block("C")
    i, = sch.get_loops(block_c)
    ca = sch.cache_read(block_c, "A", "local")
    cb = sch.cache_read(block_c, "B", "local")
    cc = sch.cache_write(block_c, "C", "local")
    ib, it, ii, ic = sch.split(i, factors=[n_b, n_t, None, n_c])
    sch.compute_at(ca, ii)
    sch.compute_at(cb, ii)
    sch.reverse_compute_at(cc, ii)
    sch.bind(ib, "blockIdx.x")
    sch.bind(it, "threadIdx.x")
    sch.parallel(ic)
    return sch

class VA(UPMEMWorkload):
    def __init__(self):
        super().__init__(profile="va", 
                         required=dict(L=65536, dtype="int32", n_b=4, n_t=16, n_c=256), 
                         symbols=["A", "B", "C"], output_symbol="C")
    def fetch_data(self):
        self.host.A = host_array(self.L, self.dtype)
        self.host.B = host_array(self.L, self.dtype)
    
    def host_version(self):
        self.host.C = self.host.A + self.host.B

if __name__ == "__main__":
    cleanup()
    va = VA()
    
    for n_b in [64, 128]:
        for n_c in [64, 128, 256]:
            va.test(vaTile, n_b=n_b, n_c=n_c, n_t=min(16, 65536 // n_b // n_c))
        va.test(vaTile, n_b=n_b, n_c=256, n_t=min(16, 256//n_b))