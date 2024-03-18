import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from tensor import host_array
from tvm.target import Target
from tvm.tir.transform import *


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
    (i,) = sch.get_loops(block_c)
    ca = sch.cache_read(block_c, "A", "local")
    cb = sch.cache_read(block_c, "B", "local")
    cc = sch.cache_write(block_c, "C", "local")
    ib, it, ii, ic = sch.split(i, factors=[n_b, n_t, None, n_c])
    # n_b = 16, n_t = 16, n_c = 256
    sch.compute_at(ca, ii)
    sch.compute_at(cb, ii)
    sch.reverse_compute_at(cc, ii)
    sch.bind(ib, "blockIdx.x")
    sch.bind(it, "threadIdx.x")
    sch.parallel(ic)
    return sch


class VA(UPMEMWorkload):
    def __init__(self):
        super().__init__(
            profile="va",
            required=dict(L=65536, dtype="int32", n_b=4, n_t=16, n_c=256),
            symbols=["A", "B", "C"],
            output_symbol="C",
        )

    def fetch_data(self):
        self.host.A = host_array(self.L, self.dtype)
        self.host.B = host_array(self.L, self.dtype)

    def host_version(self):
        self.host.C = self.host.A + self.host.B


if __name__ == "__main__":
    cleanup()
    va = VA()
    # va.test(vaTile, n_b=64, n_c=256, n_t=4)

    for n_b, n_t, n_c in [
        (1, 16, 256),
        (2, 16, 256),
        (4, 16, 256),
        (8, 16, 256),
        (16, 16, 256),
        (32, 8, 256),
        (64, 4, 256),
        (128, 2, 256),
        (256, 1, 256),
        (512, 1, 128),
        (1024, 1, 64),
        (2048, 1, 16),
        (64, 16, 64),
        (64, 8, 128),
        (64, 4, 256),
        (128, 8, 64),
        (128, 4, 128),
        (128, 2, 256),
    ]:
        va.test(vaTile, n_b=n_b, n_c=n_c, n_t=n_t)
