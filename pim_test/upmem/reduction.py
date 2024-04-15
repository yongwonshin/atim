import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from tensor import host_array
import numpy as np
import math
import argparse


def red_prim_schedule(L, dtype):
    M = 1

    @tvm.script.ir_module
    class ReductionModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": T.bool(True), "pragma_explicit_h2d": ["A"]}
            )
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
    brf = sch.rfactor(ib, factor_axis=0)  # C_rf
    _, it, _, _ = sch.get_loops(brf)
    trf = sch.rfactor(it, factor_axis=0, mem_scope="shared")  # C_rf_rf
    ca = sch.cache_read(trf, 0, "local")
    cc = sch.cache_write(trf, 0, "local")
    tib, tit, tii, _ = sch.get_loops(trf)
    tii, _ = sch.split(tii, factors=[None, n_c])
    sch.compute_at(ca, tii)
    sch.reverse_compute_at(cc, tii)
    sch.reverse_compute_at(brf, tit)
    sch.bind(tib, "blockIdx.x")
    sch.bind(tit, "threadIdx.x")
    # sch.annotate(sch.get_block("A_local"), "pragma_explicit_h2d", True)
    sch.decompose_reduction(trf, tii)
    return sch


class REDUCE(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="reduction",
            required=dict(L=8388608, dtype="int64", n_b=1024, n_t=16, n_c=64),
            symbols=["A", "B"],
            output_symbol="B",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array((self.L, 1), self.dtype, new=True)
        self.host.B = host_array((1,), self.dtype, new=True)

    def host_version(self):
        self.host.B = np.sum(self.host.A)

    def h2d(self):
        self.dev.A = tvm.nd.array(self.host.A, self.target_device, symbol="A")
        self.dev.B = tvm.nd.empty((1,), self.dtype, self.target_device)

    def benchmark_command(self, config):
        bl = int(math.log2(config["n_c"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"make clean && NR_DPUS={config['n_b']} \
            NR_TASKLETS={config['n_t']} TYPE={pbtype} BL={bl} VERSION=HANDSHAKE make && \
            ./bin/host_code -i {config['L']} -w {self.warmup} -e {self.repeat}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--schedule", default="gemvRTile", type=str)
    parser.add_argument("-i", "--L", default=8388608, type=int)
    parser.add_argument("-dtype", "--dtype", default="int64", type=str)
    parser.add_argument("-b", "--n_b", default=1024, type=int)
    parser.add_argument("-c", "--n_c", default=64, type=int)
    parser.add_argument("-t", "--n_t", default=16, type=int)

    parser.add_argument("-w", "--warmup", default=1, type=int)
    parser.add_argument("-e", "--repeat", default=3, type=int)
    parser.add_argument("-v", "--verbose", default=0, type=int)
    parser.add_argument("-bench", "--bench", default=False, action="store_true")
    parser.add_argument("-custom", "--custom", default=False, action="store_true")
    parser.add_argument("-compile_only", "--compile_only", default=False, action="store_true")

    args = parser.parse_args()

    cleanup()
    reduce = REDUCE(
        repeat=args.repeat,
        warmup=args.warmup,
        bench=args.bench,
        verbose=args.verbose,
        compile_only=args.compile_only,
    )

    if not args.custom:
        config = reduce.extract_config(args)
        reduce.benchmark(**config)
        reduce.test(crossReduction, **config)
    else:  # custom test config
        #
        configs = [
            (6553600, 1, 16, 128, "int64"),
            (6553600 * 4, 4, 16, 128, "int64"),
            (6553600 * 16, 16, 16, 128, "int64"),
            (6553600 * 64, 64, 16, 128, "int64"),
            (6553600, 1, 16, 128, "int64"),
            (6553600, 4, 16, 128, "int64"),
            (6553600, 16, 16, 128, "int64"),
            (6553600, 64, 16, 128, "int64"),
            (400000000, 256, 16, 128, "int64"),
            (400000000, 512, 16, 128, "int64"),
            (400000000, 1024, 16, 128, "int64"),
            (400000000, 2048, 16, 128, "int64"),
        ]
        for L, n_b, n_t, n_c, dtype in configs:
            reduce.benchmark(L=L, n_b=n_b, n_t=n_t, n_c=n_c, dtype=dtype)
        for L, n_b, n_t, n_c, dtype in configs:
            reduce.test(crossReduction, L=L, n_b=n_b, n_t=n_t, n_c=n_c, dtype=dtype)
