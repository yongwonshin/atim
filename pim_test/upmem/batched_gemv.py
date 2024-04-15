import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from tensor import host_array
import numpy as np
import math
import argparse


def bgemv_factory(N, M, K, dtype):
    @tvm.script.ir_module
    class BGEMVModule:
        @T.prim_func
        def main(
            a: T.handle,
            b: T.handle,
            c: T.handle,
        ):
            # We exchange data between function by handles, which are similar to pointer.
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": T.bool(True), "pragma_explicit_h2d": ["A"]}
            )
            # Create buffer from handles.
            A = T.match_buffer(a, (N, M, K), dtype=dtype)
            B = T.match_buffer(b, (N, K), dtype=dtype)
            C = T.match_buffer(c, (N, M), dtype=dtype)

            for n, i, k in T.grid(N, M, K):
                with T.block("C"):
                    v_n, v_i, v_k = T.axis.remap("SSR", [n, i, k])
                    with T.init():
                        C[v_n, v_i] = 0
                    C[v_n, v_i] = C[v_n, v_i] + A[v_n, v_i, v_k] * B[v_n, v_k]

    return BGEMVModule


def bgemvTile(N, M, K, n_bb, n_xb, n_yb, n_yt=16, n_cache=64, n_rt=128, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(bgemv_factory(N, M, K, dtype))
    block_c = sch.get_block("C")
    b, i, k = sch.get_loops(block_c)
    kb, ko = sch.split(k, factors=[n_xb, None])
    crf = sch.rfactor(kb, factor_axis=0)
    ca = sch.cache_read(crf, 0, "local")
    cb = sch.cache_read(crf, 1, "local")
    cc = sch.cache_write(crf, 0, "local")
    b, i, kb, ko = sch.get_loops(crf)
    bb, bo = sch.split(b, factors=[n_bb, None])
    yb, yo, yi, yc = sch.split(i, factors=[n_yb, n_yt, None, 2])
    ko, ki = sch.split(ko, factors=[None, n_cache])
    sch.reorder(bb, kb, yb, bo, yo, yi, yc, ko, ki)
    sch.compute_at(ca, ko)
    sch.compute_at(cb, ko)
    sch.reverse_compute_at(cc, yi)
    sch.bind(bb, "blockIdx.x")
    sch.bind(kb, "blockIdx.y")
    sch.bind(yb, "blockIdx.z")

    sch.bind(yo, "threadIdx.x")
    sch.annotate(kb, "bank", True)
    sch.annotate(yb, "bank", True)
    sch.annotate(bb, "bank", True)
    sch.decompose_reduction(crf, ko)
    return sch


class BGEMV(UPMEMWorkload):
    def __init__(self, **kwargs):
        required = dict(
            N=16, M=256, K=256, dtype="int32", n_bb=1, n_xb=1, n_yb=1, n_cache=64, n_yt=16, n_rt=64
        )
        super().__init__(
            profile="bgemv", required=required, symbols=["A", "B", "C"], output_symbol="C", **kwargs
        )

    def fetch_data(self):
        self.host.A = host_array((self.N, self.M, self.K), self.dtype, intdist=2)
        self.host.B = host_array((self.N, self.K), self.dtype, intdist=2)

    def host_version(self):
        self.host.C = np.einsum("nmk,nk->nm", self.host.A, self.host.B)

    def benchmark_command(self, config):
        bl = int(math.log2(config["n_cache"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"make clean && NR_DPUS={config['n_xb'] * config['n_yb']} \
            NR_TASKLETS={config['n_yt']} TYPE={pbtype} BL={bl} make && \
            ./bin/gemv_host -m {config['M']} -n {config['K']} -w {self.warmup} -e {self.repeat}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--N", default=16, type=int)
    parser.add_argument("-m", "--M", default=256, type=int)
    parser.add_argument("-k", "--K", default=256, type=int)
    parser.add_argument("-dtype", "--dtype", default="int32", type=str)
    parser.add_argument("-bb", "--n_bb", default=4, type=int)
    parser.add_argument("-xb", "--n_xb", default=32, type=int)
    parser.add_argument("-yb", "--n_yb", default=16, type=int)
    parser.add_argument("-c", "--n_cache", default=64, type=int)
    parser.add_argument("-yt", "--n_yt", default=16, type=int)
    parser.add_argument("-rt", "--n_rt", default=64, type=int)

    parser.add_argument("-w", "--warmup", default=1, type=int)
    parser.add_argument("-e", "--repeat", default=3, type=int)
    parser.add_argument("-v", "--verbose", default=0, type=int)
    parser.add_argument("-bench", "--bench", default=False, action="store_true")
    parser.add_argument("-custom", "--custom", default=False, action="store_true")
    parser.add_argument("-compile_only", "--compile_only", default=False, action="store_true")

    args = parser.parse_args()

    cleanup()
    gemv = BGEMV(
        repeat=args.repeat,
        warmup=args.warmup,
        bench=args.bench,
        verbose=args.verbose,
        compile_only=args.compile_only,
    )

    if not args.custom:
        config = gemv.extract_config(args)
        gemv.benchmark(**config)
        gemv.test(bgemvTile, **config)
    else:  # custom test
        configs = [(128, 480, 256, 16, 8, 4, 16, 64)]

        for n, m, k, bb, xb, yb, yt, cache in configs:
            gemv.benchmark(N=n, M=m, K=k, n_bb=bb, n_xb=xb, n_yb=yb, n_yt=yt, n_cache=cache)
        for n, m, k, bb, xb, yb, yt, cache in configs:
            gemv.test(bgemvTile, N=n, M=m, K=k, n_bb=bb, n_xb=xb, n_yb=yb, n_yt=yt, n_cache=cache)
16
