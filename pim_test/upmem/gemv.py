import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from tensor import host_array
import numpy as np
import math
import argparse


def upmem_gemv_factory(M, K, dtype):
    @tvm.script.ir_module
    class UPMEMModule:
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

    return UPMEMModule


def gemvRCTile(M, K, n_xb, n_yb, n_yt=16, n_cache=64, n_rt=16, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_gemv_factory(M, K, dtype))
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


def gemvRTile(M, K, n_yb, n_cache=64, n_yt=16, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_gemv_factory(M, K, dtype))
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
    sch = tvm.tir.Schedule(upmem_gemv_factory(M, K, dtype))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--schedule", default="gemvRTile", type=str)
    parser.add_argument("-m", "--M", default=8192, type=int)
    parser.add_argument("-k", "--K", default=8192, type=int)
    parser.add_argument("-dtype", "--dtype", default="int32", type=str)
    parser.add_argument("-xb", "--n_xb", default=16, type=int)
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
    gemv = GEMV(
        repeat=args.repeat,
        warmup=args.warmup,
        bench=args.bench,
        verbose=args.verbose,
        compile_only=args.compile_only,
    )

    schedules = {
        "gemvRTile": gemvRTile,
        "gemvRCTile": gemvRCTile,
        "StridedBankTile": StridedBankTile,
        "HBMStyleTile": HBMStyleTile,
    }
    schedule = schedules.get(args.schedule)
    if schedule is None:
        raise ValueError(f"Schedule {args.schedule} not found")

    if not args.custom:
        config = gemv.extract_config(args)
        gemv.benchmark(**config)
        gemv.test(schedule, **config)
    else:  # custom test
        # dims = [
        #     # (12288, 4096),
        #     # (4096, 4096),
        #     # (16384, 4096),
        #     # (4096, 16384),
        #     # (15360, 5120),
        #     # (5120, 5120),
        #     # (20480, 5120),
        #     # (5120, 20480),
        #     # (21504, 7168),
        #     # (7168, 7168),
        #     (28672, 7168),
        #     (7168, 28672),
        #     # (36864, 12288),
        #     # (12288, 12288),
        #     # (49152, 12288),
        #     # (12288, 49152),
        # ]
        # for m, k in dims:
        #     print(m, k)
        #     for xb, yb in (
        #         (1, 2048),
        #         (2, 1024),
        #         (4, 512),
        #         (8, 256),
        #         (16, 128),
        #         (32, 64),
        #         (64, 32),
        #         (128, 16),
        #     ):
        #         for c in (16, 32, 64):
        #             if (k // xb) < c:
        #                 continue
        #             if xb == 1:
        #                 gemv.test(gemvRTile, M=m, K=k, n_yb=yb, n_yt=16, n_rt=16, n_cache=c)
        #             else:
        #                 gemv.test(
        #                     gemvRCTile, M=m, K=k, n_xb=xb, n_yb=yb, n_yt=16, n_rt=16, n_cache=c
        #                 )
        #     gemv.dump_handtune_max()
        #     print()

        configs = [
            # (8192, 1024, 1, 64, 16, 256, 1),
            # (28672, 7168, 1, 2048, 16, 256, 16),
            # (28672, 7168, 2, 1024, 16, 256, 16),
            # (28672, 2048, 4, 512, 16, 256, 16), #
            # (12288, 4096, 8, 128, 16, 256, 16),
            # (49152, 12288, 1, 2048, 16, 256, 16),
            # (4096, 4096, 2, 512, 16, 128, 16),
            # (4096, 4092, 2, 32, 16, 128, 16),
            # (49152, 12256, 2, 1024, 16, 32, 16),
            # (49152, 12288, 4, 512, 16, 256, 16),
            # (49152, 12288, 8, 256, 16, 256, 16),
            # (49152, 12288, 16, 128, 16, 128, 16),
            # (163840, 4096, 1, 2048, 16, 128, 16),
            # (12288, 4096, 16, 128, 16, 128, 16),
            # (4096, 4096, 16, 128, 16, 128, 16),
            # (16384, 4096, 8, 256, 16, 128, 16),
            # (4096, 16384, 32, 64, 16, 128, 16),
            # (163840, 4096, 1, 2048, 16, 64, 16),
            # (15360, 5120, 8, 256, 16, 64, 16),
            # (5120, 5120, 32, 64, 16, 64, 16),
            # (20480, 5120, 8, 256, 16, 64, 16),
            # (5120, 20480, 64, 32, 16, 64, 16),
            # (21504, 7168, 64, 32, 16, 64, 16),
            # (7168, 7168, 8, 256, 16, 64, 16),
            # (28672, 7168, 64, 32, 16, 64, 16),
            # (7168, 28672, 128, 16, 16, 64, 16),
            # (36864, 12288, 16, 128, 16, 64, 16),
            # (12288, 12288, 16, 128, 16, 64, 16),
            # (49152, 12288, 16, 128, 16, 64, 16),
            (12288, 49152, 64, 32, 16, 64, 16),
        ]

        configs = [
            (1024, 200, 1, 16, 8, 64, 16),
            #(1024, 1000, 2, 16, 16, 256, 16)
        ]

        # for m, k, xb, yb, cache in configs:
        #     gemv.test(gemvRCTile, M=m, K=k, n_xb=xb, n_yb=yb, n_cache=cache, n_yt=16, n_rt=16)

        for m, k, xb, yb, yt, cache, rt in configs:
            gemv.benchmark(M=m, K=k, n_xb=xb, n_yb=yb, n_yt=yt, n_rt=rt, n_cache=cache)
        for m, k, xb, yb, yt, cache, rt in configs:
            if xb == 1:
                gemv.test(gemvRTile, M=m, K=k, n_yb=yb, n_yt=yt, n_rt=rt, n_cache=cache)
            else:
                gemv.test(gemvRCTile, M=m, K=k, n_xb=xb, n_yb=yb, n_yt=yt, n_rt=rt, n_cache=cache)
