import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from tensor import host_array
import numpy as np
import math
import argparse
from tqdm import tqdm


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
    yb, yo = sch.split(i, factors=[n_yb, None])
    yo, yi, yc = sch.split(yo, factors=[n_yt, None, 8 // np.dtype(dtype).itemsize])
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

def mtvRCTile(M, K, n_xb, n_yb, n_yt=16, n_cache=256, n_rt=1, dtype="int32", **kwargs):
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
    sch.annotate(block_c, "meta_schedule.optimization_level", 0)
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
    def __init__(self, profile="gemv", **kwargs):
        required = dict(M=8192, K=8192, dtype="int32", n_xb=1, n_yb=1, n_cache=64, n_yt=16, n_rt=64)
        super().__init__(
            profile=profile,
            required=required,
            symbols=["A", "B", "C"],
            output_symbol="C",
            **kwargs
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


def bench_handtune():
    for m, k in [
        (12288, 4096),
        (4096, 4096),
        (16384, 4096),
        (4096, 16384),
        (21504, 7168),
        (7168, 7168),
        (28672, 7168),
        (7168, 28672)
    ]:
        print(m, k)
        configs = [
            (m, k, 1, d, t, c, 1, "int32")
                for d in [256, 512, 1024, 1536, 2048]
                for t in [16, 20, 24]
                for c in [8, 16, 32, 64, 128, 256]
        ]

        max_time = 1e9
        with tqdm(total=len(configs), leave=True) as pbar:
            for m, k, xb, yb, yt, cache, rt, dtype in configs:
                try:
                    tuples = gemv.benchmark(M=m, K=k, n_xb=xb, n_yb=yb, n_yt=yt, n_rt=rt, n_cache=cache, dtype=dtype)
                    total_time = tuples[0] + tuples[1] + tuples[2]
                    if total_time < max_time:
                        max_time = total_time
                        best_config = (m, k, xb, yb, yt, cache, rt, dtype)
                except ValueError as e:
                    tuples = ("wrong", "", "")
                except RuntimeError as e:
                    tuples = ("fail", "", "")
                except TimeoutError as e:
                    tuples = ("timeout", "", "")
                tqdm.write("\t".join([str(x) for x in tuples] + [str(m), str(k), str(yb), str(yt), str(cache), dtype]))
                pbar.update(1)

        print(f"Best config: {best_config} with {max_time} ms")
        print()

def handtune():
    dims = [
        (12288, 4096),
        (4096, 4096),
        (16384, 4096),
        (4096, 16384),
        (21504, 7168),
        (7168, 7168),
        (28672, 7168),
        (7168, 28672)
    ]

    gemv.use_time_evaluator = False
    records = []

    for M, K in dims:
        print(M, K)
        for by in [512, 1024, 1536, 2048]:
            for c in [16, 32, 64, 128, 256]:
                gemv.test(
                    gemvRCTile,
                    M=M,
                    K=K,
                    n_xb=1,
                    n_yb=by,
                    n_cache=c,
                    n_yt=16,
                    n_rt=16,
                    dtype="int32",
                )
        records.append(gemv.dump_handtune_max())

    gemv.repeat = 1000
    gemv.use_time_evaluator = True
    for conf in records:
        gemv.test(gemvRCTile, **conf)

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

    gemv.print_header()

    if not args.custom:
        config = gemv.extract_config(args)
        if args.bench:
            gemv.benchmark(**config)
        gemv.test(schedule, **config)

    else:  # custom test



        confs = [
            {'M': 12288, 'K': 4096, 'dtype': 'int32', 'n_xb': 1, 'n_yb': 512, 'n_cache': 16, 'n_yt': 16, 'n_rt': 16},
            {'M': 4096, 'K': 4096, 'dtype': 'int32', 'n_xb': 1, 'n_yb': 512, 'n_cache': 16, 'n_yt': 16, 'n_rt': 16},
            {'M': 16384, 'K': 4096, 'dtype': 'int32', 'n_xb': 1, 'n_yb': 1024, 'n_cache': 256, 'n_yt': 16, 'n_rt': 16},
            {'M': 4096, 'K': 16384, 'dtype': 'int32', 'n_xb': 1, 'n_yb': 1024, 'n_cache': 16, 'n_yt': 16, 'n_rt': 16},
            {'M': 21504, 'K': 7168, 'dtype': 'int32', 'n_xb': 1, 'n_yb': 1536, 'n_cache': 128, 'n_yt': 16, 'n_rt': 16},
            {'M': 7168, 'K': 7168, 'dtype': 'int32', 'n_xb': 1, 'n_yb': 512, 'n_cache': 16, 'n_yt': 16, 'n_rt': 16},
            {'M': 28672, 'K': 7168, 'dtype': 'int32', 'n_xb': 1, 'n_yb': 2048, 'n_cache': 16, 'n_yt': 16, 'n_rt': 16},
            {'M': 7168, 'K': 28672, 'dtype': 'int32', 'n_xb': 1, 'n_yb': 512, 'n_cache': 16, 'n_yt': 16, 'n_rt': 16},
        ]
        gemv.use_time_evaluator = False

        for c in confs:
            gemv.test(schedule, **c)
        #handtune()
        # config = []

        # for bm, bk in [(16, 128)]:
        #     for c in [32]:
        #         config.append((766, 493, bm, bk, 16, c, 16, "int32"))

        # for m, k, xb, yb, yt, cache, rt, dtype in config:
        #     gemv.test(gemvRCTile, M=m, K=k, n_xb=xb, n_yb=yb, n_yt=yt, n_rt=rt, n_cache=cache, dtype=dtype)

        # gemv.opt_level = 3

        # for m, k, xb, yb, yt, cache, rt, dtype in config:
        #     gemv.test(gemvRCTile, M=m, K=k, n_xb=xb, n_yb=yb, n_yt=yt, n_rt=rt, n_cache=cache, dtype=dtype)

        # for m, k, xb, yb, cache in configs:
        #     gemv.test(gemvRCTile, M=m, K=k, n_xb=xb, n_yb=yb, n_cache=cache, n_yt=16, n_rt=16)

        # for m, k, xb, yb, yt, rt, cache, dtype in configs:
        #     # if xb == 1:
        #     #     gemv.test(gemvRTile, M=m, K=k, n_yb=yb, n_yt=yt, n_rt=rt, n_cache=cache)
        #     # else:
        #     with tvm.transform.PassContext(config={"tir.UpmemKernelOptimize": 4 }):
        #         gemv.test(gemvRCTile, M=m, K=k, n_xb=xb, n_yb=yb, n_yt=yt, n_rt=rt, n_cache=cache, dtype=dtype)
