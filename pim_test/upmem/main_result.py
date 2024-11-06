import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from tensor import host_array
import numpy as np
import math
import argparse
from bench import *
from tuned import *
import subprocess


def get_module(op_type):
    if op_type == "mtv":
        return MTV
    elif op_type == "ttv":
        return TTV
    elif op_type == "poly_gemv1":
        return GEMV
    # elif op_type == "poly_gemv2":
    #     pass
    elif op_type == "va":
        return VA
    elif op_type == "ta":
        return TA
    elif op_type == "poly_va":
        return GEVA
    elif op_type == "poly_mixed":
        pass
    elif op_type == "dot":
        return DOT
    elif op_type == "innerprod":
        pass
    elif op_type == "mmtv":
        return MMTV
    elif op_type == "red":
        return RED
    else:
        raise Exception(f"Unknown operator type: {args.type}")


def mtvRTile(M, K, n_yb, n_t=16, n_cache=256, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, xo = sch.get_loops(block_c)
    ca = sch.cache_read(block_c, 0, "local")
    cb = sch.cache_read(block_c, 1, "local")
    cc = sch.cache_write(block_c, 0, "local")
    i, k = sch.get_loops(block_c)

    rounded = math.ceil(M / n_yb / 2) * 2
    yb, yo = sch.split(i, factors=[n_yb, rounded])
    yo, yi, yc = sch.split(yo, factors=[n_t, None, 2])

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


def mtvRCTile(M, K, n_xb, n_yb, n_t=16, n_cache=64, n_rt=1, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, k = sch.get_loops(block_c)
    xb, xo = sch.split(k, factors=[n_xb, None])
    block_crf = sch.rfactor(xb, factor_axis=0)
    ca = sch.cache_read(block_crf, 0, "local")
    cb = sch.cache_read(block_crf, 1, "local")
    cc = sch.cache_write(block_crf, 0, "local")
    i, xb, k = sch.get_loops(block_crf)
    yb, yo, yi, yc = sch.split(i, factors=[n_yb, n_t, None, 2])
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
    sch.annotate(block_c, "meta_schedule.optimization_level", 4)
    return sch


def mmtvRCTile(M, N, K, n_bb, n_xb, n_yb, n_t=16, n_cache=64, n_rt=1, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    block_c = sch.get_block("C")
    b, i, k = sch.get_loops(block_c)
    kb, ko = sch.split(k, factors=[n_xb, None])
    crf = sch.rfactor(kb, factor_axis=0)
    ca = sch.cache_read(crf, 0, "local")
    cb = sch.cache_read(crf, 1, "local")
    cc = sch.cache_write(crf, 0, "local")
    b, i, kb, ko = sch.get_loops(crf)
    bb, bo = sch.split(b, factors=[n_bb, None])
    yb, yo, yi, yc = sch.split(i, factors=[n_yb, n_t, None, 2])
    ko, ki = sch.split(ko, factors=[None, n_cache])
    sch.reorder(kb, bb, yb, bo, yo, yi, yc, ko, ki)
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
    sch.annotate(block_c, "meta_schedule.optimization_level", 4)
    return sch


def mmtvRTile(M, N, K, n_bb, n_yb, n_t=16, n_cache=256, n_rt=1, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    block_c = sch.get_block("C")
    ca = sch.cache_read(block_c, 0, "local")
    cb = sch.cache_read(block_c, 1, "local")
    cc = sch.cache_write(block_c, 0, "local")
    b, i, k = sch.get_loops(block_c)
    bb, bo = sch.split(b, factors=[n_bb, None])
    yb, yo, yi, yc = sch.split(i, factors=[n_yb, n_t, None, 2])
    ko, ki = sch.split(k, factors=[None, n_cache])
    sch.reorder(bb, yb, bo, yo, yi, yc, ko, ki)
    sch.compute_at(ca, ko)
    sch.compute_at(cb, ko)
    sch.reverse_compute_at(cc, yi)
    sch.bind(bb, "blockIdx.x")
    sch.bind(yb, "blockIdx.y")
    sch.bind(yo, "threadIdx.x")
    sch.annotate(bb, "bank", True)
    sch.annotate(yb, "bank", True)
    sch.decompose_reduction(block_c, ko)
    sch.annotate(block_c, "meta_schedule.optimization_level", 4)
    return sch


def vaTile(M, n_xb, n_t=16, n_cache=256, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_va_factory(M, dtype))
    block_c = sch.get_block("C")
    (i,) = sch.get_loops(block_c)
    ca = sch.cache_read(block_c, "A", "local")
    cb = sch.cache_read(block_c, "B", "local")
    cc = sch.cache_write(block_c, "C", "local")
    # ib, it, ii, ic = sch.split(i, factors=[n_b, n_t, None, n_c])
    bytes = np.dtype(dtype).itemsize
    ib, it = sch.split(i, factors=[n_xb, math.ceil(M / n_xb / bytes) * bytes])
    it, ii, ic = sch.split(it, factors=[n_t, None, n_cache])
    sch.compute_at(ca, ii)
    sch.compute_at(cb, ii)
    sch.reverse_compute_at(cc, ii)
    sch.bind(ib, "blockIdx.x")
    sch.bind(it, "threadIdx.x")
    # sch.parallel(ic)
    return sch


class VA(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="va",
            required=dict(),
            symbols=["A", "B", "C"],
            output_symbol="C",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array(self.M, self.dtype)
        self.host.B = host_array(self.M, self.dtype)

    def host_version(self):
        self.host.C = self.host.A + self.host.B


class GEVA(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="poly",
            required=dict(),
            symbols=["A", "B", "C", "ALPHA", "BETA"],
            output_symbol="C",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array(self.M, self.dtype)
        self.host.B = host_array(self.M, self.dtype)
        self.host.ALPHA = host_array((1,), self.dtype)
        self.host.BETA = host_array((1,), self.dtype)

    def host_version(self):
        alpha_val = self.host.ALPHA[0]
        beta_val = self.host.BETA[0]
        self.host.C = alpha_val * self.host.A + beta_val * self.host.B


class TA(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="ta",
            required=dict(),
            symbols=["A", "B", "C"],
            output_symbol="C",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, self.N, self.K), self.dtype)
        self.host.B = host_array((self.M, self.N, self.K), self.dtype)

    def host_version(self):
        self.host.C = self.host.A + self.host.B


class DOT(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="reduction",
            required=dict(),
            symbols=["A", "B"],
            output_symbol="B",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, 1), self.dtype, new=True)
        self.host.B = host_array((1,), self.dtype, new=True)

    def host_version(self):
        self.host.B = np.sum(self.host.A)


class RED(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="reduction",
            required=dict(),
            symbols=["A", "B"],
            output_symbol="B",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array((self.M,), self.dtype, new=True)
        self.host.B = host_array((1,), self.dtype, new=True)

    def h2d(self):
        self.dev.A = tvm.nd.array(self.host.A, self.target_device, symbol="A")
        self.dev.B = tvm.nd.empty((1,), self.dtype, self.target_device)

    def host_version(self):
        self.host.B = np.sum(self.host.A)


class MTV(UPMEMWorkload):
    def __init__(self, **kwargs):
        required = dict()
        super().__init__(
            profile="mtv", required=required, symbols=["A", "B", "C"], output_symbol="C", **kwargs
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, self.K), self.dtype)
        self.host.B = host_array((self.K,), self.dtype)

    def host_version(self):
        self.host.C = np.dot(self.host.A, self.host.B)


class GEMV(UPMEMWorkload):
    def __init__(self, **kwargs):
        required = dict()
        super().__init__(
            profile="poly_gemv1",
            required=required,
            symbols=["A", "B", "C", "ALPHA"],
            output_symbol="C",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, self.K), self.dtype)
        self.host.B = host_array((self.K,), self.dtype)
        self.host.ALPHA = host_array((1,), self.dtype)

    def host_version(self):
        alpha_val = self.host.ALPHA[0]
        self.host.C = alpha_val * np.dot(self.host.A, self.host.B)


class TTV(UPMEMWorkload):
    def __init__(self, **kwargs):
        required = dict()
        super().__init__(
            profile="ttv", required=required, symbols=["A", "B", "C"], output_symbol="C", **kwargs
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, self.N, self.K), self.dtype)
        self.host.B = host_array((self.K,), self.dtype)

    def host_version(self):
        self.host.C = np.einsum("mnk,k->mn", self.host.A, self.host.B)

    def benchmark_command(self, config):
        bl = int(math.log2(config["n_cache"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"make clean && NR_DPUS={config['n_xb'] * config['n_yb']} \
            NR_TASKLETS={config['n_t']} TYPE={pbtype} BL={bl} make && \
            ./bin/mtv_host -m {config['M']} -n {config['K']} -w {self.warmup} -e {self.repeat}"


class MMTV(UPMEMWorkload):
    def __init__(self, **kwargs):
        required = dict()
        super().__init__(
            profile="mmtv", required=required, symbols=["A", "B", "C"], output_symbol="C", **kwargs
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, self.N, self.K), self.dtype, intdist=2)
        self.host.B = host_array((self.M, self.K), self.dtype, intdist=2)

    def host_version(self):
        self.host.C = np.einsum("mnk,mk->mn", self.host.A, self.host.B)

    def benchmark_command(self, config):
        bl = int(math.log2(config["n_cache"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"make clean && NR_DPUS={config['n_xb'] * config['n_yb']} \
            NR_TASKLETS={config['n_t']} TYPE={pbtype} BL={bl} make && \
            ./bin/mmtv_host -m {config['M']} -n {config['K']} -w {self.warmup} -e {self.repeat}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_type", required=True, type=str)
    parser.add_argument("--schedule", required=True, type=str)
    parser.add_argument("--M", required=True, type=int)
    parser.add_argument("--N", required=True, type=int)
    parser.add_argument("--K", required=True, type=int)
    parser.add_argument("--n_xb", required=False, type=int)
    parser.add_argument("--n_yb", required=False, type=int)
    parser.add_argument("--n_cache", default=256, type=int)
    parser.add_argument("--n_t", default=16, type=int)
    parser.add_argument("--n_rt", default=1, type=int)
    parser.add_argument("--dtype", default="int32", type=str)

    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--repeat", default=100, type=int)
    parser.add_argument("--verbose", default=0, type=int)
    parser.add_argument("-compile_only", "--compile_only", default=False, action="store_true")

    args = parser.parse_args()

    # cleanup()

    op_class = get_module(
        args.op_type,
    )
    workload = op_class(
        repeat=args.repeat,
        warmup=args.warmup,
        verbose=args.verbose,
        compile_only=args.compile_only,
    )
    schedules = {
        # baseline
        "vaTile": vaTile,
        "mtvRTile": mtvRTile,
        "mtvRCTile": mtvRCTile,
        "mmtvRTile": mmtvRTile,
        "mmtvRCTile": mmtvRCTile,
        # basic
        # "va_67108864_1_1_Tuned": va_67108864_1_1_Tuned,
        # "dot_33554432_1_1_Tuned": dot_33554432_1_1_Tuned,
        # "red_33554432_1_1_Tuned": red_33554432_1_1_Tuned,
        "mtv_8192_1_8192_Tuned": mtv_8192_1_8192_Tuned,
        # "ta_256_512_512_Tuned": ta_256_512_512_Tuned,
        # "ttv_256_512_512_Tuned": ttv_256_512_512_Tuned,
        "mmtv_256_512_512_Tuned": mmtv_256_512_512_Tuned,
        # "poly_gemv1_8192_1_8192_Tuned": poly_gemv1_8192_1_8192_Tuned,
        # "poly_va_67108864_1_1_Tuned": poly_va_67108864_1_1_Tuned,
        # gpt-6b
        "mtv_12288_1_4096_Tuned": mtv_12288_1_4096_Tuned,
        "mtv_4096_1_4096_Tuned": mtv_4096_1_4096_Tuned,
        "mtv_16384_1_4096_Tuned": mtv_16384_1_4096_Tuned,
        "mtv_4096_1_16384_Tuned": mtv_4096_1_16384_Tuned,
        "mmtv_16_64_256_Tuned": mmtv_16_64_256_Tuned,
        "mmtv_16_128_256_Tuned": mmtv_16_128_256_Tuned,
        "mmtv_16_256_256_Tuned": mmtv_16_256_256_Tuned,
        "mmtv_16_512_256_Tuned": mmtv_16_512_256_Tuned,
        "mmtv_64_64_256_Tuned": mmtv_64_64_256_Tuned,
        "mmtv_64_128_256_Tuned": mmtv_64_128_256_Tuned,
        "mmtv_64_256_256_Tuned": mmtv_64_256_256_Tuned,
        "mmtv_64_512_256_Tuned": mmtv_64_512_256_Tuned,
        "mmtv_256_64_256_Tuned": mmtv_256_64_256_Tuned,
        "mmtv_256_128_256_Tuned": mmtv_256_128_256_Tuned,
        "mmtv_256_256_256_Tuned": mmtv_256_256_256_Tuned,
        "mmtv_256_512_256_Tuned": mmtv_256_512_256_Tuned,
        # gpt-30b
        "mtv_21504_1_7168_Tuned": mtv_21504_1_7168_Tuned,
        "mtv_7168_1_7168_Tuned": mtv_7168_1_7168_Tuned,
        "mtv_28672_1_7168_Tuned": mtv_28672_1_7168_Tuned,
        "mtv_7168_1_28672_Tuned": mtv_7168_1_28672_Tuned,
        "mmtv_28_64_256_Tuned": mmtv_28_64_256_Tuned,
        "mmtv_28_128_256_Tuned": mmtv_28_128_256_Tuned,
        "mmtv_28_256_256_Tuned": mmtv_28_256_256_Tuned,
        "mmtv_28_512_256_Tuned": mmtv_28_512_256_Tuned,
        "mmtv_112_64_256_Tuned": mmtv_112_64_256_Tuned,
        "mmtv_112_128_256_Tuned": mmtv_112_128_256_Tuned,
        "mmtv_112_256_256_Tuned": mmtv_112_256_256_Tuned,
        "mmtv_112_512_256_Tuned": mmtv_112_512_256_Tuned,
        "mmtv_448_64_256_Tuned": mmtv_448_64_256_Tuned,
        "mmtv_448_128_256_Tuned": mmtv_448_128_256_Tuned,
        "mmtv_448_256_256_Tuned": mmtv_448_256_256_Tuned,
        "mmtv_448_512_256_Tuned": mmtv_448_512_256_Tuned,
    }
    schedule = schedules.get(args.schedule)

    if schedule is None:
        raise ValueError(f"Schedule {args.schedule} not found")

    search_space = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    if "Tuned" in args.schedule:
        workload.test(
            schedule,
            M=args.M,
            N=args.N,
            K=args.K,
            n_xb=-1,
            n_yb=-1,
            n_t=-1,
            n_rt=-1,
            n_cache=-1,
            dtype=args.dtype,
        )
    elif args.op_type == "va":
        for n_xb in search_space:
            if n_xb < 256:
                continue
            workload.test(
                schedule,
                M=args.M,
                N=args.N,
                K=args.K,
                n_xb=n_xb,
                n_yb=-1,
                n_t=args.n_t,
                n_rt=args.n_rt,
                n_cache=args.n_cache,
                dtype=args.dtype,
            )
    elif args.op_type == "mtv":
        if "RCTile" in args.schedule:
            for n_xb in search_space:
                for n_yb in search_space:
                    if n_xb < 4 or n_yb < 64:
                        continue
                    n_dpu = n_xb * n_yb
                    if n_dpu < 256 or n_dpu > 2048:
                        continue
                    workload.test(
                        schedule,
                        M=args.M,
                        N=args.N,
                        K=args.K,
                        n_xb=n_xb,
                        n_yb=n_yb,
                        n_t=args.n_t,
                        n_rt=args.n_rt,
                        n_cache=args.n_cache,
                        dtype=args.dtype,
                    )
        elif "RTile" in args.schedule:
            for n_yb in search_space:
                if n_yb < 256:
                    continue
                workload.test(
                    schedule,
                    M=args.M,
                    N=args.N,
                    K=args.K,
                    n_xb=-1,
                    n_yb=n_yb,
                    n_t=args.n_t,
                    n_rt=args.n_rt,
                    n_cache=args.n_cache,
                    dtype=args.dtype,
                )
    elif args.op_type == "mmtv":
        if "RCTile" in args.schedule:
            i = 0
            for n_bb in search_space:
                for n_xb in search_space:
                    for n_yb in search_space:
                        n_dpu = n_bb * n_xb * n_yb
                        LB = 256
                        if n_dpu < LB or n_dpu > 2048:
                            continue
                        if args.K // n_xb < 64:
                            continue
                        if n_bb >= args.M or n_yb >= args.N or n_xb >= args.K:
                            continue
                        workload.test(
                            schedule,
                            M=args.M,
                            N=args.N,
                            K=args.K,
                            n_xb=n_xb,
                            n_yb=n_yb,
                            n_bb=n_bb,
                            n_t=args.n_t,
                            n_rt=args.n_rt,
                            n_cache=args.n_cache,
                            dtype=args.dtype,
                        )
                        i += 1
            print(i)
        elif "RTile" in args.schedule:
            i = 0
            for n_bb in search_space:
                for n_yb in search_space:
                    n_dpu = n_bb * n_yb
                    LB = 256
                    if n_dpu < LB or n_dpu > 2048:
                        continue
                    if n_bb >= args.M or n_yb >= args.N:
                        continue
                    workload.test(
                        schedule,
                        M=args.M,
                        N=args.N,
                        K=args.K,
                        n_xb=-1,
                        n_yb=n_yb,
                        n_bb=n_bb,
                        n_t=args.n_t,
                        n_rt=args.n_rt,
                        n_cache=args.n_cache,
                        dtype=args.dtype,
                    )
                    i += 1
            print(i)
