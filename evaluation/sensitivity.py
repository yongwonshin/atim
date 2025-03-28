import tvm
import numpy as np
import math

import pandas as pd
from workloads import MTV, VA
from bench import upmem_mtv_factory, upmem_va_factory


def gemvRCTile(M, K, n_xb, n_yb, n_yt=16, n_cache=64, n_rt=16, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
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
    it, _ = sch.split(i, factors=[n_rt, None])
    sch.parallel(it)
    return sch


def vaTile(M, n_b, n_t, n_c, dtype):
    sch = tvm.tir.Schedule(upmem_va_factory(M, dtype))
    block_c = sch.get_block("C")
    (i,) = sch.get_loops(block_c)
    ca = sch.cache_read(block_c, "A", "local")
    cb = sch.cache_read(block_c, "B", "local")
    cc = sch.cache_write(block_c, "C", "local")
    bytes = np.dtype(dtype).itemsize
    ib, it = sch.split(i, factors=[n_b, math.ceil(M / n_b / bytes) * bytes])
    it, ii, ic = sch.split(it, factors=[n_t, None, n_c])
    sch.compute_at(ca, ii)
    sch.compute_at(cb, ii)
    sch.reverse_compute_at(cc, ii)
    sch.bind(ib, "blockIdx.x")
    sch.bind(it, "threadIdx.x")
    return sch

workload_configs = dict(
    repeat=100,
    warmup=10,
    verbose=-1,
    ignore_wrong=True,
)

mtv = MTV(**workload_configs)
va = VA(**workload_configs)

def sens_mtv(m, k):
    option = dict(M=m, K=k, dtype="int32", n_xb=1, n_yb=1, n_cache=32, n_yt=16, n_rt=16)
    prim = mtv.benchmark(**option)
    res = [float(prim[1])]
    for opt in [0, 1, 2, 4]:
        mtv.opt_level = opt
        t = mtv.test(gemvRCTile, **option)
        if t == "ERROR" or t == "WRONG":
            t = 0
        res.append(max(0.001, float(t)))
    print(f"{m}\t{k}\t" + "\t".join([f"{t:.4f}" for t in res]))
    return res


def sens_va(l):
    option = dict(M=l, n_b=32, n_t=16, n_c=64, dtype="int32")
    prim = va.benchmark(**option)
    res = [float(prim[1])]
    for opt in [0, 1, 2, 4]:
        va.opt_level = opt
        t = va.test(vaTile, **option)
        if t == "ERROR" or t == "WRONG":
            t = 0
        res.append(max(0.001, float(t)))
    print(f"{l}\t1\t" + "\t".join([f"{t:.4f}" for t in res]))
    return res


if __name__ == "__main__":
    df = pd.read_csv("./graph/result_opt_template.csv")
    mtv_var_dims = [72, 91, 123, 145, 164, 196, 212, 245]
    mtv_var_dims = [196]
    va_dims = [i * 100000 for i in [1, 2, 3, 4, 5, 6, 7, 8]]

    print("\nMTV 256 x L\nWorkload\t M\tK\tPrIM\tO0\tO1\tO2\tO4")
    for i, l in enumerate(mtv_var_dims):
        df.iloc[i, 2:] = sens_mtv(256, l)

    print("\nMTV L x 256\nM\tK\tPrIM\tO0\tO1\tO2\tO4")
    for i, l in enumerate(mtv_var_dims):
        df.iloc[8 + i, 2:] = sens_mtv(l, 256)

    print("\nMTV L x L\nM\tK\tPrIM\tO0\tO1\tO2\tO4")
    for i, l in enumerate(mtv_var_dims):
        df.iloc[16 + i, 2:] = sens_mtv(l, l)

    print("\nVA\nL\t-\tPrIM\tO0\tO1\tO2\tO4")
    for i, l in enumerate(va_dims):
        df.iloc[24 + i, 2:] = sens_va(l)

    df.to_csv("./graph/result_opt.csv", index=False)

    # 15분