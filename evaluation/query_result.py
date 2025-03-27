import logging
import tempfile

import numpy as np
import pytest
import tvm
import sys
import time
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.local_rpc import LocalRPC
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule
from typing import Callable
import os
from tvm import te, runtime, topi, tir
from tvm.meta_schedule import Database
from tvm.meta_schedule.database import JSONDatabase
from tvm.script import ir as I
from tvm.script import tir as T
import gc
from tqdm import tqdm
import multiprocessing as mp
import flush

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", default="bench_autotuner_result", type=str)
parser.add_argument("--only_show", action="store_true")
parser.add_argument("--only_run", action="store_true")
parser.add_argument("-N", type=int, default=1048576)
args = parser.parse_args()

target = Target("llvm")

N = args.N


@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((N,), "int32"), B: T.Buffer((N,), "int32"), C: T.Buffer((N,), "int32")):
        T.func_attr(
            {
                "global_symbol": "main",
                "pragma_explicit_h2d": ["A", "B"],
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        for i in range(N):
            with T.block("C"):
                v_i = T.axis.spatial(N, i)
                T.reads(A[v_i], B[v_i])
                T.writes(C[v_i])
                C[v_i] = A[v_i] + B[v_i]


def flush_worker(dummy_size):
    dummy_arr = np.random.randint(0, 50, size=dummy_size, dtype="int32")
    sum_dummy = 0
    length = len(dummy_arr)
    for i in tqdm(range(0, length, 2)):
        sum_dummy += dummy_arr[i]

def flush_cache(dummy_size=2**24, num_processes=24):
    procs = []
    for _ in range(num_processes):
        p = mp.Process(target=flush_worker, args=(dummy_size,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
    b0 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.parallel", ann_val=1536)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.vectorize", ann_val=64)
    sch.enter_postproc()
    b1 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b1, ann_key="meta_schedule.parallel")
    sch.unannotate(block_or_loop=b1, ann_key="meta_schedule.vectorize")
    (b2,) = sch.get_child_blocks(b1)
    (l3,) = sch.get_loops(block=b2)
    l4 = sch.fuse(l3, preserve_unit_iters=True)
    l5, l6 = sch.split(loop=l4, factors=[16384, None], preserve_unit_iters=True)
    sch.parallel(loop=l5)
    sch.vectorize(loop=l6)


def ndarray(size, dtype="int32"):
    arr = np.random.randint(0, 50, size=size, dtype=dtype)
    ret = tvm.nd.array(arr)
    flush.flush_cache_clflush(arr)
    return ret


def ndzeros(size, dtype="int32"):
    arr = np.zeros(size, dtype=dtype)
    ret = tvm.nd.array(arr)
    flush.flush_cache_clflush(arr)
    return ret


def query(workdir: str, only_show: False, only_run: False, manual: False) -> None:
    if not workdir:
        print("")
        return
    parsed = workdir.split("/")[-1].split("_")
    comp = parsed[1]
    m, n, k = int(parsed[2]), int(parsed[3]), int(parsed[4])
    # dtype = "int32"
    # if "red" in workdir or "dot" in workdir:
    #     dtype = "int64"

    database = JSONDatabase(work_dir=workdir)
    all_records = database.get_all_tuning_records()
    top_record = sorted(all_records, key=lambda rec: rec.run_secs[0])[0]
    assert len(top_record.run_secs) == 1
    mod = top_record.workload.mod
    schedule = ms.tir_integration.compile_tir(database, mod, target)
    if not only_run:
        schedule.trace.show()
    mod = schedule.mod
    l = tvm.lower(mod)
    f = tvm.build(schedule.mod, target=target)
    evaluator = f.time_evaluator(f.entry_name, tvm.cpu(0), number=1, repeat=100)

    times = []
    print(workdir)
    for _ in range(1):
        dlist = []
        if comp == "va":
            a = ndarray((m,))
            b = ndarray((m,))
            c = ndzeros((m,))
            t = evaluator(a, b, c)
            times.append(t.median * 1000)
            times.append(t.mean * 1000)

        if comp == "mmtv":
            a = ndarray((m, n, k))
            b = ndarray((m, k))
            c = ndzeros((m, n))
            print("Eval Start")
            t = evaluator(a, b, c)
            times.append(t.median * 1000)
            times.append(t.mean * 1000)

        if comp == "mtv":
            a = ndarray((m, k))
            b = ndarray((k,))
            c = ndzeros((m,))
            t = evaluator(a, b, c)
            times.append(t.median * 1000)
            times.append(t.mean * 1000)

        if comp == "red":
            a = ndarray((m,), dtype=np.int64)
            c = ndzeros((1,), dtype=np.int64)
            t = evaluator(a, c)
            times.append(t.median * 1000)
            times.append(t.mean * 1000)

        if comp == "ttv":
            a = ndarray((m, n, k))
            b = ndarray((k,))
            c = ndzeros((m, n))
            t = evaluator(a, b, c)
            times.append(t.median * 1000)
            times.append(t.mean * 1000)

        if comp == "polyva":
            a = ndarray((m,))
            b = ndarray((m,))
            alpha = ndarray((1,))
            beta = ndarray((1,))
            c = ndzeros((m,))
            t = evaluator(a, b, c, alpha, beta)
            times.append(t.median * 1000)
            times.append(t.mean * 1000)

        if comp == "polygemv1":
            a = ndarray((m, k))
            b = ndarray((k,))
            alpha = ndarray((1,))
            c = ndzeros((m,))
            t = evaluator(a, b, c, alpha)
            times.append(t.median * 1000)
            times.append(t.mean * 1000)

        a, b, c, alpha, beta, t = None, None, None, None, None, None
        gc.collect()

    # print(workdir)
    print(times)


if __name__ == "__main__":
    # for workdir in [
        # "./isca_mmtv_16_64_256_rev",
        # "./isca_mmtv_16_128_256_rev",
        # "./isca_mmtv_16_256_256_rev",
        # "./isca_mmtv_16_512_256_rev",
        # "", "", "", "",
        # "./isca_mmtv_64_64_256_rev",
        # "./isca_mmtv_64_128_256_rev",
        # "./isca_mmtv_64_256_256_rev",
        # "./isca_mmtv_64_512_256_rev",
        # "", "", "", "",
        # "./isca_mmtv_28_64_256_rev",
        # "./isca_mmtv_28_128_256_rev",
        # "./isca_mmtv_28_256_256_rev",
        # "./isca_mmtv_28_512_256_rev",
        # "", "", "", "",
        # "./isca_mmtv_112_64_256_rev",
        # "./isca_mmtv_112_128_256_rev",
        # "./isca_mmtv_112_256_256_rev",
        # "./isca_mmtv_112_512_256_rev",
        # "", "", "", "",

        # "./isca_mtv_12288_1_4096_rev",
        # "./isca_mtv_4096_1_4096_rev",
        # "./isca_mtv_16384_1_4096_rev",
        # "./isca_mtv_4096_1_16384_rev",
        # "./isca_mtv_21504_1_7168_rev",
        # "./isca_mtv_7168_1_7168_rev",
        # "./isca_mtv_28672_1_7168_rev",
        # "./isca_mtv_7168_1_28672_rev",

        # "./isca_va_67108864_1_1_rev",
        # "./isca_red_33554432_1_1_rev",
        # "./isca_mtv_8192_1_8192_rev",
        # "./isca_ttv_256_512_512_rev",
        # "./isca_mmtv_256_512_512_rev",
        # "./isca_polyva_67108864_1_1_rev",
        # "./isca_polygemv1_8192_1_8192_rev",

        # "./isca_va_1048576_1_1_rev",
        # "./isca_red_524288_1_1_rev",
        # "./isca_mtv_1024_1_1024_rev",
        # "./isca_ttv_32_64_512_rev",
        # "./isca_mmtv_32_64_512_rev",
        # "./isca_polyva_1048576_1_1_rev",
        # "./isca_polygemv1_1024_1_1024_rev",

        # "./archive_isca/isca_mmtv_256_512_512_rev"


    # ]:
    query(args.workdir, args.only_show, args.only_run, False)
