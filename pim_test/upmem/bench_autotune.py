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

from bench import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--op_type", required=True, type=str)
parser.add_argument("--M", required=True, type=int)
parser.add_argument("--N", required=True, type=int)
parser.add_argument("--K", required=True, type=int)
parser.add_argument("--workdir", default="bench_autotuner_result", type=str)
parser.add_argument("--reuse_cost_model", action="store_true")
args = parser.parse_args()


# GOT-J 175B, 30B, 13B, 6B
tuple_mv = {
    # "175B_qkvgen": (36864, 12288),  # 21775
    # "175B_qkvproj": (12288, 12288),  # 7278
    # "175B_fc": (49152, 12288),  # 27918
    # "175B_fcproj": (12288, 49152),  # 27847
    # "13B_qkvgen": (15360, 5120),
    # "13B_qkvproj": (5120, 5120),
    # "13B_fc": (20480, 5120),
    # "13B_fcproj": (5120, 20480),
    # "6B_qkvgen": (12288, 4096),
    # "6B_qkvproj": (4096, 4096),
    # "6B_fc": (16384, 4096),
    # "6B_fcproj": (4096, 16384),
    # "30B_qkvgen": (21504, 7168),
    # "30B_qkvproj": (7168, 7168),
    # "30B_fc": (28672, 7168),
    # "30B_fcproj": (7168, 28672),
    # "175B_fcproj_1": (12288, 49152),  # 27847
    # "175B_fcproj_2": (12288, 49152),  # 27847
    # "175B_fcproj_3": (12288, 49152),  # 27847
    # "163840_4096_1": (163840, 4096),  # 27847
    # "163840_4096_2": (163840, 4096),  # 27847
    # "163840_4096_3": (163840, 4096),  # 27847
    # "ablation": (163840, 4096),
}

# layers = {6: 16, 13: 20, 30: 28, 175: 48}
# tuple_bmv = [(16, j, 256) for j in [16, 32, 64, 128, 256, 512]] + [
#     (j, 256, 256) for j in [16, 32, 64, 128, 256, 512, 1024]
# ]
tuple_bmv = {
    # "params6B_batch1_token64": (16, 64, 256),
    # "params6B_batch1_token128": (16, 128, 256),
    # "params6B_batch1_token256": (16, 256, 256),
    # "params6B_batch1_token512": (16, 512, 256),
    # "params6B_batch16_token64": (256, 64, 256),
    # "params6B_batch16_token128": (256, 128, 256),
    # "params6B_batch16_token256": (256, 256, 256),
    # "params6B_batch16_token512": (256, 512, 256),
    # "params13B_batch1_token64": (20, 64, 256),
    # "params13B_batch1_token128": (20, 128, 256),
    # "params13B_batch1_token256": (20, 256, 256),
    # "params13B_batch1_token512": (20, 512, 256),
    # "params13B_batch16_token64": (320, 64, 256),
    # "params13B_batch16_token128": (320, 128, 256),
    # "params13B_batch16_token256": (320, 256, 256),
    # "params13B_batch16_token512": (320, 512, 256),
    # "params30B_batch1_token64": (28, 64, 256),
    # "params30B_batch1_token128": (28, 128, 256),
    # "params30B_batch1_token256": (28, 256, 256),
    # "params30B_batch1_token512": (28, 512, 256),
    # "params30B_batch16_token64": (448, 64, 256),
    # "params30B_batch16_token128": (448, 128, 256),
    # "params30B_batch16_token256": (448, 256, 256),
    # "params30B_batch16_token512_1": (448, 512, 256),
    # "params30B_batch16_token512_2": (448, 512, 256),
    # "params30B_batch16_token512_3": (448, 512, 256),
    # "params175B_batch1_token64": (48, 64, 256),
    # "params175B_batch1_token128": (48, 128, 256),
    # "params175B_batch1_token256": (48, 256, 256),
    # "params175B_batch1_token512": (48, 512, 256),
    # "params175B_batch16_token64": (768, 64, 256),
    # "params175B_batch16_token128": (768, 128, 256),
    # "params175B_batch16_token256": (768, 256, 256),
    # "params175B_batch16_token512": (768, 512, 256),
}

tuple_bench = {"default": (args.M, args.N, args.K)}
target = Target("upmem --num-cores=96")


def get_module(M, N, K, dtype):
    if args.op_type == "mtv":
        return upmem_mtv_factory(M, K, dtype)
    elif args.op_type == "ttv":
        return upmem_ttv_factory(M, N, K, dtype)
    elif args.op_type == "polygemv1":
        return upmem_poly_gemv1_factory(M, K, dtype)
    # elif args.op_type == "polygemv2":
    #     return upmem_poly_gemv2_factory(M, K, dtype)
    elif args.op_type == "va":
        return upmem_va_factory(M, dtype)
    elif args.op_type == "ta":
        return upmem_ta_factory(M, N, K, dtype)
    elif args.op_type == "polyva":
        return upmem_poly_va_factory(M, dtype)
    elif args.op_type == "polymixed":
        return upmem_poly_mixed_factory(M, N, dtype)
    elif args.op_type == "dot":
        dtype = "int64"
        return upmem_dot_factory(M, dtype)
    elif args.op_type == "red":
        dtype = "int64"
        return upmem_red_factory(M, dtype)
    elif args.op_type == "innerprod":
        dtype = "int64"
        return upmem_innerprod_factory(M, N, K, dtype)
    elif args.op_type == "mmtv":
        return upmem_mmtv_factory(M, N, K, dtype)
    else:
        raise Exception(f"Unknown operator type: {args.type}")


os.system(f"mkdir -p ./{args.workdir}")
for name, (M, N, K) in tuple_bench.items():
    start = time.time()
    with open(f"./{args.workdir}/{name}.txt", "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        mod = get_module(M, N, K, dtype="int32")
        cost_model = "xgb"
        if args.reuse_cost_model and os.path.exists(f"{args.workdir}.tar"):
            print("Cost model reused")
            cost_model = ms.CostModel.create("xgb", num_tuning_cores=1)
            cost_model.load(f"{args.workdir}.tar")
        database = ms.tir_integration.tune_tir(
            mod=mod,
            target=target,
            work_dir=f"./{args.workdir}",
            max_trials_global=1000,
            num_trials_per_iter=64,
            # num_tuning_cores=1,  # to prevent dpu allocation error
            cost_model=cost_model,
        )
        sch = ms.tir_integration.compile_tir(database, mod, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show(black_format=False, name=name)
            sch.trace.show()
        sys.stdout = original_stdout
    end = time.time()
    print("DONE ", name, " in ", end - start, " seconds")
