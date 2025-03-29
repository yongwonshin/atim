import sys
import time
import os
import argparse
import multiprocessing

from tvm import meta_schedule as ms
from tvm.target import Target

from bench import get_base_module

from tasks import poly_tasks, gptj_tasks

def tune(op_type, M, N, K, workdir, reuse_cost_model=False):
    target = Target(f"upmem --num-cores={multiprocessing.cpu_count()}")
    os.system(f"mkdir -p ./{workdir}")

    start = time.time()
    with open(f"./{workdir}/default.txt", "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        mod = get_base_module(op_type, M, N, K, dtype="int32")

        cost_model = "xgb"
        if reuse_cost_model and os.path.exists(f"{workdir}.tar"):
            print("Cost model reused")
            cost_model = ms.CostModel.create("xgb", num_tuning_cores=1)
            cost_model.load(f"{workdir}.tar")

        database = ms.tir_integration.tune_tir(
            mod=mod,
            target=target,
            work_dir=f"./{workdir}",
            max_trials_global=1000,
            num_trials_per_iter=64,
            # num_tuning_cores=1,  # to prevent dpu allocation error
            cost_model=cost_model,
        )
        sch = ms.tir_integration.compile_tir(database, mod, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show(black_format=False, name="default")
            sch.trace.show()
        sys.stdout = original_stdout
    end = time.time()
    print(f"DONE {op_type} {M} {N} {K} in {end - start} seconds")

def tune_group(tasks):
    for op_type, m, n, k in tasks:
        if not op_type:
            continue
        try:
            tune(op_type, m, n, k, f"./reproduced/tuned/{op_type}_{m}_{n}_{k}", reuse_cost_model=False)
        except Exception as e:
            print(f"Error: {op_type}, {m}, {n}, {k}")
            print(e)
            continue

tune_group(gptj_tasks)
tune_group(poly_tasks)