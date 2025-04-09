import sys
import time
import argparse
import os
import multiprocessing

sys.path.insert(0, "tvm_cputest/python")
# make sure TVM in pythonpath indicates the python directory in tvm_cputest
from tvm import meta_schedule as ms
from tvm.target import Target

from bench import get_base_module
from tasks import get_tasks

parser = argparse.ArgumentParser()
parser.add_argument("--kick-the-tires", action="store_true", help="Run CPU autotune with single workload for AE kick-the-tires.")
args = parser.parse_args()

def tune(op_type, M, N, K, workdir, reuse_cost_model=False):
    target = Target(f"llvm --num-cores={multiprocessing.cpu_count()}")
    os.system(f"mkdir -p ./{workdir}")

    start = time.time()

    mod = get_base_module(op_type, M, N, K, dtype="int32")
    cost_model = "xgb"
    if reuse_cost_model and os.path.exists(f"{workdir}.tar"):
        print("Cost model reused")
        cost_model = ms.CostModel.create("xgb")
        cost_model.load(f"{workdir}.tar")

    database = ms.tir_integration.tune_tir(
        mod=mod,
        target=target,
        work_dir=f"./{workdir}",
        max_trials_global=1000,
        num_trials_per_iter=64,
        num_tuning_cores=multiprocessing.cpu_count(),
        cost_model=cost_model,
    )
    sch = ms.tir_integration.compile_tir(database, mod, target)
    if sch is None:
        print("No valid schedule found!")
    else:
        sch.mod.show(black_format=False, name="default")
        sch.trace.show()

    end = time.time()
    print(f"DONE {op_type} {M} {N} {K} in {end - start} seconds")

for op_type, m, n, k in get_tasks("poly", args.kick_the_tires):
    if not op_type:
        continue
    try:
        tune(op_type, m, n, k, f"./reproduced/cpu_tuned/{op_type}_{m}_{n}_{k}", reuse_cost_model=False)
    except Exception as e:
        print(f"Error: {op_type}, {m}, {n}, {k}")
        print(e)
        continue