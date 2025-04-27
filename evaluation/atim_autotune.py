import sys
import time
import os
import argparse
import multiprocessing

from tvm import meta_schedule as ms
from tvm.target import Target

from bench import get_base_module
from parser_utils import get_tune_parser, args_to_tasks

def tune(op_type, M, N, K, workdir,
        reuse_cost_model=False,
        skip_existing=False,
        max_trials_global=1000,
        num_trials_per_iter=64,
        num_cores=multiprocessing.cpu_count()):
    if skip_existing and os.path.exists(f"{workdir}.tar"):
        print(f"Skipping {op_type}_{m}_{n}_{k} - existing module found")
        return
    target = Target(f"upmem --num-cores={num_cores}")
    os.system(f"mkdir -p ./{workdir}")

    start = time.time()
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
        max_trials_global=max_trials_global,
        num_trials_per_iter=num_trials_per_iter,
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

if __name__ == "__main__":
    parser = get_tune_parser()
    parser.add_argument("--num-cores", type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores to use")
    parser.add_argument("--workdir", type=str, default=f"./reproduced/tuned", help="Directory to save the tuning results")
    parser.add_argument("--max-trials-global", type=int, default=1000, help="Maximum number of trials")
    parser.add_argument("--num-trials-per-iter", type=int, default=64, help="Number of trials per iteration")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tasks where search parameters already exist")
    parser.add_argument("--reuse-cost-model", action="store_true", help="Reuse the cost model if it exists")

    args = parser.parse_args()
    tasks = args_to_tasks(args)

    for op_type, m, n, k in tasks:
        if not op_type:
            continue
        try:
            tune(op_type, m, n, k,
                 f"./{args.workdir}/{op_type}_{m}_{n}_{k}",
                 reuse_cost_model=args.reuse_cost_model,
                skip_existing=args.skip_existing,
                 max_trials_global=args.max_trials_global,
                 num_trials_per_iter=args.num_trials_per_iter,
                 num_cores=args.num_cores
                )
        except Exception as e:
            print(f"Error: {op_type}, {m}, {n}, {k}")
            print(e)
            continue