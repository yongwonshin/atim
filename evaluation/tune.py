import sys
import time
import os
import argparse

from tvm import meta_schedule as ms
from tvm.target import Target

from bench import *


def get_module(M, N, K, dtype):
    if args.op_type == "mtv":
        return upmem_mtv_factory(M, K, dtype)
    elif args.op_type == "ttv":
        return upmem_ttv_factory(M, N, K, dtype)
    elif args.op_type == "polygemv1":
        return upmem_poly_gemv1_factory(M, K, dtype)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_type", required=True, type=str)
    parser.add_argument("--M", required=True, type=int)
    parser.add_argument("--N", required=True, type=int)
    parser.add_argument("--K", required=True, type=int)
    parser.add_argument("--workdir", default="bench_autotuner_result", type=str)
    parser.add_argument("--reuse_cost_model", action="store_true")
    args = parser.parse_args()

    name, M, N, K = "default", args.M, args.N, args.K
    target = Target("upmem --num-cores=96")
    os.system(f"mkdir -p ./{args.workdir}")

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
