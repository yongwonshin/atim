from tasks import poly_tasks, gptj_tasks
import argparse
from query_result import query
import pandas as pd
import traceback
import os
import importlib.util
import sys
import multiprocessing
from tvm.meta_schedule.database import JSONDatabase
from tvm.target import Target
from tvm import meta_schedule as ms
from tvm import tir
import tvm
from workloads import get_workload

target = Target(f"upmem --num-cores={multiprocessing.cpu_count()}")

def get_pretuned_schedule(op_type, m, n, k):
    module_name = f"{op_type}_{m}_{n}_{k}"
    module_path = f"./results/tuned_modules/atim_{module_name}.py"
    full_module_name = f"module_atim_{module_name}"
    spec = importlib.util.spec_from_file_location(full_module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_module_name] = module
    spec.loader.exec_module(module)
    ir_mod = getattr(module, full_module_name)
    return tir.Schedule(ir_mod)

def get_reproduced_schedule(op_type, m, n, k):
    workdir = f"./reproduced/tuned/{op_type}_{m}_{n}_{k}"
    database = JSONDatabase(work_dir=workdir)
    all_records = database.get_all_tuning_records()
    top_record = sorted(all_records, key=lambda rec: rec.run_secs[0])[0]
    assert len(top_record.run_secs) == 1
    mod = top_record.workload.mod
    schedule = ms.tir_integration.compile_tir(database, mod, target)
    return schedule

def eval_mod(sch, op_type, m, n, k):
    op_class = get_workload(op_type)
    dtype = "int64" if op_type == "red" else "int32"
    workload = op_class(
        repeat=1000,
        warmup=10,
        compile_only=False,
        output_format="tab",
        verbose=1,
        use_time_evaluator=True
    )
    workload.test(
        lambda *args, **kwargs: sch,
        M=m,
        N=n,
        K=k,
        n_xb=-1,
        n_yb=-1,
        n_t=-1,
        n_rt=-1,
        n_cache=-1,
        dtype=dtype,
    )
    return workload.recent_time_tuple

argparser = argparse.ArgumentParser(description='Evaluate CPU performance')
argparser.add_argument('--pretuned', action='store_true', help='Use pretuned parameters')
args = argparser.parse_args()

def eval_group(df, tasks):
    results = []
    for op_type, m, n, k  in tasks:
        if not op_type:
            results.append((0, 0, 0, 0))
            continue
        try:
            if args.pretuned:
                sch = get_pretuned_schedule(op_type, m, n, k)
            else:
                sch = get_reproduced_schedule(op_type, m, n, k)
            time = eval_mod(sch, op_type, m, n, k)
            time_tuple = (time[0], time[1], time[2], time[4])
        except Exception as e:
            print(f"Error during query: {e}")
            traceback.print_exc()
            time_tuple = (0, 0, 0, 0)
        results.append(time_tuple)
    df.iloc[:, 13:17] = results

if __name__ == "__main__":
    df_gptj = pd.read_csv("./reproduced/result_gptj.csv")
    df_poly = pd.read_csv("./reproduced/result_poly.csv")

    eval_group(df_gptj, gptj_tasks)
    eval_group(df_poly, poly_tasks)

    df_gptj.to_csv("./reproduced/result_gptj.csv", index=False)
    df_poly.to_csv("./reproduced/result_poly.csv", index=False)