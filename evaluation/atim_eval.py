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
# import random
from save_csv import PolySaver, GPTJSaver

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
    if not os.path.exists(os.path.join(workdir, "database_workload.json")):
        return None
    database = JSONDatabase(work_dir=workdir)
    all_records = database.get_all_tuning_records()
    top_record = sorted(all_records, key=lambda rec: rec.run_secs[0])[0]
    assert len(top_record.run_secs) == 1
    mod = top_record.workload.mod
    schedule = ms.tir_integration.compile_tir(database, mod, target)
    return schedule

def eval_mod(sch, op_type, m, n, k):
    # return (random.random() * 10, random.random() * 10, random.random() * 10, random.random() * 10)
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

def eval_group(saver, tasks):
    for task in tasks:
        if not task[0]:
            continue
        time_tuple = (0, 0, 0, 0)
        try:
            if args.pretuned:
                sch = get_pretuned_schedule(*task)
            else:
                sch = get_reproduced_schedule(*task)
                if not sch:
                    raise FileNotFoundError(f"CPU-autotuned module not found for task {task}")
            print(f"Evaluating ATiM task {task}")
            time = eval_mod(sch, *task)
            time_tuple = (time[0], time[1], time[2], time[4])
            print(f"H2D: {time_tuple[0]:.3f} ms, Kernel: {time_tuple[1]:.3f} ms, D2H: {time_tuple[2]:.3f} ms, Total: {time_tuple[3]:.3f} ms")
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"Error processing task {task} with schedule: {e}")
        saver.set_atim(task, *time_tuple)
        saver.commit()

if __name__ == "__main__":
    csv_gptj = GPTJSaver()
    csv_poly = PolySaver()

    eval_group(csv_gptj, gptj_tasks)
    eval_group(csv_poly, poly_tasks)