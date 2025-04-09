import sys
import argparse
import importlib.util
import multiprocessing
import os
import numpy as np
import pandas as pd

env = os.environ.copy()
env["PYTHONPATH"] = f"{os.path.abspath('.')}/tvm_cputest/python:{env['PYTHONPATH']}"

sys.path.insert(0, "tvm_cputest/python")
from tvm.target import Target
from tvm import tir
from tvm import meta_schedule as ms
from tvm.meta_schedule.database import JSONDatabase
import tvm

from tasks import poly_tasks

argparser = argparse.ArgumentParser(description='Evaluate CPU performance')
argparser.add_argument('--pretuned', action='store_true', help='Use pretuned parameters')
args = argparser.parse_args()

target = Target(f"llvm --num-cores={multiprocessing.cpu_count()}")

def ndarray(size, dtype="int32"):
    arr = np.random.randint(0, 50, size=size, dtype=dtype)
    return tvm.nd.array(arr)

def ndzeros(size, dtype="int32"):
    arr = np.zeros(size, dtype=dtype)
    return tvm.nd.array(arr)

def get_pretuned_schedule(op_type, m, n, k):
    module_name = f"{op_type}_{m}_{n}_{k}"
    module_path = f"./results/cpu_tuned_modules/cpu_{module_name}.py"
    full_module_name = f"module_cpu_{module_name}"
    spec = importlib.util.spec_from_file_location(full_module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_module_name] = module
    spec.loader.exec_module(module)
    ir_mod = getattr(module, full_module_name)
    return ir_mod

def get_reproduced_schedule(op_type, m, n, k):
    workdir = "./reproduced/cpu_tuned/" + f"{op_type}_{m}_{n}_{k}"
    database = JSONDatabase(work_dir=workdir)
    all_records = database.get_all_tuning_records()
    top_record = sorted(all_records, key=lambda rec: rec.run_secs[0])[0]
    assert len(top_record.run_secs) == 1
    mod = top_record.workload.mod
    sch = ms.tir_integration.compile_tir(database, mod, target)
    return sch.mod

def eval_mod(mod, op_type, m, n, k):
    print(op_type, m, n, k)
    func = tvm.build(mod, target)
    evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number=1, repeat=100, flushing_cache=True)

    if op_type == "va":
        a = ndarray((m,))
        b = ndarray((m,))
        c = ndzeros((m,))
        t = evaluator(a, b, c)

    elif op_type == "mmtv":
        a = ndarray((m, n, k))
        b = ndarray((m, k))
        c = ndzeros((m, n))
        t = evaluator(a, b, c)

    elif op_type == "mtv":
        a = ndarray((m, k))
        b = ndarray((k,))
        c = ndzeros((m,))
        t = evaluator(a, b, c)

    elif op_type == "red":
        a = ndarray((m,), dtype=np.int64)
        c = ndzeros((1,), dtype=np.int64)
        t = evaluator(a, c)

    elif op_type == "ttv":
        a = ndarray((m, n, k))
        b = ndarray((k,))
        c = ndzeros((m, n))
        t = evaluator(a, b, c)

    elif op_type == "geva":
        a = ndarray((m,))
        b = ndarray((m,))
        alpha = ndarray((1,))
        beta = ndarray((1,))
        c = ndzeros((m,))
        t = evaluator(a, b, c, alpha, beta)

    elif op_type == "gemv":
        a = ndarray((m, k))
        b = ndarray((k,))
        alpha = ndarray((1,))
        c = ndzeros((m,))
        t = evaluator(a, b, c, alpha)

    else:
        raise ValueError(f"Unknown op_type: {op_type}")

    elapsed_time = t.median * 1000
    print(f"Elapsed time: {elapsed_time}")
    return elapsed_time

if __name__ == "__main__":
    df_poly = pd.read_csv("./reproduced/result_poly.csv")
    results = []

    for task in poly_tasks:
        if not task[0]:
            results.append(0.0)
            continue
        try:
            if args.pretuned:
                mod = get_pretuned_schedule(*task)
            else:
                mod = get_reproduced_schedule(*task)
            elapsed_time = eval_mod(mod, *task)
        except Exception as e:
            print(f"Error processing task {task} with schedule: {e}")
            elapsed_time = 0.0
        results.append(elapsed_time)

    df_poly["CPU-Autotuned"] = results
    df_poly.to_csv("./reproduced/result_poly.csv", index=False)