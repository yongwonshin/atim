import sys
import argparse
import importlib.util
import multiprocessing

import numpy as np

sys.path.insert(0, "tvm_cputest/python")
from tvm.target import Target
from tvm import tir
import tvm

from tasks import poly_tasks

argparser = argparse.ArgumentParser(description='Evaluate CPU performance')
argparser.add_argument('--pretuned', action='store_true', help='Use pretuned parameters')
args = argparser.parse_args()

target = Target(f"llvm --num-cores={multiprocessing.cpu_count()}")

def ndarray(size, dtype="int32"):
    arr = np.random.randint(0, 50, size=size, dtype=dtype)
    ret = tvm.nd.array(arr)
    # flush.flush_cache_clflush(arr)
    return ret

def ndzeros(size, dtype="int32"):
    arr = np.zeros(size, dtype=dtype)
    ret = tvm.nd.array(arr)
    # flush.flush_cache_clflush(arr)
    return ret

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

def eval_pretuned(op_type, m, n, k):
    print(op_type, m, n, k)
    mod = get_pretuned_schedule(op_type, m, n, k)
    func = tvm.build(mod, target)
    evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number=1, repeat=100, flushing_cache=True)

    times = []

    if op_type == "va":
        a = ndarray((m,))
        b = ndarray((m,))
        c = ndzeros((m,))
        t = evaluator(a, b, c)

    if op_type == "mmtv":
        a = ndarray((m, n, k))
        b = ndarray((m, k))
        c = ndzeros((m, n))
        t = evaluator(a, b, c)

    if op_type == "mtv":
        a = ndarray((m, k))
        b = ndarray((k,))
        c = ndzeros((m,))
        t = evaluator(a, b, c)

    if op_type == "red":
        a = ndarray((m,), dtype=np.int64)
        c = ndzeros((1,), dtype=np.int64)
        t = evaluator(a, c)

    if op_type == "ttv":
        a = ndarray((m, n, k))
        b = ndarray((k,))
        c = ndzeros((m, n))
        t = evaluator(a, b, c)

    if op_type == "geva":
        a = ndarray((m,))
        b = ndarray((m,))
        alpha = ndarray((1,))
        beta = ndarray((1,))
        c = ndzeros((m,))
        t = evaluator(a, b, c, alpha, beta)

    if op_type == "gemv":
        a = ndarray((m, k))
        b = ndarray((k,))
        alpha = ndarray((1,))
        c = ndzeros((m,))
        t = evaluator(a, b, c, alpha)

    else:
        return

    print(f"Elapsed time: {t.median * 1000}")

if __name__ == "__main__":
    if args.pretuned:
        for task in poly_tasks:
            eval_pretuned(*task)