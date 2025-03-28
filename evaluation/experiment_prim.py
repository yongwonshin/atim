from tqdm import tqdm
from itertools import product
from workloads import VA, RED, MTV, MMTV
import pandas as pd
import json
from tasks import gptj_tasks, poly_tasks
import argparse

warmup = 1
rep = 100
eval_warmup = 100
eval_rep = 1000
ignore_wrong = True

default_config = {
    "repeat": rep,
    "warmup": warmup,
    "bench": True,
    "compile_only": True,
    "ignore_wrong": ignore_wrong,
}

instances = {
    "va": VA(**default_config),
    "red": RED(**default_config),
    "mtv": MTV(**default_config),
    "ttv": MTV(**default_config),
    "mmtv": MMTV(**default_config),
    "geva": VA(**default_config),
    "gemv": MTV(**default_config)
}
instances["geva"].profile = "geva"
instances["gemv"].profile = "gemv"

def get_tilings(op_type, M, N, K, naive):
    if op_type == "mmtv":
        ytile = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        tasklets = [1, 2, 4, 8, 16]
        caches = [16, 32, 64, 128, 256]
        if naive:
            tasklets = [16]
            caches = [256]
        configs = []
        for y, t, c in product(ytile, tasklets, caches):
            if M * y <= 2048 and 32 <= M * y and y * t * 2 <= N:
                configs.append((y, t, c))
        return configs
    else:
        dpus = [256, 512, 1024, 2048]
        if naive:
            return [(dpu, 16, 256) for dpu in dpus]
        tasklets = [8, 16, 24]
        caches = [8, 16, 32, 64, 128, 256]
        return list(product(dpus, tasklets, caches))

def config_factory(op_type, M, N, K, tiling):
    dtype = "int32" if op_type != "red" else "int64"
    d, t, c = tiling
    if op_type in {"va", "geva", "red"}:
        return (M, d, t, c, dtype)
    elif op_type in {"mtv", "gemv"}:
        return (M, K, 1, d, t, 1, c, dtype)
    elif op_type == "ttv":
        return (M * N, K, 1, d, t, 1, c, dtype)
    elif op_type == "mmtv":
        return (M, N, K, M, d, t, c, dtype)
    else:
        raise ValueError(f"Unsupported op_type: {op_type}")

def config_to_kwargs(op_type, config):
    if op_type in {"va", "geva", "red"}:
        return dict(zip(["M", "n_b", "n_t", "n_c", "dtype"], config))
    elif op_type in {"mtv", "gemv", "ttv"}:
        return dict(zip(["M", "K", "n_xb", "n_yb", "n_yt", "n_rt", "n_cache", "dtype"], config))
    elif op_type == "mmtv":
        return dict(zip(["M", "N", "K", "n_bb", "n_yb", "n_yt", "n_cache", "dtype"], config))
    else:
        raise ValueError(f"Unsupported op_type: {op_type}")

def get_total_time(op_type, time_tuple):
    if op_type in ["va", "red", "geva"]:
        return time_tuple[1] + time_tuple[2]
    else:
        return time_tuple[0] + time_tuple[1] + time_tuple[2]

def save_search_params(op_type, M, N, K, tilings, naive, jsonfile="./reproduced/search_parameters.json"):
    best_dpus, best_tasklets, best_cache = tilings
    with open(jsonfile, "r") as f:
        search_params = json.load(f)
        key = f"{op_type}_{M}_{N}_{K}"
        tmp = search_params.get(key, {})
        prim_key = "prim" if naive else "prim-search"
        tmp[prim_key] = {
            "dpus": best_dpus,
            **({"tasklets": best_tasklets, "cache_size": best_cache} if not naive else {})
        }
        search_params[key] = tmp
        with open(jsonfile, "w") as f:
            json.dump(search_params, f, indent=4)

def load_search_params(op_type, M, N, K, naive, jsonfile="./reproduced/search_parameters.json"):
    best_tiling = {}
    with open(jsonfile, "r") as f:
        search_params = json.load(f)
        key = f"{op_type}_{M}_{N}_{K}"
        prim_key = "prim" if naive else "prim-search"
        best_tiling = search_params.get(key, {}).get(prim_key, {})

    dpus = best_tiling.get("dpus")
    if dpus is None:
        raise ValueError(f"DPUs not found for key: {key}, prim_key: {prim_key} in {jsonfile}")
    tasklets = best_tiling.get("tasklets", 16)
    cache_size = best_tiling.get("cache_size", 256)
    best_tiling = (dpus, tasklets, cache_size)
    return best_tiling


def search(op_type, M, N, K, naive):
    tilings = get_tilings(op_type, M, N, K, naive)
    instance = instances[op_type]

    max_time = 1e9
    best_tiling = tilings[0]
    with tqdm(total=len(tilings), leave=False) as pbar:
        for tiling in tilings:
            config = config_factory(op_type, M, N, K, tiling)
            kwargs = config_to_kwargs(op_type, config)
            try:
                tuples = instance.benchmark(**kwargs)
                if op_type in ["va", "red", "geva"]:
                    total_time = tuples[1] + tuples[2]
                else:
                    total_time = tuples[0] + tuples[1] + tuples[2]
                if total_time < max_time:
                    max_time = total_time
                    best_tiling = tiling
            except Exception as e:
                pass
            finally:
                pbar.update(1)

    best_dpus, best_tasklets, best_cache = best_tiling
    if naive:
        print("PrIM", op_type, M, N, K, "naive DPUs: ", best_dpus)
    else:
        print("PrIM-Search", op_type, M, N, K, "founded DPUs: ", best_dpus, "tasklets: ", best_tasklets, "cache: ", best_cache)
    return best_tiling


def eval(op_type, M, N, K, tiling):
    instance = instances[op_type]
    dpus, tasklets, cache_size = tiling

    instance.warmup, prev_warmup = eval_warmup, instance.warmup
    instance.repeat, prev_repeat = eval_rep, instance.repeat
    config = config_factory(op_type, M, N, K, tiling)
    kwargs = config_to_kwargs(op_type, config)
    best_tuple = instance.benchmark(**kwargs)
    instance.warmup, instance.repeat = prev_warmup, prev_repeat

    total_time = get_total_time(op_type, best_tuple)

    print(op_type, M, N, K, "founded DPUs: ", dpus, "tasklets: ", tasklets, "cache: ", cache_size)
    print("H2D", best_tuple[0], "Kernel", best_tuple[1], "D2H", best_tuple[2], "Total", total_time)
    print()
    return best_tuple


parser = argparse.ArgumentParser(description="Run search or evaluation.")
parser.add_argument("--search", action="store_true", help="Run the search process.")
parser.add_argument("--eval", action="store_true", help="Run the evaluation process.")
args = parser.parse_args()

if args.search:
    for naive in [True, False]:
        print(("PrIM" if naive else "PrIM-Search") + " search for Tensor programs")
        for task in poly_tasks:
            params = search(*task, naive=naive)
            save_search_params(*task, params, naive)
        print(("PrIM" if naive else "PrIM-Search") + " search for GPT-J")
        for task in gptj_tasks:
            params = search(*task, naive=naive)
            save_search_params(*task, params, naive)

if args.eval:
    df_gptj = pd.read_csv("./graph/result_gptj.csv")
    df_poly = pd.read_csv("./graph/result_poly.csv")
    for naive in [True]:
        start_col = 5 if naive else 9
        print(("PrIM" if naive else "PrIM-Search") + " evaluate for GPT-J")
        results = []
        for task in gptj_tasks:
            params = load_search_params(*task, naive)
            res = eval(*task, params)
            results.append(res)
        df_gptj.iloc[:, start_col:start_col + 4] = results
        df_gptj.to_csv("./graph/result_gptj.csv", index=False)

        print(("PrIM" if naive else "PrIM-Search") + " evaluate for Tensor programs")
        results = []
        for task in poly_tasks:
            params = load_search_params(*task, naive)
            res = eval(*task, params)
            results.append(res)
        df_poly.iloc[:, start_col:start_col + 4] = results
        df_poly.to_csv("./graph/result_poly.csv", index=False)

