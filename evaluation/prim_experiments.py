
import sys, os
from tqdm import tqdm
from itertools import product
from workloads import VA, RED, MTV, MMTV
import math
import numpy as np
import time
import pandas as pd

VERBOSE = 0
NAIVE = False
warmup = 1
rep = 100
eval_warmup = 100
eval_rep = 1000
ignore_wrong = True

def get_total_time_gemv(tuples):
    return tuples[0] + tuples[1] + tuples[2]

def get_total_time_reduce(tuples):
    return tuples[1] + tuples[2]

def run(
    instance,
    configs,
    config_label,
    reducer=get_total_time_reduce,
    collect_available_configs=False,
):
    verbose = VERBOSE
    max_time = 1e9
    best_config = configs[0]
    available_configs = []
    with tqdm(total=len(configs), leave=False) as pbar:
        for conf in configs:
            try:
                tuples = instance.benchmark(**dict(zip(config_label, conf)))
                total_time = reducer(tuples)
                if total_time < max_time:
                    max_time = total_time
                    best_config = conf
                if collect_available_configs:
                    available_configs.append(conf)
            except ValueError as e:
                tuples = ("wrong", "", "")
            except RuntimeError as e:
                tuples = ("fail", "", "")
            except TimeoutError as e:
                tuples = ("timeout", "", "")
            finally:
                if verbose >= 1:
                    with open("prim_search_results.txt", "a") as f:
                        f.write("\t".join([str(x) for x in list(tuples) + [reducer(tuples)] + list(conf)]) + "\n")
                pbar.update(1)
    if verbose >= 1:
        with open("prim_search_results.txt", "a") as f:
            f.write("\n")


    instance.warmup, prev_warmup = eval_warmup, instance.warmup
    instance.repeat, prev_repeat = eval_rep, instance.repeat
    best_tuple = instance.benchmark(**dict(zip(config_label, best_config)))
    instance.warmup, instance.repeat = prev_warmup, prev_repeat
    latency = list(best_tuple) + [reducer(best_tuple)]
    print(f"{instance.profile}\t{best_tuple[0]}\t{best_tuple[1]}\t{best_tuple[2]}\t{reducer(best_tuple)}\t{best_config}")
    return latency


default_config = {
    "repeat": rep,
    "warmup": warmup,
    "bench": True,
    "compile_only": True,
    "ignore_wrong": ignore_wrong,
}

va = VA(**default_config)
reduction = RED(**default_config)
gemv = MTV(**default_config)
bgemv = MMTV(**default_config)

def tilings(default_tasklets=16, default_caches=256):
    dpus = [256, 512, 1024, 2048]
    tasklets = [8, 16, 24]
    caches = [8, 16, 32, 64, 128, 256]
    if NAIVE:
        tasklets = [default_tasklets]
        caches = [default_caches]
    return product(dpus, tasklets, caches)

# VA, TA
def run_va(L):
    va.profile = "va"
    configs = [(L, *p, "int32") for p in tilings()]
    config_label = ["M", "n_b", "n_t", "n_c", "dtype"]
    return run(va, configs, config_label)


def run_red(L):
    configs = [(L, *p, "int64") for p in tilings()]
    config_label = ["M", "n_b", "n_t", "n_c", "dtype"]
    return run(reduction, configs, config_label)


def run_polyva(L):
    va.profile = "geva"
    configs = [(L, *p, "int32") for p in tilings()]
    config_label = ["M", "n_b", "n_t", "n_c", "dtype"]
    return run(va, configs, config_label)


def run_polygemv(M, K):
    gemv.profile = "gemv"
    configs = [ (M, K, 1, d, t, 1, c, "int32") for d, t, c in tilings() ]
    config_label = ["M", "K", "n_xb", "n_yb", "n_yt", "n_rt", "n_cache", "dtype"]
    return run(gemv, configs, config_label, reducer=get_total_time_gemv)


def run_mtv(M, K):
    gemv.profile = "mtv"
    configs = [(M, K, 1, d, t, 1, c, "int32") for d, t, c in tilings()]
    config_label = ["M", "K", "n_xb", "n_yb", "n_yt", "n_rt", "n_cache", "dtype"]
    return run(gemv, configs, config_label, reducer=get_total_time_gemv)


def run_mmtv(M, N, K):
    bgemv.profile = "mmtv"
    ytile = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    tasklets = [1, 2, 4, 8, 16]
    caches = [16, 32, 64, 128, 256]
    configs = []
    if NAIVE:
        tasklets = [16]
        caches = [256]
    for b, y, t, c in product([M], ytile, tasklets, caches):
        if M * y <= 2048 and 32 <= M * y and y * t * 2 <= N and b * y >= 32:
            configs.append((M, N, K, b, y, t, c, "int32"))
    configs_label = ["M", "N", "K", "n_bb", "n_yb", "n_yt", "n_cache", "dtype"]
    return run(bgemv, configs, configs_label, reducer=get_total_time_gemv)

def search_poly(naive=False):
    global NAIVE
    NAIVE = naive

    df_poly = pd.read_csv("./graph/result_poly.csv")

    poly_res = []
    print("POLYBENCH-S")
    poly_res.append(run_va(1048576))
    poly_res.append(run_red(524288))
    poly_res.append(run_mtv(1024, 1024))
    poly_res.append(run_mtv(32*64, 512))
    poly_res.append(run_mmtv(32, 64, 512))
    poly_res.append(run_polyva(1048576))
    poly_res.append(run_polygemv(1024, 1024))
    print()

    print("POLYBENCH-M")
    poly_res.append(run_va(16777216))
    poly_res.append(run_red(8388608))
    poly_res.append(run_mtv(4096, 4096))
    poly_res.append(run_mtv(128*256, 512))
    poly_res.append(run_mmtv(128, 256, 512))
    poly_res.append(run_polyva(16777216))
    poly_res.append(run_polygemv(4096, 4096))
    print()

    print("POLYBENCH-L")
    poly_res.append(run_va(67108864))
    poly_res.append(run_red(33554432))
    poly_res.append(run_mtv(8192, 8192))
    poly_res.append(run_mtv(256*512, 512))
    poly_res.append(run_mmtv(256, 512, 512))
    poly_res.append(run_polyva(67108864))
    poly_res.append(run_polygemv(8192, 8192))
    print()

    print("POLYBENCH-XL")
    poly_res.append([0.0, 0.0, 0.0, 0.0])
    poly_res.append(run_red(67108864))
    poly_res.append(run_mtv(8192, 16384))
    poly_res.append(run_mtv(512*512, 512))
    poly_res.append(run_mmtv(512, 512, 512))
    poly_res.append([0.0, 0.0, 0.0, 0.0])
    poly_res.append(run_polygemv(8192, 16384))
    print()
    if naive:
        df_poly.iloc[:, 5:9] = poly_res
    else:
        df_poly.iloc[:, 9:13] = poly_res
    df_poly.to_csv("./graph/result_poly.csv", index=False)


def search_gptj(naive=False):
    global NAIVE
    NAIVE = naive

    df_gptj = pd.read_csv("./graph/result_gptj.csv")
    gptj_res = []

    print("GPT-6B")
    gptj_res.append(run_mmtv(16, 64, 256))
    gptj_res.append(run_mmtv(16, 128, 256))
    gptj_res.append(run_mmtv(16, 256, 256))
    gptj_res.append(run_mmtv(16, 512, 256))
    gptj_res.append(run_mmtv(64, 64, 256))
    gptj_res.append(run_mmtv(64, 128, 256))
    gptj_res.append(run_mmtv(64, 256, 256))
    gptj_res.append(run_mmtv(64, 512, 256))
    print()

    print("GPT-13B")
    gptj_res.append(run_mmtv(28, 64, 256))
    gptj_res.append(run_mmtv(28, 128, 256))
    gptj_res.append(run_mmtv(28, 256, 256))
    gptj_res.append(run_mmtv(28, 512, 256))
    gptj_res.append(run_mmtv(112, 64, 256))
    gptj_res.append(run_mmtv(112, 128, 256))
    gptj_res.append(run_mmtv(112, 256, 256))
    gptj_res.append(run_mmtv(112, 512, 256))
    print()

    print("GPT-MTV")
    gptj_res.append(run_mtv(12288, 4096))
    gptj_res.append(run_mtv(4096, 4096))
    gptj_res.append(run_mtv(16384, 4096))
    gptj_res.append(run_mtv(4096, 16384))
    gptj_res.append(run_mtv(21504, 7168))
    gptj_res.append(run_mtv(7168, 7168))
    gptj_res.append(run_mtv(28672, 7168))
    gptj_res.append(run_mtv(7168, 28672))

    if naive:
        df_gptj.iloc[:, 5:9] = gptj_res
    else:
        df_gptj.iloc[:, 9:13] = gptj_res
    df_gptj.to_csv("./graph/result_gptj.csv", index=False)

if __name__ == "__main__":
    search_poly(True)
    search_gptj(True)
    search_poly(False)
    search_gptj(False)
