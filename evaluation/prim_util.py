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
    "gemv": MTV(**default_config),
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
