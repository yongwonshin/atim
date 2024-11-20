import sys, os

sys.path.append("/root/dev/tvm/pim_test/upmem")
os.chdir("/root/dev/tvm/pim_test/upmem")

from tqdm import tqdm
from itertools import product
from va import VA
from reduction import REDUCE
from gemv import GEMV
import math
import numpy as np


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
    verbose=0,
):
    if verbose >= 1:
        print("[[[[[[" + instance.profile + "]]]]]]]")
    max_time = 1e9
    best_config = configs[0]
    available_configs = []
    with tqdm(total=len(configs)) as pbar:
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
            if verbose >= 1:
                tqdm.write("\t".join([str(x) for x in list(tuples) + list(conf)]))
            pbar.update(1)

    instance.warmup, prev_warmup = 10, instance.warmup
    instance.repeat, prev_repeat = 1000, instance.repeat
    best_tuple = instance.benchmark(**dict(zip(config_label, best_config)))
    instance.warmup, instance.repeat = prev_warmup, prev_repeat
    print(f"Best config: {best_config} with {reducer(best_tuple)} ms / {best_tuple}")
    if verbose >= 1:
        print(len(available_configs))
        print(available_configs)
        print()
    return best_config


class BatchedGEMVBenchmark(GEMV):
    def __init__(self, **kwargs):
        super().__init__(profile="rgemv", **kwargs)
        self.required = dict(
            B=256,
            M=512,
            N=512,
            dtype="int32",
            n_yb=1,
            n_bb=1,
            # n_xb: Unneed TIR implementation
            n_yt=16,
        )
        print(self.required)

    # def host_version
    # fetch_data
    def benchmark_command(self, config):
        bl = int(math.log2(config["n_cache"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"""
            make clean &&
            NR_DPUS_Y={config["n_yb"]} \
            NR_DPUS_B={config["n_bb"]} \
            NR_TASKLETS={config["n_yt"]} \
            BL={bl} TYPE={pbtype} make &&
            ./bin/gemv_host -b {config["B"]} \
                -m {config["M"]} \
                -n {config["N"]} \
                -w {self.warmup} \
                -e {self.repeat}
        """


rep = 100
warmup = 1
va = VA(repeat=rep, warmup=warmup, bench=True, compile_only=True)
reduction = REDUCE(repeat=rep, warmup=warmup, bench=True, compile_only=True)
gemv = GEMV(repeat=rep, warmup=warmup, bench=True, compile_only=True)
bgemv = BatchedGEMVBenchmark(bench=True, warmup=warmup, repeat=rep, compile_only=True)


# VA, TA
def run_va(L):
    dpu_grid = [512, 1024, 1536, 2048]
    tasklets_grid = [16, 20, 24]
    cache_grid = [8, 16, 32, 64, 128]
    configs = [(L, *p, "int32") for p in product(dpu_grid, tasklets_grid, cache_grid)]
    config_label = ["L", "n_b", "n_t", "n_c", "dtype"]
    va.profile = "va"
    run(va, configs, config_label)


def run_red(L):
    dpus = [256, 512, 1024, 2048]
    tasklets = [16, 24]  # 20: correctness issue
    cache_size = [8, 16, 32, 64, 128]
    configs = [(L, d, t, c, "int64") for d, t, c in product(dpus, tasklets, cache_size)]
    config_label = ["L", "n_b", "n_t", "n_c", "dtype"]
    run(reduction, configs, config_label)


def run_polyva(L):
    dpu_grid = [512, 1024, 1536, 2048]
    tasklets_grid = [16, 20, 24]
    cache_grid = [8, 16, 32, 64, 128]
    configs = [(L, *p, "int32") for p in product(dpu_grid, tasklets_grid, cache_grid)]
    config_label = ["L", "n_b", "n_t", "n_c", "dtype"]
    va.profile = "va_poly"
    run(va, configs, config_label)


def run_polygemv(M, K):
    gemv.profile = "gemv_poly"
    dpus = [256, 512, 1024, 1536, 2048]
    tasklets = [16, 20, 24]  # 20: correctness issue
    cache_size = [8, 16, 32, 64, 128]
    configs = [
        (M, K, 1, d, t, 1, c, "int32") for d, t, c in product(dpus, tasklets, cache_size)
    ]
    config_label = ["M", "K", "n_xb", "n_yb", "n_yt", "n_rt", "n_cache", "dtype"]
    run(gemv, configs, config_label, reducer=get_total_time_gemv)


def run_mtv(M, K):
    gemv.profile = "gemv"
    dpus = [512, 1024, 1536, 2048]
    tasklets = [16, 24]  # 20: correctness issue
    cache_size = [16, 32, 64, 128]
    configs = [
        (M, K, 1, d, t, 1, c, "int32") for d, t, c in product(dpus, tasklets, cache_size)
    ]
    config_label = ["M", "K", "n_xb", "n_yb", "n_yt", "n_rt", "n_cache", "dtype"]
    run(gemv, configs, config_label, reducer=get_total_time_gemv)


def run_mmtv(B, M, N):
    best_configs = []
    ytile = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    tasklets = [1, 2, 4, 8, 16]
    caches = [16, 32, 64, 128]
    configs = []
    for b, y, t, c in product([B], ytile, tasklets, caches):
        if B * y <= 2048 and 32 <= B * y and y * t * 2 <= M and b * y >= 32:
            configs.append((B, M, N, b, y, t, c, "int32"))
    configs_label = ["B", "M", "N", "n_bb", "n_yb", "n_yt", "n_cache", "dtype"]
    res = run(bgemv, configs, configs_label, reducer=get_total_time_gemv)
    best_configs.append(res)


print("GPT")
run_mmtv(16, 64, 256)
run_mmtv(16, 128, 256)
run_mmtv(16, 256, 256)
run_mmtv(16, 512, 256)
run_mmtv(64, 64, 256)
run_mmtv(64, 128, 256)
run_mmtv(64, 256, 256)
run_mmtv(64, 512, 256)
run_mmtv(256, 64, 256)
run_mmtv(256, 128, 256)
run_mmtv(256, 256, 256)
run_mmtv(256, 512, 256)
run_mmtv(28, 64, 256)
run_mmtv(28, 128, 256)
run_mmtv(28, 256, 256)
run_mmtv(28, 512, 256)
run_mmtv(112, 64, 256)
run_mmtv(112, 128, 256)
run_mmtv(112, 256, 256)
run_mmtv(112, 512, 256)
run_mmtv(448, 64, 256)
run_mmtv(448, 128, 256)
run_mmtv(448, 256, 256)
run_mmtv(448, 512, 256)

# run_mtv(12288, 4096)
# run_mtv(4096, 4096)
# run_mtv(16384, 4096)
# run_mtv(4096, 16384)
# run_mtv(21504, 7168)
# run_mtv(7168, 7168)
# run_mtv(28672, 7168)
# run_mtv(7168, 28672)

# print()
# print("POLYBENCH")

# run_va(67108864) # why it breaks?=
# run_red(33554432)
# run_mtv(8192, 8192)
# run_mtv(256*512, 512)
run_mmtv(256, 512, 512)
# run_polyva(67108864)
# run_polygemv(8192, 8192)

# run_va(1048576)
# run_red(524288)
# run_mtv(1024, 1024)
# run_mtv(32*64, 512)
run_mmtv(32, 64, 512)
# run_polyva(1048576)
# run_polygemv(1024, 1024)

print()
print("SENS")
# random_toks = [26, 69, 109, 126, 153, 180, 232, 255, 272, 321, 345, 380, 399, 429, 473, 501]
# random_toks = [399]
# shapes = [(64, T, 256) for T in random_toks] + [(64, T, T) for T in random_toks]
# for s in shapes:
#     run_mmtv(*s)