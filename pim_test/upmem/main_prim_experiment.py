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

def run(instance, configs, config_label, reducer=get_total_time_reduce, collect_available_configs=False):
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
            tqdm.write("\t".join([str(x) for x in list(tuples) + list(conf)]))
            pbar.update(1)
    print(f"Best config: {best_config} with {max_time} ms")
    print(len(available_configs))
    print(available_configs)
    print()

rep = 100
warmup = 10
va = VA(repeat=rep, warmup=warmup, bench=True, compile_only=True)
reduction = REDUCE(repeat=rep, warmup=warmup, bench=True, compile_only=True)
gemv = GEMV(repeat=rep, warmup=warmup, bench=True, compile_only=True)

# VA, TA
dpu_grid = [2048]
tasklets_grid = [16, 20, 24]
cache_grid = [8, 16, 32, 64, 128]
configs = [(67108864, *p, "int32") for p in product(dpu_grid, tasklets_grid, cache_grid)]
config_label = ["L", "n_b", "n_t", "n_c", "dtype"]
va.profile = "va"
# run(va, configs, config_label)

# reduction
dpus = [512, 1024, 2048]
tasklets = [16, 24] # 20: correctness issue
cache_size = [8, 16, 32, 64, 128]
configs = [(33554432, d, t, c, "int32") for d, t, c in product(dpus, tasklets, cache_size)]
config_label = ["L", "n_b", "n_t", "n_c", "dtype"]
va.profile = "reduction"
#run(reduction, configs, config_label)

# POLYVA
dpu_grid = [512, 1024, 1536, 2048]
dpu_grid = [2048]
tasklets_grid = [16, 20, 24]
cache_grid = [8, 16, 32, 64, 128]
configs = [(67108864, *p, "int32") for p in product(dpu_grid, tasklets_grid, cache_grid)]
config_label = ["L", "n_b", "n_t", "n_c", "dtype"]
va.profile = "va_poly"
#run(va, configs, config_label)

# POLYGEMV
gemv.profile = "gemv_poly"
dpus = [256, 512, 1024, 1536, 2048]
tasklets = [16, 20, 24] # 20: correctness issue
cache_size = [8, 16, 32, 64, 128]
configs = [(8192, 8192, 1, d, t, 1, c, "int32") for d, t, c in product(dpus, tasklets, cache_size)]
config_label = ["M", "K", "n_xb", "n_yb", "n_yt", "n_rt", "n_cache", "dtype"]
#run(gemv, configs, config_label, reducer=get_total_time_gemv)


## TTV
for m, k in [(12288, 4096), (4096, 4096), (16384, 4096), (4096, 16384)]:
    gemv.profile = "gemv"
    dpus = [256, 512, 1024, 1536, 2048]
    tasklets = [16, 20, 24] # 20: correctness issue
    cache_size = [8, 16, 32, 64, 128]
    configs = [(m, k, 1, d, t, 1, c, "int32") for d, t, c in product(dpus, tasklets, cache_size)]
    config_label = ["M", "K", "n_xb", "n_yb", "n_yt", "n_rt", "n_cache", "dtype"]
    # run(gemv, configs, config_label, reducer=get_total_time_gemv)

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
            #n_xb: Unneed TIR implementation
            n_yt=16
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
            TYPE={pbtype} make &&
            ./bin/gemv_host -b {config["B"]} \
                -m {config["M"]} \
                -n {config["N"]} \
                -w {self.warmup} \
                -e {self.repeat}
        """

bgemv = BatchedGEMVBenchmark(bench=True, warmup=warmup, repeat=rep, compile_only=True)

btile = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ytile = [1, 2, 4, 8, 16, 32, 64, 128, 256]
tasklets = [1, 2, 4, 8, 16]
caches = [8, 16, 32, 64, 128]

#for (B, M, N) in [(256, 512, 512), (64, 64, 256), (64, 128, 256), (64, 256, 256), (64, 512, 256)]:
for (B, M, N) in [(256, 512, 512)]:
    configs = []
    for b, y, t, c in product([B], ytile, tasklets, caches):
        if (B * y <= 2048 and
            32 <= B * y and
            y * t * 2 <= M and b * y >= 32):
            configs.append((B, M, N, b, y, t, c, "int32"))
    configs_label = ["B", "M", "N", "n_bb", "n_yb", "n_yt", "n_cache", "dtype"]

    run(bgemv, configs, configs_label, reducer=get_total_time_gemv)