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
import time

VERBOSE = 0
NAIVE = False
TOT = 54
total_cnt = 0
rep = 100
warmup = 1

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
    global total_cnt
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


    instance.warmup, prev_warmup = 100, instance.warmup
    instance.repeat, prev_repeat = 1000, instance.repeat
    best_tuple = instance.benchmark(**dict(zip(config_label, best_config)))
    instance.warmup, instance.repeat = prev_warmup, prev_repeat
    print(f"{instance.profile}\t{best_tuple[0]}\t{best_tuple[1]}\t{best_tuple[2]}\t{reducer(best_tuple)}\t{best_config}")
    total_cnt += 1
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


va = VA(repeat=rep, warmup=warmup, bench=True, compile_only=True)
reduction = REDUCE(repeat=rep, warmup=warmup, bench=True, compile_only=True)
gemv = GEMV(repeat=rep, warmup=warmup, bench=True, compile_only=True)
bgemv = BatchedGEMVBenchmark(bench=True, warmup=warmup, repeat=rep, compile_only=True)

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
    config_label = ["L", "n_b", "n_t", "n_c", "dtype"]
    run(va, configs, config_label)


def run_red(L):
    configs = [(L, *p, "int64") for p in tilings()]
    config_label = ["L", "n_b", "n_t", "n_c", "dtype"]
    run(reduction, configs, config_label)


def run_polyva(L):
    va.profile = "va_poly"
    configs = [(L, *p, "int32") for p in tilings()]
    config_label = ["L", "n_b", "n_t", "n_c", "dtype"]
    run(va, configs, config_label)


def run_polygemv(M, K):
    gemv.profile = "gemv_poly"
    configs = [ (M, K, 1, d, t, 1, c, "int32") for d, t, c in tilings() ]
    config_label = ["M", "K", "n_xb", "n_yb", "n_yt", "n_rt", "n_cache", "dtype"]
    run(gemv, configs, config_label, reducer=get_total_time_gemv)


def run_mtv(M, K):
    gemv.profile = "gemv"
    configs = [(M, K, 1, d, t, 1, c, "int32") for d, t, c in tilings()]
    config_label = ["M", "K", "n_xb", "n_yb", "n_yt", "n_rt", "n_cache", "dtype"]
    run(gemv, configs, config_label, reducer=get_total_time_gemv)


def run_mmtv(B, M, N):
    best_configs = []
    ytile = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    tasklets = [1, 2, 4, 8, 16]
    caches = [16, 32, 64, 128, 256]
    configs = []
    if NAIVE:
        tasklets = [16]
        caches = [256]
    for b, y, t, c in product([B], ytile, tasklets, caches):
        if B * y <= 2048 and 32 <= B * y and y * t * 2 <= M and b * y >= 32:
            configs.append((B, M, N, b, y, t, c, "int32"))
    configs_label = ["B", "M", "N", "n_bb", "n_yb", "n_yt", "n_cache", "dtype"]
    res = run(bgemv, configs, configs_label, reducer=get_total_time_gemv)
    best_configs.append(res)

def search():
    global TOT, VERBOSE, NAIVE
    VERBOSE = 1
    start_time = time.time()

    NAIVE = True
    print("NAIVE")
    print("POLYBENCH-L")
    run_va(67108864)
    run_red(33554432)
    run_mtv(8192, 8192)
    run_mtv(256*512, 512)
    run_mmtv(256, 512, 512)
    run_polyva(67108864)
    run_polygemv(8192, 8192)
    print(f"Total time: {time.time() - start_time}, Total count: {total_cnt}/naive\n")

    print("POLYBENCH-S")
    run_va(1048576)
    run_red(524288)
    run_mtv(1024, 1024)
    run_mtv(32*64, 512)
    run_mmtv(32, 64, 512)
    run_polyva(1048576)
    run_polygemv(1024, 1024)
    print(f"Total time: {time.time() - start_time}, Total count: {total_cnt}/naive\n")

    NAIVE = False
    print("SEARCH")
    print("GPT-6B")
    run_mmtv(16, 64, 256)
    run_mmtv(16, 128, 256)
    run_mmtv(16, 256, 256)
    run_mmtv(16, 512, 256)
    run_mmtv(32, 64, 256)
    run_mmtv(32, 128, 256)
    run_mmtv(32, 256, 256)
    run_mmtv(32, 512, 256)
    run_mmtv(64, 64, 256)
    run_mmtv(64, 128, 256)
    run_mmtv(64, 256, 256)
    run_mmtv(64, 512, 256)
    run_mmtv(256, 64, 256)
    run_mmtv(256, 128, 256)
    run_mmtv(256, 256, 256)
    run_mmtv(256, 512, 256)
    print(f"Total time: {time.time() - start_time}, Total count: {total_cnt}/{TOT}\n")

    print("GPT-13B")
    run_mmtv(28, 64, 256)
    run_mmtv(28, 128, 256)
    run_mmtv(28, 256, 256)
    run_mmtv(28, 512, 256)
    run_mmtv(56, 64, 256)
    run_mmtv(56, 128, 256)
    run_mmtv(56, 256, 256)
    run_mmtv(56, 512, 256)
    run_mmtv(112, 64, 256)
    run_mmtv(112, 128, 256)
    run_mmtv(112, 256, 256)
    run_mmtv(112, 512, 256)
    run_mmtv(448, 64, 256)
    run_mmtv(448, 128, 256)
    run_mmtv(448, 256, 256)
    run_mmtv(448, 512, 256)
    print(f"Total time: {time.time() - start_time}, Total count: {total_cnt}/{TOT}\n")

    print("GPT-MTV")
    run_mtv(12288, 4096)
    run_mtv(4096, 4096)
    run_mtv(16384, 4096)
    run_mtv(4096, 16384)
    run_mtv(21504, 7168)
    run_mtv(7168, 7168)
    run_mtv(28672, 7168)
    run_mtv(7168, 28672)
    print(f"Total time: {time.time() - start_time}, Total count: {total_cnt}/{TOT}\n")

    NAIVE=True
    print("POLYBENCH-L")
    run_va(67108864)
    run_red(33554432)
    run_mtv(8192, 8192)
    run_mtv(256*512, 512)
    run_mmtv(256, 512, 512)
    run_polyva(67108864)
    run_polygemv(8192, 8192)
    print(f"Total time: {time.time() - start_time}, Total count: {total_cnt}/{TOT}\n")

    print("POLYBENCH-S")
    run_va(1048576)
    run_red(524288)
    run_mtv(1024, 1024)
    run_mtv(32*64, 512)
    run_mmtv(32, 64, 512)
    run_polyva(1048576)
    run_polygemv(1024, 1024)
    print(f"Total time: {time.time() - start_time}, Total count: {total_cnt}/{TOT}\n")

    print()
    print("SENS")
    random_toks = [26, 69, 109, 126, 153, 180, 232, 255, 272, 321, 345, 380, 399, 429, 473, 501]
    random_toks = [399]
    shapes = [(64, T, 256) for T in random_toks] + [(64, T, T) for T in random_toks]

def eval():
    va.warmup = 100
    va.repeat = 1000
    print(va.benchmark(L=67108864, n_b=2048, n_t=16, n_c=64, dtype="int32"))

    gemv.warmup = 100
    gemv.repeat = 1000
    print(gemv.benchmark(M=32*64, K=512, n_xb=1, n_yb=256, n_yt=8, n_rt=1, n_cache=16, dtype="int32"))

    gemv.profile = "gemv_poly"
    print(gemv.benchmark(M=1024, K=1024, n_xb=1, n_yb=256, n_yt=8, n_rt=1, n_cache=16, dtype="int32"))

if __name__ == "__main__":
    # search()
    NAIVE = True
    run_va(67108864)
    run_mtv(32*64, 512)
    run_polygemv(1024, 1024)
