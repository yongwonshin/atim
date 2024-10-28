import os, sys
import numpy as np

sys.path.append("/root/dev/tvm/python")
os.chdir("/root/dev/tvm/pim_test/upmem")

from base import cleanup, UPMEMWorkload
from gemv import GEMV, gemvRCTile
from va import VA, vaTile
import tvm
from typing import Tuple, List
from tqdm import tqdm
import time

WHITE = "97"  # 흰색
RED = "91"  # 빨간색
PURPLE = "95"  # 보라색
GREEN = "92"  # 초록색
YELLOW = "93"  # 노란색
CYAN = "96"  # 청록색
BLUE = "34"  # 파란색

all_records = []
opt_level = 7
early_skip = False


def colorize(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"


def colorize_result(times, sources):
    times_str = []
    for level in range(opt_level):
        cur = times[level]
        color = WHITE
        for c in range(level):
            if sources[level] == sources[c]:
                cur = f"${c}"
                color = BLUE
        if color != BLUE:
            if times[0] < cur:
                color = RED
            elif all([t >= cur for t in times]):
                color = CYAN
            elif all([t >= cur for t in times[:level]]):
                color = GREEN
        times_str.append(colorize(cur, color))
    return times_str


def get_min_speedup(times):
    min_times = [-1 for _ in range(opt_level)]
    for row in times:
        for i, t in enumerate(row):
            if t != -1 and (min_times[i] == -1 or t < min_times[i]):
                min_times[i] = t
    b = min_times[0]
    speedups = ["x" if t == -1 or b == -1 else f"{(b / t):.3f}" for t in min_times]
    return min_times, speedups


def run(base_name, instance, schedule, dim_conf, tile_conf, dim_label, tile_label):
    print("\t".join(["IDX", "O0", "O1", "O2", "O3", "O4", "O6", *dim_label, *tile_label]))
    for l, dim in enumerate(dim_conf):
        min_times = []
        for k, tile in enumerate(tile_conf):
            times, sources = [], []
            start = time.time()
            flag = False
            for opt in range(opt_level):
                instance.profile = f"{base_name}_task{l}_opt{opt}"
                instance.index = k
                with tvm.transform.PassContext(config={"tir.UpmemKernelOptimize": opt}):
                    try:
                        if flag:
                            raise ValueError("SKIP")
                        res = instance.test(
                            schedule, **dict(zip(dim_label, dim)), **dict(zip(tile_label, tile))
                        )
                        if res == "ERROR":
                            raise ValueError("ERROR")
                        sources.append(instance.func.imported_modules[0].get_source())
                        times.append(float(res))
                    except KeyboardInterrupt:
                        print("Caught KeyboardInterrupt")
                        sys.exit(1)
                    except:
                        sources.append("ERROR")
                        times.append(-1)
                        if early_skip and (opt == 0 or opt == 1):
                            flag = True
            times_str = colorize_result(times, sources)
            print_dim = dim if k == 0 else ["."] * len(dim_label)
            min_times.append(times)
            end = time.time()
            print(
                "\t".join(
                    [
                        str(x)
                        for x in [
                            k,
                            *times_str,
                            *print_dim,
                            *tile,
                            format((end - start) * 1e3, ".3f"),
                        ]
                    ]
                )
            )

        min_times, speedups = get_min_speedup(min_times)
        all_records.append([base_name, dim, min_times, speedups])

        print("\t".join([str(x) for x in ["MIN", *min_times, *dim]]))
        print("\t".join([str(x) for x in ["SPD", *speedups, *dim]]))
        print()


def log_uniform_int(min_val, max_val, num_vals):
    return [
        int(val)
        for val in np.sort(
            10 ** (np.random.uniform(np.log10(min_val), np.log10(max_val), num_vals))
        )
    ]


def gemv_config():
    dist = log_uniform_int(70, 512, 10)
    dist = [55, 73, 99, 139, 171, 213, 239, 271, 291, 301, 331, 363, 391]
    dim_conf = [(i, j) for i in dist for j in dist]
    # dim_conf = [(256, i) for i in [173, 193, 213, 233]]
    # dim_conf += [(512, i) for i in [373, 393, 413, 433]]
    # dim_conf = [(i, j) for i in [373, 393, 413, 433] for j in [373, 393, 413, 433]]
    dim_conf = [(400, 400)]
    tile_conf = [(4, 4, 16, 16, i, "int32") for i in (16,)]
    # for bm, bk in [(16, 128), (32, 64), (64, 32), (128, 16)]:
    # for c in [4, 8, 16, 32]:
    #     dtype = "int32"
        # dtype = "int32" if M % 2 == 0 and K % 2 == 0 else "int64"
        # tile_conf.append((1, 1, 16, 16, c, dtype))

    dim_label = ["M", "K"]
    tile_label = ["n_xb", "n_yb", "n_yt", "n_rt", "n_cache", "dtype"]
    return (
        "gemv",
        GEMV(warmup=1, repeat=10),
        gemvRCTile,
        dim_conf,
        tile_conf,
        dim_label,
        tile_label,
    )


def va_config():
    dim_conf = log_uniform_int(5000, 200000000, 30)
    tile_conf = []

    for b in [1024, 1536, 2048]:
        for c in [4, 16, 64, 256]:
            tile_conf.append((b, 16, c, "int32"))

    dim_label = ["L"]
    tile_label = ["n_b", "n_t", "n_c", "dtype"]
    return ("va", VA(warmup=1, repeat=5), vaTile, dim_conf, tile_conf, dim_label, tile_label)


cleanup()
run(*gemv_config())
# run(*va_config())

for record in all_records:
    name, dim, times, speedups = record
    print(f"{name} {dim}")
    print("O0\tO1\tO2\tO3\tO4\tO5\tO6\tM\t")
    print("\t".join([str(x) for x in times]))
    print("\t".join([str(x) for x in speedups]))
    print()
