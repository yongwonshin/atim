from main_result import *
from itertools import product
# VA
# RED
# MTV
# TTV
# MMTV
# POLYVA
# POLYGEMV1

# VA, TA

def get_tiling(op_type):
    if op_type == "va":
        return vaTile
    elif op_type == "red":
        return crossReductionCache
    elif op_type == "mtv":
        return mtvRTile
    elif op_type == "ttv":
        return ttvRTile
    elif op_type == "mmtv":
        return mmtvRTile
    elif op_type == "poly_va":
        return gevaTile
    elif op_type == "poly_gemv1":
        return polygemvRTile

best_config = {}

# HELPERS
def work_helper(workname, configs, conf_label):
    print("#######WORK", workname)
    workload = get_module(workname)(repeat = 100, warmup = 10)
    workload.use_time_evaluator = False
    for conf in configs:
        workload.test(get_tiling(workname), **dict(zip(conf_label, conf)))
    best_config[workname] = workload.dump_handtune_max()
    print(best_config[workname])
    print()

def va_helper(wl, L):
    dpu_grid = [512, 1024, 1536, 2048]
    tasklets_grid = [16, 20, 24]
    cache_grid = [8, 16, 32, 64, 128]
    configs = [(L, *p, "int32") for p in product(dpu_grid, tasklets_grid, cache_grid)]
    config_label = ["M", "n_xb", "n_t", "n_cache", "dtype"]
    work_helper(wl, configs, config_label)

def ein2_helper(wl, M, N):
    dpus = [512, 1024, 1536, 2048]
    tasklets = [16, 20, 24]
    cache_size = [16, 32, 64, 128, 256]
    configs = [(M, N, *p, "int32") for p in product(dpus, tasklets, cache_size)]
    config_label = ["M", "K", "n_yb", "n_t", "n_cache", "dtype"]
    work_helper(wl, configs, config_label)

def ein3_helper(wl, M, N, K):
    ytile = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    tasklets = [16, 20, 24]
    cache_size = [8, 16, 32, 64, 128]
    configs = []
    for y, t, c in product(ytile, tasklets, cache_size):
        if (N * y <= 2048 and
            32 <= N * y and
            y * t * 2 <= M and N * y >= 32):
            configs.append((M, N, K, M, y, t, c, "int32"))
    config_label = ["M", "N", "K", "n_bb", "n_yb", "n_t", "n_cache", "dtype"]
    work_helper(wl, configs, config_label)

def eval(workname):
    print("########EVAL ", workname)
    workload = get_module(workname)(repeat = 1000, warmup = 10, verbose = 0)
    workload.time_evaluator = True
    workload.test(get_tiling(workname), **best_config[workname])
    print()


def red_test(L):
    dpus = [256, 512, 1024, 2048]
    tasklets = [16, 24] # 20: correctness issue
    cache_size = [8, 16, 32, 64, 128]
    configs = [(L, d, t, c, "int64") for d, t, c in product(dpus, tasklets, cache_size)]
    config_label = ["M", "n_b", "n_t", "n_cache", "dtype"]
    work_helper("red", configs, config_label)

def va_test(L):
    va_helper("va", L)

def mtv_test(M, K):
    ein2_helper("mtv", M, K)

def ttv_test(M, N, K):
    ein3_helper("ttv", M, N, K)

def mmtv_test(M, N, K):
    ein3_helper("mmtv", M, N, K)

def polyva_test(L):
    va_helper("poly_va", L)

def polygemv1_test(M, K):
    ein2_helper("poly_gemv1", M, K)

if __name__ == "__main__":
    best_config["va"] = {'M': 1048576, 'n_xb': 1024, 'n_t': 20, 'n_cache': 64, 'dtype': 'int32'}
    best_config["red"] = {'M': 524288, 'n_b': 256, 'n_t': 16, 'n_cache': 16, 'dtype': 'int64'}
    best_config["mtv"] = {'M': 1024, 'K': 1024, 'n_yb': 512, 'n_t': 24, 'n_cache': 16, 'dtype': 'int32'}
    best_config["ttv"] = {'M': 32, 'N': 64, 'K': 512, 'n_bb': 32, 'n_yb': 1, 'n_t': 16, 'n_cache': 16, 'dtype': 'int32'}
    best_config["mmtv"] = {'M': 32, 'N': 64, 'K': 512, 'n_bb': 32, 'n_yb': 1, 'n_t': 16, 'n_cache': 16, 'dtype': 'int32'}
    best_config["poly_va"] = {'M': 1048576, 'n_xb': 1024, 'n_t': 16, 'n_cache': 32, 'dtype': 'int32'}
    best_config["poly_gemv1"] = {'M': 1024, 'K': 1024, 'n_yb': 512, 'n_t': 20, 'n_cache': 256, 'dtype': 'int32'}
    # va_test(1048576)
    #red_test(524288)
    #mtv_test(1024, 1024)
    #ttv_test(32, 64, 512)
    # mmtv_test(32, 64, 512)
    #polyva_test(1048576)
    #polygemv1_test(1024, 1024)

    eval("va")
    eval("red")
    eval("mtv")
    eval("ttv")
    eval("mmtv")
    eval("poly_va")
    eval("poly_gemv1")

    # va_test(67108864)
    # red_test(33554432)
    # mtv_test(8192, 8192)
    # ttv_test(256, 512, 256)
    # mmtv_test(256, 512, 256)
    # polyva_test(67108864)
    # polygemv1_test(8192, 8192)

    best_config["va"] = {'M': 67108864, 'n_xb': 2048, 'n_t': 20, 'n_cache': 32, 'dtype': 'int32'}
    best_config["red"] = {'M': 33554432, 'n_b': 1024, 'n_t': 16, 'n_cache': 32, 'dtype': 'int64'}
    best_config["ttv"] = {'M': 256, 'N': 512, 'K': 256, 'n_bb': 256, 'n_yb': 4, 'n_t': 16, 'n_cache': 16, 'dtype': 'int32'}
    best_config["mtv"] = {'M': 8192, 'K': 8192, 'n_yb': 1024, 'n_t': 16, 'n_cache': 16, 'dtype': 'int32'}
    best_config["mmtv"] = {'M': 256, 'N': 512, 'K': 256, 'n_bb': 256, 'n_yb': 4, 'n_t': 16, 'n_cache': 128, 'dtype': 'int32'}
    best_config["poly_va"] = {'M': 67108864, 'n_xb': 2048, 'n_t': 16, 'n_cache': 8, 'dtype': 'int32'}
    best_config["poly_gemv1"] = {'M': 8192, 'K': 8192, 'n_yb': 1024, 'n_t': 16, 'n_cache': 16, 'dtype': 'int32'}

    eval("va")
    eval("red")
    eval("mtv")
    eval("ttv")
    eval("mmtv")
    eval("poly_va")
    eval("poly_gemv1")


