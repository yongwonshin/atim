from prim_util import *
import argparse


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
    best_dpus_str = f"{M}*{best_dpus}" if op_type == "mmtv" else best_dpus
    if naive:
        print("PrIM", op_type, M, N, K, "naive DPUs: ", best_dpus_str)
    else:
        print(
            "PrIM-Search",
            op_type,
            M,
            N,
            K,
            "founded DPUs: ",
            best_dpus_str,
            "tasklets: ",
            best_tasklets,
            "cache: ",
            best_cache,
        )
    return best_tiling