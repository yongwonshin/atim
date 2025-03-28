from prim_util import *


def save_search_params(
    op_type, M, N, K, tilings, naive, jsonfile="./reproduced/search_parameters.json"
):
    best_dpus, best_tasklets, best_cache = tilings
    with open(jsonfile, "r") as f:
        search_params = json.load(f)
        key = f"{op_type}_{M}_{N}_{K}"
        tmp = search_params.get(key, {})
        prim_key = "prim" if naive else "prim-search"
        tmp[prim_key] = {
            "dpus": best_dpus,
            **({"tasklets": best_tasklets, "cache_size": best_cache} if not naive else {}),
        }
        search_params[key] = tmp
        with open(jsonfile, "w") as f:
            json.dump(search_params, f, indent=4)


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
        print(
            "PrIM-Search",
            op_type,
            M,
            N,
            K,
            "founded DPUs: ",
            best_dpus,
            "tasklets: ",
            best_tasklets,
            "cache: ",
            best_cache,
        )
    return best_tiling


if __name__ == "__main__":
    for naive in [True, False]:
        print(("PrIM" if naive else "PrIM-Search") + " search for Tensor programs")
        for task in poly_tasks:
            if not task[0]:
                continue
            params = search(*task, naive=naive)
            save_search_params(*task, params, naive)
        print(("PrIM" if naive else "PrIM-Search") + " search for GPT-J")
        for task in gptj_tasks:
            params = search(*task, naive=naive)
            save_search_params(*task, params, naive)
