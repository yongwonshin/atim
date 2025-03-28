from prim_util import *


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


if __name__ == "__main__":
    df_gptj = pd.read_csv("./reproduced/result_gptj.csv")
    df_poly = pd.read_csv("./reproduced/result_poly.csv")
    for naive in [True, False]:
        start_col = 5 if naive else 9
        print(("PrIM" if naive else "PrIM-Search") + " evaluate for Tensor programs")
        results = []
        for task in poly_tasks:
            if not task[0]:
                results.append((0, 0, 0))
                continue
            params = load_search_params(*task, naive)
            res = eval(*task, params)
            results.append(res)
        df_poly.iloc[:, start_col : start_col + 4] = results
        df_poly.to_csv("./reproduced/result_poly.csv", index=False)

        print(("PrIM" if naive else "PrIM-Search") + " evaluate for GPT-J")
        results = []
        for task in gptj_tasks:
            params = load_search_params(*task, naive)
            res = eval(*task, params)
            results.append(res)
        df_gptj.iloc[:, start_col : start_col + 4] = results
        df_gptj.to_csv("./reproduced/result_gptj.csv", index=False)