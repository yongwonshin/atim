from prim_util import *
import argparse


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

    dpus_str = f"{M}*{dpus}" if op_type == "mmtv" else dpus
    print(op_type, M, N, K, "founded DPUs: ", dpus_str, "tasklets: ", tasklets, "cache: ", cache_size)
    print("H2D", best_tuple[0], "Kernel", best_tuple[1], "D2H", best_tuple[2], "Total", total_time)
    print()
    return (*best_tuple, total_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonfile", type=str, default="./reproduced/prim_parameters.json")
    args = parser.parse_args()

    default_tuple = (0, 1000.0, 1000.0, 2000.0)
    df_gptj = pd.read_csv("./reproduced/result_gptj.csv")
    df_poly = pd.read_csv("./reproduced/result_poly.csv")
    for naive in [True, False]:
        start_col = 5 if naive else 9
        print(("PrIM" if naive else "PrIM-Search") + " evaluate for Tensor programs")
        results = []
        for task in poly_tasks:
            if not task[0]:
                results.append(default_tuple)
                continue
            try:
                params = load_search_params(*task, naive, jsonfile=args.jsonfile)
                res = eval(*task, params)
            except Exception as e:
                print(f"Error in {task}: {e}")
                res = default_tuple
            results.append(res)

        df_poly.iloc[:, start_col : start_col + 4] = results
        df_poly.to_csv("./reproduced/result_poly.csv", index=False)

        print(("PrIM" if naive else "PrIM-Search") + " evaluate for GPT-J")
        results = []
        for task in gptj_tasks:
            try:
                params = load_search_params(*task, naive, jsonfile=args.jsonfile)
                res = eval(*task, params)
            except Exception as e:
                print(f"Error in {task}: {e}")
                res = default_tuple
            results.append(res)
        df_gptj.iloc[:, start_col : start_col + 4] = results
        df_gptj.to_csv("./reproduced/result_gptj.csv", index=False)