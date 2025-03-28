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
        parser = argparse.ArgumentParser()
        parser.add_argument("--skip_existing", action="store_true", help="Skip tasks where search parameters already exist")
        args = parser.parse_args()

        for task in poly_tasks:
            if not task[0]:
                continue
            if args.skip_existing and search_param_exists(*task, naive):
                continue
            print(task)
            params = search(*task, naive=naive)
            save_search_params(*task, params, naive)
        print(("PrIM" if naive else "PrIM-Search") + " search for GPT-J")
        for task in gptj_tasks:
            if args.skip_existing and search_param_exists(*task, naive):
                continue
            print(task)
            params = search(*task, naive=naive)
            save_search_params(*task, params, naive)
