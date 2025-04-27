from prim_util import *
from save_csv import CSVSaver
from parser_utils import get_tune_parser, args_to_tasks
# import random

def eval(op_type, M, N, K, tiling):
    # return (random.random() * 10, random.random() * 10, random.random() * 10, random.random() * 10)
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
    parser = get_tune_parser()
    parser.add_argument("--jsonfile", type=str, default="./reproduced/prim_parameters.json")
    args = parser.parse_args()
    tasks = args_to_tasks(args)

    default_tuple = (0, 0, 0, 0)
    csv = CSVSaver()

    for naive in [True, False]:
        print(("PrIM" if naive else "PrIM-Search"))
        for task in poly_tasks:
            res = default_tuple
            if not task[0]:
                continue
            try:
                params = load_search_params(*task, naive, jsonfile=args.jsonfile)
                print("Evaluate task", task)
                res = eval(*task, params)
                print(f"H2D: {res[0]:.3f} ms, Kernel: {res[1]:.3f} ms, D2H: {res[2]:.3f} ms, Total: {res[3]:.3f} ms")
            except FileNotFoundError as e:
                print(e)
            except Exception as e:
                print(f"Error in {task}: {e}")
            csv.set_prim(task, *res, search=not naive)
            csv.commit()
        print()