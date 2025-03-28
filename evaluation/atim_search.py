from tune import tune
from tasks import poly_tasks, gptj_tasks


def tune_group(tasks):
    for op_type, m, n, k in tasks:
        if not op_type:
            continue
        try:
            tune(op_type, m, n, k, f"./reproduced/tuned/{op_type}_{m}_{n}_{k}", reuse_cost_model=False)
        except Exception as e:
            print(f"Error: {op_type}, {m}, {n}, {k}")
            print(e)
            continue

tune_group(gptj_tasks)
tune_group(poly_tasks)