from tune import tune
from tasks import poly_tasks, gptj_tasks
import argparse
from query_result import query
import pandas as pd
import os

def eval_group(df, tasks):
    results = []
    for op_type, m, n, k  in tasks:
        fname = f"./reproduced/tuned/{op_type}_{m}_{n}_{k}"
        try:
            if not os.path.exists(fname):
                print(f"File not found: {fname}")
                time_tuple = (0, 0, 0)
            else:
                time_tuple = query(fname, only_run=True)
        except Exception as e:
            print(f"Error during query: {e}")
            time_tuple = (0, 0, 0)
        results.append(time_tuple)
    df.iloc[:, 13:16] = results

if __name__ == "__main__":
    df_gptj = pd.read_csv("./graph/result_gptj.csv")
    df_poly = pd.read_csv("./graph/result_poly.csv")

    eval_group(df_gptj, gptj_tasks)
    eval_group(df_poly, poly_tasks)

    df_gptj.to_csv("./graph/result_gptj.csv", index=False)
    df_poly.to_csv("./graph/result_poly.csv", index=False)