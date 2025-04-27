import pandas as pd


class CSVSaver:
    def __init__(self):
        self.df_poly = pd.read_csv("./reproduced/result_poly.csv", dtype={col: float for col in range(5, 21)})
        self.df_gptj = pd.read_csv("./reproduced/result_gptj.csv", dtype={col: float for col in range(5, 21)})

    def get_row(self, df, task):
        try:
            if not task or not task[0]:
                return None
            wl, m, n, k = task
            condition = (
                (df["Workload"] == wl) &
                (df["M"].fillna(-1).astype(int) == m) &
                (df["N"].fillna(-1).astype(int) == n) &
                (df["K"].fillna(-1).astype(int) == k)
            )
            rows = df[condition].index
            return rows[0] if len(rows) > 0 else None
        except Exception as e:
            print(f"Error in get_row for task {task}: {e}")
            return None

    def set_cpu_autotuned(self, task, value):
        for df in [self.df_poly, self.df_gptj]:
            row = self.get_row(df, task)
            if row is not None:
                df.at[row, "CPU-Autotuned"] = value

    def set_latency_values(self, task, start_col, h2d, kernel, d2h, total):
        for df in [self.df_poly, self.df_gptj]:
            row = self.get_row(df, task)
            if row is not None:
                df.iloc[row, start_col:start_col + 4] = [h2d, kernel, d2h, total]

    def set_prim(self, task, h2d, kernel, d2h, total, search=False):
        self.set_latency_values(task, 9 if search else 5, h2d, kernel, d2h, total)

    def set_prim_search(self, task, h2d, kernel, d2h, total):
        self.set_latency_values(task, 9, h2d, kernel, d2h, total)

    def set_atim(self, task, h2d, kernel, d2h, total):
        self.set_latency_values(task, 13, h2d, kernel, d2h, total)

    def commit(self):
        self.df_gptj.to_csv("./reproduced/result_gptj.csv", index=False)
        self.df_poly.to_csv("./reproduced/result_poly.csv", index=False)