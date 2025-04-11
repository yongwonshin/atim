import pandas as pd


class CSVSaver:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, dtype={col: float for col in range(5, 21)})

    def get_row(self, task):
        try:
            wl, m, n, k = task
            condition = (
                (self.df["Workload"] == wl) &
                (self.df["M"].fillna(-1).astype(int) == m) &
                (self.df["N"].fillna(-1).astype(int) == n) &
                (self.df["K"].fillna(-1).astype(int) == k)
            )
            rows = self.df[condition].index
            if len(rows) == 0:
                return None
            return rows[0]
        except Exception as e:
            print(f"Error in {task}: {e}")
            return None

    def set_cpu_autotuned(self, task, value):
        if not task[0]:
            return
        row = self.get_row(task)
        if not row:
            return
        self.df.loc[row, "CPU-Autotuned"] = value

    def _set_values(self, task, start_col, h2d, kernel, d2h, total):
        if not task[0]:
            return
        row = self.get_row(task)
        if not row:
            return
        self.df.iloc[row, start_col:start_col + 4] = [h2d, kernel, d2h, total]

    def set_prim(self, task, h2d, kernel, d2h, total, search=False):
        self._set_values(task, 9 if search else 5, h2d, kernel, d2h, total)

    def set_prim_search(self, task, h2d, kernel, d2h, total):
        self._set_values(task, 9, h2d, kernel, d2h, total)

    def set_atim(self, task, h2d, kernel, d2h, total):
        self._set_values(task, 13, h2d, kernel, d2h, total)

    def set(self, task, column, value):
        self.df.loc[task, column] = value

    def commit(self):
        self.df.to_csv(self.csv_path, index=False)

class GPTJSaver(CSVSaver):
    def __init__(self):
        super().__init__("./reproduced/result_gptj.csv")

class PolySaver(CSVSaver):
    def __init__(self):
        super().__init__("./reproduced/result_poly.csv")