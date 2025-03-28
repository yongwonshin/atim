import pandas as pd
import matplotlib.pyplot as plt
from plot_gpt import plot_gpt
from plot_polybench import plot_polybench
from plot_branch_opt import plot_branch_opt
import argparse

plt.rcParams.update({'font.family':'Arial'})

argparser = argparse.ArgumentParser()
argparser.add_argument("--reproduced", type=bool, action="store_true", default=True)
args = argparser.parse_args()

directory = "reproduced" if args.reproduced else "results"

df_gptj = pd.read_csv(f"../{directory}/result_gptj.csv")
df_gptj = df_gptj.fillna(0).reset_index(drop=True)
normalization_factor = df_gptj.iloc[:, 5:8].astype(float).sum(axis=1)
df_gptj.iloc[:, 5:] = df_gptj.iloc[:, 5:].astype(float).div(normalization_factor, axis=0)
plot_gpt(df_gptj, f"../{directory}/plot_gpt.pdf")

df_poly = pd.read_csv(f"../{directory}/result_poly.csv")
df_poly = df_poly.fillna(0).reset_index(drop=True)
normalization_factor = df_poly.iloc[:, 5:8].astype(float).sum(axis=1)
df_poly.iloc[:, 5:] = df_poly.iloc[:, 5:].astype(float).div(normalization_factor, axis=0)
plot_polybench(df_poly, f"../{directory}/plot_polybench.pdf")

df_opt = pd.read_csv(f"../{directory}/result_opt.csv")
df_opt = df_opt.fillna(0)
df_opt.iloc[:, 2:] = df_opt.iloc[:, 2:].astype(float)
plot_branch_opt(df_opt, f"../{directory}/plot_branch_opt.pdf")