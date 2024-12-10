import logging
import tempfile

import numpy as np
import pytest
import tvm
import sys
import time
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.local_rpc import LocalRPC
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule
from typing import Callable
import os
from tvm import te, runtime, topi, tir
from tvm.meta_schedule import Database
from tvm.meta_schedule.database import JSONDatabase
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", default="bench_autotuner_result", type=str)
args = parser.parse_args()

# target = Target("upmem --num-cores=96")


def query(file_path):
    best_gflops_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                # "Best GFLOPs: " 패턴을 가진 줄에서 숫자 추출
                if "Trial #" in line:
                    match = re.search(r"Best GFLOPs: ([0-9.]+)", line)
                    if match:
                        best_gflops = float(match.group(1))
                        best_gflops_list.append(best_gflops)
                    else:
                        if len(best_gflops_list) > 0:
                            best_gflops_list.append(best_gflops_list[-1])
                        else:
                            best_gflops_list.append(0.0)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return best_gflops_list


def plot_gflops(lists, filename="gflops_plot.pdf"):
    # Define colors and labels
    colors = ["#FF3BFB", "#4CAF50", "#FB8C00", "#3b88fd"]
    labels = [
        "None (default TVM)",
        # "Without reduction tiling",
        "Balanced sampling",
        "Adaptive epsilon-greedy",
        "All (ATiM)",
    ]

    overall_max = max(max(gflops_list) for gflops_list in lists)
    overall_length = max(len(gflops_list) for gflops_list in lists)  # Maximum x-axis length

    plt.figure(figsize=(8.85 / 2.54, 3.3 / 2.54))
    for idx, (gflops_list, color, label) in enumerate(zip(lists, colors, labels), start=1):
        trials = range(len(gflops_list))
        plt.plot(trials, gflops_list, label=label, color=color, linewidth=1)

    plt.title("", fontsize=6)
    plt.xlabel("The number of trials", fontsize=6, labelpad=2)
    plt.ylabel("GFLOPS", fontsize=5.5, labelpad=2)
    plt.legend(fontsize=5.6)
    plt.grid(True, linewidth=0.25)
    plt.yticks(np.arange(0, overall_max + 5, 5), fontsize=5.5)
    plt.xticks(fontsize=6)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)  # Bottom axis spine width
    ax.spines["left"].set_linewidth(0.5)  # Left axis spine width

    ax.tick_params(length=0)  # Set tick width to 0.5pt

    plt.xlim(0, overall_length)  # Ensure the x-axis stays within the data range

    plt.tight_layout(pad=0.05)  # Reduce padding around the plot

    plt.savefig(filename + "_png", format="png")

    try:
        plt.savefig(filename, format="pdf")
        print(f"Plot saved as {filename}")
    except Exception as e:
        print(f"Failed to save as PDF. Error: {e}. Saving as PNG instead.")
        png_filename = filename.replace(".pdf", ".png")
        plt.savefig(png_filename, format="png")
        print(f"Plot saved as {png_filename}")
    finally:
        plt.close()


if __name__ == "__main__":
    base_filepath = f"{args.workdir}_rfactor+cache/logs/tvm.meta_schedule.logging.task_0_main.log"
    prim_filepath = f"{args.workdir}_prim+cache/logs/tvm.meta_schedule.logging.task_0_main.log"
    balanced_filepath = (
        f"{args.workdir}+cache+balanced/logs/tvm.meta_schedule.logging.task_0_main.log"
    )
    epsilon_filepath = (
        f"{args.workdir}+cache+epsilon/logs/tvm.meta_schedule.logging.task_0_main.log"
    )
    all_filepath = (
        f"{args.workdir}+cache+balanced+epsilon/logs/tvm.meta_schedule.logging.task_0_main.log"
    )
    base_flops = query(base_filepath)
    prim_flops = query(prim_filepath)
    balanced_flops = query(balanced_filepath)
    epsilon_flops = query(epsilon_filepath)
    all_flops = query(all_filepath)

    plt.rcParams.update({"font.family": "Arial"})

    assert len(base_flops) == 1000
    assert len(prim_flops) == 1000
    assert len(balanced_flops) == 1000
    assert len(epsilon_flops) == 1000
    assert len(all_flops) == 1000
    lists = [base_flops, epsilon_flops, balanced_flops, all_flops]
    # print(list)
    # plot_gflops([base_flops, epsilon_flops, balanced_flops, all_flops])
    for l in lists:
        print(l[-1])
