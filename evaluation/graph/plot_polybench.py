# %%
import os
os.chdir("/root/atim/evaluation/graph")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from .plot_utils import colors, hatches, adjust_color
import matplotlib.font_manager as fm
from functools import reduce


# %%


def plot_polybench(src_df, filename):
    width_ratio = [1, 1, 1, 1]
    height_ratio = [1, 1]
    ax_width = 9.5
    plot_height = 5.3
    bar_width = 0.19
    fontsize = 7
    threshold = 1.1
    latency_label = "Normalized latency"
    speedup_label = "Speedup over CPU"

    sequence = ["va", "geva", "red", "mtv", "gemv", "ttv", "mmtv"]
    captions = ["VA.", "GEVA.", "RED.", "MTV.", "GEMV.", "TTV (M×N×512).", "MMTV (M×N×512)."]
    permutation = [0, 5, 1, 2, 6, 3, 4]
    fusion = [[0, 2], [2, 3], [3, 5], [5, 7]]
    shape_labels = ["Length", "Length", "M, N", "M, N"]

    shapes_info = {
        "va": ["1048576", "16777216", "67108864"],
        "geva": ["1048576", "16777216", "67108864"],
        "red": ["524288", "8388608", "34554432", "67108864"],
        "mtv": ["1024×1024", "4096×4096", "8192×8192", "8192×16384"],
        "gemv": ["1024×1024", "4096×4096", "8192×8192", "8192×16384"],
        "ttv": ["32×64×512", "128×256×512", "256×512×512", "512×512×512"],
        "mmtv": ["32×64×512", "128×256×512", "256×512×512", "512×512×512"]
    }

    original_size_labels = ["4MB", "64MB", "256MB", "512MB"]
    start_pos = [0, 7, 14, 21]

    is_workload_prim = lambda wl: wl in ["va", "red", "mtv"]
    is_workload_simplepim = lambda wl: wl in ["va", "red"]

    def create_plot(plot_idx, fig, interval, bar_width):
        start, end = fusion[plot_idx]
        indices = []
        shapes = []
        size_labels = []
        num_shape_part = []
        num_shapes = 0
        for i in range(start, end):
            shape = shapes_info[sequence[i]]
            l = len(shape)
            shapes += shape
            num_shapes += l
            num_shape_part.append(l)
            size_labels += original_size_labels[:l]
            indices += [7 * j + permutation[i] for j in range(l)]

        df_subset = src_df.iloc[indices]
        workloads = df_subset['Workload']

        if plot_idx < 2:
            plot_row, plot_col_start, plot_col_end = 0, start, end
        else:
            plot_row, plot_col_start, plot_col_end = 1, start - 3, end - 3
        ax = fig.add_subplot(gs[plot_row, plot_col_start:plot_col_end])

        x = np.arange(num_shapes)
        ax.set_xlim(-0.5, num_shapes - 0.5)
        ax.set_ylim(0, threshold)
        ax.yaxis.grid(True, linestyle='--', zorder=0, linewidth=0.5)
        y_ticks = np.arange(interval, threshold, interval)
        ax.tick_params(axis='both', which='both', direction='in', length=0)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks], fontsize=5.5)
        ax.set_xticklabels([])
        ax.set_xticks(x)

        ax.spines['left'].set_capstyle("butt")
        ax.spines['left'].set_clip_on(False )
        ax.spines['right'].set_capstyle("butt")
        ax.spines['right'].set_clip_on(False)
        ax.spines['top'].set_visible(False)

        ax_twin = ax.twinx()
        ax_twin.spines['top'].set_visible(False)

        def plot_bars(ax, x, df, label_prefix, color, hatch):
            h2d_val = df[f'{label_prefix}-H2D'].astype(float)
            kernel_val = df[f'{label_prefix}-Kernel'].astype(float)
            after_val = df[f'{label_prefix}-After'].astype(float)
            pos = x
            ax.bar(pos, h2d_val, bar_width,
                linewidth=0, zorder=2,label=f'{label_prefix}-H2D',
                color=color, edgecolor="white", hatch=hatch['H2D'])
            ax.bar(pos, kernel_val, bar_width, bottom=h2d_val,
                linewidth=0, zorder=2, label=f'{label_prefix}-Kernel',
                color=color, hatch=hatch['Kernel'])
            adjusted_color = adjust_color(color)
            ax.bar(pos, after_val, bar_width, bottom=h2d_val + kernel_val,
                linewidth=0, zorder=2, label=f'{label_prefix}-After',
                color=adjusted_color, edgecolor="white", hatch=hatch['After'])
            ax.bar(pos, h2d_val + kernel_val + after_val, bar_width,
                linewidth=0.5, zorder=2, edgecolor='black', fill=False)
            total_height = (h2d_val + kernel_val + after_val).tolist()
            for i in range(len(x)):
                if total_height[i] > threshold:
                    ax.text(x[i], threshold, f'{total_height[i]:.2f}', ha='center', va='bottom', fontsize=fontsize - 0.5)

        def get_cpu_val(df):
            kernel_val = df['CPU-Autotuned'].astype(float)
            atim_val = df["ATiM-H2D"].astype(float) + df["ATiM-Kernel"].astype(float) + df["ATiM-After"].astype(float)
            return kernel_val / atim_val

        prim_colors = [colors["PrIM" if is_workload_prim(wl) else "PrIMC"] for wl in workloads.values]

        def x_seq(x, bar_width, n):
            return [[x + bar_width * ((-n + 1) / 2 + i)] for i in range(n)]

        cpu_xs = []
        cpu_vals = []
        for i, workload in enumerate(workloads):
            if is_workload_simplepim(workload):
                seq = x_seq(x[i], bar_width, 4)
                cpu_xs.append(seq[3][0])
                plot_bars(ax, seq[0], df_subset.iloc[[i]], 'PrIM', prim_colors[i], hatches)
                plot_bars(ax, seq[1], df_subset.iloc[[i]], 'PS', colors['PS'], hatches)
                plot_bars(ax, seq[2], df_subset.iloc[[i]], 'SimplePIM', colors['SimplePIM'], hatches)
                plot_bars(ax, seq[3], df_subset.iloc[[i]], 'ATiM', colors['ATiM'], hatches)
            else:
                seq = x_seq(x[i], bar_width, 3)
                cpu_xs.append(seq[2][0])
                plot_bars(ax, seq[0], df_subset.iloc[[i]], 'PrIM', prim_colors[i], hatches)
                plot_bars(ax, seq[1], df_subset.iloc[[i]], 'PS', colors['PS'], hatches)
                plot_bars(ax, seq[2], df_subset.iloc[[i]], 'ATiM', colors['ATiM'], hatches)
            cpu_vals.append(get_cpu_val(df_subset.iloc[[i]]))

        if end - start == 2: # hardcode
            ax_twin.plot(cpu_xs[:num_shapes // 2], cpu_vals[:num_shapes // 2],
                          marker='o', markersize=4,linewidth=1.4, color=colors["CPU"], label="CPU", zorder=3)
            ax_twin.plot(cpu_xs[num_shapes // 2:], cpu_vals[num_shapes // 2:],
                         marker='o', markersize=4,linewidth=1.4, color=colors["CPU"], label="CPU", zorder=3)
        else:
            ax_twin.plot(cpu_xs, cpu_vals,marker='o', markersize=4,linewidth=1.4,
                         color=colors["CPU"], label="CPU", zorder=3)
        ax_twin.axhline(1, color=colors["CPU"], lw=0.8, linestyle="--")

        raw_interval = ax_twin.get_ylim()[1] / 5
        twin_interval = 1
        if raw_interval < 0.7:
            twin_interval = 0.6
        elif raw_interval < 1:
            twin_interval = 1
        elif raw_interval < 2:
            twin_interval = 1.5
        else:
            twin_interval = 5
        twin_threshold = twin_interval * 5.5

        ax_twin.set_xlim(-0.5, len(x) - 0.5)
        ax_twin.set_ylim(0, twin_threshold)
        twin_y_ticks = np.arange(twin_interval, twin_threshold, twin_interval)
        ax_twin.tick_params(axis='both', which='both', direction='in', length=0)
        ax_twin.set_yticks(twin_y_ticks)
        ax_twin.set_yticklabels([f"{tick:.1f}" for tick in twin_y_ticks], fontsize=5.5)

        for i in range(num_shapes):
            ax.text(x[i], -0.04, size_labels[i], ha='center', va='top',
                    transform=ax.get_xaxis_transform(), fontsize=6)
            slabel = shapes[i]
            if workload in ["mtv", "gemv", "polygemv"]:
                m, k = slabel.split("×")
                slabel = f"{m},{k}"
            elif workload in ["ttv", "mmtv"]:
                m, k, _ = slabel.split("×")
                slabel = f"{m},{k}"
            ax.text(x[i], -0.19, slabel, ha='center', va='top',
                    transform=ax.get_xaxis_transform(), fontsize=5)
        for i in range(start, end):
            caption = f"({chr(97 + i)}) {captions[i].upper()}"
            ax.text((i - start + 0.5) / (end - start), -0.37, caption, ha='center', va='top',
                    transform=ax.transAxes, fontsize=8, fontname='Times New Roman')

        ax.spines['left'].set_bounds(-0.35, threshold)
        ax.spines['right'].set_bounds(-0.35, threshold)
        h0 = ax.axhline(-0.35, color='black', xmin=-0.12 / (end - start), lw=0.6)
        h0.set_clip_on(False)
        h1 = ax.axhline(0, color='black', xmin=-0.12 / (end - start), lw=0.6)
        h1.set_clip_on(False)

        ax.text(-0.55, -0.04, "Size", fontsize=6, ha='right', va='top', transform=ax.get_xaxis_transform())
        ax.text(-0.55, -0.19, shape_labels[plot_idx], fontsize=5, ha='right', va='top', transform=ax.get_xaxis_transform())

        if end - start == 2: # hardcode
            v0 = ax.axvline(num_shapes / 2 - 0.5, ymin=-0.3, color='black', lw=0.6)
            v0.set_clip_on(False)

        ax.set_ylabel(latency_label, fontsize=6, labelpad=3)
        ax_twin.set_ylabel(speedup_label, fontsize=6, labelpad=2)

        return ax, ax_twin


    fig = plt.figure(figsize=(7.48,  plot_height / 2.54), dpi=300, constrained_layout=True)
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1.15, 0.85], height_ratios=height_ratio, wspace=0.0, hspace=0.01, figure=fig)
    ax_twins = []
    axs = []

    for i in range(4):
        ax, ax_twin = create_plot(i, fig, 0.2, bar_width if i >= 1 else 0.16)
        axs.append(ax)
        ax_twins.append(ax_twin)

    handles = [
        Rectangle((0, 0.1), 1, 1, facecolor=colors["PrIM"], label='PrIM'),
        Rectangle((0, 0.1), 1, 0.8, facecolor=colors["PrIMC"], label='PrIM (E)'),
        Rectangle((0, 0.1), 1, 0.8, facecolor=colors["PS"], label='PrIM+search'),
        Rectangle((0, 0.1), 1, 0.8, facecolor=colors["SimplePIM"], label="SimplePIM"),
        Rectangle((0, 0.1), 1, 0.8, facecolor=colors["ATiM"], label='ATiM'),
        Rectangle((0, 0.1), 1, 0.8, facecolor='#444444', label='H2D', edgecolor="white", hatch=hatches["H2D"]),
        Rectangle((0, 0.1), 1, 0.8, facecolor='#444444', label='Kernel', edgecolor="white"),
        Rectangle((0, 0.1), 1, 0.8, facecolor='#444444', label='After Kernel', edgecolor="white", hatch=hatches["After"]),
        Line2D([0], [0], color=colors["CPU"], marker='o', linestyle='-', label='ATiM\'s\nSpeedup over\nCPU-autotuned', linewidth=1.4, markersize=4),
    ]

    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=fontsize - 1, bbox_to_anchor=(0.895, 0.91),
               borderpad=0.6, handlelength=1.2, handletextpad=0.6, columnspacing=0.8)
    fig.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")