import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from plot_utils import colors, hatches, adjust_color

def plot_polybench(src_df, filename):
    width_polybench = 19 / 2
    height = 5.3
    bar_width = 0.15
    fontsize = 7
    latency_label = "Normalized latency"

    shapes = [
        ["524288", "1024×1024", "1024×1024", "32×64×512", "32×64×512",  "1048576", "1048576"],
        ["8388608", "4096×4096", "4096×4096", "128×256×512", "128×256×512",  "16777216", "16777216"],
        ["34554432", "8192×8192", "8192×8192", "256×512×512", "256×512×512",  "67108864", "67108864"],
        ["67108864", "8192×16384", "8192×16384", "512×512×512", "512×512×512",  "", ""]
    ]
    label = [
        "(a) 4MB Tensor operation results",
        "(b) 64MB Tensor operation results",
        "(c) 256MB Tensor operation results",
        "(d) 512MB Tensor operation results"
    ]
    start_pos = [0, 7, 14, 21]

    is_workload_prim = lambda wl: wl in ["va", "red", "mtv"]
    is_workload_simplepim = lambda wl: wl in ["va", "red"]

    def create_plot(ax, shape, df_subset, width, threshold, interval=0.5, bar_width=0.25, colors=colors, hatches=hatches, caption="", batch_unit=16):
        workloads = df_subset['Workload']
        x = np.arange(len(workloads))
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(0, threshold)
        ax.yaxis.grid(True, linestyle='--', zorder=0, linewidth=0.5)
        y_ticks = np.arange(0, threshold, interval)
        ax.tick_params(axis='both', which='both', direction='in', length=0)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks], fontsize=fontsize)
        ax.set_xticklabels([])
        ax.set_xticks(x)

        ax.spines['left'].set_capstyle("butt")
        ax.spines['left'].set_clip_on(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

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

        def plot_bar_cpu(ax, x, df):
            kernel_val = df['CPU-Autotuned'].astype(float)
            ax.bar(x, kernel_val, bar_width, bottom=0, label=f'CPU', linewidth=0, zorder=2, color=colors["CPU"], edgecolor="white")
            ax.bar(x, kernel_val, bar_width, linewidth=0.5, zorder=2, edgecolor='black', fill=False)
            for pos, val in zip(x, kernel_val):
                if val > threshold:
                    ax.text(pos, threshold, f'{val:.2f}', ha='center', va='bottom', fontsize=fontsize - 0.5)

        prim_colors = [colors["PrIM" if is_workload_prim(wl) else "PrIMC"] for wl in workloads.values]

        def x_seq(x, bar_width, n):
            return [[x + bar_width * ((-n + 1) / 2 + i)] for i in range(n)]

        for i, workload in enumerate(workloads):
            if is_workload_simplepim(workload):
                seq = x_seq(x[i], bar_width, 5)
                plot_bar_cpu(ax, seq[0], df_subset.iloc[[i]])
                plot_bars(ax, seq[1], df_subset.iloc[[i]], 'PrIM', prim_colors[i], hatches)
                plot_bars(ax, seq[2], df_subset.iloc[[i]], 'PS', colors['PS'], hatches)
                plot_bars(ax, seq[3], df_subset.iloc[[i]], 'SimplePIM', colors['SimplePIM'], hatches)
                plot_bars(ax, seq[4], df_subset.iloc[[i]], 'ATiM', colors['ATiM'], hatches)
            else:
                seq = x_seq(x[i], bar_width, 4)
                plot_bar_cpu(ax, seq[0], df_subset.iloc[[i]])
                plot_bars(ax, seq[1], df_subset.iloc[[i]], 'PrIM', prim_colors[i], hatches)
                plot_bars(ax, seq[2], df_subset.iloc[[i]], 'PS', colors['PS'], hatches)
                plot_bars(ax, seq[3], df_subset.iloc[[i]], 'ATiM', colors['ATiM'], hatches)

        for i, label in enumerate(workloads):
            l = label.upper()
            ax.text(x[i], -0.04, l, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=6.5)
            ax.text(x[i], -0.2, shape[i], ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=5.5)
        if caption:
            ax.text(0.5, -0.35, caption, ha='center', va='top',
                    transform=ax.transAxes, fontsize=8, fontname='Times New Roman')

    fig = plt.figure(figsize=(7.48,  height / 2.54), dpi=300, constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[7.3, 7], wspace=0.01, figure=fig)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for i in range(4):
        indices = [7 * i + j for j in [1, 2, 6, 3, 4, 0, 5]]
        create_plot(
            axs[i],
            shapes[i],
            src_df.iloc[indices],
            width_polybench,
            threshold=1.4,
            interval=0.2,
            bar_width=bar_width,
            colors=colors,
            hatches=hatches,
            caption=label[i])
    axs[0].set_ylabel(latency_label, fontsize=fontsize)
    axs[2].set_ylabel(latency_label, fontsize=fontsize)
    axs[1].set_yticklabels([])
    axs[3].set_yticklabels([])

    handles = [
        Rectangle((0, 0.1), 1, 0.8, facecolor=colors["CPU"], label='CPU-autotuned'),
        Rectangle((0, 0.1), 1, 0.8, facecolor=colors["PrIM"], label='PrIM'),
        Rectangle((0, 0.1), 1, 0.8, facecolor=colors["PrIMC"], label='PrIM (E)'),
        Rectangle((0, 0.1), 1, 0.8, facecolor=colors["PS"], label='PrIM+search'),
        Rectangle((0, 0.1), 1, 0.8, facecolor=colors["SimplePIM"], label="SimplePIM"),
        Rectangle((0, 0.1), 1, 0.8, facecolor=colors["ATiM"], label='ATiM'),
        Rectangle((0, 0.1), 1, 0.8, facecolor='#444444', label='H2D', edgecolor="white", hatch=hatches["H2D"]),
        Rectangle((0, 0.1), 1, 0.8, facecolor='#444444', label='Kernel', edgecolor="white"),
        Rectangle((0, 0.1), 1, 0.8, facecolor='#444444', label='After Kernel', edgecolor="white", hatch=hatches["After"])
    ]

    rect = Rectangle((0.93 - 0.055, 0.07 - 0.05), 0.15, 0.5,
                    transform=fig.transFigure, facecolor='white')
    fig.patches.append(rect)
    fig.legend(handles=handles, loc='upper center', ncol=1, fontsize=fontsize - 1.5, bbox_to_anchor=(0.94, 0.53), borderpad=0.5)
    fig.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
