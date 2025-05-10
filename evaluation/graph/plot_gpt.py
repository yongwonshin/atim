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

# %%
def plot_gpt(src_df, filename):
    width_mmtv = 14
    width_mtv = 5
    height = 2.1
    height_table = 0.4
    bar_width = 0.2
    fontsize = 7
    ylabel_offset = 0.05
    ylabel_interval = 0.19
    y_params = 1.11
    latency_label = "Normalized latency  "

    threshold = 3.2 # 상위 12행
    interval = 0.5

    def create_plot(ax, df_subset, width, threshold, interval=0.5, bar_width=0.25, colors=colors, hatches=hatches, caption="", batch_unit=16):
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_capstyle("butt")
        ax.spines['bottom'].set_visible(False)
        x = np.arange(len(df_subset['Workload']))
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(0, threshold)
        ax.yaxis.grid(True, linestyle='--', zorder=0, linewidth=0.5)
        y_ticks = np.arange(0, threshold, interval)
        ax.tick_params(axis='both', which='both', direction='in', length=0)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:.1f}" if tick != 0 else "" for tick in y_ticks], fontsize=6)
        ax.set_xticklabels([])
        ax.set_xticks(x)

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
                    ax.text(x[i], threshold, f'{total_height[i]:.2f}', ha='center', va='bottom', fontsize=fontsize)

        def plot_bar_cpu(ax, x, df):
            kernel_val = df['CPU-Autotuned'].astype(float)
            ax.bar(x, kernel_val, bar_width, bottom=0, label=f'CPU', linewidth=0, zorder=2, color=colors["CPU"], edgecolor="white")
            ax.bar(x, kernel_val, bar_width, linewidth=0.5, zorder=2, edgecolor='black', fill=False)
            for pos, val in zip(x, kernel_val):
                if val > threshold:
                    ax.text(pos, threshold, f'{val:.2f}', ha='center', va='bottom', fontsize=fontsize - 0.5)

        is_mmtv = "mmtv" in df_subset["Workload"].values[0]
        prim_label = "PrIMC" if is_mmtv else "PrIM"


        plot_bars(ax, x - bar_width, df_subset, "PrIM", colors[prim_label], hatches)
        plot_bars(ax, x, df_subset, 'PS', colors['PS'], hatches)
        plot_bars(ax, x + bar_width, df_subset, 'ATiM', colors['ATiM'], hatches)

        cpu_val = df_subset['CPU-Autotuned'].astype(float)
        cpu_val /= df_subset["ATiM-H2D"].astype(float) + df_subset["ATiM-Kernel"].astype(float) + df_subset["ATiM-After"].astype(float)

        if is_mmtv:
            for j in range(3):
                s, e = j * 4, j * 4 + 4
                p = ax_twin.plot(x[s:e] + bar_width, cpu_val[s:e], marker='o', markersize=4,
                            linewidth=1.4, color=colors["CPU"], label="CPU", zorder=3)
                p[0].set_clip_on(False)
        else:
            p = ax_twin.plot(x + bar_width, cpu_val, marker='o', markersize=4,
                        linewidth=1.4, color=colors["CPU"], label="CPU", zorder=3)
            p[0].set_clip_on(False)
        ax_twin.axhline(1, color=colors["CPU"], lw=0.8, linestyle="--")

        ax.set_ylabel(latency_label, fontsize=fontsize - 0.2, labelpad=8 if is_mmtv else 5)
        ax_twin.set_ylabel("Speedup over CPU   ", fontsize=fontsize - 0.2)

        ypos = lambda x: -ylabel_offset - ylabel_interval * x
        cconf = dict(ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=fontsize - 0.5)
        hconf = dict(ha="right", va="top", transform=ax.get_xaxis_transform(), fontsize=fontsize - 0.5)

        if is_mmtv:
            B_labels = df_subset["N"].values
            M_labels = df_subset['M'].values
            ax.text(-0.5 - 0.37, -ylabel_offset, 'Batch', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=fontsize)
            ax.text(-0.5 - 0.37, -ylabel_offset - ylabel_interval, 'Token', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=fontsize)
            for i, label in enumerate(B_labels):
                if label == 16 or label == 28:
                    batch = 1
                elif label == 64 or label == 112:
                    batch = 4
                elif label == 256 or label == 448:
                    batch = 16
                if i % 4 == 1:
                    ax.text((x[i] + x[i+1]) / 2, -ylabel_offset, batch, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=fontsize)
            for i, label in enumerate(M_labels):
                ax.text(x[i], -ylabel_offset - ylabel_interval, label, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=fontsize)
            for i in range(len(x) - 1):
                yy = ax.axvline((x[i] + x[i+1]) / 2, color='black', ymin=(-height_table / height) * 2, ymax=0.01 if x[i] % 4 == 3 else (-height_table / height), lw=0.5, linestyle="--")
                yy.set_clip_on(False)
        else:
            M_labels = df_subset['N'].values
            K_labels = df_subset['K'].values
            ax.text(-0.5 - 0.15, -ylabel_offset, 'M', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=fontsize)
            ax.text(-0.5 - 0.15, -ylabel_offset - ylabel_interval, 'K', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=fontsize)
            for i, label in enumerate(M_labels):
                ax.text(x[i], -ylabel_offset, label, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=fontsize)
            for i, label in enumerate(K_labels):
                ax.text(x[i], -ylabel_offset - ylabel_interval, label, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=fontsize)
            for i in range(len(x) - 1):
                yy = ax.axvline((x[i] + x[i+1]) / 2, color='black', ymin=-(height_table / height) * 2, ymax=0, lw=0.5, linestyle="--")
                yy.set_clip_on(False)

        rpt = threshold * height_table / height
        xmin = (-0.7 if is_mmtv else -0.4) / len(x)
        h0 = ax.axhline(-0, color='black',xmin=xmin, lw=1)
        hl = ax.axhline(-rpt, color='black', lw=0.6, xmin=xmin, linestyle="--")
        h2 = ax.axhline(-rpt * 2, color='black',xmin=xmin, lw=0.6)

        h0.set_clip_on(False)
        hl.set_clip_on(False)
        h2.set_clip_on(False)

        ax.spines['left'].set_bounds(-rpt * 2, threshold)
        ax.spines['left'].set_capstyle("butt")
        ax.spines['left'].set_clip_on(False)
        ax.spines['right'].set_bounds(-rpt * 2, threshold)
        ax.spines['right'].set_capstyle("butt")
        ax.spines['right'].set_clip_on(False)

        raw_threshold = ax_twin.get_ylim()[1]
        if is_mmtv:
            twin_interval = 1.2
        else:
            twin_interval = 1.5
        twin_threshold = twin_interval * 5.5

        ax_twin.set_xlim(-0.5, len(x) - 0.5)
        ax_twin.set_ylim(0, twin_threshold)
        twin_y_ticks = np.arange(twin_interval, twin_threshold, twin_interval)
        ax_twin.tick_params(axis='both', which='both', direction='in', length=0)
        ax_twin.set_yticks(twin_y_ticks)
        ax_twin.set_yticklabels([f"{tick:.1f}" for tick in twin_y_ticks], fontsize=6)

        if caption:
            ax.text(0.5, -0.44, caption, ha='center', va='top',
                    transform=ax.transAxes, fontsize=8, fontname='Times New Roman')

    fig = plt.figure(figsize=(7.48, (height* 2 + 2) / 2.54), dpi=300, constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[width_mmtv, width_mtv], wspace=0.05, hspace=0.03, figure=fig)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]

    create_plot(axs[0], src_df.iloc[:12], width_mmtv, threshold=1.1, interval=0.2,
                bar_width=bar_width, colors=colors, hatches=hatches, caption="(a) MMTV in GPT-J 6B.", batch_unit=16)
    create_plot(axs[1], src_df.iloc[12:24], width_mmtv, threshold=1.1, interval=0.2,
                bar_width=bar_width, colors=colors, hatches=hatches, caption="(c) MMTV in GPT-J 30B.", batch_unit=28)
    mtv_1 = pd.concat([src_df.iloc[25:26], src_df.iloc[24:25], src_df.iloc[26:28]])
    create_plot(axs[2], mtv_1, width_mtv, threshold=1.1, interval=0.2,
                bar_width=bar_width, colors=colors, hatches=hatches, caption="(b) MTV in GPT-J 6B.")
    mtv_2 = pd.concat([src_df.iloc[29:30], src_df.iloc[28:29], src_df.iloc[30:32]])
    create_plot(axs[3], mtv_2, width_mtv, threshold=1.1, interval=0.2,
                bar_width=bar_width, colors=colors, hatches=hatches, caption="(d) MTV in GPT-J 30B.")

    handles = [
        Rectangle((0, 0.1), 1, 1, facecolor=colors["PrIM"], label='PrIM'),
        Rectangle((0, 0.1), 1, 1, facecolor=colors["PrIMC"], label='PrIM (E)'),
        Rectangle((0, 0.1), 1, 1, facecolor=colors["PS"], label='PrIM+search'),
        Rectangle((0, 0.1), 1, 1, facecolor=colors["ATiM"], label='ATiM'),

        Rectangle((0, 0.1), 1, 1, facecolor='#444444', label='H2D', edgecolor="white", hatch=hatches["H2D"]),
        Rectangle((0, 0.1), 1, 1, facecolor='#444444', label='Kernel', edgecolor="white"),
        Rectangle((0, 0.1), 1, 1, facecolor='#444444', label='After Kernel', edgecolor="white", hatch=hatches["After"]),
        Line2D([0], [0], color=colors["CPU"], marker='o', linestyle='-', label='ATiM\'s Speedup over CPU-autotuned', linewidth=1.4, markersize=4),
    ]

    fig.legend(handles=handles, loc='upper center', ncol=8, fontsize=fontsize - 1, bbox_to_anchor=(0.5, 1.09),
               borderpad=0.6, handlelength=1.5, handletextpad=0.6, columnspacing=1.6)

    fig.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")