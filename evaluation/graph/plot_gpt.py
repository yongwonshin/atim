import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from plot_utils import colors, hatches, adjust_color


def plot_gpt(src_df, filename):
    width_mmtv = 10
    width_mtv = 5
    height = 2.1
    height_table = 0.5
    bar_width = 0.2
    fontsize = 7
    ylabel_offset = 0.05
    ylabel_interval = 0.24
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
        ax.set_yticklabels([f"{tick:.1f}" if tick != 0 else "" for tick in y_ticks], fontsize=fontsize)
        ax.set_xticklabels([])
        ax.set_xticks(x)


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

        plot_bar_cpu(ax, x - bar_width * 3 / 2, df_subset)
        plot_bars(ax, x - bar_width / 2, df_subset, "PrIM", colors[prim_label], hatches)
        plot_bars(ax, x + bar_width / 2, df_subset, 'PS', colors['PS'], hatches)
        plot_bars(ax, x + bar_width * 3 / 2, df_subset, 'ATiM', colors['ATiM'], hatches)

        if not is_mmtv:
            ax.set_ylabel(latency_label, fontsize=fontsize - 0.2)

        ypos = lambda x: -ylabel_offset - ylabel_interval * x
        cconf = dict(ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=fontsize - 0.5)
        hconf = dict(ha="right", va="top", transform=ax.get_xaxis_transform(), fontsize=fontsize - 0.5)

        if is_mmtv:
            B_labels = df_subset["M"].values
            M_labels = df_subset['N'].values
            ax.text(-0.5 - 0.3, -ylabel_offset, 'Batch', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=fontsize - 0.5)
            ax.text(-0.5 - 0.3, -ylabel_offset - ylabel_interval, 'Token', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=fontsize - 0.5)
            for i, label in enumerate(B_labels):
                if i % 4 == 1:
                    ax.text((x[i] + x[i+1]) / 2, -ylabel_offset, label, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=fontsize)
            for i, label in enumerate(M_labels):
                ax.text(x[i], -ylabel_offset - ylabel_interval, label, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=fontsize)
            for i in range(len(x) - 1):
                yy = ax.axvline((x[i] + x[i+1]) / 2, color='black', ymin=(-height_table / height) * 2, ymax=0.01 if x[i] % 4 == 3 else (-height_table / height), lw=0.5, linestyle="--")
                yy.set_clip_on(False)
        else:
            # ax.text(-0.54, ypos(0), 'Row', **hconf)
            # ax.text(-0.54, ypos(1), 'Col', **hconf)
            M_labels = df_subset['M'].values
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
        xmin = (-0.8 if is_mmtv else -0.8) / len(x)
        h0 = ax.axhline(-0, color='black',xmin=xmin, lw=1)
        hl = ax.axhline(-rpt, color='black', lw=0.6, xmin=xmin, linestyle="--")
        h2 = ax.axhline(-rpt * 2, color='black',xmin=xmin, lw=0.6)

        h0.set_clip_on(False)
        hl.set_clip_on(False)
        h2.set_clip_on(False)

        ax.spines['left'].set_bounds(-rpt * 2, threshold + 0.2)
        ax.spines['left'].set_capstyle("butt")
        ax.spines['left'].set_clip_on(False)
        ax.spines['right'].set_bounds(-rpt * 2, threshold + 0.2)
        ax.spines['right'].set_capstyle("butt")
        ax.spines['right'].set_clip_on(False)

        if caption:
            ax.text(0.5, -0.5, caption, ha='center', va='top',
                    transform=ax.transAxes, fontsize=8, fontname='Times New Roman')

    fig = plt.figure(figsize=(7.48, (height* 2 +1) / 2.54), dpi=300, constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[width_mmtv, width_mtv], wspace=0.03, hspace=0.03, figure=fig)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]

    create_plot(axs[0], src_df.iloc[:12], width_mmtv, threshold=1.1, interval=0.2,
                bar_width=bar_width, colors=colors, hatches=hatches, caption="(a) MMTV in GPT-J 6B", batch_unit=16)
    create_plot(axs[1], src_df.iloc[12:24], width_mmtv, threshold=1.1, interval=0.2,
                bar_width=bar_width, colors=colors, hatches=hatches, caption="(c) MMTV in GPT-J 30B", batch_unit=28)
    create_plot(axs[2], src_df.iloc[24:28], width_mtv, threshold=1.1, interval=0.2,
                bar_width=bar_width, colors=colors, hatches=hatches, caption="(b) MTV in GPT-J 6B")
    create_plot(axs[3], src_df.iloc[28:32], width_mtv, threshold=1.1, interval=0.2,
                bar_width=bar_width, colors=colors, hatches=hatches, caption="(d) MTV in GPT-J 30B")


    fig.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")