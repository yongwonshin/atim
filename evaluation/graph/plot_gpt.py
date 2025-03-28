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

    def create_plot(ax, df_subset, threshold, interval=0.5, bar_width=0.25, colors=colors, hatches=hatches, caption="", batch_unit=16):
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
            common = dict(linewidth=0, zorder=2, color=color, edgecolor="white")
            ax.bar(pos, h2d_val, bar_width, label=f'{label_prefix}-H2D', hatch=hatch['H2D'], **common)
            ax.bar(pos, kernel_val, bar_width, bottom=h2d_val, label=f'{label_prefix}-Kernel', hatch=hatch['Kernel'], **common)
            adjusted_color = adjust_color(color)
            ax.bar(pos, after_val, bar_width, bottom=h2d_val + kernel_val, linewidth=0, zorder=2, label=f'{label_prefix}-After',
                color=adjusted_color, edgecolor="white", hatch=hatch['After'])
            ax.bar(pos, h2d_val + kernel_val + after_val, bar_width, linewidth=0.5, zorder=2, edgecolor='black', fill=False)
            total_height = (h2d_val + kernel_val + after_val).tolist()
            for i in range(len(x)):
                if total_height[i] > threshold:
                    ax.text(x[i], threshold, f'{total_height[i]:.2f}', ha='center', va='bottom', fontsize=fontsize)

        is_mmtv = "mmtv" in df_subset["Workload"].values[0]
        prim_label = "PrIMC" if is_mmtv else "PrIM"

        plot_bars(ax, x - bar_width, df_subset, "PrIM", colors[prim_label], hatches)
        plot_bars(ax, x, df_subset, 'PS', colors['PS'], hatches)
        plot_bars(ax, x + bar_width, df_subset, 'ATiM', colors['ATiM'], hatches)

        if not is_mmtv:
            ax.set_ylabel(latency_label, fontsize=fontsize - 0.2)

        ypos = lambda x: -ylabel_offset - ylabel_interval * x
        cconf = dict(ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=fontsize - 0.5)
        hconf = dict(ha="right", va="top", transform=ax.get_xaxis_transform(), fontsize=fontsize - 0.5)

        if is_mmtv:
            M_labels = df_subset['M'].values
            ax.text(-0.54, ypos(0), 'Token', **hconf)
            ax.text(-0.54, ypos(1), 'Batch', **hconf)
            for i, label in enumerate(M_labels):
                ax.text(x[i], -ylabel_offset, label, **cconf)
            for i, label in enumerate(["1 (16×T×256)", "4 (64×T×256)", "1 (28×T×256)", "4 (112×T×256)"]):
                ax.text((x[i * 4 + 1] + x[i * 4 + 2]) / 2, ypos(1), label, **cconf)
            for i, label in enumerate(["GPT-J 6B","GPT-J 30B"]):
                ax.text((x[i * 8 + 3] + x[i * 8 + 4]) / 2, y_params, label, **cconf)
            for i in range(len(x) - 1):
                ymin = (-height_table / height) * (2 if x[i] % 4 == 3 else 1)
                ymax = 1.2 if x[i] % 8 == 7 else 0.01
                yy = ax.axvline((x[i] + x[i+1]) / 2, color='black', ymin= ymin, ymax=ymax, lw=0.5, linestyle="--")
                yy.set_clip_on(False)
        else:
            ax.text(-0.54, ypos(0), 'Row', **hconf)
            ax.text(-0.54, ypos(1), 'Col', **hconf)
            M_labels = df_subset['N'].values
            K_labels = df_subset['K'].values
            for i, label in enumerate(M_labels):
                ax.text(x[i], ypos(0), label, **cconf)
            for i, label in enumerate(K_labels):
                ax.text(x[i], ypos(1), label, **cconf)
            for i, label in enumerate(["GPT-J 6B", "GPT-J 30B"]):
                ax.text((x[i * 4 + 1] + x[i * 4 + 2]) / 2, y_params, label, **cconf)
            for i in range(len(x) - 1):
                ymin = (-height_table / height) * (2 if x[i] % 4 == 3 else 2)
                ymax = 1.2 if x[i] % 4 == 3 else 0.01
                yy = ax.axvline((x[i] + x[i+1]) / 2, color='black', ymin= ymin, ymax=ymax, lw=0.5, linestyle="--")
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

    fig = plt.figure(figsize=(7.48, (height + 0.5) / 2.54), dpi=300, constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[width_mtv, width_mmtv], wspace=0.02, hspace=0.03, figure=fig)
    axs = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 0])]

    mtv_combined_data = src_df.iloc[16:24]
    create_plot(axs[1], mtv_combined_data, threshold=1.1, interval=0.2,
                bar_width=bar_width, colors=colors, hatches=hatches, caption="(a) MTV in GPT-J")
    mmtv_combined_data = src_df.iloc[:16]
    create_plot(axs[0], mmtv_combined_data, threshold=1.1, interval=0.2,
                bar_width=bar_width, colors=colors, hatches=hatches, caption="(b) MMTV in GPT-J")
    fig.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")