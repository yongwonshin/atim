import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


def plot_branch_opt(src_df, filename):
    height = 2.2
    bar_width = 0.15
    fontsize = 7
    latency_label = "Normalized to PrIM"

    sens_cols = {
        "O0": "#dff2eb",
        "O1": "#b9e5e8",
        "O2": "#7ab2d3",
        "O4": "#4a628a",
    }
    sens_labels = {
        "O0": "No OPT",
        "O1": "DMA",
        "O2": "DMA + LT",
        "O4": "DMA + LT + BH",
    }

    def create_plot(ax, df_subset, threshold, interval=0.5, bar_width=0.15, caption=""):
        x = np.arange(len(df_subset['M']), dtype=int)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(0, threshold)
        ax.yaxis.grid(True, linestyle='--', zorder=0, linewidth=0.5)
        y_ticks = np.arange(0, threshold, interval)
        ax.tick_params(axis='both', which='both', direction='in', length=0)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks], fontsize=fontsize-0.5)
        ax.set_xticklabels([])
        ax.set_xticks(x)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=0.5)
        ax.spines['left'].set_capstyle("butt")
        ax.spines['left'].set_clip_on(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        bars = [ax.bar(
                x + bar_width * (-3 / 2 + i),
                df_subset["PrIM"] / df_subset[level],
                bar_width,
                linewidth=0.5,
                zorder=2,
                color=sens_cols[level],
                edgecolor="black",
                label=sens_labels[level])
                for i, level in enumerate(["O0", "O1", "O2", "O4"])]

        for bar_group in bars:
            for bar in bar_group:
                height = bar.get_height()
                if height > threshold:
                    ax.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, threshold),
                                xytext=(0, 1),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=6)

        workloads = df_subset['M'].astype("int")
        for i, label in enumerate(workloads):
            ax.text(x[i], -0.04, label, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=7)
        if caption:
            ax.text(0.5, -0.3, caption, ha='center', va='top',
                    transform=ax.transAxes, fontsize=8, fontname='Times New Roman')

    fig = plt.figure(figsize=(19 / 2.54,  height / 2.54), dpi=300, constrained_layout=True)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.0, figure=fig)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]
    create_plot(axs[0], src_df.iloc[0:8], threshold=1.25, interval=0.2,
                bar_width=bar_width, caption="(a) MTV([256, L]×[L])")
    axs[0].set_ylabel(latency_label, fontsize=fontsize - 0.5)
    create_plot(axs[1], src_df.iloc[8:16], threshold=1.25, interval=0.2,
                bar_width=bar_width, caption="(b) MTV([L, 256]×[256])")
    axs[1].tick_params(axis='y', labelleft=False)
    create_plot(axs[2], src_df.iloc[16:24], threshold=1.25, interval=0.2,
                bar_width=bar_width, caption="(c) MTV([L, L]×[L])")
    axs[2].tick_params(axis='y', labelleft=False)
    create_plot(axs[3], src_df.iloc[24:32], threshold=1.25, interval=0.2,
                bar_width=bar_width, caption="(d) VA([L×100000]) with 32 DPUs")
    axs[3].tick_params(axis='y', labelleft=False)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', ncol=4, fontsize=fontsize - 1, bbox_to_anchor=(0.022,  1.19), columnspacing=0.8, handlelength=0.8, handleheight=0.8)
    fig.text(0.33, 1.085, "DMA: DMA-aware boundary check elimination (§5.3.1)   LT: Loop-bound tightening (§5.3.2)   BH: Invariant branch hoisting (§5.3.3)", ha='left', va='top', fontsize=fontsize - 1, transform=fig.transFigure)
    fig.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")