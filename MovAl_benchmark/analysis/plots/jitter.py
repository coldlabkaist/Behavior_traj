"""Four adopted jitter conditions; statistics supplied separately from drawing."""
import matplotlib.pyplot as plt
import numpy as np
from .style import COLORS, significance_label
from ..core.jitter import METHOD_ORDER, BANDS


def plot_bands(summary, keypoint, comparisons):
    fig, ax = plt.subplots(figsize=(16, 9))
    width = 0.14
    for i, band in enumerate(BANDS):
        for j, method in enumerate(METHOD_ORDER):
            row = summary[(summary.keypoint == keypoint) & (summary.band == band) &
                          (summary.method == method)].iloc[0]
            xpos = i + (j - 1.5) * width
            mean, sem = row.mean_power / 1e6, row.se_power / 1e6
            ax.bar(xpos, mean, width=width, color=COLORS[method])
            ax.errorbar(xpos, mean, yerr=sem, fmt="none", ecolor="black", capsize=5, linewidth=1.5)
        # One label for the three comparisons against the final workflow is used
        # only when all three have the same significance level.
        selected = comparisons[(comparisons.keypoint == keypoint) & (comparisons.band == band) &
                               (comparisons.group2 == "Seg-Cont MovAl")]
        labels = [significance_label(p) for p in selected.P_holm]
        if len(labels) == 3 and len(set(labels)) == 1 and labels[0] != "n.s.":
            row = summary[(summary.keypoint == keypoint) & (summary.band == band) &
                          (summary.method == "Seg-Cont MovAl")].iloc[0]
            ax.text(i + 1.5 * width, (row.mean_power + row.se_power) / 1e6 + 0.4,
                    labels[0], ha="center", va="bottom", fontsize=26)
    ax.set_xticks(range(3), list(BANDS), fontsize=26)
    ax.set_xlabel("Band (Frequency)", fontsize=30)
    ax.set_ylabel(r"Power ($10^6$ pixel$^2$)", fontsize=30)
    ax.tick_params(axis="y", labelsize=24)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in METHOD_ORDER]
    ax.legend(handles, METHOD_ORDER, ncol=2, frameon=False, fontsize=19)
    return fig


def plot_traces(traces, keypoint):
    fig, axes = plt.subplots(2, 2, figsize=(16, 7))
    for i, condition in enumerate(["Raw", "Seg-Cont"]):
        for j, method in enumerate(["SLEAP", "MovAl"]):
            name = f"{condition} {method}"
            subset = traces[(traces.method == name) & (traces.keypoint == keypoint)]
            ax = axes[i, j]
            ax.plot(subset.frame_idx, subset.jitter_log, color=COLORS[name], linewidth=1)
            ax.set_ylim(-4, 4)
            ax.set_yticks([-4, 0, 4])
            ax.set_xticks([0, 2000, 4000, 6000, 8000])
            ax.tick_params(labelsize=15)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(method, fontsize=22)
            if i == 1:
                ax.set_xlabel("Frame Index", fontsize=22)
            if j == 0:
                ax.set_ylabel(f"{condition}\nJitter (log10)", fontsize=20)
    return fig
