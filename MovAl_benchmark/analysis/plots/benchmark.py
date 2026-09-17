import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..core.identity import IDENTITY_COLUMNS
from .style import significance_label

def add_bracket(ax, x1, x2, y, label):
    ax.plot(
        [x1, x2],
        [y, y],
        color="black",
        linewidth=1.5,
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y + 0.03,
        label,
        ha="center",
        va="bottom",
        fontsize=60,
        family="Arial",
    )


def plot_identity(data, pairwise):
    method_columns = IDENTITY_COLUMNS
    method_values = {c: data[c].to_numpy(float) for c in method_columns}
    comparison_lookup = {(row["test_column"], row["reference_column"]): row
                         for row in pairwise.to_dict("records")}
    colors = {
        "raw_sleap": "#a8d5a0",
        "raw_dlc": "#e2d98b",
        "raw_yolo": "#e9a0a3",
        "seg_cont_sleap": "#1f6b37",
        "seg_cont_dlc": "#e0bd18",
        "seg_cont_yolo": "#d9342f",
    }
    positions = {
        "raw_sleap": -0.32,
        "raw_dlc": 0.00,
        "raw_yolo": 0.32,
        "seg_cont_sleap": 0.88,
        "seg_cont_dlc": 1.20,
        "seg_cont_yolo": 1.52,
    }

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(18, 11))

    for column in method_columns:
        values = method_values[column]
        x_position = positions[column]
        boxplot = ax.boxplot(
            values,
            positions=[x_position],
            widths=0.29,
            patch_artist=True,
            showfliers=False,
        )
        for box in boxplot["boxes"]:
            box.set_facecolor(colors[column])
            box.set_edgecolor("black")
            box.set_linewidth(2.6)
        for component in ("whiskers", "caps", "medians"):
            for artist in boxplot[component]:
                artist.set_color("black")
                artist.set_linewidth(2.6)

        ax.scatter(
            x_position,
            np.mean(values),
            color="black",
            marker="x",
            s=85,
            linewidths=2.0,
            zorder=4,
        )
        jitter = rng.uniform(-0.038, 0.038, size=len(values))
        ax.scatter(
            x_position + jitter,
            values,
            color="#252525",
            s=78,
            marker="D",
            alpha=0.9,
            zorder=3,
        )

    lower_bracket_y = 2.00
    upper_bracket_y = 2.20
    add_bracket(
        ax,
        positions["raw_sleap"],
        positions["raw_yolo"],
        lower_bracket_y,
        significance_label(comparison_lookup[("raw_sleap", "raw_yolo")]["q"]),
    )
    add_bracket(
        ax,
        positions["raw_dlc"],
        positions["raw_yolo"],
        upper_bracket_y,
        significance_label(comparison_lookup[("raw_dlc", "raw_yolo")]["q"]),
    )
    add_bracket(
        ax,
        positions["seg_cont_sleap"],
        positions["seg_cont_yolo"],
        lower_bracket_y,
        significance_label(
            comparison_lookup[("seg_cont_sleap", "seg_cont_yolo")]["q"]
        ),
    )
    add_bracket(
        ax,
        positions["seg_cont_dlc"],
        positions["seg_cont_yolo"],
        upper_bracket_y,
        significance_label(
            comparison_lookup[("seg_cont_dlc", "seg_cont_yolo")]["q"]
        ),
    )

    ax.set_xticks([0.0, 1.2])
    ax.set_xticklabels(["Raw", "Seg Cont"], fontsize=44, family="Arial")
    ax.set_ylabel(
        "Identity Switch Frequency (%)", fontsize=46, family="Arial", labelpad=18
    )
    ax.set_ylim(0, 2.30)
    ax.tick_params(axis="y", labelsize=30, width=2.2, length=8)
    ax.tick_params(axis="x", width=2.2, length=8)
    for label in ax.get_yticklabels():
        label.set_family("Arial")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3.5)
    ax.spines["bottom"].set_linewidth(3.5)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#4caf50", edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#e2bb0c", edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#e53935", edgecolor="none"),
    ]
    ax.legend(
        legend_handles,
        ["SLEAP", "DLC", "MovAl"],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.78),
        frameon=False,
        fontsize=34,
        prop={"family": "DejaVu Sans", "size": 34},
        handlelength=0.8,
        handleheight=1.2,
    )

    return fig


def plot_missing(summary):
    from .style import COLORS
    from ..core.missing import KEYPOINT_ORDER, CONDITION_ORDER
    fig, ax = plt.subplots(figsize=(22, 9))
    centers = np.arange(5) * 1.2
    width, gap = 0.12, 0.001
    total_width = 6 * width + 5 * gap
    ymax = 0.0
    for k, kp in enumerate(KEYPOINT_ORDER):
        for j, condition in enumerate(CONDITION_ORDER):
            inp, method = condition.rsplit(" ", 1)
            row = summary[(summary.keypoint == kp) & (summary.method == method) & (summary.input == inp)].iloc[0]
            xpos = centers[k] - total_width / 2 + j * (width + gap) + width / 2 + [-0.05, 0, 0.05][j // 2]
            ax.bar(xpos, row["mean"], width=width, color=COLORS[condition], edgecolor="black", linewidth=1, zorder=2)
            ax.errorbar(xpos, row["mean"], yerr=row["sem"], fmt="none", ecolor="black", capsize=8, linewidth=2, zorder=3)
            ymax = max(ymax, row["mean"] + row["sem"])
    ax.set_xticks(centers, KEYPOINT_ORDER, fontsize=30)
    ax.set_xlabel("Keypoints", fontsize=44)
    ax.set_ylabel("Missing Data Frequency (%)", fontsize=44)
    ax.set_xlim(centers[0] - 1.2 * 0.7, centers[-1] + 1.2 * 0.7)
    ax.set_ylim(0, ymax * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(3)
    ax.tick_params(axis="both", labelsize=30, width=2)
    return fig


def plot_rmse(data):
    from ..core.rmse import KEYPOINTS, CONDITIONS
    wide = data.pivot(index="keypoint", columns=["method", "input"], values="rmse")
    wide = wide.reindex(index=KEYPOINTS, columns=pd.MultiIndex.from_tuples(CONDITIONS))
    if wide.isna().any().any():
        raise ValueError("Missing heatmap cells")
    values = wide.to_numpy()

    plt.rcParams.update({"font.family": "Arial", "font.weight": "regular",
                         "axes.unicode_minus": False, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=0.3, aspect="auto")
    labels = [f"{condition.replace('Seg-Cont', 'Seg Cont')}\n{method}"
              for method, condition in CONDITIONS]
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=22)
    ax.set_yticks(np.arange(len(KEYPOINTS)), KEYPOINTS, fontsize=22)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.4f}", ha="center", va="center",
                    fontsize=24, color="white" if values[i, j] > 0.04 else "black")
    ax.set_xticks(np.arange(-0.5, len(labels), 2), minor=True)
    ax.set_yticks(np.arange(-0.5, len(KEYPOINTS), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.12, orientation="horizontal")
    cbar.set_label("RMSE", fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    return fig
