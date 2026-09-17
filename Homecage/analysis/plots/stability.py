"""Motif resolution and temporal-shift stability figures."""

from __future__ import annotations
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.core.temporal_shift import _mean_ci, MAGNITUDES, LABELS
from analysis.plots.export import save_figure
from analysis.plots.style import apply_style


# Motif resolution


def render_resolution(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    focal: pd.DataFrame,
    output: Path,
) -> None:
    apply_style('motif_resolution')
    ink = "#222A32"
    neutral = "#A4AAB0"
    orange = "#D97832"
    pink = "#E98EAF"
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(20.0, 8.0),
        constrained_layout=True,
        facecolor="white",
    )
    specifications = (
        (
            "novelty",
            "New behavioral information added at each K",
            "Behavioral profile difference from K − 1",
            (4, 5),
        ),
        (
            "stability",
            "Codebook stability",
            "Agreement across repeated fits",
            (5, 6),
        ),
    )
    for axis, (metric, title, ylabel, comparison) in zip(axes, specifications):
        current = metrics.pivot(index="cage", columns="k", values=metric).sort_index()
        current_summary = summary[summary["metric"] == metric].sort_values("k")
        ks = current_summary["k"].to_numpy(int)
        axis.axvspan(
            comparison[0],
            comparison[1],
            color=pink,
            alpha=0.18,
            linewidth=0,
            zorder=0,
        )
        for _, values in current.iterrows():
            axis.plot(
                ks,
                values.loc[ks].to_numpy(float),
                color=neutral,
                linewidth=1.55,
                alpha=0.55,
                zorder=1,
            )
        mean = current_summary["mean"].to_numpy(float)
        lower = mean - current_summary["ci95_low"].to_numpy(float)
        upper = current_summary["ci95_high"].to_numpy(float) - mean
        axis.plot(
            ks,
            mean,
            color=ink,
            linewidth=3.5,
            marker="o",
            markersize=10.8,
            markerfacecolor="white",
            markeredgecolor=ink,
            markeredgewidth=1.7,
            zorder=3,
        )
        axis.errorbar(
            ks,
            mean,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor=ink,
            elinewidth=2.15,
            capsize=5.8,
            capthick=2.15,
            zorder=2,
        )
        k4_mean = float(current_summary.loc[current_summary["k"] == 4, "mean"].iloc[0])
        axis.scatter(
            [4],
            [k4_mean],
            s=300,
            color=orange,
            edgecolor="white",
            linewidth=1.8,
            zorder=5,
        )
        minimum = float(current.to_numpy(float).min())
        maximum = float(current.to_numpy(float).max())
        span = maximum - minimum
        axis.set_ylim(minimum - 0.08 * span, maximum + 0.10 * span)
        axis.set_xticks(ks)
        axis.set_xlabel("Number of motifs (K)")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="center", pad=18.0, fontweight="semibold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
    stem = output / "k_cage_metric_lines"
    save_figure(figure, stem, svg_dpi=300, bbox_inches="tight")
    plt.close(figure)


# Temporal shift

INK = "#20262D"


HARD = "#747C85"


SOFT = "#4C78A8"


HARD_LIGHT = "#D7DBDF"


SOFT_LIGHT = "#C6D7E7"


def _style_temporal_shift() -> None:
    apply_style('temporal_shift')


def render_temporal_shift(data: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    _style_temporal_shift()
    figure, (axis_level, axis_drop) = plt.subplots(
        1,
        2,
        figsize=(18.0, 8.0),
        gridspec_kw={"width_ratios": (1.85, 1.0)},
    )
    x = np.arange(3, dtype=float)
    rng = np.random.default_rng(20260827)

    for metric, metric_label, color, light_color, marker, linestyle in (
        ("hard_agreement", "Hard label", HARD, HARD_LIGHT, "s", "--"),
        ("js_similarity", "Soft posterior", SOFT, SOFT_LIGHT, "o", "-"),
    ):
        means, lower, upper = [], [], []
        for magnitude in MAGNITUDES:
            row = summary[
                (summary["metric"] == metric)
                & np.isclose(summary["shift_magnitude_seconds"], magnitude)
            ].iloc[0]
            means.append(float(row["mean"]))
            lower.append(float(row["ci95_low"]))
            upper.append(float(row["ci95_high"]))
        means_array = 100.0 * np.asarray(means)
        lower_array = 100.0 * np.asarray(lower)
        upper_array = 100.0 * np.asarray(upper)
        axis_level.fill_between(
            x,
            lower_array,
            upper_array,
            color=light_color,
            alpha=0.35,
            linewidth=0,
        )
        axis_level.plot(
            x,
            means_array,
            color=color,
            linewidth=3.3,
            linestyle=linestyle,
            marker=marker,
            markersize=9,
            markerfacecolor="white",
            markeredgewidth=2.3,
            zorder=4,
        )
        for position, magnitude, mean_value, upper_ci in zip(
            x, MAGNITUDES, means_array, upper_array
        ):
            values = 100.0 * data.loc[
                np.isclose(data["shift_magnitude_seconds"], magnitude), metric
            ].to_numpy(dtype=float)
            jitter = rng.normal(0.0, 0.045, size=len(values))
            axis_level.scatter(
                position + jitter,
                values,
                s=34,
                color=color,
                alpha=0.28,
                edgecolor="none",
                zorder=2,
            )
            axis_level.text(
                position,
                min(99.2, float(upper_ci) + 0.8),
                f"{mean_value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=17,
                fontweight="bold",
                color=INK,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                    "pad": 0.25,
                },
            )
        axis_level.text(
            2.13,
            means_array[-1],
            metric_label,
            ha="left",
            va="center",
            fontsize=17,
            fontweight="bold",
            color=INK,
        )

    axis_level.set_xlim(-0.35, 2.68)
    axis_level.set_ylim(60, 100.5)
    axis_level.set_xticks(x, LABELS)
    axis_level.set_yticks(np.arange(60, 101, 10))
    axis_level.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis_level.set_ylabel("Agreement between independent analyses", labelpad=14)
    axis_level.set_title("Soft assignments retain more information", pad=18)
    axis_level.spines[["top", "right"]].set_visible(False)
    axis_level.tick_params(width=1.6, length=6)

    bar_width = 0.32
    bar_x = np.arange(2, dtype=float)
    for offset, metric, metric_label, color in (
        (-bar_width / 2, "hard_agreement", "Hard label", HARD),
        (bar_width / 2, "js_similarity", "Soft posterior", SOFT),
    ):
        heights, low_errors, high_errors = [], [], []
        for magnitude in MAGNITUDES[1:]:
            pivot = data.pivot(
                index="held_out_cage",
                columns="shift_magnitude_seconds",
                values=metric,
            ).sort_index()
            baseline_column = pivot.columns[
                int(np.argmin(np.abs(pivot.columns.to_numpy(float))))
            ]
            shifted_column = pivot.columns[
                int(np.argmin(np.abs(pivot.columns.to_numpy(float) - magnitude)))
            ]
            decrease = pivot[baseline_column].to_numpy(float) - pivot[shifted_column].to_numpy(float)
            mean, low, high = _mean_ci(decrease)
            heights.append(100.0 * mean)
            low_errors.append(100.0 * (mean - low))
            high_errors.append(100.0 * (high - mean))
        bars = axis_drop.bar(
            bar_x + offset,
            heights,
            width=bar_width * 0.88,
            color=color,
            alpha=0.72,
            edgecolor=INK,
            linewidth=1.4,
            yerr=np.vstack([low_errors, high_errors]),
            error_kw={"ecolor": INK, "elinewidth": 1.7, "capsize": 5, "capthick": 1.7},
        )
        for bar, value, upper_error in zip(bars, heights, high_errors):
            axis_drop.text(
                bar.get_x() + bar.get_width() / 2,
                value + upper_error + 0.65,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
                color=INK,
            )

    axis_drop.set_xlim(-0.55, 1.75)
    axis_drop.set_ylim(0, 20.0)
    axis_drop.set_xticks(bar_x, LABELS[1:])
    axis_drop.set_yticks(np.arange(0, 21, 4))
    axis_drop.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis_drop.set_ylabel("Decrease from no shift (%)", labelpad=12)
    axis_drop.set_title("Hard labels amplify the apparent decrease", pad=18)
    axis_drop.legend(
        handles=[
            Patch(facecolor=HARD, edgecolor=INK, alpha=0.72, label="Hard label"),
            Patch(facecolor=SOFT, edgecolor=INK, alpha=0.72, label="Soft posterior"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=False,
        fontsize=15,
        handlelength=1.3,
        handletextpad=0.55,
        borderaxespad=0.1,
    )
    axis_drop.spines[["top", "right"]].set_visible(False)
    axis_drop.tick_params(width=1.6, length=6)

    figure.suptitle(
        "Soft posterior similarity is more robust to temporal shifts",
        fontsize=29,
        fontweight="bold",
        y=0.99,
    )
    figure.subplots_adjust(left=0.075, right=0.975, bottom=0.15, top=0.82, wspace=0.32)
    base = output_dir / "hard_vs_soft_temporal_shift_comparison"
    save_figure(figure, base, bbox_inches="tight")
    plt.close(figure)


