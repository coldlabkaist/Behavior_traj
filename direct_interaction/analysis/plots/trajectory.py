"""Individual trajectories and mean/SEM panels for Fig5D and FigS9AB."""
from __future__ import annotations

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from core.statistics import METRICS, sem
from plots.style import CONDITIONS, COLORS, DISPLAY_LABELS, TITLE_LABELS, apply_style, save_figure

def plot_category(
    df: pd.DataFrame,
    category: str,
    stats: pd.DataFrame,
    output_dir: Path,
    sex_group: str,
    dpi: int,
    width: float,
    height: float,
    mean_focus: bool,
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(width, height), facecolor="white")
    fig.subplots_adjust(left=0.105, right=0.93, bottom=0.21, top=0.73, wspace=0.28)

    weeks = sorted(df["week"].dropna().astype(int).unique())
    for ax, (metric, label) in zip(axes, METRICS):
        draw_metric_panel(ax, df, category, metric, label, weeks, stats, mean_focus)

    fig.text(
        0.5,
        0.955,
        TITLE_LABELS.get(category, f"{category} behavior"),
        ha="center",
        va="top",
        fontsize=24,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[condition],
            marker="o",
            markersize=8,
            linewidth=3.3,
            label=DISPLAY_LABELS[condition],
        )
        for condition in CONDITIONS
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        fontsize=16,
        handlelength=1.8,
        ncol=1,
        labelspacing=0.35,
    )

    stem = f"direct_interaction_{category.lower()}_{sex_group}_individual_trajectory_visible"
    save_figure(fig, output_dir, stem, dpi)


def draw_metric_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    category: str,
    metric: str,
    label: str,
    weeks: list[int],
    stats: pd.DataFrame,
    mean_focus: bool,
) -> None:
    col = f"{category}_{metric}"
    offset = {"cont": -0.07, "exp": 0.07}
    rng = np.random.default_rng(20260727)
    all_values = pd.to_numeric(df[col], errors="coerce").dropna()
    y_upper = compute_y_upper(df, category, metric, mean_focus)

    for condition in CONDITIONS:
        condition_df = df[df["condition"] == condition].copy()
        for _mouse_id, mouse_df in condition_df.groupby("mouse_id", sort=False):
            mouse_df = mouse_df.sort_values("week")
            x = mouse_df["week"].astype(float).to_numpy() + offset[condition]
            y = pd.to_numeric(mouse_df[col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(y)
            if valid.sum() >= 2:
                ax.plot(
                    x[valid],
                    y[valid],
                    color=COLORS[condition],
                    alpha=0.09 if mean_focus else 0.16,
                    linewidth=0.85 if mean_focus else 1.0,
                    zorder=1,
                )
        for week in weeks:
            values = pd.to_numeric(
                condition_df.loc[condition_df["week"].astype(int) == int(week), col],
                errors="coerce",
            ).dropna()
            x_jitter = week + offset[condition] + rng.uniform(-0.018, 0.018, len(values))
            ax.scatter(
                x_jitter,
                values,
                s=8 if mean_focus else 12,
                color=COLORS[condition],
                alpha=0.26 if mean_focus else 0.48,
                edgecolor="none",
                zorder=2,
            )

        means = []
        sems = []
        xs = []
        for week in weeks:
            values = pd.to_numeric(
                condition_df.loc[condition_df["week"].astype(int) == int(week), col],
                errors="coerce",
            ).dropna()
            xs.append(week + offset[condition])
            means.append(float(values.mean()) if len(values) else math.nan)
            sems.append(sem(values))
        xs_arr = np.asarray(xs, dtype=float)
        means_arr = np.asarray(means, dtype=float)
        sems_arr = np.asarray(sems, dtype=float)
        valid_summary = np.isfinite(means_arr) & np.isfinite(sems_arr)
        if mean_focus and valid_summary.sum() >= 2:
            ax.fill_between(
                xs_arr[valid_summary],
                means_arr[valid_summary] - sems_arr[valid_summary],
                means_arr[valid_summary] + sems_arr[valid_summary],
                color=COLORS[condition],
                alpha=0.18,
                linewidth=0,
                zorder=3,
            )
        ax.errorbar(
            xs_arr,
            means_arr,
            yerr=sems,
            color=COLORS[condition],
            marker="o",
            markersize=9.2 if mean_focus else 7.5,
            markerfacecolor=COLORS[condition] if mean_focus else "white",
            markeredgecolor="white" if mean_focus else COLORS[condition],
            markeredgewidth=1.8 if mean_focus else 2.0,
            linewidth=4.2 if mean_focus else 3.2,
            capsize=5,
            elinewidth=2.8 if mean_focus else 2.2,
            capthick=2.8 if mean_focus else 2.2,
            zorder=5,
        )

    add_significance(ax, df, category, metric, weeks, stats, y_upper)
    ax.set_xlim(min(weeks) - 0.38, max(weeks) + 0.38)
    ax.set_ylim(0, y_upper)
    ax.set_xticks(weeks)
    ax.set_xticklabels([f"{week}W" for week in weeks], fontsize=17)
    ax.set_ylabel(label, fontsize=18)
    ax.set_xlabel("Week", fontsize=18)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.7)
    ax.spines["bottom"].set_linewidth(1.7)


def compute_y_upper(
    df: pd.DataFrame,
    category: str,
    metric: str,
    mean_focus: bool,
) -> float:
    col = f"{category}_{metric}"
    all_values = pd.to_numeric(df[col], errors="coerce").dropna()
    if all_values.empty:
        return 1.0
    step = 20 if metric == "total_seconds" else 10
    if not mean_focus:
        return nice_upper(float(all_values.max()) * 1.26, step)

    group_max = 0.0
    for condition in CONDITIONS:
        for week in sorted(df["week"].dropna().astype(int).unique()):
            values = pd.to_numeric(
                df.loc[
                    (df["condition"] == condition)
                    & (df["week"].astype(int) == int(week)),
                    col,
                ],
                errors="coerce",
            ).dropna()
            if len(values):
                group_max = max(group_max, float(values.mean()) + sem(values))
    # Keep all observations visible unless one extreme outlier alone drives the axis.
    central_max = float(np.nanpercentile(all_values.to_numpy(dtype=float), 97.5))
    visible_target = max(group_max * 1.28, central_max * 1.08)
    visible_target = min(visible_target, float(all_values.max()) * 1.12)
    return nice_upper(visible_target, step)


def add_significance(
    ax: plt.Axes,
    df: pd.DataFrame,
    category: str,
    metric: str,
    weeks: list[int],
    stats: pd.DataFrame,
    y_upper: float,
) -> None:
    col = f"{category}_{metric}"
    metric_stats = stats[stats["metric"] == metric]
    for week in weeks:
        row = metric_stats[metric_stats["week"].astype(int) == int(week)]
        if row.empty:
            continue
        marker = str(row.iloc[0].get("stars", ""))
        if not marker or marker == "nan":
            continue
        values = pd.to_numeric(
            df.loc[df["week"].astype(int) == int(week), col],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        y = min(float(values.max()) + y_upper * 0.08, y_upper * 0.88)
        x1 = week - 0.17
        x2 = week + 0.17
        ax.plot([x1, x2], [y, y], color="black", linewidth=1.8, clip_on=False, zorder=5)
        ax.text(
            week,
            y + y_upper * 0.025,
            marker,
            ha="center",
            va="bottom",
            fontsize=21,
            clip_on=False,
            zorder=6,
        )


def nice_upper(value: float, step: int) -> float:
    if not np.isfinite(value) or value <= 0:
        return float(step)
    return float(math.ceil(value / step) * step)

