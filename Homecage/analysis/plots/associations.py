"""Maternal behavior and offspring BOI association figures."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.core.rank_tests import common_slope
from analysis.plots.export import save_figure
from analysis.plots.style import apply_style, CONTROL_COLOR, VPA_COLOR


# Maternal association


INK = "#202934"


ZERO = "#DCE2E8"


PANELS = (
    {
        "input": "pnd10_sustained_proximity_matched_litters.csv",
        "predictor": "filtered_total_pct_1s",
        "title": "PND10 maternal proximity and W3 BOI",
        "xlabel": "PND10 proximity time",
        "xlim": (0.0, 100.0),
        "xticks": [0, 20, 40, 60, 80, 100],
        "stats_anchor": "right",
        "output": "01_pnd10_sustained_proximity_w3_boi_spearman.png",
        "stats": "01_pnd10_sustained_proximity_w3_boi_spearman_stats.csv",
    },
    {
        "input": "04_pnd10_maternal_movement_w3_matched_litters.csv",
        "predictor": "avg_velocity_mm_s",
        "title": "PND10 maternal movement and W3 BOI",
        "xlabel": "PND10 maternal movement (mm/s)",
        "xlim": (0.0, 70.0),
        "xticks": [0, 10, 20, 30, 40, 50, 60, 70],
        "stats_anchor": "left",
        "output": "04_pnd10_maternal_movement_w3_boi_spearman.png",
        "stats": "04_pnd10_maternal_movement_w3_boi_spearman_stats.csv",
    },
    {
        "input": "05_pnd10_20_mean_maternal_movement_w3_matched_litters.csv",
        "predictor": "avg_velocity_mm_s",
        "title": "PND10–20 mean maternal movement and W3 BOI",
        "xlabel": "PND10–20 mean maternal movement (mm/s)",
        "xlim": (0.0, 70.0),
        "xticks": [0, 10, 20, 30, 40, 50, 60, 70],
        "stats_anchor": "left",
        "output": "05_pnd10_20_mean_maternal_movement_w3_boi_spearman.png",
        "stats": "05_pnd10_20_mean_maternal_movement_w3_boi_spearman_stats.csv",
    },
)


def p_label(value: float) -> str:
    if value < 0.001:
        return "p<.001"
    return f"p={value:.3f}".replace("0.", ".")


def significance_stars(value: float) -> str:
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def render(data: pd.DataFrame, statistics: pd.DataFrame, config: dict, output_dir: Path) -> Path:
    apply_style('maternal_association')
    figure, axis = plt.subplots(figsize=(7.9, 5.8))
    predictor = config["predictor"]
    condition = data["condition"].to_numpy()
    x_all = data[predictor].to_numpy(float)
    y_all = data["score_3w"].to_numpy(float)

    slope = common_slope(x_all, y_all, condition)
    grid = np.linspace(config["xlim"][0], config["xlim"][1], 200)
    axis.plot(
        grid,
        float(y_all.mean()) + slope * (grid - float(x_all.mean())),
        color=INK,
        linewidth=2.6,
        alpha=0.80,
        zorder=1,
    )

    for group, color, marker in (
        ("control", CONTROL_COLOR, "o"),
        ("vpa", VPA_COLOR, "s"),
    ):
        current = data.loc[data["condition"].eq(group)]
        x = current[predictor].to_numpy(float)
        y = current["score_3w"].to_numpy(float)
        group_slope, intercept = np.polyfit(x, y, 1)
        group_grid = np.linspace(float(x.min()), float(x.max()), 100)
        axis.plot(
            group_grid,
            intercept + group_slope * group_grid,
            color=color,
            linewidth=2.25,
            zorder=2,
        )
        axis.scatter(
            x,
            y,
            s=74,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.95,
            zorder=3,
        )

    axis.axhline(0, color=ZERO, linewidth=1.0, zorder=0)
    axis.set_xlim(*config["xlim"])
    axis.set_ylim(-0.70, 1.12)
    axis.set_xticks(config["xticks"])
    axis.set_yticks([-0.5, 0.0, 0.5, 1.0])
    axis.set_xlabel(config["xlabel"], fontsize=24, labelpad=16)
    axis.set_ylabel("W3 BOI", fontsize=24, labelpad=16)
    axis.set_title(config["title"], fontsize=23.25, fontweight="bold", pad=22)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.45)
    axis.tick_params(axis="both", which="major", labelsize=20.5, width=1.45, length=6)

    if config["stats_anchor"] == "right":
        x_position, alignment = 0.97, "right"
    else:
        x_position, alignment = 0.04, "left"
    label_map = {
        "all_litters_condition_adjusted": ("All", INK, 0.965),
        "control": ("Control", CONTROL_COLOR, 0.875),
        "vpa": ("VPA", VPA_COLOR, 0.785),
    }
    for row in statistics.itertuples(index=False):
        prefix, color, y_position = label_map[row.analysis]
        stars = significance_stars(float(row.exact_permutation_p))
        star_label = f"  {stars}" if stars else ""
        axis.text(
            x_position,
            y_position,
            f"{prefix}: ρ = {row.spearman_rho:.2f},  {p_label(row.exact_permutation_p)}{star_label}",
            transform=axis.transAxes,
            ha=alignment,
            va="top",
            color=color,
            fontsize=20.5 if prefix != "All" else 21.5,
            fontweight="bold",
            zorder=10,
        )

    figure.tight_layout(pad=1.15)
    path = output_dir / config["output"]
    save_figure(figure, path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


