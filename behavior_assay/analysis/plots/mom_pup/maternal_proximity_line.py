from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.plots.mom_pup.maternal_summary_talk_poster import COLORS, CONDITIONS, OFFSETS, PNDS, STYLES, _save_bundle, _stars, _style_axis

matplotlib.use("Agg")

METRIC = "pct_time_sustained_body_scale_proximity"

def plot_proximity_line(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    tests: pd.DataFrame,
    profile: str,
) -> None:
    style = STYLES[profile]
    rng = np.random.default_rng(20260830)

    for condition in CONDITIONS:
        group = metrics[metrics["condition"] == condition].copy()
        offset = OFFSETS[condition]
        for _, subject in group.groupby("subject_id"):
            subject = subject.sort_values("pnd")
            ax.plot(
                subject["pnd"].to_numpy(dtype=float) + offset,
                pd.to_numeric(subject[METRIC], errors="coerce"),
                color=COLORS[condition],
                alpha=0.16,
                linewidth=style.individual_line_width,
                zorder=1,
            )

        means: list[float] = []
        sems: list[float] = []
        for pnd in PNDS:
            values = pd.to_numeric(
                group.loc[group["pnd"] == pnd, METRIC],
                errors="coerce",
            ).dropna()
            x = pnd + offset + rng.uniform(-0.045, 0.045, size=len(values))
            ax.scatter(
                x,
                values,
                s=style.point_size,
                facecolors="white",
                edgecolors=COLORS[condition],
                linewidths=1.55,
                alpha=0.95,
                zorder=2,
            )
            means.append(float(values.mean()))
            sems.append(float(values.sem()))
        ax.errorbar(
            np.asarray(PNDS, dtype=float) + offset,
            means,
            yerr=sems,
            color=COLORS[condition],
            marker="o",
            markersize=10,
            linewidth=style.main_line_width,
            elinewidth=2.2,
            capsize=5,
            capthick=2.2,
            zorder=3,
        )

    stat_rows = tests.set_index("pnd")
    for pnd in PNDS:
        p_value = float(stat_rows.loc[pnd, "p_holm_across_3_pnd"])
        ax.text(
            pnd,
            104.0,
            _stars(p_value),
            ha="center",
            va="bottom",
            fontsize=style.star_size,
        )

    ax.set_xlim(9.3, 20.7)
    ax.set_ylim(0, 112)
    ax.set_xticks(PNDS)
    ax.set_xlabel("Postnatal day", fontsize=style.label_size, labelpad=14)
    ax.set_ylabel("Time near pup (%)", fontsize=style.label_size, labelpad=14)
    ax.set_title("Maternal proximity", fontsize=style.title_size, pad=20)
    _style_axis(ax, style)

def plot_movement_matched_export(
    metrics: pd.DataFrame,
    tests: pd.DataFrame,
    output_dir: Path,
    *,
    metric: str = METRIC,
    title: str = "Maternal proximity",
    y_label: str = "Time near pup (%)",
    stem: str = "maternal_proximity_by_pnd",
    star_size: float = 20,
    ns_size: float = 22,
) -> list[Path]:
    """Export with the exact canvas and styling used by the 1200 dpi movement plot."""
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    rng = np.random.default_rng(20260830)
    offsets = {"Control": -0.16, "VPA": 0.16}

    for condition in CONDITIONS:
        group = metrics[metrics["condition"] == condition].copy()
        offset = offsets[condition]
        for _, subject in group.groupby("subject_id"):
            subject = subject.sort_values("pnd")
            ax.plot(
                subject["pnd"].to_numpy(dtype=float) + offset,
                pd.to_numeric(subject[metric], errors="coerce"),
                color=COLORS[condition],
                alpha=0.16,
                linewidth=0.8,
                zorder=1,
            )

        means: list[float] = []
        sems: list[float] = []
        for pnd in PNDS:
            values = pd.to_numeric(
                group.loc[group["pnd"] == pnd, metric],
                errors="coerce",
            ).dropna()
            x = pnd + offset + rng.uniform(-0.045, 0.045, size=len(values))
            ax.scatter(
                x,
                values,
                s=30,
                facecolors="white",
                edgecolors=COLORS[condition],
                linewidths=1.15,
                alpha=0.9,
                zorder=2,
            )
            means.append(float(values.mean()))
            sems.append(float(values.sem()))
        ax.errorbar(
            np.asarray(PNDS, dtype=float) + offset,
            means,
            yerr=sems,
            color=COLORS[condition],
            marker="o",
            markersize=7,
            linewidth=2.2,
            capsize=4,
            zorder=3,
        )

    y_max = float(pd.to_numeric(metrics[metric], errors="coerce").max())
    ax.set_ylim(0, y_max * 1.18)
    stat_rows = tests.set_index("pnd")
    for pnd in PNDS:
        p_value = float(stat_rows.loc[pnd, "p_holm_across_3_pnd"])
        significance_label = _stars(p_value)
        ax.text(
            pnd,
            y_max * 1.08,
            significance_label,
            ha="center",
            va="bottom",
            fontsize=ns_size if significance_label == "ns" else star_size,
        )

    ax.set_xticks(PNDS)
    ax.set_xlabel("Postnatal day", fontsize=20, labelpad=10)
    ax.set_ylabel(y_label, fontsize=20, labelpad=12)
    ax.set_title(title, fontsize=26, pad=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=17,
        width=1.6,
        length=6,
        pad=6,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths
