from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PNDS = (10, 15, 20)

COLORS = {"Control": "#304F78", "VPA": "#C4475B"}

OFFSETS = {"Control": -0.16, "VPA": 0.16}

METRIC = "total_stable_proximity_time_sec"

def _stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"

def _save(fig: plt.Figure, output_dir: Path) -> None:
    stem = "stable_proximity_by_pnd"
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        output_dir / f"{stem}_1200dpi.png",
        dpi=1200,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / f"{stem}_600dpi.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

def plot_stable_proximity(
    metrics: pd.DataFrame,
    simple_effects: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    rng = np.random.default_rng(20260831)

    for condition in ("Control", "VPA"):
        group = metrics.loc[metrics["condition"] == condition].copy()
        offset = OFFSETS[condition]

        for _, subject in group.groupby("subject_id"):
            subject = subject.sort_values("pnd")
            ax.plot(
                subject["pnd"].to_numpy(dtype=float) + offset,
                pd.to_numeric(subject[METRIC], errors="coerce"),
                color=COLORS[condition],
                alpha=0.16,
                linewidth=0.8,
                zorder=1,
            )

        means: list[float] = []
        sems: list[float] = []
        for pnd in PNDS:
            values = pd.to_numeric(
                group.loc[group["pnd"] == pnd, METRIC],
                errors="coerce",
            ).dropna()
            x = pnd + offset + rng.uniform(-0.045, 0.045, len(values))
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

    y_max = float(pd.to_numeric(metrics[METRIC], errors="coerce").max())
    ax.set_ylim(0, y_max * 1.18)
    stat_rows = simple_effects.set_index("pnd")
    for pnd in PNDS:
        if pnd not in stat_rows.index:
            continue
        p_value = float(stat_rows.loc[pnd, "p_holm_across_3_pnd"])
        ax.text(
            pnd,
            y_max * 1.08,
            _stars(p_value),
            ha="center",
            va="bottom",
            fontsize=20,
        )

    ax.set_xticks(PNDS)
    ax.set_xlabel("Postnatal day", fontsize=20, labelpad=10)
    ax.set_ylabel("Stable proximity time (s)", fontsize=20, labelpad=12)
    ax.set_title("Stable maternal proximity", fontsize=26, pad=16)
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
    _save(fig, output_dir)
