from __future__ import annotations

import argparse

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

PNDS = (10, 15, 20)

COLORS = {"Control": "#304F78", "VPA": "#C4475B"}

OFFSETS = {"Control": -0.16, "VPA": 0.16}

METRIC = "pct_time_stable_proximity"

STEM = "stable_proximity_percentage_by_pnd"

from analysis.plots.mom_pup.stable_proximity_pct_rm_anova import _stars

from analysis.plots.mom_pup.stable_proximity_pct_rm_anova import _save

from analysis.plots.mom_pup.stable_proximity_pct_rm_anova import plot_percentage

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot stable proximity as a percentage of valid observation time "
            "with repeated cage trajectories and Holm-adjusted contrasts."
        )
    )
    parser.add_argument(
        "--metrics",
        default=str(
            ROOT
            / "output"
            / "mom_pup"
            / "stable_proximity_percentage"
            / "session_metrics_percentage.csv"
        ),
    )
    parser.add_argument(
        "--tests",
        default=str(
            ROOT
            / "output"
            / "mom_pup"
            / "stable_proximity_percentage"
            / "mixed_model_simple_effects_group_within_pnd.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "mom_pup"),
    )
    parser.add_argument("--min-duration-sec", type=float, default=3.0)
    args = parser.parse_args(argv)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_percentage(
        pd.read_csv(args.metrics),
        pd.read_csv(args.tests),
        output_dir,
        float(args.min_duration_sec),
    )

    for name in (
        f"{STEM}.png",
        f"{STEM}_1200dpi.png",
        f"{STEM}_600dpi.tiff",
        f"{STEM}.svg",
    ):
        print(output_dir / name)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
