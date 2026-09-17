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

METRIC = "total_stable_proximity_time_sec"

from analysis.plots.mom_pup.stable_proximity_rm_anova import _stars

from analysis.plots.mom_pup.stable_proximity_rm_anova import _save

from analysis.plots.mom_pup.stable_proximity_rm_anova import plot_stable_proximity

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot stable maternal proximity with repeated-measures cage "
            "trajectories and Holm-adjusted within-PND group contrasts."
        )
    )
    parser.add_argument(
        "--metrics",
        default=str(
            ROOT
            / "output"
            / "mom_pup"
            / "behavior_metrics"
            / "session_metrics.csv"
        ),
    )
    parser.add_argument(
        "--tests",
        default=str(
            ROOT
            / "output"
            / "mom_pup"
            / "stable_proximity"
            / "mixed_model_simple_effects_group_within_pnd.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "mom_pup"),
    )
    args = parser.parse_args(argv)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    metrics = pd.read_csv(args.metrics)
    simple_effects = pd.read_csv(args.tests)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_stable_proximity(metrics, simple_effects, output_dir)

    for name in (
        "stable_proximity_by_pnd.png",
        "stable_proximity_by_pnd_1200dpi.png",
        "stable_proximity_by_pnd_600dpi.tiff",
        "stable_proximity_by_pnd.svg",
    ):
        print(output_dir / name)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
