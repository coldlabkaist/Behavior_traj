from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from analysis.plots.mom_pup.maternal_summary_talk_poster import COLORS, CONDITIONS, OFFSETS, PNDS, STYLES, _save_bundle, _stars, _style_axis
from analysis.plots.mom_pup.maternal_proximity_line import plot_movement_matched_export
from analysis.plots.mom_pup.maternal_proximity_line import plot_proximity_line

matplotlib.use("Agg")

from analysis.plots.mom_pup.maternal_proximity_line import METRIC

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot longitudinal maternal proximity with individual cage "
            "trajectories and Holm-adjusted within-PND group contrasts."
        )
    )
    from analysis.paths import FINAL, REPRODUCED
    parser.add_argument("--metrics", default=str(FINAL / "Fig5F/data/cage_proximity.csv"))
    parser.add_argument("--tests", default=str(FINAL / "Fig5F/stat/holm_group_comparisons.csv"))
    parser.add_argument("--output-dir", default=str(REPRODUCED / "Fig5F/figure"))
    parser.add_argument("--metric-column", default=METRIC)
    parser.add_argument("--title", default="Maternal proximity")
    parser.add_argument("--y-label", default="Proximity time (%)")
    parser.add_argument("--stem", default="maternal_proximity")
    parser.add_argument("--star-size", type=float, default=32)
    parser.add_argument("--ns-size", type=float, default=22)
    parser.add_argument(
        "--single-export-only",
        action="store_true",
        help="Only create the movement-matched PNG/SVG bundle (default behavior).",
    )
    parser.add_argument("--include-talk-profiles", action="store_true", help="Also export the optional talk/poster layouts.")
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
    tests = pd.read_csv(args.tests)
    output_dir = Path(args.output_dir)

    if args.metric_column not in metrics.columns:
        raise ValueError(f"Missing metric column: {args.metric_column}")

    for path in plot_movement_matched_export(
        metrics,
        tests,
        output_dir,
        metric=args.metric_column,
        title=args.title,
        y_label=args.y_label,
        stem=args.stem,
        star_size=args.star_size,
        ns_size=args.ns_size,
    ):
        print(path)

    if args.single_export_only or not args.include_talk_profiles:
        return 0

    for profile in ("talk", "poster"):
        size = (12.8, 7.4) if profile == "talk" else (11.8, 7.2)
        fig, ax = plt.subplots(figsize=size)
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.85)
        plot_proximity_line(ax, metrics, tests, profile)
        suffix = "_poster" if profile == "poster" else ""
        paths = _save_bundle(
            fig,
            output_dir,
            f"talk_maternal_proximity_line{suffix}",
        )
        plt.close(fig)
        for path in paths:
            print(path)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
