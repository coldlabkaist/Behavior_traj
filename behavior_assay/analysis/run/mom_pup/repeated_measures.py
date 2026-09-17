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
from analysis.plots.mom_pup.maternal_summary_talk_poster import STYLES, _plot_proximity, _save_bundle

matplotlib.use("Agg")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export maternal-proximity talk/poster figures annotated with "
            "Tukey-adjusted mixed-model comparisons."
        )
    )
    parser.add_argument(
        "--metrics",
        default=str(
            ROOT
            / "output"
            / "mom_pup"
            / "body_scale_proximity"
            / "session_metrics.csv"
        ),
    )
    parser.add_argument(
        "--tests",
        default=str(
            ROOT
            / "output"
            / "mom_pup"
            / "body_scale_proximity"
            / "tukey_hsd_emm_same_pnd_control_vs_vpa.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "work" / "talkfigure"),
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
    tests = pd.read_csv(args.tests).rename(
        columns={"p_tukey_hsd": "p_welch_holm_across_pnd"}
    )
    output_dir = Path(args.output_dir)

    for profile in ("talk", "poster"):
        style = STYLES[profile]
        figure_size = (11.2, 7.5) if profile == "talk" else (10.5, 7.4)
        fig, ax = plt.subplots(figsize=figure_size)
        fig.subplots_adjust(left=0.15, right=0.97, bottom=0.16, top=0.86)
        _plot_proximity(ax, metrics, tests, style)
        suffix = "_poster" if profile == "poster" else ""
        paths = _save_bundle(
            fig,
            output_dir,
            f"talk_maternal_proximity_rm_anova_tukey{suffix}",
        )
        plt.close(fig)
        for path in paths:
            print(path)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
