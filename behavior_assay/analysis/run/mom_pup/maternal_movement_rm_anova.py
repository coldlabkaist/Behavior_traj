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
import pandas as pd
from analysis.plots.mom_pup.focused_results import plot_locomotion

matplotlib.use("Agg")

METRIC = "avg_velocity_mm_s"

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot maternal movement using Holm-adjusted simple effects "
            "from the mixed repeated-measures model."
        )
    )
    from analysis.paths import FINAL, REPRODUCED
    parser.add_argument("--metrics", default=str(FINAL / "Fig5H/data/cage_velocity.csv"))
    parser.add_argument("--tests", default=str(FINAL / "Fig5H/stat/holm_group_comparisons.csv"))
    parser.add_argument("--output-dir", default=str(REPRODUCED / "Fig5H/figure"))
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
    plot_tests = simple_effects.loc[
        :, ["pnd", "p_holm_across_3_pnd"]
    ].rename(
        columns={"p_holm_across_3_pnd": "p_holm_within_metric"}
    )
    plot_tests.insert(1, "metric", METRIC)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_locomotion(metrics, plot_tests, output_dir)
    for name in ("maternal_locomotion_by_pnd.png", "maternal_locomotion_by_pnd.svg"):
        print(output_dir / name)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
