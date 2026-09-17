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
from analysis.plots.mom_pup.maternal_proximity_bout_count import plot_bout_count

matplotlib.use("Agg")

from analysis.plots.mom_pup.maternal_proximity_bout_count import DEFAULT_STEM

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot counts of body-scale maternal proximity bouts lasting at "
            "least one second."
        )
    )
    parser.add_argument(
        "--metrics",
        default=str(
            ROOT
            / "output"
            / "mom_pup"
            / "body_scale_proximity_bouts_1s"
            / "session_metrics.csv"
        ),
    )
    parser.add_argument(
        "--tests",
        default=str(
            ROOT
            / "output"
            / "mom_pup"
            / "body_scale_proximity_bouts_1s"
            / "mixed_model_simple_effects_group_within_pnd.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "mom_pup"),
    )
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--stem", default=DEFAULT_STEM)
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
    plot_bout_count(
        pd.read_csv(args.metrics),
        pd.read_csv(args.tests),
        output_dir,
        args.min_duration_sec,
        args.stem,
    )
    for name in (
        f"{args.stem}.png",
        f"{args.stem}_1200dpi.png",
        f"{args.stem}_600dpi.tiff",
        f"{args.stem}.svg",
    ):
        print(output_dir / name)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
