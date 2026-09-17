from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import pandas as pd
from analysis.core.three_chamber.talk_figure import _path
from analysis.plots.three_chamber.talk_figure import plot_talk_figure

from analysis.core.three_chamber.talk_figure import PHASE_CONFIG

def main() -> int:
	parser = argparse.ArgumentParser(description="Create compact two-panel three-chamber talk figures.")
	parser.add_argument(
		"--sample-summary",
		default="output/3chamber/barplots/preference_sample_summary__Nose__contact20__n1__first5min__bout0.5s__gap0.5s__reviewed.csv",
	)
	parser.add_argument("--phase", choices=sorted(PHASE_CONFIG), default="nov")
	parser.add_argument("--output-dir", required=True)
	args = parser.parse_args()

	sample_path = _path(args.sample_summary)
	if not sample_path.exists():
		raise FileNotFoundError(f"Sample summary not found: {sample_path}")
	output_dir = _path(args.output_dir)
	output_path = output_dir / f"three_chamber_{args.phase}_talk.png"
	plot_talk_figure(pd.read_csv(sample_path), phase=args.phase, output_path=output_path)
	print(f"Wrote {output_path}")
	print(f"Wrote {output_path.with_suffix('.pdf')}")
	print(f"Wrote {output_path.with_suffix('.svg')}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
