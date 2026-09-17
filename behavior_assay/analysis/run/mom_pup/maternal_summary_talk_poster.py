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
from analysis.plots.mom_pup.maternal_summary_talk_poster import create_individual_figures
from analysis.plots.mom_pup.maternal_summary_talk_poster import create_summary_figure

matplotlib.use("Agg")

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Export large-type maternal summary figures for talk and poster use.")
	parser.add_argument(
		"--proximity-metrics",
		default=str(ROOT / "output" / "mom_pup" / "body_scale_proximity" / "session_metrics.csv"),
	)
	parser.add_argument(
		"--proximity-tests",
		default=str(ROOT / "output" / "mom_pup" / "body_scale_proximity" / "between_group_tests.csv"),
	)
	parser.add_argument(
		"--movement-metrics",
		default=str(ROOT / "output" / "mom_pup" / "behavior_metrics" / "session_metrics.csv"),
	)
	parser.add_argument(
		"--movement-tests",
		default=str(ROOT / "output" / "mom_pup" / "development" / "stats" / "between_group_tests.csv"),
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
	proximity_metrics = pd.read_csv(args.proximity_metrics)
	proximity_tests = pd.read_csv(args.proximity_tests)
	movement_metrics = pd.read_csv(args.movement_metrics)
	movement_tests = pd.read_csv(args.movement_tests)
	output_dir = Path(args.output_dir)

	all_paths: list[Path] = []
	for profile in ("talk", "poster"):
		all_paths.extend(
			create_summary_figure(
				proximity_metrics,
				proximity_tests,
				movement_metrics,
				movement_tests,
				output_dir,
				profile,
			)
		)
		all_paths.extend(
			create_individual_figures(
				proximity_metrics,
				proximity_tests,
				movement_metrics,
				movement_tests,
				output_dir,
				profile,
			)
		)
	for path in all_paths:
		print(path)
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
