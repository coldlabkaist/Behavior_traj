from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import matplotlib
import pandas as pd
from analysis.core.mom_pup.common import DEFAULT_OUTPUT_DIR, to_abs_path
from analysis.plots.mom_pup.focused_results import plot_locomotion
from analysis.plots.mom_pup.focused_results import plot_pnd10_stable_proximity

matplotlib.use("Agg")

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Create the two focused mother-pup result figures.")
	parser.add_argument(
		"--metrics",
		default=str(DEFAULT_OUTPUT_DIR / "behavior_metrics" / "session_metrics.csv"),
	)
	parser.add_argument(
		"--between-stats",
		default=str(DEFAULT_OUTPUT_DIR / "development" / "stats" / "between_group_tests.csv"),
	)
	parser.add_argument(
		"--stable-stats",
		default=str(DEFAULT_OUTPUT_DIR / "development" / "stats" / "stable_proximity_tests.csv"),
	)
	parser.add_argument(
		"--output-dir",
		default=str(DEFAULT_OUTPUT_DIR / "focused_figures"),
	)
	args = parser.parse_args(argv)

	df = pd.read_csv(to_abs_path(args.metrics))
	between = pd.read_csv(to_abs_path(args.between_stats))
	stable_stats = pd.read_csv(to_abs_path(args.stable_stats))
	output_dir = to_abs_path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	plot_locomotion(df, between, output_dir)
	plot_pnd10_stable_proximity(df, stable_stats, output_dir)
	print(f"Saved focused figures to {output_dir}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
