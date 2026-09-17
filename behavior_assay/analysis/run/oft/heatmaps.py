from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from analysis.core.oft.common import DEFAULT_OUTPUT_DIR, to_abs_path
from analysis.plots.oft.heatmaps import _session_probability
from analysis.plots.oft.heatmaps import plot_group_densities

matplotlib.use("Agg")

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Create equal-session-weight OFT group heatmaps.")
	parser.add_argument("--preprocess-summary", default=str(DEFAULT_OUTPUT_DIR / "preprocessed" / "preprocess_summary.csv"))
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "figures" / "heatmaps"))
	parser.add_argument("--analysis-seconds", type=float, default=600.0)
	parser.add_argument("--bins", type=int, default=100)
	parser.add_argument("--sigma", type=float, default=2.5)
	args = parser.parse_args(argv)

	summary = pd.read_csv(to_abs_path(args.preprocess_summary))
	output_dir = to_abs_path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	group_densities: dict[str, np.ndarray] = {}
	for condition, group in summary.groupby("condition"):
		session_maps = [
			_session_probability(
				Path(str(row["preprocessed_path"])),
				fps=float(row["fps"]),
				seconds=float(args.analysis_seconds),
				bins=int(args.bins),
			)
			for row in group.to_dict(orient="records")
		]
		density = np.mean(session_maps, axis=0)
		density = gaussian_filter(density, sigma=float(args.sigma))
		group_densities[str(condition)] = density
		np.savetxt(output_dir / f"density_{str(condition).lower()}.csv", density, delimiter=",")

	return plot_group_densities(group_densities, output_dir)

if __name__ == '__main__':
    raise SystemExit(main())
