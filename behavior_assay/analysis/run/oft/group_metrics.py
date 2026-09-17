from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.core.oft.analyze_sessions import METRICS
from analysis.core.oft.common import DEFAULT_OUTPUT_DIR, to_abs_path
from analysis.plots.oft.group_metrics import plot_core_oft_bars

matplotlib.use("Agg")

from analysis.plots.oft.group_metrics import COLORS

from analysis.plots.oft.group_metrics import CONDITIONS

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Plot OFT animal values with cage means and FDR-adjusted tests.")
	parser.add_argument("--metrics", default=str(DEFAULT_OUTPUT_DIR / "metrics" / "session_metrics.csv"))
	parser.add_argument("--tests", default=str(DEFAULT_OUTPUT_DIR / "stats" / "group_tests.csv"))
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "figures" / "group_metrics"))
	args = parser.parse_args(argv)

	df = pd.read_csv(to_abs_path(args.metrics))
	tests = pd.read_csv(to_abs_path(args.tests))
	output_dir = to_abs_path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	rng = np.random.default_rng(20260728)

	for metric, ylabel in METRICS.items():
		fig, ax = plt.subplots(figsize=(4.4, 5.0))
		for x_pos, condition in enumerate(CONDITIONS):
			group = df[df["condition"] == condition]
			values = pd.to_numeric(group[metric], errors="coerce")
			valid = values.notna()
			jitter = rng.uniform(-0.12, 0.12, valid.sum())
			ax.scatter(
				x_pos + jitter,
				values[valid],
				s=34,
				facecolors="white",
				edgecolors=COLORS[condition],
				linewidths=1.3,
				alpha=0.85,
				zorder=2,
			)
			cage_means = group.assign(_value=values).groupby("cage_id")["_value"].mean().dropna()
			ax.scatter(
				np.full(len(cage_means), x_pos),
				cage_means,
				marker="D",
				s=45,
				color=COLORS[condition],
				edgecolors="black",
				linewidths=0.5,
				zorder=3,
				label=f"{condition} cage mean" if metric == next(iter(METRICS)) else None,
			)
			if valid.any():
				mean = values[valid].mean()
				sem = values[valid].sem() if valid.sum() > 1 else 0.0
				ax.errorbar(
					x_pos,
					mean,
					yerr=sem,
					fmt="_",
					markersize=20,
					color="black",
					capsize=5,
					linewidth=1.6,
					zorder=4,
				)

		test = tests[
			(tests["analysis_unit"] == "animal")
			& (tests["metric"] == metric)
		]
		if not test.empty and pd.notna(test.iloc[0][p_column]):
			q_value = float(test.iloc[0][p_column])
			ax.text(0.5, 0.98, f"animal-level FDR q={q_value:.3g}", ha="center", va="top", transform=ax.transAxes, fontsize=9)
		ax.set_xticks([0, 1], ["Control", "VPA"])
		ax.set_ylabel(ylabel)
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)
		ax.set_title(metric.replace("_", " "), fontsize=11)
		fig.tight_layout()
		fig.savefig(output_dir / f"{metric}.png", dpi=250)
		plt.close(fig)

	plot_core_oft_bars(df, tests, output_dir)
	print(f"Saved {len(METRICS)} metric figures and core OFT panel to {output_dir}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
