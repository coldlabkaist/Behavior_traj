from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import matplotlib
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from analysis.core.mom_pup.common import DEFAULT_OUTPUT_DIR, to_abs_path
from analysis.core.mom_pup.development import between_group_tests
from analysis.core.mom_pup.development import group_summary
from analysis.core.mom_pup.development import longitudinal_model
from analysis.plots.mom_pup.development import plot_metric
from analysis.core.mom_pup.development import stable_proximity_tests
from analysis.core.mom_pup.development import within_group_tests

matplotlib.use("Agg")

try:
	import pingouin as pg
except ImportError:
	pg = None

from analysis.core.mom_pup.development import PNDS

from analysis.core.mom_pup.development import METRICS

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Run longitudinal mom-pup statistics and developmental plots.")
	parser.add_argument("--metrics", default=str(DEFAULT_OUTPUT_DIR / "behavior_metrics" / "session_metrics.csv"))
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "development"))
	args = parser.parse_args(argv)

	df = pd.read_csv(to_abs_path(args.metrics))
	df["pnd"] = pd.to_numeric(df["pnd"], errors="coerce").astype("Int64")
	df = df[df["pnd"].isin(PNDS) & df["condition"].isin(["Control", "VPA"])].copy()
	if df.duplicated(["subject_id", "pnd"]).any():
		raise ValueError("Duplicate subject_id/PND rows found; fix the manifest before statistics")

	output_dir = to_abs_path(args.output_dir)
	stats_dir = output_dir / "stats"
	figure_dir = output_dir / "figures"
	stats_dir.mkdir(parents=True, exist_ok=True)
	figure_dir.mkdir(parents=True, exist_ok=True)

	between_tables: list[pd.DataFrame] = []
	within_tables: list[pd.DataFrame] = []
	model_tables: list[pd.DataFrame] = []
	for metric, ylabel in METRICS.items():
		if metric not in df:
			print(f"Skipping missing metric: {metric}")
			continue
		between = between_group_tests(df, metric)
		within = within_group_tests(df, metric)
		model = longitudinal_model(df, metric)
		between_tables.append(between)
		within_tables.append(within)
		model_tables.append(model)
		plot_metric(df, metric, ylabel, between, figure_dir / f"trajectory_{metric}.png")

	between_df = pd.concat(between_tables, ignore_index=True)
	within_df = pd.concat(within_tables, ignore_index=True)
	model_df = pd.concat(model_tables, ignore_index=True, sort=False)
	valid_between = between_df["p_value"].notna()
	between_df["p_fdr_all_metrics_timepoints"] = np.nan
	if valid_between.any():
		between_df.loc[valid_between, "p_fdr_all_metrics_timepoints"] = multipletests(
			between_df.loc[valid_between, "p_value"],
			method="fdr_bh",
		)[1]
	valid_within = within_df["p_value"].notna()
	within_df["p_fdr_all_metrics_pairs"] = np.nan
	if valid_within.any():
		within_df.loc[valid_within, "p_fdr_all_metrics_pairs"] = multipletests(
			within_df.loc[valid_within, "p_value"],
			method="fdr_bh",
		)[1]
	if "p_value" in model_df:
		valid_model = model_df["p_value"].notna()
		model_df["p_fdr_all_model_terms"] = np.nan
		if valid_model.any():
			model_df.loc[valid_model, "p_fdr_all_model_terms"] = multipletests(
				model_df.loc[valid_model, "p_value"],
				method="fdr_bh",
			)[1]

	group_summary(df).to_csv(stats_dir / "group_summary.csv", index=False)
	between_df.to_csv(stats_dir / "between_group_tests.csv", index=False)
	within_df.to_csv(stats_dir / "within_group_tests.csv", index=False)
	model_df.to_csv(stats_dir / "longitudinal_models.csv", index=False)
	if {
		"total_stable_proximity_time_sec",
		"n_stable_proximity_bouts",
	}.issubset(df.columns):
		stable_proximity_tests(df).to_csv(
			stats_dir / "stable_proximity_tests.csv",
			index=False,
		)
	print(f"Rows used: {len(df)}")
	print(f"Subjects: {df['subject_id'].nunique()}")
	print(f"Saved statistics to {stats_dir}")
	print(f"Saved figures to {figure_dir}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
