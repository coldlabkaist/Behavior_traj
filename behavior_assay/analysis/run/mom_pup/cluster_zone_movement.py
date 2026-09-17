from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import matplotlib
import pandas as pd
from analysis.core.mom_pup.behavior_metrics import BehaviorConfig, build_position_table, identify_initial_cluster, identify_real_pups, load_session
from analysis.core.mom_pup.common import DEFAULT_OUTPUT_DIR, to_abs_path
from analysis.core.mom_pup.cluster_zone_movement import _as_bool
from analysis.core.mom_pup.cluster_zone_movement import add_zone_occupancy
from analysis.core.mom_pup.cluster_zone_movement import between_group_tests
from analysis.core.mom_pup.cluster_zone_movement import group_summary
from analysis.core.mom_pup.cluster_zone_movement import interaction_tests
from analysis.core.mom_pup.cluster_zone_movement import occupancy_outputs
from analysis.plots.mom_pup.cluster_zone_movement import plot_occupancy
from analysis.core.mom_pup.cluster_zone_movement import process_session
from analysis.core.mom_pup.cluster_zone_movement import stationary_outputs
from analysis.core.mom_pup.cluster_zone_movement import within_group_tests

matplotlib.use("Agg")

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description="Compare maternal movement near versus outside the initial pup cluster.",
	)
	parser.add_argument(
		"--manifest",
		default=str(DEFAULT_OUTPUT_DIR / "manifest" / "session_manifest.csv"),
	)
	parser.add_argument(
		"--output-dir",
		default=str(DEFAULT_OUTPUT_DIR / "cluster_zone_movement"),
	)
	parser.add_argument(
		"--focused-output-dir",
		default=str(DEFAULT_OUTPUT_DIR / "focused_figures"),
	)
	parser.add_argument(
		"--thresholds-mm",
		type=float,
		nargs="+",
		default=[40.0, 60.0, 80.0],
	)
	parser.add_argument("--primary-threshold-mm", type=float, default=60.0)
	args = parser.parse_args(argv)

	thresholds_mm = sorted(set(float(value) for value in args.thresholds_mm))
	if float(args.primary_threshold_mm) not in thresholds_mm:
		parser.error("--primary-threshold-mm must be included in --thresholds-mm")

	config = BehaviorConfig()
	manifest = pd.read_csv(to_abs_path(args.manifest))
	manifest = manifest[_as_bool(manifest["include"])].copy()
	output_dir = to_abs_path(args.output_dir)
	focused_output_dir = to_abs_path(args.focused_output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	focused_output_dir.mkdir(parents=True, exist_ok=True)

	metric_rows: list[dict[str, object]] = []
	issue_rows: list[dict[str, object]] = []
	for row in manifest.to_dict(orient="records"):
		session_id = str(row["session_id"])
		print(f"Zone movement: {session_id} ...")
		try:
			session_rows, notes = process_session(row, config, thresholds_mm)
			metric_rows.extend(session_rows)
			for note in notes:
				issue_rows.append(
					{
						"session_id": session_id,
						"issue": "analysis_warning",
						"detail": note,
					}
				)
		except Exception as exc:
			issue_rows.append(
				{
					"session_id": session_id,
					"issue": "analysis_error",
					"detail": str(exc),
				}
			)

	metrics = add_zone_occupancy(pd.DataFrame(metric_rows))
	issues = pd.DataFrame(
		issue_rows,
		columns=["session_id", "issue", "detail"],
	)
	interaction = interaction_tests(metrics)
	within = within_group_tests(metrics)
	between = between_group_tests(metrics)
	summary = group_summary(metrics)
	occupancy_between, occupancy_summary = occupancy_outputs(metrics)
	(
		stationary_interaction,
		stationary_within,
		stationary_between,
		stationary_summary,
	) = stationary_outputs(metrics)

	metrics.to_csv(output_dir / "session_zone_metrics.csv", index=False)
	interaction.to_csv(output_dir / "interaction_tests.csv", index=False)
	within.to_csv(output_dir / "within_group_tests.csv", index=False)
	between.to_csv(output_dir / "between_group_tests.csv", index=False)
	summary.to_csv(output_dir / "group_summary.csv", index=False)
	occupancy_between.to_csv(
		output_dir / "occupancy_between_group_tests.csv",
		index=False,
	)
	occupancy_summary.to_csv(
		output_dir / "occupancy_group_summary.csv",
		index=False,
	)
	stationary_interaction.to_csv(
		output_dir / "stationary_interaction_tests.csv",
		index=False,
	)
	stationary_within.to_csv(
		output_dir / "stationary_within_group_tests.csv",
		index=False,
	)
	stationary_between.to_csv(
		output_dir / "stationary_between_group_tests.csv",
		index=False,
	)
	stationary_summary.to_csv(
		output_dir / "stationary_group_summary.csv",
		index=False,
	)
	issues.to_csv(output_dir / "processing_issues.csv", index=False)
	plot_occupancy(
		metrics,
		occupancy_between,
		threshold_mm=float(args.primary_threshold_mm),
		output_dir=focused_output_dir,
	)

	print(f"Completed sessions: {metrics['session_id'].nunique()}")
	print(f"Zone metric rows: {len(metrics)}")
	print(f"QC issue rows: {len(issues)}")
	print(f"Wrote {output_dir}")
	print(f"Wrote focused plot to {focused_output_dir}")
	return 0 if not (issues["issue"] == "analysis_error").any() else 1

if __name__ == '__main__':
    raise SystemExit(main())
