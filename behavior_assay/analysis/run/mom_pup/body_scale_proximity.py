from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import matplotlib
import pandas as pd
from analysis.core.mom_pup.behavior_metrics import BehaviorConfig, _as_bool, _find_runs, build_position_table, identify_real_pups, load_session
from analysis.core.mom_pup.common import DEFAULT_OUTPUT_DIR, to_abs_path
from analysis.core.mom_pup.body_scale_proximity import between_group_tests
from analysis.core.mom_pup.body_scale_proximity import group_summary
from analysis.plots.mom_pup.body_scale_proximity import plot_body_scale_proximity
from analysis.core.mom_pup.body_scale_proximity import process_session

matplotlib.use("Agg")

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Measure time within a pose-derived body-scale distance of any pup."
		)
	)
	parser.add_argument(
		"--manifest",
		default=str(DEFAULT_OUTPUT_DIR / "manifest" / "session_manifest.csv"),
	)
	parser.add_argument(
		"--output-dir",
		default=str(DEFAULT_OUTPUT_DIR / "body_scale_proximity"),
	)
	parser.add_argument(
		"--figure-dir",
		default=str(DEFAULT_OUTPUT_DIR / "focused_figures"),
	)
	parser.add_argument("--score-threshold", type=float, default=0.5)
	parser.add_argument("--min-bout-duration-sec", type=float, default=1.0)
	args = parser.parse_args(argv)

	config = BehaviorConfig(score_threshold=float(args.score_threshold))
	manifest = pd.read_csv(to_abs_path(args.manifest))
	manifest = manifest[_as_bool(manifest["include"])].copy()
	output_dir = to_abs_path(args.output_dir)
	figure_dir = to_abs_path(args.figure_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	session_rows: list[dict[str, object]] = []
	issue_rows: list[dict[str, object]] = []
	for row in manifest.to_dict(orient="records"):
		print(f"Processing {row['session_id']} ...")
		try:
			session_rows.append(
				process_session(
					row,
					config,
					min_bout_duration_sec=float(args.min_bout_duration_sec),
				)
			)
		except Exception as exc:
			issue_rows.append(
				{
					"session_id": row["session_id"],
					"issue": "analysis_error",
					"detail": str(exc),
				}
			)

	metrics = pd.DataFrame(session_rows).sort_values(
		["condition", "subject_id", "pnd"]
	)
	if metrics.duplicated(["subject_id", "pnd"]).any():
		raise ValueError("Duplicate subject_id/PND rows found")
	tests = between_group_tests(metrics)
	summary = group_summary(metrics)
	issues = pd.DataFrame(
		issue_rows,
		columns=["session_id", "issue", "detail"],
	)

	metrics.to_csv(output_dir / "session_metrics.csv", index=False)
	tests.to_csv(output_dir / "between_group_tests.csv", index=False)
	summary.to_csv(output_dir / "group_summary.csv", index=False)
	issues.to_csv(output_dir / "processing_issues.csv", index=False)
	plot_body_scale_proximity(metrics, tests, figure_dir)

	print(f"Completed sessions: {len(metrics)}")
	print(f"QC issue rows: {len(issues)}")
	print(f"Saved metrics and statistics to {output_dir}")
	print(f"Saved focused figure to {figure_dir}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
