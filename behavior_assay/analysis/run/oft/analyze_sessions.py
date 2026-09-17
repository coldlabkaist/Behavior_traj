from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path
import pandas as pd
from analysis.core.oft.common import CENTRE_FRACTION, DEFAULT_OUTPUT_DIR, to_abs_path
from analysis.core.oft.analyze_sessions import _group_summary
from analysis.core.oft.analyze_sessions import _test_table
from analysis.core.oft.analyze_sessions import analyze_session

from analysis.core.oft.analyze_sessions import METRICS

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Compute OFT session metrics and group statistics.")
	parser.add_argument("--preprocess-summary", default=str(DEFAULT_OUTPUT_DIR / "preprocessed" / "preprocess_summary.csv"))
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--analysis-seconds", type=float, default=600.0)
	parser.add_argument("--centre-fraction", type=float, default=CENTRE_FRACTION)
	parser.add_argument("--entry-buffer", type=float, default=0.03)
	parser.add_argument("--max-step-cm", type=float, default=5.0)
	parser.add_argument(
		"--smoothing-window",
		type=int,
		default=5,
		help="Odd centered rolling-median window for Body_C trajectory; use 1 to disable",
	)
	args = parser.parse_args(argv)
	if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
		parser.error("--smoothing-window must be a positive odd integer")

	summary = pd.read_csv(to_abs_path(args.preprocess_summary))
	output_dir = to_abs_path(args.output_dir)
	metrics_dir = output_dir / "metrics"
	stats_dir = output_dir / "stats"
	qc_dir = output_dir / "qc"
	for directory in (metrics_dir, stats_dir, qc_dir):
		directory.mkdir(parents=True, exist_ok=True)

	rows: list[dict[str, object]] = []
	issues: list[dict[str, object]] = []
	for meta in summary.to_dict(orient="records"):
		try:
			df = pd.read_csv(Path(str(meta["preprocessed_path"])))
			metrics = analyze_session(
				df,
				fps=float(meta["fps"]),
				arena_cm=float(meta["arena_cm"]),
				analysis_seconds=float(args.analysis_seconds),
				centre_fraction=float(args.centre_fraction),
				entry_buffer=float(args.entry_buffer),
				max_step_cm=float(args.max_step_cm),
				smoothing_window=int(args.smoothing_window),
			)
			rows.append(
				{
					"session_id": meta["session_id"],
					"subject_id": meta["subject_id"],
					"cage_id": meta["cage_id"],
					"condition": meta["condition"],
					"sex": meta["sex"],
					"animal_id": meta["animal_id"],
					"fps": meta["fps"],
					"arena_cm": meta["arena_cm"],
					"centre_fraction": float(args.centre_fraction),
					"entry_buffer_fraction": float(args.entry_buffer),
					"max_step_cm": float(args.max_step_cm),
					"trajectory_smoothing_method": "centered_rolling_median",
					"trajectory_smoothing_window_frames": int(args.smoothing_window),
					**metrics,
					"preprocessed_path": meta["preprocessed_path"],
					"raw_path": meta["raw_path"],
				}
			)
		except Exception as exc:
			issues.append({"session_id": meta["session_id"], "issue": "analysis_error", "detail": str(exc)})

	session_metrics = pd.DataFrame(rows)
	session_path = metrics_dir / "session_metrics.csv"
	session_metrics.to_csv(session_path, index=False)

	metric_columns = list(METRICS)
	cage_metrics = (
		session_metrics.groupby(["condition", "cage_id"], as_index=False)[metric_columns].mean()
		if not session_metrics.empty
		else pd.DataFrame(columns=["condition", "cage_id", *metric_columns])
	)
	cage_path = metrics_dir / "cage_mean_metrics.csv"
	cage_metrics.to_csv(cage_path, index=False)

	group_summary = pd.concat(
		[
			_group_summary(session_metrics, unit="animal"),
			_group_summary(cage_metrics, unit="cage"),
		],
		ignore_index=True,
	)
	group_summary.to_csv(stats_dir / "group_summary.csv", index=False)
	tests = pd.concat(
		[
			_test_table(session_metrics, unit="animal"),
			_test_table(cage_metrics, unit="cage"),
		],
		ignore_index=True,
	)
	tests.to_csv(stats_dir / "group_tests.csv", index=False)
	pd.DataFrame(issues, columns=["session_id", "issue", "detail"]).to_csv(
		qc_dir / "analysis_issues.csv",
		index=False,
	)
	print(f"Analyzed sessions: {len(session_metrics)}")
	print(f"Cage means: {len(cage_metrics)}")
	print(f"Analysis errors: {len(issues)}")
	print(f"Wrote {session_path}")
	print(f"Wrote {cage_path}")
	print(f"Wrote {stats_dir / 'group_summary.csv'}")
	print(f"Wrote {stats_dir / 'group_tests.csv'}")
	return 0 if not issues else 1

if __name__ == '__main__':
    raise SystemExit(main())
