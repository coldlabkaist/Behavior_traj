from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import pandas as pd
from analysis.core.mom_pup.common import DEFAULT_OUTPUT_DIR, to_abs_path
from analysis.core.mom_pup.behavior_metrics import BehaviorConfig
from analysis.core.mom_pup.behavior_metrics import _as_bool
from analysis.core.mom_pup.behavior_metrics import process_session

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Compute longitudinal mother-pup clustering, dispersion, approach, and locomotion metrics.")
	parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_DIR / "manifest" / "session_manifest.csv"))
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "behavior_metrics"))
	parser.add_argument("--score-threshold", type=float, default=0.5)
	parser.add_argument("--cluster-threshold-mm", type=float, default=50.0)
	parser.add_argument("--mom-close-threshold-mm", type=float, default=60.0)
	parser.add_argument("--stray-threshold-mm", type=float, default=80.0)
	parser.add_argument("--stable-proximity-min-duration-sec", type=float, default=3.0)
	args = parser.parse_args(argv)

	config = BehaviorConfig(
		score_threshold=float(args.score_threshold),
		cluster_threshold_mm=float(args.cluster_threshold_mm),
		mom_close_threshold_mm=float(args.mom_close_threshold_mm),
		stray_threshold_mm=float(args.stray_threshold_mm),
		stable_proximity_min_duration_sec=float(args.stable_proximity_min_duration_sec),
	)
	manifest = pd.read_csv(to_abs_path(args.manifest))
	manifest = manifest[_as_bool(manifest["include"])].copy()
	output_dir = to_abs_path(args.output_dir)
	qc_dir = output_dir / "qc"
	output_dir.mkdir(parents=True, exist_ok=True)
	qc_dir.mkdir(parents=True, exist_ok=True)

	session_rows: list[dict[str, object]] = []
	event_rows: list[dict[str, object]] = []
	issue_rows: list[dict[str, object]] = []
	for row in manifest.to_dict(orient="records"):
		session_id = str(row["session_id"])
		print(f"Processing {session_id} ...")
		try:
			session, events, notes = process_session(row, config)
			session["qc_status"] = "warning" if notes else "ok"
			session["qc_notes"] = ";".join(notes)
			session_rows.append(session)
			event_rows.extend(events)
			for note in notes:
				issue_rows.append({"session_id": session_id, "issue": "analysis_warning", "detail": note})
		except Exception as exc:
			issue_rows.append({"session_id": session_id, "issue": "analysis_error", "detail": str(exc)})

	session_df = pd.DataFrame(session_rows)
	event_df = pd.DataFrame(event_rows)
	issues_df = pd.DataFrame(issue_rows, columns=["session_id", "issue", "detail"])
	session_path = output_dir / "session_metrics.csv"
	event_path = output_dir / "dispersion_events.csv"
	issues_path = qc_dir / "processing_issues.csv"
	session_df.to_csv(session_path, index=False)
	event_df.to_csv(event_path, index=False)
	issues_df.to_csv(issues_path, index=False)
	print(f"Manifest sessions: {len(manifest)}")
	print(f"Completed sessions: {len(session_df)}")
	print(f"Dispersion events: {len(event_df)}")
	print(f"QC issue rows: {len(issues_df)}")
	print(f"Wrote {session_path}")
	print(f"Wrote {event_path}")
	print(f"Wrote {issues_path}")
	return 0 if not (issues_df["issue"] == "analysis_error").any() else 1

if __name__ == '__main__':
    raise SystemExit(main())
