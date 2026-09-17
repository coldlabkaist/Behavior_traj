from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import matplotlib
import pandas as pd
from analysis.core.mom_pup.behavior_metrics import BehaviorConfig, build_position_table, identify_real_pups, load_session
from analysis.core.mom_pup.common import DEFAULT_OUTPUT_DIR, to_abs_path
from analysis.core.mom_pup.pup_zone_dwell import DwellConfig
from analysis.core.mom_pup.pup_zone_dwell import _as_bool
from analysis.core.mom_pup.pup_zone_dwell import interaction_tests
from analysis.plots.mom_pup.pup_zone_dwell import plot_dwell
from analysis.core.mom_pup.pup_zone_dwell import process_session
from analysis.core.mom_pup.pup_zone_dwell import zone_tests

matplotlib.use("Agg")

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Test whether mothers dwell longer in a pup-centered zone than "
			"in geometry-matched non-pup zones."
		),
	)
	parser.add_argument(
		"--manifest",
		default=str(DEFAULT_OUTPUT_DIR / "manifest" / "session_manifest.csv"),
	)
	parser.add_argument(
		"--output-dir",
		default=str(DEFAULT_OUTPUT_DIR / "pup_zone_dwell"),
	)
	parser.add_argument(
		"--focused-output-dir",
		default=str(DEFAULT_OUTPUT_DIR / "focused_figures"),
	)
	args = parser.parse_args(argv)

	behavior_config = BehaviorConfig()
	dwell_config = DwellConfig()
	manifest = pd.read_csv(to_abs_path(args.manifest))
	manifest = manifest[
		_as_bool(manifest["include"])
		& (pd.to_numeric(manifest["pnd"], errors="coerce") == dwell_config.pnd)
	].copy()
	output_dir = to_abs_path(args.output_dir)
	focused_output_dir = to_abs_path(args.focused_output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	focused_output_dir.mkdir(parents=True, exist_ok=True)

	metric_rows: list[dict[str, object]] = []
	event_rows: list[dict[str, object]] = []
	qc_rows: list[dict[str, object]] = []
	issue_rows: list[dict[str, object]] = []
	for row in manifest.to_dict(orient="records"):
		session_id = str(row["session_id"])
		print(f"Pup-zone dwell: {session_id} ...")
		try:
			session_metrics, session_events, session_qc = process_session(
				row,
				behavior_config,
				dwell_config,
			)
			metric_rows.extend(session_metrics)
			event_rows.extend(session_events)
			qc_rows.append(session_qc)
		except Exception as exc:
			issue_rows.append(
				{
					"session_id": session_id,
					"issue": "analysis_error",
					"detail": str(exc),
				}
			)

	metrics = pd.DataFrame(metric_rows)
	events = pd.DataFrame(event_rows)
	qc = pd.DataFrame(qc_rows)
	issues = pd.DataFrame(
		issue_rows,
		columns=["session_id", "issue", "detail"],
	)
	if metrics.empty:
		raise RuntimeError("no pup-zone dwell metrics were produced")
	interaction, contrasts = interaction_tests(metrics)
	zone_results = zone_tests(metrics)

	metrics.to_csv(output_dir / "session_zone_metrics.csv", index=False)
	events.to_csv(output_dir / "visit_events.csv", index=False)
	qc.to_csv(output_dir / "zone_definition_qc.csv", index=False)
	issues.to_csv(output_dir / "processing_issues.csv", index=False)
	interaction.to_csv(output_dir / "group_by_zone_interaction.csv", index=False)
	contrasts.to_csv(output_dir / "subject_dwell_contrasts.csv", index=False)
	zone_results.to_csv(output_dir / "within_zone_group_tests.csv", index=False)
	plot_dwell(
		metrics,
		interaction,
		zone_results,
		focused_output_dir,
	)

	print(f"Completed sessions: {metrics['session_id'].nunique()}")
	print(f"Visit events: {len(events)}")
	print(f"QC issue rows: {len(issues)}")
	print(f"Wrote {output_dir}")
	print(f"Wrote focused plot to {focused_output_dir}")
	return 0 if not len(issues) else 1

if __name__ == '__main__':
    raise SystemExit(main())
