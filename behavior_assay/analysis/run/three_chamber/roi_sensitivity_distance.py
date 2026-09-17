from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from analysis.core.three_chamber.roi_sensitivity_distance import _build_roi_lookup
from analysis.core.three_chamber.roi_sensitivity_distance import _build_sample_df
from analysis.core.three_chamber.roi_sensitivity_distance import _compute_session_metrics
from analysis.core.three_chamber.roi_sensitivity_distance import _format_minutes_suffix
from analysis.core.three_chamber.roi_sensitivity_distance import _normalize_session_n
from analysis.core.three_chamber.roi_sensitivity_distance import _parse_phase_list
from analysis.core.three_chamber.roi_sensitivity_distance import _parse_radius_scales
from analysis.core.three_chamber.roi_sensitivity_distance import _parse_session_n_filter
from analysis.plots.three_chamber.roi_sensitivity_distance import _plot_distance_phase
from analysis.plots.three_chamber.roi_sensitivity_distance import _plot_sensitivity_curve
from analysis.plots.three_chamber.roi_sensitivity_distance import _summarize_distance_stats
from analysis.core.three_chamber.roi_sensitivity_distance import _summarize_group
from analysis.plots.three_chamber.roi_sensitivity_distance import _summarize_sensitivity_stats
from analysis.core.three_chamber.roi_sensitivity_distance import _to_abs_path

import argparse
from analysis.core.three_chamber.roi_sensitivity_distance import DEFAULT_RADIUS_SCALES

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run ROI radius sensitivity and Nose-to-cup distance analysis for 3-chamber data.")
	parser.add_argument("--keypoint", default="Nose", help="Keypoint used for distance and ROI sensitivity.")
	parser.add_argument("--preprocess-summary", default="output/3chamber/preprocessed/preprocess_summary.csv")
	parser.add_argument("--roi-summary", default="output/3chamber/roi/roi_summary.csv")
	parser.add_argument("--output-dir", default="output/3chamber/roi_sensitivity")
	parser.add_argument("--radius-scales", default=DEFAULT_RADIUS_SCALES)
	parser.add_argument("--phases", default="soc,nov")
	parser.add_argument("--session-n", default="", help="Optional comma-separated session filter, e.g. 1, 2, or 1,2.")
	parser.add_argument("--sample-unit", choices=["trial_mean", "session"], default="trial_mean")
	parser.add_argument("--fps", type=float, default=30.0, help="Frames per second for time-window filtering and seconds conversion.")
	parser.add_argument(
		"--max-minutes",
		type=float,
		default=0.0,
		help="Analyze only the first N minutes of each recording. Default 0 uses the full session.",
	)
	return parser.parse_args()

def main() -> None:
	args = parse_args()
	phases = _parse_phase_list(args.phases)
	radius_scales = _parse_radius_scales(args.radius_scales)
	output_dir = _to_abs_path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	preprocess_df = pd.read_csv(_to_abs_path(args.preprocess_summary))
	roi_df = pd.read_csv(_to_abs_path(args.roi_summary))
	work = preprocess_df[preprocess_df["phase"].astype(str).str.lower().isin(phases)].copy()
	if args.session_n:
		allowed = set(_parse_session_n_filter(args.session_n))
		work["_session_n_norm"] = work["session_n"].map(_normalize_session_n)
		work = work[work["_session_n_norm"].isin(allowed)].copy()
		work = work.drop(columns=["_session_n_norm"])
	work["fps"] = float(args.fps)
	roi_lookup = _build_roi_lookup(roi_df)

	sensitivity_rows: list[dict[str, object]] = []
	distance_rows: list[dict[str, object]] = []
	issue_rows: list[dict[str, object]] = []
	for row in work.to_dict(orient="records"):
		s_rows, d_row, issues = _compute_session_metrics(row, roi_lookup, radius_scales, args.keypoint, float(args.max_minutes))
		sensitivity_rows.extend(s_rows)
		if d_row is not None:
			distance_rows.append(d_row)
		issue_rows.extend(issues)

	sensitivity_session_df = pd.DataFrame(sensitivity_rows)
	distance_session_df = pd.DataFrame(distance_rows)
	issues_df = pd.DataFrame(issue_rows)
	sensitivity_sample_df = _build_sample_df(sensitivity_session_df, args.sample_unit, "sensitivity")
	distance_sample_df = _build_sample_df(distance_session_df, args.sample_unit, "distance")
	sensitivity_stats_df = _summarize_sensitivity_stats(sensitivity_sample_df)
	distance_stats_df = _summarize_distance_stats(distance_sample_df)
	sensitivity_group_df = _summarize_group(
		sensitivity_sample_df,
		["target_time_s", "opposite_time_s", "preference_index", "target_visit_count", "opposite_visit_count"],
		["phase", "radius_scale", "condition"],
	)
	distance_group_df = _summarize_group(
		distance_sample_df,
		["target_distance_mean_norm", "opposite_distance_mean_norm", "distance_delta_norm", "distance_preference_index"],
		["phase", "condition"],
	)

	session_suffix = ""
	if args.session_n:
		session_suffix = "__n" + "_".join(_parse_session_n_filter(args.session_n))
	minutes_suffix = _format_minutes_suffix(float(args.max_minutes))
	suffix = f"{args.keypoint}{session_suffix}{minutes_suffix}"

	paths = {
		"sensitivity_session": output_dir / f"roi_sensitivity_session_summary__{suffix}.csv",
		"sensitivity_sample": output_dir / f"roi_sensitivity_sample_summary__{suffix}.csv",
		"sensitivity_group": output_dir / f"roi_sensitivity_group_summary__{suffix}.csv",
		"sensitivity_stats": output_dir / f"roi_sensitivity_stats__{suffix}.csv",
		"distance_session": output_dir / f"distance_session_summary__{suffix}.csv",
		"distance_sample": output_dir / f"distance_sample_summary__{suffix}.csv",
		"distance_group": output_dir / f"distance_group_summary__{suffix}.csv",
		"distance_stats": output_dir / f"distance_stats__{suffix}.csv",
		"issues": output_dir / f"roi_sensitivity_distance_issues__{suffix}.csv",
		"sensitivity_plot": output_dir / f"roi_sensitivity_curve__{suffix}.png",
	}
	sensitivity_session_df.to_csv(paths["sensitivity_session"], index=False)
	sensitivity_sample_df.to_csv(paths["sensitivity_sample"], index=False)
	sensitivity_group_df.to_csv(paths["sensitivity_group"], index=False)
	sensitivity_stats_df.to_csv(paths["sensitivity_stats"], index=False)
	distance_session_df.to_csv(paths["distance_session"], index=False)
	distance_sample_df.to_csv(paths["distance_sample"], index=False)
	distance_group_df.to_csv(paths["distance_group"], index=False)
	distance_stats_df.to_csv(paths["distance_stats"], index=False)
	issues_df.to_csv(paths["issues"], index=False)

	_plot_sensitivity_curve(sensitivity_sample_df, sensitivity_stats_df, paths["sensitivity_plot"])
	distance_plot_paths = []
	for phase in phases:
		if phase not in set(distance_sample_df["phase"].astype(str)):
			continue
		out_path = output_dir / f"distance_metrics__{phase}__{suffix}.png"
		_plot_distance_phase(distance_sample_df, distance_stats_df, phase, out_path)
		distance_plot_paths.append(out_path)

	print(f"Keypoint: {args.keypoint}")
	print(f"FPS: {float(args.fps):g}")
	print(f"Radius scales: {','.join(f'{v:.2f}' for v in radius_scales)}")
	if args.session_n:
		print(f"Session n filter: {args.session_n}")
	if float(args.max_minutes) > 0:
		print(f"Analysis window: first {float(args.max_minutes):g} min")
	print(f"Sessions analyzed: {distance_session_df['session_key'].nunique() if not distance_session_df.empty else 0}")
	print(f"Samples analyzed: {distance_sample_df['sample_key'].nunique() if not distance_sample_df.empty and 'sample_key' in distance_sample_df.columns else 0}")
	print(f"Issues: {len(issues_df)}")
	for path in paths.values():
		print(f"Wrote {path.relative_to(ROOT)}")
	for path in distance_plot_paths:
		print(f"Wrote {path.relative_to(ROOT)}")
	show = sensitivity_stats_df[
		(sensitivity_stats_df["metric"].eq("preference_index"))
		& (sensitivity_stats_df["radius_scale"].isin([min(radius_scales), 1.5, max(radius_scales)]))
	].copy()
	if not show.empty:
		print(show[["analysis", "phase", "radius_scale", "condition", "comparison", "n", "pvalue", "p_label"]].to_string(index=False))
	show_dist = distance_stats_df[distance_stats_df["metric"].isin(["distance_preference_index", "distance_delta_norm"])].copy()
	if not show_dist.empty:
		print(show_dist[["analysis", "phase", "condition", "metric", "comparison", "n", "pvalue", "p_label"]].to_string(index=False))

if __name__ == '__main__':
    raise SystemExit(main())
