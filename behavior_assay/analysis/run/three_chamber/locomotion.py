from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import math
import numpy as np
import pandas as pd
from analysis.core.three_chamber.homography import CANONICAL_WIDTH, apply_homography, wall_homography
from analysis.core.three_chamber.locomotion import _analysis_phase
from analysis.plots.three_chamber.locomotion import _analyze_open_spatial_preference_strength
from analysis.core.three_chamber.locomotion import _build_sample_df
from analysis.core.three_chamber.locomotion import _condition_sort_key
from analysis.core.three_chamber.locomotion import _filter_session_n_with_matching_hab
from analysis.core.three_chamber.locomotion import _parse_phase_list
from analysis.core.three_chamber.locomotion import _parse_session_n_filter
from analysis.core.three_chamber.locomotion import _phase_sort_key
from analysis.plots.three_chamber.locomotion import _plot_metric
from analysis.plots.three_chamber.locomotion import _plot_metric_by_sex
from analysis.plots.three_chamber.locomotion import _plot_open_spatial_preference
from analysis.plots.three_chamber.locomotion import _plot_open_spatial_preference_by_sex
from analysis.core.three_chamber.locomotion import _point_in_polygon_mask
from analysis.core.three_chamber.locomotion import _sem
from analysis.core.three_chamber.locomotion import _to_abs_path

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Plot 3-chamber Body_C locomotion summaries by phase.")
	parser.add_argument(
		"--manifest",
		default="output/3chamber/manifest/session_manifest.csv",
		help="Manifest CSV produced by build_manifest.py",
	)
	parser.add_argument(
		"--preprocess-summary",
		default="output/3chamber/preprocessed/preprocess_summary.csv",
		help="Preprocess summary CSV produced by preprocess_sessions.py",
	)
	parser.add_argument(
		"--roi-vertices",
		default="output/3chamber/roi/roi_vertices.csv",
		help="ROI vertices CSV produced by build_roi_geometry.py",
	)
	parser.add_argument(
		"--output-dir",
		default="output/3chamber/locomotion",
		help="Directory to write locomotion figures and summary CSVs",
	)
	parser.add_argument("--keypoint", default="Body_C", help="Keypoint to use for locomotion, e.g. Body_C")
	parser.add_argument("--phases", default="hab_closed,hab_open,soc,nov", help="Comma-separated phase list to summarize")
	parser.add_argument("--fps", type=float, default=30.0, help="Frames per second for duration and speed metrics")
	parser.add_argument(
		"--session-n",
		default="",
		help="Optional session number filter for soc/nov, e.g. 1, n_1, or 1,2. Matching habituation rows are retained for the same subject/trial.",
	)
	parser.add_argument(
		"--coord-mode",
		default="wall_bbox",
		choices=["wall_bbox", "homography"],
		help="Coordinate mode for arena-normalized metrics. wall_bbox keeps the current wall bounding-box normalization; homography rectifies each session using the wall corners.",
	)
	parser.add_argument(
		"--sample-unit",
		default="trial_mean",
		choices=["trial_mean", "session"],
		help="Sample unit for plotting/statistics. trial_mean averages n1/n2 within subject_id + trial_id before plotting.",
	)
	args = parser.parse_args(argv)

	manifest_path = _to_abs_path(args.manifest)
	preprocess_path = _to_abs_path(args.preprocess_summary)
	roi_vertices_path = _to_abs_path(args.roi_vertices)
	output_dir = _to_abs_path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	phase_list = _parse_phase_list(args.phases)
	session_n_filter = _parse_session_n_filter(args.session_n)
	file_suffix = "__homography" if args.coord_mode == "homography" else ""
	file_suffix += f"__n{'_'.join(session_n_filter)}" if session_n_filter else ""
	for path in (manifest_path, preprocess_path, roi_vertices_path):
		if not path.exists():
			raise FileNotFoundError(f"Required input not found: {path}")

	preprocess_df = pd.read_csv(preprocess_path)
	roi_vertices_df = pd.read_csv(roi_vertices_path)

	sessions_df = preprocess_df.copy()
	if "phase_detail" not in sessions_df.columns:
		sessions_df["phase_detail"] = sessions_df.apply(_analysis_phase, axis=1)
	if "hab_type" not in sessions_df.columns:
		sessions_df["hab_type"] = sessions_df["phase_detail"].map(lambda value: str(value).replace("hab_", "") if str(value).startswith("hab_") else "")
	sessions_df["analysis_phase"] = sessions_df.apply(_analysis_phase, axis=1)
	sessions_df = sessions_df[sessions_df["analysis_phase"].isin(phase_list)].copy()
	if session_n_filter:
		sessions_df = _filter_session_n_with_matching_hab(sessions_df, session_n_filter)
	if sessions_df.empty:
		print("No sessions matched the requested filters.")
		return 0

	xcol = f"{args.keypoint}.x"
	ycol = f"{args.keypoint}.y"
	session_rows: list[dict[str, object]] = []
	issue_rows: list[dict[str, object]] = []

	for row in sessions_df.to_dict(orient="records"):
		session_key = str(row["session_key"])
		preprocessed_csv = _to_abs_path(str(row.get("output_path", "")))
		if not preprocessed_csv.exists():
			issue_rows.append({"session_key": session_key, "issue": "preprocessed_file_not_found", "detail": str(preprocessed_csv)})
			continue

		df = pd.read_csv(preprocessed_csv)
		if xcol not in df.columns or ycol not in df.columns:
			issue_rows.append({"session_key": session_key, "issue": "missing_keypoint_columns", "detail": args.keypoint})
			continue

		wall_vertices = roi_vertices_df[(roi_vertices_df["session_key"] == session_key) & (roi_vertices_df["roi_id"] == "wall")].sort_values("vertex_idx")
		if wall_vertices.empty:
			issue_rows.append({"session_key": session_key, "issue": "missing_wall_vertices", "detail": ""})
			continue
		wall_poly = wall_vertices[["x_norm", "y_norm"]].astype(float).to_numpy()

		x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
		y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
		valid = np.isfinite(x) & np.isfinite(y)

		if args.coord_mode == "homography":
			try:
				homography = wall_homography(wall_poly)
			except ValueError as exc:
				issue_rows.append({"session_key": session_key, "issue": "invalid_wall_homography", "detail": str(exc)})
				continue
			coord = apply_homography(np.column_stack([x, y]), homography)
			x_norm = coord[:, 0]
			y_norm = coord[:, 1]
			wall_coord = apply_homography(wall_poly, homography)
			valid &= np.isfinite(x_norm) & np.isfinite(y_norm)
			valid &= _point_in_polygon_mask(x_norm, y_norm, wall_coord)
		else:
			valid &= _point_in_polygon_mask(x, y, wall_poly)

			min_x = float(np.min(wall_poly[:, 0]))
			max_x = float(np.max(wall_poly[:, 0]))
			min_y = float(np.min(wall_poly[:, 1]))
			max_y = float(np.max(wall_poly[:, 1]))
			width = max(max_x - min_x, 1e-9)
			height = max(max_y - min_y, 1e-9)
			x_norm = (x - min_x) / width
			y_norm = (y - min_y) / height

		valid_frames = int(valid.sum())
		if valid_frames < 2:
			issue_rows.append({"session_key": session_key, "issue": "insufficient_valid_frames", "detail": str(valid_frames)})
			continue

		valid_x_fraction = x_norm[valid] / (CANONICAL_WIDTH if args.coord_mode == "homography" else 1.0)
		left_zone_frames = int((valid_x_fraction < (1.0 / 3.0)).sum())
		center_zone_frames = int(((valid_x_fraction >= (1.0 / 3.0)) & (valid_x_fraction <= (2.0 / 3.0))).sum())
		right_zone_frames = int((valid_x_fraction > (2.0 / 3.0)).sum())
		left_zone_time_s = left_zone_frames / max(float(args.fps), 1e-9)
		center_zone_time_s = center_zone_frames / max(float(args.fps), 1e-9)
		right_zone_time_s = right_zone_frames / max(float(args.fps), 1e-9)
		left_zone_pct = 100.0 * left_zone_frames / valid_frames
		center_zone_pct = 100.0 * center_zone_frames / valid_frames
		right_zone_pct = 100.0 * right_zone_frames / valid_frames
		side_time_den = left_zone_time_s + right_zone_time_s
		spatial_preference_index = ((right_zone_time_s - left_zone_time_s) / side_time_den) if side_time_den > 0 else math.nan
		step_valid = valid[:-1] & valid[1:]
		dx_px = np.diff(x)
		dy_px = np.diff(y)
		dx_norm = np.diff(x_norm)
		dy_norm = np.diff(y_norm)
		step_distance_px = np.sqrt(dx_px * dx_px + dy_px * dy_px)
		step_distance_norm = np.sqrt(dx_norm * dx_norm + dy_norm * dy_norm)

		total_distance_px = float(np.nansum(step_distance_px[step_valid]))
		total_distance_norm = float(np.nansum(step_distance_norm[step_valid]))
		valid_steps = int(step_valid.sum())
		duration_s = float(valid_frames / max(float(args.fps), 1e-9))
		duration_min = duration_s / 60.0
		distance_per_min_px = total_distance_px / duration_min if duration_min > 0 else math.nan
		distance_per_min_norm = total_distance_norm / duration_min if duration_min > 0 else math.nan
		mean_speed_px_per_s = total_distance_px / duration_s if duration_s > 0 else math.nan
		mean_speed_norm_per_s = total_distance_norm / duration_s if duration_s > 0 else math.nan

		session_rows.append(
			{
				"session_key": session_key,
				"condition": row.get("condition", ""),
				"phase": str(row.get("analysis_phase", "")).lower(),
				"raw_phase": str(row.get("phase", "")).lower(),
				"phase_detail": row.get("phase_detail", row.get("analysis_phase", "")),
				"hab_type": row.get("hab_type", ""),
				"subject_id": row.get("subject_id", ""),
				"sex": row.get("sex", ""),
				"trial_id": row.get("trial_id", pd.NA),
				"session_n": row.get("session_n", pd.NA),
				"layout_raw": row.get("layout_raw", ""),
				"keypoint": args.keypoint,
				"coord_mode": args.coord_mode,
				"fps": args.fps,
				"valid_frames": valid_frames,
				"valid_steps": valid_steps,
				"duration_s": duration_s,
				"duration_min": duration_min,
				"total_distance_px": total_distance_px,
				"total_distance_norm": total_distance_norm,
				"distance_per_min_px": distance_per_min_px,
				"distance_per_min_norm": distance_per_min_norm,
				"mean_speed_px_per_s": mean_speed_px_per_s,
				"mean_speed_norm_per_s": mean_speed_norm_per_s,
				"left_zone_frames": left_zone_frames,
				"center_zone_frames": center_zone_frames,
				"right_zone_frames": right_zone_frames,
				"left_zone_time_s": left_zone_time_s,
				"center_zone_time_s": center_zone_time_s,
				"right_zone_time_s": right_zone_time_s,
				"left_zone_pct": left_zone_pct,
				"center_zone_pct": center_zone_pct,
				"right_zone_pct": right_zone_pct,
				"spatial_preference_index": spatial_preference_index,
			}
		)

	session_df = pd.DataFrame(session_rows)
	sample_df = _build_sample_df(session_df, args.sample_unit) if not session_df.empty else session_df.copy()
	issue_df = pd.DataFrame(issue_rows)
	if session_df.empty:
		print("No valid sessions were available after QC.")
		issue_path = output_dir / f"locomotion_issues__{args.keypoint}{file_suffix}.csv"
		issue_df.to_csv(issue_path, index=False)
		print(f"Wrote {issue_path.relative_to(ROOT).as_posix()}")
		return 0

	group_rows: list[dict[str, object]] = []
	sex_group_rows: list[dict[str, object]] = []
	for (phase, condition), group in sample_df.groupby(["phase", "condition"], sort=False):
		group_rows.append(
			{
				"phase": phase,
				"condition": condition,
				"sample_unit": args.sample_unit,
				"sample_count": len(group),
				"total_distance_norm": float(pd.to_numeric(group["total_distance_norm"], errors="coerce").mean()),
				"total_distance_norm_sem": _sem(group["total_distance_norm"]),
				"total_distance_px": float(pd.to_numeric(group["total_distance_px"], errors="coerce").mean()),
				"total_distance_px_sem": _sem(group["total_distance_px"]),
				"distance_per_min_norm": float(pd.to_numeric(group["distance_per_min_norm"], errors="coerce").mean()),
				"distance_per_min_norm_sem": _sem(group["distance_per_min_norm"]),
				"distance_per_min_px": float(pd.to_numeric(group["distance_per_min_px"], errors="coerce").mean()),
				"distance_per_min_px_sem": _sem(group["distance_per_min_px"]),
				"mean_speed_norm_per_s": float(pd.to_numeric(group["mean_speed_norm_per_s"], errors="coerce").mean()),
				"mean_speed_norm_per_s_sem": _sem(group["mean_speed_norm_per_s"]),
				"left_zone_time_s": float(pd.to_numeric(group["left_zone_time_s"], errors="coerce").mean()),
				"left_zone_time_s_sem": _sem(group["left_zone_time_s"]),
				"center_zone_time_s": float(pd.to_numeric(group["center_zone_time_s"], errors="coerce").mean()),
				"center_zone_time_s_sem": _sem(group["center_zone_time_s"]),
				"right_zone_time_s": float(pd.to_numeric(group["right_zone_time_s"], errors="coerce").mean()),
				"right_zone_time_s_sem": _sem(group["right_zone_time_s"]),
				"spatial_preference_index": float(pd.to_numeric(group["spatial_preference_index"], errors="coerce").mean()),
				"spatial_preference_index_sem": _sem(group["spatial_preference_index"]),
			}
		)
	if "sex" in sample_df.columns:
		for (phase, condition, sex), group in sample_df.groupby(["phase", "condition", "sex"], sort=False):
			sex_group_rows.append(
				{
					"phase": phase,
					"condition": condition,
					"sex": sex,
					"sample_unit": args.sample_unit,
					"sample_count": len(group),
					"total_distance_norm": float(pd.to_numeric(group["total_distance_norm"], errors="coerce").mean()),
					"total_distance_norm_sem": _sem(group["total_distance_norm"]),
					"total_distance_px": float(pd.to_numeric(group["total_distance_px"], errors="coerce").mean()),
					"total_distance_px_sem": _sem(group["total_distance_px"]),
					"distance_per_min_norm": float(pd.to_numeric(group["distance_per_min_norm"], errors="coerce").mean()),
					"distance_per_min_norm_sem": _sem(group["distance_per_min_norm"]),
					"distance_per_min_px": float(pd.to_numeric(group["distance_per_min_px"], errors="coerce").mean()),
					"distance_per_min_px_sem": _sem(group["distance_per_min_px"]),
					"mean_speed_norm_per_s": float(pd.to_numeric(group["mean_speed_norm_per_s"], errors="coerce").mean()),
					"mean_speed_norm_per_s_sem": _sem(group["mean_speed_norm_per_s"]),
					"left_zone_time_s": float(pd.to_numeric(group["left_zone_time_s"], errors="coerce").mean()),
					"left_zone_time_s_sem": _sem(group["left_zone_time_s"]),
					"center_zone_time_s": float(pd.to_numeric(group["center_zone_time_s"], errors="coerce").mean()),
					"center_zone_time_s_sem": _sem(group["center_zone_time_s"]),
					"right_zone_time_s": float(pd.to_numeric(group["right_zone_time_s"], errors="coerce").mean()),
					"right_zone_time_s_sem": _sem(group["right_zone_time_s"]),
					"spatial_preference_index": float(pd.to_numeric(group["spatial_preference_index"], errors="coerce").mean()),
					"spatial_preference_index_sem": _sem(group["spatial_preference_index"]),
				}
			)

	group_df = pd.DataFrame(group_rows).sort_values(
		["phase", "condition"],
		key=lambda s: s.map(lambda x: _phase_sort_key(str(x))[0] if s.name == "phase" else _condition_sort_key(str(x))[0]),
	)
	sex_group_df = pd.DataFrame(sex_group_rows)

	total_phase_list = [phase for phase in ("hab_open", "nov") if phase in phase_list]
	total_plot_path = output_dir / f"locomotion_total_distance__{args.keypoint}{file_suffix}.png"
	total_plot_path, total_stats_rows = _plot_metric(
		sample_df,
		phase_list=total_phase_list,
		metric_col="total_distance_norm",
		ylabel="Total Distance",
		title="Movement Distance",
		output_path=total_plot_path,
		minimal_axes=True,
	)
	total_sex_plot_path = output_dir / f"locomotion_total_distance_by_sex__{args.keypoint}{file_suffix}.png"
	total_sex_plot_path, total_sex_stats_rows = _plot_metric_by_sex(
		sample_df,
		phase_list=total_phase_list,
		metric_col="total_distance_norm",
		ylabel="Total distance traveled (arena-normalized units)",
		title=f"{args.keypoint} locomotion",
		output_path=total_sex_plot_path,
	)

	per_min_plot_path = output_dir / f"locomotion_distance_per_min__{args.keypoint}{file_suffix}.png"
	per_min_plot_path, per_min_stats_rows = _plot_metric(
		sample_df,
		phase_list=phase_list,
		metric_col="distance_per_min_norm",
		ylabel="Distance traveled per min (arena-normalized units)",
		title=f"{args.keypoint} locomotion per minute",
		output_path=per_min_plot_path,
	)
	per_min_sex_plot_path = output_dir / f"locomotion_distance_per_min_by_sex__{args.keypoint}{file_suffix}.png"
	per_min_sex_plot_path, per_min_sex_stats_rows = _plot_metric_by_sex(
		sample_df,
		phase_list=phase_list,
		metric_col="distance_per_min_norm",
		ylabel="Distance traveled per min (arena-normalized units)",
		title=f"{args.keypoint} locomotion per minute",
		output_path=per_min_sex_plot_path,
	)

	open_spatial_path = output_dir / f"open_spatial_preference__{args.keypoint}{file_suffix}.png"
	open_spatial_plot_path, open_spatial_stats_rows = _plot_open_spatial_preference(
		sample_df,
		output_path=open_spatial_path,
	)
	open_spatial_sex_path = output_dir / f"open_spatial_preference_by_sex__{args.keypoint}{file_suffix}.png"
	open_spatial_sex_plot_path, open_spatial_sex_stats_rows = _plot_open_spatial_preference_by_sex(
		sample_df,
		output_path=open_spatial_sex_path,
	)
	spatial_class_df, spatial_class_group_df, spatial_class_stats_df = (
		_analyze_open_spatial_preference_strength(sample_df)
	)
	stats_df = pd.DataFrame(total_stats_rows + per_min_stats_rows + open_spatial_stats_rows)
	sex_stats_df = pd.DataFrame(total_sex_stats_rows + per_min_sex_stats_rows + open_spatial_sex_stats_rows)

	session_path = output_dir / f"locomotion_session_summary__{args.keypoint}{file_suffix}.csv"
	sample_path = output_dir / f"locomotion_sample_summary__{args.keypoint}{file_suffix}.csv"
	group_path = output_dir / f"locomotion_group_summary__{args.keypoint}{file_suffix}.csv"
	sex_group_path = output_dir / f"locomotion_sex_group_summary__{args.keypoint}{file_suffix}.csv"
	stats_path = output_dir / f"locomotion_stats__{args.keypoint}{file_suffix}.csv"
	sex_stats_path = output_dir / f"locomotion_sex_stats__{args.keypoint}{file_suffix}.csv"
	issue_path = output_dir / f"locomotion_issues__{args.keypoint}{file_suffix}.csv"
	spatial_class_path = output_dir / f"open_spatial_preference_classification__{args.keypoint}{file_suffix}.csv"
	spatial_class_group_path = output_dir / f"open_spatial_preference_classification_group__{args.keypoint}{file_suffix}.csv"
	spatial_class_stats_path = output_dir / f"open_spatial_preference_classification_stats__{args.keypoint}{file_suffix}.csv"
	session_df.to_csv(session_path, index=False)
	sample_df.to_csv(sample_path, index=False)
	group_df.to_csv(group_path, index=False)
	sex_group_df.to_csv(sex_group_path, index=False)
	stats_df.to_csv(stats_path, index=False)
	sex_stats_df.to_csv(sex_stats_path, index=False)
	issue_df.to_csv(issue_path, index=False)
	spatial_class_df.to_csv(spatial_class_path, index=False)
	spatial_class_group_df.to_csv(spatial_class_group_path, index=False)
	spatial_class_stats_df.to_csv(spatial_class_stats_path, index=False)

	print(f"Keypoint: {args.keypoint}")
	print(f"Phases summarized: {', '.join(phase_list)}")
	print(f"Coordinate mode: {args.coord_mode}")
	print(f"Sample unit: {args.sample_unit}")
	print(f"Sessions summarized: {len(session_df)}")
	print(f"Samples plotted: {len(sample_df)}")
	if session_n_filter:
		print(f"Session n filter: {', '.join(session_n_filter)}")
	print(f"Sessions with issues: {len(issue_df)}")
	print(f"Wrote {total_plot_path.relative_to(ROOT).as_posix()}")
	if total_sex_plot_path is not None:
		print(f"Wrote {total_sex_plot_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {per_min_plot_path.relative_to(ROOT).as_posix()}")
	if per_min_sex_plot_path is not None:
		print(f"Wrote {per_min_sex_plot_path.relative_to(ROOT).as_posix()}")
	if open_spatial_plot_path is not None:
		print(f"Wrote {open_spatial_plot_path.relative_to(ROOT).as_posix()}")
	if open_spatial_sex_plot_path is not None:
		print(f"Wrote {open_spatial_sex_plot_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {session_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {sample_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {group_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {sex_group_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {stats_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {sex_stats_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {issue_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {spatial_class_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {spatial_class_group_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {spatial_class_stats_path.relative_to(ROOT).as_posix()}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
