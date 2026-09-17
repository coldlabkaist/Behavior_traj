from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.core.three_chamber.side_occupancy import _build_sample_df
from analysis.core.three_chamber.side_occupancy import _count_visits
from analysis.core.three_chamber.side_occupancy import _normalize_session_n
from analysis.core.three_chamber.side_occupancy import _parse_phase_list
from analysis.core.three_chamber.side_occupancy import _parse_session_n_filter
from analysis.plots.three_chamber.side_occupancy import _plot_phase_bars
from analysis.core.three_chamber.side_occupancy import _point_in_polygon_mask
from analysis.core.three_chamber.side_occupancy import _sem
from analysis.core.three_chamber.side_occupancy import _to_abs_path

from analysis.core.three_chamber.side_occupancy import PHASE_CONFIG

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Plot 3-chamber side occupancy preference using wall-normalized side chambers.")
	parser.add_argument("--preprocess-summary", default="output/3chamber/preprocessed/preprocess_summary.csv")
	parser.add_argument("--manifest", default="output/3chamber/manifest/session_manifest.csv")
	parser.add_argument("--roi-vertices", default="output/3chamber/roi/roi_vertices.csv")
	parser.add_argument("--output-dir", default="output/3chamber/side_occupancy")
	parser.add_argument("--keypoint", default="Body_C")
	parser.add_argument("--phases", default="soc,nov")
	parser.add_argument("--fps", type=float, default=30.0)
	parser.add_argument(
		"--side-width",
		type=float,
		default=1.0 / 3.0,
		help="Width of each side chamber as a fraction of wall-normalized x range.",
	)
	parser.add_argument(
		"--session-n",
		default="",
		help="Optional comma-separated session_n filter, e.g. 1, 2, or 1,2.",
	)
	parser.add_argument(
		"--sample-unit",
		default="trial_mean",
		choices=["trial_mean", "session"],
		help="trial_mean averages n1/n2 within subject_id + trial_id before plotting.",
	)
	args = parser.parse_args(argv)

	preprocess_path = _to_abs_path(args.preprocess_summary)
	manifest_path = _to_abs_path(args.manifest)
	roi_vertices_path = _to_abs_path(args.roi_vertices)
	output_dir = _to_abs_path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	for path in (preprocess_path, manifest_path, roi_vertices_path):
		if not path.exists():
			raise FileNotFoundError(f"Required input not found: {path}")

	phase_list = _parse_phase_list(args.phases)
	session_n_filter = _parse_session_n_filter(args.session_n)
	file_suffix = f"__n{'_'.join(session_n_filter)}" if session_n_filter else ""

	preprocess_df = pd.read_csv(preprocess_path)
	manifest_df = pd.read_csv(manifest_path)
	roi_vertices_df = pd.read_csv(roi_vertices_path)

	sessions_df = preprocess_df.merge(
		manifest_df[
			[
				"session_key",
				"condition",
				"phase",
				"subject_id",
				"sex",
				"trial_id",
				"session_n",
				"layout_raw",
				"social_side",
				"empty_side",
				"familiar_side",
				"novel_side",
			]
		].drop_duplicates("session_key"),
		on=["session_key", "condition", "phase", "subject_id", "sex", "trial_id", "session_n", "layout_raw", "social_side", "empty_side", "familiar_side", "novel_side"],
		how="inner",
	)
	sessions_df = sessions_df[sessions_df["phase"].isin(phase_list)].copy()
	if session_n_filter:
		sessions_df["session_n_norm"] = sessions_df["session_n"].map(_normalize_session_n)
		sessions_df = sessions_df[sessions_df["session_n_norm"].isin(session_n_filter)].copy()

	xcol = f"{args.keypoint}.x"
	ycol = f"{args.keypoint}.y"
	session_rows: list[dict[str, object]] = []
	issue_rows: list[dict[str, object]] = []

	for row in sessions_df.to_dict(orient="records"):
		session_key = str(row["session_key"])
		phase = str(row.get("phase", "")).lower()
		output_path = _to_abs_path(str(row.get("output_path", "")))
		if not output_path.exists():
			issue_rows.append({"session_key": session_key, "issue": "preprocessed_file_not_found", "detail": str(output_path)})
			continue
		wall_vertices = roi_vertices_df[(roi_vertices_df["session_key"] == session_key) & (roi_vertices_df["roi_id"] == "wall")].sort_values("vertex_idx")
		if wall_vertices.empty:
			issue_rows.append({"session_key": session_key, "issue": "missing_wall_vertices", "detail": ""})
			continue
		wall_poly = wall_vertices[["x_norm", "y_norm"]].astype(float).to_numpy()
		df = pd.read_csv(output_path)
		if xcol not in df.columns or ycol not in df.columns:
			issue_rows.append({"session_key": session_key, "issue": "missing_keypoint_columns", "detail": args.keypoint})
			continue
		x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
		y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
		valid = np.isfinite(x) & np.isfinite(y)
		valid &= _point_in_polygon_mask(x, y, wall_poly)
		valid_frames = int(valid.sum())
		if valid_frames < 1:
			issue_rows.append({"session_key": session_key, "issue": "no_valid_points_inside_wall", "detail": args.keypoint})
			continue

		min_x = float(np.min(wall_poly[:, 0]))
		max_x = float(np.max(wall_poly[:, 0]))
		width = max(max_x - min_x, 1e-9)
		x_norm = (x - min_x) / width
		left_mask = valid & (x_norm < float(args.side_width))
		right_mask = valid & (x_norm > (1.0 - float(args.side_width)))
		left_frames = int(left_mask.sum())
		right_frames = int(right_mask.sum())
		left_time_s = left_frames / max(float(args.fps), 1e-9)
		right_time_s = right_frames / max(float(args.fps), 1e-9)
		left_visit_count = _count_visits(left_mask)
		right_visit_count = _count_visits(right_mask)

		side_time_map = {"l": left_time_s, "r": right_time_s}
		side_visit_map = {"l": left_visit_count, "r": right_visit_count}
		empty_side_time_s = math.nan
		social_side_time_s = math.nan
		empty_side_visit_count = math.nan
		social_side_visit_count = math.nan
		social_side_preference_index = math.nan
		familiar_side_time_s = math.nan
		novel_side_time_s = math.nan
		familiar_side_visit_count = math.nan
		novel_side_visit_count = math.nan
		novel_side_preference_index = math.nan

		if phase == "soc":
			social_side = str(row.get("social_side", "")).lower()
			empty_side = str(row.get("empty_side", "")).lower()
			social_side_time_s = side_time_map.get(social_side, math.nan)
			empty_side_time_s = side_time_map.get(empty_side, math.nan)
			social_side_visit_count = side_visit_map.get(social_side, math.nan)
			empty_side_visit_count = side_visit_map.get(empty_side, math.nan)
			den = social_side_time_s + empty_side_time_s
			social_side_preference_index = ((social_side_time_s - empty_side_time_s) / den) if np.isfinite(den) and den > 0 else math.nan
		elif phase == "nov":
			familiar_side = str(row.get("familiar_side", "")).lower()
			novel_side = str(row.get("novel_side", "")).lower()
			familiar_side_time_s = side_time_map.get(familiar_side, math.nan)
			novel_side_time_s = side_time_map.get(novel_side, math.nan)
			familiar_side_visit_count = side_visit_map.get(familiar_side, math.nan)
			novel_side_visit_count = side_visit_map.get(novel_side, math.nan)
			den = novel_side_time_s + familiar_side_time_s
			novel_side_preference_index = ((novel_side_time_s - familiar_side_time_s) / den) if np.isfinite(den) and den > 0 else math.nan

		session_rows.append(
			{
				"session_key": session_key,
				"condition": row.get("condition", ""),
				"phase": phase,
				"subject_id": row.get("subject_id", ""),
				"sex": row.get("sex", ""),
				"trial_id": row.get("trial_id", pd.NA),
				"session_n": row.get("session_n", pd.NA),
				"layout_raw": row.get("layout_raw", ""),
				"keypoint": args.keypoint,
				"fps": args.fps,
				"side_width": args.side_width,
				"valid_frames": valid_frames,
				"left_side_frames": left_frames,
				"right_side_frames": right_frames,
				"left_side_time_s": left_time_s,
				"right_side_time_s": right_time_s,
				"left_side_visit_count": left_visit_count,
				"right_side_visit_count": right_visit_count,
				"empty_side_time_s": empty_side_time_s,
				"social_side_time_s": social_side_time_s,
				"empty_side_visit_count": empty_side_visit_count,
				"social_side_visit_count": social_side_visit_count,
				"social_side_preference_index": social_side_preference_index,
				"familiar_side_time_s": familiar_side_time_s,
				"novel_side_time_s": novel_side_time_s,
				"familiar_side_visit_count": familiar_side_visit_count,
				"novel_side_visit_count": novel_side_visit_count,
				"novel_side_preference_index": novel_side_preference_index,
			}
		)

	session_df = pd.DataFrame(session_rows)
	sample_df = _build_sample_df(session_df, args.sample_unit) if not session_df.empty else session_df.copy()
	issue_df = pd.DataFrame(issue_rows)
	stats_rows: list[dict[str, object]] = []
	group_rows: list[dict[str, object]] = []
	figure_paths: list[Path] = []

	for phase in phase_list:
		phase_df = sample_df[sample_df["phase"] == phase].copy()
		if phase_df.empty:
			continue
		figure_path, phase_stats = _plot_phase_bars(
			phase_df,
			phase=phase,
			output_dir=output_dir,
			keypoint=args.keypoint,
			file_suffix=file_suffix,
		)
		figure_paths.append(figure_path)
		stats_rows.extend(phase_stats)
		cfg = PHASE_CONFIG[phase]
		for condition, group in phase_df.groupby("condition", sort=False):
			group_rows.append(
				{
					"phase": phase,
					"condition": condition,
					"sample_unit": args.sample_unit,
					"sample_count": len(group),
					cfg["time_left_col"]: float(pd.to_numeric(group[cfg["time_left_col"]], errors="coerce").mean()),
					f"{cfg['time_left_col']}_sem": _sem(group[cfg["time_left_col"]]),
					cfg["time_right_col"]: float(pd.to_numeric(group[cfg["time_right_col"]], errors="coerce").mean()),
					f"{cfg['time_right_col']}_sem": _sem(group[cfg["time_right_col"]]),
					cfg["visit_left_col"]: float(pd.to_numeric(group[cfg["visit_left_col"]], errors="coerce").mean()),
					f"{cfg['visit_left_col']}_sem": _sem(group[cfg["visit_left_col"]]),
					cfg["visit_right_col"]: float(pd.to_numeric(group[cfg["visit_right_col"]], errors="coerce").mean()),
					f"{cfg['visit_right_col']}_sem": _sem(group[cfg["visit_right_col"]]),
					cfg["pref_col"]: float(pd.to_numeric(group[cfg["pref_col"]], errors="coerce").mean()),
					f"{cfg['pref_col']}_sem": _sem(group[cfg["pref_col"]]),
				}
			)

	stats_df = pd.DataFrame(stats_rows)
	group_df = pd.DataFrame(group_rows)
	session_path = output_dir / f"side_occupancy_session_summary__{args.keypoint}{file_suffix}.csv"
	sample_path = output_dir / f"side_occupancy_sample_summary__{args.keypoint}{file_suffix}.csv"
	group_path = output_dir / f"side_occupancy_group_summary__{args.keypoint}{file_suffix}.csv"
	stats_path = output_dir / f"side_occupancy_stats__{args.keypoint}{file_suffix}.csv"
	issue_path = output_dir / f"side_occupancy_issues__{args.keypoint}{file_suffix}.csv"
	session_df.to_csv(session_path, index=False)
	sample_df.to_csv(sample_path, index=False)
	group_df.to_csv(group_path, index=False)
	stats_df.to_csv(stats_path, index=False)
	issue_df.to_csv(issue_path, index=False)

	print(f"Keypoint: {args.keypoint}")
	print(f"Side width: {args.side_width:.3f}")
	print(f"Sample unit: {args.sample_unit}")
	if session_n_filter:
		print(f"Session n filter: {', '.join(session_n_filter)}")
	print(f"Sessions summarized: {len(session_df)}")
	print(f"Samples plotted: {len(sample_df)}")
	print(f"Sessions with issues: {len(issue_df)}")
	for path in figure_paths:
		print(f"Wrote {path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {session_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {sample_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {group_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {stats_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {issue_path.relative_to(ROOT).as_posix()}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
