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
from analysis.core.three_chamber.contact_events import close_short_gaps, count_visits, filter_short_bouts
from analysis.core.three_chamber.contact_overrides import AUTO, apply_overrides, load_overrides, override_arrays
from analysis.core.three_chamber.homography import apply_homography, wall_homography
from analysis.core.three_chamber.preference_bars import _buffer_polygon
from analysis.core.three_chamber.preference_bars import _build_sample_df
from analysis.core.three_chamber.preference_bars import _compute_circle_mask
from analysis.core.three_chamber.preference_bars import _compute_shared_panel_ylims
from analysis.core.three_chamber.preference_bars import _count_visits
from analysis.core.three_chamber.preference_bars import _filter_first_minutes
from analysis.core.three_chamber.preference_bars import _filter_short_bouts
from analysis.core.three_chamber.preference_bars import _format_minutes_suffix
from analysis.core.three_chamber.preference_bars import _holm_adjust
from analysis.core.three_chamber.preference_bars import _infer_image_size
from analysis.core.three_chamber.preference_bars import _normalize_session_n
from analysis.core.three_chamber.preference_bars import _parse_phase_list
from analysis.core.three_chamber.preference_bars import _parse_session_n_filter
from analysis.plots.three_chamber.preference_bars import _plot_phase_bars
from analysis.plots.three_chamber.preference_bars import _plot_phase_bars_by_sex
from analysis.core.three_chamber.preference_bars import _point_in_polygon_mask
from analysis.core.three_chamber.preference_bars import _roi_output_tag
from analysis.core.three_chamber.preference_bars import _sem
from analysis.core.three_chamber.preference_bars import _to_abs_path

from analysis.core.three_chamber.preference_bars import PHASE_CONFIG

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Plot 3-chamber investigation bar plots and preference indices.")
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
		"--roi-summary",
		default="output/3chamber/roi/roi_summary.csv",
		help="ROI summary CSV produced by build_roi_geometry.py",
	)
	parser.add_argument(
		"--roi-vertices",
		default="output/3chamber/roi/roi_vertices.csv",
		help="ROI vertices CSV produced by build_roi_geometry.py",
	)
	parser.add_argument(
		"--output-dir",
		default="output/3chamber/barplots",
		help="Directory to write bar plot figures and summary CSVs",
	)
	parser.add_argument("--keypoint", default="Nose", help="Keypoint to use for ROI investigation, e.g. Nose or Body_C")
	parser.add_argument("--phases", default="soc,nov", help="Comma-separated phase list to plot")
	parser.add_argument(
		"--roi-mode",
		default="contact",
		choices=["contact", "circle", "polygon"],
		help="Investigation ROI definition. contact buffers the pinned cup polygon; circle uses centroid and radius; polygon uses pins only.",
	)
	parser.add_argument("--radius-scale", type=float, default=1.2, help="ROI scale. contact uses (scale - 1) as the polygon buffer fraction.")
	parser.add_argument(
		"--coord-mode",
		default="normalized",
		choices=["normalized", "wall_bbox", "image", "homography"],
		help=(
			"Coordinate mode. normalized uses x_norm/y_norm and session-specific normalized cup ROIs. "
			"wall_bbox restores coordinates to raw pixels and uses raw-pixel cup ROIs. "
			"image is an alias for wall_bbox; homography requires calibrated floor-plane wall corners."
		),
	)
	parser.add_argument(
		"--session-n",
		default="",
		help="Optional session number filter, e.g. 1, n_1, or 1,2",
	)
	parser.add_argument("--fps", type=float, default=30.0, help="Frames per second for converting ROI frames to seconds")
	parser.add_argument(
		"--sample-unit",
		default="trial_mean",
		choices=["trial_mean", "session"],
		help="Sample unit for plotting/statistics. trial_mean averages n1/n2 within subject_id + trial_id before plotting.",
	)
	parser.add_argument(
		"--max-minutes",
		type=float,
		default=0.0,
		help="Analyze only the first N minutes of each recording. Default 0 uses the full session.",
	)
	parser.add_argument(
		"--max-gap-seconds",
		type=float,
		default=0.5,
		help="Fill bounded FALSE gaps up to this duration before minimum-bout filtering. Default: 0.5 s.",
	)
	parser.add_argument(
		"--min-bout-seconds",
		type=float,
		default=0.0,
		help="Remove continuous ROI entries shorter than this duration. Default 0 keeps all entries.",
	)
	parser.add_argument(
		"--exclude-body-in-cup",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Exclude Nose contact frames when Body_C is inside either pinned cup polygon (default: enabled).",
	)
	parser.add_argument(
		"--contact-overrides",
		default="output/3chamber/contact_review/contact_overrides.csv",
		help="Manual frame-range corrections saved by review_contacts_gui.py.",
	)
	parser.add_argument(
		"--use-contact-overrides",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Apply saved manual contact corrections when present (default: enabled).",
	)
	args = parser.parse_args(argv)
	if float(args.max_gap_seconds) < 0:
		parser.error("--max-gap-seconds must be >= 0")
	if float(args.min_bout_seconds) < 0:
		parser.error("--min-bout-seconds must be >= 0")
	if args.roi_mode == "contact":
		if float(args.radius_scale) < 1.0:
			parser.error("contact ROI requires --radius-scale >= 1.0")
		args.coord_mode = "wall_bbox"

	manifest_path = _to_abs_path(args.manifest)
	preprocess_path = _to_abs_path(args.preprocess_summary)
	roi_summary_path = _to_abs_path(args.roi_summary)
	roi_vertices_path = _to_abs_path(args.roi_vertices)
	output_dir = _to_abs_path(args.output_dir)
	override_path = _to_abs_path(args.contact_overrides)
	output_dir.mkdir(parents=True, exist_ok=True)

	phase_list = _parse_phase_list(args.phases)
	session_n_filter = _parse_session_n_filter(args.session_n)
	file_suffix = "" if args.roi_mode == "contact" or args.coord_mode == "normalized" else ("__raw" if args.coord_mode in {"wall_bbox", "image"} else "__homography")
	file_suffix += f"__n{'_'.join(session_n_filter)}" if session_n_filter else ""
	file_suffix += _format_minutes_suffix(float(args.max_minutes))
	file_suffix += f"__bout{float(args.min_bout_seconds):g}s" if float(args.min_bout_seconds) > 0 else ""
	file_suffix += f"__gap{float(args.max_gap_seconds):g}s" if float(args.max_gap_seconds) > 0 else ""
	for path in (manifest_path, preprocess_path, roi_summary_path, roi_vertices_path):
		if not path.exists():
			raise FileNotFoundError(f"Required input not found: {path}")

	manifest_df = pd.read_csv(manifest_path)
	preprocess_df = pd.read_csv(preprocess_path)
	roi_summary_df = pd.read_csv(roi_summary_path)
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
	if sessions_df.empty:
		print("No sessions matched the requested phases.")
		return 0
	override_table = load_overrides(override_path) if bool(args.use_contact_overrides) else pd.DataFrame()
	if not override_table.empty:
		relevant_session_keys = set(sessions_df["session_key"].astype(str))
		relevant_overrides = override_table[override_table["session_key"].astype(str).isin(relevant_session_keys)].copy()
	else:
		relevant_overrides = override_table.copy()
	if not relevant_overrides.empty:
		file_suffix += "__reviewed"

	xcol = f"{args.keypoint}.x"
	ycol = f"{args.keypoint}.y"
	exclude_body_in_cup = bool(args.exclude_body_in_cup) and args.roi_mode == "contact" and args.keypoint.lower() != "body_c"
	session_rows: list[dict[str, object]] = []
	issue_rows: list[dict[str, object]] = []

	for row in sessions_df.to_dict(orient="records"):
		session_key = str(row["session_key"])
		phase = str(row["phase"]).lower()
		preprocessed_csv = _to_abs_path(str(row["output_path"]))
		if not preprocessed_csv.exists():
			issue_rows.append({"session_key": session_key, "issue": "preprocessed_file_not_found", "detail": str(preprocessed_csv)})
			continue

		df = pd.read_csv(preprocessed_csv)
		df = _filter_first_minutes(df, max_minutes=float(args.max_minutes), fps=float(args.fps))
		if xcol not in df.columns or ycol not in df.columns:
			issue_rows.append({"session_key": session_key, "issue": "missing_keypoint_columns", "detail": args.keypoint})
			continue

		x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
		y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
		body_x: np.ndarray | None = None
		body_y: np.ndarray | None = None
		if exclude_body_in_cup:
			if "Body_C.x" not in df.columns or "Body_C.y" not in df.columns:
				issue_rows.append({"session_key": session_key, "issue": "missing_body_center_columns", "detail": "Body_C"})
				continue
			body_x = pd.to_numeric(df["Body_C.x"], errors="coerce").to_numpy(dtype=float)
			body_y = pd.to_numeric(df["Body_C.y"], errors="coerce").to_numpy(dtype=float)

		session_vertices = roi_vertices_df[roi_vertices_df["session_key"] == session_key].copy()
		wall_vertices = session_vertices[session_vertices["roi_id"] == "wall"].sort_values("vertex_idx")
		if wall_vertices.empty:
			issue_rows.append({"session_key": session_key, "issue": "missing_wall_vertices", "detail": ""})
			continue

		use_wall_bbox = args.coord_mode in {"wall_bbox", "image"}
		if use_wall_bbox:
			try:
				image_width, image_height = _infer_image_size(session_vertices)
			except ValueError as exc:
				issue_rows.append({"session_key": session_key, "issue": "invalid_image_size", "detail": str(exc)})
				continue
			x = x * image_width
			y = y * image_height
			if body_x is not None and body_y is not None:
				body_x = body_x * image_width
				body_y = body_y * image_height
			wall_poly = wall_vertices[["x", "y"]].astype(float).to_numpy()
		else:
			wall_poly = wall_vertices[["x_norm", "y_norm"]].astype(float).to_numpy()

		homography = None
		if args.coord_mode == "homography":
			try:
				homography = wall_homography(wall_poly)
			except ValueError as exc:
				issue_rows.append({"session_key": session_key, "issue": "invalid_wall_homography", "detail": str(exc)})
				continue
			points = apply_homography(np.column_stack([x, y]), homography)
			x = points[:, 0]
			y = points[:, 1]
			wall_poly = apply_homography(wall_poly, homography)

		valid = np.isfinite(x) & np.isfinite(y)
		valid &= _point_in_polygon_mask(x, y, wall_poly)
		if not bool(np.any(valid)):
			issue_rows.append({"session_key": session_key, "issue": "no_valid_points_inside_wall", "detail": args.keypoint})
			continue

		body_in_any_cup = np.zeros(len(df), dtype=bool)
		if body_x is not None and body_y is not None:
			body_finite = np.isfinite(body_x) & np.isfinite(body_y)
			for cup_roi_id in ("chamber_l", "chamber_r"):
				cup_vertices = session_vertices[session_vertices["roi_id"] == cup_roi_id].sort_values("vertex_idx")
				if cup_vertices.empty:
					continue
				poly_cols = ["x", "y"] if use_wall_bbox else ["x_norm", "y_norm"]
				cup_poly = cup_vertices[poly_cols].astype(float).to_numpy()
				body_in_any_cup |= body_finite & _point_in_polygon_mask(body_x, body_y, cup_poly)

		roi_masks: dict[str, np.ndarray] = {}
		body_in_cup_frames_removed: dict[str, int] = {"chamber_l": 0, "chamber_r": 0}
		nose_in_cup_frames_removed: dict[str, int] = {"chamber_l": 0, "chamber_r": 0}
		for roi_id in ("chamber_l", "chamber_r"):
			if args.roi_mode == "circle":
				if args.coord_mode == "homography":
					verts = roi_vertices_df[(roi_vertices_df["session_key"] == session_key) & (roi_vertices_df["roi_id"] == roi_id)].sort_values("vertex_idx")
					if verts.empty or homography is None:
						continue
					poly = apply_homography(verts[["x_norm", "y_norm"]].astype(float).to_numpy(), homography)
					poly = poly[np.isfinite(poly).all(axis=1)]
					if poly.shape[0] < 3:
						continue
					cx = float(np.mean(poly[:, 0]))
					cy = float(np.mean(poly[:, 1]))
					radius = float(np.mean(np.sqrt((poly[:, 0] - cx) ** 2 + (poly[:, 1] - cy) ** 2))) * float(args.radius_scale)
				elif args.coord_mode == "normalized":
					roi_row = roi_summary_df[(roi_summary_df["session_key"] == session_key) & (roi_summary_df["roi_id"] == roi_id)]
					if roi_row.empty:
						continue
					r0 = roi_row.iloc[0]
					cx = float(r0["centroid_x_norm"])
					cy = float(r0["centroid_y_norm"])
					radius = float(r0["mean_radius_norm"]) * float(args.radius_scale)
				else:
					roi_row = roi_summary_df[(roi_summary_df["session_key"] == session_key) & (roi_summary_df["roi_id"] == roi_id)]
					if roi_row.empty:
						continue
					r0 = roi_row.iloc[0]
					cx = float(r0["centroid_x"])
					cy = float(r0["centroid_y"])
					radius = float(r0["mean_radius_px"]) * float(args.radius_scale)
				mask = _compute_circle_mask(
					x,
					y,
					cx=cx,
					cy=cy,
					radius=radius,
				)
			elif args.roi_mode == "contact":
				verts = roi_vertices_df[(roi_vertices_df["session_key"] == session_key) & (roi_vertices_df["roi_id"] == roi_id)].sort_values("vertex_idx")
				roi_row = roi_summary_df[(roi_summary_df["session_key"] == session_key) & (roi_summary_df["roi_id"] == roi_id)]
				if verts.empty or roi_row.empty:
					continue
				poly = verts[["x", "y"]].astype(float).to_numpy()
				buffer_px = float(roi_row.iloc[0]["mean_radius_px"]) * (float(args.radius_scale) - 1.0)
				buffered_poly = _buffer_polygon(poly, buffer_px)
				if buffered_poly.shape[0] < 3:
					continue
				mask = _point_in_polygon_mask(x, y, buffered_poly)
				if body_x is not None and body_y is not None:
					body_in_cup_frames_removed[roi_id] = int(np.sum(valid & mask & body_in_any_cup))
					mask &= ~body_in_any_cup
			else:
				verts = roi_vertices_df[(roi_vertices_df["session_key"] == session_key) & (roi_vertices_df["roi_id"] == roi_id)].sort_values("vertex_idx")
				if verts.empty:
					continue
				poly_cols = ["x", "y"] if use_wall_bbox else ["x_norm", "y_norm"]
				poly = verts[poly_cols].astype(float).to_numpy()
				if args.coord_mode == "homography" and homography is not None:
					poly = apply_homography(poly, homography)
				mask = _point_in_polygon_mask(x, y, poly)
			roi_masks[roi_id] = valid & mask

		if "chamber_l" not in roi_masks or "chamber_r" not in roi_masks:
			issue_rows.append({"session_key": session_key, "issue": "missing_chamber_roi", "detail": args.roi_mode})
			continue

		max_gap_frames = int(math.ceil(float(args.max_gap_seconds) * float(args.fps))) if float(args.max_gap_seconds) > 0 else 0
		min_bout_frames = int(math.ceil(float(args.min_bout_seconds) * float(args.fps))) if float(args.min_bout_seconds) > 0 else 0
		left_raw_mask = roi_masks["chamber_l"]
		right_raw_mask = roi_masks["chamber_r"]
		left_raw_count = int(left_raw_mask.sum())
		right_raw_count = int(right_raw_mask.sum())
		gap_eligible = valid.copy()
		if exclude_body_in_cup:
			gap_eligible &= ~body_in_any_cup
		left_closed, left_gap_mask, _, _ = close_short_gaps(
			left_raw_mask,
			max_gap_frames,
			eligible=gap_eligible & ~right_raw_mask,
		)
		right_closed, right_gap_mask, _, _ = close_short_gaps(
			right_raw_mask,
			max_gap_frames,
			eligible=gap_eligible & ~left_raw_mask,
		)
		left_mask, left_short_frames_removed, left_short_bouts_removed = _filter_short_bouts(left_closed, min_bout_frames)
		right_mask, right_short_frames_removed, right_short_bouts_removed = _filter_short_bouts(right_closed, min_bout_frames)
		left_gap_mask &= left_mask
		right_gap_mask &= right_mask
		left_gap_frames_filled = int(left_gap_mask.sum())
		right_gap_frames_filled = int(right_gap_mask.sum())
		left_gaps_closed = _count_visits(left_gap_mask)
		right_gaps_closed = _count_visits(right_gap_mask)
		left_auto_mask = left_mask.copy()
		right_auto_mask = right_mask.copy()
		if "frame_idx" in df.columns:
			pose_frames_numeric = pd.to_numeric(df["frame_idx"], errors="coerce")
			if pose_frames_numeric.isna().any():
				issue_rows.append({"session_key": session_key, "issue": "invalid_frame_idx", "detail": "Manual overrides were not applied"})
				pose_frames = np.arange(len(df), dtype=int)
				left_override = np.full(len(df), AUTO, dtype=np.int8)
				right_override = np.full(len(df), AUTO, dtype=np.int8)
			else:
				pose_frames = np.rint(pose_frames_numeric.to_numpy(dtype=float)).astype(int)
				left_override, right_override = override_arrays(
					relevant_overrides,
					session_key=session_key,
					pose_frames=pose_frames,
				)
		else:
			pose_frames = np.arange(len(df), dtype=int)
			left_override, right_override = override_arrays(
				relevant_overrides,
				session_key=session_key,
				pose_frames=pose_frames,
			)
		left_mask, right_mask = apply_overrides(
			left_auto_mask,
			right_auto_mask,
			left_override,
			right_override,
		)
		manual_override_mask = (left_override != AUTO) | (right_override != AUTO)
		manual_override_frames = int(manual_override_mask.sum())
		left_manual_frames_added = int(np.sum(~left_auto_mask & left_mask))
		left_manual_frames_removed = int(np.sum(left_auto_mask & ~left_mask))
		right_manual_frames_added = int(np.sum(~right_auto_mask & right_mask))
		right_manual_frames_removed = int(np.sum(right_auto_mask & ~right_mask))
		roi_masks["chamber_l"] = left_mask
		roi_masks["chamber_r"] = right_mask
		valid_count = int(np.sum(valid | left_mask | right_mask))
		left_count = int(left_mask.sum())
		right_count = int(right_mask.sum())
		left_time_s = left_count / max(float(args.fps), 1e-9)
		right_time_s = right_count / max(float(args.fps), 1e-9)
		left_visit_count = _count_visits(roi_masks["chamber_l"])
		right_visit_count = _count_visits(roi_masks["chamber_r"])
		left_pct = 100.0 * left_count / valid_count
		right_pct = 100.0 * right_count / valid_count

		social_pct = math.nan
		empty_pct = math.nan
		social_time_s = math.nan
		empty_time_s = math.nan
		social_visit_count = math.nan
		empty_visit_count = math.nan
		social_within_roi_pct = math.nan
		empty_within_roi_pct = math.nan
		social_preference_index = math.nan
		familiar_pct = math.nan
		novel_pct = math.nan
		familiar_time_s = math.nan
		novel_time_s = math.nan
		familiar_visit_count = math.nan
		novel_visit_count = math.nan
		familiar_within_roi_pct = math.nan
		novel_within_roi_pct = math.nan
		novel_preference_index = math.nan

		if phase == "soc":
			social_side = str(row.get("social_side", "")).lower()
			empty_side = str(row.get("empty_side", "")).lower()
			side_map = {"l": left_pct, "r": right_pct}
			time_map = {"l": left_time_s, "r": right_time_s}
			visit_map = {"l": left_visit_count, "r": right_visit_count}
			social_pct = side_map.get(social_side, math.nan)
			empty_pct = side_map.get(empty_side, math.nan)
			social_time_s = time_map.get(social_side, math.nan)
			empty_time_s = time_map.get(empty_side, math.nan)
			social_visit_count = visit_map.get(social_side, math.nan)
			empty_visit_count = visit_map.get(empty_side, math.nan)
			den = social_pct + empty_pct
			if np.isfinite(den) and den > 0:
				social_within_roi_pct = 100.0 * social_pct / den
				empty_within_roi_pct = 100.0 * empty_pct / den
			time_den = social_time_s + empty_time_s
			social_preference_index = ((social_time_s - empty_time_s) / time_den) if np.isfinite(time_den) and time_den > 0 else math.nan
		elif phase == "nov":
			familiar_side = str(row.get("familiar_side", "")).lower()
			novel_side = str(row.get("novel_side", "")).lower()
			side_map = {"l": left_pct, "r": right_pct}
			time_map = {"l": left_time_s, "r": right_time_s}
			visit_map = {"l": left_visit_count, "r": right_visit_count}
			familiar_pct = side_map.get(familiar_side, math.nan)
			novel_pct = side_map.get(novel_side, math.nan)
			familiar_time_s = time_map.get(familiar_side, math.nan)
			novel_time_s = time_map.get(novel_side, math.nan)
			familiar_visit_count = visit_map.get(familiar_side, math.nan)
			novel_visit_count = visit_map.get(novel_side, math.nan)
			den = novel_pct + familiar_pct
			if np.isfinite(den) and den > 0:
				novel_within_roi_pct = 100.0 * novel_pct / den
				familiar_within_roi_pct = 100.0 * familiar_pct / den
			time_den = novel_time_s + familiar_time_s
			novel_preference_index = ((novel_time_s - familiar_time_s) / time_den) if np.isfinite(time_den) and time_den > 0 else math.nan

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
				"coord_mode": args.coord_mode,
				"roi_mode": args.roi_mode,
				"radius_scale": args.radius_scale,
				"contact_buffer_fraction": (float(args.radius_scale) - 1.0) if args.roi_mode == "contact" else math.nan,
				"fps": args.fps,
				"max_minutes": args.max_minutes if float(args.max_minutes) > 0 else math.nan,
				"max_gap_seconds": float(args.max_gap_seconds),
				"max_gap_frames": max_gap_frames,
				"min_bout_seconds": float(args.min_bout_seconds),
				"min_bout_frames": min_bout_frames,
				"exclude_body_in_cup": exclude_body_in_cup,
				"contact_overrides_applied": manual_override_frames > 0,
				"valid_frames": valid_count,
				"left_raw_frames": left_raw_count,
				"right_raw_frames": right_raw_count,
				"left_gap_frames_filled": left_gap_frames_filled,
				"right_gap_frames_filled": right_gap_frames_filled,
				"left_gaps_closed": left_gaps_closed,
				"right_gaps_closed": right_gaps_closed,
				"left_short_frames_removed": left_short_frames_removed,
				"right_short_frames_removed": right_short_frames_removed,
				"left_short_bouts_removed": left_short_bouts_removed,
				"right_short_bouts_removed": right_short_bouts_removed,
				"left_body_in_cup_frames_removed": body_in_cup_frames_removed["chamber_l"],
				"right_body_in_cup_frames_removed": body_in_cup_frames_removed["chamber_r"],
				"left_nose_in_cup_frames_removed": nose_in_cup_frames_removed["chamber_l"],
				"right_nose_in_cup_frames_removed": nose_in_cup_frames_removed["chamber_r"],
				"manual_override_frames": manual_override_frames,
				"left_manual_frames_added": left_manual_frames_added,
				"left_manual_frames_removed": left_manual_frames_removed,
				"right_manual_frames_added": right_manual_frames_added,
				"right_manual_frames_removed": right_manual_frames_removed,
				"left_frames": left_count,
				"right_frames": right_count,
				"left_pct": left_pct,
				"right_pct": right_pct,
				"left_time_s": left_time_s,
				"right_time_s": right_time_s,
				"left_visit_count": left_visit_count,
				"right_visit_count": right_visit_count,
				"empty_pct": empty_pct,
				"social_pct": social_pct,
				"empty_time_s": empty_time_s,
				"social_time_s": social_time_s,
				"empty_visit_count": empty_visit_count,
				"social_visit_count": social_visit_count,
				"empty_within_roi_pct": empty_within_roi_pct,
				"social_within_roi_pct": social_within_roi_pct,
				"social_preference_index": social_preference_index,
				"familiar_pct": familiar_pct,
				"novel_pct": novel_pct,
				"familiar_time_s": familiar_time_s,
				"novel_time_s": novel_time_s,
				"familiar_visit_count": familiar_visit_count,
				"novel_visit_count": novel_visit_count,
				"familiar_within_roi_pct": familiar_within_roi_pct,
				"novel_within_roi_pct": novel_within_roi_pct,
				"novel_preference_index": novel_preference_index,
			}
		)

	session_df = pd.DataFrame(session_rows)
	sample_df = _build_sample_df(session_df, args.sample_unit) if not session_df.empty else session_df.copy()
	issue_df = pd.DataFrame(issue_rows)
	shared_panel_ylims = _compute_shared_panel_ylims(sample_df, phase_list, sex_stratified=False) if not sample_df.empty else {}
	shared_sex_panel_ylims = _compute_shared_panel_ylims(sample_df, phase_list, sex_stratified=True) if not sample_df.empty else {}
	figure_paths: list[Path] = []
	group_rows: list[dict[str, object]] = []
	sex_group_rows: list[dict[str, object]] = []
	stats_rows: list[dict[str, object]] = []
	sex_stats_rows: list[dict[str, object]] = []

	for phase in phase_list:
		phase_df = sample_df[sample_df["phase"] == phase].copy()
		if phase_df.empty:
			continue
		figure_path, phase_stats_rows = _plot_phase_bars(
			phase_df,
			phase=phase,
			output_dir=output_dir,
			keypoint=args.keypoint,
			roi_mode=args.roi_mode,
			radius_scale=args.radius_scale,
			file_suffix=file_suffix,
			shared_panel_ylims=shared_panel_ylims,
		)
		figure_paths.append(figure_path)
		stats_rows.extend(phase_stats_rows)
		sex_figure_path, phase_sex_stats_rows = _plot_phase_bars_by_sex(
			phase_df,
			phase=phase,
			output_dir=output_dir,
			keypoint=args.keypoint,
			roi_mode=args.roi_mode,
			radius_scale=args.radius_scale,
			file_suffix=file_suffix,
			shared_panel_ylims=shared_sex_panel_ylims,
		)
		if sex_figure_path is not None:
			figure_paths.append(sex_figure_path)
		sex_stats_rows.extend(phase_sex_stats_rows)
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
		if "sex" in phase_df.columns:
			for (condition, sex), group in phase_df.groupby(["condition", "sex"], sort=False):
				sex_group_rows.append(
					{
						"phase": phase,
						"condition": condition,
						"sex": sex,
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

	group_df = pd.DataFrame(group_rows)
	sex_group_df = pd.DataFrame(sex_group_rows)
	stats_df = pd.DataFrame(stats_rows)
	if not stats_df.empty:
		stats_df["pvalue_holm"] = math.nan
		variance_mask = stats_df["panel"].astype(str) == "preference_index_variance"
		stats_df.loc[variance_mask, "pvalue_holm"] = _holm_adjust(stats_df.loc[variance_mask, "pvalue"])
	sex_stats_df = pd.DataFrame(sex_stats_rows)
	roi_tag = _roi_output_tag(args.roi_mode, float(args.radius_scale))
	session_path = output_dir / f"preference_session_summary__{args.keypoint}__{roi_tag}{file_suffix}.csv"
	sample_path = output_dir / f"preference_sample_summary__{args.keypoint}__{roi_tag}{file_suffix}.csv"
	group_path = output_dir / f"preference_group_summary__{args.keypoint}__{roi_tag}{file_suffix}.csv"
	sex_group_path = output_dir / f"preference_sex_group_summary__{args.keypoint}__{roi_tag}{file_suffix}.csv"
	stats_path = output_dir / f"preference_stats__{args.keypoint}__{roi_tag}{file_suffix}.csv"
	sex_stats_path = output_dir / f"preference_sex_stats__{args.keypoint}__{roi_tag}{file_suffix}.csv"
	issue_path = output_dir / f"preference_issues__{args.keypoint}__{roi_tag}{file_suffix}.csv"
	session_df.to_csv(session_path, index=False)
	sample_df.to_csv(sample_path, index=False)
	group_df.to_csv(group_path, index=False)
	sex_group_df.to_csv(sex_group_path, index=False)
	stats_df.to_csv(stats_path, index=False)
	sex_stats_df.to_csv(sex_stats_path, index=False)
	issue_df.to_csv(issue_path, index=False)

	print(f"Keypoint: {args.keypoint}")
	print(f"Coordinate mode: {args.coord_mode}")
	print(f"ROI mode: {args.roi_mode}")
	print(f"Radius scale: {args.radius_scale:.2f}")
	if args.roi_mode == "contact":
		print(f"Contact buffer: {(float(args.radius_scale) - 1.0) * 100.0:.0f}% of cup radius")
	print(f"Sample unit: {args.sample_unit}")
	if session_n_filter:
		print(f"Session n filter: {', '.join(session_n_filter)}")
	if float(args.max_minutes) > 0:
		print(f"Analysis window: first {float(args.max_minutes):g} min")
	if float(args.min_bout_seconds) > 0:
		print(f"Minimum bout: {float(args.min_bout_seconds):g} s ({int(math.ceil(float(args.min_bout_seconds) * float(args.fps)))} frames)")
	if float(args.max_gap_seconds) > 0:
		print(f"Maximum gap: {float(args.max_gap_seconds):g} s ({int(math.ceil(float(args.max_gap_seconds) * float(args.fps)))} frames)")
	print(f"Exclude Body_C inside pinned cup: {'yes' if exclude_body_in_cup else 'no'}")
	print(f"Exclude {args.keypoint} inside pinned cup from contact ROI: no")
	print(f"Manual contact overrides: {'enabled' if bool(args.use_contact_overrides) else 'disabled'}")
	if bool(args.use_contact_overrides):
		print(f"Override file: {override_path.relative_to(ROOT).as_posix() if override_path.is_relative_to(ROOT) else override_path}")
		print(f"Override ranges applied: {len(relevant_overrides)} across {relevant_overrides['session_key'].nunique() if not relevant_overrides.empty else 0} sessions")
	print(f"Sessions summarized: {len(session_df)}")
	print(f"Samples plotted: {len(sample_df)}")
	print(f"Sessions with issues: {len(issue_df)}")
	for fig_path in figure_paths:
		print(f"Wrote {fig_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {session_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {sample_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {group_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {sex_group_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {stats_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {sex_stats_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {issue_path.relative_to(ROOT).as_posix()}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
