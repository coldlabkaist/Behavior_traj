from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from collections import Counter
import pandas as pd
from analysis.core.three_chamber.build_roi_geometry import _add_issue
from analysis.core.three_chamber.build_roi_geometry import _normalize_roi_ids
from analysis.core.three_chamber.build_roi_geometry import _polygon_metrics
from analysis.core.three_chamber.build_roi_geometry import _to_abs_path

from analysis.core.three_chamber.build_roi_geometry import REQUIRED_ROIS

from analysis.core.three_chamber.build_roi_geometry import EXPECTED_VERTEX_COUNTS

from analysis.core.three_chamber.build_roi_geometry import META_COLUMNS

from analysis.core.three_chamber.build_roi_geometry import ISSUE_COLUMNS

from analysis.core.three_chamber.build_roi_geometry import VERTEX_COLUMNS

from analysis.core.three_chamber.build_roi_geometry import SUMMARY_COLUMNS

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description="Build ROI geometry tables from 3-chamber pins files listed in the manifest."
	)
	parser.add_argument(
		"manifest",
		nargs="?",
		default="output/3chamber/manifest/session_manifest.csv",
		help="Manifest CSV produced by build_manifest.py",
	)
	parser.add_argument(
		"--output-dir",
		default="output/3chamber/roi",
		help="Directory to write roi_vertices.csv, roi_summary.csv, and roi_issues.csv",
	)
	args = parser.parse_args(argv)

	manifest_path = _to_abs_path(args.manifest)
	if not manifest_path.exists():
		raise FileNotFoundError(f"Manifest not found: {manifest_path}")

	output_dir = _to_abs_path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	manifest_df = pd.read_csv(manifest_path)
	if manifest_df.empty:
		print("Manifest is empty.")
		return 0

	vertex_rows: list[dict[str, object]] = []
	summary_rows: list[dict[str, object]] = []
	issue_rows: list[dict[str, object]] = []

	for row in manifest_df.to_dict(orient="records"):
		meta = {column: row.get(column, "") for column in META_COLUMNS}
		pins_rel = str(row.get("pins_path", "")).strip()
		if not pins_rel:
			_add_issue(issue_rows, meta, "missing_pins_path")
			continue

		pins_path = _to_abs_path(pins_rel)
		if not pins_path.exists():
			_add_issue(issue_rows, meta, "pins_file_not_found", pins_rel)
			continue

		pins_df = pd.read_csv(pins_path)
		pins_df.columns = [str(column).strip() for column in pins_df.columns]
		missing_columns = sorted({"id", "x", "y"} - set(pins_df.columns))
		if missing_columns:
			_add_issue(issue_rows, meta, "missing_required_columns", ",".join(missing_columns))
			continue

		pins_df = pins_df.reset_index(names="vertex_idx")
		pins_df["id"] = pins_df["id"].astype(str).str.strip().str.lower()
		pins_df = _normalize_roi_ids(pins_df)
		present_rois = set(pins_df["id"].dropna())

		for roi_id in REQUIRED_ROIS:
			if roi_id not in present_rois:
				_add_issue(issue_rows, meta, "missing_required_roi", roi_id)

		for roi_id, group in pins_df.groupby("id", sort=False):
			group = group.sort_values("vertex_idx").reset_index(drop=True)
			point_count = len(group)
			if point_count < 3:
				_add_issue(issue_rows, meta, "too_few_vertices", f"{roi_id}:{point_count}")

			expected_count = EXPECTED_VERTEX_COUNTS.get(roi_id)
			if expected_count is not None and point_count != expected_count:
				_add_issue(
					issue_rows,
					meta,
					"unexpected_vertex_count",
					f"{roi_id}:{point_count} (expected {expected_count})",
				)

			points_xy = [
				(float(x), float(y))
				for x, y in zip(group["x"].tolist(), group["y"].tolist())
				if pd.notna(x) and pd.notna(y)
			]
			points_norm = (
				[
					(float(x_norm), float(y_norm))
					for x_norm, y_norm in zip(group["x_norm"].tolist(), group["y_norm"].tolist())
					if pd.notna(x_norm) and pd.notna(y_norm)
				]
				if {"x_norm", "y_norm"}.issubset(group.columns)
				else []
			)

			if len(points_xy) != point_count:
				_add_issue(issue_rows, meta, "missing_xy_vertices", f"{roi_id}:{point_count - len(points_xy)}")
			if {"x_norm", "y_norm"}.issubset(group.columns) and len(points_norm) != point_count:
				_add_issue(
					issue_rows,
					meta,
					"missing_norm_vertices",
					f"{roi_id}:{point_count - len(points_norm)}",
				)

			metrics_xy = _polygon_metrics(points_xy)
			metrics_norm = _polygon_metrics(points_norm) if points_norm else _polygon_metrics([])

			summary_rows.append(
				{
					**meta,
					"roi_id": roi_id,
					"point_count": point_count,
					"centroid_x": metrics_xy["centroid_x"],
					"centroid_y": metrics_xy["centroid_y"],
					"centroid_x_norm": metrics_norm["centroid_x"],
					"centroid_y_norm": metrics_norm["centroid_y"],
					"area_px2": metrics_xy["area"],
					"area_norm2": metrics_norm["area"],
					"perimeter_px": metrics_xy["perimeter"],
					"perimeter_norm": metrics_norm["perimeter"],
					"mean_radius_px": metrics_xy["mean_radius"],
					"mean_radius_norm": metrics_norm["mean_radius"],
					"max_radius_px": metrics_xy["max_radius"],
					"max_radius_norm": metrics_norm["max_radius"],
					"min_x": metrics_xy["min_x"],
					"max_x": metrics_xy["max_x"],
					"min_y": metrics_xy["min_y"],
					"max_y": metrics_xy["max_y"],
					"bbox_width_px": metrics_xy["bbox_width"],
					"bbox_height_px": metrics_xy["bbox_height"],
					"min_x_norm": metrics_norm["min_x"],
					"max_x_norm": metrics_norm["max_x"],
					"min_y_norm": metrics_norm["min_y"],
					"max_y_norm": metrics_norm["max_y"],
					"bbox_width_norm": metrics_norm["bbox_width"],
					"bbox_height_norm": metrics_norm["bbox_height"],
				}
			)

			for vertex in group.to_dict(orient="records"):
				vertex_rows.append(
					{
						**meta,
						"roi_id": roi_id,
						"frame": vertex.get("frame", pd.NA),
						"vertex_idx": vertex.get("vertex_idx", pd.NA),
						"x": vertex.get("x", pd.NA),
						"y": vertex.get("y", pd.NA),
						"x_norm": vertex.get("x_norm", pd.NA),
						"y_norm": vertex.get("y_norm", pd.NA),
					}
				)

	vertices_df = pd.DataFrame(vertex_rows, columns=VERTEX_COLUMNS)
	summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
	issues_df = pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS)

	if not issues_df.empty and not vertices_df.empty:
		resolved_issue_indices: list[int] = []
		fallback_vertex_rows: list[pd.DataFrame] = []
		fallback_summary_rows: list[pd.DataFrame] = []
		for issue_idx, issue_row in issues_df.iterrows():
			if issue_row.get("issue") != "missing_required_roi" or issue_row.get("detail") != "wall":
				continue
			missing_session_key = issue_row.get("session_key")
			match_cols = ["condition", "date", "subject_id", "trial_id"]
			donor_mask = (vertices_df["roi_id"] == "wall") & (vertices_df["session_key"] != missing_session_key)
			for col in match_cols:
				donor_mask &= vertices_df[col].astype(str) == str(issue_row.get(col, ""))
			donor_vertices = vertices_df[donor_mask].copy()
			if donor_vertices.empty:
				donor_mask = (
					(vertices_df["roi_id"] == "wall")
					& (vertices_df["session_key"] != missing_session_key)
					& (vertices_df["condition"].astype(str) == str(issue_row.get("condition", "")))
					& (vertices_df["date"].astype(str) == str(issue_row.get("date", "")))
				)
				donor_vertices = vertices_df[donor_mask].copy()
			if donor_vertices.empty:
				continue
			donor_key = donor_vertices.iloc[0]["session_key"]
			donor_vertices = donor_vertices[donor_vertices["session_key"] == donor_key].copy()
			for col in META_COLUMNS:
				donor_vertices[col] = issue_row.get(col, "")
			fallback_vertex_rows.append(donor_vertices)

			donor_summary = summary_df[(summary_df["roi_id"] == "wall") & (summary_df["session_key"] == donor_key)].copy()
			if not donor_summary.empty:
				for col in META_COLUMNS:
					donor_summary[col] = issue_row.get(col, "")
				fallback_summary_rows.append(donor_summary)
			resolved_issue_indices.append(int(issue_idx))

		if fallback_vertex_rows:
			vertices_df = pd.concat([vertices_df, *fallback_vertex_rows], ignore_index=True)
		if fallback_summary_rows:
			summary_df = pd.concat([summary_df, *fallback_summary_rows], ignore_index=True)
		if resolved_issue_indices:
			issues_df = issues_df.drop(index=resolved_issue_indices).reset_index(drop=True)

	if not vertices_df.empty:
		vertices_df = vertices_df.sort_values(
			["date", "subject_id", "phase", "trial_id", "session_n", "roi_id", "vertex_idx"],
			na_position="last",
		).reset_index(drop=True)
	if not summary_df.empty:
		summary_df = summary_df.sort_values(
			["date", "subject_id", "phase", "trial_id", "session_n", "roi_id"],
			na_position="last",
		).reset_index(drop=True)
	if not issues_df.empty:
		issues_df = issues_df.sort_values(
			["date", "subject_id", "phase", "trial_id", "session_n", "issue", "detail"],
			na_position="last",
		).reset_index(drop=True)

	vertices_path = output_dir / "roi_vertices.csv"
	summary_path = output_dir / "roi_summary.csv"
	issues_path = output_dir / "roi_issues.csv"
	vertices_df.to_csv(vertices_path, index=False)
	summary_df.to_csv(summary_path, index=False)
	issues_df.to_csv(issues_path, index=False)

	issue_counter = Counter(issues_df["issue"].tolist()) if not issues_df.empty else Counter()
	print(f"Sessions in manifest: {len(manifest_df)}")
	print(f"ROI summary rows: {len(summary_df)}")
	print(f"ROI vertex rows: {len(vertices_df)}")
	print(f"Sessions with ROI issues: {issues_df['session_key'].nunique() if not issues_df.empty else 0}")
	for issue, count in sorted(issue_counter.items()):
		print(f" - {issue}: {count}")
	print(f"Wrote {vertices_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {summary_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {issues_path.relative_to(ROOT).as_posix()}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
