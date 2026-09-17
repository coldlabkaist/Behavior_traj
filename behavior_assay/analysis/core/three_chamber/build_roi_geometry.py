from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT, resolve_input_path
import math
from pathlib import Path
import pandas as pd

REQUIRED_ROIS = ("wall", "chamber_l", "chamber_r")

EXPECTED_VERTEX_COUNTS = {
	"wall": 4,
	"chamber_l": 10,
	"chamber_r": 10,
}

LETTER_ROI_MAP = {
	**{letter: "chamber_l" for letter in "abcdefghij"},
	**{letter: "chamber_r" for letter in "klmnopqrst"},
	**{letter: "wall" for letter in "uvwx"},
}

META_COLUMNS = [
	"date",
	"condition",
	"session_key",
	"canonical_stem",
	"session_stem",
	"subject_id",
	"sex",
	"phase",
	"trial_id",
	"session_n",
	"s1_side",
	"s2_side",
	"social_side",
	"empty_side",
	"familiar_side",
	"novel_side",
	"heatmap_target_label",
	"heatmap_target_side",
	"mirror_lr_for_target_right",
	"layout_raw",
	"pose_path",
	"pins_path",
]

ISSUE_COLUMNS = META_COLUMNS + ["issue", "detail"]

VERTEX_COLUMNS = META_COLUMNS + ["roi_id", "frame", "vertex_idx", "x", "y", "x_norm", "y_norm"]

SUMMARY_COLUMNS = META_COLUMNS + [
	"roi_id",
	"point_count",
	"centroid_x",
	"centroid_y",
	"centroid_x_norm",
	"centroid_y_norm",
	"area_px2",
	"area_norm2",
	"perimeter_px",
	"perimeter_norm",
	"mean_radius_px",
	"mean_radius_norm",
	"max_radius_px",
	"max_radius_norm",
	"min_x",
	"max_x",
	"min_y",
	"max_y",
	"bbox_width_px",
	"bbox_height_px",
	"min_x_norm",
	"max_x_norm",
	"min_y_norm",
	"max_y_norm",
	"bbox_width_norm",
	"bbox_height_norm",
]

def _to_abs_path(path_str: str) -> Path:
    return resolve_input_path(path_str)

def _polygon_metrics(points: list[tuple[float, float]]) -> dict[str, float]:
	if not points:
		return {
			"point_count": 0,
			"centroid_x": math.nan,
			"centroid_y": math.nan,
			"area": math.nan,
			"perimeter": math.nan,
			"mean_radius": math.nan,
			"max_radius": math.nan,
			"min_x": math.nan,
			"max_x": math.nan,
			"min_y": math.nan,
			"max_y": math.nan,
			"bbox_width": math.nan,
			"bbox_height": math.nan,
		}

	closed = points + [points[0]]
	cross_sum = 0.0
	cx_num = 0.0
	cy_num = 0.0
	perimeter = 0.0
	for (x1, y1), (x2, y2) in zip(closed[:-1], closed[1:]):
		cross = x1 * y2 - x2 * y1
		cross_sum += cross
		cx_num += (x1 + x2) * cross
		cy_num += (y1 + y2) * cross
		perimeter += math.dist((x1, y1), (x2, y2))

	area = abs(cross_sum) / 2.0
	if abs(cross_sum) > 1e-12:
		centroid_x = cx_num / (3.0 * cross_sum)
		centroid_y = cy_num / (3.0 * cross_sum)
	else:
		centroid_x = sum(x for x, _ in points) / len(points)
		centroid_y = sum(y for _, y in points) / len(points)

	radii = [math.dist((x, y), (centroid_x, centroid_y)) for x, y in points]
	xs = [x for x, _ in points]
	ys = [y for _, y in points]

	return {
		"point_count": len(points),
		"centroid_x": centroid_x,
		"centroid_y": centroid_y,
		"area": area,
		"perimeter": perimeter,
		"mean_radius": (sum(radii) / len(radii) if radii else math.nan),
		"max_radius": (max(radii) if radii else math.nan),
		"min_x": min(xs),
		"max_x": max(xs),
		"min_y": min(ys),
		"max_y": max(ys),
		"bbox_width": max(xs) - min(xs),
		"bbox_height": max(ys) - min(ys),
	}

def _add_issue(issue_rows: list[dict[str, object]], meta: dict[str, object], issue: str, detail: str = "") -> None:
	issue_rows.append(
		{
			**meta,
			"issue": issue,
			"detail": detail,
		}
	)

def _normalize_roi_ids(pins_df: pd.DataFrame) -> pd.DataFrame:
	out = pins_df.copy()
	roi_ids = set(out["id"].dropna().astype(str))
	if REQUIRED_ROIS[0] in roi_ids or REQUIRED_ROIS[1] in roi_ids or REQUIRED_ROIS[2] in roi_ids:
		return out
	if roi_ids and roi_ids.issubset(set(LETTER_ROI_MAP)):
		out["id"] = out["id"].map(LETTER_ROI_MAP)
		return out
	if roi_ids and all(len(str(roi_id)) == 1 and str(roi_id).isalpha() for roi_id in roi_ids):
		coord_x = "x_norm" if "x_norm" in out.columns else "x"
		coord_y = "y_norm" if "y_norm" in out.columns else "y"
		work = out.copy()
		work[coord_x] = pd.to_numeric(work[coord_x], errors="coerce")
		work[coord_y] = pd.to_numeric(work[coord_y], errors="coerce")
		work = work.dropna(subset=[coord_x, coord_y]).copy()
		if len(work) < 24:
			return out

		y = work[coord_y]
		wall_candidates = work[(y <= 0.20) | (y >= 0.85)].copy()
		if len(wall_candidates) >= 4:
			top = wall_candidates.nsmallest(2, coord_y)
			bottom = wall_candidates.nlargest(2, coord_y)
			wall_idx = pd.Index(top.index.tolist() + bottom.index.tolist()).drop_duplicates()
		else:
			center_y = float(y.median())
			wall_idx = (y - center_y).abs().nlargest(4).index

		chamber = work.drop(index=wall_idx, errors="ignore").copy()
		if len(chamber) < 20:
			return out

		x_sorted = chamber.sort_values(coord_x)
		x_values = x_sorted[coord_x].to_numpy(dtype=float)
		gaps = x_values[1:] - x_values[:-1]
		if len(gaps) == 0:
			return out
		split_pos = int(gaps.argmax()) + 1
		left_idx = x_sorted.iloc[:split_pos].index
		right_idx = x_sorted.iloc[split_pos:].index
		if len(left_idx) < 10 or len(right_idx) < 10:
			return out

		# Keep the original click order within each side; this drops occasional extra clicks.
		left_idx = chamber.loc[left_idx].sort_values("vertex_idx").head(10).index
		right_idx = chamber.loc[right_idx].sort_values("vertex_idx").head(10).index
		wall_idx = work.loc[wall_idx].sort_values("vertex_idx").head(4).index
		keep_idx = list(left_idx) + list(right_idx) + list(wall_idx)
		out = out.loc[keep_idx].copy()
		out.loc[left_idx, "id"] = "chamber_l"
		out.loc[right_idx, "id"] = "chamber_r"
		out.loc[wall_idx, "id"] = "wall"
	return out
