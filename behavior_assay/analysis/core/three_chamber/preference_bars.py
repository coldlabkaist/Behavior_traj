from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT, resolve_input_path
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from analysis.core.three_chamber.contact_geometry import buffer_polygon
from analysis.core.three_chamber.contact_events import close_short_gaps, count_visits, filter_short_bouts

FIGURE_DPI = 1200

PHASE_CONFIG = {
	"soc": {
		"title": "Sociability test",
		"left_label": "E",
		"right_label": "S",
		"time_left_col": "empty_time_s",
		"time_right_col": "social_time_s",
		"visit_left_col": "empty_visit_count",
		"visit_right_col": "social_visit_count",
		"pref_col": "social_preference_index",
		"pref_ylabel": "Preference index",
	},
	"nov": {
		"title": "Social Novelty test",
		"left_label": "F",
		"right_label": "N",
		"time_left_col": "familiar_time_s",
		"time_right_col": "novel_time_s",
		"visit_left_col": "familiar_visit_count",
		"visit_right_col": "novel_visit_count",
		"pref_col": "novel_preference_index",
		"pref_ylabel": "Preference index",
	},
}

PREFERRED_CONDITION_ORDER = ["Control", "control", "VPA", "vpa"]

CONDITION_COLORS = {
	"Control": "#304F78",
	"control": "#304F78",
	"VPA": "#C4475B",
	"vpa": "#C4475B",
}

SEX_ORDER = ["m", "f"]

SEX_COLORS = {
	"m": "#2f6db3",
	"f": "#c04c7a",
}

def _to_abs_path(path_str: str) -> Path:
    return resolve_input_path(path_str)

def _parse_phase_list(arg: str) -> list[str]:
	phases = [token.strip().lower() for token in str(arg).split(",") if token.strip()]
	if not phases:
		raise ValueError("At least one phase must be provided")
	return phases

def _normalize_session_n(value: object) -> str | None:
	if value is None or pd.isna(value):
		return None
	text = str(value).strip().lower()
	if not text:
		return None
	if text.startswith("n_"):
		text = text[2:]
	elif text.startswith("n"):
		text = text[1:]
	try:
		num = float(text)
	except ValueError:
		return None
	if not np.isfinite(num):
		return None
	if abs(num - round(num)) < 1e-9:
		return str(int(round(num)))
	return str(num)

def _parse_session_n_filter(arg: str) -> list[str]:
	values = []
	for token in str(arg).split(","):
		norm = _normalize_session_n(token)
		if norm is not None and norm not in values:
			values.append(norm)
	return values

def _format_session_n_values(series: pd.Series) -> str:
	values = sorted(pd.to_numeric(series, errors="coerce").dropna().astype(int).unique())
	return ",".join(str(v) for v in values)

def _format_minutes_suffix(max_minutes: float) -> str:
	if not np.isfinite(max_minutes) or max_minutes <= 0:
		return ""
	if abs(max_minutes - round(max_minutes)) < 1e-9:
		return f"__first{int(round(max_minutes))}min"
	return f"__first{max_minutes:g}min".replace(".", "p")

def _filter_first_minutes(df: pd.DataFrame, *, max_minutes: float, fps: float) -> pd.DataFrame:
	if not np.isfinite(max_minutes) or max_minutes <= 0:
		return df
	max_frames = int(round(max_minutes * 60.0 * max(float(fps), 1e-9)))
	if max_frames <= 0:
		return df.iloc[0:0].copy()
	if "frame_idx" in df.columns:
		frame_values = pd.to_numeric(df["frame_idx"], errors="coerce")
		if frame_values.notna().any():
			frame_min = int(frame_values.dropna().min())
			return df.loc[frame_values <= frame_min + max_frames - 1].copy()
	return df.head(max_frames).copy()

def _condition_sort_key(condition: str) -> tuple[int, str]:
	if condition in PREFERRED_CONDITION_ORDER:
		return (PREFERRED_CONDITION_ORDER.index(condition), condition)
	return (len(PREFERRED_CONDITION_ORDER), condition)

def _sex_sort_key(sex: str) -> tuple[int, str]:
	sex = str(sex).lower()
	if sex in SEX_ORDER:
		return (SEX_ORDER.index(sex), sex)
	return (len(SEX_ORDER), sex)

def _sex_label(sex: str) -> str:
	sex = str(sex).lower()
	if sex == "m":
		return "M"
	if sex == "f":
		return "F"
	return sex.upper() if sex else "?"

def _sex_color(sex: str) -> str:
	return SEX_COLORS.get(str(sex).lower(), "#777777")

def _point_in_polygon_mask(x: np.ndarray, y: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
	if polygon_xy.shape[0] < 3:
		return np.zeros(len(x), dtype=bool)
	cx = float(np.mean(polygon_xy[:, 0]))
	cy = float(np.mean(polygon_xy[:, 1]))
	angles = np.arctan2(polygon_xy[:, 1] - cy, polygon_xy[:, 0] - cx)
	order = np.argsort(angles)
	poly = polygon_xy[order]
	px = poly[:, 0]
	py = poly[:, 1]
	px_next = np.roll(px, -1)
	py_next = np.roll(py, -1)
	inside = np.zeros(len(x), dtype=bool)
	for x1, y1, x2, y2 in zip(px, py, px_next, py_next):
		crosses = (y1 > y) != (y2 > y)
		x_intersect = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
		inside ^= crosses & (x < x_intersect)
	return inside

def _infer_image_size(vertices_df: pd.DataFrame) -> tuple[float, float]:
	"""Infer source image dimensions from paired pixel and normalized pin coordinates."""
	required = {"x", "y", "x_norm", "y_norm"}
	if not required.issubset(vertices_df.columns):
		raise ValueError("ROI vertices must contain x, y, x_norm, and y_norm")

	x = pd.to_numeric(vertices_df["x"], errors="coerce").to_numpy(dtype=float)
	y = pd.to_numeric(vertices_df["y"], errors="coerce").to_numpy(dtype=float)
	x_norm = pd.to_numeric(vertices_df["x_norm"], errors="coerce").to_numpy(dtype=float)
	y_norm = pd.to_numeric(vertices_df["y_norm"], errors="coerce").to_numpy(dtype=float)
	x_valid = np.isfinite(x) & np.isfinite(x_norm) & (np.abs(x_norm) > 1e-9)
	y_valid = np.isfinite(y) & np.isfinite(y_norm) & (np.abs(y_norm) > 1e-9)
	if not x_valid.any() or not y_valid.any():
		raise ValueError("Could not infer image dimensions from ROI vertices")

	width = float(np.median(x[x_valid] / x_norm[x_valid]))
	height = float(np.median(y[y_valid] / y_norm[y_valid]))
	if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
		raise ValueError(f"Invalid inferred image dimensions: {width} x {height}")
	return width, height

def _compute_circle_mask(
	x: np.ndarray,
	y: np.ndarray,
	*,
	cx: float,
	cy: float,
	radius: float,
) -> np.ndarray:
	d2 = (x - float(cx)) ** 2 + (y - float(cy)) ** 2
	return d2 <= float(radius) ** 2

def _buffer_polygon(polygon_xy: np.ndarray, distance: float) -> np.ndarray:
	return buffer_polygon(polygon_xy, distance, resolution=16)

def _roi_output_tag(roi_mode: str, radius_scale: float) -> str:
	if roi_mode == "contact":
		return f"contact{int(round(max(radius_scale - 1.0, 0.0) * 100.0))}"
	return f"{roi_mode}_r{radius_scale:.2f}"

def _count_visits(mask: np.ndarray) -> int:
	return count_visits(mask)

def _filter_short_bouts(mask: np.ndarray, min_frames: int) -> tuple[np.ndarray, int, int]:
	return filter_short_bouts(mask, min_frames)

def _sem(series: pd.Series) -> float:
	values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
	if values.size <= 1:
		return 0.0
	return float(np.std(values, ddof=1) / math.sqrt(values.size))

def _paired_ttest(left: pd.Series, right: pd.Series) -> dict[str, object]:
	pair_df = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
	if len(pair_df) < 2:
		return {"test": "paired_ttest", "n": len(pair_df), "statistic": math.nan, "pvalue": math.nan}
	stat = stats.ttest_rel(pair_df["left"], pair_df["right"], nan_policy="omit")
	return {"test": "paired_ttest", "n": len(pair_df), "statistic": float(stat.statistic), "pvalue": float(stat.pvalue)}

def _welch_ttest(group_a: pd.Series, group_b: pd.Series) -> dict[str, object]:
	a = pd.to_numeric(group_a, errors="coerce").dropna().to_numpy(dtype=float)
	b = pd.to_numeric(group_b, errors="coerce").dropna().to_numpy(dtype=float)
	if len(a) < 2 or len(b) < 2:
		return {"test": "welch_ttest", "n_a": len(a), "n_b": len(b), "statistic": math.nan, "pvalue": math.nan}
	stat = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
	return {"test": "welch_ttest", "n_a": len(a), "n_b": len(b), "statistic": float(stat.statistic), "pvalue": float(stat.pvalue)}

def _brown_forsythe_test(group_a: pd.Series, group_b: pd.Series) -> dict[str, object]:
	a = pd.to_numeric(group_a, errors="coerce").dropna().to_numpy(dtype=float)
	b = pd.to_numeric(group_b, errors="coerce").dropna().to_numpy(dtype=float)
	variance_a = float(np.var(a, ddof=1)) if len(a) > 1 else math.nan
	variance_b = float(np.var(b, ddof=1)) if len(b) > 1 else math.nan
	if len(a) < 2 or len(b) < 2:
		return {
			"test": "brown_forsythe",
			"n_a": len(a),
			"n_b": len(b),
			"variance_a": variance_a,
			"variance_b": variance_b,
			"variance_ratio_b_over_a": math.nan,
			"statistic": math.nan,
			"pvalue": math.nan,
		}
	stat = stats.levene(a, b, center="median")
	variance_ratio = variance_b / variance_a if np.isfinite(variance_a) and variance_a > 0 else math.nan
	return {
		"test": "brown_forsythe",
		"n_a": len(a),
		"n_b": len(b),
		"variance_a": variance_a,
		"variance_b": variance_b,
		"variance_ratio_b_over_a": variance_ratio,
		"statistic": float(stat.statistic),
		"pvalue": float(stat.pvalue),
	}

def _holm_adjust(pvalues: pd.Series) -> np.ndarray:
	values = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
	adjusted = np.full(values.shape, np.nan, dtype=float)
	finite_indices = np.flatnonzero(np.isfinite(values))
	if finite_indices.size == 0:
		return adjusted
	order = finite_indices[np.argsort(values[finite_indices])]
	ranked = values[order]
	m = len(ranked)
	ranked_adjusted = np.maximum.accumulate((m - np.arange(m)) * ranked)
	adjusted[order] = np.minimum(ranked_adjusted, 1.0)
	return adjusted

def _onesample_ttest(values: pd.Series, popmean: float = 0.0) -> dict[str, object]:
	arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
	if len(arr) < 2:
		return {"test": "onesample_ttest", "n": len(arr), "popmean": popmean, "statistic": math.nan, "pvalue": math.nan}
	stat = stats.ttest_1samp(arr, popmean=popmean, nan_policy="omit")
	return {
		"test": "onesample_ttest",
		"n": len(arr),
		"popmean": popmean,
		"statistic": float(stat.statistic),
		"pvalue": float(stat.pvalue),
	}

def _aggregate_trial_means(session_df: pd.DataFrame) -> pd.DataFrame:
	if session_df.empty:
		return session_df.copy()
	group_cols = ["condition", "phase", "subject_id", "trial_id"]
	numeric_cols = [
		"valid_frames",
		"left_frames",
		"right_frames",
		"left_pct",
		"right_pct",
		"left_time_s",
		"right_time_s",
		"left_visit_count",
		"right_visit_count",
		"empty_pct",
		"social_pct",
		"empty_time_s",
		"social_time_s",
		"empty_visit_count",
		"social_visit_count",
		"empty_within_roi_pct",
		"social_within_roi_pct",
		"social_preference_index",
		"familiar_pct",
		"novel_pct",
		"familiar_time_s",
		"novel_time_s",
		"familiar_visit_count",
		"novel_visit_count",
		"familiar_within_roi_pct",
		"novel_within_roi_pct",
		"novel_preference_index",
		"left_raw_frames",
		"right_raw_frames",
		"left_gap_frames_filled",
		"right_gap_frames_filled",
		"left_gaps_closed",
		"right_gaps_closed",
		"left_short_frames_removed",
		"right_short_frames_removed",
		"left_short_bouts_removed",
		"right_short_bouts_removed",
		"left_body_in_cup_frames_removed",
		"right_body_in_cup_frames_removed",
		"left_nose_in_cup_frames_removed",
		"right_nose_in_cup_frames_removed",
		"manual_override_frames",
		"left_manual_frames_added",
		"left_manual_frames_removed",
		"right_manual_frames_added",
		"right_manual_frames_removed",
	]
	agg_map: dict[str, str] = {col: "mean" for col in numeric_cols if col in session_df.columns}
	for col in ("sex", "keypoint", "roi_mode", "radius_scale", "contact_buffer_fraction", "fps", "max_minutes", "max_gap_seconds", "max_gap_frames", "min_bout_seconds", "min_bout_frames", "exclude_body_in_cup", "contact_overrides_applied"):
		if col in session_df.columns:
			agg_map[col] = "first"
	grouped = session_df.groupby(group_cols, dropna=False, as_index=False).agg(agg_map)
	session_counts = (
		session_df.groupby(group_cols, dropna=False)["session_key"]
		.count()
		.reset_index(name="sessions_averaged")
	)
	session_n_values = (
		session_df.groupby(group_cols, dropna=False)["session_n"]
		.apply(_format_session_n_values)
		.reset_index(name="session_n_values")
	)
	grouped = grouped.merge(session_counts, on=group_cols, how="left").merge(session_n_values, on=group_cols, how="left")
	grouped["sample_unit"] = "trial_mean"
	grouped["sample_key"] = grouped.apply(
		lambda row: f"{row['condition']}__{row['subject_id']}__id{int(row['trial_id'])}" if pd.notna(row["trial_id"]) else f"{row['condition']}__{row['subject_id']}",
		axis=1,
	)
	return grouped

def _build_sample_df(session_df: pd.DataFrame, sample_unit: str) -> pd.DataFrame:
	if sample_unit == "trial_mean":
		return _aggregate_trial_means(session_df)
	sample_df = session_df.copy()
	sample_df["sessions_averaged"] = 1
	sample_df["session_n_values"] = sample_df["session_n"].map(_normalize_session_n).fillna("")
	sample_df["sample_unit"] = "session"
	sample_df["sample_key"] = sample_df["session_key"].astype(str)
	return sample_df

def _paired_panel_top_limit(finite_values: list[float], group_count: int, *, sex_stratified: bool = False) -> float:
	ymax = max(finite_values, default=0.0)
	yrange = max(ymax, 1.0)
	if sex_stratified:
		step = 0.07 * yrange
		return max(ymax + 0.24 * yrange + group_count * step, 1.0)
	step = 0.12 * yrange
	return max(ymax + 0.3 * yrange + max(group_count - 1, 0) * step, 1.0)

def _extend_panel_values(finite_values: list[float], left_vals: pd.Series, right_vals: pd.Series) -> None:
	left_vals = pd.to_numeric(left_vals, errors="coerce")
	right_vals = pd.to_numeric(right_vals, errors="coerce")
	for value in np.concatenate([left_vals.dropna().to_numpy(dtype=float), right_vals.dropna().to_numpy(dtype=float)]):
		if np.isfinite(value):
			finite_values.append(float(value))
	for top in (float(left_vals.mean()) + _sem(left_vals), float(right_vals.mean()) + _sem(right_vals)):
		if np.isfinite(top):
			finite_values.append(float(top))

def _compute_shared_panel_ylims(sample_df: pd.DataFrame, phase_list: list[str], *, sex_stratified: bool = False) -> dict[str, float]:
	limits = {"investigation_time_s": 1.0, "visit_count": 1.0}
	for phase in phase_list:
		if phase not in PHASE_CONFIG:
			continue
		cfg = PHASE_CONFIG[phase]
		phase_df = sample_df[sample_df["phase"] == phase].copy()
		if phase_df.empty:
			continue

		if sex_stratified:
			if "sex" not in phase_df.columns:
				continue
			phase_df["sex"] = phase_df["sex"].astype(str).str.lower().str.strip()
			phase_df = phase_df[phase_df["sex"].isin(SEX_ORDER)].copy()
			if phase_df.empty:
				continue
			condition_order = sorted(phase_df["condition"].dropna().astype(str).unique().tolist(), key=_condition_sort_key)
			sex_order = [sex for sex in SEX_ORDER if sex in set(phase_df["sex"])]
			group_defs: list[tuple[str, str]] = []
			for condition in condition_order:
				condition_df = phase_df[phase_df["condition"].astype(str) == condition]
				for sex in sex_order:
					if not condition_df[condition_df["sex"] == sex].empty:
						group_defs.append((condition, sex))
		else:
			condition_order = sorted(phase_df["condition"].dropna().astype(str).unique().tolist(), key=_condition_sort_key)
			group_defs = [(condition, "") for condition in condition_order]

		for panel_name, left_col, right_col in (
			("investigation_time_s", cfg["time_left_col"], cfg["time_right_col"]),
			("visit_count", cfg["visit_left_col"], cfg["visit_right_col"]),
		):
			finite_values: list[float] = []
			for condition, sex in group_defs:
				group = phase_df[phase_df["condition"].astype(str) == condition].copy()
				if sex_stratified:
					group = group[group["sex"] == sex].copy()
				_extend_panel_values(finite_values, group[left_col], group[right_col])
			limits[panel_name] = max(
				limits[panel_name],
				_paired_panel_top_limit(finite_values, len(group_defs), sex_stratified=sex_stratified),
			)
	return limits
