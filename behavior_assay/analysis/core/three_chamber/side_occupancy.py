from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT, resolve_input_path
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

FIGURE_DPI = 300

PHASE_CONFIG = {
	"soc": {
		"title": "Sociability side occupancy",
		"left_label": "E side",
		"right_label": "S side",
		"time_left_col": "empty_side_time_s",
		"time_right_col": "social_side_time_s",
		"visit_left_col": "empty_side_visit_count",
		"visit_right_col": "social_side_visit_count",
		"pref_col": "social_side_preference_index",
	},
	"nov": {
		"title": "Social Novelty side occupancy",
		"left_label": "F side",
		"right_label": "N side",
		"time_left_col": "familiar_side_time_s",
		"time_right_col": "novel_side_time_s",
		"visit_left_col": "familiar_side_visit_count",
		"visit_right_col": "novel_side_visit_count",
		"pref_col": "novel_side_preference_index",
	},
}

PREFERRED_CONDITION_ORDER = ["Control", "control", "VPA", "vpa"]

CONDITION_COLORS = {
	"Control": "#304F78",
	"control": "#304F78",
	"VPA": "#C4475B",
	"vpa": "#C4475B",
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

def _condition_sort_key(condition: str) -> tuple[int, str]:
	if condition in PREFERRED_CONDITION_ORDER:
		return (PREFERRED_CONDITION_ORDER.index(condition), condition)
	return (len(PREFERRED_CONDITION_ORDER), condition)

def _point_in_polygon_mask(x: np.ndarray, y: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
	if polygon_xy.shape[0] < 3:
		return np.zeros(len(x), dtype=bool)
	cx = float(np.mean(polygon_xy[:, 0]))
	cy = float(np.mean(polygon_xy[:, 1]))
	angles = np.arctan2(polygon_xy[:, 1] - cy, polygon_xy[:, 0] - cx)
	poly = polygon_xy[np.argsort(angles)]
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

def _count_visits(mask: np.ndarray) -> int:
	mask = np.asarray(mask, dtype=bool)
	if mask.size == 0:
		return 0
	return int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1]))

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

def _onesample_ttest(values: pd.Series, popmean: float = 0.0) -> dict[str, object]:
	arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
	if len(arr) < 2:
		return {"test": "onesample_ttest", "n": len(arr), "popmean": popmean, "statistic": math.nan, "pvalue": math.nan}
	stat = stats.ttest_1samp(arr, popmean=popmean, nan_policy="omit")
	return {"test": "onesample_ttest", "n": len(arr), "popmean": popmean, "statistic": float(stat.statistic), "pvalue": float(stat.pvalue)}

def _aggregate_trial_means(session_df: pd.DataFrame) -> pd.DataFrame:
	if session_df.empty:
		return session_df.copy()
	group_cols = ["condition", "phase", "subject_id", "trial_id"]
	numeric_cols = [
		"valid_frames",
		"left_side_frames",
		"right_side_frames",
		"left_side_time_s",
		"right_side_time_s",
		"left_side_visit_count",
		"right_side_visit_count",
		"empty_side_time_s",
		"social_side_time_s",
		"empty_side_visit_count",
		"social_side_visit_count",
		"social_side_preference_index",
		"familiar_side_time_s",
		"novel_side_time_s",
		"familiar_side_visit_count",
		"novel_side_visit_count",
		"novel_side_preference_index",
	]
	agg_map: dict[str, str] = {col: "mean" for col in numeric_cols if col in session_df.columns}
	for col in ("sex", "keypoint", "fps"):
		if col in session_df.columns:
			agg_map[col] = "first"
	grouped = session_df.groupby(group_cols, dropna=False, as_index=False).agg(agg_map)
	counts = session_df.groupby(group_cols, dropna=False)["session_key"].count().reset_index(name="sessions_averaged")
	session_n_values = session_df.groupby(group_cols, dropna=False)["session_n"].apply(_format_session_n_values).reset_index(name="session_n_values")
	grouped = grouped.merge(counts, on=group_cols, how="left").merge(session_n_values, on=group_cols, how="left")
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
	sample_df["session_n_values"] = sample_df["session_n"].map(lambda value: _normalize_session_n(value) or "")
	sample_df["sample_unit"] = "session"
	sample_df["sample_key"] = sample_df["session_key"].astype(str)
	return sample_df
