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
	"hab_closed": {"title": "Habituation closed"},
	"hab_open": {"title": "Habituation open"},
	"soc": {"title": "Sociability test"},
	"nov": {"title": "Social Novelty test"},
}

PHASE_ORDER = ["hab_closed", "hab_open", "soc", "nov"]

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
	phases: list[str] = []
	for token in str(arg).split(","):
		phase = token.strip().lower()
		if not phase:
			continue
		if phase == "hab":
			phase = "hab_closed"
		if phase not in phases:
			phases.append(phase)
	if not phases:
		raise ValueError("At least one phase must be provided")
	return phases

def _analysis_phase(row: pd.Series | dict[str, object]) -> str:
	phase = str(row.get("phase", "")).strip().lower()
	if phase == "hab":
		detail = str(row.get("phase_detail", "")).strip().lower()
		if detail in {"hab_open", "hab_closed"}:
			return detail
		hab_type = str(row.get("hab_type", "")).strip().lower()
		if hab_type in {"open", "closed"}:
			return f"hab_{hab_type}"
		return "hab_closed"
	return phase

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

def _key_text(value: object) -> str:
	if value is None or pd.isna(value):
		return ""
	try:
		num = float(value)
	except (TypeError, ValueError):
		return str(value).strip()
	if np.isfinite(num) and abs(num - round(num)) < 1e-9:
		return str(int(round(num)))
	return str(value).strip()

def _filter_session_n_with_matching_hab(sessions_df: pd.DataFrame, session_n_filter: list[str]) -> pd.DataFrame:
	if not session_n_filter:
		return sessions_df

	work = sessions_df.copy()
	work["session_n_norm"] = work["session_n"].map(_normalize_session_n)
	work["_subject_key"] = work["subject_id"].map(_key_text)
	work["_trial_key"] = work["trial_id"].map(_key_text)

	test_mask = work["analysis_phase"].isin(["soc", "nov"]) & work["session_n_norm"].isin(session_n_filter)
	test_keys = set(
		zip(
			work.loc[test_mask, "condition"].map(str),
			work.loc[test_mask, "_subject_key"],
			work.loc[test_mask, "_trial_key"],
		)
	)

	hab_mask = work["analysis_phase"].isin(["hab_closed", "hab_open"])
	hab_keys = list(zip(work["condition"].map(str), work["_subject_key"], work["_trial_key"]))
	hab_match_mask = hab_mask & pd.Series([key in test_keys for key in hab_keys], index=work.index)

	return work[test_mask | hab_match_mask].drop(columns=["session_n_norm", "_subject_key", "_trial_key"]).copy()

def _condition_sort_key(condition: str) -> tuple[int, str]:
	if condition in PREFERRED_CONDITION_ORDER:
		return (PREFERRED_CONDITION_ORDER.index(condition), condition)
	return (len(PREFERRED_CONDITION_ORDER), condition)

def _phase_sort_key(phase: str) -> tuple[int, str]:
	if phase in PHASE_ORDER:
		return (PHASE_ORDER.index(phase), phase)
	return (len(PHASE_ORDER), phase)

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

def _sem(series: pd.Series) -> float:
	values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
	if values.size <= 1:
		return 0.0
	return float(np.std(values, ddof=1) / math.sqrt(values.size))

def _welch_ttest(group_a: pd.Series, group_b: pd.Series) -> dict[str, object]:
	a = pd.to_numeric(group_a, errors="coerce").dropna().to_numpy(dtype=float)
	b = pd.to_numeric(group_b, errors="coerce").dropna().to_numpy(dtype=float)
	if len(a) < 2 or len(b) < 2:
		return {"test": "welch_ttest", "n_a": len(a), "n_b": len(b), "statistic": math.nan, "pvalue": math.nan}
	stat = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
	return {"test": "welch_ttest", "n_a": len(a), "n_b": len(b), "statistic": float(stat.statistic), "pvalue": float(stat.pvalue)}

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
		"valid_steps",
		"duration_s",
		"duration_min",
		"total_distance_px",
		"total_distance_norm",
		"distance_per_min_px",
		"distance_per_min_norm",
		"mean_speed_px_per_s",
		"mean_speed_norm_per_s",
		"left_zone_frames",
		"center_zone_frames",
		"right_zone_frames",
		"left_zone_time_s",
		"center_zone_time_s",
		"right_zone_time_s",
		"left_zone_pct",
		"center_zone_pct",
		"right_zone_pct",
		"spatial_preference_index",
	]
	agg_map: dict[str, str] = {col: "mean" for col in numeric_cols if col in session_df.columns}
	for col in ("sex", "keypoint", "fps", "raw_phase", "phase_detail", "hab_type"):
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
