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

DEFAULT_RADIUS_SCALES = "1.0,1.25,1.5,1.75,2.0,2.25,2.5"

PREFERRED_CONDITION_ORDER = ["Control", "control", "VPA", "vpa"]

CONDITION_COLORS = {
	"Control": "#304F78",
	"control": "#304F78",
	"VPA": "#C4475B",
	"vpa": "#C4475B",
}

PHASE_CONFIG = {
	"soc": {
		"title": "Sociability test",
		"target_label": "S",
		"opposite_label": "E",
		"target_side_col": "social_side",
		"opposite_side_col": "empty_side",
		"pref_name": "social",
	},
	"nov": {
		"title": "Social Novelty test",
		"target_label": "N",
		"opposite_label": "F",
		"target_side_col": "novel_side",
		"opposite_side_col": "familiar_side",
		"pref_name": "novel",
	},
}

def _to_abs_path(path_str: str) -> Path:
    return resolve_input_path(path_str)

def _parse_phase_list(arg: str) -> list[str]:
	phases = [token.strip().lower() for token in str(arg).split(",") if token.strip()]
	if not phases:
		raise ValueError("At least one phase must be provided")
	return phases

def _parse_radius_scales(arg: str) -> list[float]:
	values = []
	for token in str(arg).split(","):
		token = token.strip()
		if not token:
			continue
		value = float(token)
		if value <= 0:
			raise ValueError(f"Radius scale must be positive: {value}")
		values.append(value)
	if not values:
		raise ValueError("At least one radius scale is required")
	return sorted(dict.fromkeys(values))

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

def _condition_color(condition: str) -> str:
	return CONDITION_COLORS.get(str(condition), "#777777")

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

def _welch_ttest(group_a: pd.Series, group_b: pd.Series) -> dict[str, object]:
	a = pd.to_numeric(group_a, errors="coerce").dropna().to_numpy(dtype=float)
	b = pd.to_numeric(group_b, errors="coerce").dropna().to_numpy(dtype=float)
	if len(a) < 2 or len(b) < 2:
		return {"test": "welch_ttest", "n_a": len(a), "n_b": len(b), "statistic": math.nan, "pvalue": math.nan}
	stat = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
	return {"test": "welch_ttest", "n_a": len(a), "n_b": len(b), "statistic": float(stat.statistic), "pvalue": float(stat.pvalue)}

def _count_visits(mask: np.ndarray) -> int:
	mask = np.asarray(mask, dtype=bool)
	if mask.size == 0:
		return 0
	return int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1]))

def _side_to_roi_id(side: object) -> str | None:
	text = str(side).strip().lower()
	if text in {"l", "left", "chamber_l"}:
		return "chamber_l"
	if text in {"r", "right", "chamber_r"}:
		return "chamber_r"
	return None

def _safe_div(num: float, den: float) -> float:
	if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
		return math.nan
	return float(num / den)

def _build_roi_lookup(roi_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
	lookup: dict[tuple[str, str], dict[str, float]] = {}
	use_df = roi_df[roi_df["roi_id"].isin(["chamber_l", "chamber_r"])].copy()
	for row in use_df.to_dict(orient="records"):
		lookup[(str(row["session_key"]), str(row["roi_id"]))] = {
			"cx": float(row["centroid_x_norm"]),
			"cy": float(row["centroid_y_norm"]),
			"radius": float(row["mean_radius_norm"]),
		}
	return lookup

def _read_keypoint_xy(path: Path, keypoint: str, *, max_minutes: float, fps: float) -> tuple[np.ndarray, np.ndarray]:
	cols = [f"{keypoint}.x", f"{keypoint}.y"]
	df = pd.read_csv(path, usecols=lambda col: col in cols or col == "frame_idx")
	df = _filter_first_minutes(df, max_minutes=max_minutes, fps=fps)
	if not set(cols).issubset(df.columns):
		missing = sorted(set(cols) - set(df.columns))
		raise ValueError(f"Missing keypoint columns in {path}: {missing}")
	x = pd.to_numeric(df[cols[0]], errors="coerce").to_numpy(dtype=float)
	y = pd.to_numeric(df[cols[1]], errors="coerce").to_numpy(dtype=float)
	valid = np.isfinite(x) & np.isfinite(y)
	return x[valid], y[valid]

def _base_session_meta(row: dict[str, object], keypoint: str) -> dict[str, object]:
	return {
		"date": row.get("date", ""),
		"condition": row.get("condition", ""),
		"session_key": row.get("session_key", ""),
		"subject_id": row.get("subject_id", ""),
		"sex": row.get("sex", ""),
		"phase": row.get("phase", ""),
		"trial_id": row.get("trial_id", math.nan),
		"session_n": row.get("session_n", math.nan),
		"layout_raw": row.get("layout_raw", ""),
		"keypoint": keypoint,
		"fps": row.get("fps", math.nan),
	}

def _compute_session_metrics(
	row: dict[str, object],
	roi_lookup: dict[tuple[str, str], dict[str, float]],
	radius_scales: list[float],
	keypoint: str,
	max_minutes: float,
) -> tuple[list[dict[str, object]], dict[str, object] | None, list[dict[str, object]]]:
	issues: list[dict[str, object]] = []
	base_meta = _base_session_meta(row, keypoint)
	phase = str(row.get("phase", "")).lower()
	if phase not in PHASE_CONFIG:
		return [], None, []
	cfg = PHASE_CONFIG[phase]
	target_roi_id = _side_to_roi_id(row.get(cfg["target_side_col"]))
	opposite_roi_id = _side_to_roi_id(row.get(cfg["opposite_side_col"]))
	if target_roi_id is None or opposite_roi_id is None:
		issues.append({**base_meta, "issue": "missing_target_or_opposite_side"})
		return [], None, issues

	session_key = str(row.get("session_key", ""))
	target_roi = roi_lookup.get((session_key, target_roi_id))
	opposite_roi = roi_lookup.get((session_key, opposite_roi_id))
	if target_roi is None or opposite_roi is None:
		issues.append({**base_meta, "issue": "missing_roi_geometry"})
		return [], None, issues

	output_path = _to_abs_path(str(row.get("output_path", "")))
	if not output_path.exists():
		issues.append({**base_meta, "issue": "missing_preprocessed_file", "detail": str(output_path)})
		return [], None, issues

	fps = float(row.get("fps", 30.0))
	if not np.isfinite(fps) or fps <= 0:
		fps = 30.0
	try:
		x, y = _read_keypoint_xy(output_path, keypoint, max_minutes=max_minutes, fps=fps)
	except Exception as exc:
		issues.append({**base_meta, "issue": "failed_to_read_keypoint", "detail": str(exc)})
		return [], None, issues

	valid_frames = int(len(x))
	if valid_frames == 0:
		issues.append({**base_meta, "issue": "no_valid_keypoint_frames"})
		return [], None, issues

	target_dist = np.sqrt((x - target_roi["cx"]) ** 2 + (y - target_roi["cy"]) ** 2)
	opposite_dist = np.sqrt((x - opposite_roi["cx"]) ** 2 + (y - opposite_roi["cy"]) ** 2)
	target_dist_mean = float(np.mean(target_dist))
	opposite_dist_mean = float(np.mean(opposite_dist))
	target_dist_median = float(np.median(target_dist))
	opposite_dist_median = float(np.median(opposite_dist))
	distance_pref = _safe_div(opposite_dist_mean - target_dist_mean, opposite_dist_mean + target_dist_mean)
	median_distance_pref = _safe_div(opposite_dist_median - target_dist_median, opposite_dist_median + target_dist_median)
	distance_row = {
		**base_meta,
		"target_label": cfg["target_label"],
		"opposite_label": cfg["opposite_label"],
		"target_side": row.get(cfg["target_side_col"]),
		"opposite_side": row.get(cfg["opposite_side_col"]),
		"valid_frames": valid_frames,
		"max_minutes": max_minutes if max_minutes > 0 else math.nan,
		"target_distance_mean_norm": target_dist_mean,
		"opposite_distance_mean_norm": opposite_dist_mean,
		"target_distance_median_norm": target_dist_median,
		"opposite_distance_median_norm": opposite_dist_median,
		"distance_delta_norm": opposite_dist_mean - target_dist_mean,
		"median_distance_delta_norm": opposite_dist_median - target_dist_median,
		"distance_preference_index": distance_pref,
		"median_distance_preference_index": median_distance_pref,
	}

	sensitivity_rows: list[dict[str, object]] = []
	for radius_scale in radius_scales:
		target_radius = target_roi["radius"] * radius_scale
		opposite_radius = opposite_roi["radius"] * radius_scale
		target_mask = target_dist <= target_radius
		opposite_mask = opposite_dist <= opposite_radius
		target_frames = int(np.sum(target_mask))
		opposite_frames = int(np.sum(opposite_mask))
		target_time = target_frames / fps
		opposite_time = opposite_frames / fps
		total_roi_time = target_time + opposite_time
		pref_index = _safe_div(target_time - opposite_time, total_roi_time)
		sensitivity_rows.append(
			{
				**base_meta,
				"target_label": cfg["target_label"],
				"opposite_label": cfg["opposite_label"],
				"target_side": row.get(cfg["target_side_col"]),
				"opposite_side": row.get(cfg["opposite_side_col"]),
				"valid_frames": valid_frames,
				"max_minutes": max_minutes if max_minutes > 0 else math.nan,
				"radius_scale": radius_scale,
				"target_radius_norm": target_radius,
				"opposite_radius_norm": opposite_radius,
				"target_frames": target_frames,
				"opposite_frames": opposite_frames,
				"target_time_s": target_time,
				"opposite_time_s": opposite_time,
				"target_visit_count": _count_visits(target_mask),
				"opposite_visit_count": _count_visits(opposite_mask),
				"target_pct_total": target_frames / valid_frames * 100.0,
				"opposite_pct_total": opposite_frames / valid_frames * 100.0,
				"preference_index": pref_index,
			}
		)
	return sensitivity_rows, distance_row, issues

def _aggregate_trial_means(session_df: pd.DataFrame, metric_kind: str) -> pd.DataFrame:
	if session_df.empty:
		return session_df.copy()
	group_cols = ["condition", "phase", "subject_id", "trial_id"]
	if metric_kind == "sensitivity":
		group_cols.append("radius_scale")
		numeric_cols = [
			"valid_frames",
			"target_frames",
			"opposite_frames",
			"target_time_s",
			"opposite_time_s",
			"target_visit_count",
			"opposite_visit_count",
			"target_pct_total",
			"opposite_pct_total",
			"preference_index",
			"target_radius_norm",
			"opposite_radius_norm",
		]
	else:
		numeric_cols = [
			"valid_frames",
			"target_distance_mean_norm",
			"opposite_distance_mean_norm",
			"target_distance_median_norm",
			"opposite_distance_median_norm",
			"distance_delta_norm",
			"median_distance_delta_norm",
			"distance_preference_index",
			"median_distance_preference_index",
		]
	agg_map: dict[str, str] = {col: "mean" for col in numeric_cols if col in session_df.columns}
	for col in ("sex", "keypoint", "fps", "max_minutes", "target_label", "opposite_label"):
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

def _build_sample_df(session_df: pd.DataFrame, sample_unit: str, metric_kind: str) -> pd.DataFrame:
	if sample_unit == "trial_mean":
		return _aggregate_trial_means(session_df, metric_kind)
	sample_df = session_df.copy()
	sample_df["sessions_averaged"] = 1
	sample_df["session_n_values"] = sample_df["session_n"].map(lambda value: _normalize_session_n(value) or "")
	sample_df["sample_unit"] = "session"
	sample_df["sample_key"] = sample_df["session_key"].astype(str)
	return sample_df

def _summarize_group(sample_df: pd.DataFrame, metric_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
	rows = []
	for key, group in sample_df.groupby(group_cols, dropna=False):
		key_values = key if isinstance(key, tuple) else (key,)
		row = dict(zip(group_cols, key_values))
		for metric in metric_cols:
			vals = pd.to_numeric(group[metric], errors="coerce").dropna()
			row[f"{metric}_n"] = int(vals.size)
			row[f"{metric}_mean"] = float(vals.mean()) if vals.size else math.nan
			row[f"{metric}_sem"] = _sem(vals)
		rows.append(row)
	return pd.DataFrame(rows)

