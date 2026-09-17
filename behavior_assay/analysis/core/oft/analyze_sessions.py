from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import math
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

METRICS = {
	"time_centre_s": "Time in centre (s)",
	"time_periphery_s": "Time in periphery (s)",
	"pct_centre": "Valid tracked time in centre (%)",
	"centre_entries": "Centre entries",
	"total_distance_cm": "Total distance (cm)",
	"distance_centre_cm": "Distance in centre (cm)",
	"mean_velocity_cm_s": "Mean velocity during valid steps (cm/s)",
}

def _window(df: pd.DataFrame, frames: int) -> pd.DataFrame:
	frame_values = pd.to_numeric(df["frame_idx"], errors="coerce")
	start = int(frame_values.min())
	return df[(frame_values >= start) & (frame_values < start + frames)].sort_values("frame_idx").copy()

def _count_centre_entries(
	body_x: np.ndarray,
	body_y: np.ndarray,
	nose_x: np.ndarray,
	nose_y: np.ndarray,
	*,
	centre_low: float,
	centre_high: float,
	entry_buffer: float,
) -> int:
	entries = 0
	inside = False
	for bx, by, nx, ny in zip(body_x, body_y, nose_x, nose_y):
		if not np.all(np.isfinite([bx, by])):
			inside = False
			continue
		deep_inside = (
			centre_low + entry_buffer <= bx <= centre_high - entry_buffer
			and centre_low + entry_buffer <= by <= centre_high - entry_buffer
		)
		deep_outside = not (
			centre_low - entry_buffer <= bx <= centre_high + entry_buffer
			and centre_low - entry_buffer <= by <= centre_high + entry_buffer
		)
		if not inside and deep_inside and np.all(np.isfinite([nx, ny])):
			if centre_low <= nx <= centre_high and centre_low <= ny <= centre_high:
				inside = True
				entries += 1
		elif inside and deep_outside:
			inside = False
	return entries

def analyze_session(
	df: pd.DataFrame,
	*,
	fps: float,
	arena_cm: float,
	analysis_seconds: float,
	centre_fraction: float,
	entry_buffer: float,
	max_step_cm: float,
	smoothing_window: int,
) -> dict[str, object]:
	if df["track"].astype(str).nunique() != 1:
		raise ValueError("OFT analysis requires exactly one track per session")
	n_frames_requested = int(round(float(fps) * float(analysis_seconds)))
	work = _window(df, n_frames_requested)
	bx = pd.to_numeric(work["Body_C.x"], errors="coerce").to_numpy(dtype=float)
	by = pd.to_numeric(work["Body_C.y"], errors="coerce").to_numpy(dtype=float)
	nx = pd.to_numeric(work["Nose.x"], errors="coerce").to_numpy(dtype=float)
	ny = pd.to_numeric(work["Nose.y"], errors="coerce").to_numpy(dtype=float)

	valid_body = np.isfinite(bx) & np.isfinite(by)
	if smoothing_window > 1:
		bx_smoothed = (
			pd.Series(bx)
			.rolling(smoothing_window, center=True, min_periods=1)
			.median()
			.to_numpy(dtype=float)
		)
		by_smoothed = (
			pd.Series(by)
			.rolling(smoothing_window, center=True, min_periods=1)
			.median()
			.to_numpy(dtype=float)
		)
		bx = np.where(valid_body, bx_smoothed, np.nan)
		by = np.where(valid_body, by_smoothed, np.nan)
	centre_low = (1.0 - float(centre_fraction)) / 2.0
	centre_high = 1.0 - centre_low
	centre = (
		valid_body
		& (bx >= centre_low)
		& (bx <= centre_high)
		& (by >= centre_low)
		& (by <= centre_high)
	)
	periphery = valid_body & ~centre

	dx = np.diff(bx) * float(arena_cm)
	dy = np.diff(by) * float(arena_cm)
	step_distance = np.hypot(dx, dy)
	consecutive_frames = np.diff(pd.to_numeric(work["frame_idx"], errors="coerce").to_numpy(dtype=float)) == 1
	valid_steps = np.isfinite(step_distance) & consecutive_frames & (step_distance <= float(max_step_cm))
	rejected_large_steps = np.isfinite(step_distance) & consecutive_frames & (step_distance > float(max_step_cm))
	total_distance = float(step_distance[valid_steps].sum())
	step_mid_x = (bx[:-1] + bx[1:]) / 2.0
	step_mid_y = (by[:-1] + by[1:]) / 2.0
	centre_steps = (
		valid_steps
		& (step_mid_x >= centre_low)
		& (step_mid_x <= centre_high)
		& (step_mid_y >= centre_low)
		& (step_mid_y <= centre_high)
	)
	distance_centre = float(step_distance[centre_steps].sum())
	valid_step_time = float(valid_steps.sum()) / float(fps)

	valid_frames = int(valid_body.sum())
	return {
		"requested_duration_s": float(analysis_seconds),
		"frames_in_window": len(work),
		"elapsed_duration_s": round(len(work) / float(fps), 3),
		"valid_body_frames": valid_frames,
		"tracking_coverage_pct": round(100 * valid_frames / len(work), 3) if len(work) else np.nan,
		"untracked_time_s": round((len(work) - valid_frames) / float(fps), 3),
		"time_centre_s": round(float(centre.sum()) / float(fps), 3),
		"time_periphery_s": round(float(periphery.sum()) / float(fps), 3),
		"pct_centre": round(100 * float(centre.sum()) / valid_frames, 3) if valid_frames else np.nan,
		"centre_entries": _count_centre_entries(
			bx,
			by,
			nx,
			ny,
			centre_low=centre_low,
			centre_high=centre_high,
			entry_buffer=float(entry_buffer),
		),
		"total_distance_cm": round(total_distance, 3),
		"distance_centre_cm": round(distance_centre, 3),
		"mean_velocity_cm_s": round(total_distance / valid_step_time, 3) if valid_step_time else np.nan,
		"valid_movement_steps": int(valid_steps.sum()),
		"rejected_large_steps": int(rejected_large_steps.sum()),
	}

def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
	if len(a) < 2 or len(b) < 2:
		return np.nan
	var_a = np.var(a, ddof=1)
	var_b = np.var(b, ddof=1)
	pooled_denom = len(a) + len(b) - 2
	if pooled_denom <= 0:
		return np.nan
	pooled = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / pooled_denom)
	if pooled == 0:
		return np.nan
	d = (np.mean(b) - np.mean(a)) / pooled
	correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
	return float(d * correction)

def _test_table(df: pd.DataFrame, *, unit: str) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for metric in METRICS:
		control = pd.to_numeric(df.loc[df["condition"] == "Control", metric], errors="coerce").dropna().to_numpy()
		vpa = pd.to_numeric(df.loc[df["condition"] == "VPA", metric], errors="coerce").dropna().to_numpy()
		if len(control) >= 2 and len(vpa) >= 2:
			t_stat, p_value = stats.ttest_ind(vpa, control, equal_var=False)
			levene_f, levene_p = stats.levene(control, vpa, center="median")
		else:
			t_stat = p_value = levene_f = levene_p = np.nan
		rows.append(
			{
				"analysis_unit": unit,
				"metric": metric,
				"n_control": len(control),
				"n_vpa": len(vpa),
				"mean_control": np.mean(control) if len(control) else np.nan,
				"mean_vpa": np.mean(vpa) if len(vpa) else np.nan,
				"mean_difference_vpa_minus_control": np.mean(vpa) - np.mean(control) if len(control) and len(vpa) else np.nan,
				"welch_t": t_stat,
				"p_value": p_value,
				"hedges_g": _hedges_g(control, vpa),
				"brown_forsythe_f": levene_f,
				"brown_forsythe_p": levene_p,
			}
		)
	out = pd.DataFrame(rows)
	valid = out["p_value"].notna()
	out["p_value_fdr_bh"] = np.nan
	if valid.any():
		out.loc[valid, "p_value_fdr_bh"] = multipletests(out.loc[valid, "p_value"], method="fdr_bh")[1]
	return out

def _group_summary(df: pd.DataFrame, *, unit: str) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for condition, group in df.groupby("condition"):
		for metric in METRICS:
			values = pd.to_numeric(group[metric], errors="coerce").dropna()
			rows.append(
				{
					"analysis_unit": unit,
					"condition": condition,
					"metric": metric,
					"n": len(values),
					"mean": values.mean(),
					"sd": values.std(ddof=1),
					"sem": values.sem(),
					"median": values.median(),
					"q25": values.quantile(0.25),
					"q75": values.quantile(0.75),
				}
			)
	return pd.DataFrame(rows)
