from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from analysis.core.mom_pup.behavior_metrics import BehaviorConfig, _as_bool, _find_runs, build_position_table, identify_real_pups, load_session

PNDS = [10, 15, 20]

CONDITIONS = ["Control", "VPA"]

COLORS = {"Control": "#304F78", "VPA": "#C4475B"}

METRIC = "pct_time_body_scale_proximity"

def pose_length_mm(
	df: pd.DataFrame,
	tracks: list[str],
	*,
	width_mm: float,
	depth_mm: float,
	score_threshold: float,
) -> float:
	subset = df[df["track"].isin(tracks)].copy()
	nose_score = pd.to_numeric(subset["Nose.score"], errors="coerce")
	tail_score = pd.to_numeric(subset["Tail.score"], errors="coerce")
	valid = (
		(nose_score >= float(score_threshold))
		& (tail_score >= float(score_threshold))
	)
	dx = (
		pd.to_numeric(subset.loc[valid, "Nose.x"], errors="coerce")
		- pd.to_numeric(subset.loc[valid, "Tail.x"], errors="coerce")
	) * float(width_mm)
	dy = (
		pd.to_numeric(subset.loc[valid, "Nose.y"], errors="coerce")
		- pd.to_numeric(subset.loc[valid, "Tail.y"], errors="coerce")
	) * float(depth_mm)
	length = np.hypot(dx, dy)
	length = length[np.isfinite(length) & (length > 5.0) & (length < 150.0)]
	return float(np.median(length)) if len(length) else np.nan

def process_session(
	row: dict[str, object],
	config: BehaviorConfig,
	*,
	min_bout_duration_sec: float = 1.0,
) -> dict[str, object]:
	df, duplicate_rows = load_session(row)
	pups, total_frames = identify_real_pups(df, config)
	if "mom" not in df["track"].unique():
		raise ValueError("mother track missing after normalization")
	if not pups:
		raise ValueError("no valid pup tracks")

	width_mm = float(row["cage_width_mm"])
	depth_mm = float(row["cage_depth_mm"])
	fps = float(row["fps"])
	positions = build_position_table(
		df,
		["mom", *pups],
		total_frames,
		width_mm=width_mm,
		depth_mm=depth_mm,
		score_threshold=config.score_threshold,
	)

	mom_x = positions["mom"]["x_mm"].to_numpy()
	mom_y = positions["mom"]["y_mm"].to_numpy()
	pup_x = np.column_stack(
		[positions[pup]["x_mm"].to_numpy() for pup in pups]
	)
	pup_y = np.column_stack(
		[positions[pup]["y_mm"].to_numpy() for pup in pups]
	)
	distance = np.hypot(
		pup_x - mom_x[:, np.newaxis],
		pup_y - mom_y[:, np.newaxis],
	)
	pup_present = np.isfinite(distance)
	nearest_distance = np.min(
		np.where(pup_present, distance, np.inf),
		axis=1,
	)
	valid = (
		np.isfinite(mom_x)
		& np.isfinite(mom_y)
		& np.isfinite(nearest_distance)
		& (nearest_distance < np.inf)
	)

	mom_length_mm = pose_length_mm(
		df,
		["mom"],
		width_mm=width_mm,
		depth_mm=depth_mm,
		score_threshold=config.score_threshold,
	)
	pup_length_mm = pose_length_mm(
		df,
		pups,
		width_mm=width_mm,
		depth_mm=depth_mm,
		score_threshold=config.score_threshold,
	)
	if not np.isfinite(mom_length_mm) or not np.isfinite(pup_length_mm):
		raise ValueError("insufficient valid Nose-Tail pose for body-scale threshold")
	if not valid.any():
		raise ValueError("no frames with valid mother and pup positions")

	proximity_threshold_mm = 0.5 * (mom_length_mm + pup_length_mm)
	close = valid & (nearest_distance <= proximity_threshold_mm)
	min_bout_frames = max(1, int(round(float(min_bout_duration_sec) * fps)))
	proximity_bouts = _find_runs(close, min_bout_frames)
	bout_durations_sec = np.asarray(
		[(end - start + 1) / fps for start, end in proximity_bouts],
		dtype=float,
	)
	valid_nearest = nearest_distance[valid]
	return {
		"session_id": row["session_id"],
		"subject_id": row["subject_id"],
		"condition_code": row["condition_code"],
		"condition": row["condition"],
		"pnd": int(row["pnd"]),
		"source_file": row["file_name"],
		"fps": fps,
		"cage_width_mm": width_mm,
		"cage_depth_mm": depth_mm,
		"score_threshold": config.score_threshold,
		"n_pups_detected": len(pups),
		"duplicate_rows_removed": duplicate_rows,
		"mom_body_length_mm": mom_length_mm,
		"pup_body_length_mm": pup_length_mm,
		"body_scale_threshold_mm": proximity_threshold_mm,
		"n_valid_frames": int(valid.sum()),
		"valid_frame_pct": 100.0 * float(valid.mean()),
		"mean_valid_pups_per_frame": float(
			pup_present[valid].sum(axis=1).mean()
		),
		"mean_nearest_pup_distance_mm": float(valid_nearest.mean()),
		"median_nearest_pup_distance_mm": float(np.median(valid_nearest)),
		"n_body_scale_proximity_frames": int(close.sum()),
		"body_scale_proximity_time_sec": float(close.sum()) / fps,
		"proximity_bout_min_duration_sec": float(min_bout_duration_sec),
		"n_body_scale_proximity_bouts": int(len(proximity_bouts)),
		"body_scale_proximity_bout_time_sec": float(bout_durations_sec.sum()),
		"pct_time_sustained_body_scale_proximity": (
			100.0
			* float(bout_durations_sec.sum())
			* fps
			/ float(valid.sum())
		),
		"avg_body_scale_proximity_bout_duration_sec": (
			float(bout_durations_sec.mean()) if len(bout_durations_sec) else np.nan
		),
		METRIC: 100.0 * float(close.sum()) / float(valid.sum()),
	}

def exact_permutation_p(control: np.ndarray, vpa: np.ndarray) -> float:
	pooled = np.concatenate([control, vpa])
	n_control = len(control)
	observed = float(control.mean() - vpa.mean())
	n_extreme = 0
	n_total = 0
	for indices in itertools.combinations(range(len(pooled)), n_control):
		mask = np.zeros(len(pooled), dtype=bool)
		mask[list(indices)] = True
		difference = float(pooled[mask].mean() - pooled[~mask].mean())
		n_extreme += abs(difference) >= abs(observed) - 1e-12
		n_total += 1
	return float(n_extreme / n_total) if n_total else np.nan

def hedges_g(control: np.ndarray, vpa: np.ndarray) -> float:
	n_control = len(control)
	n_vpa = len(vpa)
	if n_control < 2 or n_vpa < 2:
		return np.nan
	pooled_variance = (
		(n_control - 1) * np.var(control, ddof=1)
		+ (n_vpa - 1) * np.var(vpa, ddof=1)
	) / (n_control + n_vpa - 2)
	if pooled_variance <= 0:
		return np.nan
	cohens_d = (control.mean() - vpa.mean()) / np.sqrt(pooled_variance)
	correction = 1.0 - 3.0 / (4.0 * (n_control + n_vpa) - 9.0)
	return float(correction * cohens_d)

def group_summary(metrics: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for pnd in PNDS:
		for condition in CONDITIONS:
			values = pd.to_numeric(
				metrics.loc[
					(metrics["pnd"] == pnd)
					& (metrics["condition"] == condition),
					METRIC,
				],
				errors="coerce",
			).dropna()
			rows.append(
				{
					"pnd": pnd,
					"condition": condition,
					"metric": METRIC,
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

def between_group_tests(metrics: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for pnd in PNDS:
		subset = metrics[metrics["pnd"] == pnd]
		control = pd.to_numeric(
			subset.loc[subset["condition"] == "Control", METRIC],
			errors="coerce",
		).dropna().to_numpy()
		vpa = pd.to_numeric(
			subset.loc[subset["condition"] == "VPA", METRIC],
			errors="coerce",
		).dropna().to_numpy()
		if len(control) >= 2 and len(vpa) >= 2:
			welch = stats.ttest_ind(control, vpa, equal_var=False)
			student = stats.ttest_ind(control, vpa, equal_var=True)
			mann_whitney = stats.mannwhitneyu(
				control,
				vpa,
				alternative="two-sided",
			)
			p_permutation = exact_permutation_p(control, vpa)
		else:
			welch = student = mann_whitney = None
			p_permutation = np.nan
		rows.append(
			{
				"pnd": pnd,
				"metric": METRIC,
				"n_control": len(control),
				"n_vpa": len(vpa),
				"mean_control": float(np.mean(control)) if len(control) else np.nan,
				"mean_vpa": float(np.mean(vpa)) if len(vpa) else np.nan,
				"mean_difference_control_minus_vpa": (
					float(np.mean(control) - np.mean(vpa))
					if len(control) and len(vpa)
					else np.nan
				),
				"hedges_g_control_minus_vpa": hedges_g(control, vpa),
				"welch_t": welch.statistic if welch is not None else np.nan,
				"welch_df": welch.df if welch is not None else np.nan,
				"p_welch_two_sided": (
					welch.pvalue if welch is not None else np.nan
				),
				"student_t": (
					student.statistic if student is not None else np.nan
				),
				"p_student_two_sided": (
					student.pvalue if student is not None else np.nan
				),
				"mann_whitney_u": (
					mann_whitney.statistic
					if mann_whitney is not None
					else np.nan
				),
				"p_mann_whitney_two_sided": (
					mann_whitney.pvalue
					if mann_whitney is not None
					else np.nan
				),
				"p_exact_permutation_two_sided": p_permutation,
			}
		)

	results = pd.DataFrame(rows)
	valid = results["p_welch_two_sided"].notna()
	results["p_welch_holm_across_pnd"] = np.nan
	if valid.any():
		results.loc[valid, "p_welch_holm_across_pnd"] = multipletests(
			results.loc[valid, "p_welch_two_sided"],
			method="holm",
		)[1]
	return results
