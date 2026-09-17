from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from analysis.core.mom_pup.behavior_metrics import BehaviorConfig, build_position_table, identify_initial_cluster, identify_real_pups, load_session

PNDS = [10, 15, 20]

CONDITIONS = ["Control", "VPA"]

ZONES = ["Near", "Outside"]

COLORS = {"Control": "#304F78", "VPA": "#C4475B"}

def _as_bool(series: pd.Series) -> pd.Series:
	return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

def _cluster_centroid(
	positions: dict[str, pd.DataFrame],
	cluster_ids: set[str],
) -> tuple[np.ndarray, np.ndarray]:
	cluster = sorted(cluster_ids)
	x = np.column_stack([positions[pup]["x_mm"].to_numpy() for pup in cluster])
	y = np.column_stack([positions[pup]["y_mm"].to_numpy() for pup in cluster])
	present = np.isfinite(x) & np.isfinite(y)
	with np.errstate(invalid="ignore"):
		centroid_x = np.nanmean(x, axis=1)
		centroid_y = np.nanmean(y, axis=1)
	any_present = present.any(axis=1)
	centroid_x[~any_present] = np.nan
	centroid_y[~any_present] = np.nan
	return centroid_x, centroid_y

def _zone_row(
	*,
	base: dict[str, object],
	zone: str,
	threshold_mm: float,
	mask: np.ndarray,
	step_distance: np.ndarray,
	step_speed: np.ndarray,
	fps: float,
	config: BehaviorConfig,
) -> dict[str, object]:
	n_steps = int(mask.sum())
	time_sec = n_steps / float(fps)
	path_length = float(step_distance[mask].sum())
	return {
		**base,
		"threshold_mm": float(threshold_mm),
		"zone": zone,
		"n_valid_steps": n_steps,
		"time_sec": round(time_sec, 3),
		"path_length_mm": round(path_length, 3),
		"movement_rate_mm_s": round(path_length / time_sec, 3) if time_sec else np.nan,
		"pct_stationary": round(
			100 * np.mean(step_speed[mask] <= config.stationary_velocity_mm_s),
			3,
		)
		if n_steps
		else np.nan,
	}

def process_session(
	row: dict[str, object],
	config: BehaviorConfig,
	thresholds_mm: list[float],
) -> tuple[list[dict[str, object]], list[str]]:
	df, _ = load_session(row)
	pups, total_frames = identify_real_pups(df, config)
	notes: list[str] = []
	if "mom" not in df["track"].unique():
		return [], ["mother track missing after normalization"]
	if len(pups) < 2:
		return [], [f"fewer than two real pup tracks ({len(pups)})"]

	fps = float(row["fps"])
	positions = build_position_table(
		df,
		["mom", *pups],
		total_frames,
		width_mm=float(row["cage_width_mm"]),
		depth_mm=float(row["cage_depth_mm"]),
		score_threshold=config.score_threshold,
	)
	initial_cluster, n_votes = identify_initial_cluster(
		positions,
		pups,
		total_frames,
		config,
	)
	if initial_cluster is None:
		return [], ["no initial pup cluster found"]

	centroid_x, centroid_y = _cluster_centroid(positions, initial_cluster)
	mom_x = positions["mom"]["x_mm"].to_numpy()
	mom_y = positions["mom"]["y_mm"].to_numpy()
	frame_distance = np.hypot(mom_x - centroid_x, mom_y - centroid_y)

	step_distance = np.hypot(np.diff(mom_x), np.diff(mom_y))
	step_speed = step_distance * fps
	valid_step = (
		np.isfinite(step_distance)
		& (step_distance <= config.max_plausible_displacement_mm)
		& np.isfinite(frame_distance[:-1])
		& np.isfinite(frame_distance[1:])
	)
	step_cluster_distance = (frame_distance[:-1] + frame_distance[1:]) / 2.0

	base = {
		"session_id": row["session_id"],
		"subject_id": row["subject_id"],
		"condition_code": row["condition_code"],
		"condition": row["condition"],
		"pnd": int(row["pnd"]),
		"source_file": row["file_name"],
		"n_cluster_pups": len(initial_cluster),
		"initial_cluster_ids": ",".join(sorted(initial_cluster)),
		"n_init_votes_used": n_votes,
		"fps": fps,
	}
	rows: list[dict[str, object]] = []
	for threshold_mm in thresholds_mm:
		near = valid_step & (step_cluster_distance <= float(threshold_mm))
		outside = valid_step & (step_cluster_distance > float(threshold_mm))
		rows.append(
			_zone_row(
				base=base,
				zone="Near",
				threshold_mm=threshold_mm,
				mask=near,
				step_distance=step_distance,
				step_speed=step_speed,
				fps=fps,
				config=config,
			)
		)
		rows.append(
			_zone_row(
				base=base,
				zone="Outside",
				threshold_mm=threshold_mm,
				mask=outside,
				step_distance=step_distance,
				step_speed=step_speed,
				fps=fps,
				config=config,
			)
		)
	return rows, notes

def _adjust_holm(
	df: pd.DataFrame,
	*,
	group_columns: list[str],
	p_column: str = "p_value",
) -> pd.DataFrame:
	output = df.copy()
	output["p_holm"] = np.nan
	for _, indices in output.groupby(group_columns, dropna=False).groups.items():
		group_index = list(indices)
		valid_index = [
			index
			for index in group_index
			if pd.notna(output.at[index, p_column])
		]
		if valid_index:
			output.loc[valid_index, "p_holm"] = multipletests(
				output.loc[valid_index, p_column],
				method="holm",
			)[1]
	return output

def interaction_tests(metrics: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for threshold_mm in sorted(metrics["threshold_mm"].unique()):
		for pnd in PNDS:
			subset = metrics[
				(metrics["threshold_mm"] == threshold_mm)
				& (metrics["pnd"] == pnd)
			]
			wide = (
				subset.pivot_table(
					index=["session_id", "subject_id", "condition"],
					columns="zone",
					values="movement_rate_mm_s",
					aggfunc="first",
				)
				.reset_index()
				.dropna(subset=ZONES)
			)
			wide["near_minus_outside_mm_s"] = wide["Near"] - wide["Outside"]
			control = wide.loc[
				wide["condition"] == "Control",
				"near_minus_outside_mm_s",
			]
			vpa = wide.loc[
				wide["condition"] == "VPA",
				"near_minus_outside_mm_s",
			]
			if len(control) >= 2 and len(vpa) >= 2:
				t_stat, p_value = stats.ttest_ind(vpa, control, equal_var=False)
			else:
				t_stat = p_value = np.nan
			rows.append(
				{
					"threshold_mm": threshold_mm,
					"pnd": pnd,
					"n_control": len(control),
					"n_vpa": len(vpa),
					"mean_near_minus_outside_control_mm_s": control.mean(),
					"mean_near_minus_outside_vpa_mm_s": vpa.mean(),
					"interaction_difference_vpa_minus_control_mm_s": (
						vpa.mean() - control.mean()
					),
					"welch_t": t_stat,
					"p_value": p_value,
				}
			)
	return _adjust_holm(
		pd.DataFrame(rows),
		group_columns=["threshold_mm"],
	)

def within_group_tests(metrics: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for threshold_mm in sorted(metrics["threshold_mm"].unique()):
		for condition in CONDITIONS:
			for pnd in PNDS:
				subset = metrics[
					(metrics["threshold_mm"] == threshold_mm)
					& (metrics["condition"] == condition)
					& (metrics["pnd"] == pnd)
				]
				wide = (
					subset.pivot_table(
						index=["session_id", "subject_id"],
						columns="zone",
						values="movement_rate_mm_s",
						aggfunc="first",
					)
					.dropna(subset=ZONES)
				)
				if len(wide) >= 2:
					t_stat, p_value = stats.ttest_rel(wide["Near"], wide["Outside"])
				else:
					t_stat = p_value = np.nan
				rows.append(
					{
						"threshold_mm": threshold_mm,
						"condition": condition,
						"pnd": pnd,
						"n": len(wide),
						"mean_near_mm_s": wide["Near"].mean(),
						"mean_outside_mm_s": wide["Outside"].mean(),
						"mean_near_minus_outside_mm_s": (
							wide["Near"] - wide["Outside"]
						).mean(),
						"paired_t": t_stat,
						"p_value": p_value,
					}
				)
	return _adjust_holm(
		pd.DataFrame(rows),
		group_columns=["threshold_mm", "condition"],
	)

def between_group_tests(metrics: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for threshold_mm in sorted(metrics["threshold_mm"].unique()):
		for zone in ZONES:
			for pnd in PNDS:
				subset = metrics[
					(metrics["threshold_mm"] == threshold_mm)
					& (metrics["zone"] == zone)
					& (metrics["pnd"] == pnd)
				]
				control = pd.to_numeric(
					subset.loc[
						subset["condition"] == "Control",
						"movement_rate_mm_s",
					],
					errors="coerce",
				).dropna()
				vpa = pd.to_numeric(
					subset.loc[
						subset["condition"] == "VPA",
						"movement_rate_mm_s",
					],
					errors="coerce",
				).dropna()
				if len(control) >= 2 and len(vpa) >= 2:
					t_stat, p_value = stats.ttest_ind(vpa, control, equal_var=False)
				else:
					t_stat = p_value = np.nan
				rows.append(
					{
						"threshold_mm": threshold_mm,
						"zone": zone,
						"pnd": pnd,
						"n_control": len(control),
						"n_vpa": len(vpa),
						"mean_control_mm_s": control.mean(),
						"mean_vpa_mm_s": vpa.mean(),
						"welch_t": t_stat,
						"p_value": p_value,
					}
				)
	return _adjust_holm(
		pd.DataFrame(rows),
		group_columns=["threshold_mm", "zone"],
	)

def group_summary(metrics: pd.DataFrame) -> pd.DataFrame:
	grouped = metrics.groupby(
		["threshold_mm", "pnd", "condition", "zone"],
		sort=True,
	)["movement_rate_mm_s"]
	return grouped.agg(
		n="count",
		mean="mean",
		sd="std",
		sem="sem",
		median="median",
	).reset_index()

def add_zone_occupancy(metrics: pd.DataFrame) -> pd.DataFrame:
	output = metrics.copy()
	output["valid_zone_time_sec"] = output.groupby(
		["session_id", "threshold_mm"],
	)["time_sec"].transform("sum")
	output["zone_occupancy_pct"] = np.where(
		output["valid_zone_time_sec"] > 0,
		100 * output["time_sec"] / output["valid_zone_time_sec"],
		np.nan,
	)
	return output

def occupancy_outputs(
	metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	occupancy = metrics.copy()
	occupancy["movement_rate_mm_s"] = pd.to_numeric(
		occupancy["zone_occupancy_pct"],
		errors="coerce",
	)
	between = between_group_tests(occupancy).rename(
		columns={
			"mean_control_mm_s": "mean_control_pct",
			"mean_vpa_mm_s": "mean_vpa_pct",
		}
	)
	summary = group_summary(occupancy)
	return between, summary

def stationary_outputs(
	metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	stationary = metrics.copy()
	stationary["movement_rate_mm_s"] = pd.to_numeric(
		stationary["pct_stationary"],
		errors="coerce",
	)
	interaction = interaction_tests(stationary).rename(
		columns={
			"mean_near_minus_outside_control_mm_s": "mean_near_minus_outside_control_pct_points",
			"mean_near_minus_outside_vpa_mm_s": "mean_near_minus_outside_vpa_pct_points",
			"interaction_difference_vpa_minus_control_mm_s": (
				"interaction_difference_vpa_minus_control_pct_points"
			),
		}
	)
	within = within_group_tests(stationary).rename(
		columns={
			"mean_near_mm_s": "mean_near_pct",
			"mean_outside_mm_s": "mean_outside_pct",
			"mean_near_minus_outside_mm_s": "mean_near_minus_outside_pct_points",
		}
	)
	between = between_group_tests(stationary).rename(
		columns={
			"mean_control_mm_s": "mean_control_pct",
			"mean_vpa_mm_s": "mean_vpa_pct",
		}
	)
	summary = group_summary(stationary)
	return interaction, within, between, summary
