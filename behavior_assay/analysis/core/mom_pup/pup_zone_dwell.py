from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from analysis.core.mom_pup.behavior_metrics import BehaviorConfig, build_position_table, identify_real_pups, load_session

CONDITIONS = ["Control", "VPA"]

COLORS = {"Control": "#304F78", "VPA": "#C4475B"}

ZONES = ["Pup zone", "Matched zone"]

@dataclass(frozen=True)
class DwellConfig:
	pnd: int = 10
	pup_core_bin_mm: float = 5.0
	pup_core_search_radius_mm: float = 50.0
	entry_radius_mm: float = 60.0
	exit_radius_mm: float = 70.0
	min_visit_duration_sec: float = 0.5
	bridge_gap_sec: float = 1.0
	matched_grid_step_mm: float = 10.0
	matched_area_resolution_mm: float = 2.0
	matched_min_separation_mm: float = 90.0
	n_matched_zones: int = 3

def _as_bool(series: pd.Series) -> pd.Series:
	return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

def _find_runs(mask: np.ndarray, min_length: int = 1) -> list[tuple[int, int]]:
	padded = np.r_[False, mask.astype(bool), False]
	changes = np.flatnonzero(padded[1:] != padded[:-1])
	return [
		(int(start), int(end - 1))
		for start, end in zip(changes[::2], changes[1::2])
		if end - start >= int(min_length)
	]

def _bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
	output = mask.astype(bool).copy()
	if max_gap <= 0:
		return output
	for start, end in _find_runs(~output):
		length = end - start + 1
		if (
			length <= max_gap
			and start > 0
			and end < len(output) - 1
			and output[start - 1]
			and output[end + 1]
		):
			output[start : end + 1] = True
	return output

def _pup_core_center(
	positions: dict[str, pd.DataFrame],
	pup_ids: list[str],
	*,
	width_mm: float,
	depth_mm: float,
	config: DwellConfig,
) -> tuple[tuple[float, float], int, float]:
	x_values = np.concatenate(
		[
			positions[pup]["x_mm"].to_numpy(dtype=float)
			for pup in pup_ids
		]
	)
	y_values = np.concatenate(
		[
			positions[pup]["y_mm"].to_numpy(dtype=float)
			for pup in pup_ids
		]
	)
	valid = np.isfinite(x_values) & np.isfinite(y_values)
	x_values = x_values[valid]
	y_values = y_values[valid]
	if not len(x_values):
		raise ValueError("no valid pup positions")

	n_x_bins = max(1, int(round(width_mm / config.pup_core_bin_mm)))
	n_y_bins = max(1, int(round(depth_mm / config.pup_core_bin_mm)))
	histogram, x_edges, y_edges = np.histogram2d(
		x_values,
		y_values,
		bins=(n_x_bins, n_y_bins),
		range=((0.0, width_mm), (0.0, depth_mm)),
	)
	peak_x_index, peak_y_index = np.unravel_index(
		int(np.argmax(histogram)),
		histogram.shape,
	)
	peak_x = float(
		(x_edges[peak_x_index] + x_edges[peak_x_index + 1]) / 2.0
	)
	peak_y = float(
		(y_edges[peak_y_index] + y_edges[peak_y_index + 1]) / 2.0
	)
	core = np.hypot(x_values - peak_x, y_values - peak_y) <= (
		config.pup_core_search_radius_mm
	)
	if not core.any():
		raise ValueError("no pup positions within the core search radius")
	center = (
		float(np.median(x_values[core])),
		float(np.median(y_values[core])),
	)
	return center, int(core.sum()), float(core.mean())

def _arena_sample_points(
	width_mm: float,
	depth_mm: float,
	resolution_mm: float,
) -> tuple[np.ndarray, np.ndarray, float]:
	x = np.arange(
		resolution_mm / 2.0,
		width_mm,
		resolution_mm,
		dtype=float,
	)
	y = np.arange(
		resolution_mm / 2.0,
		depth_mm,
		resolution_mm,
		dtype=float,
	)
	grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
	return grid_x.ravel(), grid_y.ravel(), resolution_mm**2

def _accessible_circle_area(
	center: tuple[float, float],
	*,
	radius_mm: float,
	sample_x: np.ndarray,
	sample_y: np.ndarray,
	cell_area_mm2: float,
) -> float:
	inside = (
		(sample_x - center[0]) ** 2
		+ (sample_y - center[1]) ** 2
	) <= radius_mm**2
	return float(inside.sum() * cell_area_mm2)

def _wall_profile(
	center: tuple[float, float],
	width_mm: float,
	depth_mm: float,
) -> np.ndarray:
	x, y = center
	return np.sort(
		np.asarray(
			[x, width_mm - x, y, depth_mm - y],
			dtype=float,
		)
	)[:2]

def _matched_centers(
	pup_center: tuple[float, float],
	*,
	width_mm: float,
	depth_mm: float,
	config: DwellConfig,
) -> tuple[list[tuple[float, float]], float, list[float]]:
	sample_x, sample_y, cell_area = _arena_sample_points(
		width_mm,
		depth_mm,
		config.matched_area_resolution_mm,
	)
	pup_area = _accessible_circle_area(
		pup_center,
		radius_mm=config.entry_radius_mm,
		sample_x=sample_x,
		sample_y=sample_y,
		cell_area_mm2=cell_area,
	)
	pup_wall = _wall_profile(pup_center, width_mm, depth_mm)
	candidate_x = np.arange(
		0.0,
		width_mm + config.matched_grid_step_mm / 2.0,
		config.matched_grid_step_mm,
	)
	candidate_y = np.arange(
		0.0,
		depth_mm + config.matched_grid_step_mm / 2.0,
		config.matched_grid_step_mm,
	)
	candidates: list[tuple[float, float, float, float, float]] = []
	for x in candidate_x:
		for y in candidate_y:
			center = (float(x), float(y))
			center_distance = float(
				np.hypot(x - pup_center[0], y - pup_center[1])
			)
			if center_distance < config.matched_min_separation_mm:
				continue
			area = _accessible_circle_area(
				center,
				radius_mm=config.entry_radius_mm,
				sample_x=sample_x,
				sample_y=sample_y,
				cell_area_mm2=cell_area,
			)
			relative_area_difference = abs(area - pup_area) / max(pup_area, 1.0)
			wall_difference = float(
				np.mean(
					np.abs(
						_wall_profile(center, width_mm, depth_mm)
						- pup_wall
					)
				)
			) / config.entry_radius_mm
			score = 4.0 * relative_area_difference + wall_difference
			candidates.append(
				(
					score,
					relative_area_difference,
					-center_distance,
					x,
					y,
				)
			)
	if not candidates:
		raise ValueError("no matched-zone candidates")

	selected: list[tuple[float, float]] = []
	selected_areas: list[float] = []
	area_matched_candidates = [
		candidate
		for candidate in candidates
		if candidate[1] <= 0.05
	]
	selection_pool = area_matched_candidates or candidates
	for _, _, _, x, y in sorted(selection_pool):
		center = (float(x), float(y))
		if any(
			np.hypot(center[0] - other[0], center[1] - other[1])
			< config.matched_min_separation_mm
			for other in selected
		):
			continue
		selected.append(center)
		selected_areas.append(
			_accessible_circle_area(
				center,
				radius_mm=config.entry_radius_mm,
				sample_x=sample_x,
				sample_y=sample_y,
				cell_area_mm2=cell_area,
			)
		)
		if len(selected) >= config.n_matched_zones:
			break
	if not selected:
		raise ValueError("no non-overlapping matched zone selected")
	return selected, pup_area, selected_areas

def _zone_visits(
	mom_x: np.ndarray,
	mom_y: np.ndarray,
	center: tuple[float, float],
	*,
	fps: float,
	config: DwellConfig,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
	distance = np.hypot(mom_x - center[0], mom_y - center[1])
	valid = np.isfinite(distance)
	inside = np.zeros(len(distance), dtype=bool)
	state = False
	for frame_index, value in enumerate(distance):
		if not np.isfinite(value):
			state = False
		elif not state and value <= config.entry_radius_mm:
			state = True
		elif state and value >= config.exit_radius_mm:
			state = False
		inside[frame_index] = state
	inside = _bridge_short_gaps(
		inside,
		max(0, int(round(config.bridge_gap_sec * fps))),
	)
	runs = _find_runs(
		inside,
		max(1, int(round(config.min_visit_duration_sec * fps))),
	)
	return valid, runs

def _visit_summary(
	runs: list[tuple[int, int]],
	*,
	fps: float,
	recording_frames: int,
) -> dict[str, float | int]:
	durations = np.asarray(
		[(end - start + 1) / fps for start, end in runs],
		dtype=float,
	)
	total_time = float(durations.sum())
	return {
		"n_visits": int(len(durations)),
		"total_dwell_time_sec": total_time,
		"occupancy_pct": (
			100.0 * total_time / (recording_frames / fps)
			if recording_frames
			else np.nan
		),
		"mean_visit_duration_sec": (
			float(durations.mean()) if len(durations) else np.nan
		),
		"median_visit_duration_sec": (
			float(np.median(durations)) if len(durations) else np.nan
		),
		"max_visit_duration_sec": (
			float(durations.max()) if len(durations) else np.nan
		),
	}

def process_session(
	row: dict[str, object],
	behavior_config: BehaviorConfig,
	dwell_config: DwellConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
	df, _ = load_session(row)
	pup_ids, total_frames = identify_real_pups(df, behavior_config)
	if "mom" not in df["track"].unique():
		raise ValueError("mother track missing after normalization")
	if len(pup_ids) < 2:
		raise ValueError(f"fewer than two real pup tracks ({len(pup_ids)})")

	fps = float(row["fps"])
	width_mm = float(row["cage_width_mm"])
	depth_mm = float(row["cage_depth_mm"])
	positions = build_position_table(
		df,
		["mom", *pup_ids],
		total_frames,
		width_mm=width_mm,
		depth_mm=depth_mm,
		score_threshold=behavior_config.score_threshold,
	)
	pup_center, n_core_observations, core_observation_fraction = (
		_pup_core_center(
			positions,
			pup_ids,
			width_mm=width_mm,
			depth_mm=depth_mm,
			config=dwell_config,
		)
	)
	matched_centers, pup_area, matched_areas = _matched_centers(
		pup_center,
		width_mm=width_mm,
		depth_mm=depth_mm,
		config=dwell_config,
	)

	mom_x = positions["mom"]["x_mm"].to_numpy(dtype=float)
	mom_y = positions["mom"]["y_mm"].to_numpy(dtype=float)
	base = {
		"session_id": row["session_id"],
		"subject_id": row["subject_id"],
		"condition_code": row["condition_code"],
		"condition": row["condition"],
		"pnd": int(row["pnd"]),
		"source_file": row["file_name"],
		"fps": fps,
		"pup_center_x_mm": pup_center[0],
		"pup_center_y_mm": pup_center[1],
		"n_real_pups": len(pup_ids),
		"n_pup_core_observations": n_core_observations,
		"pup_core_observation_fraction": core_observation_fraction,
		"entry_radius_mm": dwell_config.entry_radius_mm,
		"exit_radius_mm": dwell_config.exit_radius_mm,
	}
	metric_rows: list[dict[str, object]] = []
	event_rows: list[dict[str, object]] = []

	valid, pup_runs = _zone_visits(
		mom_x,
		mom_y,
		pup_center,
		fps=fps,
		config=dwell_config,
	)
	pup_summary = _visit_summary(
		pup_runs,
		fps=fps,
		recording_frames=total_frames,
	)
	metric_rows.append(
		{
			**base,
			"zone": "Pup zone",
			"n_reference_zones": 1,
			"accessible_area_mm2": pup_area,
			**pup_summary,
		}
	)
	for visit_index, (start, end) in enumerate(pup_runs, start=1):
		event_rows.append(
			{
				**base,
				"zone": "Pup zone",
				"zone_index": 0,
				"visit_index": visit_index,
				"start_frame": start,
				"end_frame": end,
				"duration_sec": (end - start + 1) / fps,
			}
		)

	matched_summaries: list[dict[str, float | int]] = []
	all_matched_durations: list[float] = []
	for zone_index, center in enumerate(matched_centers, start=1):
		valid, runs = _zone_visits(
			mom_x,
			mom_y,
			center,
			fps=fps,
			config=dwell_config,
		)
		matched_summaries.append(
			_visit_summary(
				runs,
				fps=fps,
				recording_frames=total_frames,
			)
		)
		for visit_index, (start, end) in enumerate(runs, start=1):
			duration = (end - start + 1) / fps
			all_matched_durations.append(duration)
			event_rows.append(
				{
					**base,
					"zone": "Matched zone",
					"zone_index": zone_index,
					"zone_center_x_mm": center[0],
					"zone_center_y_mm": center[1],
					"visit_index": visit_index,
					"start_frame": start,
					"end_frame": end,
					"duration_sec": duration,
				}
			)
	matched_duration_array = np.asarray(all_matched_durations, dtype=float)
	metric_rows.append(
		{
			**base,
			"zone": "Matched zone",
			"n_reference_zones": len(matched_centers),
			"accessible_area_mm2": float(np.mean(matched_areas)),
			"n_visits": float(
				np.mean([item["n_visits"] for item in matched_summaries])
			),
			"total_dwell_time_sec": float(
				np.mean(
					[item["total_dwell_time_sec"] for item in matched_summaries]
				)
			),
			"occupancy_pct": float(
				np.mean([item["occupancy_pct"] for item in matched_summaries])
			),
			"mean_visit_duration_sec": (
				float(matched_duration_array.mean())
				if len(matched_duration_array)
				else np.nan
			),
			"median_visit_duration_sec": (
				float(np.median(matched_duration_array))
				if len(matched_duration_array)
				else np.nan
			),
			"max_visit_duration_sec": (
				float(matched_duration_array.max())
				if len(matched_duration_array)
				else np.nan
			),
		}
	)
	qc = {
		**base,
		"matched_centers": ";".join(
			f"{center[0]:.1f},{center[1]:.1f}"
			for center in matched_centers
		),
		"n_matched_zones": len(matched_centers),
		"pup_zone_accessible_area_mm2": pup_area,
		"matched_zone_mean_accessible_area_mm2": float(
			np.mean(matched_areas)
		),
		"matched_zone_area_difference_pct": (
			100.0 * (float(np.mean(matched_areas)) - pup_area) / pup_area
		),
	}
	return metric_rows, event_rows, qc

def _exact_permutation_p(
	control: np.ndarray,
	vpa: np.ndarray,
) -> float:
	values = np.r_[control, vpa]
	n_control = len(control)
	observed = float(control.mean() - vpa.mean())
	differences: list[float] = []
	indices = np.arange(len(values))
	for selected in combinations(indices, n_control):
		mask = np.zeros(len(values), dtype=bool)
		mask[list(selected)] = True
		differences.append(
			float(values[mask].mean() - values[~mask].mean())
		)
	differences_array = np.asarray(differences)
	return float(
		np.mean(np.abs(differences_array) >= abs(observed))
	)

def interaction_tests(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	index_columns = ["session_id", "subject_id", "condition"]
	contrasts = metrics[index_columns].drop_duplicates().copy()
	rows: list[dict[str, object]] = []
	for metric in (
		"occupancy_pct",
		"mean_visit_duration_sec",
		"n_visits",
	):
		wide = (
			metrics.pivot_table(
				index=index_columns,
				columns="zone",
				values=metric,
				aggfunc="first",
			)
			.reset_index()
			.dropna(subset=ZONES)
		)
		safe_metric = metric.removesuffix("_sec")
		pup_column = f"pup_zone_{safe_metric}"
		matched_column = f"matched_zone_{safe_metric}"
		contrast_column = f"pup_minus_matched_{safe_metric}"
		wide = wide.rename(
			columns={
				"Pup zone": pup_column,
				"Matched zone": matched_column,
			}
		)
		wide[contrast_column] = (
			wide[pup_column] - wide[matched_column]
		)
		contrasts = contrasts.merge(
			wide[
				index_columns
				+ [pup_column, matched_column, contrast_column]
			],
			on=index_columns,
			how="left",
		)
		control = pd.to_numeric(
			wide.loc[
				wide["condition"] == "Control",
				contrast_column,
			],
			errors="coerce",
		).dropna().to_numpy()
		vpa = pd.to_numeric(
			wide.loc[
				wide["condition"] == "VPA",
				contrast_column,
			],
			errors="coerce",
		).dropna().to_numpy()
		if metric == "mean_visit_duration_sec":
			analysis_control = np.log1p(
				wide.loc[
					wide["condition"] == "Control",
					pup_column,
				].to_numpy(dtype=float)
			) - np.log1p(
				wide.loc[
					wide["condition"] == "Control",
					matched_column,
				].to_numpy(dtype=float)
			)
			analysis_vpa = np.log1p(
				wide.loc[
					wide["condition"] == "VPA",
					pup_column,
				].to_numpy(dtype=float)
			) - np.log1p(
				wide.loc[
					wide["condition"] == "VPA",
					matched_column,
				].to_numpy(dtype=float)
			)
			analysis_scale = "log1p"
		else:
			analysis_control = control
			analysis_vpa = vpa
			analysis_scale = "raw"
		if len(control) >= 2 and len(vpa) >= 2:
			welch_t, p_welch = stats.ttest_ind(
				control,
				vpa,
				equal_var=False,
			)
			student_t, p_student = stats.ttest_ind(
				control,
				vpa,
				equal_var=True,
			)
			p_permutation = _exact_permutation_p(control, vpa)
			_, p_welch_analysis_scale = stats.ttest_ind(
				analysis_control,
				analysis_vpa,
				equal_var=False,
			)
		else:
			welch_t = p_welch = student_t = p_student = p_permutation = np.nan
			p_welch_analysis_scale = np.nan
		rows.append(
			{
				"pnd": int(metrics["pnd"].iloc[0]),
				"metric": metric,
				"contrast": "(Pup zone - Matched zone): Control vs VPA",
				"n_control": len(control),
				"n_vpa": len(vpa),
				"mean_contrast_control": (
					float(control.mean()) if len(control) else np.nan
				),
				"mean_contrast_vpa": (
					float(vpa.mean()) if len(vpa) else np.nan
				),
				"interaction_control_minus_vpa": (
					float(control.mean() - vpa.mean())
					if len(control) and len(vpa)
					else np.nan
				),
				"welch_t": welch_t,
				"p_welch_two_sided": p_welch,
				"student_t": student_t,
				"p_student_two_sided": p_student,
				"p_exact_permutation_two_sided": p_permutation,
				"analysis_scale": analysis_scale,
				"p_welch_analysis_scale_two_sided": p_welch_analysis_scale,
			}
		)
	return pd.DataFrame(rows), contrasts

def zone_tests(metrics: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for metric in (
		"occupancy_pct",
		"mean_visit_duration_sec",
		"n_visits",
	):
		for zone in ZONES:
			subset = metrics[metrics["zone"] == zone]
			control = pd.to_numeric(
				subset.loc[
					subset["condition"] == "Control",
					metric,
				],
				errors="coerce",
			).dropna()
			vpa = pd.to_numeric(
				subset.loc[
					subset["condition"] == "VPA",
					metric,
				],
				errors="coerce",
			).dropna()
			if len(control) >= 2 and len(vpa) >= 2:
				t_stat, p_value = stats.ttest_ind(
					control,
					vpa,
					equal_var=False,
				)
				_, p_mann_whitney = stats.mannwhitneyu(
					control,
					vpa,
					alternative="two-sided",
				)
				if metric == "mean_visit_duration_sec":
					_, p_welch_analysis_scale = stats.ttest_ind(
						np.log1p(control),
						np.log1p(vpa),
						equal_var=False,
					)
					analysis_scale = "log1p"
				else:
					p_welch_analysis_scale = p_value
					analysis_scale = "raw"
			else:
				t_stat = p_value = np.nan
				p_mann_whitney = p_welch_analysis_scale = np.nan
				analysis_scale = (
					"log1p"
					if metric == "mean_visit_duration_sec"
					else "raw"
				)
			rows.append(
				{
					"metric": metric,
					"zone": zone,
					"n_control": len(control),
					"n_vpa": len(vpa),
					"mean_control": control.mean(),
					"mean_vpa": vpa.mean(),
					"welch_t": t_stat,
					"p_welch_two_sided": p_value,
					"p_mann_whitney_two_sided": p_mann_whitney,
					"analysis_scale": analysis_scale,
					"p_welch_analysis_scale_two_sided": (
						p_welch_analysis_scale
					),
				}
			)
	return pd.DataFrame(rows)
