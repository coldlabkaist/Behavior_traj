from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from data_loader.processing.deduplicate import deduplicate_frame_track

@dataclass(frozen=True)
class BehaviorConfig:
	score_threshold: float = 0.5
	min_track_coverage_frac: float = 0.02
	min_track_frames_abs: int = 90
	cluster_threshold_mm: float = 50.0
	mom_close_threshold_mm: float = 60.0
	stray_threshold_mm: float = 80.0
	stray_min_duration_frames: int = 30
	initial_vote_frames: int = 30
	initial_search_frames: int = 3000
	max_plausible_displacement_mm: float = 50.0
	stationary_velocity_mm_s: float = 10.0
	stable_proximity_speed_window_sec: float = 1.0
	stable_proximity_min_duration_sec: float = 3.0
	bout_min_duration_frames: int = 15
	approach_search_limit_sec: float = 120.0

def _as_bool(series: pd.Series) -> pd.Series:
	return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

def load_session(row: dict[str, object]) -> tuple[pd.DataFrame, int]:
	raw = pd.read_csv(Path(str(row["raw_path"])))
	duplicate_rows = int(raw.duplicated(["track", "frame_idx"]).sum())
	raw = deduplicate_frame_track(raw, policy="highest_instance_score")
	raw["track"] = raw["track"].astype(str)
	mother_raw = str(row["mother_track_raw"])
	raw["track"] = raw["track"].replace({mother_raw: "mom"})
	frame = pd.to_numeric(raw["frame_idx"], errors="coerce")
	frame_start = int(frame.min())
	raw["frame_idx"] = (frame - frame_start).astype("Int64")
	return raw, duplicate_rows

def identify_real_pups(df: pd.DataFrame, config: BehaviorConfig) -> tuple[list[str], int]:
	total_frames = int(pd.to_numeric(df["frame_idx"], errors="coerce").max()) + 1
	counts = df.groupby("track")["frame_idx"].nunique()
	threshold = max(config.min_track_frames_abs, config.min_track_coverage_frac * total_frames)
	real = counts[counts >= threshold].index.astype(str).tolist()
	pups = sorted(
		(track for track in real if track != "mom"),
		key=lambda value: (
			0,
			int("".join(char for char in value if char.isdigit())),
		)
		if any(char.isdigit() for char in value)
		else (1, value),
	)
	return pups, total_frames

def build_position_table(
	df: pd.DataFrame,
	tracks: list[str],
	total_frames: int,
	*,
	width_mm: float,
	depth_mm: float,
	score_threshold: float,
) -> dict[str, pd.DataFrame]:
	full_index = pd.RangeIndex(0, total_frames)
	output: dict[str, pd.DataFrame] = {}
	for track in tracks:
		sub = (
			df[df["track"] == track]
			.set_index("frame_idx")[["Body_C.x", "Body_C.y", "Body_C.score"]]
			.reindex(full_index)
		)
		score = pd.to_numeric(sub["Body_C.score"], errors="coerce")
		good = score >= float(score_threshold)
		x = pd.to_numeric(sub["Body_C.x"], errors="coerce")
		y = pd.to_numeric(sub["Body_C.y"], errors="coerce")
		output[track] = pd.DataFrame(
			{
				"x_mm": np.where(good, x * float(width_mm), np.nan),
				"y_mm": np.where(good, y * float(depth_mm), np.nan),
			},
			index=full_index,
		)
	return output

class UnionFind:
	def __init__(self, items: list[str]):
		self.parent = {item: item for item in items}

	def find(self, value: str) -> str:
		while self.parent[value] != value:
			self.parent[value] = self.parent[self.parent[value]]
			value = self.parent[value]
		return value

	def union(self, a: str, b: str) -> None:
		root_a, root_b = self.find(a), self.find(b)
		if root_a != root_b:
			self.parent[root_a] = root_b

def cluster_frame(frame_positions: dict[str, tuple[float, float]], threshold_mm: float) -> list[frozenset[str]]:
	ids = list(frame_positions)
	union_find = UnionFind(ids)
	for index, first in enumerate(ids):
		for second in ids[index + 1 :]:
			x1, y1 = frame_positions[first]
			x2, y2 = frame_positions[second]
			if np.hypot(x1 - x2, y1 - y2) <= float(threshold_mm):
				union_find.union(first, second)
	groups: dict[str, list[str]] = {}
	for item in ids:
		groups.setdefault(union_find.find(item), []).append(item)
	return [frozenset(group) for group in groups.values() if len(group) >= 2]

def frame_positions_at(
	positions: dict[str, pd.DataFrame],
	pup_ids: list[str],
	frame_idx: int,
) -> dict[str, tuple[float, float]]:
	output: dict[str, tuple[float, float]] = {}
	for pup in pup_ids:
		x = positions[pup]["x_mm"].iat[frame_idx]
		y = positions[pup]["y_mm"].iat[frame_idx]
		if np.isfinite(x) and np.isfinite(y):
			output[pup] = (float(x), float(y))
	return output

def identify_initial_cluster(
	positions: dict[str, pd.DataFrame],
	pup_ids: list[str],
	total_frames: int,
	config: BehaviorConfig,
) -> tuple[set[str] | None, int]:
	votes: list[frozenset[str]] = []
	for frame_idx in range(min(config.initial_search_frames, total_frames)):
		frame_positions = frame_positions_at(positions, pup_ids, frame_idx)
		if len(frame_positions) < 2:
			continue
		clusters = cluster_frame(frame_positions, config.cluster_threshold_mm)
		if not clusters:
			continue
		max_size = max(map(len, clusters))
		candidates = [cluster for cluster in clusters if len(cluster) == max_size]
		if len(candidates) > 1:
			mom_x = positions["mom"]["x_mm"].iat[frame_idx]
			mom_y = positions["mom"]["y_mm"].iat[frame_idx]
			if np.isfinite(mom_x) and np.isfinite(mom_y):
				candidates.sort(
					key=lambda cluster: np.hypot(
						np.mean([frame_positions[pup][0] for pup in cluster]) - mom_x,
						np.mean([frame_positions[pup][1] for pup in cluster]) - mom_y,
					)
				)
			else:
				candidates.sort(key=lambda cluster: sorted(cluster))
		votes.append(candidates[0])
		if len(votes) >= config.initial_vote_frames:
			break
	if not votes:
		return None, 0
	return set(Counter(votes).most_common(1)[0][0]), len(votes)

def _find_runs(mask: np.ndarray, min_length: int) -> list[tuple[int, int]]:
	padded = np.r_[False, mask.astype(bool), False]
	changes = np.flatnonzero(padded[1:] != padded[:-1])
	return [
		(int(start), int(end - 1))
		for start, end in zip(changes[::2], changes[1::2])
		if end - start >= int(min_length)
	]

def compute_mom_locomotion(
	positions: dict[str, pd.DataFrame],
	total_frames: int,
	*,
	fps: float,
	config: BehaviorConfig,
) -> dict[str, object]:
	x = positions["mom"]["x_mm"].to_numpy()
	y = positions["mom"]["y_mm"].to_numpy()
	step_distance = np.hypot(np.diff(x), np.diff(y))
	valid = np.isfinite(step_distance) & (step_distance <= config.max_plausible_displacement_mm)
	total_path = float(step_distance[valid].sum())
	valid_time = int(valid.sum()) / float(fps)
	step_velocity = step_distance * float(fps)
	stationary = valid & (step_velocity <= config.stationary_velocity_mm_s)
	active = valid & (step_velocity > config.stationary_velocity_mm_s)
	active_time_sec = int(active.sum()) / float(fps)
	active_path = float(step_distance[active].sum())
	return {
		"total_path_length_mm": round(total_path, 3),
		"avg_velocity_mm_s": round(total_path / valid_time, 3) if valid_time else np.nan,
		"pct_time_stationary": round(100 * stationary.sum() / valid.sum(), 3) if valid.any() else np.nan,
		"pct_time_active": round(100 * active.sum() / valid.sum(), 3) if valid.any() else np.nan,
		"active_time_sec": round(active_time_sec, 3),
		"avg_speed_while_moving_mm_s": round(active_path / active_time_sec, 3)
		if active_time_sec
		else np.nan,
		"n_valid_movement_steps": int(valid.sum()),
		"mom_movement_coverage_pct": round(100 * valid.sum() / max(total_frames - 1, 1), 3),
	}

def compute_full_clustering(
	positions: dict[str, pd.DataFrame],
	pup_ids: list[str],
	total_frames: int,
	config: BehaviorConfig,
) -> dict[str, object]:
	n_clusters: list[int] = []
	cluster_sizes: list[int] = []
	for frame_idx in range(total_frames):
		frame_positions = frame_positions_at(positions, pup_ids, frame_idx)
		if not frame_positions:
			continue
		clusters = cluster_frame(frame_positions, config.cluster_threshold_mm)
		n_clusters.append(len(clusters))
		cluster_sizes.extend(map(len, clusters))
	return {
		"n_valid_frames_clustering": len(n_clusters),
		"avg_n_clusters_per_frame": round(float(np.mean(n_clusters)), 4) if n_clusters else np.nan,
		"max_simultaneous_clusters": int(np.max(n_clusters)) if n_clusters else np.nan,
		"pct_frames_fragmented": round(100 * np.mean(np.asarray(n_clusters) >= 2), 3) if n_clusters else np.nan,
		"avg_pups_per_cluster": round(float(np.mean(cluster_sizes)), 4) if cluster_sizes else np.nan,
	}

def compute_mom_cluster(
	positions: dict[str, pd.DataFrame],
	cluster_ids: set[str],
	total_frames: int,
	*,
	fps: float,
	config: BehaviorConfig,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
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
	mom_x = positions["mom"]["x_mm"].to_numpy()
	mom_y = positions["mom"]["y_mm"].to_numpy()
	distance = np.hypot(mom_x - centroid_x, mom_y - centroid_y)
	valid = np.isfinite(distance)
	close = valid & (distance <= config.mom_close_threshold_mm)
	mom_dx = np.diff(mom_x)
	mom_dy = np.diff(mom_y)
	step_distance = np.hypot(mom_dx, mom_dy)
	step_valid = np.isfinite(step_distance) & (step_distance <= config.max_plausible_displacement_mm)
	frame_speed = np.full(total_frames, np.nan, dtype=float)
	frame_speed[1:] = np.where(step_valid, step_distance * float(fps), np.nan)
	speed_window_frames = max(
		1,
		int(round(config.stable_proximity_speed_window_sec * float(fps))),
	)
	median_speed = (
		pd.Series(frame_speed)
		.rolling(
			window=speed_window_frames,
			center=True,
			min_periods=max(1, speed_window_frames // 2),
		)
		.median()
		.to_numpy()
	)
	frame_stationary = np.isfinite(median_speed) & (
		median_speed <= config.stationary_velocity_mm_s
	)
	stable_proximity = valid & close & frame_stationary
	stable_runs = _find_runs(
		stable_proximity,
		max(1, int(round(config.stable_proximity_min_duration_sec * float(fps)))),
	)
	stable_durations = [
		(end - start + 1) / float(fps)
		for start, end in stable_runs
	]
	target_dx = centroid_x[:-1] - mom_x[:-1]
	target_dy = centroid_y[:-1] - mom_y[:-1]
	target_distance = np.hypot(target_dx, target_dy)
	active_far = (
		np.isfinite(step_distance)
		& np.isfinite(target_distance)
		& (step_distance <= config.max_plausible_displacement_mm)
		& (step_distance * float(fps) > config.stationary_velocity_mm_s)
		& (target_distance > config.mom_close_threshold_mm)
	)
	alignment = np.full(step_distance.shape, np.nan, dtype=float)
	alignment[active_far] = (
		mom_dx[active_far] * target_dx[active_far]
		+ mom_dy[active_far] * target_dy[active_far]
	) / (step_distance[active_far] * target_distance[active_far])
	toward_cluster = active_far & (alignment > 0)
	return (
		{
			"n_valid_frames_mom_cluster": int(valid.sum()),
			"pct_video_valid_mom_cluster": round(100 * valid.sum() / total_frames, 3),
			"avg_mom_dist_to_cluster_mm": round(float(np.nanmean(distance)), 3) if valid.any() else np.nan,
			"time_mom_close_to_cluster_sec": round(close.sum() / float(fps), 3),
			"pct_time_mom_close_to_cluster": round(100 * close.sum() / valid.sum(), 3) if valid.any() else np.nan,
			"total_stable_proximity_time_sec": round(float(sum(stable_durations)), 3),
			"n_stable_proximity_bouts": len(stable_runs),
			"avg_stable_proximity_bout_duration_sec": round(
				float(np.mean(stable_durations)),
				3,
			)
			if stable_durations
			else np.nan,
			"n_active_steps_far_from_cluster": int(active_far.sum()),
			"pct_active_steps_toward_cluster_when_far": round(
				100 * toward_cluster.sum() / active_far.sum(),
				3,
			)
			if active_far.any()
			else np.nan,
		},
		close,
		valid,
	)

def compute_approach_bouts(
	close: np.ndarray,
	valid: np.ndarray,
	*,
	fps: float,
	active_time_sec: float,
	config: BehaviorConfig,
) -> dict[str, object]:
	runs = _find_runs(valid & close, config.bout_min_duration_frames)
	durations = [(end - start + 1) / float(fps) for start, end in runs]
	active_minutes = float(active_time_sec) / 60.0
	return {
		"n_approach_bouts": len(runs),
		"avg_bout_duration_sec": round(float(np.mean(durations)), 3) if durations else np.nan,
		"approach_bouts_per_active_min": round(len(runs) / active_minutes, 3)
		if active_minutes > 0
		else np.nan,
	}

def compute_dispersion_events(
	positions: dict[str, pd.DataFrame],
	cluster_ids: set[str],
	*,
	fps: float,
	config: BehaviorConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
	cluster = sorted(cluster_ids)
	mom_x = positions["mom"]["x_mm"].to_numpy()
	mom_y = positions["mom"]["y_mm"].to_numpy()
	pup_xy = {
		pup: (positions[pup]["x_mm"].to_numpy(), positions[pup]["y_mm"].to_numpy())
		for pup in cluster
	}
	events: list[dict[str, object]] = []
	search_frames = int(round(config.approach_search_limit_sec * float(fps)))
	for pup in cluster:
		pup_x, pup_y = pup_xy[pup]
		others = [other for other in cluster if other != pup]
		if not others:
			continue
		other_x = np.column_stack([pup_xy[other][0] for other in others])
		other_y = np.column_stack([pup_xy[other][1] for other in others])
		with np.errstate(invalid="ignore"):
			ref_x = np.nanmean(other_x, axis=1)
			ref_y = np.nanmean(other_y, axis=1)
		other_present = np.isfinite(other_x) & np.isfinite(other_y)
		any_other = other_present.any(axis=1)
		ref_x[~any_other] = np.nan
		ref_y[~any_other] = np.nan
		distance_from_cluster = np.hypot(pup_x - ref_x, pup_y - ref_y)
		is_stray = (
			np.isfinite(pup_x)
			& np.isfinite(pup_y)
			& any_other
			& (distance_from_cluster > config.stray_threshold_mm)
		)
		for start, end in _find_runs(is_stray, config.stray_min_duration_frames):
			event_distance = np.hypot(mom_x[start : end + 1] - pup_x[start : end + 1], mom_y[start : end + 1] - pup_y[start : end + 1])
			event_valid = np.isfinite(event_distance)
			if not event_valid.any():
				continue
			close_frames = event_valid & (event_distance <= config.mom_close_threshold_mm)
			search_end = min(start + search_frames, len(mom_x))
			search_distance = np.hypot(
				mom_x[start:search_end] - pup_x[start:search_end],
				mom_y[start:search_end] - pup_y[start:search_end],
			)
			first_close = np.flatnonzero(np.isfinite(search_distance) & (search_distance <= config.mom_close_threshold_mm))
			latency = float(first_close[0]) / float(fps) if len(first_close) else np.nan
			events.append(
				{
					"pup_track": pup,
					"start_frame_relative": start,
					"end_frame_relative": end,
					"duration_sec": round((end - start + 1) / float(fps), 3),
					"avg_mom_dist_during_event_mm": round(float(np.nanmean(event_distance)), 3),
					"mom_close_time_during_event_sec": round(close_frames.sum() / float(fps), 3),
					"mom_first_approach_latency_sec": round(latency, 3) if np.isfinite(latency) else np.nan,
					"mom_approached_within_window": bool(np.isfinite(latency)),
				}
			)

	approach_latencies = [
		float(event["mom_first_approach_latency_sec"])
		for event in events
		if pd.notna(event["mom_first_approach_latency_sec"])
	]
	return (
		{
			"n_dispersion_events": len(events),
			"n_pups_with_dispersion": len({event["pup_track"] for event in events}),
			"avg_mom_dist_to_stray_pup_mm": round(
				float(np.mean([event["avg_mom_dist_during_event_mm"] for event in events])),
				3,
			)
			if events
			else np.nan,
			"total_mom_close_time_all_events_sec": round(
				float(sum(event["mom_close_time_during_event_sec"] for event in events)),
				3,
			),
			"total_dispersion_duration_sec": round(
				float(sum(event["duration_sec"] for event in events)),
				3,
			),
			"avg_mom_first_approach_latency_sec": round(float(np.mean(approach_latencies)), 3)
			if approach_latencies
			else np.nan,
			"mom_approach_success_rate_pct": round(100 * len(approach_latencies) / len(events), 3)
			if events
			else np.nan,
		},
		events,
	)

def process_session(row: dict[str, object], config: BehaviorConfig) -> tuple[dict[str, object], list[dict[str, object]], list[str]]:
	df, duplicate_rows = load_session(row)
	pups, total_frames = identify_real_pups(df, config)
	fps = float(row["fps"])
	width_mm = float(row["cage_width_mm"])
	depth_mm = float(row["cage_depth_mm"])
	notes: list[str] = []
	base = {
		"session_id": row["session_id"],
		"subject_id": row["subject_id"],
		"condition_code": row["condition_code"],
		"condition": row["condition"],
		"pnd": int(row["pnd"]),
		"source_file": row["file_name"],
		"frame_start_raw": int(row["frame_start"]),
		"frame_end_raw": int(row["frame_end"]),
		"total_frames": total_frames,
		"duration_sec": round(total_frames / fps, 3),
		"fps": fps,
		"cage_width_mm": width_mm,
		"cage_depth_mm": depth_mm,
		"score_threshold": config.score_threshold,
		"cluster_threshold_mm": config.cluster_threshold_mm,
		"mom_close_threshold_mm": config.mom_close_threshold_mm,
		"stray_threshold_mm": config.stray_threshold_mm,
		"stray_min_duration_sec": config.stray_min_duration_frames / fps,
		"approach_bout_min_duration_sec": config.bout_min_duration_frames / fps,
		"approach_search_limit_sec": config.approach_search_limit_sec,
		"stable_proximity_speed_window_sec": config.stable_proximity_speed_window_sec,
		"stable_proximity_min_duration_sec": config.stable_proximity_min_duration_sec,
		"n_pups_detected": len(pups),
		"pup_track_ids": ",".join(pups),
		"duplicate_rows_removed": duplicate_rows,
	}
	if "mom" not in df["track"].unique():
		notes.append("mother track missing after normalization")
		return base, [], notes

	mom_positions = build_position_table(
		df,
		["mom"],
		total_frames,
		width_mm=width_mm,
		depth_mm=depth_mm,
		score_threshold=config.score_threshold,
	)
	base.update(compute_mom_locomotion(mom_positions, total_frames, fps=fps, config=config))
	if len(pups) < 2:
		notes.append(f"fewer than two real pup tracks ({len(pups)})")
		return base, [], notes

	positions = build_position_table(
		df,
		["mom", *pups],
		total_frames,
		width_mm=width_mm,
		depth_mm=depth_mm,
		score_threshold=config.score_threshold,
	)
	base.update(compute_full_clustering(positions, pups, total_frames, config))
	initial_cluster, n_votes = identify_initial_cluster(positions, pups, total_frames, config)
	if initial_cluster is None:
		notes.append("no initial pup cluster found")
		base.update({"initial_cluster_ids": "", "n_cluster_pups": 0, "n_init_votes_used": 0})
		return base, [], notes

	base.update(
		{
			"initial_cluster_ids": ",".join(sorted(initial_cluster)),
			"n_cluster_pups": len(initial_cluster),
			"n_init_votes_used": n_votes,
		}
	)
	cluster_summary, close, valid = compute_mom_cluster(
		positions,
		initial_cluster,
		total_frames,
		fps=fps,
		config=config,
	)
	base.update(cluster_summary)
	base.update(
		compute_approach_bouts(
			close,
			valid,
			fps=fps,
			active_time_sec=float(base["active_time_sec"]),
			config=config,
		)
	)
	dispersion_summary, events = compute_dispersion_events(
		positions,
		initial_cluster,
		fps=fps,
		config=config,
	)
	base.update(dispersion_summary)
	for event in events:
		event.update(
			{
				"session_id": row["session_id"],
				"subject_id": row["subject_id"],
				"condition_code": row["condition_code"],
				"condition": row["condition"],
				"pnd": int(row["pnd"]),
				"start_frame_raw": int(row["frame_start"]) + int(event["start_frame_relative"]),
				"end_frame_raw": int(row["frame_start"]) + int(event["end_frame_relative"]),
			}
		)
	return base, events, notes
