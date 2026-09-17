from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd


def pad_frame_track_grid(df: pd.DataFrame, *, tracks: List[str] | None = None, frames: List[int] | None = None) -> pd.DataFrame:
	"""
	Create a dense grid of (frame_idx, track) rows, merging original data onto it.
	If tracks/frames are None, infer from the dataframe.
	"""
	if "frame_idx" not in df.columns or "track" not in df.columns:
		raise ValueError("DataFrame must contain 'frame_idx' and 'track'")

	work = df.copy()
	work["track"] = work["track"].astype(str)
	work["frame_idx"] = pd.to_numeric(work["frame_idx"], errors="coerce").astype("Int64")

	if frames is None:
		frames = sorted(int(f) for f in work["frame_idx"].dropna().unique())
	if tracks is None:
		tracks = sorted(work["track"].dropna().astype(str).unique().tolist())

	grid = pd.MultiIndex.from_product([frames, tracks], names=["frame_idx", "track"]).to_frame(index=False)
	padded = grid.merge(work, on=["frame_idx", "track"], how="left").sort_values(["frame_idx", "track"]).reset_index(drop=True)
	return padded


def _count_long_nan_runs(mask: np.ndarray, max_gap: int) -> int:
	"""
	Count contiguous True-runs in mask longer than max_gap.
	"""
	if max_gap is None or max_gap <= 0:
		return 0
	count = 0
	run = 0
	for v in mask:
		if v:
			run += 1
		else:
			if run > max_gap:
				count += 1
			run = 0
	# tail
	if run > max_gap:
		count += 1
	return count


def _rolling_mean_nan_aware(s: pd.Series, window: int) -> pd.Series:
	if window is None or window <= 1:
		return s
	return s.rolling(window=window, center=True, min_periods=1).mean()


def _kalman_body_c_track(
	x: pd.Series,
	y: pd.Series,
	score: pd.Series | None,
	score_threshold: float,
	q_pos: float,
	q_vel: float,
	r_base: float,
	max_predict: int,
	gate_mahalanobis_sq: float | None,
) -> Tuple[pd.Series, pd.Series, int, int]:
	"""
	Run a constant-velocity Kalman filter on one track.
	Measurement is (x,y). Missing if NaN or score < threshold.
	Prediction-only runs longer than max_predict are masked back to NaN.
	Returns: (x_filled, y_filled, filled_x_count, filled_y_count)
	"""
	T = len(x)
	# State: [x, y, vx, vy]
	F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
	H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
	Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(float)

	# Initialize with first available measurement
	x_out = np.full(T, np.nan, dtype=float)
	y_out = np.full(T, np.nan, dtype=float)

	z_x = x.to_numpy(dtype=float)
	z_y = y.to_numpy(dtype=float)
	s = score.to_numpy(dtype=float) if score is not None else np.full(T, 1.0, dtype=float)
	meas_ok = (~np.isnan(z_x)) & (~np.isnan(z_y)) & (s >= score_threshold)

	# Find first measurement to initialize
	if not meas_ok.any():
		return pd.Series(x_out, index=x.index), pd.Series(y_out, index=y.index), 0, 0
	first = int(np.argmax(meas_ok))
	state = np.array([z_x[first], z_y[first], 0.0, 0.0], dtype=float)
	P = np.eye(4, dtype=float)

	filled_x = 0
	filled_y = 0
	predict_run = 0

	for t in range(T):
		if t < first:
			continue
		# Predict
		state = F @ state
		P = F @ P @ F.T + Q

		if meas_ok[t]:
			# Measurement noise scales with inverse of confidence
			r_scale = 1.0 / max(s[t], 1e-6)
			R = (r_base * r_scale) * np.eye(2)
			z = np.array([z_x[t], z_y[t]])
			S = H @ P @ H.T + R
			K = P @ H.T @ np.linalg.inv(S)
			innov = z - (H @ state)
			# Optional: outlier gating (reject huge innovations)
			if gate_mahalanobis_sq is not None:
				try:
					d2 = float(innov.T @ np.linalg.inv(S) @ innov)
				except Exception:
					d2 = float("inf")
				if d2 > float(gate_mahalanobis_sq):
					# Treat as missing measurement: skip update
					predict_run += 1
				else:
					state = state + K @ innov
					P = (np.eye(4) - K @ H) @ P
					predict_run = 0
			else:
				state = state + K @ innov
				P = (np.eye(4) - K @ H) @ P
				predict_run = 0
		else:
			predict_run += 1

		x_pred, y_pred = state[0], state[1]
		if not meas_ok[t]:
			# Count as filled if we created a value where it was missing
			if np.isnan(z_x[t]) and not np.isnan(x_pred):
				filled_x += 1
			if np.isnan(z_y[t]) and not np.isnan(y_pred):
				filled_y += 1
			# Mask overly long predictions
			if predict_run > max_predict:
				x_pred = np.nan
				y_pred = np.nan

		x_out[t] = x_pred
		y_out[t] = y_pred

	return pd.Series(x_out, index=x.index), pd.Series(y_out, index=y.index), filled_x, filled_y


def interpolate_body_c(
	df: pd.DataFrame,
	*,
	method: str = "linear",  # "linear" | "kalman"
	max_gap: int = 5,
	smooth_window: int | None = None,
	score_threshold: float = 0.0,
	kalman_q_pos: float = 1e-3,
	kalman_q_vel: float = 1e-2,
	kalman_r_base: float = 1e-2,
	kalman_max_predict: int = 30,
	kalman_gate_mahalanobis_sq: float | None = None,
) -> tuple[pd.DataFrame, Dict[str, int]]:
	"""
	Pad grid (if needed externally) and fill Body_C.x/Body_C.y NaNs.
	method:
	- linear: fill interior gaps up to max_gap; no edge extrapolation
	- kalman: constant-velocity Kalman; score-aware R; limit long predict-only runs
	Returns (filled_df, summary_counts)
	"""
	return interpolate_keypoints(
		df,
		keypoints=["Body_C"],
		method=method,
		max_gap=max_gap,
		smooth_window=smooth_window,
		score_threshold=score_threshold,
		kalman_q_pos=kalman_q_pos,
		kalman_q_vel=kalman_q_vel,
		kalman_r_base=kalman_r_base,
		kalman_max_predict=kalman_max_predict,
		kalman_gate_mahalanobis_sq=kalman_gate_mahalanobis_sq,
	)


def interpolate_keypoints(
	df: pd.DataFrame,
	*,
	keypoints: Iterable[str],
	method: str = "linear",  # "linear" | "kalman"
	max_gap: int = 5,
	smooth_window: int | None = None,
	score_threshold: float = 0.0,
	kalman_q_pos: float = 1e-3,
	kalman_q_vel: float = 1e-2,
	kalman_r_base: float = 1e-2,
	kalman_max_predict: int = 30,
	kalman_gate_mahalanobis_sq: float | None = None,
) -> tuple[pd.DataFrame, Dict[str, int]]:
	"""
	Pad grid (if needed externally) and fill <keypoint>.x / <keypoint>.y NaNs.
	method:
	- linear: fill interior gaps up to max_gap; if max_gap <= 0, fill all interior gaps
	- kalman: constant-velocity Kalman; score-aware R; limit long predict-only runs
	Returns (filled_df, summary_counts)
	"""
	keypoint_list = [str(kp) for kp in keypoints]
	if not keypoint_list:
		raise ValueError("At least one keypoint must be provided")

	for kp in keypoint_list:
		for channel in ("x", "y"):
			col = f"{kp}.{channel}"
			if col not in df.columns:
				raise ValueError("Missing required column: %s" % col)

	# Ensure unique (frame, track)
	if df.duplicated(["frame_idx", "track"]).any():
		raise ValueError("DataFrame must be deduplicated to unique (frame_idx, track)")

	work = df.copy()

	long_gap_counts = {kp: 0 for kp in keypoint_list}
	filled_x_counts = {kp: 0 for kp in keypoint_list}
	filled_y_counts = {kp: 0 for kp in keypoint_list}
	interp_limit = None if max_gap is None or max_gap <= 0 else int(max_gap)

	def _interp_track_linear(g: pd.DataFrame) -> pd.DataFrame:
		g = g.sort_values("frame_idx").reset_index(drop=True)
		for kp in keypoint_list:
			sx = g[f"{kp}.x"].copy()
			sy = g[f"{kp}.y"].copy()
			long_gap_counts[kp] += _count_long_nan_runs(sx.isna().to_numpy(), max_gap)
			orig_na_x = sx.isna()
			orig_na_y = sy.isna()
			sx_i = sx.interpolate(method="linear", limit=interp_limit, limit_direction="both")
			sy_i = sy.interpolate(method="linear", limit=interp_limit, limit_direction="both")
			# Remove edge extrapolation
			edge_x = orig_na_x & (sx.ffill().isna() | sx.bfill().isna())
			edge_y = orig_na_y & (sy.ffill().isna() | sy.bfill().isna())
			sx_i.loc[edge_x] = np.nan
			sy_i.loc[edge_y] = np.nan
			filled_x_counts[kp] += int((orig_na_x & sx_i.notna()).sum())
			filled_y_counts[kp] += int((orig_na_y & sy_i.notna()).sum())
			# Optional smoothing
			sx_s = _rolling_mean_nan_aware(sx_i, smooth_window) if smooth_window and smooth_window > 1 else sx_i
			sy_s = _rolling_mean_nan_aware(sy_i, smooth_window) if smooth_window and smooth_window > 1 else sy_i
			g[f"{kp}.x"] = sx_s
			g[f"{kp}.y"] = sy_s
		return g

	def _interp_track_kalman(g: pd.DataFrame) -> pd.DataFrame:
		g = g.sort_values("frame_idx").reset_index(drop=True)
		for kp in keypoint_list:
			sx = g[f"{kp}.x"].copy()
			sy = g[f"{kp}.y"].copy()
			sc = g[f"{kp}.score"] if f"{kp}.score" in g.columns else None
			x_f, y_f, fx, fy = _kalman_body_c_track(
				sx, sy, sc, score_threshold,
				q_pos=kalman_q_pos, q_vel=kalman_q_vel, r_base=kalman_r_base, max_predict=kalman_max_predict,
				gate_mahalanobis_sq=kalman_gate_mahalanobis_sq,
			)
			filled_x_counts[kp] += fx
			filled_y_counts[kp] += fy
			g[f"{kp}.x"] = x_f
			g[f"{kp}.y"] = y_f
		return g

	if method == "kalman":
		out = work.groupby("track", group_keys=False).apply(_interp_track_kalman)
	else:
		out = work.groupby("track", group_keys=False).apply(_interp_track_linear)

	summary: Dict[str, int | str] = {"method": method}
	for kp in keypoint_list:
		summary[f"{kp}_filled_x"] = int(filled_x_counts[kp])
		summary[f"{kp}_filled_y"] = int(filled_y_counts[kp])
		summary[f"{kp}_long_gaps"] = int(long_gap_counts[kp])
		summary[f"{kp}_remaining_nan_x"] = int(out[f"{kp}.x"].isna().sum())
		summary[f"{kp}_remaining_nan_y"] = int(out[f"{kp}.y"].isna().sum())

	if keypoint_list == ["Body_C"]:
		summary["filled_x"] = int(summary["Body_C_filled_x"])
		summary["filled_y"] = int(summary["Body_C_filled_y"])
		summary["long_gaps"] = int(summary["Body_C_long_gaps"])
		summary["remaining_nan_x"] = int(summary["Body_C_remaining_nan_x"])
		summary["remaining_nan_y"] = int(summary["Body_C_remaining_nan_y"])
	return out, summary
