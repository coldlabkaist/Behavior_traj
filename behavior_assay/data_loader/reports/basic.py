from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def per_frame_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Returns DataFrame with columns:
	- frame_idx
	- num_tracks_present
	- num_tracks_expected (max tracks observed in file)
	- missing_tracks (num_tracks_expected - num_tracks_present)
	- duplicate_tracks (count of tracks appearing more than once within the frame)
	"""
	if "frame_idx" not in df.columns or "track" not in df.columns:
		raise ValueError("DataFrame must contain 'frame_idx' and 'track'")

	# Determine expected track set as those that appear at least once
	all_tracks = df["track"].astype(str).unique().tolist()
	expected = len(all_tracks)

	grp = df.groupby("frame_idx")
	rows = []
	for frame, g in grp:
		tracks = g["track"].astype(str)
		n_present = tracks.nunique()
		missing = max(expected - n_present, 0)
		# Duplicate count: total rows - nunique tracks
		duplicates = max(len(tracks) - n_present, 0)
		rows.append({
			"frame_idx": int(frame),
			"num_tracks_present": int(n_present),
			"num_tracks_expected": int(expected),
			"missing_tracks": int(missing),
			"duplicate_tracks": int(duplicates),
		})
	return pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)


def summarize_file_level(df: pd.DataFrame) -> Dict[str, int]:
	pf = per_frame_missing_and_duplicates(df)
	missing_frames = int((pf["missing_tracks"] > 0).sum() if not pf.empty else 0)
	duplicate_frames = int((pf["duplicate_tracks"] > 0).sum() if not pf.empty else 0)
	return {
		"frames": int(len(pf)),
		"tracks_expected": int(pf["num_tracks_expected"].max() if not pf.empty else 0),
		"missing_frames": missing_frames,
		"duplicate_frames": duplicate_frames,
	}


def per_track_missing_and_duplicate_frames(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Per-track breakdown of frame counts:
	- missing_frames: number of frames where this track is absent
	- duplicate_frames: number of frames where this track appears more than once
	"""
	if "frame_idx" not in df.columns or "track" not in df.columns:
		raise ValueError("DataFrame must contain 'frame_idx' and 'track'")

	df = df.copy()
	df["track"] = df["track"].astype(str)
	frames_total = df["frame_idx"].nunique()

	# Frames where track present at least once
	present_frames = df.groupby("track")["frame_idx"].nunique()

	# Frames where track count > 1
	dup_flags = (
		df.groupby(["track", "frame_idx"]).size().reset_index(name="count")
		.pipe(lambda d: d[d["count"] > 1])
	)
	dup_frames_by_track = dup_flags.groupby("track")["frame_idx"].nunique()

	tracks = df["track"].unique().tolist()
	rows = []
	for t in tracks:
		present = int(present_frames.get(t, 0))
		missing_frames = int(max(frames_total - present, 0))
		duplicate_frames = int(dup_frames_by_track.get(t, 0))
		rows.append({
			"track": t,
			"missing_frames": missing_frames,
			"duplicate_frames": duplicate_frames,
		})
	return pd.DataFrame(rows).sort_values("track").reset_index(drop=True)
