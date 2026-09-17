from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import pandas as pd
from data_loader.processing.deduplicate import deduplicate_frame_track
from data_loader.processing.interpolate import interpolate_keypoints, pad_frame_track_grid

KEYPOINTS = ("Body_C", "Nose")

def _as_bool(series: pd.Series) -> pd.Series:
	return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

def preprocess_one(
	raw_path: Path,
	*,
	frame_start: int,
	frame_end: int,
	score_threshold: float,
	max_gap: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
	raw = pd.read_csv(raw_path)
	raw_rows = len(raw)
	duplicate_rows = int(raw.duplicated(["frame_idx", "track"]).sum())
	work = deduplicate_frame_track(raw, policy="highest_instance_score")
	work["frame_idx"] = pd.to_numeric(work["frame_idx"], errors="coerce").astype("Int64")
	work["track"] = work["track"].astype(str)

	masked_counts: dict[str, int] = {}
	for keypoint in KEYPOINTS:
		xcol, ycol, scol = f"{keypoint}.x", f"{keypoint}.y", f"{keypoint}.score"
		if xcol not in work or ycol not in work:
			raise ValueError(f"Missing required keypoint columns for {keypoint}")
		x = pd.to_numeric(work[xcol], errors="coerce")
		y = pd.to_numeric(work[ycol], errors="coerce")
		score = pd.to_numeric(work.get(scol), errors="coerce")
		low_score = score < float(score_threshold)
		if keypoint == "Body_C":
			out_of_bounds = (x < 0) | (x > 1) | (y < 0) | (y > 1)
		else:
			# The nose may physically extend slightly beyond a floor-calibrated
			# arena boundary. Only reject implausibly distant coordinates.
			out_of_bounds = (x < -0.15) | (x > 1.15) | (y < -0.15) | (y > 1.15)
		masked_counts[f"{keypoint}_low_score_rows"] = int(low_score.fillna(False).sum())
		masked_counts[f"{keypoint}_out_of_bounds_rows"] = int(out_of_bounds.fillna(False).sum())
		work[xcol] = x.mask(low_score | out_of_bounds)
		work[ycol] = y.mask(low_score | out_of_bounds)

	tracks = sorted(work["track"].dropna().unique().tolist())
	frames = list(range(int(frame_start), int(frame_end) + 1))
	padded = pad_frame_track_grid(work, tracks=tracks, frames=frames)
	inserted_rows = len(padded) - len(work)
	filled, interp_summary = interpolate_keypoints(
		padded,
		keypoints=KEYPOINTS,
		method="linear",
		max_gap=int(max_gap),
		score_threshold=float(score_threshold),
	)
	filled = filled.sort_values(["frame_idx", "track"]).reset_index(drop=True)

	body_valid = (
		pd.to_numeric(filled["Body_C.x"], errors="coerce").notna()
		& pd.to_numeric(filled["Body_C.y"], errors="coerce").notna()
	)
	nose_valid = (
		pd.to_numeric(filled["Nose.x"], errors="coerce").notna()
		& pd.to_numeric(filled["Nose.y"], errors="coerce").notna()
	)
	summary = {
		"raw_rows": raw_rows,
		"duplicate_rows_removed": duplicate_rows,
		"padded_rows_inserted": inserted_rows,
		"output_rows": len(filled),
		"track_count": len(tracks),
		"body_valid_frames": int(body_valid.sum()),
		"body_coverage_pct": round(100 * body_valid.mean(), 3) if len(filled) else 0.0,
		"nose_valid_frames": int(nose_valid.sum()),
		"nose_coverage_pct": round(100 * nose_valid.mean(), 3) if len(filled) else 0.0,
		**masked_counts,
		**interp_summary,
	}
	return filled, summary
