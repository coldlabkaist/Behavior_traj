from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.core.mom_pup.common import DEFAULT_OUTPUT_DIR, DEFAULT_RAW_DIR, parse_session_stem, to_abs_path

REQUIRED_COLUMNS = {
	"track",
	"frame_idx",
	"instance.score",
	"Body_C.x",
	"Body_C.y",
	"Body_C.score",
}

MOTHER_LABEL_CANDIDATES = ("mom", "dam", "mother", "track_0")

EXPECTED_PNDS = {10, 15, 20}

def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()

def _mother_track(tracks: list[str]) -> str:
	lower_to_actual = {track.lower(): track for track in tracks}
	for candidate in MOTHER_LABEL_CANDIDATES:
		if candidate in lower_to_actual:
			return lower_to_actual[candidate]
	return ""

def inspect_file(path: Path, *, fps: float, cage_width_mm: float, cage_depth_mm: float, calculate_hash: bool) -> tuple[dict[str, object], list[dict[str, str]]]:
	meta = parse_session_stem(path.stem)
	issues: list[dict[str, str]] = []
	try:
		df = pd.read_csv(path)
	except Exception as exc:
		return (
			{
				**meta,
				"file_name": path.name,
				"raw_path": str(path.resolve()),
				"include": False,
				"issues": "read_error",
			},
			[{"issue": "read_error", "detail": str(exc)}],
		)

	missing = sorted(REQUIRED_COLUMNS - set(df.columns))
	if missing:
		issues.append({"issue": "missing_columns", "detail": ",".join(missing)})
	tracks = sorted(df["track"].dropna().astype(str).unique().tolist()) if "track" in df else []
	mother_track = _mother_track(tracks)
	if not bool(meta["metadata_ok"]):
		issues.append({"issue": "metadata_parse_failed", "detail": path.stem})
	if not mother_track:
		issues.append({"issue": "mother_track_not_found", "detail": ",".join(tracks)})
	if len(tracks) < 3:
		issues.append({"issue": "too_few_tracks", "detail": len(tracks)})

	frame = pd.to_numeric(df.get("frame_idx"), errors="coerce")
	valid_frame = frame.dropna()
	frame_start = int(valid_frame.min()) if not valid_frame.empty else np.nan
	frame_end = int(valid_frame.max()) if not valid_frame.empty else np.nan
	frame_span = int(frame_end - frame_start + 1) if not valid_frame.empty else 0
	unique_frames = int(valid_frame.nunique())
	duplicate_rows = (
		int(df.duplicated(["track", "frame_idx"]).sum())
		if {"track", "frame_idx"}.issubset(df.columns)
		else 0
	)
	if duplicate_rows:
		issues.append({"issue": "duplicate_frame_track_rows", "detail": duplicate_rows})

	fatal = {"read_error", "missing_columns", "metadata_parse_failed", "mother_track_not_found", "too_few_tracks"}
	include = not any(issue["issue"] in fatal for issue in issues)
	row = {
		**meta,
		"file_name": path.name,
		"raw_path": str(path.resolve()),
		"sha256": _sha256(path) if calculate_hash else "",
		"file_size_bytes": path.stat().st_size,
		"row_count": len(df),
		"track_count_raw": len(tracks),
		"track_labels_raw": ",".join(tracks),
		"mother_track_raw": mother_track,
		"frame_start": frame_start,
		"frame_end": frame_end,
		"frame_span": frame_span,
		"unique_frames": unique_frames,
		"duration_sec": round(frame_span / float(fps), 3) if frame_span else np.nan,
		"duplicate_frame_track_rows": duplicate_rows,
		"fps": float(fps),
		"cage_width_mm": float(cage_width_mm),
		"cage_depth_mm": float(cage_depth_mm),
		"include": include,
		"issues": ";".join(issue["issue"] for issue in issues),
	}
	return row, issues
