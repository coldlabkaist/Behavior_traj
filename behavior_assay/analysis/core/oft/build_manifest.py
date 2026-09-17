from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.core.oft.common import ARENA_CM, DEFAULT_OUTPUT_DIR, DEFAULT_RAW_DIR, FPS, parse_oft_stem, to_abs_path

REQUIRED_COLUMNS = {
	"track",
	"frame_idx",
	"instance.score",
	"Nose.x",
	"Nose.y",
	"Nose.score",
	"Body_C.x",
	"Body_C.y",
	"Body_C.score",
}

def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()

def _bool_issue(issues: list[dict[str, str]], code: str, detail: object) -> None:
	issues.append({"issue": code, "detail": str(detail)})

def inspect_file(path: Path, *, fps: float, arena_cm: float, calculate_hash: bool) -> tuple[dict[str, object], list[dict[str, str]]]:
	meta = parse_oft_stem(path.stem)
	issues: list[dict[str, str]] = []
	try:
		df = pd.read_csv(path)
	except Exception as exc:
		row = {
			**meta,
			"raw_path": str(path.resolve()),
			"file_name": path.name,
			"include": False,
			"issues": "read_error",
			"fps": fps,
			"arena_cm": arena_cm,
		}
		return row, [{"issue": "read_error", "detail": str(exc)}]

	missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
	if missing_columns:
		_bool_issue(issues, "missing_columns", ",".join(missing_columns))

	frame = pd.to_numeric(df.get("frame_idx"), errors="coerce")
	valid_frame = frame.dropna()
	frame_start = int(valid_frame.min()) if not valid_frame.empty else np.nan
	frame_end = int(valid_frame.max()) if not valid_frame.empty else np.nan
	frame_span = int(frame_end - frame_start + 1) if not valid_frame.empty else 0
	unique_frames = int(valid_frame.nunique())
	frame_coverage = 100.0 * unique_frames / frame_span if frame_span else 0.0
	track_count = int(df["track"].astype(str).nunique()) if "track" in df else 0
	duplicate_key_rows = (
		int(df.duplicated(["track", "frame_idx"]).sum())
		if {"track", "frame_idx"}.issubset(df.columns)
		else 0
	)
	exact_duplicate_rows = int(df.duplicated().sum())

	if not bool(meta["metadata_ok"]):
		_bool_issue(issues, "metadata_parse_failed", path.stem)
	if str(meta.get("filename_condition", "")) and meta["filename_condition"] != meta["condition"]:
		_bool_issue(
			issues,
			"filename_condition_conflict",
			f"filename={meta['filename_condition']};manifest={meta['condition']}",
		)
	if track_count != 1:
		_bool_issue(issues, "unexpected_track_count", track_count)
	if duplicate_key_rows:
		_bool_issue(issues, "duplicate_frame_track_rows", duplicate_key_rows)
	if frame_span and unique_frames < frame_span:
		_bool_issue(issues, "missing_frame_rows", frame_span - unique_frames)

	out_of_range = 0
	for column in ("Body_C.x", "Body_C.y", "Nose.x", "Nose.y"):
		if column in df:
			values = pd.to_numeric(df[column], errors="coerce")
			out_of_range += int(((values < 0) | (values > 1)).sum())
	if out_of_range:
		_bool_issue(issues, "coordinates_outside_unit_range", out_of_range)

	fatal_codes = {"read_error", "missing_columns", "metadata_parse_failed", "unexpected_track_count"}
	include = not any(issue["issue"] in fatal_codes for issue in issues)
	row = {
		**meta,
		"file_name": path.name,
		"raw_path": str(path.resolve()),
		"sha256": _sha256(path) if calculate_hash else "",
		"file_size_bytes": path.stat().st_size,
		"row_count": len(df),
		"track_count": track_count,
		"frame_start": frame_start,
		"frame_end": frame_end,
		"frame_span": frame_span,
		"unique_frames": unique_frames,
		"frame_coverage_pct": round(frame_coverage, 3),
		"duplicate_frame_track_rows": duplicate_key_rows,
		"exact_duplicate_rows": exact_duplicate_rows,
		"coordinates_outside_unit_range": out_of_range,
		"fps": float(fps),
		"arena_cm": float(arena_cm),
		"include": include,
		"issues": ";".join(issue["issue"] for issue in issues),
	}
	return row, issues
