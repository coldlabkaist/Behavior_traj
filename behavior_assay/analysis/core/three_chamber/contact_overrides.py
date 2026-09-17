from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

OVERRIDE_COLUMNS = (
	"session_key",
	"start_frame",
	"end_frame",
	"left_override",
	"right_override",
	"updated_at",
)

AUTO = np.int8(-1)

FALSE = np.int8(0)

TRUE = np.int8(1)

def empty_override_table() -> pd.DataFrame:
	return pd.DataFrame(columns=OVERRIDE_COLUMNS)

def load_overrides(path: Path) -> pd.DataFrame:
	if not path.exists() or path.stat().st_size == 0:
		return empty_override_table()
	df = pd.read_csv(path)
	missing = [column for column in OVERRIDE_COLUMNS if column not in df.columns]
	if missing:
		raise ValueError(f"Override file is missing columns: {', '.join(missing)}")
	df = df.loc[:, OVERRIDE_COLUMNS].copy()
	for column in ("start_frame", "end_frame", "left_override", "right_override"):
		df[column] = pd.to_numeric(df[column], errors="coerce")
	df = df.dropna(subset=["session_key", "start_frame", "end_frame", "left_override", "right_override"])
	df["session_key"] = df["session_key"].astype(str)
	df["start_frame"] = df["start_frame"].astype(int)
	df["end_frame"] = df["end_frame"].astype(int)
	for column in ("left_override", "right_override"):
		df[column] = df[column].astype(int)
		if not df[column].isin([-1, 0, 1]).all():
			raise ValueError(f"{column} must contain only -1, 0, or 1")
	return df.sort_values(["session_key", "start_frame", "end_frame"]).reset_index(drop=True)

def override_arrays(
	overrides: pd.DataFrame,
	*,
	session_key: str,
	pose_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	frames = np.asarray(pose_frames, dtype=int)
	left = np.full(frames.shape, AUTO, dtype=np.int8)
	right = np.full(frames.shape, AUTO, dtype=np.int8)
	if overrides.empty:
		return left, right
	rows = overrides[overrides["session_key"].astype(str) == str(session_key)]
	for row in rows.itertuples(index=False):
		mask = (frames >= int(row.start_frame)) & (frames <= int(row.end_frame))
		left[mask] = np.int8(row.left_override)
		right[mask] = np.int8(row.right_override)
	return left, right

def apply_overrides(
	auto_left: np.ndarray,
	auto_right: np.ndarray,
	left_override: np.ndarray,
	right_override: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	left = np.asarray(auto_left, dtype=bool).copy()
	right = np.asarray(auto_right, dtype=bool).copy()
	left_edit = np.asarray(left_override, dtype=np.int8)
	right_edit = np.asarray(right_override, dtype=np.int8)
	if not (left.shape == right.shape == left_edit.shape == right_edit.shape):
		raise ValueError("Automatic masks and override arrays must have identical shapes")
	left[left_edit != AUTO] = left_edit[left_edit != AUTO].astype(bool)
	right[right_edit != AUTO] = right_edit[right_edit != AUTO].astype(bool)
	return left, right

def compress_overrides(
	*,
	session_key: str,
	pose_frames: np.ndarray,
	left_override: np.ndarray,
	right_override: np.ndarray,
	updated_at: str | None = None,
) -> pd.DataFrame:
	frames = np.asarray(pose_frames, dtype=int)
	left = np.asarray(left_override, dtype=np.int8)
	right = np.asarray(right_override, dtype=np.int8)
	if not (frames.shape == left.shape == right.shape):
		raise ValueError("pose_frames and override arrays must have identical shapes")
	keep = ((left != AUTO) | (right != AUTO)) & (frames >= 0)
	indices = np.flatnonzero(keep)
	if indices.size == 0:
		return empty_override_table()

	stamp = updated_at or datetime.now().isoformat(timespec="seconds")
	rows: list[dict[str, object]] = []
	start_index = int(indices[0])
	previous_index = start_index
	for index in indices[1:]:
		index = int(index)
		same_state = left[index] == left[previous_index] and right[index] == right[previous_index]
		consecutive_frame = frames[index] == frames[previous_index] + 1
		if not (same_state and consecutive_frame):
			rows.append(
				{
					"session_key": str(session_key),
					"start_frame": int(frames[start_index]),
					"end_frame": int(frames[previous_index]),
					"left_override": int(left[start_index]),
					"right_override": int(right[start_index]),
					"updated_at": stamp,
				}
			)
			start_index = index
		previous_index = index
	rows.append(
		{
			"session_key": str(session_key),
			"start_frame": int(frames[start_index]),
			"end_frame": int(frames[previous_index]),
			"left_override": int(left[start_index]),
			"right_override": int(right[start_index]),
			"updated_at": stamp,
		}
	)
	return pd.DataFrame(rows, columns=OVERRIDE_COLUMNS)

def replace_session_overrides(
	overrides: pd.DataFrame,
	*,
	session_key: str,
	session_rows: pd.DataFrame,
) -> pd.DataFrame:
	remaining = overrides[overrides["session_key"].astype(str) != str(session_key)].copy()
	combined = pd.concat([remaining, session_rows], ignore_index=True)
	if combined.empty:
		return empty_override_table()
	return combined.loc[:, OVERRIDE_COLUMNS].sort_values(["session_key", "start_frame", "end_frame"]).reset_index(drop=True)

def save_overrides(path: Path, overrides: pd.DataFrame) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	table = overrides.loc[:, OVERRIDE_COLUMNS] if not overrides.empty else empty_override_table()
	temporary = path.with_suffix(path.suffix + ".tmp")
	table.to_csv(temporary, index=False)
	temporary.replace(path)
