from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import numpy as np

def count_visits(mask: np.ndarray) -> int:
	mask = np.asarray(mask, dtype=bool)
	if mask.size == 0:
		return 0
	return int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1]))

def close_short_gaps(
	mask: np.ndarray,
	max_gap_frames: int,
	*,
	eligible: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
	"""Fill short FALSE runs bounded by TRUE frames.

	Only frames marked by ``eligible`` can be filled. This prevents contact
	bridges across invalid tracking, wall exits, Body_C exclusions, or the
	opposite cup ROI.
	"""
	closed = np.asarray(mask, dtype=bool).copy()
	filled = np.zeros(closed.shape, dtype=bool)
	if closed.size == 0 or max_gap_frames < 1:
		return closed, filled, 0, 0

	if eligible is None:
		eligible_mask = np.ones(closed.shape, dtype=bool)
	else:
		eligible_mask = np.asarray(eligible, dtype=bool)
		if eligible_mask.shape != closed.shape:
			raise ValueError("eligible must have the same shape as mask")

	edges = np.diff(np.concatenate(([0], closed.astype(np.int8), [0])))
	starts = np.flatnonzero(edges == 1)
	ends = np.flatnonzero(edges == -1)
	closed_gaps = 0
	for gap_start, gap_end in zip(ends[:-1], starts[1:]):
		gap_length = int(gap_end - gap_start)
		if gap_length < 1 or gap_length > int(max_gap_frames):
			continue
		if not bool(np.all(eligible_mask[gap_start:gap_end])):
			continue
		closed[gap_start:gap_end] = True
		filled[gap_start:gap_end] = True
		closed_gaps += 1
	return closed, filled, int(filled.sum()), closed_gaps

def filter_short_bouts(mask: np.ndarray, min_frames: int) -> tuple[np.ndarray, int, int]:
	filtered = np.asarray(mask, dtype=bool).copy()
	if filtered.size == 0 or min_frames <= 1:
		return filtered, 0, 0
	edges = np.diff(np.concatenate(([0], filtered.astype(np.int8), [0])))
	starts = np.flatnonzero(edges == 1)
	ends = np.flatnonzero(edges == -1)
	short = (ends - starts) < int(min_frames)
	removed_frames = 0
	for start, end in zip(starts[short], ends[short]):
		removed_frames += int(end - start)
		filtered[start:end] = False
	return filtered, removed_frames, int(short.sum())
