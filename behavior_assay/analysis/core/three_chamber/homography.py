from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import numpy as np

ARENA_WIDTH_CM = 60.0

ARENA_HEIGHT_CM = 40.0

CANONICAL_WIDTH = ARENA_WIDTH_CM / ARENA_HEIGHT_CM

CANONICAL_HEIGHT = 1.0

CANONICAL_WALL = np.array(
	[
		[0.0, 0.0],
		[CANONICAL_WIDTH, 0.0],
		[CANONICAL_WIDTH, CANONICAL_HEIGHT],
		[0.0, CANONICAL_HEIGHT],
	],
	dtype=float,
)

def order_wall_corners(points_xy: np.ndarray) -> np.ndarray:
	"""Return wall corners as top-left, top-right, bottom-right, bottom-left."""
	points = np.asarray(points_xy, dtype=float)
	points = points[np.isfinite(points).all(axis=1)]
	if points.shape[0] < 4:
		raise ValueError("At least four finite wall points are required for homography")
	if points.shape[0] > 4:
		centroid = points.mean(axis=0)
		dist = np.sqrt(((points - centroid) ** 2).sum(axis=1))
		points = points[np.argsort(dist)[-4:]]

	y_order = np.argsort(points[:, 1])
	top = points[y_order[:2]]
	bottom = points[y_order[-2:]]
	top = top[np.argsort(top[:, 0])]
	bottom = bottom[np.argsort(bottom[:, 0])]
	top_left, top_right = top
	bottom_left, bottom_right = bottom
	return np.array([top_left, top_right, bottom_right, bottom_left], dtype=float)

def homography_matrix(src_xy: np.ndarray, dst_xy: np.ndarray = CANONICAL_WALL) -> np.ndarray:
	"""Compute a projective transform mapping src_xy to dst_xy."""
	src = np.asarray(src_xy, dtype=float)
	dst = np.asarray(dst_xy, dtype=float)
	if src.shape != (4, 2) or dst.shape != (4, 2):
		raise ValueError("Homography requires source and destination arrays with shape (4, 2)")

	rows: list[list[float]] = []
	for (x, y), (u, v) in zip(src, dst):
		rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
		rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
	a = np.asarray(rows, dtype=float)
	_, _, vh = np.linalg.svd(a)
	h = vh[-1].reshape(3, 3)
	if abs(h[2, 2]) > 1e-12:
		h = h / h[2, 2]
	return h

def wall_homography(wall_xy: np.ndarray) -> np.ndarray:
	ordered_wall = order_wall_corners(wall_xy)
	return homography_matrix(ordered_wall, CANONICAL_WALL)

def apply_homography(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
	points = np.asarray(points_xy, dtype=float)
	if points.size == 0:
		return points.reshape((-1, 2))
	flat = points.reshape((-1, 2))
	out = np.full_like(flat, np.nan, dtype=float)
	finite = np.isfinite(flat).all(axis=1)
	if finite.any():
		hom = np.column_stack([flat[finite], np.ones(int(finite.sum()), dtype=float)])
		mapped = hom @ np.asarray(matrix, dtype=float).T
		den = mapped[:, 2]
		valid_den = np.abs(den) > 1e-12
		tmp = np.full((mapped.shape[0], 2), np.nan, dtype=float)
		tmp[valid_den] = mapped[valid_den, :2] / den[valid_den, None]
		out[finite] = tmp
	return out.reshape(points.shape)
