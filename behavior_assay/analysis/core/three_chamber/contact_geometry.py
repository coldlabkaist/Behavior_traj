from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import math
import cv2
import numpy as np

try:
	from shapely.geometry import Polygon
except ModuleNotFoundError:  # The moval environment intentionally stays lightweight.
	Polygon = None

def _convex_round_buffer(polygon_xy: np.ndarray, distance: float, resolution: int) -> np.ndarray:
	"""Buffer a convex hull with round joins without requiring Shapely."""
	poly = np.asarray(polygon_xy, dtype=float)
	poly = poly[np.isfinite(poly).all(axis=1)]
	if poly.shape[0] < 3:
		return np.empty((0, 2), dtype=float)
	hull = cv2.convexHull(poly.astype(np.float32), clockwise=False, returnPoints=True).reshape(-1, 2).astype(float)
	if hull.shape[0] < 3:
		return np.empty((0, 2), dtype=float)
	distance = max(float(distance), 0.0)
	if distance <= 0:
		return hull

	signed_area = 0.5 * float(
		np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) - np.roll(hull[:, 0], -1) * hull[:, 1])
	)
	direction = 1.0 if signed_area > 0 else -1.0
	edges = np.roll(hull, -1, axis=0) - hull
	lengths = np.linalg.norm(edges, axis=1)
	keep = lengths > 1e-12
	if not np.all(keep):
		hull = hull[keep]
		edges = np.roll(hull, -1, axis=0) - hull
		lengths = np.linalg.norm(edges, axis=1)
	if hull.shape[0] < 3:
		return np.empty((0, 2), dtype=float)

	if direction > 0:
		normals = np.column_stack([edges[:, 1], -edges[:, 0]]) / lengths[:, None]
	else:
		normals = np.column_stack([-edges[:, 1], edges[:, 0]]) / lengths[:, None]

	points: list[np.ndarray] = []
	max_step = math.pi / max(2 * int(resolution), 1)
	for index, vertex in enumerate(hull):
		previous_normal = normals[index - 1]
		next_normal = normals[index]
		start = math.atan2(previous_normal[1], previous_normal[0])
		end = math.atan2(next_normal[1], next_normal[0])
		if direction > 0:
			while end <= start:
				end += 2.0 * math.pi
		else:
			while end >= start:
				end -= 2.0 * math.pi
		steps = max(int(math.ceil(abs(end - start) / max_step)), 1)
		angles = np.linspace(start, end, steps + 1)
		arc = vertex + distance * np.column_stack([np.cos(angles), np.sin(angles)])
		if points:
			arc = arc[1:]
		points.extend(arc)
	return np.asarray(points, dtype=float)

def buffer_polygon(polygon_xy: np.ndarray, distance: float, resolution: int = 16) -> np.ndarray:
	poly = np.asarray(polygon_xy, dtype=float)
	poly = poly[np.isfinite(poly).all(axis=1)]
	if poly.shape[0] < 3:
		return np.empty((0, 2), dtype=float)
	center = np.mean(poly, axis=0)
	angles = np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0])
	ordered = poly[np.argsort(angles)]
	if Polygon is None:
		return _convex_round_buffer(ordered, distance, resolution)
	buffered = Polygon(ordered).convex_hull.buffer(max(float(distance), 0.0), resolution=resolution)
	if buffered.is_empty or buffered.geom_type != "Polygon":
		return np.empty((0, 2), dtype=float)
	return np.asarray(buffered.exterior.coords[:-1], dtype=float)
