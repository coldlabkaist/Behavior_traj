from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from analysis.paths import resolve_input_path
from analysis.tools.three_chamber.pin_rois_gui import _canonical_key, _path_similarity, _scan_videos
from analysis.core.three_chamber.preference_bars import _buffer_polygon, _infer_image_size, _point_in_polygon_mask
from analysis.core.three_chamber.contact_events import close_short_gaps, filter_short_bouts
from analysis.core.three_chamber.contact_overrides import AUTO, FALSE, TRUE, apply_overrides, compress_overrides, load_overrides, override_arrays, replace_session_overrides, save_overrides


REVIEW_COLUMNS = ("session_key", "status", "note", "reviewed_at")
PIN_COLORS = {
	"chamber_l": (255, 220, 0),
	"chamber_r": (0, 150, 255),
	"wall": (80, 210, 120),
}
ANALYSIS_COLOR = (255, 255, 0)
TRUE_COLOR = (70, 230, 70)
GAP_FILLED_COLOR = (255, 90, 190)
SHORT_BOUT_COLOR = (0, 220, 255)
BODY_EXCLUDED_COLOR = (50, 50, 255)
MANUAL_TRUE_COLOR = (80, 255, 255)
MANUAL_FALSE_COLOR = (60, 60, 255)
NOSE_COLOR = (245, 245, 245)
BODY_COLOR = (255, 80, 210)


def _to_abs_path(path_value: object) -> Path:
    return resolve_input_path(path_value)


def _normalize_session_n(value: object) -> str:
	if value is None or pd.isna(value):
		return ""
	text = str(value).strip().lower().removeprefix("n_").removeprefix("n")
	try:
		value_float = float(text)
	except ValueError:
		return ""
	return str(int(round(value_float))) if np.isfinite(value_float) else ""


def _ordered_polygon(points: np.ndarray) -> np.ndarray:
	poly = np.asarray(points, dtype=float)
	poly = poly[np.isfinite(poly).all(axis=1)]
	if poly.shape[0] < 3:
		return poly
	center = np.mean(poly, axis=0)
	angles = np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0])
	return poly[np.argsort(angles)]


def _poly_int(points: np.ndarray) -> np.ndarray:
	return np.rint(_ordered_polygon(points)).astype(np.int32).reshape((-1, 1, 2))


def _load_sessions(manifest_path: Path, preprocess_path: Path) -> pd.DataFrame:
	manifest = pd.read_csv(manifest_path)
	preprocess = pd.read_csv(preprocess_path)
	meta_columns = [
		"session_key",
		"canonical_stem",
		"condition",
		"date",
		"subject_id",
		"sex",
		"phase",
		"trial_id",
		"session_n",
		"social_side",
		"empty_side",
		"familiar_side",
		"novel_side",
		"pose_path",
		"pins_path",
	]
	meta = manifest[[column for column in meta_columns if column in manifest.columns]].drop_duplicates("session_key")
	preprocess = preprocess.drop(columns=[column for column in meta.columns if column != "session_key" and column in preprocess.columns])
	sessions = preprocess.merge(meta, on="session_key", how="inner")
	sessions["phase"] = sessions["phase"].astype(str).str.lower()
	sessions["session_n_norm"] = sessions["session_n"].map(_normalize_session_n)
	sessions["video_key"] = sessions["pose_path"].map(lambda value: _canonical_key(Path(str(value))))
	return sessions.sort_values(["condition", "date", "subject_id", "phase", "trial_id", "session_n_norm"]).reset_index(drop=True)


def _video_index(videos: list[Path]) -> dict[str, list[Path]]:
	index: dict[str, list[Path]] = {}
	for video in videos:
		index.setdefault(_canonical_key(video), []).append(video)
	return index


def _resolve_video(row: pd.Series, index: dict[str, list[Path]]) -> Path | None:
	candidates = index.get(str(row["video_key"]), [])
	if not candidates:
		return None
	if len(candidates) == 1:
		return candidates[0]
	pins_path = _to_abs_path(row["pins_path"])
	scored = sorted(
		((_path_similarity(video, pins_path), video) for video in candidates),
		key=lambda item: (-item[0], item[1].as_posix().lower()),
	)
	return scored[0][1]


@dataclass
class ContactData:
	frame_count: int
	pose_frame_idx: np.ndarray
	nose_x: np.ndarray
	nose_y: np.ndarray
	body_x: np.ndarray
	body_y: np.ndarray
	valid_wall: np.ndarray
	body_in_cup: np.ndarray
	raw_hit: dict[str, np.ndarray]
	body_excluded: dict[str, np.ndarray]
	pre_bout_hit: dict[str, np.ndarray]
	gap_filled: dict[str, np.ndarray]
	auto_final_hit: dict[str, np.ndarray]
	final_hit: dict[str, np.ndarray]
	pinned_polygons: dict[str, np.ndarray]
	analysis_polygons: dict[str, np.ndarray]
	role_labels: dict[str, str]
	inferred_width: float
	inferred_height: float
	min_bout_frames: int
	max_gap_frames: int

	@property
	def final_any(self) -> np.ndarray:
		return self.final_hit["chamber_l"] | self.final_hit["chamber_r"]


def _role_labels(row: pd.Series) -> dict[str, str]:
	labels = {"chamber_l": "L", "chamber_r": "R"}
	phase = str(row.get("phase", "")).lower()
	if phase == "soc":
		for side, label in ((row.get("social_side", ""), "S"), (row.get("empty_side", ""), "E")):
			if str(side).lower() in {"l", "r"}:
				labels[f"chamber_{str(side).lower()}"] = label
	elif phase == "nov":
		for side, label in ((row.get("familiar_side", ""), "F"), (row.get("novel_side", ""), "N")):
			if str(side).lower() in {"l", "r"}:
				labels[f"chamber_{str(side).lower()}"] = label
	return labels


def _build_contact_data(
	row: pd.Series,
	roi_vertices: pd.DataFrame,
	roi_summary: pd.DataFrame,
	*,
	buffer_percent: float,
	max_gap_seconds: float,
	min_bout_seconds: float,
	fps: float,
	max_minutes: float,
	exclude_body_in_cup: bool,
) -> ContactData:
	pose_path = _to_abs_path(row["output_path"])
	pose = pd.read_csv(pose_path).sort_values("frame_idx").reset_index(drop=True)
	for column in ("frame_idx", "Nose.x", "Nose.y", "Body_C.x", "Body_C.y"):
		if column not in pose.columns:
			raise ValueError(f"Missing pose column: {column}")

	vertices = roi_vertices[roi_vertices["session_key"] == row["session_key"]].copy()
	summary = roi_summary[roi_summary["session_key"] == row["session_key"]].copy()
	if vertices.empty:
		raise ValueError("No ROI vertices for this session")
	image_width, image_height = _infer_image_size(vertices)

	frame_values = pd.to_numeric(pose["frame_idx"], errors="coerce").to_numpy(dtype=float)
	finite_frames = frame_values[np.isfinite(frame_values)]
	if finite_frames.size == 0:
		raise ValueError("No finite frame_idx values")
	frame_min = int(np.min(finite_frames))
	relative_frames = np.rint(frame_values - frame_min).astype(int)
	pose_span = max(int(np.max(relative_frames)) + 1, 1)
	if max_minutes > 0:
		limit = max(int(round(max_minutes * 60.0 * fps)), 1)
		frame_count = min(pose_span, limit)
	else:
		frame_count = pose_span

	def place(column: str, scale: float = 1.0) -> np.ndarray:
		values = pd.to_numeric(pose[column], errors="coerce").to_numpy(dtype=float) * scale
		out = np.full(frame_count, np.nan, dtype=float)
		keep = (relative_frames >= 0) & (relative_frames < frame_count)
		out[relative_frames[keep]] = values[keep]
		return out

	pose_frame_idx = np.full(frame_count, -1, dtype=int)
	keep_frames = (relative_frames >= 0) & (relative_frames < frame_count)
	pose_frame_idx[relative_frames[keep_frames]] = np.rint(frame_values[keep_frames]).astype(int)
	nose_x = place("Nose.x", image_width)
	nose_y = place("Nose.y", image_height)
	body_x = place("Body_C.x", image_width)
	body_y = place("Body_C.y", image_height)

	pinned: dict[str, np.ndarray] = {}
	for roi_id in ("chamber_l", "chamber_r", "wall"):
		poly_rows = vertices[vertices["roi_id"] == roi_id].sort_values("vertex_idx")
		poly = poly_rows[["x", "y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
		poly = _ordered_polygon(poly)
		if poly.shape[0] < 3:
			raise ValueError(f"Incomplete ROI: {roi_id}")
		pinned[roi_id] = poly

	finite_nose = np.isfinite(nose_x) & np.isfinite(nose_y)
	valid_wall = finite_nose & _point_in_polygon_mask(nose_x, nose_y, pinned["wall"])
	body_finite = np.isfinite(body_x) & np.isfinite(body_y)
	body_in_cup = np.zeros(frame_count, dtype=bool)
	for roi_id in ("chamber_l", "chamber_r"):
		body_in_cup |= body_finite & _point_in_polygon_mask(body_x, body_y, pinned[roi_id])

	analysis_polygons: dict[str, np.ndarray] = {}
	raw_hit: dict[str, np.ndarray] = {}
	body_excluded: dict[str, np.ndarray] = {}
	pre_bout_hit: dict[str, np.ndarray] = {}
	gap_filled: dict[str, np.ndarray] = {}
	final_hit: dict[str, np.ndarray] = {}
	max_gap_frames = int(math.ceil(max(max_gap_seconds, 0.0) * max(fps, 1e-9)))
	min_bout_frames = int(math.ceil(max(min_bout_seconds, 0.0) * max(fps, 1e-9)))
	before_bout_by_roi: dict[str, np.ndarray] = {}
	for roi_id in ("chamber_l", "chamber_r"):
		summary_row = summary[summary["roi_id"] == roi_id]
		if summary_row.empty:
			raise ValueError(f"Missing ROI summary: {roi_id}")
		mean_radius = float(summary_row.iloc[0]["mean_radius_px"])
		buffer_px = mean_radius * max(buffer_percent, 0.0) / 100.0
		analysis_poly = _buffer_polygon(pinned[roi_id], buffer_px)
		if analysis_poly.shape[0] < 3:
			raise ValueError(f"Could not buffer ROI: {roi_id}")
		analysis_polygons[roi_id] = analysis_poly
		raw = valid_wall & _point_in_polygon_mask(nose_x, nose_y, analysis_poly)
		excluded = raw & body_in_cup if exclude_body_in_cup else np.zeros(frame_count, dtype=bool)
		before_bout = raw & ~excluded
		raw_hit[roi_id] = raw
		body_excluded[roi_id] = excluded
		pre_bout_hit[roi_id] = before_bout
		before_bout_by_roi[roi_id] = before_bout

	gap_eligible = valid_wall.copy()
	if exclude_body_in_cup:
		gap_eligible &= ~body_in_cup
	for roi_id, opposite_id in (("chamber_l", "chamber_r"), ("chamber_r", "chamber_l")):
		closed, filled, _, _ = close_short_gaps(
			before_bout_by_roi[roi_id],
			max_gap_frames,
			eligible=gap_eligible & ~before_bout_by_roi[opposite_id],
		)
		filtered, _, _ = filter_short_bouts(closed, min_bout_frames)
		gap_filled[roi_id] = filled & filtered
		final_hit[roi_id] = filtered

	return ContactData(
		frame_count=frame_count,
		pose_frame_idx=pose_frame_idx,
		nose_x=nose_x,
		nose_y=nose_y,
		body_x=body_x,
		body_y=body_y,
		valid_wall=valid_wall,
		body_in_cup=body_in_cup,
		raw_hit=raw_hit,
		body_excluded=body_excluded,
		pre_bout_hit=pre_bout_hit,
		gap_filled=gap_filled,
		auto_final_hit={roi_id: mask.copy() for roi_id, mask in final_hit.items()},
		final_hit=final_hit,
		pinned_polygons=pinned,
		analysis_polygons=analysis_polygons,
		role_labels=_role_labels(row),
		inferred_width=image_width,
		inferred_height=image_height,
		min_bout_frames=min_bout_frames,
		max_gap_frames=max_gap_frames,
	)


class VideoCanvas(QtWidgets.QGraphicsView):
	def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
		super().__init__(parent)
		self.setScene(QtWidgets.QGraphicsScene(self))
		self.setBackgroundBrush(QtGui.QColor("#111315"))
		self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
		self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
		self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
		self._has_image = False
		self._panning = False
		self._pan_position = QtCore.QPoint()

	def set_rgb_frame(self, rgb_frame: np.ndarray, *, reset_view: bool = False) -> None:
		height, width = rgb_frame.shape[:2]
		image = QtGui.QImage(
			rgb_frame.data,
			width,
			height,
			int(rgb_frame.strides[0]),
			QtGui.QImage.Format.Format_RGB888,
		).copy()
		self.scene().clear()
		self.scene().addPixmap(QtGui.QPixmap.fromImage(image))
		self.scene().setSceneRect(0, 0, width, height)
		self._has_image = True
		if reset_view:
			self.fit_image()

	def fit_image(self) -> None:
		if not self._has_image:
			return
		self.resetTransform()
		self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

	def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
		if not self._has_image:
			return
		factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
		current = self.transform().m11()
		if (factor > 1 and current < 8.0) or (factor < 1 and current > 0.15):
			self.scale(factor, factor)

	def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
		if event.button() == QtCore.Qt.MouseButton.MiddleButton:
			self._panning = True
			self._pan_position = event.position().toPoint()
			self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
			event.accept()
			return
		super().mousePressEvent(event)

	def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
		if self._panning:
			position = event.position().toPoint()
			delta = position - self._pan_position
			self._pan_position = position
			self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
			self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
			event.accept()
			return
		super().mouseMoveEvent(event)

	def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
		if event.button() == QtCore.Qt.MouseButton.MiddleButton and self._panning:
			self._panning = False
			self.unsetCursor()
			event.accept()
			return
		super().mouseReleaseEvent(event)


class ContactTimeline(QtWidgets.QWidget):
	frame_requested = QtCore.Signal(int)

	def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
		super().__init__(parent)
		self.setMinimumHeight(54)
		self.setMaximumHeight(66)
		self.setMouseTracking(True)
		self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
		self.left_mask = np.zeros(0, dtype=bool)
		self.right_mask = np.zeros(0, dtype=bool)
		self.left_gap = np.zeros(0, dtype=bool)
		self.right_gap = np.zeros(0, dtype=bool)
		self.left_manual = np.zeros(0, dtype=np.int8)
		self.right_manual = np.zeros(0, dtype=np.int8)
		self.left_label = "L"
		self.right_label = "R"
		self.fps = 30.0
		self.position = 0
		self._dragging = False

	def set_data(
		self,
		left_mask: np.ndarray,
		right_mask: np.ndarray,
		*,
		left_gap: np.ndarray,
		right_gap: np.ndarray,
		left_manual: np.ndarray,
		right_manual: np.ndarray,
		left_label: str,
		right_label: str,
		fps: float,
	) -> None:
		self.left_mask = np.asarray(left_mask, dtype=bool).copy()
		self.right_mask = np.asarray(right_mask, dtype=bool).copy()
		self.left_gap = np.asarray(left_gap, dtype=bool).copy()
		self.right_gap = np.asarray(right_gap, dtype=bool).copy()
		self.left_manual = np.asarray(left_manual, dtype=np.int8).copy()
		self.right_manual = np.asarray(right_manual, dtype=np.int8).copy()
		self.left_label = str(left_label)
		self.right_label = str(right_label)
		self.fps = max(float(fps), 1e-9)
		self.position = min(self.position, max(len(self.left_mask) - 1, 0))
		self.update()

	def clear_data(self) -> None:
		self.left_mask = np.zeros(0, dtype=bool)
		self.right_mask = np.zeros(0, dtype=bool)
		self.left_gap = np.zeros(0, dtype=bool)
		self.right_gap = np.zeros(0, dtype=bool)
		self.left_manual = np.zeros(0, dtype=np.int8)
		self.right_manual = np.zeros(0, dtype=np.int8)
		self.position = 0
		self.update()

	def set_position(self, frame_index: int) -> None:
		new_position = min(max(int(frame_index), 0), max(len(self.left_mask) - 1, 0))
		if new_position != self.position:
			self.position = new_position
			self.update()

	def _plot_rect(self) -> QtCore.QRectF:
		return QtCore.QRectF(52.0, 5.0, max(float(self.width()) - 62.0, 1.0), max(float(self.height()) - 20.0, 1.0))

	def _frame_from_x(self, x: float) -> int:
		if self.left_mask.size == 0:
			return 0
		rect = self._plot_rect()
		fraction = min(max((float(x) - rect.left()) / max(rect.width(), 1.0), 0.0), 1.0)
		return int(round(fraction * (len(self.left_mask) - 1)))

	@staticmethod
	def _bouts(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		if mask.size == 0:
			return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
		edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
		return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)

	def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
		painter = QtGui.QPainter(self)
		painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
		painter.fillRect(self.rect(), QtGui.QColor("#f5f6f7"))
		rect = self._plot_rect()
		painter.fillRect(rect, QtGui.QColor("#25292d"))
		middle = rect.top() + rect.height() / 2.0
		painter.setPen(QtGui.QPen(QtGui.QColor("#60666b"), 1))
		painter.drawLine(QtCore.QPointF(rect.left(), middle), QtCore.QPointF(rect.right(), middle))

		font = painter.font()
		font.setPointSize(8)
		font.setBold(True)
		painter.setFont(font)
		painter.setPen(QtGui.QColor("#202428"))
		painter.drawText(QtCore.QRectF(2, rect.top(), 47, rect.height() / 2.0), QtCore.Qt.AlignmentFlag.AlignCenter, f"L / {self.left_label}")
		painter.drawText(QtCore.QRectF(2, middle, 47, rect.height() / 2.0), QtCore.Qt.AlignmentFlag.AlignCenter, f"R / {self.right_label}")

		frame_count = len(self.left_mask)
		if frame_count > 0:
			for mask, color, top, bottom in (
				(self.left_mask, QtGui.QColor("#00c8ff"), rect.top() + 2.0, middle - 2.0),
				(self.right_mask, QtGui.QColor("#f28e1c"), middle + 2.0, rect.bottom() - 2.0),
			):
				starts, ends = self._bouts(mask)
				painter.setPen(QtCore.Qt.PenStyle.NoPen)
				painter.setBrush(color)
				for start, end in zip(starts, ends):
					x0 = rect.left() + rect.width() * float(start) / frame_count
					x1 = rect.left() + rect.width() * float(end) / frame_count
					painter.drawRect(QtCore.QRectF(x0, top, max(x1 - x0, 1.5), max(bottom - top, 1.0)))

			for mask, top, bottom in (
				(self.left_gap, rect.top() + 2.0, middle - 2.0),
				(self.right_gap, middle + 2.0, rect.bottom() - 2.0),
			):
				starts, ends = self._bouts(mask)
				painter.setPen(QtCore.Qt.PenStyle.NoPen)
				painter.setBrush(QtGui.QColor("#dc5cff"))
				for start, end in zip(starts, ends):
					x0 = rect.left() + rect.width() * float(start) / frame_count
					x1 = rect.left() + rect.width() * float(end) / frame_count
					painter.drawRect(QtCore.QRectF(x0, top, max(x1 - x0, 1.5), max(bottom - top, 1.0)))

			for manual, top, bottom in (
				(self.left_manual, rect.top() + 2.0, middle - 2.0),
				(self.right_manual, middle + 2.0, rect.bottom() - 2.0),
			):
				for state, color in ((TRUE, "#f7ff5c"), (FALSE, "#ff4050")):
					starts, ends = self._bouts(manual == state)
					painter.setPen(QtCore.Qt.PenStyle.NoPen)
					painter.setBrush(QtGui.QColor(color))
					for start, end in zip(starts, ends):
						x0 = rect.left() + rect.width() * float(start) / frame_count
						x1 = rect.left() + rect.width() * float(end) / frame_count
						height = max(bottom - top, 1.0) if state == TRUE else 3.0
						y = top if state == TRUE else (top + bottom - height) / 2.0
						painter.drawRect(QtCore.QRectF(x0, y, max(x1 - x0, 1.5), height))

			position_x = rect.left() + rect.width() * float(self.position) / max(frame_count - 1, 1)
			painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
			painter.drawLine(QtCore.QPointF(position_x, rect.top()), QtCore.QPointF(position_x, rect.bottom()))
			painter.setPen(QtGui.QPen(QtGui.QColor("#111111"), 1))
			painter.drawLine(QtCore.QPointF(position_x + 2, rect.top()), QtCore.QPointF(position_x + 2, rect.bottom()))

			total_seconds = int(round((frame_count - 1) / self.fps))
			painter.setPen(QtGui.QColor("#45494d"))
			painter.drawText(QtCore.QRectF(rect.left(), rect.bottom() + 1, 90, 13), QtCore.Qt.AlignmentFlag.AlignLeft, "0:00")
			minutes, seconds = divmod(total_seconds, 60)
			painter.drawText(QtCore.QRectF(rect.right() - 90, rect.bottom() + 1, 90, 13), QtCore.Qt.AlignmentFlag.AlignRight, f"{minutes}:{seconds:02d}")
		else:
			painter.setPen(QtGui.QColor("#c5c9cc"))
			painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "Load a matched video session")

	def _request_at(self, x: float) -> None:
		if self.left_mask.size:
			self.frame_requested.emit(self._frame_from_x(x))

	def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
		if event.button() == QtCore.Qt.MouseButton.LeftButton:
			self._dragging = True
			self._request_at(event.position().x())
			event.accept()
			return
		super().mousePressEvent(event)

	def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
		frame_index = self._frame_from_x(event.position().x())
		if self.left_mask.size:
			left = bool(self.left_mask[frame_index])
			right = bool(self.right_mask[frame_index])
			gap = bool(self.left_gap[frame_index] or self.right_gap[frame_index])
			manual = self.left_manual[frame_index] != AUTO or self.right_manual[frame_index] != AUTO
			seconds = frame_index / self.fps
			QtWidgets.QToolTip.showText(
				event.globalPosition().toPoint(),
				f"frame {frame_index} | {seconds:.2f} s | L/{self.left_label}={int(left)}, R/{self.right_label}={int(right)} | gap-filled={int(gap)} | manual={int(manual)}",
				self,
			)
		if self._dragging:
			self._request_at(event.position().x())
			event.accept()
			return
		super().mouseMoveEvent(event)

	def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
		if event.button() == QtCore.Qt.MouseButton.LeftButton:
			self._dragging = False
			event.accept()
			return
		super().mouseReleaseEvent(event)


class ContactReviewWindow(QtWidgets.QMainWindow):
	def __init__(
		self,
		*,
		manifest_path: Path,
		preprocess_path: Path,
		roi_vertices_path: Path,
		roi_summary_path: Path,
		review_path: Path,
		override_path: Path,
		video_dir: Path | None,
	) -> None:
		super().__init__()
		self.setWindowTitle("Three-chamber contact review")
		self.resize(1620, 940)
		self.setMinimumSize(1180, 720)
		self.manifest_path = manifest_path
		self.preprocess_path = preprocess_path
		self.roi_vertices_path = roi_vertices_path
		self.roi_summary_path = roi_summary_path
		self.review_path = review_path
		self.override_path = override_path
		self.sessions = _load_sessions(manifest_path, preprocess_path)
		self.roi_vertices = pd.read_csv(roi_vertices_path)
		self.roi_summary = pd.read_csv(roi_summary_path)
		self.review_rows = self._load_reviews()
		self.override_table = load_overrides(override_path)
		self.video_root = video_dir
		self.videos: list[Path] = []
		self.videos_by_key: dict[str, list[Path]] = {}
		self.visible_indices: list[int] = []
		self.current_row_index: int | None = None
		self.current_video: Path | None = None
		self.capture: cv2.VideoCapture | None = None
		self.contact: ContactData | None = None
		self.left_override = np.zeros(0, dtype=np.int8)
		self.right_override = np.zeros(0, dtype=np.int8)
		self.range_start: int | None = None
		self.range_end: int | None = None
		self.current_frame = 0
		self.video_frame_count = 0
		self.video_width = 0
		self.video_height = 0
		self.video_fps = 0.0
		self.rebuild_process: QtCore.QProcess | None = None

		self.play_timer = QtCore.QTimer(self)
		self.play_timer.timeout.connect(self._advance_playback)
		self.frame_timer = QtCore.QTimer(self)
		self.frame_timer.setSingleShot(True)
		self.frame_timer.setInterval(50)
		self.frame_timer.timeout.connect(self._render_current_frame)

		self._build_ui()
		self._bind_shortcuts()
		if video_dir and video_dir.exists():
			self._set_video_root(video_dir)
		else:
			self._refresh_session_list()

	def _build_ui(self) -> None:
		central = QtWidgets.QWidget()
		self.setCentralWidget(central)
		layout = QtWidgets.QVBoxLayout(central)
		layout.setContentsMargins(8, 8, 8, 8)
		layout.setSpacing(6)

		path_row = QtWidgets.QHBoxLayout()
		video_button = QtWidgets.QPushButton("Load video folder")
		video_button.clicked.connect(self._choose_video_folder)
		add_videos_button = QtWidgets.QPushButton("Add video files")
		add_videos_button.clicked.connect(self._choose_video_files)
		self.video_path_edit = QtWidgets.QLineEdit(str(self.video_root or ""))
		self.video_path_edit.setReadOnly(True)
		path_row.addWidget(video_button)
		path_row.addWidget(self.video_path_edit, 1)
		path_row.addWidget(add_videos_button)
		path_row.addSpacing(12)
		path_row.addWidget(QtWidgets.QLabel("Analysis: Nose | cup polygon + buffer | wall filter"))
		layout.addLayout(path_row)

		splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
		layout.addWidget(splitter, 1)

		left_panel = QtWidgets.QWidget()
		left_layout = QtWidgets.QVBoxLayout(left_panel)
		left_layout.setContentsMargins(0, 0, 4, 0)
		filter_row = QtWidgets.QHBoxLayout()
		self.phase_combo = QtWidgets.QComboBox()
		self.phase_combo.addItems(["SOC + NOV", "SOC", "NOV"])
		self.phase_combo.currentIndexChanged.connect(self._refresh_session_list)
		self.session_combo = QtWidgets.QComboBox()
		self.session_combo.addItems(["n1", "All sessions"])
		self.session_combo.currentIndexChanged.connect(self._refresh_session_list)
		filter_row.addWidget(self.phase_combo)
		filter_row.addWidget(self.session_combo)
		left_layout.addLayout(filter_row)
		self.search_edit = QtWidgets.QLineEdit()
		self.search_edit.setPlaceholderText("Filter condition/date/subject...")
		self.search_edit.textChanged.connect(self._refresh_session_list)
		left_layout.addWidget(self.search_edit)
		self.session_list = QtWidgets.QListWidget()
		self.session_list.setFont(QtGui.QFont("Consolas", 9))
		self.session_list.currentRowChanged.connect(self._session_selected)
		left_layout.addWidget(self.session_list, 1)

		review_group = QtWidgets.QGroupBox("Session review")
		review_layout = QtWidgets.QVBoxLayout(review_group)
		self.review_status = QtWidgets.QComboBox()
		self.review_status.addItems(["Unreviewed", "OK", "Issue"])
		self.review_note = QtWidgets.QLineEdit()
		self.review_note.setPlaceholderText("Optional note")
		review_save = QtWidgets.QPushButton("Save review + overrides (Ctrl+S)")
		review_save.clicked.connect(self._save_all)
		review_layout.addWidget(self.review_status)
		review_layout.addWidget(self.review_note)
		review_layout.addWidget(review_save)
		left_layout.addWidget(review_group)

		viewer_panel = QtWidgets.QWidget()
		viewer_layout = QtWidgets.QVBoxLayout(viewer_panel)
		viewer_layout.setContentsMargins(4, 0, 0, 0)
		self.header_label = QtWidgets.QLabel("Select a video folder and session.")
		header_font = self.header_label.font()
		header_font.setBold(True)
		self.header_label.setFont(header_font)
		viewer_layout.addWidget(self.header_label)
		self.canvas = VideoCanvas()
		viewer_layout.addWidget(self.canvas, 1)

		frame_row = QtWidgets.QHBoxLayout()
		self.play_button = QtWidgets.QPushButton("Play")
		self.play_button.clicked.connect(self._toggle_play)
		frame_row.addWidget(self.play_button)
		for label, delta in (("-30", -30), ("-1", -1), ("+1", 1), ("+30", 30)):
			button = QtWidgets.QPushButton(label)
			button.clicked.connect(lambda _checked=False, step=delta: self._step_frame(step))
			frame_row.addWidget(button)
		prev_true = QtWidgets.QPushButton("Previous TRUE")
		prev_true.clicked.connect(lambda: self._jump_true(-1))
		next_true = QtWidgets.QPushButton("Next TRUE")
		next_true.clicked.connect(lambda: self._jump_true(1))
		frame_row.addWidget(prev_true)
		frame_row.addWidget(next_true)
		self.true_only_check = QtWidgets.QCheckBox("Play TRUE only")
		frame_row.addWidget(self.true_only_check)
		self.speed_combo = QtWidgets.QComboBox()
		self.speed_combo.addItems(["0.5x", "1x", "2x"])
		self.speed_combo.setCurrentText("1x")
		frame_row.addWidget(self.speed_combo)
		fit_button = QtWidgets.QPushButton("Fit")
		fit_button.clicked.connect(self.canvas.fit_image)
		frame_row.addWidget(fit_button)
		viewer_layout.addLayout(frame_row)

		self.contact_timeline = ContactTimeline()
		self.contact_timeline.frame_requested.connect(self._set_frame)
		viewer_layout.addWidget(self.contact_timeline)
		self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
		self.frame_slider.setRange(0, 1)
		self.frame_slider.valueChanged.connect(self._schedule_frame)
		viewer_layout.addWidget(self.frame_slider)

		manual_group = QtWidgets.QGroupBox("Manual correction (final interaction mask)")
		manual_layout = QtWidgets.QVBoxLayout(manual_group)
		range_layout = QtWidgets.QHBoxLayout()
		range_layout.addWidget(QtWidgets.QLabel("Start frame"))
		self.range_start_spin = QtWidgets.QSpinBox()
		self.range_start_spin.setRange(-1, 0)
		self.range_start_spin.setSpecialValueText("Current")
		self.range_start_spin.setValue(-1)
		self.range_start_spin.valueChanged.connect(self._range_inputs_changed)
		range_layout.addWidget(self.range_start_spin)
		mark_start = QtWidgets.QPushButton("Start = current [")
		mark_start.clicked.connect(self._mark_range_start)
		range_layout.addWidget(mark_start)
		range_layout.addWidget(QtWidgets.QLabel("End frame"))
		self.range_end_spin = QtWidgets.QSpinBox()
		self.range_end_spin.setRange(-1, 0)
		self.range_end_spin.setSpecialValueText("Current")
		self.range_end_spin.setValue(-1)
		self.range_end_spin.valueChanged.connect(self._range_inputs_changed)
		range_layout.addWidget(self.range_end_spin)
		mark_end = QtWidgets.QPushButton("End = current ]")
		mark_end.clicked.connect(self._mark_range_end)
		range_layout.addWidget(mark_end)
		current_only = QtWidgets.QPushButton("Current frame only")
		current_only.clicked.connect(self._set_current_frame_range)
		range_layout.addWidget(current_only)
		self.range_label = QtWidgets.QLabel("Range: current frame")
		range_layout.addWidget(self.range_label)
		range_layout.addStretch(1)
		manual_layout.addLayout(range_layout)
		action_layout = QtWidgets.QHBoxLayout()
		self.left_true_button = QtWidgets.QPushButton("Set L TRUE (1)")
		self.left_true_button.clicked.connect(lambda: self._apply_manual_state("left"))
		self.right_true_button = QtWidgets.QPushButton("Set R TRUE (2)")
		self.right_true_button.clicked.connect(lambda: self._apply_manual_state("right"))
		set_false = QtWidgets.QPushButton("Set FALSE (0)")
		set_false.clicked.connect(lambda: self._apply_manual_state("false"))
		reset_auto = QtWidgets.QPushButton("Reset auto (Backspace)")
		reset_auto.clicked.connect(lambda: self._apply_manual_state("auto"))
		action_layout.addWidget(self.left_true_button)
		action_layout.addWidget(self.right_true_button)
		action_layout.addWidget(set_false)
		action_layout.addWidget(reset_auto)
		exclude_bout = QtWidgets.QPushButton("Exclude current bout (Del)")
		exclude_bout.clicked.connect(self._exclude_current_bout)
		action_layout.addWidget(exclude_bout)
		save_manual = QtWidgets.QPushButton("Save overrides")
		save_manual.clicked.connect(self._persist_current_overrides)
		action_layout.addWidget(save_manual)
		action_layout.addStretch(1)
		manual_layout.addLayout(action_layout)
		viewer_layout.addWidget(manual_group)

		option_row = QtWidgets.QHBoxLayout()
		option_row.addWidget(QtWidgets.QLabel("Buffer (% cup radius)"))
		self.buffer_spin = QtWidgets.QDoubleSpinBox()
		self.buffer_spin.setRange(0.0, 100.0)
		self.buffer_spin.setValue(20.0)
		self.buffer_spin.setDecimals(1)
		option_row.addWidget(self.buffer_spin)
		option_row.addWidget(QtWidgets.QLabel("Maximum gap (s)"))
		self.gap_spin = QtWidgets.QDoubleSpinBox()
		self.gap_spin.setRange(0.0, 10.0)
		self.gap_spin.setValue(0.5)
		self.gap_spin.setSingleStep(0.1)
		option_row.addWidget(self.gap_spin)
		option_row.addWidget(QtWidgets.QLabel("Minimum bout (s)"))
		self.bout_spin = QtWidgets.QDoubleSpinBox()
		self.bout_spin.setRange(0.0, 10.0)
		self.bout_spin.setValue(0.5)
		self.bout_spin.setSingleStep(0.1)
		option_row.addWidget(self.bout_spin)
		option_row.addWidget(QtWidgets.QLabel("FPS"))
		self.fps_spin = QtWidgets.QDoubleSpinBox()
		self.fps_spin.setRange(1.0, 240.0)
		self.fps_spin.setValue(30.0)
		option_row.addWidget(self.fps_spin)
		option_row.addWidget(QtWidgets.QLabel("Analyze first (min)"))
		self.minutes_spin = QtWidgets.QDoubleSpinBox()
		self.minutes_spin.setRange(0.0, 60.0)
		self.minutes_spin.setValue(5.0)
		self.minutes_spin.setSpecialValueText("Full")
		option_row.addWidget(self.minutes_spin)
		self.exclude_body_check = QtWidgets.QCheckBox("Exclude Body_C in pinned cup")
		self.exclude_body_check.setChecked(True)
		option_row.addWidget(self.exclude_body_check)
		recalculate = QtWidgets.QPushButton("Recalculate")
		recalculate.clicked.connect(self._recalculate)
		option_row.addWidget(recalculate)
		rebuild = QtWidgets.QPushButton("Save + rebuild barplots")
		rebuild.clicked.connect(self._rebuild_barplots)
		option_row.addWidget(rebuild)
		option_row.addStretch(1)
		viewer_layout.addLayout(option_row)

		self.frame_status = QtWidgets.QLabel("No frame loaded")
		self.frame_status.setWordWrap(True)
		self.frame_status.setStyleSheet("padding: 5px; background: #eceff1;")
		viewer_layout.addWidget(self.frame_status)

		splitter.addWidget(left_panel)
		splitter.addWidget(viewer_panel)
		splitter.setSizes([360, 1260])
		self.statusBar().showMessage(
			"Automatic: green=TRUE, purple=gap-filled, yellow=short bout removed, red=Body_C exclusion. "
			"Manual timeline: yellow=forced TRUE, red=forced FALSE."
		)

	def _bind_shortcuts(self) -> None:
		bindings = [
			("Space", self._toggle_play),
			("Left", lambda: self._step_frame(-1)),
			("Right", lambda: self._step_frame(1)),
			("Ctrl+Left", lambda: self._jump_true(-1)),
			("Ctrl+Right", lambda: self._jump_true(1)),
			("[", self._mark_range_start),
			("]", self._mark_range_end),
			("1", lambda: self._apply_manual_state("left")),
			("2", lambda: self._apply_manual_state("right")),
			("0", lambda: self._apply_manual_state("false")),
			("Backspace", lambda: self._apply_manual_state("auto")),
			("Delete", self._exclude_current_bout),
			("Ctrl+S", self._save_all),
			("Escape", self.canvas.fit_image),
		]
		self.shortcuts = []
		for key, callback in bindings:
			shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
			shortcut.activated.connect(callback)
			self.shortcuts.append(shortcut)

	def _choose_video_folder(self) -> None:
		selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Select video root", str(self.video_root or ROOT))
		if selected:
			self._set_video_root(Path(selected).resolve())

	def _choose_video_files(self) -> None:
		selected, _ = QtWidgets.QFileDialog.getOpenFileNames(
			self,
			"Select videos",
			str(self.video_root or ROOT),
			"Video files (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All files (*.*)",
		)
		if not selected:
			return
		paths = [Path(path).resolve() for path in selected]
		self.videos = sorted(set(self.videos + paths), key=lambda path: path.as_posix().lower())
		self.videos_by_key = _video_index(self.videos)
		self.video_path_edit.setText(f"{len(self.videos)} selected video files")
		self._refresh_session_list()
		self.statusBar().showMessage(f"Loaded {len(paths)} additional videos; {len(self.videos)} total.")

	def _set_video_root(self, root: Path) -> None:
		self.video_root = root
		self.video_path_edit.setText(str(root))
		QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
		try:
			self.videos = _scan_videos(root)
			self.videos_by_key = _video_index(self.videos)
		finally:
			QtWidgets.QApplication.restoreOverrideCursor()
		self._refresh_session_list()
		self.statusBar().showMessage(f"Loaded {len(self.videos)} videos from {root}")

	def _refresh_session_list(self) -> None:
		phase_text = self.phase_combo.currentText().lower()
		search = self.search_edit.text().strip().lower()
		self.visible_indices = []
		self.session_list.blockSignals(True)
		self.session_list.clear()
		for index, row in self.sessions.iterrows():
			phase = str(row["phase"]).lower()
			if phase not in {"soc", "nov"}:
				continue
			if phase_text != "soc + nov" and phase != phase_text:
				continue
			if self.session_combo.currentText() == "n1" and str(row["session_n_norm"]) != "1":
				continue
			label = (
				f"{row.get('condition', '')} {row.get('date', '')} {row.get('subject_id', '')} "
				f"{phase.upper()} n{row.get('session_n_norm', '')} id{row.get('trial_id', '')}"
			)
			if search and search not in label.lower() and search not in str(row["session_key"]).lower():
				continue
			video = _resolve_video(row, self.videos_by_key)
			review = self.review_rows.get(str(row["session_key"]), {})
			status = "OK" if video else "--"
			review_mark = {"OK": "+", "Issue": "!"}.get(review.get("status", ""), " ")
			manual_mark = "M" if bool(np.any(self.override_table["session_key"].astype(str) == str(row["session_key"]))) else " "
			self.session_list.addItem(f"[{status}][{review_mark}][{manual_mark}] {label}")
			self.visible_indices.append(int(index))
		self.session_list.blockSignals(False)
		if self.visible_indices:
			self.session_list.setCurrentRow(0)
		else:
			self._release_capture()
			self.header_label.setText("No sessions matched the current filters.")

	def _session_selected(self, list_index: int) -> None:
		if list_index < 0 or list_index >= len(self.visible_indices):
			return
		self._load_session(self.visible_indices[list_index])

	def _release_capture(self) -> None:
		self.play_timer.stop()
		self.play_button.setText("Play")
		if self.capture is not None:
			self.capture.release()
			self.capture = None

	def _load_session(self, row_index: int) -> None:
		self._release_capture()
		self.current_row_index = row_index
		row = self.sessions.loc[row_index]
		self.current_video = _resolve_video(row, self.videos_by_key)
		self._load_review_fields(str(row["session_key"]))
		if self.current_video is None:
			self.contact = None
			self.contact_timeline.clear_data()
			self.header_label.setText(f"No matching video: {row['session_key']}")
			return
		self.capture = cv2.VideoCapture(str(self.current_video))
		if not self.capture.isOpened():
			QtWidgets.QMessageBox.critical(self, "Video error", f"Could not open:\n{self.current_video}")
			self._release_capture()
			return
		self.video_frame_count = max(int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
		self.video_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.video_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.video_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
		try:
			self.contact = self._calculate(row)
		except Exception as exc:
			self.contact = None
			QtWidgets.QMessageBox.critical(self, "Analysis error", f"{row['session_key']}\n\n{exc}")
			return
		self._load_manual_overrides(row)
		max_frame = min(self.video_frame_count, self.contact.frame_count) - 1
		self.frame_slider.blockSignals(True)
		self.frame_slider.setRange(0, max(max_frame, 0))
		self.current_frame = 0
		self.frame_slider.setValue(0)
		self.frame_slider.blockSignals(False)
		self._update_timeline()
		self._render_current_frame(reset_view=True)

	def _calculate(self, row: pd.Series) -> ContactData:
		return _build_contact_data(
			row,
			self.roi_vertices,
			self.roi_summary,
			buffer_percent=float(self.buffer_spin.value()),
			max_gap_seconds=float(self.gap_spin.value()),
			min_bout_seconds=float(self.bout_spin.value()),
			fps=float(self.fps_spin.value()),
			max_minutes=float(self.minutes_spin.value()),
			exclude_body_in_cup=bool(self.exclude_body_check.isChecked()),
		)

	def _load_manual_overrides(self, row: pd.Series) -> None:
		if self.contact is None:
			return
		self.left_true_button.setText(f"Set L / {self.contact.role_labels['chamber_l']} TRUE (1)")
		self.right_true_button.setText(f"Set R / {self.contact.role_labels['chamber_r']} TRUE (2)")
		self.left_override, self.right_override = override_arrays(
			self.override_table,
			session_key=str(row["session_key"]),
			pose_frames=self.contact.pose_frame_idx,
		)
		maximum = max(self.contact.frame_count - 1, 0)
		self.range_start_spin.setMaximum(maximum)
		self.range_end_spin.setMaximum(maximum)
		self._reset_range_inputs()
		self._refresh_manual_masks()
		self._update_range_label()

	def _refresh_manual_masks(self) -> None:
		if self.contact is None:
			return
		left, right = apply_overrides(
			self.contact.auto_final_hit["chamber_l"],
			self.contact.auto_final_hit["chamber_r"],
			self.left_override,
			self.right_override,
		)
		self.contact.final_hit["chamber_l"] = left
		self.contact.final_hit["chamber_r"] = right

	def _selected_range(self) -> tuple[int, int]:
		if self.contact is None:
			return 0, 0
		start = self.range_start if self.range_start is not None else self.current_frame
		end = self.range_end if self.range_end is not None else self.current_frame
		return min(start, end), max(start, end)

	def _range_inputs_changed(self, _value: int | None = None) -> None:
		self.range_start = None if self.range_start_spin.value() < 0 else int(self.range_start_spin.value())
		self.range_end = None if self.range_end_spin.value() < 0 else int(self.range_end_spin.value())
		self._update_range_label()

	def _reset_range_inputs(self) -> None:
		self.range_start_spin.setValue(-1)
		self.range_end_spin.setValue(-1)
		self.range_start = None
		self.range_end = None

	def _set_current_frame_range(self) -> None:
		if self.contact is None:
			return
		self.range_start_spin.setValue(self.current_frame)
		self.range_end_spin.setValue(self.current_frame)
		self._update_range_label()

	def _update_range_label(self) -> None:
		if self.range_start is None and self.range_end is None:
			self.range_label.setText(f"Range: current frame {self.current_frame}")
			return
		start, end = self._selected_range()
		self.range_label.setText(f"Range: {start} - {end} ({end - start + 1} frames)")

	def _mark_range_start(self) -> None:
		if self.contact is None:
			return
		self.range_start_spin.setValue(self.current_frame)
		self.range_end_spin.setValue(-1)
		self._update_range_label()

	def _mark_range_end(self) -> None:
		if self.contact is None:
			return
		if self.range_start is None:
			self.range_start_spin.setValue(self.current_frame)
		self.range_end_spin.setValue(self.current_frame)
		self._update_range_label()

	def _apply_manual_state(self, state: str) -> None:
		if self.contact is None or self.current_row_index is None:
			return
		start, end = self._selected_range()
		slice_ = slice(start, end + 1)
		if state == "left":
			self.left_override[slice_] = TRUE
			self.right_override[slice_] = FALSE
		elif state == "right":
			self.left_override[slice_] = FALSE
			self.right_override[slice_] = TRUE
		elif state == "false":
			self.left_override[slice_] = FALSE
			self.right_override[slice_] = FALSE
		elif state == "auto":
			self.left_override[slice_] = AUTO
			self.right_override[slice_] = AUTO
		else:
			raise ValueError(f"Unknown manual state: {state}")
		self._reset_range_inputs()
		self._refresh_manual_masks()
		self._update_range_label()
		self._persist_current_overrides(refresh_list=False)
		self._update_timeline()
		self._render_current_frame()

	def _exclude_current_bout(self) -> None:
		if self.contact is None:
			return
		mask = self.contact.final_any
		if not bool(mask[self.current_frame]):
			self.statusBar().showMessage("Current frame is not inside a final TRUE bout.")
			return
		start = self.current_frame
		end = self.current_frame
		while start > 0 and mask[start - 1]:
			start -= 1
		while end + 1 < mask.size and mask[end + 1]:
			end += 1
		self.range_start = start
		self.range_end = end
		self._apply_manual_state("false")

	def _persist_current_overrides(self, *, refresh_list: bool = True) -> None:
		if self.contact is None or self.current_row_index is None:
			return
		session_key = str(self.sessions.loc[self.current_row_index, "session_key"])
		rows = compress_overrides(
			session_key=session_key,
			pose_frames=self.contact.pose_frame_idx,
			left_override=self.left_override,
			right_override=self.right_override,
		)
		self.override_table = replace_session_overrides(
			self.override_table,
			session_key=session_key,
			session_rows=rows,
		)
		save_overrides(self.override_path, self.override_table)
		self.statusBar().showMessage(f"Saved {len(rows)} override ranges to {self.override_path}")
		if refresh_list:
			current_list_row = self.session_list.currentRow()
			self._refresh_session_list()
			if current_list_row >= 0:
				self.session_list.setCurrentRow(min(current_list_row, self.session_list.count() - 1))

	def _recalculate(self) -> None:
		if self.current_row_index is None:
			return
		row = self.sessions.loc[self.current_row_index]
		try:
			self.contact = self._calculate(row)
		except Exception as exc:
			QtWidgets.QMessageBox.critical(self, "Analysis error", str(exc))
			return
		self._load_manual_overrides(row)
		max_frame = min(self.video_frame_count, self.contact.frame_count) - 1
		self.frame_slider.setRange(0, max(max_frame, 0))
		self.current_frame = min(self.current_frame, max(max_frame, 0))
		self.frame_slider.setValue(self.current_frame)
		self._update_timeline()
		self._render_current_frame()

	def _update_timeline(self) -> None:
		if self.contact is None:
			self.contact_timeline.clear_data()
			return
		self.contact_timeline.set_data(
			self.contact.final_hit["chamber_l"],
			self.contact.final_hit["chamber_r"],
			left_gap=self.contact.gap_filled["chamber_l"] & self.contact.final_hit["chamber_l"],
			right_gap=self.contact.gap_filled["chamber_r"] & self.contact.final_hit["chamber_r"],
			left_manual=self.left_override,
			right_manual=self.right_override,
			left_label=self.contact.role_labels["chamber_l"],
			right_label=self.contact.role_labels["chamber_r"],
			fps=float(self.fps_spin.value()),
		)
		self.contact_timeline.set_position(self.current_frame)

	def _schedule_frame(self, frame_index: int) -> None:
		self.current_frame = int(frame_index)
		self._update_range_label()
		self.frame_timer.start()

	def _step_frame(self, delta: int) -> None:
		if self.contact is None:
			return
		self._set_frame(self.current_frame + int(delta))

	def _set_frame(self, frame_index: int) -> None:
		if self.contact is None:
			return
		maximum = self.frame_slider.maximum()
		self.current_frame = min(max(int(frame_index), 0), maximum)
		self.frame_slider.blockSignals(True)
		self.frame_slider.setValue(self.current_frame)
		self.frame_slider.blockSignals(False)
		self.contact_timeline.set_position(self.current_frame)
		self._update_range_label()
		self._render_current_frame()

	def _jump_true(self, direction: int) -> None:
		if self.contact is None:
			return
		indices = np.flatnonzero(self.contact.final_any)
		if direction > 0:
			candidates = indices[indices > self.current_frame]
			target = candidates[0] if candidates.size else (indices[0] if indices.size else None)
		else:
			candidates = indices[indices < self.current_frame]
			target = candidates[-1] if candidates.size else (indices[-1] if indices.size else None)
		if target is not None:
			self._set_frame(int(target))

	def _toggle_play(self) -> None:
		if self.contact is None:
			return
		if self.play_timer.isActive():
			self.play_timer.stop()
			self.play_button.setText("Play")
			return
		speed = float(self.speed_combo.currentText().removesuffix("x"))
		interval = max(int(round(1000.0 / (float(self.fps_spin.value()) * speed))), 5)
		if self.true_only_check.isChecked():
			interval = max(interval, 120)
		self.play_timer.start(interval)
		self.play_button.setText("Pause")

	def _advance_playback(self) -> None:
		if self.contact is None:
			return
		if self.true_only_check.isChecked():
			indices = np.flatnonzero(self.contact.final_any)
			candidates = indices[indices > self.current_frame]
			if candidates.size:
				self._set_frame(int(candidates[0]))
				return
		else:
			if self.current_frame < self.frame_slider.maximum():
				self._set_frame(self.current_frame + 1)
				return
		self.play_timer.stop()
		self.play_button.setText("Play")

	def _frame_state(self, frame_index: int) -> tuple[str, tuple[int, int, int]]:
		assert self.contact is not None
		manual = self.left_override[frame_index] != AUTO or self.right_override[frame_index] != AUTO
		if manual:
			manual_sides = [
				roi
				for roi, override in (("chamber_l", self.left_override), ("chamber_r", self.right_override))
				if override[frame_index] == TRUE
			]
			if manual_sides:
				labels = "/".join(self.contact.role_labels[side] for side in manual_sides)
				return f"MANUAL TRUE: {labels}", MANUAL_TRUE_COLOR
			return "MANUAL FALSE", MANUAL_FALSE_COLOR
		gap_sides = [roi for roi in ("chamber_l", "chamber_r") if self.contact.gap_filled[roi][frame_index]]
		if gap_sides:
			labels = "/".join(self.contact.role_labels[side] for side in gap_sides)
			return f"TRUE (gap-filled): {labels}", GAP_FILLED_COLOR
		final_sides = [roi for roi in ("chamber_l", "chamber_r") if self.contact.final_hit[roi][frame_index]]
		if final_sides:
			labels = "/".join(self.contact.role_labels[side] for side in final_sides)
			return f"TRUE: {labels}", TRUE_COLOR
		excluded = [roi for roi in ("chamber_l", "chamber_r") if self.contact.body_excluded[roi][frame_index]]
		if excluded:
			labels = "/".join(self.contact.role_labels[side] for side in excluded)
			return f"EXCLUDED (Body_C in cup): {labels}", BODY_EXCLUDED_COLOR
		short = [
			roi
			for roi in ("chamber_l", "chamber_r")
			if self.contact.pre_bout_hit[roi][frame_index] and not self.contact.final_hit[roi][frame_index]
		]
		if short:
			labels = "/".join(self.contact.role_labels[side] for side in short)
			return f"REMOVED (< minimum bout): {labels}", SHORT_BOUT_COLOR
		if not self.contact.valid_wall[frame_index]:
			return "FALSE: Nose missing/outside wall", (130, 130, 130)
		return "FALSE", (210, 210, 210)

	def _render_current_frame(self, *, reset_view: bool = False) -> None:
		if self.capture is None or self.contact is None or self.current_row_index is None:
			return
		frame_index = min(self.current_frame, self.contact.frame_count - 1)
		self.contact_timeline.set_position(frame_index)
		pose_frame = int(self.contact.pose_frame_idx[frame_index])
		video_frame = pose_frame if pose_frame >= 0 else frame_index
		self.capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
		ok, frame = self.capture.read()
		if not ok:
			self.statusBar().showMessage(f"Could not read video frame {video_frame} (analysis frame {frame_index})")
			return
		row = self.sessions.loc[self.current_row_index]
		for roi_id in ("wall", "chamber_l", "chamber_r"):
			cv2.polylines(frame, [_poly_int(self.contact.pinned_polygons[roi_id])], True, PIN_COLORS[roi_id], 2, cv2.LINE_AA)
		for roi_id in ("chamber_l", "chamber_r"):
			thickness = 5 if self.contact.final_hit[roi_id][frame_index] else 2
			color = TRUE_COLOR if self.contact.final_hit[roi_id][frame_index] else ANALYSIS_COLOR
			cv2.polylines(frame, [_poly_int(self.contact.analysis_polygons[roi_id])], True, color, thickness, cv2.LINE_AA)
			center = np.mean(self.contact.pinned_polygons[roi_id], axis=0)
			label = f"{roi_id[-1].upper()} / {self.contact.role_labels[roi_id]}"
			cv2.putText(frame, label, (int(center[0] - 30), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, PIN_COLORS[roi_id], 2, cv2.LINE_AA)

		trail_start = max(frame_index - int(round(float(self.fps_spin.value()))), 0)
		trail = np.column_stack([
			self.contact.nose_x[trail_start : frame_index + 1],
			self.contact.nose_y[trail_start : frame_index + 1],
		])
		trail = trail[np.isfinite(trail).all(axis=1)]
		if trail.shape[0] >= 2:
			trail_points = np.rint(trail).astype(np.int32).reshape((-1, 1, 2))
			cv2.polylines(frame, [trail_points], False, (235, 235, 235), 1, cv2.LINE_AA)

		state_text, state_color = self._frame_state(frame_index)
		nx, ny = self.contact.nose_x[frame_index], self.contact.nose_y[frame_index]
		if np.isfinite(nx) and np.isfinite(ny):
			cv2.circle(frame, (int(round(nx)), int(round(ny))), 11, state_color, -1, cv2.LINE_AA)
			cv2.circle(frame, (int(round(nx)), int(round(ny))), 13, (20, 20, 20), 2, cv2.LINE_AA)
			cv2.putText(frame, "Nose", (int(nx + 16), int(ny - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2, cv2.LINE_AA)
		bx, by = self.contact.body_x[frame_index], self.contact.body_y[frame_index]
		if np.isfinite(bx) and np.isfinite(by):
			point = (int(round(bx)), int(round(by)))
			cv2.drawMarker(frame, point, BODY_COLOR, cv2.MARKER_CROSS, 20, 3, cv2.LINE_AA)
			cv2.putText(frame, "Body_C", (point[0] + 14, point[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BODY_COLOR, 2, cv2.LINE_AA)

		cv2.rectangle(frame, (12, 12), (680, 70), (15, 15, 15), -1)
		cv2.putText(frame, state_text, (28, 53), cv2.FONT_HERSHEY_SIMPLEX, 1.05, state_color, 3, cv2.LINE_AA)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		self.canvas.set_rgb_frame(rgb, reset_view=reset_view)

		analysis_seconds = frame_index / max(float(self.fps_spin.value()), 1e-9)
		video_seconds = video_frame / max(float(self.fps_spin.value()), 1e-9)
		self.header_label.setText(
			f"{row['condition']} | {row['subject_id']} | {str(row['phase']).upper()} n{row['session_n_norm']} "
			f"| analysis frame {frame_index} ({analysis_seconds:.2f} s) / "
			f"video+pose frame {video_frame} ({video_seconds:.2f} s) | {self.current_video.name}"
		)
		left = self.contact.role_labels["chamber_l"]
		right = self.contact.role_labels["chamber_r"]
		self.frame_status.setText(
			f"{state_text}    Left={left}: raw {int(self.contact.raw_hit['chamber_l'][frame_index])}, "
			f"final {int(self.contact.final_hit['chamber_l'][frame_index])}    "
			f"Right={right}: raw {int(self.contact.raw_hit['chamber_r'][frame_index])}, "
			f"final {int(self.contact.final_hit['chamber_r'][frame_index])}    "
			f"manual L/R={int(self.left_override[frame_index])}/{int(self.right_override[frame_index])}    "
			f"gap-filled={int(self.contact.gap_filled['chamber_l'][frame_index] or self.contact.gap_filled['chamber_r'][frame_index])}    "
			f"Body_C in pinned cup={int(self.contact.body_in_cup[frame_index])}    "
			f"inferred image={self.contact.inferred_width:.1f}x{self.contact.inferred_height:.1f}, "
			f"video={self.video_width}x{self.video_height}"
		)
		background = "#fffbd0" if state_color == MANUAL_TRUE_COLOR else "#ffd9dd" if state_color == MANUAL_FALSE_COLOR else "#f4dcff" if state_color == GAP_FILLED_COLOR else "#dff5df" if state_color == TRUE_COLOR else "#fff4cf" if state_color == SHORT_BOUT_COLOR else "#f8d9d9" if state_color == BODY_EXCLUDED_COLOR else "#eceff1"
		self.frame_status.setStyleSheet(f"padding: 5px; background: {background};")

	def _load_reviews(self) -> dict[str, dict[str, str]]:
		if not self.review_path.exists():
			return {}
		try:
			with self.review_path.open("r", encoding="utf-8-sig", newline="") as handle:
				return {row["session_key"]: row for row in csv.DictReader(handle) if row.get("session_key")}
		except (OSError, csv.Error, KeyError):
			return {}

	def _load_review_fields(self, session_key: str) -> None:
		row = self.review_rows.get(session_key, {})
		status = row.get("status", "Unreviewed")
		index = self.review_status.findText(status)
		self.review_status.setCurrentIndex(index if index >= 0 else 0)
		self.review_note.setText(row.get("note", ""))

	def _save_review(self, *, refresh_list: bool = True) -> None:
		if self.current_row_index is None:
			return
		session_key = str(self.sessions.loc[self.current_row_index, "session_key"])
		self.review_rows[session_key] = {
			"session_key": session_key,
			"status": self.review_status.currentText(),
			"note": self.review_note.text().strip(),
			"reviewed_at": datetime.now().isoformat(timespec="seconds"),
		}
		self.review_path.parent.mkdir(parents=True, exist_ok=True)
		with self.review_path.open("w", encoding="utf-8", newline="") as handle:
			writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
			writer.writeheader()
			writer.writerows(self.review_rows[key] for key in sorted(self.review_rows))
		self.statusBar().showMessage(f"Saved review to {self.review_path}")
		if refresh_list:
			current_list_row = self.session_list.currentRow()
			self._refresh_session_list()
			if current_list_row >= 0:
				self.session_list.setCurrentRow(min(current_list_row, self.session_list.count() - 1))

	def _save_all(self) -> None:
		self._persist_current_overrides(refresh_list=False)
		self._save_review(refresh_list=True)

	def _rebuild_barplots(self) -> None:
		if self.rebuild_process is not None and self.rebuild_process.state() != QtCore.QProcess.ProcessState.NotRunning:
			self.statusBar().showMessage("Barplot aggregation is already running.")
			return
		self._persist_current_overrides(refresh_list=False)
		phase_text = self.phase_combo.currentText().lower()
		phases = "soc,nov" if phase_text == "soc + nov" else phase_text
		arguments = [
			str(ROOT / "analysis/run/three_chamber/preference_bars.py"),
			"--keypoint", "Nose",
			"--roi-mode", "contact",
			"--radius-scale", f"{1.0 + float(self.buffer_spin.value()) / 100.0:g}",
			"--phases", phases,
			"--fps", f"{float(self.fps_spin.value()):g}",
			"--max-minutes", f"{float(self.minutes_spin.value()):g}",
			"--max-gap-seconds", f"{float(self.gap_spin.value()):g}",
			"--min-bout-seconds", f"{float(self.bout_spin.value()):g}",
			"--contact-overrides", str(self.override_path),
		]
		if self.session_combo.currentText() == "n1":
			arguments.extend(["--session-n", "1"])
		if not self.exclude_body_check.isChecked():
			arguments.append("--no-exclude-body-in-cup")
		self.rebuild_process = QtCore.QProcess(self)
		self.rebuild_process.setWorkingDirectory(str(ROOT))
		self.rebuild_process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
		self.rebuild_process.finished.connect(self._barplots_finished)
		self.rebuild_process.start(sys.executable, arguments)
		self.statusBar().showMessage("Rebuilding reviewed barplots...")

	def _barplots_finished(self, exit_code: int, _exit_status: QtCore.QProcess.ExitStatus) -> None:
		if self.rebuild_process is None:
			return
		output = bytes(self.rebuild_process.readAllStandardOutput()).decode("utf-8", errors="replace")
		if exit_code == 0:
			self.statusBar().showMessage("Reviewed barplots rebuilt successfully.")
			QtWidgets.QMessageBox.information(self, "Aggregation complete", "Reviewed barplots and CSV summaries were rebuilt.")
		else:
			QtWidgets.QMessageBox.critical(self, "Aggregation failed", output[-4000:] or f"Exit code: {exit_code}")

	def closeEvent(self, event: QtGui.QCloseEvent) -> None:
		self._release_capture()
		event.accept()


def _self_check(
	manifest_path: Path,
	preprocess_path: Path,
	roi_vertices_path: Path,
	roi_summary_path: Path,
) -> int:
	sessions = _load_sessions(manifest_path, preprocess_path)
	roi_vertices = pd.read_csv(roi_vertices_path)
	roi_summary = pd.read_csv(roi_summary_path)
	matched = sessions[(sessions["phase"].isin(["soc", "nov"])) & (sessions["session_n_norm"] == "1")]
	if matched.empty:
		raise RuntimeError("No n1 SOC/NOV sessions found")
	row = matched.iloc[0]
	contact = _build_contact_data(
		row,
		roi_vertices,
		roi_summary,
		buffer_percent=20.0,
		max_gap_seconds=0.5,
		min_bout_seconds=0.5,
		fps=30.0,
		max_minutes=5.0,
		exclude_body_in_cup=True,
	)
	print(f"Python: {sys.executable}")
	print(f"OpenCV: {cv2.__version__}")
	print(f"PySide6: {QtCore.__version__}")
	print(f"Sessions available: {len(sessions)}")
	print(f"Checked: {row['session_key']}")
	print(f"Frames: {contact.frame_count}")
	print(f"Left final TRUE: {int(contact.final_hit['chamber_l'].sum())}")
	print(f"Right final TRUE: {int(contact.final_hit['chamber_r'].sum())}")
	print("Contact review GUI self-check passed.")
	return 0


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Review three-chamber contact classifications overlaid on source videos.")
	parser.add_argument("--video-dir", default="", help="Optional video root to scan recursively on startup.")
	parser.add_argument("--manifest", default="output/3chamber/manifest/session_manifest.csv")
	parser.add_argument("--preprocess-summary", default="output/3chamber/preprocessed/preprocess_summary.csv")
	parser.add_argument("--roi-vertices", default="output/3chamber/roi/roi_vertices.csv")
	parser.add_argument("--roi-summary", default="output/3chamber/roi/roi_summary.csv")
	parser.add_argument("--review-output", default="output/3chamber/contact_review/contact_review.csv")
	parser.add_argument("--contact-overrides", default="output/3chamber/contact_review/contact_overrides.csv")
	parser.add_argument("--self-check", action="store_true", help="Validate dependencies and one session without opening the GUI.")
	args = parser.parse_args(argv)
	manifest_path = _to_abs_path(args.manifest)
	preprocess_path = _to_abs_path(args.preprocess_summary)
	roi_vertices_path = _to_abs_path(args.roi_vertices)
	roi_summary_path = _to_abs_path(args.roi_summary)
	review_path = _to_abs_path(args.review_output)
	override_path = _to_abs_path(args.contact_overrides)
	for path in (manifest_path, preprocess_path, roi_vertices_path, roi_summary_path):
		if not path.exists():
			raise FileNotFoundError(path)
	if args.self_check:
		return _self_check(manifest_path, preprocess_path, roi_vertices_path, roi_summary_path)
	video_dir = Path(args.video_dir).expanduser().resolve() if args.video_dir else None
	app = QtWidgets.QApplication(sys.argv[:1])
	app.setApplicationName("Three-chamber contact review")
	window = ContactReviewWindow(
		manifest_path=manifest_path,
		preprocess_path=preprocess_path,
		roi_vertices_path=roi_vertices_path,
		roi_summary_path=roi_summary_path,
		review_path=review_path,
		override_path=override_path,
		video_dir=video_dir,
	)
	window.show()
	return int(app.exec())


if __name__ == "__main__":
	raise SystemExit(main())
