from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
from PySide6 import QtCore, QtGui, QtWidgets


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from analysis.paths import RAW_DATA
from analysis.core.three_chamber.build_manifest import _normalize_stem


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
ROI_ORDER = ("chamber_l", "chamber_r", "wall")
ROI_LABELS = {
	"chamber_l": "Left cup",
	"chamber_r": "Right cup",
	"wall": "Wall",
}
ROI_COUNTS = {
	"chamber_l": 10,
	"chamber_r": 10,
	"wall": 4,
}
ROI_COLORS = {
	"chamber_l": "#00c8ff",
	"chamber_r": "#ff9d1c",
	"wall": "#9ce04a",
}
CSV_COLUMNS = ("id", "frame", "x", "y", "x_norm", "y_norm")


def _canonical_key(path: Path) -> str:
	return _normalize_stem(path.stem).lower()


def _scan_videos(root: Path) -> list[Path]:
	return sorted(
		(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
		key=lambda path: path.as_posix().lower(),
	)


def _row_coordinate(row: dict[str, str], axis: str) -> float:
	norm_key = f"{axis}_norm"
	try:
		norm_value = float(row.get(norm_key, "nan"))
		if math.isfinite(norm_value):
			return norm_value
	except (TypeError, ValueError):
		pass
	return float(row[axis])


def _reduce_polygon_rows(rows: list[dict[str, str]], target_count: int) -> list[dict[str, str]]:
	rows = list(rows)
	while len(rows) > target_count:
		try:
			points = [(_row_coordinate(row, "x"), _row_coordinate(row, "y")) for row in rows]
		except (KeyError, TypeError, ValueError):
			return rows[:target_count]
		contributions: list[float] = []
		for index, point in enumerate(points):
			previous = points[index - 1]
			following = points[(index + 1) % len(points)]
			before = math.dist(previous, point) + math.dist(point, following)
			after = math.dist(previous, following)
			contributions.append(before - after)
		rows.pop(min(range(len(rows)), key=contributions.__getitem__))
	return rows


def _group_pins_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
	grouped = {roi_id: [] for roi_id in ROI_ORDER}
	standard_rows = [row for row in rows if str(row.get("id", "")).strip().lower() in grouped]
	if standard_rows:
		for row in standard_rows:
			grouped[str(row["id"]).strip().lower()].append(row)
		return grouped

	legacy_rows = [row for row in rows if re.fullmatch(r"[a-z]", str(row.get("id", "")).strip().lower())]
	if len(legacy_rows) < 20:
		return grouped

	if len(legacy_rows) >= 24:
		cup_rows = legacy_rows[:-4]
		grouped["wall"] = legacy_rows[-4:]
	else:
		cup_rows = legacy_rows

	try:
		x_values = [_row_coordinate(row, "x") for row in cup_rows]
	except (KeyError, TypeError, ValueError):
		x_values = [float(index) for index in range(len(cup_rows))]
	sorted_x = sorted(set(x_values))
	if len(sorted_x) < 2:
		return grouped
	largest_gap_index = max(range(len(sorted_x) - 1), key=lambda index: sorted_x[index + 1] - sorted_x[index])
	split_x = (sorted_x[largest_gap_index] + sorted_x[largest_gap_index + 1]) / 2.0
	left_rows = [row for row, x_value in zip(cup_rows, x_values) if x_value < split_x]
	right_rows = [row for row, x_value in zip(cup_rows, x_values) if x_value >= split_x]
	grouped["chamber_l"] = _reduce_polygon_rows(left_rows, ROI_COUNTS["chamber_l"])
	grouped["chamber_r"] = _reduce_polygon_rows(right_rows, ROI_COUNTS["chamber_r"])
	return grouped


def _pins_counts(path: Path) -> dict[str, int]:
	counts = {roi_id: 0 for roi_id in ROI_ORDER}
	if not path.exists():
		return counts
	try:
		with path.open("r", encoding="utf-8-sig", newline="") as handle:
			grouped = _group_pins_rows(list(csv.DictReader(handle)))
			for roi_id in ROI_ORDER:
				counts[roi_id] = len(grouped[roi_id])
	except (OSError, csv.Error):
		return {roi_id: 0 for roi_id in ROI_ORDER}
	return counts


def _is_complete_pins(path: Path) -> bool:
	counts = _pins_counts(path)
	return all(counts[roi_id] == ROI_COUNTS[roi_id] for roi_id in ROI_ORDER)


def _write_pins_csv(
	path: Path,
	points: dict[str, list[tuple[float, float]]],
	*,
	frame_index: int,
	image_width: int,
	image_height: int,
) -> None:
	if image_width <= 0 or image_height <= 0:
		raise ValueError("Image dimensions must be positive")
	path.parent.mkdir(parents=True, exist_ok=True)
	temp_path = path.with_suffix(path.suffix + ".tmp")
	try:
		with temp_path.open("w", encoding="utf-8", newline="") as handle:
			writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
			writer.writeheader()
			for roi_id in ROI_ORDER:
				for x, y in points[roi_id]:
					x_int = int(round(x))
					y_int = int(round(y))
					writer.writerow(
						{
							"id": roi_id,
							"frame": int(frame_index),
							"x": x_int,
							"y": y_int,
							"x_norm": f"{x_int / image_width:.6f}",
							"y_norm": f"{y_int / image_height:.6f}",
						}
					)
		os.replace(temp_path, path)
	finally:
		if temp_path.exists():
			temp_path.unlink()


def _path_similarity(video: Path, pins: Path) -> int:
	video_parts = {part.lower() for part in video.parent.parts}
	pins_parts = {part.lower() for part in pins.parent.parts}
	common_parts = video_parts & pins_parts
	score = len(common_parts)
	for part in common_parts:
		if re.fullmatch(r"\d{8}", part):
			score += 10
		elif part in {"control", "vpa"}:
			score += 5
	return score


class DraggablePointItem(QtWidgets.QGraphicsEllipseItem):
	def __init__(
		self,
		*,
		roi_id: str,
		point_index: int,
		x: float,
		y: float,
		radius: float,
		color: QtGui.QColor,
		on_moved,
		on_released,
	) -> None:
		super().__init__(-radius, -radius, radius * 2, radius * 2)
		self.roi_id = roi_id
		self.point_index = point_index
		self._on_moved = on_moved
		self._on_released = on_released
		self._ready = False
		self.setPen(QtGui.QPen(QtGui.QColor("black"), 1))
		self.setBrush(QtGui.QBrush(color))
		self.setFlags(
			QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
			| QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
			| QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
			| QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
		)
		self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
		self.setZValue(10)
		self.setPos(x, y)
		self._ready = True

	def itemChange(self, change, value):
		if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene() is not None:
			rect = self.scene().sceneRect()
			value = QtCore.QPointF(
				min(max(value.x(), rect.left()), rect.right()),
				min(max(value.y(), rect.top()), rect.bottom()),
			)
		result = super().itemChange(change, value)
		if self._ready and change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
			point = self.pos()
			self._on_moved(self.roi_id, self.point_index, float(point.x()), float(point.y()))
		return result

	def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
		super().mouseReleaseEvent(event)
		QtCore.QTimer.singleShot(0, self._on_released)


class RoiCanvas(QtWidgets.QGraphicsView):
	point_clicked = QtCore.Signal(float, float)
	point_moved = QtCore.Signal(str, int, float, float)
	point_drag_finished = QtCore.Signal()
	undo_requested = QtCore.Signal()

	def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
		super().__init__(parent)
		self.setScene(QtWidgets.QGraphicsScene(self))
		self.setBackgroundBrush(QtGui.QColor("#151719"))
		self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
		self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
		self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
		self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
		self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
		self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
		self._rgb_frame = None
		self._points: dict[str, list[tuple[float, float]]] = {roi_id: [] for roi_id in ROI_ORDER}
		self._active_roi = "chamber_l"
		self._panning = False
		self._pan_position = QtCore.QPoint()

	def set_frame(self, rgb_frame, *, reset_view: bool = False) -> None:
		self._rgb_frame = rgb_frame.copy()
		self._redraw_scene()
		if reset_view:
			self.fit_image()

	def set_points(self, points: dict[str, list[tuple[float, float]]], active_roi: str) -> None:
		self._points = points
		self._active_roi = active_roi
		self._redraw_scene()

	def _redraw_scene(self) -> None:
		scene = self.scene()
		scene.clear()
		if self._rgb_frame is None:
			return
		height, width = self._rgb_frame.shape[:2]
		stride = int(self._rgb_frame.strides[0])
		image = QtGui.QImage(
			self._rgb_frame.data,
			width,
			height,
			stride,
			QtGui.QImage.Format.Format_RGB888,
		).copy()
		scene.addPixmap(QtGui.QPixmap.fromImage(image))
		scene.setSceneRect(0, 0, width, height)

		for roi_id in ROI_ORDER:
			points = self._points[roi_id]
			color = QtGui.QColor(ROI_COLORS[roi_id])
			line_width = 4.0 if roi_id == self._active_roi else 2.5
			pen = QtGui.QPen(color, line_width)
			pen.setCosmetic(True)
			for index in range(1, len(points)):
				x1, y1 = points[index - 1]
				x2, y2 = points[index]
				scene.addLine(x1, y1, x2, y2, pen)
			if len(points) == ROI_COUNTS[roi_id]:
				x1, y1 = points[-1]
				x2, y2 = points[0]
				scene.addLine(x1, y1, x2, y2, pen)

			for point_index, (x, y) in enumerate(points, start=1):
				radius = 7.0 if roi_id == self._active_roi else 5.5
				point_item = DraggablePointItem(
					roi_id=roi_id,
					point_index=point_index - 1,
					x=x,
					y=y,
					radius=radius,
					color=color,
					on_moved=self.point_moved.emit,
					on_released=self.point_drag_finished.emit,
				)
				scene.addItem(point_item)
				text_item = scene.addText(str(point_index), QtGui.QFont("Segoe UI", 8, QtGui.QFont.Weight.Bold))
				text_item.setDefaultTextColor(QtGui.QColor("white"))
				text_item.setPos(x + 7, y - 20)
				text_item.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)

	def fit_image(self) -> None:
		if self._rgb_frame is None:
			return
		self.resetTransform()
		self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

	def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
		if self._rgb_frame is None:
			return
		factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
		current_scale = self.transform().m11()
		if factor > 1 and current_scale >= 8.0:
			return
		if factor < 1 and current_scale <= 0.15:
			return
		self.scale(factor, factor)

	def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
		if event.button() == QtCore.Qt.MouseButton.MiddleButton:
			self._panning = True
			self._pan_position = event.position().toPoint()
			self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
			event.accept()
			return
		if event.button() == QtCore.Qt.MouseButton.RightButton:
			self.undo_requested.emit()
			event.accept()
			return
		if event.button() == QtCore.Qt.MouseButton.LeftButton and self._rgb_frame is not None:
			clicked_item = self.itemAt(event.position().toPoint())
			if isinstance(clicked_item, DraggablePointItem):
				super().mousePressEvent(event)
				return
			point = self.mapToScene(event.position().toPoint())
			if self.sceneRect().contains(point):
				self.point_clicked.emit(float(point.x()), float(point.y()))
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
			self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
			event.accept()
			return
		super().mouseReleaseEvent(event)


class RoiPinningWindow(QtWidgets.QMainWindow):
	def __init__(self, *, video_dir: Path | None = None, pins_root: Path | None = None, initial_frame: int = 0) -> None:
		super().__init__()
		self.setWindowTitle("Three-chamber ROI pinning")
		self.resize(1540, 920)
		self.setMinimumSize(1120, 700)

		self.video_root = video_dir.resolve() if video_dir else None
		self.pins_root = (pins_root or RAW_DATA["3chamber"]).resolve()
		self.initial_frame = max(int(initial_frame), 0)
		self.videos: list[Path] = []
		self.pins_index: dict[str, list[Path]] = defaultdict(list)
		self.current_index = -1
		self.current_video: Path | None = None
		self.current_pins_path: Path | None = None
		self.capture: cv2.VideoCapture | None = None
		self.frame_count = 0
		self.fps = 0.0
		self.frame_index = 0
		self.image_width = 0
		self.image_height = 0
		self.points: dict[str, list[tuple[float, float]]] = {roi_id: [] for roi_id in ROI_ORDER}
		self.active_roi = "chamber_l"
		self.dirty = False
		self._changing_selection = False

		self.frame_timer = QtCore.QTimer(self)
		self.frame_timer.setSingleShot(True)
		self.frame_timer.setInterval(80)
		self.frame_timer.timeout.connect(self._read_selected_frame)

		self._build_ui()
		self._bind_shortcuts()
		self._rebuild_pins_index()
		if self.video_root and self.video_root.exists():
			self._set_videos(_scan_videos(self.video_root))

	def _build_ui(self) -> None:
		central = QtWidgets.QWidget()
		self.setCentralWidget(central)
		root_layout = QtWidgets.QVBoxLayout(central)
		root_layout.setContentsMargins(8, 8, 8, 8)
		root_layout.setSpacing(6)

		path_row = QtWidgets.QHBoxLayout()
		self.video_dir_edit = QtWidgets.QLineEdit(str(self.video_root or ""))
		self.video_dir_edit.setReadOnly(True)
		self.pins_root_edit = QtWidgets.QLineEdit(str(self.pins_root))
		self.pins_root_edit.setReadOnly(True)
		video_folder_button = QtWidgets.QPushButton("Video folder")
		video_folder_button.clicked.connect(self._choose_video_folder)
		add_files_button = QtWidgets.QPushButton("Add video files")
		add_files_button.clicked.connect(self._choose_video_files)
		pins_root_button = QtWidgets.QPushButton("Pins root")
		pins_root_button.clicked.connect(self._choose_pins_root)
		path_row.addWidget(video_folder_button)
		path_row.addWidget(self.video_dir_edit, 2)
		path_row.addWidget(add_files_button)
		path_row.addSpacing(12)
		path_row.addWidget(pins_root_button)
		path_row.addWidget(self.pins_root_edit, 2)
		root_layout.addLayout(path_row)

		splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
		root_layout.addWidget(splitter, 1)
		left_panel = QtWidgets.QWidget()
		left_layout = QtWidgets.QVBoxLayout(left_panel)
		left_layout.setContentsMargins(0, 0, 4, 0)
		nav_row = QtWidgets.QHBoxLayout()
		previous_button = QtWidgets.QPushButton("Previous")
		previous_button.clicked.connect(lambda: self._move_video(-1))
		next_button = QtWidgets.QPushButton("Next")
		next_button.clicked.connect(lambda: self._move_video(1))
		next_unfinished_button = QtWidgets.QPushButton("Next unfinished")
		next_unfinished_button.clicked.connect(self._next_unfinished)
		nav_row.addWidget(previous_button)
		nav_row.addWidget(next_button)
		nav_row.addWidget(next_unfinished_button)
		left_layout.addLayout(nav_row)

		self.video_list = QtWidgets.QListWidget()
		self.video_list.setFont(QtGui.QFont("Consolas", 9))
		self.video_list.currentRowChanged.connect(self._on_video_selected)
		left_layout.addWidget(self.video_list, 1)

		output_group = QtWidgets.QGroupBox("Current pins file")
		output_layout = QtWidgets.QVBoxLayout(output_group)
		self.output_path_edit = QtWidgets.QLineEdit()
		self.output_path_edit.setReadOnly(True)
		save_as_button = QtWidgets.QPushButton("Save as...")
		save_as_button.clicked.connect(self._save_as)
		output_layout.addWidget(self.output_path_edit)
		output_layout.addWidget(save_as_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
		left_layout.addWidget(output_group)

		viewer_panel = QtWidgets.QWidget()
		viewer_layout = QtWidgets.QVBoxLayout(viewer_panel)
		viewer_layout.setContentsMargins(4, 0, 0, 0)
		viewer_header = QtWidgets.QHBoxLayout()
		self.video_info_label = QtWidgets.QLabel("Load a video folder or video files.")
		font = self.video_info_label.font()
		font.setBold(True)
		self.video_info_label.setFont(font)
		hint_label = QtWidgets.QLabel("Pin the OUTER WIRE CUP EDGE, not the dark lid")
		hint_label.setStyleSheet("color: #b34a00; font-weight: 600;")
		viewer_header.addWidget(self.video_info_label)
		viewer_header.addStretch(1)
		viewer_header.addWidget(hint_label)
		viewer_layout.addLayout(viewer_header)

		self.canvas = RoiCanvas()
		self.canvas.point_clicked.connect(self._add_point)
		self.canvas.point_moved.connect(self._move_point)
		self.canvas.point_drag_finished.connect(self._finish_point_drag)
		self.canvas.undo_requested.connect(self._undo_active)
		viewer_layout.addWidget(self.canvas, 1)

		frame_row = QtWidgets.QHBoxLayout()
		for label, delta in (("-300", -300), ("-30", -30)):
			button = QtWidgets.QPushButton(label)
			button.clicked.connect(lambda _checked=False, step=delta: self._step_frame(step))
			frame_row.addWidget(button)
		self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
		self.frame_slider.setRange(0, 1)
		self.frame_slider.valueChanged.connect(self._schedule_frame)
		frame_row.addWidget(self.frame_slider, 1)
		for label, delta in (("+30", 30), ("+300", 300)):
			button = QtWidgets.QPushButton(label)
			button.clicked.connect(lambda _checked=False, step=delta: self._step_frame(step))
			frame_row.addWidget(button)
		fit_button = QtWidgets.QPushButton("Fit")
		fit_button.clicked.connect(self.canvas.fit_image)
		frame_row.addWidget(fit_button)
		viewer_layout.addLayout(frame_row)

		roi_row = QtWidgets.QHBoxLayout()
		self.roi_button_group = QtWidgets.QButtonGroup(self)
		self.roi_buttons: dict[str, QtWidgets.QRadioButton] = {}
		self.count_labels: dict[str, QtWidgets.QLabel] = {}
		for roi_id in ROI_ORDER:
			button = QtWidgets.QRadioButton(ROI_LABELS[roi_id])
			button.toggled.connect(lambda checked, selected=roi_id: self._select_roi(selected) if checked else None)
			self.roi_button_group.addButton(button)
			self.roi_buttons[roi_id] = button
			count_label = QtWidgets.QLabel()
			count_label.setStyleSheet(f"color: {ROI_COLORS[roi_id]}; font-weight: 600;")
			self.count_labels[roi_id] = count_label
			roi_row.addWidget(button)
			roi_row.addWidget(count_label)
			roi_row.addSpacing(8)
		self.roi_buttons["chamber_l"].setChecked(True)
		undo_button = QtWidgets.QPushButton("Undo")
		undo_button.clicked.connect(self._undo_active)
		clear_button = QtWidgets.QPushButton("Clear active")
		clear_button.clicked.connect(self._clear_active)
		clear_cups_button = QtWidgets.QPushButton("Clear both cups")
		clear_cups_button.clicked.connect(self._clear_cups)
		save_button = QtWidgets.QPushButton("Save (Ctrl+S)")
		save_button.clicked.connect(self._save)
		roi_row.addWidget(undo_button)
		roi_row.addWidget(clear_button)
		roi_row.addWidget(clear_cups_button)
		roi_row.addStretch(1)
		roi_row.addWidget(save_button)
		viewer_layout.addLayout(roi_row)
		self._update_point_counts()

		splitter.addWidget(left_panel)
		splitter.addWidget(viewer_panel)
		splitter.setSizes([340, 1200])
		self.statusBar().showMessage("Pin the OUTER WIRE CUP EDGE, not the dark lid.")

	def _bind_shortcuts(self) -> None:
		shortcuts = [
			("Ctrl+S", self._save),
			("Backspace", self._undo_active),
			("Delete", self._clear_active),
			("Return", lambda: self._move_video(1)),
			("Escape", self.canvas.fit_image),
			("L", lambda: self.roi_buttons["chamber_l"].setChecked(True)),
			("R", lambda: self.roi_buttons["chamber_r"].setChecked(True)),
			("W", lambda: self.roi_buttons["wall"].setChecked(True)),
		]
		self.shortcuts = []
		for key_sequence, callback in shortcuts:
			shortcut = QtGui.QShortcut(QtGui.QKeySequence(key_sequence), self)
			shortcut.activated.connect(callback)
			self.shortcuts.append(shortcut)

	def _select_roi(self, roi_id: str) -> None:
		self.active_roi = roi_id
		self.canvas.set_points(self.points, self.active_roi)

	def _choose_video_folder(self) -> None:
		selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Select video root", str(self.video_root or ROOT))
		if not selected:
			return
		self.video_root = Path(selected).resolve()
		self.video_dir_edit.setText(str(self.video_root))
		self._set_videos(_scan_videos(self.video_root))

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
		if self.video_root is None:
			self.video_root = Path(os.path.commonpath([str(path.parent) for path in paths])).resolve()
			self.video_dir_edit.setText(str(self.video_root))
		self._set_videos(sorted(set(self.videos + paths), key=lambda path: path.as_posix().lower()))

	def _choose_pins_root(self) -> None:
		selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Select pins root", str(self.pins_root))
		if not selected:
			return
		self.pins_root = Path(selected).resolve()
		self.pins_root_edit.setText(str(self.pins_root))
		self._rebuild_pins_index()
		self._refresh_video_list()
		if self.current_video:
			self.current_pins_path = self._resolve_pins_path(self.current_video)
			self.output_path_edit.setText(str(self.current_pins_path))

	def _set_videos(self, videos: Iterable[Path]) -> None:
		if not self._confirm_discard_or_save():
			return
		self.videos = list(videos)
		self.current_index = -1
		self._refresh_video_list()
		if self.videos:
			self.video_list.setCurrentRow(0)
			self.statusBar().showMessage(f"Loaded {len(self.videos)} videos.")
		else:
			self.statusBar().showMessage("No supported videos found.")

	def _rebuild_pins_index(self) -> None:
		self.pins_index = defaultdict(list)
		if not self.pins_root.exists():
			return
		for path in self.pins_root.rglob("*_pins.csv"):
			if path.is_file():
				self.pins_index[_canonical_key(path)].append(path.resolve())

	def _default_new_pins_path(self, video: Path) -> Path:
		filename = f"{video.stem}_pins.csv"
		if self.pins_root.name.lower() == "object":
			return self.pins_root / filename
		if self.video_root:
			try:
				rel_parent = video.parent.relative_to(self.video_root)
			except ValueError:
				rel_parent = Path()
			return self.pins_root / rel_parent / "object" / filename
		return self.pins_root / filename

	def _resolve_pins_path(self, video: Path) -> Path:
		candidates = self.pins_index.get(_canonical_key(video), [])
		if len(candidates) == 1:
			return candidates[0]
		if len(candidates) > 1:
			scored = sorted(
				((_path_similarity(video, path), path) for path in candidates),
				key=lambda item: (-item[0], item[1].as_posix()),
			)
			if len(scored) == 1 or scored[0][0] > scored[1][0]:
				return scored[0][1]
		return self._default_new_pins_path(video)

	def _video_status(self, video: Path) -> str:
		pins_path = self._resolve_pins_path(video)
		if _is_complete_pins(pins_path):
			return "OK"
		if pins_path.exists():
			return "!!"
		return "  "

	def _refresh_video_list(self) -> None:
		selected = self.current_index
		self._changing_selection = True
		self.video_list.clear()
		for video in self.videos:
			self.video_list.addItem(f"[{self._video_status(video)}] {video.name}")
		if 0 <= selected < len(self.videos):
			self.video_list.setCurrentRow(selected)
		self._changing_selection = False

	def _on_video_selected(self, index: int) -> None:
		if self._changing_selection or index < 0 or index == self.current_index:
			return
		previous = self.current_index
		if not self._confirm_discard_or_save():
			self._changing_selection = True
			self.video_list.setCurrentRow(previous)
			self._changing_selection = False
			return
		self._load_video(index)

	def _move_video(self, delta: int) -> None:
		if not self.videos:
			return
		target = min(max(self.current_index + delta, 0), len(self.videos) - 1)
		if target != self.current_index:
			self.video_list.setCurrentRow(target)

	def _next_unfinished(self) -> None:
		if not self.videos:
			return
		for offset in range(1, len(self.videos) + 1):
			index = (self.current_index + offset) % len(self.videos)
			if not _is_complete_pins(self._resolve_pins_path(self.videos[index])):
				self.video_list.setCurrentRow(index)
				return
		self.statusBar().showMessage("All videos have complete pins files.")

	def _release_capture(self) -> None:
		if self.capture is not None:
			self.capture.release()
		self.capture = None

	def _load_video(self, index: int) -> None:
		self._release_capture()
		self.current_index = index
		self.current_video = self.videos[index]
		self.current_pins_path = self._resolve_pins_path(self.current_video)
		self.output_path_edit.setText(str(self.current_pins_path))
		self.capture = cv2.VideoCapture(str(self.current_video))
		if not self.capture.isOpened():
			QtWidgets.QMessageBox.critical(self, "Video error", f"Could not open:\n{self.current_video}")
			self._release_capture()
			return
		self.frame_count = max(int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
		self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
		self.image_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.image_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.frame_slider.blockSignals(True)
		self.frame_slider.setRange(0, max(self.frame_count - 1, 1))
		self.frame_index = min(self.initial_frame, self.frame_count - 1)
		self.frame_slider.setValue(self.frame_index)
		self.frame_slider.blockSignals(False)
		self.points = {roi_id: [] for roi_id in ROI_ORDER}
		self._load_existing_pins()
		self.dirty = False
		self._update_point_counts()
		self._read_frame(self.frame_index, reset_view=True)

	def _load_existing_pins(self) -> None:
		path = self.current_pins_path
		if path is None or not path.exists():
			return
		try:
			with path.open("r", encoding="utf-8-sig", newline="") as handle:
				rows = list(csv.DictReader(handle))
			grouped = _group_pins_rows(rows)
			for roi_id in ROI_ORDER:
				for row in grouped[roi_id]:
					try:
						x_norm = float(row.get("x_norm", "nan"))
						y_norm = float(row.get("y_norm", "nan"))
						x = x_norm * self.image_width if math.isfinite(x_norm) else float(row["x"])
						y = y_norm * self.image_height if math.isfinite(y_norm) else float(row["y"])
					except (TypeError, ValueError, KeyError):
						continue
					self.points[roi_id].append((x, y))
		except (OSError, csv.Error) as exc:
			QtWidgets.QMessageBox.warning(self, "Pins warning", f"Could not read existing pins:\n{path}\n\n{exc}")

	def _schedule_frame(self, frame_index: int) -> None:
		if self.capture is None:
			return
		self.frame_index = min(max(int(frame_index), 0), self.frame_count - 1)
		self.frame_timer.start()

	def _read_selected_frame(self) -> None:
		self._read_frame(self.frame_index)

	def _step_frame(self, delta: int) -> None:
		if self.capture is None:
			return
		self.frame_index = min(max(self.frame_index + delta, 0), self.frame_count - 1)
		self.frame_slider.blockSignals(True)
		self.frame_slider.setValue(self.frame_index)
		self.frame_slider.blockSignals(False)
		self._read_frame(self.frame_index)

	def _read_frame(self, frame_index: int, *, reset_view: bool = False) -> None:
		if self.capture is None:
			return
		self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
		ok, frame = self.capture.read()
		if not ok:
			self.statusBar().showMessage(f"Could not read frame {frame_index}.")
			return
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		seconds = frame_index / self.fps if self.fps > 0 else 0.0
		self.video_info_label.setText(
			f"{self.current_index + 1}/{len(self.videos)}  {self.current_video.name if self.current_video else ''}"
			f"  |  frame {frame_index}/{self.frame_count - 1}  ({seconds:.2f} s)"
		)
		self.canvas.set_frame(rgb, reset_view=reset_view)
		self.canvas.set_points(self.points, self.active_roi)

	def _add_point(self, x: float, y: float) -> None:
		roi_id = self.active_roi
		if len(self.points[roi_id]) >= ROI_COUNTS[roi_id]:
			self.statusBar().showMessage(f"{ROI_LABELS[roi_id]} already has {ROI_COUNTS[roi_id]} points. Clear or undo first.")
			return
		self.points[roi_id].append((x, y))
		self.dirty = True
		self._update_point_counts()
		if len(self.points[roi_id]) == ROI_COUNTS[roi_id]:
			if roi_id == "chamber_l":
				self.roi_buttons["chamber_r"].setChecked(True)
			elif roi_id == "chamber_r" and len(self.points["wall"]) != ROI_COUNTS["wall"]:
				self.roi_buttons["wall"].setChecked(True)
			self.statusBar().showMessage(f"Completed {ROI_LABELS[roi_id]}.")
		self.canvas.set_points(self.points, self.active_roi)

	def _undo_active(self) -> None:
		if self.points[self.active_roi]:
			self.points[self.active_roi].pop()
			self.dirty = True
			self._update_point_counts()
			self.canvas.set_points(self.points, self.active_roi)

	def _move_point(self, roi_id: str, point_index: int, x: float, y: float) -> None:
		if roi_id not in self.points or not 0 <= point_index < len(self.points[roi_id]):
			return
		self.points[roi_id][point_index] = (x, y)
		self.dirty = True
		self.statusBar().showMessage(f"Moved {ROI_LABELS[roi_id]} point {point_index + 1}.")

	def _finish_point_drag(self) -> None:
		self.canvas.set_points(self.points, self.active_roi)

	def _clear_active(self) -> None:
		roi_id = self.active_roi
		if not self.points[roi_id]:
			return
		answer = QtWidgets.QMessageBox.question(self, "Clear ROI", f"Clear all {ROI_LABELS[roi_id]} points?")
		if answer == QtWidgets.QMessageBox.StandardButton.Yes:
			self.points[roi_id] = []
			self.dirty = True
			self._update_point_counts()
			self.canvas.set_points(self.points, self.active_roi)

	def _clear_cups(self) -> None:
		if not (self.points["chamber_l"] or self.points["chamber_r"]):
			return
		answer = QtWidgets.QMessageBox.question(self, "Clear cups", "Clear both cup ROIs and keep wall points?")
		if answer == QtWidgets.QMessageBox.StandardButton.Yes:
			self.points["chamber_l"] = []
			self.points["chamber_r"] = []
			self.roi_buttons["chamber_l"].setChecked(True)
			self.dirty = True
			self._update_point_counts()
			self.canvas.set_points(self.points, self.active_roi)

	def _update_point_counts(self) -> None:
		for roi_id in ROI_ORDER:
			self.count_labels[roi_id].setText(f"{len(self.points[roi_id])}/{ROI_COUNTS[roi_id]}")

	def _validate_points(self) -> list[str]:
		return [
			f"{ROI_LABELS[roi_id]}: {len(self.points[roi_id])}/{ROI_COUNTS[roi_id]}"
			for roi_id in ROI_ORDER
			if len(self.points[roi_id]) != ROI_COUNTS[roi_id]
		]

	def _write_pins(self, path: Path) -> Path | None:
		issues = self._validate_points()
		if issues:
			QtWidgets.QMessageBox.critical(self, "Incomplete ROI", "Each ROI needs the exact point count:\n\n" + "\n".join(issues))
			return None
		try:
			_write_pins_csv(
				path,
				self.points,
				frame_index=self.frame_index,
				image_width=self.image_width,
				image_height=self.image_height,
			)
		except (OSError, ValueError) as exc:
			QtWidgets.QMessageBox.critical(self, "Save error", str(exc))
			return None
		self.dirty = False
		self._rebuild_pins_index()
		self._refresh_video_list()
		self.statusBar().showMessage(f"Updated {path.name}.")
		return path

	def _save(self) -> bool:
		if self.current_pins_path is None:
			return False
		return self._write_pins(self.current_pins_path) is not None

	def _save_as(self) -> None:
		if self.current_video is None:
			return
		initial = self.current_pins_path or self._default_new_pins_path(self.current_video)
		selected, _ = QtWidgets.QFileDialog.getSaveFileName(
			self,
			"Save pins CSV",
			str(initial),
			"CSV (*.csv)",
		)
		if not selected:
			return
		path = Path(selected).resolve()
		if not path.name.lower().endswith("_pins.csv"):
			path = path.with_name(f"{path.stem}_pins.csv")
		if self._write_pins(path):
			self.current_pins_path = path
			self.output_path_edit.setText(str(path))

	def _confirm_discard_or_save(self) -> bool:
		if not self.dirty:
			return True
		box = QtWidgets.QMessageBox(self)
		box.setWindowTitle("Unsaved changes")
		box.setText("Save changes before leaving this video?")
		box.setStandardButtons(
			QtWidgets.QMessageBox.StandardButton.Save
			| QtWidgets.QMessageBox.StandardButton.Discard
			| QtWidgets.QMessageBox.StandardButton.Cancel
		)
		answer = box.exec()
		if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
			return False
		if answer == QtWidgets.QMessageBox.StandardButton.Save:
			return self._save()
		return True

	def closeEvent(self, event: QtGui.QCloseEvent) -> None:
		if self._confirm_discard_or_save():
			self._release_capture()
			event.accept()
		else:
			event.ignore()


def _self_check() -> int:
	print(f"Python: {sys.executable}")
	print(f"OpenCV: {cv2.__version__}")
	print(f"PySide6: {QtCore.__version__}")
	print("ROI GUI dependencies are available.")
	return 0


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Batch GUI for pinning three-chamber cup and wall ROIs.")
	parser.add_argument("--video-dir", default="", help="Optional video root to scan recursively on startup.")
	parser.add_argument("--pins-root", default=str(RAW_DATA["3chamber"]), help="Root used to find and save *_pins.csv files.")
	parser.add_argument("--frame", type=int, default=0, help="Initial frame index for each video.")
	parser.add_argument("--self-check", action="store_true", help="Check GUI dependencies without opening a window.")
	args = parser.parse_args(argv)
	if args.self_check:
		return _self_check()

	video_dir = Path(args.video_dir).expanduser().resolve() if args.video_dir else None
	pins_root = Path(args.pins_root).expanduser()
	if not pins_root.is_absolute():
		pins_root = (ROOT / pins_root).resolve()
	app = QtWidgets.QApplication(sys.argv[:1])
	app.setApplicationName("Three-chamber ROI pinning")
	window = RoiPinningWindow(video_dir=video_dir, pins_root=pins_root, initial_frame=args.frame)
	window.show()
	return int(app.exec())


if __name__ == "__main__":
	raise SystemExit(main())
