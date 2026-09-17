from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from analysis.core.three_chamber.spatial_maps import _roi_label_position
from analysis.core.three_chamber.spatial_maps import _roi_output_tag

from analysis.core.three_chamber.spatial_maps import FIGURE_DPI

from analysis.core.three_chamber.spatial_maps import PHASE_LABELS

from analysis.core.three_chamber.spatial_maps import PHASE_TITLES

from analysis.core.three_chamber.spatial_maps import PHASE_FIGURE_TITLES

def _draw_overlay(
	ax: plt.Axes,
	*,
	occupancy_support: np.ndarray,
	object_circles: dict[str, tuple[float, float, float, float]],
	analysis_circles: dict[str, tuple[float, float, float, float]],
	object_polygons: dict[str, np.ndarray] | None = None,
	analysis_polygons: dict[str, np.ndarray] | None = None,
	x_max: float,
) -> None:
	def _envelope_mask(mask: np.ndarray) -> np.ndarray:
		# Keep the arena silhouette while filling internal unvisited holes.
		out = np.zeros_like(mask, dtype=bool)
		for y_idx in range(mask.shape[1]):
			x_idx = np.flatnonzero(mask[:, y_idx])
			if x_idx.size:
				out[x_idx.min() : x_idx.max() + 1, y_idx] = True
		for x_idx in range(mask.shape[0]):
			y_idx = np.flatnonzero(mask[x_idx, :])
			if y_idx.size:
				out[x_idx, y_idx.min() : y_idx.max() + 1] = True
		return out

	occupancy_envelope = _envelope_mask(occupancy_support)

	def _plot_heat_masked_circle(circle: tuple[float, float, float, float]) -> None:
		cx, cy, radius_x, radius_y = circle
		theta = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=True)
		xs = cx + radius_x * np.cos(theta)
		ys = cy + radius_y * np.sin(theta)
		bins_x, bins_y = occupancy_envelope.shape
		xi = np.floor((xs / x_max) * bins_x).astype(int)
		yi = np.floor(ys * bins_y).astype(int)
		in_bounds = (xs >= 0.0) & (xs <= x_max) & (ys >= 0.0) & (ys <= 1.0)
		xi = np.clip(xi, 0, bins_x - 1)
		yi = np.clip(yi, 0, bins_y - 1)
		has_occupancy = in_bounds & occupancy_envelope[xi, yi]
		plot_x = xs.copy()
		plot_y = ys.copy()
		plot_x[~has_occupancy] = np.nan
		plot_y[~has_occupancy] = np.nan
		ax.plot(plot_x, plot_y, color="#00E5FF", linewidth=1.45, linestyle="-")

	def _plot_heat_masked_polygon(poly: np.ndarray) -> None:
		closed = np.vstack([poly, poly[0]])
		bins_x, bins_y = occupancy_envelope.shape
		xi = np.floor((closed[:, 0] / x_max) * bins_x).astype(int)
		yi = np.floor(closed[:, 1] * bins_y).astype(int)
		in_bounds = (closed[:, 0] >= 0.0) & (closed[:, 0] <= x_max) & (closed[:, 1] >= 0.0) & (closed[:, 1] <= 1.0)
		xi = np.clip(xi, 0, bins_x - 1)
		yi = np.clip(yi, 0, bins_y - 1)
		has_occupancy = in_bounds & occupancy_envelope[xi, yi]
		plot_x = closed[:, 0].copy()
		plot_y = closed[:, 1].copy()
		plot_x[~has_occupancy] = np.nan
		plot_y[~has_occupancy] = np.nan
		ax.plot(plot_x, plot_y, color="#00E5FF", linewidth=1.45, linestyle="-")

	ax.add_patch(Rectangle((0.0, 0.0), x_max, 1.0, fill=False, edgecolor="white", linewidth=1.5))
	for roi_id in ("chamber_l", "chamber_r"):
		object_poly = (object_polygons or {}).get(roi_id)
		if object_poly is not None and object_poly.shape[0] >= 3:
			closed = np.vstack([object_poly, object_poly[0]])
			ax.plot(closed[:, 0], closed[:, 1], color="white", linewidth=1.2, linestyle=(0, (5, 4)))
		object_circle = object_circles.get(roi_id)
		if object_poly is None and object_circle is not None:
			cx, cy, radius_x, radius_y = object_circle
			ax.add_patch(
				Ellipse(
					(cx, cy),
					width=2.0 * radius_x,
					height=2.0 * radius_y,
					fill=False,
					edgecolor="white",
					linewidth=1.2,
					linestyle=(0, (5, 4)),
				)
			)
		analysis_poly = (analysis_polygons or {}).get(roi_id)
		if analysis_poly is not None and analysis_poly.shape[0] >= 3:
			_plot_heat_masked_polygon(analysis_poly)
		analysis_circle = analysis_circles.get(roi_id)
		if analysis_poly is None and analysis_circle is not None:
			_plot_heat_masked_circle(analysis_circle)

def _save_figure(
	fig: plt.Figure,
	out_path: Path,
	*,
	pad_inches: float = 0.06,
	square_canvas: bool = False,
) -> None:
	save_kwargs = {
		"dpi": FIGURE_DPI,
		"facecolor": "white",
	}
	if not square_canvas:
		save_kwargs["bbox_inches"] = "tight"
		save_kwargs["pad_inches"] = pad_inches
	fig.savefig(out_path, **save_kwargs)
	fig.savefig(out_path.with_suffix(".svg"), format="svg", **save_kwargs)

def _save_phase_figure(
	*,
	phase: str,
	condition_order: list[str],
	group_heatmaps: dict[tuple[str, str], np.ndarray],
	group_support_masks: dict[tuple[str, str], np.ndarray],
	group_counts: dict[tuple[str, str], int],
	group_object_circles: dict[tuple[str, str, str], tuple[float, float, float, float]],
	group_analysis_circles: dict[tuple[str, str, str], tuple[float, float, float, float]],
	group_object_polygons: dict[tuple[str, str, str], np.ndarray] | None,
	group_analysis_polygons: dict[tuple[str, str, str], np.ndarray] | None,
	output_dir: Path,
	keypoint: str,
	target_side: str,
	roi_radius_scale: float,
	file_suffix: str,
	x_max: float,
	display_width_height_ratio: float,
	font_scale: float = 1.0,
) -> Path | None:
	conditions = [condition for condition in condition_order if (phase, condition) in group_heatmaps]
	if not conditions:
		return None

	fig, axes = plt.subplots(
		1,
		len(conditions),
		figsize=(3.55 * len(conditions) + 0.45, 4.5),
		constrained_layout=False,
	)
	fig.subplots_adjust(left=0.035, right=0.94, bottom=0.055, top=0.77, wspace=0.09)
	axes = np.atleast_1d(axes).astype(object)
	cmap = plt.get_cmap("turbo").copy()
	phase_vmax = max(float(group_heatmaps[(phase, condition)].max()) for condition in conditions)
	norm = Normalize(vmin=0.0, vmax=max(phase_vmax, 1e-9))
	left_label, right_label = PHASE_LABELS.get(phase, ("L", "R"))
	im = None

	for ax, condition in zip(axes, conditions):
		heat = group_heatmaps[(phase, condition)]
		ax.set_facecolor(cmap(0.0))
		im = ax.imshow(heat.T, extent=(0.0, x_max, 1.0, 0.0), cmap=cmap, norm=norm)
		occupancy_support = group_support_masks.get((phase, condition), heat > 0.0)
		_draw_overlay(
			ax,
			occupancy_support=occupancy_support,
			object_circles={
				"chamber_l": group_object_circles.get((phase, condition, "chamber_l")),
				"chamber_r": group_object_circles.get((phase, condition, "chamber_r")),
			},
			analysis_circles={
				"chamber_l": group_analysis_circles.get((phase, condition, "chamber_l")),
				"chamber_r": group_analysis_circles.get((phase, condition, "chamber_r")),
			},
			object_polygons={
				"chamber_l": (group_object_polygons or {}).get((phase, condition, "chamber_l")),
				"chamber_r": (group_object_polygons or {}).get((phase, condition, "chamber_r")),
			},
			analysis_polygons={
				"chamber_l": (group_analysis_polygons or {}).get((phase, condition, "chamber_l")),
				"chamber_r": (group_analysis_polygons or {}).get((phase, condition, "chamber_r")),
			},
			x_max=x_max,
		)
		ax.set_xlim(0.0, x_max)
		ax.set_ylim(1.0, 0.0)
		ax.set_aspect("auto")
		ax.set_box_aspect(1.0)
		ax.set_xticks([])
		ax.set_yticks([])
		for spine in ax.spines.values():
			spine.set_color("#4A4A4A")
			spine.set_linewidth(0.8)
		ax.set_title(
			f"{condition} (n={group_counts.get((phase, condition), 0)})",
			fontsize=16 * font_scale,
			pad=7 * font_scale,
		)

		for roi_id, label in (("chamber_l", left_label), ("chamber_r", right_label)):
			position = _roi_label_position(
				phase=phase,
				condition=condition,
				roi_id=roi_id,
				group_object_circles=group_object_circles,
				group_object_polygons=group_object_polygons,
			)
			if position is not None:
				ax.text(
					position[0],
					position[1],
					label,
					ha="center",
					va="center",
					color="white",
					fontsize=16 * font_scale,
					fontweight="bold",
					zorder=8,
				)

	if im is not None:
		cax = inset_axes(
			axes[-1],
			width="2.8%",
			height="100%",
			loc="lower left",
			bbox_to_anchor=(1.035, 0.0, 1.0, 1.0),
			bbox_transform=axes[-1].transAxes,
			borderpad=0.0,
		)
		cbar = fig.colorbar(im, cax=cax)
		cbar.set_ticks([])
		cbar.outline.set_linewidth(0.7)
		cbar.outline.set_edgecolor("#4A4A4A")

	fig.suptitle(
		PHASE_FIGURE_TITLES.get(phase, f"Spatial Map of {phase.upper()}"),
		fontsize=20 * font_scale,
		y=0.98,
	)
	roi_tag = _roi_output_tag("contact" if group_analysis_polygons else "circle", roi_radius_scale)
	out_path = output_dir / f"spatial_map__{phase}__{keypoint}__target_{target_side}__{roi_tag}{file_suffix}.png"
	_save_figure(fig, out_path)
	plt.close(fig)
	return out_path

def _save_combined_figure(
	*,
	phase_list: list[str],
	condition_order: list[str],
	group_heatmaps: dict[tuple[str, str], np.ndarray],
	group_support_masks: dict[tuple[str, str], np.ndarray],
	group_object_circles: dict[tuple[str, str, str], tuple[float, float, float, float]],
	group_analysis_circles: dict[tuple[str, str, str], tuple[float, float, float, float]],
	group_object_polygons: dict[tuple[str, str, str], np.ndarray] | None,
	group_analysis_polygons: dict[tuple[str, str, str], np.ndarray] | None,
	output_dir: Path,
	keypoint: str,
	target_side: str,
	roi_radius_scale: float,
	density_mode: str,
	file_suffix: str,
	x_max: float,
	display_width_height_ratio: float,
) -> Path:
	nrows = len(condition_order)
	ncols = len(phase_list)
	fig, axes = plt.subplots(nrows, ncols, figsize=(6.1 * ncols + 0.8, 3.4 * nrows + 0.4), constrained_layout=True)
	axes = np.asarray(axes, dtype=object).reshape(nrows, ncols)
	cmap = plt.get_cmap("turbo").copy()
	phase_images: dict[str, object] = {}
	global_vmax = max((float(heat.max()) for heat in group_heatmaps.values()), default=1.0)
	norm = Normalize(vmin=0.0, vmax=max(global_vmax, 1e-9))

	for col_idx, phase in enumerate(phase_list):
		left_label, right_label = PHASE_LABELS.get(phase, ("L", "R"))
		for row_idx, condition in enumerate(condition_order):
			ax = axes[row_idx, col_idx]
			heat = group_heatmaps.get((phase, condition))
			if heat is None:
				ax.set_visible(False)
				continue
			ax.set_facecolor(cmap(0.0))
			im = ax.imshow(heat.T, extent=(0.0, x_max, 1.0, 0.0), cmap=cmap, norm=norm)
			phase_images[phase] = im
			occupancy_support = group_support_masks.get((phase, condition), heat > 0.0)
			_draw_overlay(
				ax,
				occupancy_support=occupancy_support,
				object_circles={
					"chamber_l": group_object_circles.get((phase, condition, "chamber_l")),
					"chamber_r": group_object_circles.get((phase, condition, "chamber_r")),
				},
				analysis_circles={
					"chamber_l": group_analysis_circles.get((phase, condition, "chamber_l")),
					"chamber_r": group_analysis_circles.get((phase, condition, "chamber_r")),
				},
				object_polygons={
					"chamber_l": (group_object_polygons or {}).get((phase, condition, "chamber_l")),
					"chamber_r": (group_object_polygons or {}).get((phase, condition, "chamber_r")),
				},
				analysis_polygons={
					"chamber_l": (group_analysis_polygons or {}).get((phase, condition, "chamber_l")),
					"chamber_r": (group_analysis_polygons or {}).get((phase, condition, "chamber_r")),
				},
				x_max=x_max,
			)
			ax.set_xlim(0.0, x_max)
			ax.set_ylim(1.0, 0.0)
			# Coordinates remain wall-normalized; only render the known apparatus
			# width:height ratio so circular cups are not stretched on a square map.
			ax.set_aspect(x_max / display_width_height_ratio, adjustable="box")
			ax.set_xticks([])
			ax.set_yticks([])
			ax.text(0.0, -0.06, condition, transform=ax.transAxes, fontsize=12, color="black", ha="left", va="top")
			ax.text(0.06, 1.03, left_label, transform=ax.transAxes, fontsize=12, color="black", ha="left", va="bottom")
			ax.text(0.94, 1.03, right_label, transform=ax.transAxes, fontsize=12, color="black", ha="right", va="bottom")
			if row_idx == 0:
				ax.set_title(PHASE_TITLES.get(phase, f"{phase.upper()} occupancy"), fontsize=14, pad=16)

	for col_idx, phase in enumerate(phase_list):
		im = phase_images.get(phase)
		if im is None:
			continue
		cbar = fig.colorbar(im, ax=axes[:, col_idx].tolist(), fraction=0.035, pad=0.02)
		cbar_label = "Pooled occupancy probability" if density_mode == "pooled" else "Mean occupancy probability"
		cbar.set_label(cbar_label, fontsize=11)

	legend_handles = [
		Line2D([0], [0], color="white", linewidth=1.2, linestyle=(0, (5, 4)), label="Pinned chamber"),
		Line2D([0], [0], color="#00E5FF", linewidth=1.45, linestyle="-", label=f"Analysis ROI ({_roi_output_tag('contact', roi_radius_scale)})" if group_analysis_polygons else f"Analysis ROI (x{roi_radius_scale:.2f})"),
	]
	fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=10)

	roi_tag = _roi_output_tag("contact" if group_analysis_polygons else "circle", roi_radius_scale)
	out_path = output_dir / f"spatial_map_combined__{keypoint}__target_{target_side}__{roi_tag}{file_suffix}.png"
	_save_figure(fig, out_path, pad_inches=0.1)
	plt.close(fig)
	return out_path
