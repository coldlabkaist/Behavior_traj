from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from analysis.core.mom_pup.talk_occupancy import _load_selected_sessions
from analysis.core.mom_pup.talk_occupancy import _session_histograms
from analysis.core.mom_pup.talk_occupancy import _shared_log_norm

matplotlib.use("Agg")

from analysis.core.mom_pup.talk_occupancy import CONTROL

from analysis.core.mom_pup.talk_occupancy import VPA

from analysis.core.mom_pup.talk_occupancy import PNDS

def _draw_panel(
	ax: plt.Axes,
	histogram: np.ndarray,
	*,
	norm: LogNorm,
	cmap,
	border_color: str,
	label: str,
	label_size: float,
) -> None:
	ax.imshow(
		histogram.T,
		extent=(0.0, 1.0, 0.0, 1.0),
		origin="lower",
		interpolation="nearest",
		cmap=cmap,
		norm=norm,
		aspect="equal",
	)
	ax.invert_yaxis()
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_facecolor("white")
	for spine in ax.spines.values():
		spine.set_color(border_color)
		spine.set_linewidth(4.2)
	ax.text(
		0.5,
		0.035,
		label,
		transform=ax.transAxes,
		ha="center",
		va="bottom",
		fontsize=label_size,
		color="black",
		bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8},
	)

def plot_talk_occupancy(
	manifest_path: Path,
	output_dir: Path,
	*,
	bins: int = 100,
	vmax_quantile: float = 0.995,
) -> list[Path]:
	sessions = _load_selected_sessions(manifest_path)
	histograms = {
		key: _session_histograms(df, bins=bins)
		for key, df in sessions.items()
	}
	norm = _shared_log_norm(histograms, vmax_quantile=vmax_quantile)
	cmap = mpl.colormaps.get_cmap("Reds").copy()
	cmap.set_under("white")

	fig = plt.figure(figsize=(13.6, 5.9), facecolor="white")
	grid = fig.add_gridspec(
		2,
		8,
		width_ratios=(1.0, 1.0, 0.23, 1.0, 1.0, 0.23, 1.0, 1.0),
		left=0.062,
		right=0.995,
		bottom=0.025,
		top=0.88,
		wspace=0.055,
		hspace=0.06,
	)

	condition_rows = (("Control", CONTROL, 0), ("VPA", VPA, 1))
	pnd_starts = {10: 0, 15: 3, 20: 6}
	group_axes: dict[int, tuple[plt.Axes, plt.Axes]] = {}
	row_axes: dict[str, list[plt.Axes]] = {"Control": [], "VPA": []}
	label_size = 16

	for condition, border_color, row in condition_rows:
		for pnd in PNDS:
			start = pnd_starts[pnd]
			pup_ax = fig.add_subplot(grid[row, start])
			mom_ax = fig.add_subplot(grid[row, start + 1])
			row_axes[condition].extend((pup_ax, mom_ax))
			pup_h, mom_h = histograms[(condition, pnd)]
			_draw_panel(
				pup_ax,
				pup_h,
				norm=norm,
				cmap=cmap,
				border_color=border_color,
				label="Pup occ",
				label_size=label_size,
			)
			_draw_panel(
				mom_ax,
				mom_h,
				norm=norm,
				cmap=cmap,
				border_color=border_color,
				label="Mom occ",
				label_size=label_size,
			)
			if row == 0:
				group_axes[pnd] = (pup_ax, mom_ax)

	fig.canvas.draw()
	for condition, color in (("Control", CONTROL), ("VPA", VPA)):
		positions = [ax.get_position() for ax in row_axes[condition]]
		y_center = (
			min(position.y0 for position in positions)
			+ max(position.y1 for position in positions)
		) / 2.0
		fig.text(
			0.026,
			y_center,
			condition,
			ha="center",
			va="center",
			rotation=90,
			fontsize=24,
			fontweight="bold",
			color=color,
		)
	for pnd, (left_ax, right_ax) in group_axes.items():
		left = left_ax.get_position().x0
		right = right_ax.get_position().x1
		fig.text(
			(left + right) / 2.0,
			0.932,
			f"PND {pnd}",
			ha="center",
			va="center",
			fontsize=25,
			color="black",
			fontweight="bold",
		)

	output_dir.mkdir(parents=True, exist_ok=True)
	base = output_dir / "mom_pup_occupancy_talk"
	paths = [
		base.with_suffix(".png"),
		base.with_name(f"{base.name}_1200dpi.png"),
		base.with_suffix(".svg"),
	]
	fig.savefig(paths[0], dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.03)
	fig.savefig(paths[1], dpi=1200, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.03)
	fig.savefig(paths[2], facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.03)
	plt.close(fig)
	return paths
