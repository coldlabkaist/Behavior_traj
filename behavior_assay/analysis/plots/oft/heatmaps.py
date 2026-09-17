from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

matplotlib.use("Agg")

def _session_probability(path: Path, *, fps: float, seconds: float, bins: int) -> np.ndarray:
	df = pd.read_csv(path, usecols=["frame_idx", "track", "Body_C.x", "Body_C.y"])
	df = df.sort_values("frame_idx")
	start = int(pd.to_numeric(df["frame_idx"], errors="coerce").min())
	df = df[pd.to_numeric(df["frame_idx"], errors="coerce") < start + int(round(fps * seconds))]
	x = pd.to_numeric(df["Body_C.x"], errors="coerce").to_numpy(dtype=float)
	y = pd.to_numeric(df["Body_C.y"], errors="coerce").to_numpy(dtype=float)
	valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
	hist, _, _ = np.histogram2d(x[valid], y[valid], bins=bins, range=[[0, 1], [0, 1]])
	return hist / hist.sum() if hist.sum() else hist

def _draw(ax: plt.Axes, density: np.ndarray, *, title: str, vmax: float):
	image = ax.imshow(
		density.T,
		origin="lower",
		extent=[0, 1, 0, 1],
		cmap="YlOrRd",
		vmin=0,
		vmax=vmax,
		interpolation="bilinear",
		aspect="equal",
	)
	ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#222222", linewidth=1.5))
	ax.add_patch(plt.Rectangle((0.25, 0.25), 0.5, 0.5, fill=False, edgecolor="#222222", linewidth=1.2, linestyle="--"))
	ax.set_title(title, fontsize=22, pad=12)
	ax.set_xticks([])
	ax.set_yticks([])
	return image

def plot_group_densities(group_densities: dict[str, np.ndarray], output_dir: Path) -> int:
	output_dir.mkdir(parents=True, exist_ok=True)
	if not group_densities:
		print("No sessions found.")
		return 1
	vmax = max(float(np.nanmax(value)) for value in group_densities.values())
	fig, axes = plt.subplots(
		1,
		len(group_densities),
		figsize=(9.2, 5.4),
		squeeze=False,
		layout="compressed",
	)
	layout_engine = fig.get_layout_engine()
	if layout_engine is not None:
		layout_engine.set(
			w_pad=0.03,
			h_pad=0.04,
			wspace=0.02,
			hspace=0.02,
		)
	for ax, condition in zip(axes[0], ("Control", "VPA")):
		if condition not in group_densities:
			ax.axis("off")
			continue
		_draw(ax, group_densities[condition], title=condition, vmax=vmax)
	fig.suptitle("Open Field occupancy", fontsize=27)
	figure_path = output_dir / "group_comparison.png"
	fig.savefig(figure_path, dpi=300, bbox_inches="tight")
	fig.savefig(output_dir / "group_comparison.svg", bbox_inches="tight")
	plt.close(fig)

	colorbar_fig, colorbar_ax = plt.subplots(figsize=(5.0, 1.25))
	colorbar_fig.subplots_adjust(left=0.08, right=0.92, bottom=0.50, top=0.68)
	mappable = ScalarMappable(
		norm=Normalize(vmin=0.0, vmax=vmax),
		cmap="YlOrRd",
	)
	mappable.set_array([])
	colorbar = colorbar_fig.colorbar(
		mappable,
		cax=colorbar_ax,
		orientation="horizontal",
	)
	colorbar.set_ticks([])
	colorbar.set_label("Mean occupancy probability", fontsize=18, labelpad=8)
	colorbar_path = output_dir / "shared_colorbar.png"
	colorbar_fig.savefig(colorbar_path, dpi=300, bbox_inches="tight")
	colorbar_fig.savefig(output_dir / "shared_colorbar.svg", bbox_inches="tight")
	plt.close(colorbar_fig)
	print(f"Wrote {figure_path}")
	print(f"Wrote {colorbar_path}")
	return 0
