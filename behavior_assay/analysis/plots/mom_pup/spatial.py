from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap
from analysis.core.mom_pup.spatial import _apply_axes_style
from analysis.core.mom_pup.spatial import _groupbar
from analysis.core.mom_pup.spatial import _hist2d_counts
from analysis.core.mom_pup.spatial import _imshow_hist
from analysis.core.mom_pup.spatial import _sem

def plot_pup_mom_heatmaps(
	df: pd.DataFrame,
	out_png: Path,
	*,
	bins: int,
	xlim: Tuple[float, float],
	ylim: Tuple[float, float],
	norm,
	cmap: Colormap,
	title: str,
	invert_y: bool = True,
):
	# Expect columns
	required = {"Body_C.x","Body_C.y","track"}
	if not required.issubset(df.columns):
		raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")

	df = df.copy()
	df["track"] = df["track"].astype(str)

	mom_xy = df[df["track"] == "track_0"][ ["Body_C.x","Body_C.y"] ].dropna()
	pup_xy = df[df["track"] != "track_0"][ ["Body_C.x","Body_C.y"] ].dropna()

	mom_h = _hist2d_counts(mom_xy["Body_C.x"].to_numpy(), mom_xy["Body_C.y"].to_numpy(), bins, xlim, ylim)
	pup_h = _hist2d_counts(pup_xy["Body_C.x"].to_numpy(), pup_xy["Body_C.y"].to_numpy(), bins, xlim, ylim)

	plot_histogram_pair(pup_h, mom_h, out_png, xlim=xlim, ylim=ylim, norm=norm, cmap=cmap, title=title, invert_y=invert_y)

def plot_histogram_pair(pup_h, mom_h, out_png, *, xlim=(0, 1), ylim=(0, 1), norm, cmap, title, invert_y=True):
	fig, axes = plt.subplots(
		1,
		2,
		figsize=(9.2, 5.2),
		layout="compressed",
	)
	layout_engine = fig.get_layout_engine()
	if layout_engine is not None:
		layout_engine.set(
			w_pad=0.02,
			h_pad=0.02,
			wspace=0.01,
			hspace=0.01,
		)
	_imshow_hist(axes[0], pup_h, xlim, ylim, norm=norm, cmap=cmap, title="Pup occupancy", invert_y=invert_y)
	_imshow_hist(axes[1], mom_h, xlim, ylim, norm=norm, cmap=cmap, title="Mom occupancy", invert_y=invert_y)
	fig.suptitle(title, fontsize=24)
	_save_figure_bundle(fig, out_png)
	plt.close(fig)

def plot_per_track_heatmaps(
	df: pd.DataFrame,
	out_png: Path,
	*,
	bins: int,
	xlim: Tuple[float, float],
	ylim: Tuple[float, float],
	norm,
	cmap: Colormap,
	title: str,
	max_cols: int = 2,
	invert_y: bool = True,
):
	# Render one heatmap per track as a grid
	required = {"Body_C.x", "Body_C.y", "track"}
	if not required.issubset(df.columns):
		raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")

	work = df.copy()
	work["track"] = work["track"].astype(str)
	tracks = sorted(work["track"].dropna().unique().tolist())
	if not tracks:
		return

	n = len(tracks)
	cols = min(max_cols, n)
	rows = int(np.ceil(n / cols))

	fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), constrained_layout=True)
	axes_arr = np.array(axes).reshape(-1)

	for i, t in enumerate(tracks):
		ax = axes_arr[i]
		xy = work[work["track"] == t][["Body_C.x", "Body_C.y"]].dropna()
		h = _hist2d_counts(xy["Body_C.x"].to_numpy(), xy["Body_C.y"].to_numpy(), bins, xlim, ylim) if not xy.empty else np.zeros((bins, bins), dtype=float)
		_imshow_hist(ax, h, xlim, ylim, norm=norm, cmap=cmap, title=t, invert_y=invert_y)

	# Turn off unused axes
	for j in range(n, len(axes_arr)):
		axes_arr[j].axis("off")

	fig.suptitle(title, fontsize=24)
	fig.savefig(out_png, dpi=150)
	plt.close(fig)

def _save_figure_bundle(fig, out_png: Path):
	"""Save the display image plus publication-quality raster and vector copies."""
	out_png = Path(out_png)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
	fig.savefig(out_png, dpi=300, **save_kwargs)
	fig.savefig(out_png.with_suffix(".svg"), **save_kwargs)

def save_shared_colorbar(out_path: Path, *, norm, cmap: Colormap, label: str = "count"):
	sm = ScalarMappable(norm=norm, cmap=cmap)
	sm.set_array([])
	fig, ax = plt.subplots(figsize=(1.6, 5.0))
	fig.subplots_adjust(left=0.4, right=0.8)
	cbar = fig.colorbar(sm, cax=ax, orientation='vertical')
	cbar.set_label(label, fontsize=14)
	cbar.ax.tick_params(labelsize=12)
	_save_figure_bundle(fig, out_path)
	plt.close(fig)

def generate_summary_plots(stats_csv: Path, out_dir: Path):
	df = pd.read_csv(stats_csv)
	df["pnd"] = pd.to_numeric(df["pnd"], errors="coerce")
	# Coverage
	cov_long = pd.DataFrame({
		"pnd": pd.concat([df["pnd"], df["pnd"]], ignore_index=True),
		"role": ["Pup"] * len(df) + ["Mom"] * len(df),
		"value": pd.concat([df["pup_coverage"], df["mom_coverage"]], ignore_index=True),
	})
	cov_grp = cov_long.groupby(["pnd", "role"])['value']
	cov_stats = cov_grp.mean().reset_index(name="mean")
	cov_stats["sem"] = cov_grp.apply(_sem).values
	# Total distance
	dist_long = pd.DataFrame({
		"pnd": pd.concat([df["pnd"], df["pnd"]], ignore_index=True),
		"role": ["Pup"] * len(df) + ["Mom"] * len(df),
		"value": pd.concat([df["pup_total_distance"], df["mom_total_distance"]], ignore_index=True),
	})
	dist_grp = dist_long.groupby(["pnd", "role"])['value']
	dist_stats = dist_grp.mean().reset_index(name="mean")
	dist_stats["sem"] = dist_grp.apply(_sem).values
	# Pup pairwise distance (no Mom)
	pp_grp = df.groupby("pnd")["pup_mean_pairwise_distance"]
	pp_stats = pp_grp.mean().reset_index(name="mean")
	pp_stats["sem"] = pp_grp.apply(_sem).values
	# Mom–Pup mean distance
	mp_grp = df.groupby("pnd")["mom_pup_mean_distance"]
	mp_stats = mp_grp.mean().reset_index(name="mean")
	mp_stats["sem"] = mp_grp.apply(_sem).values

	pnd_order = sorted(cov_stats["pnd"].dropna().unique().tolist())
	colors = {"Pup": "#d62728", "Mom": "#1f77b4"}

	# Plot coverage (grouped)
	fig, ax = plt.subplots(figsize=(6, 4))
	_groupbar(ax, cov_stats, pnd_order, ["Pup", "Mom"], colors, ylabel="Coverage", title="Mean coverage by PND")
	_apply_axes_style(ax, title="Mean coverage by PND", xlabel="PND", ylabel="Coverage")
	fig.tight_layout()
	fig.savefig(out_dir / "summary_coverage.png", dpi=150)
	plt.close(fig)
	# Plot total distance (grouped)
	fig, ax = plt.subplots(figsize=(6, 4))
	_groupbar(ax, dist_stats, pnd_order, ["Pup", "Mom"], colors, ylabel="Total distance", title="Mean total distance by PND")
	_apply_axes_style(ax, title="Mean total distance by PND", xlabel="PND", ylabel="Total distance")
	fig.tight_layout()
	fig.savefig(out_dir / "summary_total_distance.png", dpi=150)
	plt.close(fig)

	# Separate per-role plots
	for metric_name, stats_df, ylabel in [
		("coverage", cov_stats, "Coverage"),
		("total_distance", dist_stats, "Total distance"),
	]:
		for role in ["Pup", "Mom"]:
			dfr = stats_df[stats_df["role"] == role]
			x = np.arange(len(pnd_order))
			y = [float(dfr[dfr["pnd"] == p]["mean"]) if not dfr[dfr["pnd"] == p].empty else 0.0 for p in pnd_order]
			yerr = [float(dfr[dfr["pnd"] == p]["sem"]) if not dfr[dfr["pnd"] == p].empty else 0.0 for p in pnd_order]
			fig, ax = plt.subplots(figsize=(6, 4))
			ax.bar(x, y, yerr=yerr, capsize=3, color=colors[role])
			ax.set_xticks(x)
			ax.set_xticklabels([str(p) for p in pnd_order])
			_apply_axes_style(ax, title=f"{role} {ylabel} by PND", xlabel="PND", ylabel=ylabel)
			fig.tight_layout()
			fig.savefig(out_dir / f"summary_{metric_name}_{role.lower()}.png", dpi=150)
			plt.close(fig)

	# Plot pup pairwise distance (single)
	fig, ax = plt.subplots(figsize=(6, 4))
	x = np.arange(len(pnd_order))
	y = [float(pp_stats[pp_stats["pnd"] == p]["mean"].iloc[0]) if not pp_stats[pp_stats["pnd"] == p].empty else 0.0 for p in pnd_order]
	yerr = [float(pp_stats[pp_stats["pnd"] == p]["sem"].iloc[0]) if not pp_stats[pp_stats["pnd"] == p].empty else 0.0 for p in pnd_order]
	ax.bar(x, y, yerr=yerr, capsize=3, color="#d62728")
	ax.set_xticks(x)
	ax.set_xticklabels([str(p) for p in pnd_order])
	_apply_axes_style(ax, title="Pup pairwise distance by PND", xlabel="PND", ylabel="Pup mean pairwise distance")
	fig.tight_layout()
	fig.savefig(out_dir / "summary_pup_pairwise_distance.png", dpi=150)
	plt.close(fig)

	# Plot Mom–Pup mean distance (single)
	fig, ax = plt.subplots(figsize=(6, 4))
	y = [float(mp_stats[mp_stats["pnd"] == p]["mean"].iloc[0]) if not mp_stats[mp_stats["pnd"] == p].empty else 0.0 for p in pnd_order]
	yerr = [float(mp_stats[mp_stats["pnd"] == p]["sem"].iloc[0]) if not mp_stats[mp_stats["pnd"] == p].empty else 0.0 for p in pnd_order]
	ax.bar(x, y, yerr=yerr, capsize=3, color="#555555")
	ax.set_xticks(x)
	ax.set_xticklabels([str(p) for p in pnd_order])
	_apply_axes_style(ax, title="Mom–Pup mean distance by PND", xlabel="PND", ylabel="Mom–Pup mean distance")
	fig.tight_layout()
	fig.savefig(out_dir / "summary_mom_pup_mean_distance.png", dpi=150)
	plt.close(fig)
