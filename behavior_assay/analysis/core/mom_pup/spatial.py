from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from typing import Tuple, Dict, List
import re
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap

def _hist2d_counts(x: np.ndarray, y: np.ndarray, bins: int, xlim: Tuple[float,float], ylim: Tuple[float,float]) -> np.ndarray:
	# Return raw 2D histogram counts
	h, _, _ = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
	return h

def _hist2d_probs(x: np.ndarray, y: np.ndarray, bins: int, xlim: Tuple[float,float], ylim: Tuple[float,float]) -> np.ndarray:
	h = _hist2d_counts(x, y, bins, xlim, ylim)
	t = float(h.sum())
	return (h / t) if t > 0 else h

def _apply_ticks(ax, xlim: Tuple[float,float], ylim: Tuple[float,float]):
	# Set tick spacing to 0.1 on both axes (like trajectory.py)
	x_ticks = np.round(np.arange(xlim[0], xlim[1] + 1e-9, 0.1), 2)
	y_ticks = np.round(np.arange(ylim[0], ylim[1] + 1e-9, 0.1), 2)
	ax.set_xticks(x_ticks)
	ax.set_yticks(y_ticks)

def _imshow_hist(
	ax,
	h: np.ndarray,
	xlim: Tuple[float, float],
	ylim: Tuple[float, float],
	*,
	norm,
	cmap: Colormap,
	title: str,
	invert_y: bool = True,
):
	# Render provided histogram with fixed color scale; no colorbar here
	xedges = np.linspace(xlim[0], xlim[1], h.shape[0] + 1)
	yedges = np.linspace(ylim[0], ylim[1], h.shape[1] + 1)
	im = ax.imshow(
		h.T,
		extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
		origin="lower",
		interpolation="nearest",
		cmap=cmap,
		aspect="equal",
		norm=norm,
	)
	ax.set_title(title, fontsize=20)
	# remove axis labels and ticks
	ax.set_xlabel("")
	ax.set_ylabel("")
	ax.set_xticks([])
	ax.set_yticks([])
	if invert_y:
		ax.invert_yaxis()
	return im

def _track_total_distance(g: pd.DataFrame) -> float:
	# g must have Body_C.x/y and frame_idx for a single track
	g = g.sort_values("frame_idx")
	x = g["Body_C.x"].to_numpy(dtype=float)
	y = g["Body_C.y"].to_numpy(dtype=float)
	mask = (~np.isnan(x)) & (~np.isnan(y))
	x = x[mask]
	y = y[mask]
	if x.size < 2:
		return 0.0
	dx = np.diff(x)
	dy = np.diff(y)
	return float(np.sqrt(dx * dx + dy * dy).sum())

def _role_total_distance(df: pd.DataFrame, role: str) -> float:
	# role: "Pup" (tracks != track_0) or "Mom" (track_0)
	role_df = df[df["track"].astype(str) != "track_0"] if role == "Pup" else df[df["track"] .astype(str) == "track_0"]
	if role_df.empty:
		return 0.0
	# compute per-track total distance, then average for fair comparison
	dists: list[float] = []
	for _, g in role_df.groupby("track"):
		dists.append(_track_total_distance(g))
	return float(np.mean(dists)) if len(dists) > 0 else 0.0

def _coverage_fraction(x: np.ndarray, y: np.ndarray, *, bins: int, xlim: Tuple[float,float], ylim: Tuple[float,float]) -> float:
	# Proportion of grid cells visited at least once
	h, _, _ = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
	cells = h.size
	if cells == 0:
		return 0.0
	return float((h > 0).sum()) / float(cells)

def _pup_mean_pairwise_distance(df: pd.DataFrame) -> float:
	# Average pairwise distance among pups across frames (only frames where >=2 pups present)
	pup_df = df[df["track"].astype(str) != "track_0"][ ["frame_idx","track","Body_C.x","Body_C.y"] ]
	pup_df = pup_df.dropna(subset=["Body_C.x","Body_C.y"])  # require both coords
	if pup_df.empty:
		return 0.0
	sum_d = 0.0
	n_pairs = 0
	for frame, g in pup_df.groupby("frame_idx"):
		coords = g[["Body_C.x","Body_C.y"]].to_numpy(dtype=float)
		k = coords.shape[0]
		if k < 2:
			continue
		diff = coords[:, None, :] - coords[None, :, :]
		dmat = np.sqrt((diff * diff).sum(axis=2))
		iu = np.triu_indices(k, 1)
		ds = dmat[iu]
		sum_d += float(ds.sum())
		n_pairs += int(ds.size)
	return float(sum_d / n_pairs) if n_pairs > 0 else 0.0

def _alltracks_mean_pairwise_distance(df: pd.DataFrame) -> float:
	# Average pairwise distance among all tracks across frames (only frames where >=2 tracks present)
	work = df[["frame_idx", "track", "Body_C.x", "Body_C.y"]].copy()
	work = work.dropna(subset=["Body_C.x", "Body_C.y"])
	if work.empty:
		return 0.0
	sum_d = 0.0
	n_pairs = 0
	for _, g in work.groupby("frame_idx"):
		coords = g[["Body_C.x", "Body_C.y"]].to_numpy(dtype=float)
		k = coords.shape[0]
		if k < 2:
			continue
		diff = coords[:, None, :] - coords[None, :, :]
		dmat = np.sqrt((diff * diff).sum(axis=2))
		iu = np.triu_indices(k, 1)
		ds = dmat[iu]
		sum_d += float(ds.sum())
		n_pairs += int(ds.size)
	return float(sum_d / n_pairs) if n_pairs > 0 else 0.0

def _mom_pup_mean_distance(df: pd.DataFrame) -> float:
	# Mean distance between Mom (track_0) and each Pup per frame, averaged over all frames/pups available
	df = df.copy()
	df["track"] = df["track"].astype(str)
	mom = df[df["track"] == "track_0"][ ["frame_idx","Body_C.x","Body_C.y"] ].rename(columns={"Body_C.x":"mx","Body_C.y":"my"})
	pups = df[df["track"] != "track_0"][ ["frame_idx","Body_C.x","Body_C.y"] ].rename(columns={"Body_C.x":"px","Body_C.y":"py"})
	if mom.empty or pups.empty:
		return 0.0
	# Join by frame to get mom location for each pup observation
	joined = pups.merge(mom, on="frame_idx", how="inner")
	joined = joined.dropna(subset=["px","py","mx","my"])  # require both
	if joined.empty:
		return 0.0
	d = np.sqrt((joined["px"].to_numpy() - joined["mx"].to_numpy())**2 + (joined["py"].to_numpy() - joined["my"].to_numpy())**2)
	return float(np.mean(d)) if d.size > 0 else 0.0

def _infer_title_from_name(name: str) -> str:
	m1 = re.search(r"PND\s*(\d+)", name, re.IGNORECASE)
	if not m1:
		m1 = re.search(r"PND(\d+)", name, re.IGNORECASE)
	pnd = m1.group(1) if m1 else None
	m2 = re.search(r"PUP[_-]?(\d+)", name, re.IGNORECASE)
	pup = m2.group(1) if m2 else None
	if pnd and pup:
		return f"PND {pnd} cage {int(pup):02d}"
	if pnd:
		return f"PND {pnd}"
	return name

def _entropy_norm(p: np.ndarray) -> float:
	p = p[p > 0]
	if p.size == 0:
		return 0.0
	H = float(-(p * np.log(p)).sum())
	Hmax = float(np.log(p.size))
	return float(H / Hmax) if Hmax > 0 else 0.0

def _gini(p: np.ndarray) -> float:
	p = p[p > 0]
	if p.size == 0:
		return 0.0
	p_sorted = np.sort(p)
	n = p_sorted.size
	cum = np.cumsum(p_sorted)
	# Gini over probabilities (sum=1): 1 - 2 * AUC(Lorenz)
	lorenz_auc = float(cum.sum()) / (n * float(cum[-1])) - (n + 1) / (2 * n)
	return max(0.0, min(1.0, 1.0 - 2.0 * lorenz_auc))

def _top_mass_area(p: np.ndarray, mass: float) -> float:
	p = p[p > 0]
	if p.size == 0:
		return 0.0
	idx = np.argsort(p)[::-1]
	p_sorted = p[idx]
	cum = np.cumsum(p_sorted)
	k = int(np.searchsorted(cum, mass, side='left')) + 1
	return float(k) / float(p.size)

def _cov_spread(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
	if x.size < 2:
		return 0.0, 0.0
	C = np.cov(np.vstack([x, y]))
	vals = np.linalg.eigvalsh(C)
	vals = np.maximum(vals, 0.0)
	trace = float(vals.sum())
	rog = float(np.sqrt(trace))
	anisotropy = float((vals[-1] / (vals[0] + 1e-12)) if vals[0] > 0 else np.inf)
	return rog, anisotropy

def _affinity_xy(x: np.ndarray, y: np.ndarray, *, wall_band: float, center_r: float) -> Tuple[float, float, float]:
	if x.size == 0:
		return 0.0, 0.0, 0.0
	# Distances to borders and center
	dx = np.minimum(x, 1.0 - x)
	dy = np.minimum(y, 1.0 - y)
	wall = ((dx < wall_band) | (dy < wall_band)).mean()
	corner = ((x < wall_band) & (y < wall_band)) | ((x > 1 - wall_band) & (y < wall_band)) | ((x < wall_band) & (y > 1 - wall_band)) | ((x > 1 - wall_band) & (y > 1 - wall_band))
	corner = float(corner.mean())
	cx = x - 0.5
	cy = y - 0.5
	center = float((np.hypot(cx, cy) < center_r).mean())
	return float(wall), float(corner), center

def compute_spatial_stats(df: pd.DataFrame, *, bins: int, xlim: Tuple[float,float], ylim: Tuple[float,float], wall_band: float = 0.05, center_r: float = 0.15) -> Dict[str, float]:
	# Pup: all except track_0; Mom: track_0
	df = df.copy()
	df["track"] = df["track"].astype(str)
	roles = {
		"Pup": df[df["track"] != "track_0"][ ["Body_C.x","Body_C.y","track","frame_idx"] ].dropna(subset=["Body_C.x","Body_C.y"]),
		"Mom": df[df["track"] == "track_0"][ ["Body_C.x","Body_C.y","track","frame_idx"] ].dropna(subset=["Body_C.x","Body_C.y"]),
	}
	out: Dict[str, float] = {}
	for role, xy in roles.items():
		x = xy["Body_C.x"].to_numpy()
		y = xy["Body_C.y"].to_numpy()
		p = _hist2d_probs(x, y, bins, xlim, ylim).ravel()
		# coverage_any below will be overridden for Pup to be per-track mean
		coverage_any = float((p > 0).mean())
		if role == "Pup":
			covs: List[float] = []
			for _, g in xy.groupby("track"):
				covs.append(_coverage_fraction(g["Body_C.x"].to_numpy(), g["Body_C.y"].to_numpy(), bins=bins, xlim=xlim, ylim=ylim))
			coverage_any = float(np.mean(covs)) if len(covs) > 0 else 0.0
		entropy_n = _entropy_norm(p)
		gini = _gini(p)
		area50 = _top_mass_area(p, 0.50)
		area95 = _top_mass_area(p, 0.95)
		rog, aniso = _cov_spread(x, y)
		wall, corner, center = _affinity_xy(x, y, wall_band=wall_band, center_r=center_r)
		total_dist = _role_total_distance(xy, role)
		prefix = f"{role.lower()}"
		out.update({
			f"{prefix}_coverage_any": coverage_any,
			f"{prefix}_coverage": coverage_any,
			f"{prefix}_entropy_norm": entropy_n,
			f"{prefix}_gini": gini,
			f"{prefix}_top50_area": area50,
			f"{prefix}_top95_area": area95,
			f"{prefix}_radius_of_gyration": rog,
			f"{prefix}_anisotropy": aniso,
			f"{prefix}_wall_affinity": wall,
			f"{prefix}_corner_affinity": corner,
			f"{prefix}_center_bias": center,
			f"{prefix}_total_distance": total_dist,
		})
	# Pup pairwise mean distance
	out["pup_mean_pairwise_distance"] = _pup_mean_pairwise_distance(df)
	# Mom-Pup mean distance
	out["mom_pup_mean_distance"] = _mom_pup_mean_distance(df)
	return out

def compute_track_spatial_stats(
	df: pd.DataFrame,
	*,
	bins: int,
	xlim: Tuple[float, float],
	ylim: Tuple[float, float],
	wall_band: float = 0.05,
	center_r: float = 0.15,
) -> list[Dict[str, object]]:
	# One stats row per track
	df = df.copy()
	df["track"] = df["track"].astype(str)
	rows: list[Dict[str, object]] = []
	for t, g in df.groupby("track"):
		xy = g[["Body_C.x", "Body_C.y", "frame_idx"]].dropna(subset=["Body_C.x", "Body_C.y"])
		x = xy["Body_C.x"].to_numpy()
		y = xy["Body_C.y"].to_numpy()
		p = _hist2d_probs(x, y, bins, xlim, ylim).ravel()
		coverage = float((p > 0).mean()) if p.size > 0 else 0.0
		entropy_n = _entropy_norm(p)
		gini = _gini(p)
		area50 = _top_mass_area(p, 0.50)
		area95 = _top_mass_area(p, 0.95)
		rog, aniso = _cov_spread(x, y)
		wall, corner, center = _affinity_xy(x, y, wall_band=wall_band, center_r=center_r)
		total_dist = _track_total_distance(xy.rename(columns={"Body_C.x": "Body_C.x", "Body_C.y": "Body_C.y"}))
		rows.append({
			"track": t,
			"coverage": coverage,
			"entropy_norm": entropy_n,
			"gini": gini,
			"top50_area": area50,
			"top95_area": area95,
			"radius_of_gyration": rog,
			"anisotropy": aniso,
			"wall_affinity": wall,
			"corner_affinity": corner,
			"center_bias": center,
			"total_distance": total_dist,
		})
	return rows

def _parse_pnd_from_name(name: str) -> str | None:
	m = re.search(r"PND(\d+)", name, re.IGNORECASE)
	return m.group(1) if m else None

def _sem(vals: pd.Series) -> float:
	vals = pd.to_numeric(vals, errors='coerce').dropna()
	n = len(vals)
	if n <= 1:
		return 0.0
	return float(vals.std(ddof=1) / np.sqrt(n))

def _apply_axes_style(ax, *, title: str, xlabel: str, ylabel: str, title_size: int = 24, label_size: int = 18, tick_size: int = 16, spine_lw: float = 2.0):
	# Remove top/right spines, keep bottom/left thicker
	for side in ["top", "right"]:
		ax.spines[side].set_visible(False)
	for side in ["bottom", "left"]:
		ax.spines[side].set_visible(True)
		ax.spines[side].set_linewidth(spine_lw)
	# Ticks and labels
	ax.tick_params(axis='both', labelsize=tick_size, width=spine_lw)
	ax.set_xlabel(xlabel, fontsize=label_size)
	ax.set_ylabel(ylabel, fontsize=label_size)
	ax.set_title(title, fontsize=title_size)

def _groupbar(ax, df: pd.DataFrame, x_order: List, hue_order: List, colors: Dict[str, str], ylabel: str, title: str):
	width = 0.35
	x = np.arange(len(x_order))
	for i, hue in enumerate(hue_order):
		dfi = df[df["role"] == hue]
		y = [float(dfi[dfi["pnd"] == p]["mean"].iloc[0]) if not dfi[dfi["pnd"] == p].empty else 0.0 for p in x_order]
		yerr = [float(dfi[dfi["pnd"] == p]["sem"].iloc[0]) if not dfi[dfi["pnd"] == p].empty else 0.0 for p in x_order]
		ax.bar(x + (i - (len(hue_order)-1)/2)*width, y, width=width, label=hue, color=colors.get(hue, None), yerr=yerr, capsize=3)
	ax.set_xticks(x)
	ax.set_xticklabels([str(p) for p in x_order])
	# Style will be applied by caller via _apply_axes_style
	ax.legend(fontsize=12)
