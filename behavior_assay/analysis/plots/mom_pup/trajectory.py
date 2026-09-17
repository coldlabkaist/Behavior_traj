from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analysis.core.mom_pup.trajectory import _build_track_labels

from analysis.core.mom_pup.trajectory import TRACK_COLORS

from analysis.core.mom_pup.trajectory import TRACK_COLORS_FIXED

def _build_track_colors(tracks: list[str], label_mode: str) -> dict[str, tuple[float, float, float, float] | tuple[float, float, float]]:
	if label_mode == "mom_pup":
		return {t: TRACK_COLORS.get(t, (0.5, 0.5, 0.5)) for t in tracks}
	# Generic: fixed mapping for track_0..3, fallback to tab10 in track order
	cmap = mpl.colormaps.get_cmap("tab10")
	out: dict[str, tuple] = {}
	for i, t in enumerate(tracks):
		if t in TRACK_COLORS_FIXED:
			out[t] = mpl.colors.to_rgba(TRACK_COLORS_FIXED[t])
		else:
			out[t] = cmap(i % 10)
	return out

def plot_trajectories(
	df: pd.DataFrame,
	title: str,
	out_png: Path,
	*,
	label_mode: str = "mom_pup",
	invert_y: bool = True,
	score_threshold: float | None = None,
	max_jump: float | None = None,
	xlim: tuple[float, float] | None = None,
	ylim: tuple[float, float] | None = None,
):
	# Expect Body_C.x, Body_C.y, frame_idx, track
	if not {"Body_C.x","Body_C.y","frame_idx","track"}.issubset(df.columns):
		raise ValueError("Required columns missing: Body_C.x, Body_C.y, frame_idx, track")
	tracks = sorted(df["track"].astype(str).unique().tolist())
	labels = _build_track_labels(tracks, label_mode)
	colors = _build_track_colors(tracks, label_mode)
	plt.figure(figsize=(10, 8))
	for t in tracks:
		g = df[df["track"].astype(str) == t].sort_values("frame_idx")
		color = colors.get(t, (0.5, 0.5, 0.5))
		lw = 2.0 if (label_mode == "mom_pup" and t == "track_0") else 1.0
		x = pd.to_numeric(g["Body_C.x"], errors="coerce").astype(float).to_numpy()
		y = pd.to_numeric(g["Body_C.y"], errors="coerce").astype(float).to_numpy()

		# Optional: mask low-confidence points (prevents connecting through bad detections)
		if score_threshold is not None and "Body_C.score" in g.columns:
			s = pd.to_numeric(g["Body_C.score"], errors="coerce").astype(float).to_numpy()
			bad = ~(s >= float(score_threshold))
			x[bad] = float("nan")
			y[bad] = float("nan")

		# Optional: mask points outside expected plotting bounds
		if xlim is not None:
			bad = (x < float(xlim[0])) | (x > float(xlim[1]))
			x[bad] = float("nan")
			y[bad] = float("nan")
		if ylim is not None:
			bad = (y < float(ylim[0])) | (y > float(ylim[1]))
			x[bad] = float("nan")
			y[bad] = float("nan")

		# Optional: break line on large jumps (prevents long straight spikes)
		if max_jump is not None:
			dx = x[1:] - x[:-1]
			dy = y[1:] - y[:-1]
			jump = (dx * dx + dy * dy) ** 0.5
			cut = jump > float(max_jump)
			# break at i+1 (segment boundary between i and i+1)
			x[1:][cut] = float("nan")
			y[1:][cut] = float("nan")

		plt.plot(x, y, linewidth=lw, color=color)
	ax = plt.gca()
	if invert_y:
		ax.invert_yaxis()  # if coords are image-like
	plt.title(title, fontsize=20)
	# remove axis labels and ticks
	ax.set_xlabel("")
	ax.set_ylabel("")
	ax.set_xticks([])
	ax.set_yticks([])
	# Legend intentionally omitted on main figure
	plt.tight_layout()
	plt.savefig(out_png, dpi=150)
	plt.close()

	# Save a separate (vertical) legend image for the tracks present
	handles = [
		Line2D(
			[0],
			[0],
			color=colors.get(t, (0.5, 0.5, 0.5)),
			lw=(2.0 if (label_mode == "mom_pup" and t == "track_0") else 3),
			label=labels.get(t, t),
		)
		for t in tracks
	]
	if handles:
		# Height scales with number of handles; single-column vertical legend
		height = max(2.0, 0.6 * len(handles))
		fig, ax2 = plt.subplots(figsize=(4, height))
		ax2.axis('off')
		ax2.legend(handles=handles, loc='center left', ncol=1, frameon=False, fontsize=14)
		legend_png = out_png.with_name(f"legend__{out_png.stem}.png")
		fig.tight_layout()
		fig.savefig(legend_png, dpi=150)
		plt.close(fig)
