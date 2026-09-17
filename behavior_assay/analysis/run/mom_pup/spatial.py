from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pathlib import Path
import argparse
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import Normalize, LogNorm, PowerNorm
from analysis.core.mom_pup.spatial import _alltracks_mean_pairwise_distance
from analysis.core.mom_pup.spatial import _hist2d_counts
from analysis.core.mom_pup.spatial import _infer_title_from_name
from analysis.core.mom_pup.spatial import _parse_pnd_from_name
from analysis.core.mom_pup.spatial import compute_spatial_stats
from analysis.core.mom_pup.spatial import compute_track_spatial_stats
from analysis.plots.mom_pup.spatial import generate_summary_plots
from analysis.plots.mom_pup.spatial import plot_per_track_heatmaps
from analysis.plots.mom_pup.spatial import plot_pup_mom_heatmaps
from analysis.plots.mom_pup.spatial import save_shared_colorbar

def main(argv=None):
	p = argparse.ArgumentParser(description="Plot side-by-side 2D occupancy heatmaps: Pup (left) and Mom (right)")
	p.add_argument("input", nargs="?", default="output/mom_pup/preprocessed", help="Input CSV file or directory (e.g., output/mom_pup/preprocessed/)")
	p.add_argument("--glob", default="*.csv", help="Glob pattern for input files if directory is given")
	p.add_argument("--bins", type=int, default=100, help="Number of bins per axis for 2D histogram")
	p.add_argument("--xlim", type=float, nargs=2, default=(0.0, 1.0), help="X axis limits (min max)")
	p.add_argument("--ylim", type=float, nargs=2, default=(0.0, 1.0), help="Y axis limits (min max)")
	p.add_argument("--output-dir", default="output/mom_pup", help="Directory to save heatmap figures")
	p.add_argument("--mode", choices=["mom_pup", "per_track"], default="mom_pup", help="mom_pup: legacy 2-panel Pup/Mom; per_track: grid of tracks")
	p.add_argument("--invert-y", action=argparse.BooleanOptionalAction, default=True, help="If set, invert y-axis (image-like coords). Use --no-invert-y to keep y increasing upward.")
	p.add_argument("--norm", choices=["linear","log","gamma"], default="log", help="Color normalization (log brightens sparse bins)")
	p.add_argument("--gamma", type=float, default=0.5, help="Gamma for PowerNorm when --norm gamma")
	p.add_argument("--vmax-quantile", type=float, default=0.995, help="Shared scale: clip vmax to this upper quantile across all files (exclude zeros)")
	p.add_argument("--stats-csv", default="", help="If set, write per-file spatial stats CSV to this path")
	p.add_argument("--wall-band", type=float, default=0.05, help="Wall proximity band for affinity metrics")
	p.add_argument("--center-radius", type=float, default=0.15, help="Center radius for center bias metric")
	p.add_argument("--summary-plots", action="store_true", help="If set, read --stats-csv and create summary bar plots for key metrics")
	args = p.parse_args(argv)

	in_path = Path(args.input)
	files = [in_path] if in_path.is_file() else sorted(in_path.glob(args.glob))
	if not files:
		print("No CSV files found for heatmaps.")
		return 0

	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# Choose white->red colormap and force zeros to white via 'under'
	cmap = mpl.colormaps.get_cmap("Reds")
	try:
		cmap = cmap.copy()
	except Exception:
		pass
	cmap.set_under((1.0, 1.0, 1.0, 1.0))

	# Determine a global color scale across all selected files using quantile clipping
	all_vals: list[np.ndarray] = []
	for f in files:
		df = pd.read_csv(f)
		df["track"] = df["track"].astype(str)
		if args.mode == "mom_pup":
			mom_xy = df[df["track"] == "track_0"][ ["Body_C.x","Body_C.y"] ].dropna()
			pup_xy = df[df["track"] != "track_0"][ ["Body_C.x","Body_C.y"] ].dropna()
			if not mom_xy.empty:
				mom_h = _hist2d_counts(mom_xy["Body_C.x"].to_numpy(), mom_xy["Body_C.y"].to_numpy(), args.bins, (args.xlim[0], args.xlim[1]), (args.ylim[0], args.ylim[1]))
				all_vals.append(mom_h.ravel())
			if not pup_xy.empty:
				pup_h = _hist2d_counts(pup_xy["Body_C.x"].to_numpy(), pup_xy["Body_C.y"].to_numpy(), args.bins, (args.xlim[0], args.xlim[1]), (args.ylim[0], args.ylim[1]))
				all_vals.append(pup_h.ravel())
		else:
			for _, g in df.groupby("track"):
				xy = g[["Body_C.x", "Body_C.y"]].dropna()
				if xy.empty:
					continue
				h = _hist2d_counts(xy["Body_C.x"].to_numpy(), xy["Body_C.y"].to_numpy(), args.bins, (args.xlim[0], args.xlim[1]), (args.ylim[0], args.ylim[1]))
				all_vals.append(h.ravel())

	if all_vals:
		vals = np.concatenate(all_vals)
		vals = vals[vals > 0]
		if vals.size == 0:
			vmin, vmax = 0.0, 1.0
		else:
			vmin, vmax = 1.0, float(np.quantile(vals, args.vmax_quantile))
	else:
		vmin, vmax = 1.0, 1.0

	# Build normalization
	if args.norm == "linear":
		norm = Normalize(vmin=vmin, vmax=vmax)
		label = "count"
	elif args.norm == "log":
		norm = LogNorm(vmin=max(vmin, 1.0), vmax=max(vmax, vmin + 1.0))
		label = "count (log)"
	else:
		norm = PowerNorm(gamma=max(args.gamma, 1e-3), vmin=vmin, vmax=vmax)
		label = f"count (gamma={args.gamma})"

	# Save one shared colorbar image (Reds) for all figures in this run
	cbar_path = out_dir / "spatial__colorbar_Reds.png"
	save_shared_colorbar(cbar_path, norm=norm, cmap=cmap, label=label)
	print(f"Saved {cbar_path}")

	# Stats rows accumulator
	rows: List[Dict[str, object]] = []

	# Render per-file heatmaps without colorbars, using shared scale and Reds colormap
	for f in files:
		df = pd.read_csv(f)
		title = _infer_title_from_name(f.name)
		if args.mode == "mom_pup":
			png_path = out_dir / f"spatial__{f.stem}.png"
			plot_pup_mom_heatmaps(
				df,
				png_path,
				bins=args.bins,
				xlim=(args.xlim[0], args.xlim[1]),
				ylim=(args.ylim[0], args.ylim[1]),
				norm=norm,
				cmap=cmap,
				title=title,
				invert_y=args.invert_y,
			)
		else:
			png_path = out_dir / f"spatial_tracks__{f.stem}.png"
			plot_per_track_heatmaps(
				df,
				png_path,
				bins=args.bins,
				xlim=(args.xlim[0], args.xlim[1]),
				ylim=(args.ylim[0], args.ylim[1]),
				norm=norm,
				cmap=cmap,
				title=title,
				max_cols=2,
				invert_y=args.invert_y,
			)
		print(f"Saved {png_path}")

		if args.stats_csv:
			if args.mode == "mom_pup":
				stats = compute_spatial_stats(
					df,
					bins=args.bins,
					xlim=(args.xlim[0], args.xlim[1]),
					ylim=(args.ylim[0], args.ylim[1]),
					wall_band=args.wall_band,
					center_r=args.center_radius,
				)
				row: Dict[str, object] = {"file": f.name, "pnd": _parse_pnd_from_name(f.name)}
				row.update(stats)
				rows.append(row)
			else:
				track_rows = compute_track_spatial_stats(
					df,
					bins=args.bins,
					xlim=(args.xlim[0], args.xlim[1]),
					ylim=(args.ylim[0], args.ylim[1]),
					wall_band=args.wall_band,
					center_r=args.center_radius,
				)
				for tr in track_rows:
					row2: Dict[str, object] = {"file": f.name, "track": tr["track"]}
					row2.update(tr)
					# Optional file-level context
					row2["alltracks_mean_pairwise_distance"] = _alltracks_mean_pairwise_distance(df)
					rows.append(row2)

	# Write stats CSV if requested
	if args.stats_csv and rows:
		stats_path = Path(args.stats_csv)
		pd.DataFrame(rows).to_csv(stats_path, index=False)
		print(f"Saved stats {stats_path}")
		if args.summary_plots and args.mode == "mom_pup":
			generate_summary_plots(stats_path, out_dir)
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
