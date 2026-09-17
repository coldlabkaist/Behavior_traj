from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pathlib import Path
import argparse
import pandas as pd
from analysis.core.mom_pup.trajectory import infer_title_from_name
from analysis.core.mom_pup.trajectory import load_csvs
from analysis.plots.mom_pup.trajectory import plot_trajectories

def main(argv=None):
	p = argparse.ArgumentParser(description="Plot Body_C XY trajectories per track from interpolated CSVs")
	p.add_argument("input", nargs="?", default="output/mom_pup/preprocessed", help="Input CSV file or directory (e.g., output/mom_pup/preprocessed/)")
	p.add_argument("--glob", default="*.csv", help="Glob pattern for input files if directory is given")
	p.add_argument("--output-dir", default="output/mom_pup", help="Directory to save trajectory plots")
	p.add_argument("--label-mode", choices=["mom_pup", "generic"], default="mom_pup", help="Legend labels: mom/pup (legacy) or generic Track N")
	p.add_argument("--invert-y", action=argparse.BooleanOptionalAction, default=True, help="If set, invert y-axis (image-like coords). Use --no-invert-y to keep y increasing upward.")
	p.add_argument("--score-threshold", type=float, default=None, help="If set and Body_C.score exists, mask points below this threshold (breaks lines).")
	p.add_argument("--max-jump", type=float, default=None, help="If set, break lines when step-to-step distance exceeds this value (in same units as x/y).")
	p.add_argument("--xlim", type=float, nargs=2, default=None, help="If set, mask points outside [xmin xmax] (breaks lines).")
	p.add_argument("--ylim", type=float, nargs=2, default=None, help="If set, mask points outside [ymin ymax] (breaks lines).")
	args = p.parse_args(argv)

	in_path = Path(args.input)
	files = load_csvs(in_path) if in_path.is_file() else sorted(in_path.glob(args.glob))
	if not files:
		print("No CSV files found for plotting.")
		return 0

	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	for f in files:
		df = pd.read_csv(f)
		title = infer_title_from_name(f.name)
		png_path = out_dir / f"traj__{f.stem}.png"
		plot_trajectories(
			df,
			title,
			png_path,
			label_mode=args.label_mode,
			invert_y=args.invert_y,
			score_threshold=args.score_threshold,
			max_jump=args.max_jump,
			xlim=(tuple(args.xlim) if args.xlim is not None else None),
			ylim=(tuple(args.ylim) if args.ylim is not None else None),
		)
		print(f"Saved {png_path}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
