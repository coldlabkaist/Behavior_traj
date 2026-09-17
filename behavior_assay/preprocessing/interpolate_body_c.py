from __future__ import annotations

from pathlib import Path
import argparse
import sys

# ensure project root on sys.path when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

import pandas as pd

from data_loader.io.schema import Schema
from data_loader.io.csv_loader import read_pose_csv
from data_loader.processing.deduplicate import deduplicate_frame_track
from data_loader.processing.interpolate import pad_frame_track_grid, interpolate_body_c


def _infer_preprocessed_output_dir(in_path: Path) -> Path:
	path_str = str(in_path).replace("\\", "/").lower()
	name = in_path.name.lower()
	parent_name = in_path.parent.name.lower() if in_path.parent else ""

	if "/data/mom_pup" in path_str or name in {"mom_pup", "fig5efh"} or parent_name in {"mom_pup", "fig5efh"}:
		return ROOT / "output" / "mom_pup" / "preprocessed"
	if "/data/objr" in path_str or name == "objr" or parent_name == "objr":
		return ROOT / "output" / "objr" / "preprocessed"
	return ROOT / "output" / "preprocessed"


def main(argv=None):
	p = argparse.ArgumentParser(description="Pad and interpolate Body_C coordinates")
	p.add_argument("input", help="Input CSV file or directory")
	p.add_argument("--output-dir", default="", help="Directory to write outputs (default: infer project-specific output/.../preprocessed)")
	p.add_argument("--dedup", action="store_true", help="Deduplicate before interpolation")
	p.add_argument("--dedup-policy", default="highest_instance_score", choices=["highest_instance_score","highest_kp_score_sum","first"], help="Deduplication policy")
	p.add_argument("--pad", action="store_true", help="Pad to dense (frame,track) grid before interpolation")
	p.add_argument("--method", default="linear", choices=["linear","kalman"], help="Interpolation method")
	p.add_argument("--max-gap", type=int, default=5, help="Max gap length to interpolate (linear)")
	p.add_argument("--smooth-window", type=int, default=0, help="Optional moving-average window after fill (linear)")
	p.add_argument("--score-threshold", type=float, default=0.0, help="Kalman: ignore measurements with score below threshold")
	p.add_argument("--kalman-q-pos", type=float, default=1e-3, help="Kalman process noise for position")
	p.add_argument("--kalman-q-vel", type=float, default=1e-2, help="Kalman process noise for velocity")
	p.add_argument("--kalman-r-base", type=float, default=1e-2, help="Kalman base measurement noise")
	p.add_argument("--kalman-max-predict", type=int, default=30, help="Max consecutive prediction-only steps to keep (mask beyond)")
	p.add_argument("--kalman-gate", type=float, default=0.0, help="Kalman: Mahalanobis^2 gate to reject outlier measurements (0 disables).")
	args = p.parse_args(argv)

	in_path = Path(args.input)
	output_dir = Path(args.output_dir) if str(args.output_dir).strip() else _infer_preprocessed_output_dir(in_path)
	if not output_dir.is_absolute():
		output_dir = ROOT / output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	files = [in_path] if in_path.is_file() else sorted(in_path.glob("*.csv"))
	if not files:
		print("No CSV files found.")
		return 0

	schema = Schema(keypoints=["Nose","Body_C","Ear_L","Ear_R","Neck","Tail"])

	for f in files:
		df, _ = read_pose_csv(f, schema)
		# auto-dedup if needed
		if df.duplicated(["frame_idx", "track"]).any():
			print(f"{f.name}: duplicates detected; applying dedup with policy={args.dedup_policy}")
			df = deduplicate_frame_track(df, policy=args.dedup_policy)
		elif args.dedup:
			df = deduplicate_frame_track(df, policy=args.dedup_policy)
		if args.pad:
			df = pad_frame_track_grid(df)
		filled, summary = interpolate_body_c(
			df,
			method=args.method,
			max_gap=args.max_gap,
			smooth_window=(args.smooth_window if args.smooth_window > 1 else None),
			score_threshold=args.score_threshold,
			kalman_q_pos=args.kalman_q_pos,
			kalman_q_vel=args.kalman_q_vel,
			kalman_r_base=args.kalman_r_base,
			kalman_max_predict=args.kalman_max_predict,
			kalman_gate_mahalanobis_sq=(None if args.kalman_gate <= 0 else float(args.kalman_gate)),
		)
		out_file = output_dir / f"interp__{args.method}__{f.name}"
		filled.to_csv(out_file, index=False)
		print(f"{f.name}: method={summary['method']}, filled_x={summary['filled_x']}, filled_y={summary['filled_y']}, long_gaps={summary['long_gaps']}, remaining_nan_x={summary['remaining_nan_x']}, remaining_nan_y={summary['remaining_nan_y']}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
