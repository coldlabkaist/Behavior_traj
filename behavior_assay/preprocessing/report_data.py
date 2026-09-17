from __future__ import annotations

from pathlib import Path
import argparse
import sys

# ensure project root on sys.path when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from data_loader.io.schema import Schema
from data_loader.io.csv_loader import read_pose_csv
from data_loader.reports.basic import summarize_file_level, per_track_missing_and_duplicate_frames
from data_loader.processing.deduplicate import deduplicate_frame_track


def _discover_pose_csvs(data_dir: Path) -> list[Path]:
	def is_pose_csv(path: Path) -> bool:
		return path.suffix.lower() == ".csv" and not path.name.lower().endswith("_pins.csv")

	direct = sorted([p for p in data_dir.glob("*.csv") if is_pose_csv(p)])
	if direct:
		return direct
	return sorted([p for p in data_dir.rglob("*.csv") if is_pose_csv(p)])


def main(argv=None):
	parser = argparse.ArgumentParser(description="Summarize pose CSVs for missing and duplicate tracks")
	parser.add_argument("data_dir", nargs="?", default="data", help="Directory containing CSV files")
	parser.add_argument("--dedup", action="store_true", help="Apply deduplication before reporting")
	parser.add_argument("--dedup-policy", default="highest_instance_score", choices=["highest_instance_score","highest_kp_score_sum","first"], help="Deduplication policy")
	args = parser.parse_args(argv)

	data_dir = Path(args.data_dir)
	csvs = _discover_pose_csvs(data_dir)
	if not csvs:
		print(f"No CSV files found in {data_dir}")
		return 0

	# Minimal keypoints for validation; will not enforce presence of all
	schema = Schema(keypoints=["Nose","Body_C","Ear_L","Ear_R","Neck","Tail"])

	print("file,frames,tracks_expected,missing_frames,duplicate_frames")
	for csv_path in csvs:
		try:
			df, _ = read_pose_csv(csv_path, schema)
			if args.dedup:
				df = deduplicate_frame_track(df, policy=args.dedup_policy)
			stats = summarize_file_level(df)
			print(
				f"{csv_path.name},{stats['frames']},{stats['tracks_expected']},{stats['missing_frames']},{stats['duplicate_frames']}"
			)
			# Per-track breakdown
			pt = per_track_missing_and_duplicate_frames(df)
			for _, r in pt.iterrows():
				print(f"  track={r['track']}, missing_frames={r['missing_frames']}, duplicate_frames={r['duplicate_frames']}")
		except Exception as e:
			print(f"{csv_path.name},ERROR,{e}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
