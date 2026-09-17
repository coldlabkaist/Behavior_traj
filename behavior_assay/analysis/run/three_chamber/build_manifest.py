from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.paths import RAW_DATA
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from analysis.core.three_chamber.build_manifest import _build_rows
from analysis.core.three_chamber.build_manifest import _collect_files
from analysis.core.three_chamber.build_manifest import _parse_condition_map

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description="Build a session manifest for 3-chamber pose files and pins files."
	)
	parser.add_argument(
		"input",
		nargs="?",
		default=str(RAW_DATA["3chamber"]),
		help="Root directory containing dated 3-chamber folders.",
	)
	parser.add_argument(
		"--output-dir",
		default="output/3chamber/manifest",
		help="Directory to write session_manifest.csv and session_manifest_issues.csv",
	)
	parser.add_argument(
		"--condition-map",
		default="",
		help="Optional fallback DATE=LABEL pairs for flat layouts without condition folders",
	)
	parser.add_argument(
		"--exclude-dirs",
		default="tmp",
		help="Comma-separated directory names to exclude recursively (default: tmp)",
	)
	args = parser.parse_args(argv)

	input_root = Path(args.input)
	if not input_root.is_absolute():
		input_root = ROOT / input_root
	if not input_root.exists():
		raise FileNotFoundError(f"Input directory not found: {input_root}")

	output_dir = Path(args.output_dir)
	if not output_dir.is_absolute():
		output_dir = ROOT / output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	excluded_dirs = {value.strip().lower() for value in str(args.exclude_dirs).split(",") if value.strip()}
	pose_map, pins_map = _collect_files(input_root, excluded_dirs=excluded_dirs)
	condition_map = _parse_condition_map(args.condition_map)
	rows = _build_rows(pose_map, pins_map, condition_map=condition_map)
	if not rows:
		print("No 3-chamber CSV files found.")
		return 0

	df = pd.DataFrame(rows).sort_values(
		["condition", "date", "subject_id", "phase", "phase_detail", "trial_id", "session_n", "layout_raw", "session_stem"],
		na_position="last",
	).reset_index(drop=True)
	issues_df = df[df["issues"].astype(str) != ""].reset_index(drop=True)

	manifest_path = output_dir / "session_manifest.csv"
	issues_path = output_dir / "session_manifest_issues.csv"
	df.to_csv(manifest_path, index=False)
	issues_df.to_csv(issues_path, index=False)

	issue_counter = Counter()
	for issue_blob in issues_df["issues"]:
		for issue in str(issue_blob).split(";"):
			if issue:
				issue_counter[issue] += 1

	print(f"Scanned pose sessions: {sum(len(paths) for paths in pose_map.values())}")
	print(f"Scanned pins sessions: {sum(len(paths) for paths in pins_map.values())}")
	print(f"Manifest rows: {len(df)}")
	print(f"Complete matches: {int(df['is_complete'].sum())}")
	print(f"Rows with issues: {len(issues_df)}")
	if "condition" in df.columns:
		for condition, count in df["condition"].fillna("").value_counts().sort_index().items():
			print(f"Condition {condition or 'unknown'}: {count}")
	for issue, count in sorted(issue_counter.items()):
		print(f" - {issue}: {count}")
	print(f"Wrote {manifest_path.relative_to(ROOT).as_posix()}")
	print(f"Wrote {issues_path.relative_to(ROOT).as_posix()}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
