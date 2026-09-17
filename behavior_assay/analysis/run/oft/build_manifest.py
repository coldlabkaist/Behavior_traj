from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import pandas as pd
from analysis.core.oft.common import ARENA_CM, DEFAULT_OUTPUT_DIR, DEFAULT_RAW_DIR, FPS, parse_oft_stem, to_abs_path
from analysis.core.oft.build_manifest import inspect_file

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Build a QC-aware manifest for OFT pose CSVs.")
	parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "manifest"))
	parser.add_argument("--fps", type=float, default=FPS)
	parser.add_argument("--arena-cm", type=float, default=ARENA_CM)
	parser.add_argument("--no-hash", action="store_true", help="Skip SHA-256 calculation")
	args = parser.parse_args(argv)

	raw_dir = to_abs_path(args.raw_dir)
	output_dir = to_abs_path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	files = sorted(path for path in raw_dir.glob("*.csv") if path.is_file())

	rows: list[dict[str, object]] = []
	issue_rows: list[dict[str, object]] = []
	for path in files:
		row, issues = inspect_file(
			path,
			fps=float(args.fps),
			arena_cm=float(args.arena_cm),
			calculate_hash=not args.no_hash,
		)
		rows.append(row)
		for issue in issues:
			issue_rows.append(
				{
					"session_id": row.get("session_id", path.stem),
					"file_name": path.name,
					**issue,
				}
			)

	manifest = pd.DataFrame(rows)
	if not manifest.empty:
		manifest = manifest.sort_values(["condition", "cage_id", "sex", "animal_id"]).reset_index(drop=True)
	issues_df = pd.DataFrame(issue_rows, columns=["session_id", "file_name", "issue", "detail"])
	manifest_path = output_dir / "session_manifest.csv"
	issues_path = output_dir / "session_manifest_issues.csv"
	manifest.to_csv(manifest_path, index=False)
	issues_df.to_csv(issues_path, index=False)

	print(f"Raw OFT files: {len(files)}")
	print(f"Included sessions: {int(manifest['include'].sum()) if not manifest.empty else 0}")
	print(f"QC issue rows: {len(issues_df)}")
	if not manifest.empty:
		for condition, count in manifest["condition"].value_counts().sort_index().items():
			print(f"{condition}: {count}")
	print(f"Wrote {manifest_path}")
	print(f"Wrote {issues_path}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
