from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import pandas as pd
from analysis.core.mom_pup.common import DEFAULT_OUTPUT_DIR, DEFAULT_RAW_DIR, parse_session_stem, to_abs_path
from analysis.core.mom_pup.build_manifest import inspect_file

from analysis.core.mom_pup.build_manifest import EXPECTED_PNDS

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Build a longitudinal mom-pup manifest with source-file QC.")
	parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "manifest"))
	parser.add_argument("--fps", type=float, default=30.0)
	parser.add_argument("--cage-width-mm", type=float, default=320.0)
	parser.add_argument("--cage-depth-mm", type=float, default=200.0)
	parser.add_argument("--no-hash", action="store_true")
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
			cage_width_mm=float(args.cage_width_mm),
			cage_depth_mm=float(args.cage_depth_mm),
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
		manifest = manifest.sort_values(["condition", "subject_id", "pnd"]).reset_index(drop=True)
		for subject_id, group in manifest.groupby("subject_id"):
			pnds = set(pd.to_numeric(group["pnd"], errors="coerce").dropna().astype(int))
			missing_pnds = sorted(EXPECTED_PNDS - pnds)
			if missing_pnds:
				issue_rows.append(
					{
						"session_id": "",
						"file_name": "",
						"issue": "subject_missing_pnd",
						"detail": f"{subject_id}:{','.join(map(str, missing_pnds))}",
					}
				)
			if group["condition"].nunique() > 1:
				issue_rows.append(
					{
						"session_id": "",
						"file_name": "",
						"issue": "subject_condition_conflict",
						"detail": subject_id,
					}
				)
		duplicates = manifest.duplicated(["subject_id", "pnd"], keep=False)
		for row in manifest[duplicates].to_dict(orient="records"):
			issue_rows.append(
				{
					"session_id": row["session_id"],
					"file_name": row["file_name"],
					"issue": "duplicate_subject_pnd",
					"detail": f"{row['subject_id']}:PND{row['pnd']}",
				}
			)

	issues_df = pd.DataFrame(issue_rows, columns=["session_id", "file_name", "issue", "detail"])
	manifest_path = output_dir / "session_manifest.csv"
	issues_path = output_dir / "session_manifest_issues.csv"
	manifest.to_csv(manifest_path, index=False)
	issues_df.to_csv(issues_path, index=False)
	print(f"Raw mom-pup files: {len(files)}")
	print(f"Included sessions: {int(manifest['include'].sum()) if not manifest.empty else 0}")
	print(f"Subjects: {manifest['subject_id'].nunique() if not manifest.empty else 0}")
	if not manifest.empty:
		for condition, count in manifest["condition"].value_counts().sort_index().items():
			print(f"{condition}: {count}")
	print(f"QC issue rows: {len(issues_df)}")
	print(f"Wrote {manifest_path}")
	print(f"Wrote {issues_path}")
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
