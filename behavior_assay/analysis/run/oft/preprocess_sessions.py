from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path
import pandas as pd
from analysis.core.oft.common import DEFAULT_OUTPUT_DIR, safe_filename, to_abs_path
from analysis.core.oft.preprocess_sessions import _as_bool
from analysis.core.oft.preprocess_sessions import preprocess_one

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Deduplicate, pad, confidence-mask, and interpolate OFT pose sessions.")
	parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_DIR / "manifest" / "session_manifest.csv"))
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "preprocessed"))
	parser.add_argument("--score-threshold", type=float, default=0.5)
	parser.add_argument("--max-gap", type=int, default=5, help="Maximum interior gap filled per keypoint")
	args = parser.parse_args(argv)

	manifest_path = to_abs_path(args.manifest)
	output_dir = to_abs_path(args.output_dir)
	session_dir = output_dir / "sessions"
	session_dir.mkdir(parents=True, exist_ok=True)
	manifest = pd.read_csv(manifest_path)
	manifest = manifest[_as_bool(manifest["include"])].copy()

	summary_rows: list[dict[str, object]] = []
	issue_rows: list[dict[str, object]] = []
	for row in manifest.to_dict(orient="records"):
		session_id = str(row["session_id"])
		output_path = session_dir / f"{safe_filename(session_id)}__preprocessed.csv"
		try:
			filled, summary = preprocess_one(
				Path(str(row["raw_path"])),
				frame_start=int(row["frame_start"]),
				frame_end=int(row["frame_end"]),
				score_threshold=float(args.score_threshold),
				max_gap=int(args.max_gap),
			)
			filled.to_csv(output_path, index=False)
			summary_rows.append(
				{
					"session_id": session_id,
					"subject_id": row["subject_id"],
					"cage_id": row["cage_id"],
					"condition": row["condition"],
					"sex": row["sex"],
					"animal_id": row["animal_id"],
					"fps": row["fps"],
					"arena_cm": row["arena_cm"],
					"score_threshold": float(args.score_threshold),
					"max_gap": int(args.max_gap),
					**summary,
					"raw_path": row["raw_path"],
					"preprocessed_path": str(output_path.resolve()),
				}
			)
		except Exception as exc:
			issue_rows.append({"session_id": session_id, "issue": "preprocess_error", "detail": str(exc)})

	summary_df = pd.DataFrame(summary_rows)
	issues_df = pd.DataFrame(issue_rows, columns=["session_id", "issue", "detail"])
	summary_path = output_dir / "preprocess_summary.csv"
	issues_path = output_dir / "preprocess_issues.csv"
	summary_df.to_csv(summary_path, index=False)
	issues_df.to_csv(issues_path, index=False)
	print(f"Manifest sessions: {len(manifest)}")
	print(f"Preprocessed sessions: {len(summary_df)}")
	print(f"Errors: {len(issues_df)}")
	print(f"Wrote {summary_path}")
	print(f"Wrote {issues_path}")
	return 0 if issues_df.empty else 1

if __name__ == '__main__':
    raise SystemExit(main())
