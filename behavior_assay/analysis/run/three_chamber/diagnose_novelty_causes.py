from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from analysis.core.three_chamber.diagnose_novelty_causes import _attach_behavior_covariates
from analysis.core.three_chamber.diagnose_novelty_causes import _attach_metadata
from analysis.core.three_chamber.diagnose_novelty_causes import _build_stats
from analysis.core.three_chamber.diagnose_novelty_causes import _cage_summary
from analysis.core.three_chamber.diagnose_novelty_causes import _group_summary
from analysis.core.three_chamber.diagnose_novelty_causes import _leave_one_batch_out
from analysis.plots.three_chamber.diagnose_novelty_causes import _plot_qc
from analysis.core.three_chamber.diagnose_novelty_causes import _to_abs_path
from analysis.core.three_chamber.diagnose_novelty_causes import _write_report

import argparse

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Trace batch, sex, cage, locomotion, and open-bias causes of the social-novelty result.")
	parser.add_argument("--preference-sample", default="output/3chamber/barplots/preference_sample_summary__Nose__contact20__n1__first5min.csv")
	parser.add_argument("--manifest", default="output/3chamber/manifest/session_manifest.csv")
	parser.add_argument("--locomotion-sample", default="output/3chamber/locomotion/locomotion_sample_summary__Body_C.csv")
	parser.add_argument("--output-dir", default="output/3chamber/diagnostics")
	return parser.parse_args()

def main() -> None:
	args = parse_args()
	preference_path = _to_abs_path(args.preference_sample)
	manifest_path = _to_abs_path(args.manifest)
	locomotion_path = _to_abs_path(args.locomotion_sample)
	output_dir = _to_abs_path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	sample_df = _attach_metadata(pd.read_csv(preference_path), pd.read_csv(manifest_path))
	sample_df = _attach_behavior_covariates(sample_df, locomotion_path)
	batch_df = _group_summary(sample_df, ["condition", "date"])
	sex_df = _group_summary(sample_df, ["condition", "sex"])
	cage_df = _cage_summary(sample_df)
	leave_one_df = _leave_one_batch_out(sample_df)
	stats_df = _build_stats(sample_df, cage_df)

	paths = {
		"sample": output_dir / "novelty_sample_with_metadata.csv",
		"batch": output_dir / "novelty_batch_summary.csv",
		"sex": output_dir / "novelty_sex_summary.csv",
		"cage": output_dir / "novelty_cage_summary.csv",
		"leave_one": output_dir / "novelty_leave_one_batch_out.csv",
		"stats": output_dir / "novelty_cause_stats.csv",
		"plot": output_dir / "novelty_cause_qc.png",
		"report": output_dir / "novelty_cause_report.md",
	}
	sample_df.to_csv(paths["sample"], index=False)
	batch_df.to_csv(paths["batch"], index=False)
	sex_df.to_csv(paths["sex"], index=False)
	cage_df.to_csv(paths["cage"], index=False)
	leave_one_df.to_csv(paths["leave_one"], index=False)
	stats_df.to_csv(paths["stats"], index=False)
	_plot_qc(sample_df, cage_df, leave_one_df, paths["plot"])
	_write_report(paths["report"], sample_df, batch_df, sex_df, leave_one_df, stats_df)

	print(f"Novelty samples: {len(sample_df)}")
	print(batch_df[["condition", "date", "preference_index_n", "preference_index_mean"]].to_string(index=False))
	print("\nKey tests:")
	show = stats_df[stats_df["axis"].isin(["condition", "sex_condition_interaction", "cage_mean_condition", "locomotion_condition"])]
	print(show[["axis", "metric", "comparison", "pvalue", "p_label"]].to_string(index=False))
	for path in paths.values():
		print(f"Wrote {path.relative_to(ROOT).as_posix()}")

if __name__ == '__main__':
    raise SystemExit(main())
